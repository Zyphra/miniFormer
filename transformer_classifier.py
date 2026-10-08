import torch
from torch import nn
from torch.nn import functional as F
from linear_attention_transformer import LinearAttentionTransformer


def sinusoidal_encoding(n_positions, dim):
    """Return (n_positions, dim) sine/cosine encodings; dim must be even."""
    positions = torch.arange(n_positions, dtype=torch.float32)[:, None]
    frequencies = 10000 ** (-torch.arange(0, dim, 2).float() / dim)
    angles = positions * frequencies
    return torch.cat((angles.sin(), angles.cos()), dim=-1)


class TransformerClassifier(nn.Module):
    """Map (batch_size, channels, coarse_time, codebook_dim) to class logits.

    normalize=True gives additive positional vectors unit L2 norm.
    RoPE rotations and projected input embeddings are left unchanged.
    """

    def __init__(
        self,
        n_classes,
        channels=16,
        coarse_time=32,
        codebook_dim=8,
        model_dim=64,
        heads=2,
        depth=2,
        temporal_pos="sinusoidal",
        channel_pos="spatial",
        aggregation_mode="CLS_token",
        normalize=False,
        channel_coords=None,
    ):
        super().__init__()
        if temporal_pos not in ("sinusoidal", "rotary"):
            raise ValueError("temporal_pos must be 'sinusoidal' or 'rotary'")
        if channel_pos not in (None, "learned", "spatial"):
            raise ValueError("channel_pos must be None, 'learned', or 'spatial'")
        if aggregation_mode not in (
            "CLS_token", "mean_pool_final", "attn_pool_final", "flatten_final"
        ):
            raise ValueError(
                "aggregation_mode must be 'CLS_token', 'mean_pool_final', "
                "'attn_pool_final', or 'flatten_final'"
            )
        if heads < 1 or model_dim % heads:
            raise ValueError("model_dim must be divisible by a positive heads count")
        pos_dim = model_dim if temporal_pos == "sinusoidal" else model_dim // heads
        if pos_dim < 2 or pos_dim % 2:
            raise ValueError("The positional embedding dimension must be positive and even")

        self.input_shape = (channels, coarse_time, codebook_dim)
        self.temporal_pos = temporal_pos
        self.channel_pos = channel_pos
        self.aggregation_mode = aggregation_mode
        self.normalize = normalize

        #  we will make linear embeddings out of the codebooks
        # standardizes model_dim for fair comparison across models
        self.input_projection = nn.Linear(codebook_dim, model_dim)

        # self.register_buffer: not a trainable parameter, saved w/ models state_dict
        self.register_buffer("time_encoding", sinusoidal_encoding(coarse_time, pos_dim))

        if channel_pos == "learned":
            self.channel_embedding = nn.Parameter(torch.randn(channels, model_dim) * 0.02)
        elif channel_pos == "spatial":
            # take (C,3) coordinates --> project to model_dim space
            if channel_coords is None or tuple(channel_coords.shape) != (channels, 3):
                raise ValueError("spatial encoding requires (channels, 3) channel_coords")
            self.register_buffer("channel_coords", channel_coords.detach().float().clone())
            self.channel_projection = nn.Linear(3, model_dim)

        num_tokens = channels * coarse_time
        classifier_dim = model_dim
        if aggregation_mode == "CLS_token":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
            num_tokens += 1
        elif aggregation_mode == "attn_pool_final":
            self.pool_score = nn.Linear(model_dim, 1)
        elif aggregation_mode == "flatten_final":
            classifier_dim = num_tokens * model_dim

        self.transformer = LinearAttentionTransformer(
            dim=model_dim, depth=depth, heads=heads,
            max_seq_len=num_tokens,
            causal=False,
        )
        self.classification_head = nn.Linear(classifier_dim, n_classes)

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != self.input_shape:
            raise ValueError(f"Expected a batch of tensors with shape {self.input_shape}")
        batch_size, channels, coarse_time, codebook_dim = x.shape
        x = self.input_projection(x)  # (batch_size, channels, coarse_time, model_dim)

        rotary_pos = None
        temporal = self.time_encoding.to(x)
        if self.temporal_pos == "sinusoidal":
            if self.normalize:
                temporal = F.normalize(temporal, dim=-1, eps=1e-6)
            x = x + temporal[None, None, :, :]
        else:
            # Repeat time positions for each channel; RoPE acts inside attention.
            rotary_pos = temporal.repeat(channels, 1)[None, :, :]

        channel = None
        if self.channel_pos == "learned":
            channel = self.channel_embedding
        elif self.channel_pos == "spatial":
            channel = self.channel_projection(self.channel_coords)
        if channel is not None:
            if self.normalize:
                channel = F.normalize(channel, dim=-1, eps=1e-6)
            x = x + channel.to(x)[None, :, None, :]

        x = x.flatten(1, 2)  # (batch_size, channels * coarse_time, model_dim)
        if self.aggregation_mode == "CLS_token":
            cls = self.cls_token.to(x).expand(batch_size, -1, -1)
            x = torch.cat((cls, x), dim=1)
            if rotary_pos is not None:
                # Position zero has sin=0, cos=1, so CLS is not rotated.
                rotary_pos = torch.cat((temporal[None, :1, :], rotary_pos), dim=1)

        x = self.transformer(x, pos_emb=rotary_pos)
        if self.aggregation_mode == "CLS_token":
            pooled = x[:, 0]
        elif self.aggregation_mode == "mean_pool_final":
            pooled = x.mean(dim=1)
        elif self.aggregation_mode == "attn_pool_final":
            weights = self.pool_score(x).softmax(dim=1)
            pooled = (weights * x).sum(dim=1)
        else:  # flatten_final
            pooled = x.flatten(1)
        return self.classification_head(pooled)


def train(model, dataloader, optimizer):
    pass


def test(model, dataloader):
    pass
