"""The single-codebook EEG DAC; architecture comes from the checkpoint."""

from pathlib import Path
import torch


class Tokenizer:
    def __init__(self, repo, checkpoint, device="cpu", target_std=0.3, clip=1.0):
        import dac
        from audiotools import AudioSignal
        from audiotools.data.transforms import EEGNormalize

        assert Path(dac.__file__).resolve().is_relative_to(Path(repo).resolve())
        weights = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = weights["metadata"]["kwargs"]
        assert config["n_codebooks"] == 1, "The (B,C,T) contract requires one codebook"
        assert config["sample_rate"] == 256
        assert target_std > 0 and clip > 0
        self.model = dac.DAC(**config)
        self.model.load_state_dict(weights["state_dict"], strict=True)
        self.model.to(device).eval().requires_grad_(False)
        self.device, self.frequency = device, 256
        hop = int(self.model.hop_length)
        assert 1280 % hop == 0, "Five seconds must divide into whole tokens"
        self.tokens_per_window = 1280 // hop
        self.normalize = EEGNormalize(target_std=target_std, clip=clip)
        self.audio_signal = AudioSignal
        self.metadata = {"architecture": config, "target_std": target_std, "clip": clip,
                         "filter": None, "normalization": "EEGNormalize per channel/window",
                         "token_stride_seconds": hop / 256}

    @property
    def codebook(self):
        return self.model.quantizer.quantizers[0].codebook.weight.detach().cpu().clone()

    def prepare(self, eeg, metadata):
        assert metadata["frequency_hz"] == self.frequency
        return eeg

    @torch.no_grad()
    def encode(self, windows):
        b, c, n = windows.shape
        assert n == 1280
        x = windows.to(device=self.device, dtype=torch.float32).reshape(b * c, 1, n)
        # Deterministic native transform, without the random augmentation wrapper.
        x = self.normalize._transform(self.audio_signal(x, sample_rate=256)).audio_data
        _, ids, _, _, _ = self.model.encode(self.model.preprocess(x, 256))
        assert ids.shape == (b * c, 1, self.tokens_per_window)
        return ids.reshape(b, c, self.tokens_per_window)
