"""LaBraM VQNSP tokenizer, with joint attention across channels and time."""

import ast
from pathlib import Path
import sys
import torch

from .pipeline import filter_recording, source_module


def channel_ids(repo):
    """Read upstream's channel vocabulary without importing its training dependencies."""
    tree = ast.parse((Path(repo) / "utils.py").read_text())
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name) and target.id == "standard_1020":
                    names = ast.literal_eval(statement.value)
                    return {name.upper(): index + 1 for index, name in enumerate(names)}
    raise ValueError(f"No standard_1020 channel vocabulary in {repo}/utils.py")


def resolve_channel_ids(names, vocabulary):
    """Use NeuralBench's anode fallback for unsupported bipolar position IDs."""
    resolved = {}
    for name in names:
        position = name.upper()
        if position not in vocabulary and position.count("-") == 1:
            position = position.split("-")[0]
        if position not in vocabulary:
            raise ValueError(f"No LaBraM channel embedding for {name}")
        resolved[name] = vocabulary[position]
    return resolved


class Tokenizer:
    frequency, tokens_per_window = 200, 5

    def __init__(self, repo, checkpoint, device="cpu"):
        source = source_module(repo, "modeling_vqnsp.py")
        for name in ("modeling_finetune", "norm_ema_quantizer"):
            assert Path(sys.modules[name].__file__).resolve().parent == Path(repo).resolve()
        weights = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
        # Upstream lists 136 names, but this checkpoint has only 128 channel embeddings.
        self.channel_ids = {name: index for name, index in channel_ids(repo).items()
                            if index < weights["encoder.pos_embed"].shape[1]}
        k, d = weights["quantize.embedding.weight"].shape
        self.model = source.vqnsp_encoder_base_decoder_3x200x12(n_code=k, code_dim=d)
        self.model.load_state_dict(weights, strict=True)
        assert self.model.quantize.embedding.initted.item(), "Uninitialized codebook"
        self.model.to(device).eval().requires_grad_(False)
        self.device = device
        self.metadata = {"architecture": "vqnsp_encoder_base_decoder_3x200x12",
                         "filter_hz": [0.1, 75], "notch_hz": 50,
                         "resampling": "MNE FFT, whole recording, 256->200 Hz",
                         "scaling": "V->uV, float32, divide by 100", "montage": "unchanged",
                         "channel_position_policy": "native ID; first-electrode fallback for unsupported bipolar names",
                         "token_patch_seconds": 1, "token_stride_seconds": 1}

    @property
    def codebook(self):
        return self.model.quantize.embedding.weight.detach().cpu().clone()

    def prepare(self, eeg, metadata):
        mapping = resolve_channel_ids(metadata["channel_names"], self.channel_ids)
        if "input_channel_ids" not in self.metadata:
            print(f"LaBraM channel position IDs: {mapping}", flush=True)
        self.metadata["input_channel_ids"] = mapping
        self.input_chans = [0] + list(mapping.values())
        raw = filter_recording(eeg, metadata).resample(200, verbose=False)
        return torch.from_numpy(raw.get_data(units="uV"))

    @torch.no_grad()
    def encode(self, windows):
        b, c, n = windows.shape
        assert n == 1000 and len(self.input_chans) == c + 1
        x = (windows.to(device=self.device, dtype=torch.float32) / 100).reshape(b, c, 5, 200)
        ids = self.model.get_codebook_indices(x, input_chans=self.input_chans)
        return ids.reshape(b, c, self.tokens_per_window)
