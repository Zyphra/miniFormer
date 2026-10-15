"""LaBraM VQNSP tokenizer, with joint attention across channels and time."""

from pathlib import Path
import sys
import torch

from .pipeline import filter_recording, source_module


# This benchmark's electrodes in upstream utils.standard_1020 (CLS is index 0).
CHANNEL_IDS = dict(zip(
    "FP1 F3 C3 P3 O1 F7 T3 T5 FZ FP2 F4 C4 P4 O2 F8 T4 T6 CZ PZ".split(),
    [1, 18, 40, 62, 81, 16, 89, 90, 20, 3, 22, 44, 66, 83, 24, 91, 92, 42, 64]))


class Tokenizer:
    frequency, tokens_per_window = 200, 5

    def __init__(self, repo, checkpoint, device="cpu"):
        source = source_module(repo, "modeling_vqnsp.py")
        for name in ("modeling_finetune", "norm_ema_quantizer"):
            assert Path(sys.modules[name].__file__).resolve().parent == Path(repo).resolve()
        weights = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
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
                         "token_patch_seconds": 1, "token_stride_seconds": 1}

    @property
    def codebook(self):
        return self.model.quantize.embedding.weight.detach().cpu().clone()

    def prepare(self, eeg, metadata):
        self.input_chans = [0] + [CHANNEL_IDS[c.upper()] for c in metadata["channel_names"]]
        raw = filter_recording(eeg, metadata).resample(200, verbose=False)
        return torch.from_numpy(raw.get_data(units="uV"))

    @torch.no_grad()
    def encode(self, windows):
        b, c, n = windows.shape
        assert n == 1000 and len(self.input_chans) == c + 1
        x = (windows.to(device=self.device, dtype=torch.float32) / 100).reshape(b, c, 5, 200)
        ids = self.model.get_codebook_indices(x, input_chans=self.input_chans)
        return ids.reshape(b, c, self.tokens_per_window)
