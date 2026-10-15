"""TFM's five-second TUEV preprocessing, retaining the cache's EEG montage."""

import numpy as np
from scipy.signal import resample
import torch

from .pipeline import filter_recording, source_module


class Tokenizer:
    frequency, tokens_per_window = 256, 9

    def __init__(self, repo, checkpoint, device="cpu"):
        source = source_module(repo, "models/tfm_token.py")
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
        k, d = weights["quantizer.embedding.weight"].shape
        self.model = source.get_tfm_tokenizer_2x2x8(code_book_size=k, emb_size=d)
        self.model.load_state_dict(weights, strict=True)
        self.model.to(device).eval().requires_grad_(False)
        self.device = device
        self.metadata = {"architecture": "get_tfm_tokenizer_2x2x8", "filter_hz": [0.1, 75],
                         "notch_hz": 50, "resampling": "scipy FFT, each window, 256->200 Hz",
                         "scaling": "V->uV; divide by channel/window abs q95 + 1e-8",
                         "montage": "unchanged; transfer from TUEV bipolar training",
                         "stft": "200-point Hann, hop 100, center=False, omit Nyquist",
                         "token_patch_seconds": 1, "token_stride_seconds": 0.5}

    @property
    def codebook(self):
        return self.model.quantizer.embedding.weight.detach().cpu().clone()

    def prepare(self, eeg, metadata):
        return torch.from_numpy(filter_recording(eeg, metadata).get_data(units="uV"))

    def inputs(self, windows):
        assert windows.shape[-1] == 1280
        x = resample(windows.numpy(), 1000, axis=-1)
        x /= np.quantile(np.abs(x), .95, axis=-1, method="linear", keepdims=True) + 1e-8
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device).flatten(0, 1)
        spectrum = torch.stft(x, n_fft=200, hop_length=100, center=False, onesided=True,
                              return_complex=True, window=torch.hann_window(200, device=x.device))
        return spectrum[:, :100].abs(), x

    @torch.no_grad()
    def encode(self, windows):
        _, ids, _ = self.model.tokenize(*self.inputs(windows))
        return ids.reshape(windows.shape[0], windows.shape[1], self.tokens_per_window)
