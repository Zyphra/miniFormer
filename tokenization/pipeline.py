"""Read the EEG cache and yield tokens; no NeuroAI, EDF reader, or disk writes."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter

import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_module(repo, relative):
    """Load an upstream file without shadowing miniFormer's models package."""
    path = Path(repo).resolve() / relative
    sys.path.append(str(Path(repo).resolve()))
    spec = importlib.util.spec_from_file_location("upstream_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def filter_recording(eeg, metadata):
    """Shared TFM/LaBraM full-recording filter, retaining volts and channels."""
    import mne

    info = mne.create_info(metadata["channel_names"], metadata["frequency_hz"], "eeg")
    raw = mne.io.RawArray(eeg.numpy().copy(), info, verbose=False)
    raw.filter(0.1, 75.0, verbose=False)
    return raw.notch_filter(50.0, verbose=False)


def read_dataset(path):
    data = json.loads((Path(path) / "cache.json").read_text())
    assert data["schema_version"] == 2
    signal = data["signal_metadata"]
    assert (signal["signal_unit"], signal["dtype"], signal["processing_stage"]) == (
        "V", "float64", "before_model_preprocessing")
    assert signal["frequency_hz"] == 256
    assert len(set(signal["channel_names"])) == len(signal["channel_names"])
    assert [r["recording_id"] for r in data["recordings"]] == list(range(len(data["recordings"])))
    assert [w["example_id"] for w in data["windows"]] == list(range(len(data["windows"])))
    durations = {w["duration"] for w in data["windows"]}
    assert len(durations) == 1 and next(iter(durations)) > 0, "Expected one positive window duration"
    return data


def batches(dataset, tokenizer, batch_size=32):
    """Yield CPU int64 ids (B,C,T), example_ids (B,), and recording_id."""
    assert batch_size > 0 and not tokenizer.model.training
    assert not any(p.requires_grad for p in tokenizer.model.parameters())
    data = read_dataset(dataset)
    signal = data["signal_metadata"]
    channels = len(signal["channel_names"])
    examples = [[] for _ in data["recordings"]]
    for window in data["windows"]:
        assert 0 <= window["recording_id"] < len(examples)
        examples[window["recording_id"]].append(window)
    codebook = tokenizer.codebook
    assert codebook.ndim == 2 and codebook.dtype == torch.float32
    assert torch.isfinite(codebook).all()
    started, reported, n_examples = perf_counter(), perf_counter(), 0
    for recording_index, (row, windows) in enumerate(zip(data["recordings"], examples)):
        path = Path(dataset) / row["signal_file"]
        assert sha256(path) == row["signal_sha256"], path
        cached = torch.load(path, map_location="cpu", weights_only=True)
        eeg = cached["eeg"]
        expected_ids = torch.tensor([w["example_id"] for w in windows], dtype=torch.int64)
        assert torch.equal(cached["example_ids"], expected_ids)
        assert eeg.dtype == torch.float64 and eeg.shape == (channels, row["n_samples"])
        assert torch.isfinite(eeg).all()
        prepared = tokenizer.prepare(eeg, signal)
        assert prepared.ndim == 2 and prepared.shape[0] == channels
        assert torch.isfinite(prepared).all()
        for start in range(0, len(windows), batch_size):
            selected = windows[start:start + batch_size]
            segments = []
            for w in selected:
                assert w["start_sample"] == w["start"] * signal["frequency_hz"]
                assert w["stop_sample"] == (w["start"] + w["duration"]) * signal["frequency_hz"]
                assert 0 <= w["start_sample"] < w["stop_sample"] <= eeg.shape[-1]
                left = w["start"] * tokenizer.frequency
                right = (w["start"] + w["duration"]) * tokenizer.frequency
                assert left == int(left) and right == int(right)
                assert 0 <= left < right <= prepared.shape[-1]
                segments.append(prepared[:, int(left):int(right)])
            ids = tokenizer.encode(torch.stack(segments)).detach().cpu()
            assert ids.dtype == torch.int64
            assert ids.shape == (len(selected), channels, tokenizer.tokens_per_window)
            assert 0 <= ids.min() and ids.max() < len(codebook)
            yield {"recording_id": row["recording_id"], "ids": ids,
                   "example_ids": expected_ids[start:start + batch_size]}
        n_examples += len(windows)
        if perf_counter() - reported >= 60 or recording_index + 1 == len(examples):
            print(f"Tokenized {recording_index + 1}/{len(examples)} recordings, "
                  f"{n_examples:,} windows in {perf_counter() - started:.1f}s", flush=True)
            reported = perf_counter()
    assert torch.equal(tokenizer.codebook, codebook), "Inference changed the codebook"
