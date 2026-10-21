"""Real-checkpoint integration check; run once in each tokenizer's environment."""

import argparse
import ast
import importlib
import json
import os
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
import torch

from tokenization.cache import provenance, write_cache
from tokenization.pipeline import batches, read_dataset, sha256


def upstream_definitions(path, names, namespace):
    """Execute exact upstream definitions without unrelated training imports (test only)."""
    nodes = [n for n in ast.parse(path.read_text()).body
             if getattr(n, "name", None) in names or
             (isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
              and n.targets[0].id in names)]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def check(name, repo, checkpoint, dataset):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    module = importlib.import_module(f"tokenization.{name}")
    tokenizer = module.Tokenizer(repo, checkpoint)
    data = read_dataset(dataset)
    row = data["recordings"][0]
    cached = torch.load(dataset / row["signal_file"], map_location="cpu", weights_only=True)
    original = cached["eeg"].clone()
    signal = data["signal_metadata"]
    prepared = tokenizer.prepare(cached["eeg"], signal)
    windows = prepared[:, :10 * tokenizer.frequency].unfold(-1, 5 * tokenizer.frequency,
                                                            5 * tokenizer.frequency).permute(1, 0, 2)
    book = tokenizer.codebook
    with torch.no_grad():
        actual = tokenizer.encode(windows)
        if name == "dac":
            x = cached["windows"][:2].float().reshape(38, 1, 1280)
            x = ((x - x.mean(-1, keepdim=True)) * (.3 / x.std(-1, keepdim=True).clamp_min(1e-8))).clamp(-1, 1)
            native = tokenizer.model.encode(tokenizer.model.preprocess(x, 256))[1].reshape(2, 19, -1)
        else:
            import mne

            raw = mne.io.RawArray(original.numpy().copy(), mne.create_info(
                signal["channel_names"], 256, "eeg"), verbose=False)
            raw.filter(0.1, 75.0, verbose=False).notch_filter(50.0, verbose=False)
            assert raw.ch_names == signal["channel_names"]
            if name == "tfm":
                from einops import rearrange
                from scipy.signal import resample

                expected = raw.get_data(units="uV")
                np.testing.assert_array_equal(prepared.numpy(), expected)
                loader = upstream_definitions(repo / "datasets/data_loaders.py", {"TUEVloader"},
                    dict(Dataset=torch.utils.data.Dataset, os=os, pickle=pickle,
                         np=np, torch=torch, resample=resample))["TUEVloader"]
                stft = upstream_definitions(repo / "utils/utils.py", {"get_stft_torch"},
                    dict(torch=torch, rearrange=rearrange))["get_stft_torch"]
                with TemporaryDirectory() as directory:
                    evaluation = Path(directory) / "processed_eval"
                    evaluation.mkdir()
                    for i in range(2):
                        with (evaluation / f"{i}.pkl").open("wb") as stream:
                            pickle.dump({"signal": expected[:, i * 1280:(i + 1) * 1280], "label": [1]}, stream)
                    upstream = loader(directory, "test", resampling_rate=200)
                    upstream.data_files.sort()
                    x = torch.stack([upstream[i][0] for i in range(2)])
                spectrum, temporal = tokenizer.inputs(windows)
                assert torch.equal(temporal, x.flatten(0, 1))
                assert torch.equal(spectrum, stft(x).flatten(0, 1))
                quantized, ids, _ = tokenizer.model.tokenize(stft(x).flatten(0, 1), x.flatten(0, 1))
                native = ids.reshape(2, 19, 9)
                torch.testing.assert_close(book[native], quantized.reshape(2, 19, 9, -1))
            else:
                native_utils = upstream_definitions(repo / "utils.py",
                    {"standard_1020", "get_input_chans"}, {})
                channels = native_utils["get_input_chans"]([c.upper() for c in signal["channel_names"]])
                assert tokenizer.input_chans == channels
                raw.resample(200, verbose=False)
                np.testing.assert_array_equal(prepared.numpy(), raw.get_data(units="uV"))
                x = torch.from_numpy(raw.get_data(units="uV")[:, :2000]).reshape(19, 2, 1000)
                x = (x.permute(1, 0, 2).float() / 100).reshape(2, 19, 5, 200)
                result = tokenizer.model.get_tokens(x, input_chans=channels)
                native = result["token"].reshape(2, 19, 5)
                torch.testing.assert_close(book[native], result["quantize"].reshape(2, 19, 5, -1))
        assert torch.equal(actual, native), "Adapter differs from native inference"
        assert torch.equal(actual, torch.cat([tokenizer.encode(w) for w in windows.split(1)]))
    assert torch.equal(cached["eeg"], original), "Preprocessing mutated source EEG"
    assert torch.equal(book, tokenizer.codebook)
    # Consumers can backpropagate through a classifier using these frozen features.
    projection = torch.nn.Linear(book.shape[1], 2)
    projection(book[actual]).sum().backward()
    assert torch.isfinite(projection.weight.grad).all()

    if name == "dac":
        short = module.Tokenizer(repo, checkpoint, window_samples=256)
        short_windows = windows[:, :, :256]
        x = short_windows.float().reshape(-1, 1, 256)
        x = ((x - x.mean(-1, keepdim=True)) * (.3 / x.std(-1, keepdim=True).clamp_min(1e-8))).clamp(-1, 1)
        native = short.model.encode(short.model.preprocess(x, 256))[1].reshape(2, 19, -1)
        assert short.tokens_per_window == 256 // short.model.hop_length
        assert torch.equal(short.encode(short_windows), native), "One-second adapter differs from native inference"

    with TemporaryDirectory() as directory:
        fixture = Path(directory) / "eeg"
        fixture.mkdir()
        # Keep two real examples with their exact original IDs, labels and split.
        data["recordings"], data["windows"] = [row], data["windows"][:2]
        cached["example_ids"], cached["windows"] = cached["example_ids"][:2], cached["windows"][:2]
        torch.save(cached, fixture / row["signal_file"])
        row["signal_sha256"] = sha256(fixture / row["signal_file"])
        (fixture / "cache.json").write_text(json.dumps(data))
        emitted = list(batches(fixture, tokenizer, batch_size=1))
        assert torch.equal(torch.cat([b["ids"] for b in emitted]), actual)
        assert torch.cat([b["example_ids"] for b in emitted]).tolist() == [0, 1]
        output = Path(directory) / "tokens"
        run = provenance(repo, checkpoint)
        write_cache(fixture, tokenizer, output, run, batch_size=2)
        saved = torch.load(output / "000.pt", weights_only=True)
        assert torch.equal(saved["ids"], actual) and saved["example_ids"].tolist() == [0, 1]
        assert torch.equal(torch.load(output / "codebook.pt", weights_only=True), book)
        meta = json.loads((output / "metadata.json").read_text())
        assert meta["dataset_sha256"] == sha256(fixture / "cache.json")
        for filename, digest in meta["files"].items():
            assert sha256(output / filename) == digest
        with TestCase().assertRaisesRegex(AssertionError, "already exists"):
            write_cache(fixture, tokenizer, output, run)
        data["signal_metadata"]["signal_unit"] = "uV"
        (fixture / "cache.json").write_text(json.dumps(data))
        failed = Path(directory) / "failed"
        with TestCase().assertRaises(AssertionError):
            write_cache(fixture, tokenizer, failed, run)
        assert not failed.exists()
        data["signal_metadata"]["signal_unit"] = "V"
        row["signal_sha256"] = "corrupted"
        (fixture / "cache.json").write_text(json.dumps(data))
        with TestCase().assertRaises(AssertionError):
            write_cache(fixture, tokenizer, failed, run)
        assert not failed.exists() and not list(Path(directory).glob(".tokens-*"))
    print(f"PASS {name}: native preprocessing/IDs/codebook, batching, cache equivalence, failure checks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokenizer", choices=("dac", "tfm", "labram"))
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("dataset cache/depression/data"))
    args = parser.parse_args()
    check(args.tokenizer, args.repo.resolve(), args.checkpoint.resolve(), args.dataset.resolve())
