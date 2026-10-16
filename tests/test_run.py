"""CPU integration checks: python -m tests.test_run."""

import csv
from inspect import signature
from itertools import product
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import torch

from tokenization.cache import write_cache
from tokenization import dac
from tokenization.pipeline import sha256
from training import run as runner


class ToyTokenizer:
    """Small deterministic fixture; the production batching and training paths are real."""
    frequency, tokens_per_window, device = 256, 5, "cpu"
    metadata = {"test_fixture": True}
    prepared = 0

    def __init__(self, *args):
        self.model = torch.nn.Identity().eval().requires_grad_(False)
        self.codebook = torch.arange(128, dtype=torch.float32).reshape(8, 16) / 128

    def prepare(self, eeg, metadata):
        assert eeg.shape == (2, 2560)
        ToyTokenizer.prepared += 1
        return eeg

    def encode(self, windows):
        return windows[:, :, ::256].long() % 8


def check(directory):
    torch.set_num_threads(1)
    dataset, cache, output = (directory / name for name in ("eeg", "tokens", "results"))
    dataset.mkdir()
    data = {"schema_version": 2, "label_mapping": {"a": 0, "b": 1},
            "signal_metadata": {"channel_names": ["C3", "C4"], "channel_positions_m": [[1, 0, 0], [0, 1, 0]],
                "frequency_hz": 256, "signal_unit": "V", "dtype": "float64",
                "processing_stage": "before_model_preprocessing"}, "recordings": [], "windows": []}
    for r, split in enumerate(("train", "train", "val", "val", "test", "test")):
        examples = torch.tensor([2 * r, 2 * r + 1])
        signal_file = f"{r:03d}.pt"
        eeg = (torch.arange(5120).reshape(2, 2560) // 256 + r).double()
        torch.save({"eeg": eeg, "example_ids": examples}, dataset / signal_file)
        data["recordings"].append(dict(recording_id=r, subject=f"subject{r}", label=r % 2,
            split=split, n_samples=2560, signal_file=signal_file, signal_sha256=sha256(dataset / signal_file)))
        data["windows"].extend(dict(example_id=2 * r + i, recording_id=r, start=5 * i, duration=5,
            start_sample=i * 1280, stop_sample=(i + 1) * 1280) for i in range(2))
    metadata = dataset / "cache.json"
    metadata.write_text(json.dumps(data))
    write_cache(dataset, ToyTokenizer(), cache, {"fixture": True, "tokenizer": "dac"})
    # Reverse both shard and example order: alignment must use IDs, not loading order.
    cache_meta = json.loads((cache / "metadata.json").read_text())
    for file in cache.glob("[0-9]*.pt"):
        chunk = torch.load(file, weights_only=True)
        torch.save({k: v.flip(0) for k, v in chunk.items()}, file)
        cache_meta["files"][file.name] = sha256(file)
    cache_meta["files"] = dict(reversed(list(cache_meta["files"].items())))
    (cache / "metadata.json").write_text(json.dumps(cache_meta))
    before = {p: sha256(p) for folder in (dataset, cache) for p in folder.iterdir()}
    sources = [{"name": "cached", "cache": str(cache)},
               {"name": "live", "tokenizer": "dac", "repo": "/fixture", "checkpoint": "/fixture/weights.pth"},
               {"name": "repeat", "cache": str(cache)}]
    names, seeds, data_seed = [s["name"] for s in sources], [33, 34, 35], 71
    requested = [11, 3, 7, 1, 9, 5, 0, 2]
    _, expected, labels, splits = runner.read_labels(metadata, requested)
    assert expected.tolist() == [0, 1, 2, 3, 5, 7, 9, 11]
    assert labels.tolist() == [0, 0, 1, 1, 0, 1, 0, 1]
    assert {s: expected[i].tolist() for s, i in splits.items()} == {
        "train": [0, 1, 2, 3], "val": [5, 7], "test": [9, 11]}
    actual_test, actual_train = runner.test, runner.train
    evaluations, states, orders, initial_weights = [], [], [], []

    def evaluate(model, loader):
        metrics = actual_test(model, loader)
        phase = len(evaluations) % 4
        evaluations.append(loader)
        if phase < 3:
            metrics["balanced_accuracy"] = [.6, .9, .7][phase]
        return metrics

    def train(model, loader, *args):
        assert loader.generator.initial_seed() == data_seed
        epoch_batches = list(loader)
        orders.append(torch.cat([inputs[:, 0, 0] for inputs, _ in epoch_batches]).tolist())
        initial_weights.append(model[1].input_projection.weight.detach().clone())
        metrics = actual_train(model, epoch_batches, *args)
        states.append({k: v.detach().clone() for k, v in model[1].state_dict().items()})
        assert model[0].weight.grad is None
        return metrics

    ToyTokenizer.prepared = 0
    with patch.object(dac, "Tokenizer", ToyTokenizer), \
         patch("tokenization.cache.provenance", return_value={"fixture": True}), \
         patch.object(runner, "test", side_effect=evaluate), patch.object(runner, "train", side_effect=train):
        rows = runner.run(sources, metadata, output, epochs=3, batch_size=2,
                          seeds=seeds, data_seed=data_seed, example_ids=requested)
    assert ToyTokenizer.prepared == 6, "Live preprocessing ran more than once per recording"
    assert len(evaluations) == 36 and len(rows) == 9
    for i in range(9):
        assert evaluations[4 * i] is evaluations[4 * i + 1] is evaluations[4 * i + 2]
        assert evaluations[4 * i + 3] is not evaluations[4 * i]
        assert orders[3 * i:3 * i + 3] == orders[:3], "Batch order depends on classifier seed"
    assert len({tuple(order) for order in orders[:3]}) > 1, "Shuffle never changes between epochs"
    assert all(not torch.equal(initial_weights[3 * i], initial_weights[3 * j]) for i, j in ((0, 1), (0, 2), (1, 2)))
    assert all(sha256(p) == digest for p, digest in before.items())
    assert set(before) == {p for folder in (dataset, cache) for p in folder.iterdir()}, "Source files changed"
    with (output / "comparison.csv").open() as stream:
        detailed = list(csv.DictReader(stream))
    assert [(r["source"], int(r["seed"])) for r in detailed] == list(product(names, seeds))
    assert all(r["tokenizer"] == "dac" and int(r["data_seed"]) == data_seed for r in detailed)
    with (output / "summary.csv").open() as stream:
        summary = list(csv.DictReader(stream))
    assert [r["source"] for r in summary] == names
    for result in summary:
        assert int(result["n_seeds"]) == 3 and int(result["data_seed"]) == data_seed
        group = [r for r in detailed if r["source"] == result["source"]]
        for metric in ("val_balanced_accuracy", "test_balanced_accuracy", "test_cross_entropy"):
            values = [float(r[metric]) for r in group]
            mean = sum(values) / 3
            std = math.sqrt(sum((v - mean)**2 for v in values) / 2)
            assert math.isclose(float(result[f"{metric}_mean"]), mean, abs_tol=1e-12)
            assert math.isclose(float(result[f"{metric}_std"]), std, abs_tol=1e-12)
    assert float(summary[0]["test_cross_entropy_std"]) > 0
    saved = []
    for index, (name, seed) in enumerate(product(names, seeds)):
        folder = output / name / f"seed_{seed}"
        assert {p.name for p in folder.iterdir()} == {"config.json", "metrics.json", "classifier.pt"}
        metrics = json.loads((folder / "metrics.json").read_text())
        config = json.loads((folder / "config.json").read_text())
        assert metrics["best_epoch"] == 2 and metrics["test"]["n_examples"] == 2
        assert config["classifier"]["codebook_dim"] == 16
        assert config["seed"] == seed and config["data_seed"] == data_seed
        for key in ("model_dim", "heads", "depth", "temporal_pos", "channel_pos", "aggregation_mode", "normalize"):
            assert config["classifier"][key] == signature(runner.TransformerClassifier).parameters[key].default
        state = torch.load(folder / "classifier.pt", weights_only=True)
        assert state.keys() == states[1].keys()
        assert all(torch.equal(v, states[3 * index + 1][k]) for k, v in state.items())
        assert any(not torch.equal(v, states[3 * index + 2][k]) for k, v in state.items())
        saved.append((state, metrics, config["ids_tensor_sha256"]))
    for i in range(6):
        assert saved[i][1:] == saved[i + 3][1:]
        assert all(torch.equal(v, saved[i + 3][0][k]) for k, v in saved[i][0].items())
    case = TestCase()
    for incomplete in (rows[:-1], rows[:-1] + [rows[0]], rows + [rows[0]],
                       rows[:-1] + [{**rows[-1], "seed": 99}]):
        with case.assertRaisesRegex(AssertionError, "source/seed"):
            runner.summarize(incomplete, names, seeds)
    for invalid_seeds in ([], [33], [33, 33]):
        with case.assertRaisesRegex(AssertionError, "seed"):
            runner.run(sources, metadata, directory / "invalid", seeds=invalid_seeds)
    assert not (directory / "invalid").exists()
    with case.assertRaises(FileExistsError):
        runner.run(sources, metadata, output)
    with case.assertRaisesRegex(AssertionError, "duplicate"):
        runner.read_labels(metadata, [1, 1])
    data["recordings"][2]["subject"] = "subject0"
    metadata.write_text(json.dumps(data))
    with case.assertRaisesRegex(AssertionError, "Subject crosses splits"):
        runner.read_labels(metadata)
    data["recordings"][2]["subject"] = "subject2"
    metadata.write_text(json.dumps(data))
    original = (cache / "metadata.json").read_text()
    for key, value, message in (("dataset_sha256", "wrong", "Dataset metadata"),
                                ("channel_names", ["C4", "C3"], "Channel order")):
        (cache / "metadata.json").write_text(json.dumps({**cache_meta, key: value}))
        with case.assertRaisesRegex(AssertionError, message):
            runner.load_tokens(sources[0], metadata, data)
    (cache / "metadata.json").write_text(original)
    file = cache / "000.pt"
    chunk = torch.load(file, weights_only=True)
    for ids, message in (([1, 1], "Duplicate token"), ([0, 2], "Duplicate token"), ([0, 12], "Unknown token")):
        chunk["example_ids"] = torch.tensor(ids)
        torch.save(chunk, file)
        cache_meta["files"][file.name] = sha256(file)
        (cache / "metadata.json").write_text(json.dumps(cache_meta))
        with case.assertRaisesRegex(AssertionError, message):
            runner.load_tokens(sources[0], metadata, data)
    chunk["example_ids"] = torch.tensor([0])
    chunk["ids"] = chunk["ids"][:1]
    torch.save(chunk, file)
    cache_meta["files"][file.name] = sha256(file)
    (cache / "metadata.json").write_text(json.dumps(cache_meta))
    with case.assertRaisesRegex(AssertionError, "Missing requested"):
        runner.run(sources[:1], metadata, directory / "missing", example_ids=requested)
    file.write_bytes(b"corrupt")
    with case.assertRaisesRegex(AssertionError, "Corrupt cache"):
        runner.load_tokens(sources[0], metadata, data)
    print("PASS: seed grid, summaries, fixed batch order, cached/live equivalence, checkpoint selection, failures")


if __name__ == "__main__":
    with TemporaryDirectory() as directory:
        check(Path(directory))
