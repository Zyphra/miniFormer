"""Train one transformer per cached or live token source, using fixed dataset splits."""

import argparse
import csv
import hashlib
import importlib
from importlib.metadata import version
from inspect import signature
import json
import math
import os
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

from models.transformer_classifier import TransformerClassifier, train, test
from tokenization.pipeline import batches, sha256


def read_labels(path, example_ids=None):
    """Resolve requested examples through recording metadata; never create splits."""
    data = json.loads(Path(path).read_text())
    assert data["schema_version"] == 2
    recordings = {r["recording_id"]: r for r in data["recordings"]}
    windows = {w["example_id"]: w for w in data["windows"]}
    assert len(recordings) == len(data["recordings"]) and len(windows) == len(data["windows"])
    requested = sorted(windows) if example_ids is None else sorted(example_ids)
    assert requested and len(set(requested)) == len(requested), "Empty or duplicate example IDs"
    assert all(type(i) is int and i >= 0 for i in requested)
    assert set(requested) <= windows.keys(), "Unknown example IDs"
    classes = list(range(len(data["label_mapping"])))
    assert len(classes) >= 2 and sorted(data["label_mapping"].values()) == classes
    rows = [recordings[windows[i]["recording_id"]] for i in requested]
    labels = torch.tensor([r["label"] for r in rows], dtype=torch.long)
    assert all(type(r["label"]) is int and r["label"] in classes for r in rows)
    assert all(r["split"] in ("train", "val", "test") for r in recordings.values())
    subjects = [{r["subject"] for r in recordings.values() if r["split"] == s}
                for s in ("train", "val", "test")]
    assert sum(map(len, subjects)) == len(set.union(*subjects)), "Subject crosses splits"
    splits = {s: torch.tensor([i for i, r in enumerate(rows) if r["split"] == s])
              for s in ("train", "val", "test")}
    for s, indices in splits.items():
        assert len(indices) and labels[indices].unique().tolist() == classes, f"Missing classes in {s}"
    signal = data["signal_metadata"]
    assert len(set(signal["channel_names"])) == len(signal["channel_names"])
    coords = torch.tensor(signal["channel_positions_m"], dtype=torch.float32)
    assert coords.shape == (len(signal["channel_names"]), 3) and torch.isfinite(coords).all()
    return data, torch.tensor(requested), labels, splits


def load_tokens(source, dataset_metadata, data, device="cpu", batch_size=32):
    """Return CPU IDs, example IDs, codebook and provenance from one explicit source."""
    dataset_metadata = Path(dataset_metadata).resolve()
    dataset_hash = sha256(dataset_metadata)
    channels = data["signal_metadata"]["channel_names"]
    if "cache" in source:
        assert set(source) == {"name", "cache"}
        cache = Path(source["cache"]).resolve()
        metadata = json.loads((cache / "metadata.json").read_text())
        assert metadata["schema_version"] == 1
        assert metadata["dataset_sha256"] == dataset_hash, "Dataset metadata differs from token cache"
        assert metadata["channel_names"] == channels, "Channel order differs from token cache"
        chunks = []
        for filename, digest in metadata["files"].items():
            assert Path(filename).name == filename and filename.endswith(".pt")
            assert sha256(cache / filename) == digest, f"Corrupt cache file: {filename}"
            if filename != "codebook.pt":
                chunks.append(torch.load(cache / filename, map_location="cpu", weights_only=True))
        assert "codebook.pt" in metadata["files"]
        codebook = torch.load(cache / "codebook.pt", map_location="cpu", weights_only=True)
        assert list(codebook.shape) == metadata["codebook_shape"]
        metadata = {**metadata, "cache": str(cache), "metadata_sha256": sha256(cache / "metadata.json")}
    else:
        assert set(source) == {"name", "tokenizer", "repo", "checkpoint"}
        assert source["tokenizer"] in ("dac", "tfm", "labram")
        assert dataset_metadata.name == "cache.json", "Live input must be the EEG cache's cache.json"
        from tokenization.cache import provenance

        adapter = importlib.import_module(f"tokenization.{source['tokenizer']}")
        tokenizer = adapter.Tokenizer(source["repo"], source["checkpoint"], device)
        metadata = {"provenance": provenance(source["repo"], source["checkpoint"]),
                    "tokenizer": source["tokenizer"], "preprocessing": tokenizer.metadata,
                    "dataset_sha256": dataset_hash, "channel_names": channels,
                    "tokens_per_window": tokenizer.tokens_per_window}
        chunks = list(batches(dataset_metadata.parent, tokenizer, batch_size))
        codebook = tokenizer.codebook
    assert chunks, "No token examples"
    assert metadata["tokens_per_window"] > 0
    assert codebook.ndim == 2 and min(codebook.shape) > 0 and codebook.dtype == torch.float32
    assert torch.isfinite(codebook).all()
    for chunk in chunks:
        ids, examples = chunk["ids"], chunk["example_ids"]
        assert examples.ndim == 1 and examples.dtype == torch.int64 and len(examples) > 0
        assert ids.dtype == torch.int64
        assert ids.shape == (len(examples), len(channels), metadata["tokens_per_window"])
        assert 0 <= ids.min() and ids.max() < len(codebook), "Token ID outside codebook"
    ids = torch.cat([c["ids"] for c in chunks])
    examples = torch.cat([c["example_ids"] for c in chunks])
    order = examples.argsort()
    examples = examples[order]
    assert len(examples.unique()) == len(examples), "Duplicate token example IDs"
    assert set(examples.tolist()) <= {w["example_id"] for w in data["windows"]}, "Unknown token example IDs"
    return ids[order], examples, codebook, metadata


def run(sources, dataset_metadata, output, epochs=30, batch_size=32, learning_rate=1e-3,
        seed=42, device="cpu", example_ids=None):
    assert epochs > 0 and batch_size > 0 and math.isfinite(learning_rate) and learning_rate > 0
    names = [s["name"] for s in sources]
    assert names and len(set(names)) == len(names), "Duplicate source names"
    assert all(n and n not in (".", "..") and Path(n).name == n for n in names)
    dataset_metadata, output = Path(dataset_metadata).resolve(), Path(output).resolve()
    assert not output.exists(), f"Output already exists: {output}"
    data, requested, labels, splits = read_labels(dataset_metadata, example_ids)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    output.mkdir(parents=True)
    comparison = []
    for source in sources:
        torch.manual_seed(seed)
        ids, examples, codebook, metadata = load_tokens(source, dataset_metadata, data, device, batch_size)
        indices = torch.searchsorted(examples, requested)
        assert (indices < len(examples)).all(), "Missing requested token examples"
        assert torch.equal(examples[indices], requested), "Missing requested token examples"
        ids = ids[indices]
        # Tokenizer initialization must not change classifier initialization or minibatch order.
        torch.manual_seed(seed)
        classifier_args = signature(TransformerClassifier).bind(
            n_classes=len(data["label_mapping"]), channels=ids.shape[1],
            coarse_time=ids.shape[2], codebook_dim=codebook.shape[1],
            channel_coords=data["signal_metadata"]["channel_positions_m"])
        classifier_args.apply_defaults()
        classifier_config = classifier_args.arguments
        classifier = TransformerClassifier(**{**classifier_config,
            "channel_coords": torch.tensor(classifier_config["channel_coords"], dtype=torch.float32)})
        model = torch.nn.Sequential(torch.nn.Embedding.from_pretrained(codebook.clone(), freeze=True),
                                    classifier).to(device)
        loaders = {s: DataLoader(TensorDataset(ids[i], labels[i]), batch_size=batch_size,
                    shuffle=s == "train", generator=torch.Generator().manual_seed(seed))
                   for s, i in splits.items()}
        optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
        destination = output / source["name"]
        destination.mkdir()
        config = {"source": source, "tokenization": metadata, "dataset_metadata": str(dataset_metadata),
            "dataset_sha256": sha256(dataset_metadata), "label_mapping": data["label_mapping"],
            "split_example_ids": {s: requested[i].tolist() for s, i in splits.items()},
            "classifier": classifier_config, "epochs": epochs, "batch_size": batch_size,
            "seed": seed, "device": device, "optimizer": {"name": "Adam", **optimizer.defaults},
            "deterministic_algorithms": True, "torch_num_threads": torch.get_num_threads(),
            "max_grad_norm": 1.0, "selection": "highest validation balanced accuracy; earliest tie",
            "python": sys.version, "packages": {n: version(n) for n in ("torch", "linear_attention_transformer")},
            "source_sha256": {str(p.resolve()): sha256(p) for p in (
                Path(__file__), Path(__file__).parents[1] / "models/transformer_classifier.py")},
            "codebook_tensor_sha256": hashlib.sha256(codebook.numpy().tobytes()).hexdigest(),
            "ids_tensor_sha256": hashlib.sha256(ids.numpy().tobytes()).hexdigest()}
        (destination / "config.json").write_text(json.dumps(config, indent=2, allow_nan=False) + "\n")
        history, best_score, best_epoch = [], -1.0, 0
        for epoch in range(1, epochs + 1):
            training = train(model, loaders["train"], optimizer)
            assert not training["missing_gradients"], training["missing_gradients"]
            validation = test(model, loaders["val"])
            history.append({"epoch": epoch, "train": training, "val": validation})
            score = validation["balanced_accuracy"]
            assert math.isfinite(score)
            if score > best_score:
                best_score, best_epoch = score, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()}
            print(f"{source['name']} epoch={epoch}/{epochs} val_balanced_accuracy={score:.6f}", flush=True)
        assert best_epoch > 0
        classifier.load_state_dict(best_state, strict=True)
        testing = test(model, loaders["test"])
        assert torch.equal(model[0].weight.detach().cpu(), codebook), "Classifier changed the codebook"
        torch.save(best_state, destination / "classifier.pt")
        metrics = {"best_epoch": best_epoch, "val": history[best_epoch - 1]["val"],
                   "test": testing, "history": history}
        (destination / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
        comparison.append({"source": source["name"], "best_epoch": best_epoch,
            "val_balanced_accuracy": best_score, "test_balanced_accuracy": testing["balanced_accuracy"],
            "test_cross_entropy": testing["cross_entropy"], "test_examples": testing["n_examples"]})
        del model, classifier, optimizer, loaders, ids, codebook
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=json.loads, nargs="+", required=True,
                        help='JSON objects: {"name":...,"cache":...} or {"name":...,"tokenizer":...,"repo":...,"checkpoint":...}')
    parser.add_argument("--dataset-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--example-ids", type=int, nargs="+")
    run(**vars(parser.parse_args()))
