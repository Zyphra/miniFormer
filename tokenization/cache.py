"""Optional disk consumer of tokenization.pipeline.batches."""

import argparse
import importlib
from importlib.metadata import version
from itertools import groupby
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import torch

from .pipeline import batches, read_dataset, sha256


def provenance(repo, checkpoint):
    repo, checkpoint = Path(repo).resolve(), Path(checkpoint).resolve()
    git = ["git", f"--git-dir={repo / '.git'}", f"--work-tree={repo}"]
    files = subprocess.check_output(git + ["ls-files", "*.py"], text=True)
    sources = {str(repo / p): sha256(repo / p) for p in files.splitlines()}
    # DAC also imports its separately installed audiotools checkout.
    for name, module in list(sys.modules.items()):
        if name.startswith("audiotools") and getattr(module, "__file__", None):
            sources[module.__file__] = sha256(module.__file__)
    sources.update({str(p): sha256(p) for p in Path(__file__).parent.glob("*.py")})
    return {"repository": str(repo), "revision": subprocess.check_output(
                git + ["rev-parse", "HEAD"], text=True).strip(),
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
            "source_sha256": sources, "python": sys.version,
            "packages": {name: version(name) for name in (
                "torch", "numpy", "scipy", "mne", "timm", "einops", "linear_attention_transformer")
                if name in sys.modules or name == "mne"}}


def write_cache(dataset, tokenizer, output, run_metadata, batch_size=32):
    """Publish a complete cache atomically; existing output is an error."""
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    assert not output.exists(), f"Cache already exists: {output}"
    data = read_dataset(dataset)
    metadata = {"schema_version": 1, "provenance": run_metadata,
                "dataset": str(dataset), "dataset_sha256": sha256(dataset / "cache.json"),
                "preprocessing": tokenizer.metadata, "channel_names": data["signal_metadata"]["channel_names"],
                "tokens_per_window": tokenizer.tokens_per_window,
                "device": str(tokenizer.device), "batch_size": batch_size, "files": {}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".tokens-", dir=output.parent) as scratch:
        staging = Path(scratch) / "data"
        staging.mkdir()
        codebook = tokenizer.codebook
        torch.save(codebook, staging / "codebook.pt")
        metadata["codebook_shape"] = list(codebook.shape)
        metadata["files"]["codebook.pt"] = sha256(staging / "codebook.pt")
        seen = []
        for recording_id, group in groupby(batches(dataset, tokenizer, batch_size),
                                           key=lambda batch: batch["recording_id"]):
            chunks = list(group)
            result = {key: torch.cat([b[key] for b in chunks]) for key in ("ids", "example_ids")}
            seen.extend(result["example_ids"].tolist())
            filename = f"{recording_id:03d}.pt"
            assert not (staging / filename).exists()
            torch.save(result, staging / filename)
            metadata["files"][filename] = sha256(staging / filename)
            print(f"{filename}: {tuple(result['ids'].shape)}", flush=True)
        assert sorted(seen) == list(range(len(data["windows"]))), "Missing or duplicated examples"
        assert torch.equal(codebook, tokenizer.codebook)
        (staging / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        assert not output.exists(), f"Cache already exists: {output}"
        staging.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokenizer", choices=("dac", "tfm", "labram"))
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("dataset cache/depression/data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    assert not args.output.exists(), f"Cache already exists: {args.output}"
    module = importlib.import_module(f"tokenization.{args.tokenizer}")
    window = read_dataset(args.dataset)["windows"][0]
    options = {"window_samples": window["stop_sample"] - window["start_sample"]} \
        if args.tokenizer == "dac" else {}
    tokenizer = module.Tokenizer(args.repo, args.checkpoint, args.device, **options)
    run_metadata = provenance(args.repo, args.checkpoint)
    run_metadata["tokenizer"] = args.tokenizer
    write_cache(args.dataset, tokenizer, args.output, run_metadata, args.batch_size)


if __name__ == "__main__":
    main()
