"""One-factor architecture ablations. Run: python -m results.synthetic_runs.sinusoid_amplitude.arch_ablations.run"""

import argparse
import csv
import json
import os
import shutil
from datetime import datetime, timezone
from importlib.metadata import version
from inspect import signature
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")
os.environ["XDG_CACHE_HOME"] = str(ROOT / ".cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from models.transformer_classifier import TransformerClassifier
from results.synthetic_runs.sinusoid_amplitude.arch_ablations.baselines import power_metrics, run_bag_of_tokens
from results.synthetic_runs.sinusoid_amplitude.arch_ablations.report import save_report
from synthetic_data.sinusoid_amplitude.generate import evaluate, make_dataset
from synthetic_data.sinusoid_amplitude.train_transformer import run_experiment


ABLATIONS = {
    "temporal_pos": ("sinusoidal", "rotary"),
    "channel_pos": ("learned", "spatial"),
    "aggregation_mode": ("CLS_token", "mean_pool_final", "attn_pool_final", "flatten_final"),
    "depth": tuple(range(2, 11)),
}
SIZES = (0.25, 0.5, *range(1, 11))


def experiment_plan(sizes=SIZES):
    """Default settings are shared references, so each configuration trains once."""
    defaults = signature(TransformerClassifier).parameters
    runs, comparisons = {"default": {}}, {}
    for parameter, values in ABLATIONS.items():
        comparisons[parameter] = {}
        for value in values:
            name = "default" if value == defaults[parameter].default else f"{parameter}/{value}"
            runs[name] = {} if name == "default" else {parameter: value}
            comparisons[parameter][str(value)] = f"1x/{name}"
    comparisons["dataset_size"] = {str(size): f"{size}x/default" for size in sizes}
    assert len(runs) == 14 and all(len(overrides) <= 1 for overrides in runs.values())
    return runs, comparisons


def save_comparison(output, comparisons, results):
    """One table and figure per sweep, with the same three baseline references."""
    rows = []
    for parameter, experiments in comparisons.items():
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        scores = {name: [] for name in ("normalize=False", "normalize=True", "svm", "kmeans", "bag_of_tokens")}
        for value, name in experiments.items():
            size = name.split("/")[0]
            models = {**results[name], **results[f"{size}/baselines"]}
            for model, metrics in models.items():
                scores[model].append(metrics["test"]["balanced_accuracy"])
                rows.append({"ablation": parameter, "value": value, "model": model,
                             "run": name if model.startswith("normalize=") else f"{size}/baselines",
                             "n_train": metrics["train"]["n_examples"], "n_test": metrics["test"]["n_examples"],
                             "train_balanced_accuracy": metrics["train"]["balanced_accuracy"],
                             "test_balanced_accuracy": metrics["test"]["balanced_accuracy"]})
        labels = list(experiments)
        if parameter == "dataset_size":
            labels = [sum(results[name]["normalize=False"][split]["n_examples"] for split in ("train", "test"))
                      for name in experiments.values()]
        positions = labels if parameter == "dataset_size" else range(len(experiments))
        for (name, values), marker, style in zip(scores.items(), ("o", "s", "^", "D", "x"), ("-", "-", "--", ":", "-.")):
            ax.plot(positions, values, marker=marker, linestyle=style, label=name)
        axis_labels = {"temporal_pos": "Temporal position", "channel_pos": "Channel position",
                       "aggregation_mode": "Aggregation mode", "depth": "Number of layers",
                       "dataset_size": "Dataset size (recordings)"}
        ax.set(xticks=positions, xticklabels=labels, ylim=(0, 1),
               xlabel=axis_labels[parameter], ylabel="Mean final test balanced accuracy")
        if parameter == "dataset_size":
            ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=0.2)
        ax.legend()
        fig.savefig(output / f"{parameter}.png", dpi=160)
        fig.savefig(output / f"{parameter}.pdf")
        plt.close(fig)
    with (output / "comparison.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def copy_run(source, output, data_path, expected):
    """Reuse only an explicitly selected, matching completed training run."""
    config = json.loads((source / "config.json").read_text())
    assert all(config[key] == value for key, value in expected.items()), (source, expected)
    metrics = json.loads((source / "metrics.json").read_text())
    assert metrics["config"] == config
    assert all(len(history) == config["epochs"] + 1 for history in metrics["histories"].values())
    shutil.copytree(source, output)
    config.update(output=str(output), data_path=str(data_path))
    metrics["config"] = config
    save_json(output / "config.json", config)
    save_json(output / "metrics.json", metrics)
    names = {"false": "normalize=False", "true": "normalize=True", "bag_of_tokens": "bag_of_tokens"}
    return {names[name]: {split: history[-1][split] for split in ("train", "test")}
            for name, history in metrics["histories"].items()}


def run_seed(output, suite, seed, reuse):
    generation, training = {**suite["generation"], "seed": seed}, {**suite["training"], "seed": seed}
    output.mkdir()
    save_json(output / "config.json", {**suite, "generation": generation, "training": training,
                                       "seeds": [seed], "split_seed": seed + 1,
                                       "default_config": "1x/default/config.json"})
    reuse_runs, reused_results = {}, {}
    if reuse is not None:
        previous = json.loads((reuse / "config.json").read_text())
        assert seed == 42 and previous["training"] == training
        assert previous["packages"] == suite["packages"]
        assert set(generation) - set(previous["generation"]) <= {"boost_channels"}
        assert all(generation[key] == value for key, value in previous["generation"].items())
        assert generation["boost_channels"] is None
        reuse_runs = {suite["comparisons"]["depth"][depth]: name
                      for depth, name in previous["comparisons"]["depth"].items()}
        reused_results = json.loads((reuse / "results.json").read_text())
        save_json(output / "reuse.json", {"source": str(reuse), "transformers": reuse_runs,
                                          "datasets_and_baselines": previous["comparisons"]["dataset_size"]})
    results = {}
    defaults = {name: signature(TransformerClassifier).parameters[name].default
                for name in ("model_dim", "heads", *ABLATIONS)}
    for size in suite["sizes"]:
        directory = output / f"{size}x"
        directory.mkdir()
        count = size * generation["samples_per_class"]
        assert count == int(count) and count >= 2, "Each size must give an integer >= 2 recordings per class"
        settings = {**generation, "samples_per_class": int(count)}
        data_path = directory / "dataset.npz"
        reuse_data = reuse is not None and size >= 1
        if reuse_data:
            assert str(size) in previous["comparisons"]["dataset_size"]
            old_settings = json.loads((reuse / f"{size}x/dataset_config.json").read_text())
            assert all(settings[key] == value for key, value in old_settings.items())
            os.link(reuse / f"{size}x/dataset.npz", data_path)
            baseline_scores = reused_results[f"{size}x/baselines"].copy()
        else:
            data = make_dataset(**settings)
            baseline_scores = power_metrics(data, evaluate(data, seed=seed + 1))
            np.savez_compressed(data_path, **data)
            del data
        save_json(directory / "dataset_config.json", settings)

        for name, overrides in (suite["runs_at_1x"] if size == 1 else {"default": {}}).items():
            key = f"{size}x/{name}"
            if key in reuse_runs:
                results[key] = copy_run(reuse / reuse_runs[key], directory / name, data_path,
                                        {**training, **defaults, **overrides})
                assert results[key] == reused_results[reuse_runs[key]]
                print(f"seed={seed}: reused {reuse_runs[key]} as {key}", flush=True)
            else:
                models, histories, _ = run_experiment(data_path, output=directory / name, **training, **overrides)
                results[key] = {f"normalize={normalize}": {split: history[-1][split] for split in ("train", "test")}
                                for normalize, history in histories.items()}
                del models
            save_json(output / "results.json", results)
        config = json.loads((directory / "default/config.json").read_text())
        if reuse_data:
            copied = copy_run(reuse / f"{size}x/baselines/bag_of_tokens", directory / "baselines/bag_of_tokens",
                              data_path, {key: config[key] for key in ("epochs", "batch_size", "seed", "optimizer_settings")})
            assert copied["bag_of_tokens"] == baseline_scores["bag_of_tokens"]
        else:
            baseline_scores["bag_of_tokens"] = run_bag_of_tokens(data_path, directory / "baselines/bag_of_tokens", config)
        save_json(directory / "baselines/metrics.json", baseline_scores)
        results[f"{size}x/baselines"] = baseline_scores
        save_json(output / "results.json", results)
    return results


def average_results(seeds):
    """Equal-weight seed means; retain individual scores in the per-seed results."""
    assert seeds and all(seed.keys() == seeds[0].keys() for seed in seeds)
    averaged = {}
    for name, models in seeds[0].items():
        assert all(seed[name].keys() == models.keys() for seed in seeds)
        averaged[name] = {}
        for model in models:
            averaged[name][model] = {}
            for split in ("train", "test"):
                metrics = [seed[name][model][split] for seed in seeds]
                assert len({item["n_examples"] for item in metrics}) == 1
                scores = [item["balanced_accuracy"] for item in metrics]
                assert all(0 <= score <= 1 for score in scores)
                averaged[name][model][split] = {"balanced_accuracy": float(np.mean(scores)),
                                                "n_examples": metrics[0]["n_examples"]}
    return averaged


def run_suite(output, epochs=30, batch_size=32, learning_rate=1e-3, seeds=(42, 43, 44),
              sizes=SIZES, reuse_seed42=None):
    assert shutil.which("pdflatex"), "Install pdflatex before training; it is required for summary.pdf"
    assert epochs >= 1 and batch_size >= 1 and np.isfinite(learning_rate) and learning_rate > 0
    assert len(set(seeds)) == len(seeds) > 0 and all(isinstance(seed, int) and seed >= 0 for seed in seeds)
    assert tuple(sorted(set(sizes))) == tuple(sizes) and 1 in sizes and min(sizes) > 0
    assert reuse_seed42 is None or 42 in seeds
    output = Path(output).resolve()
    assert output.is_relative_to(ROOT) and output != ROOT, "All artifacts belong under arch_ablations"
    runs, comparisons = experiment_plan(sizes)
    suite = {
        "generation": {name: parameter.default for name, parameter in signature(make_dataset).parameters.items()},
        "training": dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate),
        "seeds": list(seeds), "sizes": list(sizes), "runs_at_1x": runs, "comparisons": comparisons,
        "normalize": [False, True], "default_config": f"seed_{seeds[0]}/1x/default/config.json",
        "packages": {name: version(name) for name in (
            "torch", "numpy", "scipy", "scikit-learn", "matplotlib", "linear-attention-transformer",
        )}, "threads": torch.get_num_threads(),
    }
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "config.json", suite)
    results = []
    for seed in seeds:
        reuse = Path(reuse_seed42).resolve() if seed == 42 and reuse_seed42 is not None else None
        results.append(run_seed(output / f"seed_{seed}", suite, seed, reuse))
    averaged = average_results(results)
    save_json(output / "results.json", averaged)
    save_comparison(output, comparisons, averaged)
    save_report(output)
    return averaged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--reuse-seed42", type=Path, help="Completed original seed-42 suite; reuse its depth sweep and baselines")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ"))
    args = parser.parse_args()
    assert args.threads >= 1
    torch.set_num_threads(args.threads)
    with threadpool_limits(limits=args.threads):
        run_suite(args.output, args.epochs, args.batch_size, args.learning_rate, args.seeds,
                  reuse_seed42=args.reuse_seed42)
    print(f"Saved ablations: {args.output}")


if __name__ == "__main__":
    main()
