"""Compare positional normalization on raw EEG patches.

Run from the repository root: python -m synthetic_data.train_transformer
Each run saves config.json, metrics, and plots in its own results folder.
Training/evaluation live in
models/transformer_classifier.py; this file prepares data and plots diagnostics.
"""

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.transformer_classifier import TransformerClassifier, train, test


def load_data(path):
    """Use the saved recording-level split and preserve raw EEG amplitudes."""
    with np.load(path) as data:
        signals = torch.from_numpy(data["X"].copy()).float()
        labels = torch.from_numpy(data["y"].copy()).long()
        coordinates = torch.from_numpy(data["channel_positions"].copy()).float()
        time = data["time"]
        train_indices = data["train_indices"]
        test_indices = data["test_indices"]

    if tuple(signals.shape[1:]) != (19, 1280):
        raise ValueError("This setup expects 19 channels and five seconds at 256 Hz")
    if not np.allclose(time, np.arange(1280) / 256, rtol=0, atol=1e-10):
        raise ValueError("Expected a uniform 256 Hz time grid")
    if coordinates.shape != (19, 3):
        raise ValueError("Expected one XYZ coordinate per channel, in channel order")
    combined = np.concatenate((train_indices, test_indices))
    if not np.array_equal(np.sort(combined), np.arange(len(labels))):
        raise ValueError("Train/test indices must partition the recordings without overlap")

    # (B, C, 1280) -> (B, C, 10, 128): ten non-overlapping half-second windows.
    patches = signals.unfold(dimension=-1, size=128, step=128)
    training = TensorDataset(patches[train_indices], labels[train_indices])
    testing = TensorDataset(patches[test_indices], labels[test_indices])
    n_classes = int(labels.max()) + 1
    return training, testing, coordinates, n_classes


def run_experiment(data_path, epochs=30, batch_size=32, learning_rate=1e-3, seed=42, output=None):
    """Save the initial configuration, then return models, histories, and config."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    training, testing, coordinates, n_classes = load_data(data_path)
    train_evaluation = DataLoader(training, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(testing, batch_size=batch_size, shuffle=False)

    torch.manual_seed(seed)
    model_settings = dict(
        n_classes=n_classes,
        channels=19,
        coarse_time=10,
        codebook_dim=128,  # Here each input is a raw waveform patch, not a codebook lookup.
        model_dim=64,
        heads=2,
        depth=2,
        temporal_pos="sinusoidal",
        channel_pos="spatial",
        aggregation_mode="CLS_token",
    )
    base_model = TransformerClassifier(**model_settings, channel_coords=coordinates, normalize=False)
    optimizer_settings = {"lr": learning_rate, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0}
    config = {
        **model_settings,
        "model_class": "models.transformer_classifier.TransformerClassifier",
        "normalize": [False, True],
        "channel_coords": coordinates.tolist(),
        "sampling_rate": 256, "window_seconds": 0.5, "stride_samples": 128,
        "epochs": epochs, "batch_size": batch_size,
        "learning_rate": learning_rate, "optimizer": "Adam", "max_grad_norm": 1.0,
        "optimizer_settings": optimizer_settings,
        "seed": seed, "n_train": len(training), "n_test": len(testing),
        "parameters": sum(p.numel() for p in base_model.parameters()),
        "coordinate_units": "meters", "device": "cpu",
        "threads": torch.get_num_threads(), "torch_version": str(torch.__version__),
        "data_path": str(Path(data_path).resolve()),
    }
    with np.load(data_path) as data:
        config["channel_names"] = data["channel_names"].tolist()
        config["coordinate_frame"] = str(data["coordinate_frame"])
    started = datetime.now(timezone.utc)
    if output is None:
        results = Path(__file__).resolve().parents[1] / "results/synthetic_data/sinusoid_amplitude"
        output = results / started.strftime("transformer_%Y%m%dT%H%M%S_%fZ")
    output = Path(output).resolve()
    # A new folder per run protects previous experiments from accidental overwrite.
    output.mkdir(parents=True, exist_ok=False)
    config["started_at_utc"] = started.isoformat()
    config["output"] = str(output)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Initial configuration: {output / 'config.json'}", flush=True)
    print(json.dumps({**model_settings, "normalize": config["normalize"]}, indent=2), flush=True)
    print("Shapes: (B,19,1280) -> (B,19,10,128) -> (B,190,64) -> (B,191,64) -> (B,K)", flush=True)
    models, histories = {}, {}

    for normalize in config["normalize"]:
        # Same initial weights and minibatch order isolate the normalize flag.
        model = copy.deepcopy(base_model)
        model.normalize = normalize
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(training, batch_size=batch_size, shuffle=True, generator=generator)
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_settings)
        history = [{"epoch": 0, "train": test(model, train_evaluation),
                    "test": test(model, test_loader)}]

        for epoch in range(1, epochs + 1):
            optimization = train(model, train_loader, optimizer, max_grad_norm=1.0)
            # Both curves evaluate the same, fixed end-of-epoch weights.
            train_metrics = test(model, train_evaluation)
            test_metrics = test(model, test_loader)
            history.append({"epoch": epoch, "train": train_metrics,
                            "test": test_metrics, "optimization": optimization})
            print(
                f"normalize={normalize} epoch={epoch:02d}/{epochs} "
                f"CE train/test={train_metrics['cross_entropy']:.3f}/{test_metrics['cross_entropy']:.3f} "
                f"balanced accuracy={train_metrics['balanced_accuracy']:.1%}/{test_metrics['balanced_accuracy']:.1%} "
                f"gradient L2={np.mean(optimization['gradient_norms']):.3f}",
                flush=True,
            )
        models[normalize] = model
        histories[normalize] = history

    return models, histories, config


def plot_diagnostics(histories, config):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    curves, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    curves.suptitle("Raw EEG patches · 0.5 s · 19 channels · 2 heads · 2 layers")
    metrics = [("cross_entropy", "Cross-entropy (nats)"),
               ("balanced_accuracy", "Balanced accuracy"),
               ("prediction_entropy", "Prediction entropy (nats)")]
    for row, (normalize, history) in enumerate(histories.items()):
        epochs = [entry["epoch"] for entry in history]
        for column, (key, title) in enumerate(metrics):
            ax = axes[row, column]
            for split, color in (("train", "tab:blue"), ("test", "tab:orange")):
                ax.plot(epochs, [entry[split][key] for entry in history], label=split, color=color)
            baseline = 1 / config["n_classes"] if key == "balanced_accuracy" else np.log(config["n_classes"])
            ax.axhline(baseline, color="gray", ls=":", label="Uniform predictions")
            ax.set(title=f"normalize={normalize} · {title}", xlabel="Epoch")
            if key == "balanced_accuracy":
                ax.set_ylim(0, 1)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)

    gradients, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
    gradients.suptitle("Gradient checks · measured before clipping at global L2 = 1")
    for column, (normalize, history) in enumerate(histories.items()):
        steps = [norm for entry in history[1:] for norm in entry["optimization"]["gradient_norms"]]
        axes[0, column].plot(np.arange(1, len(steps) + 1), steps, lw=0.8)
        axes[0, column].axhline(1, color="gray", ls=":", label="Clipping threshold")
        axes[0, column].set(title=f"normalize={normalize}", xlabel="Optimizer step",
                            ylabel="Global gradient L2 norm", yscale="log")
        axes[0, column].legend()
        names = list(history[1]["optimization"]["parameter_gradient_norms"])
        matrix = np.array([[entry["optimization"]["parameter_gradient_norms"][name]
                            for entry in history[1:]] for name in names])
        heatmap = axes[1, column].imshow(np.maximum(matrix, 1e-12), aspect="auto",
                                          norm=LogNorm(vmin=1e-8, vmax=max(1, matrix.max())), cmap="viridis")
        axes[1, column].set(yticks=np.arange(len(names)), yticklabels=names,
                            xlabel="Epoch", title="Mean gradient L2 per parameter")
        tick_indices = np.unique(np.linspace(0, len(history) - 2, min(6, len(history) - 1)).astype(int))
        axes[1, column].set_xticks(tick_indices, tick_indices + 1)
        axes[1, column].tick_params(axis="y", labelsize=6)
        gradients.colorbar(heatmap, ax=axes[1, column], label="Gradient L2")

    confusions, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    confusions.suptitle("Final-epoch predictions · rows = true class")
    for column, (normalize, history) in enumerate(histories.items()):
        for row, split in enumerate(("train", "test")):
            metrics = history[-1][split]
            matrix = np.asarray(metrics["confusion_matrix"])
            ax = axes[row, column]
            ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max())
            for i, j in np.ndindex(matrix.shape):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                        color="white" if matrix[i, j] > matrix.max() / 2 else "black")
            ax.set(title=f"normalize={normalize} · {split} · BA={metrics['balanced_accuracy']:.1%}",
                   xlabel="Predicted class", ylabel="True class",
                   xticks=range(config["n_classes"]), yticks=range(config["n_classes"]))
    return {"learning_curves": curves, "gradients": gradients, "confusion_matrices": confusions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "outputs/dataset.npz")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path,
                        help="New run directory; defaults to a dated folder in results/synthetic_data/sinusoid_amplitude")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    import matplotlib
    matplotlib.use("Agg")

    models, histories, config = run_experiment(
        args.data, args.epochs, args.batch_size, args.learning_rate, args.seed, args.output,
    )
    figures = plot_diagnostics(histories, config)
    output = Path(config["output"])
    for name, figure in figures.items():
        figure.savefig(output / f"{name}.png", dpi=160)
    (output / "metrics.json").write_text(json.dumps({"config": config, "histories": histories}, indent=2) + "\n")
    print(f"Config, plots, and metrics: {output}")


if __name__ == "__main__":
    main()
