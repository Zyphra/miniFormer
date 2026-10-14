"""Compare positional normalization on raw EEG patches.

Run from the repository root: python -m synthetic_data.sinusoid_amplitude.train_transformer
Each run saves config.json, metrics, and plots in its own results folder.
Training/evaluation live in
models/transformer_classifier.py; this file prepares data and plots diagnostics.
"""

import argparse
import copy
import json
from datetime import datetime, timezone
from inspect import signature
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


def fit(model, training, testing, config, loss_function=torch.nn.functional.cross_entropy):
    """Shared epoch loop; reset randomness and minibatch order for every model."""
    assert config["epochs"] >= 1 and config["batch_size"] >= 1
    torch.manual_seed(config["seed"])
    train_loader = DataLoader(training, batch_size=config["batch_size"], shuffle=True,
                              generator=torch.Generator().manual_seed(config["seed"]))
    train_evaluation = DataLoader(training, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(testing, batch_size=config["batch_size"], shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), **config["optimizer_settings"])
    history = [{"epoch": 0, "train": test(model, train_evaluation),
                "test": test(model, test_loader)}]
    for epoch in range(1, config["epochs"] + 1):
        optimization = train(model, train_loader, optimizer, max_grad_norm=config["max_grad_norm"],
                             loss_function=loss_function)
        assert not optimization["missing_gradients"], optimization["missing_gradients"]
        train_metrics, test_metrics = test(model, train_evaluation), test(model, test_loader)
        history.append({"epoch": epoch, "train": train_metrics,
                        "test": test_metrics, "optimization": optimization})
        print(
            f"epoch={epoch:02d}/{config['epochs']} "
            f"CE train/test={train_metrics['cross_entropy']:.3f}/{test_metrics['cross_entropy']:.3f} "
            f"balanced accuracy={train_metrics['balanced_accuracy']:.1%}/{test_metrics['balanced_accuracy']:.1%} "
            f"gradient L2={np.mean(optimization['gradient_norms']):.3f}", flush=True,
        )
    return history


def run_experiment(data_path, epochs=30, batch_size=32, learning_rate=1e-3, seed=42,
                   output=None, **model_overrides):
    """Save the initial configuration, then return models, histories, and config."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    training, testing, coordinates, n_classes = load_data(data_path)
    assert set(model_overrides) <= {"temporal_pos", "channel_pos", "aggregation_mode", "depth"}

    torch.manual_seed(seed)
    model_settings = dict(
        n_classes=n_classes,
        channels=19,
        coarse_time=10,
        codebook_dim=128,  # Here each input is a raw waveform patch, not a codebook lookup.
        **{name: signature(TransformerClassifier).parameters[name].default for name in
           ("model_dim", "heads", "depth", "temporal_pos", "channel_pos", "aggregation_mode")},
    )
    model_settings.update(model_overrides)
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
        results = Path(__file__).resolve().parents[2] / "results/synthetic_runs/sinusoid_amplitude"
        output = results / started.strftime("transformer_%Y%m%dT%H%M%S_%fZ")
    output = Path(output).resolve()
    # A new folder per run protects previous experiments from accidental overwrite.
    output.mkdir(parents=True, exist_ok=False)
    config["started_at_utc"] = started.isoformat()
    config["output"] = str(output)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Initial configuration: {output / 'config.json'}", flush=True)
    print(json.dumps({**model_settings, "normalize": config["normalize"]}, indent=2), flush=True)
    models, histories = {}, {}

    for normalize in config["normalize"]:
        # Same initial weights and minibatch order isolate the normalize flag.
        model = copy.deepcopy(base_model)
        model.normalize = normalize
        print(f"normalize={normalize}", flush=True)
        models[normalize] = model
        histories[normalize] = fit(model, training, testing, config)

    save_diagnostics(histories, config)
    return models, histories, config


def plot_diagnostics(histories, config):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    histories = {f"normalize={key}" if isinstance(key, bool) else key: value
                 for key, value in histories.items()}
    n_variants = len(histories)
    curves, axes = plt.subplots(n_variants, 3, figsize=(13, 3.5 * n_variants),
                               constrained_layout=True, squeeze=False)
    curves.suptitle(config["model_class"])
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
            ax.set(title=f"{normalize} · {title}", xlabel="Epoch")
            if key == "balanced_accuracy":
                ax.set_ylim(0, 1)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)

    n_parameters = len(next(iter(histories.values()))[1]["optimization"]["parameter_gradient_norms"])
    gradients, axes = plt.subplots(2, n_variants, figsize=(7 * n_variants, max(12, n_parameters * 0.18)),
                                  constrained_layout=True, squeeze=False)
    gradients.suptitle("Gradient checks · measured before clipping at global L2 = 1")
    for column, (normalize, history) in enumerate(histories.items()):
        steps = [norm for entry in history[1:] for norm in entry["optimization"]["gradient_norms"]]
        axes[0, column].plot(np.arange(1, len(steps) + 1), steps, lw=0.8, marker=".", markersize=2)
        axes[0, column].axhline(1, color="gray", ls=":", label="Clipping threshold")
        axes[0, column].set(title=normalize, xlabel="Optimizer step",
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

    confusions, axes = plt.subplots(2, n_variants, figsize=(4.5 * n_variants, 8),
                                   constrained_layout=True, squeeze=False)
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
            ax.set(title=f"{normalize} · {split} · BA={metrics['balanced_accuracy']:.1%}",
                   xlabel="Predicted class", ylabel="True class",
                   xticks=range(config["n_classes"]), yticks=range(config["n_classes"]))
    return {"learning_curves": curves, "gradients": gradients, "confusion_matrices": confusions}


def save_diagnostics(histories, config):
    import matplotlib.pyplot as plt

    output = Path(config["output"])
    (output / "metrics.json").write_text(
        json.dumps({"config": config, "histories": histories}, indent=2, allow_nan=False) + "\n"
    )
    for name, figure in plot_diagnostics(histories, config).items():
        figure.savefig(output / f"{name}.png", dpi=160)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "outputs/dataset.npz")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path,
                        help="New run directory; defaults to a dated folder in results/synthetic_runs/sinusoid_amplitude")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    import matplotlib
    matplotlib.use("Agg")

    _, _, config = run_experiment(
        args.data, args.epochs, args.batch_size, args.learning_rate, args.seed, args.output,
    )
    output = Path(config["output"])
    print(f"Config, plots, and metrics: {output}")


if __name__ == "__main__":
    main()
