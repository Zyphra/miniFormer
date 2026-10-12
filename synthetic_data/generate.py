"""Generate K-class EEG-like signals with adjustable dominant-frequency spacing."""

import argparse
import json
from pathlib import Path

import numpy as np


def frequency_power(signals, sampling_rate, frequencies):
    """Squared Fourier projections at exact frequencies, including off-grid tones.

    These equal FFT-bin power for bin-aligned frequencies. Closely spaced
    components contribute to each other's projections over a finite window.
    """
    n_times = signals.shape[-1]
    time = np.arange(n_times) / sampling_rate
    basis = np.exp(-2j * np.pi * time[:, None] * np.asarray(frequencies))
    spectrum = signals[:, 0, :] @ basis
    return 2 * np.abs(spectrum) ** 2 / n_times**2


def make_dataset(
    n_classes=4,
    samples_per_class=200,
    sampling_rate=250,
    duration=5.0,
    dominant_amplitude=2.0,
    background_amplitude=0.5,
    amplitude_jitter=0.1,
    noise_std=0.5,
    seed=42,
    frequency_start=10.0,
    frequency_spacing=0.075,
):
    """Each class boosts a different frequency; phase and noise vary per sample.

    Returns a dictionary suitable for np.savez_compressed. X has shape
    (examples, 1, time), y has integer labels 0..K-1, and power is (examples, K).
    Class k boosts frequency_start + k * frequency_spacing Hz. Close tones
    overlap in power space; separation is not guaranteed even without noise.
    """
    if not isinstance(n_classes, (int, np.integer)) or n_classes < 2:
        raise ValueError("n_classes must be an integer >= 2")
    if not isinstance(samples_per_class, (int, np.integer)) or samples_per_class < 2:
        raise ValueError("samples_per_class must be an integer >= 2")
    parameters = [sampling_rate, duration, dominant_amplitude,
                  background_amplitude, amplitude_jitter, noise_std,
                  frequency_start, frequency_spacing]
    if not np.all(np.isfinite(parameters)):
        raise ValueError("All signal parameters must be finite")
    if sampling_rate <= 0 or duration <= 0:
        raise ValueError("sampling_rate and duration must be positive")
    n_times = round(sampling_rate * duration)
    if n_times < 2 or not np.isclose(n_times, sampling_rate * duration, rtol=0, atol=1e-8):
        raise ValueError("sampling_rate * duration must be an integer >= 2")
    if noise_std < 0 or not 0 <= amplitude_jitter <= background_amplitude:
        raise ValueError("Require noise_std >= 0 and 0 <= jitter <= background amplitude")
    if dominant_amplitude - background_amplitude <= 2 * amplitude_jitter:
        raise ValueError("Dominant amplitude must exceed background + 2 * jitter")

    if frequency_start <= 0 or frequency_spacing <= 0:
        raise ValueError("frequency_start and frequency_spacing must be positive")
    # Preserve sub-bin spacing instead of snapping multiple classes to one bin.
    frequencies = frequency_start + frequency_spacing * np.arange(n_classes)
    if (not np.all(np.isfinite(frequencies))
            or np.any(frequencies >= sampling_rate / 2)
            or len(np.unique(frequencies)) != n_classes):
        raise ValueError("Class frequencies must be distinct and below the Nyquist frequency")
    time = np.arange(n_times) / sampling_rate
    rng = np.random.default_rng(seed)
    labels = rng.permutation(np.repeat(np.arange(n_classes), samples_per_class))
    class_amplitudes = np.full((n_classes, n_classes), background_amplitude, dtype=float)
    np.fill_diagonal(class_amplitudes, dominant_amplitude)
    amplitudes = class_amplitudes[labels] + rng.uniform(
        -amplitude_jitter, amplitude_jitter, size=(len(labels), n_classes)
    )
    phases = rng.uniform(0, 2 * np.pi, size=amplitudes.shape)
    signals = rng.normal(0, noise_std, size=(len(labels), n_times))
    for j, frequency in enumerate(frequencies):
        signals += amplitudes[:, j, None] * np.sin(
            2 * np.pi * frequency * time[None, :] + phases[:, j, None]
        )
    signals = signals[:, None, :].astype(np.float32)
    return {
        "X": signals,
        "y": labels,
        "power": frequency_power(signals, sampling_rate, frequencies),
        "time": time,
        "frequencies": frequencies,
        "class_amplitudes": class_amplitudes,
    }


def stratified_split(labels, seed=43):
    """Split independent examples approximately 80/20 within each class."""
    rng = np.random.default_rng(seed)
    train, test = [], []
    for label in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == label))
        n_test = min(len(indices) - 1, max(1, round(0.2 * len(indices))))
        test.extend(indices[:n_test])
        train.extend(indices[n_test:])
    return rng.permutation(train), rng.permutation(test)


def evaluate(data, seed=43):
    """Fit nearest class means using training power only; return linear scores."""
    train, test = stratified_split(data["y"], seed)
    power, labels = data["power"], data["y"]
    n_classes = len(data["frequencies"])
    means = np.array([power[train][labels[train] == k].mean(axis=0)
                      for k in range(n_classes)])
    # argmin ||p - mean_k||^2 = argmax (2 * mean_k @ p - ||mean_k||^2).
    weights = 2 * means
    bias = -(means**2).sum(axis=1)
    predictions = (power @ weights.T + bias).argmax(axis=1)
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(confusion, (labels[test], predictions[test]), 1)
    other_power = power.copy()
    other_power[np.arange(len(labels)), labels] = -np.inf
    margins = power[np.arange(len(labels)), labels] - other_power.max(axis=1)
    data.update(train_indices=train, test_indices=test,
                linear_weights=weights, linear_bias=bias)
    return {
        "train_accuracy": float(np.mean(predictions[train] == labels[train])),
        "test_accuracy": float(np.mean(predictions[test] == labels[test])),
        "argmax_power_test_accuracy": float(np.mean(power[test].argmax(axis=1) == labels[test])),
        "minimum_power_margin_all_examples": float(margins.min()),
        "all_examples_separated_by_argmax_power": bool(np.all(margins > 0)),
        "confusion_matrix_test": confusion.tolist(),
    }


def plot_dataset(data, metrics, sampling_rate, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    signals, labels, power = data["X"][:, 0, :], data["y"], data["power"]
    frequencies, time = data["frequencies"], data["time"]
    n_classes = len(frequencies)
    colors = plt.cm.tab10(np.arange(n_classes) % 10)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    fig.suptitle(f"Synthetic EEG-like signals · {n_classes} classes · {len(labels):,} examples", fontsize=17)

    # Cap waveform/spectrum overlays to keep a large-K run readable.
    shown = min(n_classes, 8)
    visible = time < min(1.0, time[-1] + 1 / sampling_rate)
    example_indices = [np.flatnonzero(labels == k)[0] for k in range(shown)]
    offsets = np.arange(shown)[::-1] * 2.5 * np.max(np.abs(signals[example_indices][:, visible]))
    for k, index in enumerate(example_indices):
        axes[0, 0].plot(time[visible], signals[index, visible] + offsets[k], color=colors[k], lw=0.8)
    axes[0, 0].set(yticks=offsets, yticklabels=[f"Class {k}: {frequencies[k]:g} Hz" for k in range(shown)],
                   xlabel="Time (s)")

    # Zero padding samples the spectrum more densely for display; it does not
    # add frequency resolution or change the features used by the classifier.
    n_fft = 8 * len(time)
    fft_frequencies = np.fft.rfftfreq(n_fft, 1 / sampling_rate)
    display_power = 2 * np.abs(np.fft.rfft(signals, n=n_fft, axis=-1)) ** 2 / len(time)**2
    padding = max(0.5, 0.2 * np.ptp(frequencies), 2 * sampling_rate / len(time))
    for k in range(shown):
        axes[0, 1].plot(fft_frequencies, display_power[labels == k].mean(axis=0),
                            color=colors[k], label=f"Class {k}", alpha=0.85)
        axes[0, 1].axvline(frequencies[k], color=colors[k], ls=":", alpha=0.4)
    axes[0, 1].set(xlim=(max(0, frequencies.min() - padding),
                         min(sampling_rate / 2, frequencies.max() + padding)),
                   xlabel="Frequency (Hz)", ylabel="Mean spectral projection power (a.u.²)",
                   title="Class-average spectra")
    axes[0, 1].legend(ncols=2, fontsize=8)

    train, test = data["train_indices"], data["test_indices"]
    center = power[train].mean(axis=0)
    _, singular, components = np.linalg.svd(power[train] - center, full_matrices=False)
    projected = (power[test] - center) @ components[:2].T
    explained = singular**2 / (singular**2).sum()
    for k in range(n_classes):
        points = projected[labels[test] == k]
        axes[1, 0].scatter(points[:, 0], points[:, 1], color=colors[k], s=14, alpha=0.65, label=f"Class {k}")
    axes[1, 0].set(xlabel=f"PC 1 ({explained[0]:.1%})", ylabel=f"PC 2 ({explained[1]:.1%})",
                   title="Held-out power features · PCA fitted on training data")
    if n_classes <= 8:
        axes[1, 0].legend(ncols=2, fontsize=8)

    explanation = axes[1, 1]
    explanation.set_axis_off()
    explanation.set_title("Signal generation")
    explanation.text(
        0.5, 0.84,
        r"$x_i(t) = \sum_{j=0}^{K-1} A_{ij}\,\sin(2\pi f_j t + \phi_{ij}) + \epsilon_i(t)$",
        transform=explanation.transAxes, ha="center", va="center", fontsize=17,
    )
    definitions = [
        (r"$K$", "Number of classes and sinusoidal components"),
        (r"$f_j$", "Component frequency, fixed across the dataset"),
        (r"$A_{ij}$", "Component amplitude in example i"),
        (r"$\phi_{ij}$", "Random phase per example and component"),
        (r"$\epsilon_i(t)$", "Additive Gaussian noise"),
    ]
    for row, (symbol, description) in enumerate(definitions):
        y = 0.64 - 0.105 * row
        explanation.text(0.05, y, symbol, transform=explanation.transAxes,
                         va="center", fontsize=13)
        explanation.text(0.19, y, description, transform=explanation.transAxes,
                         va="center", fontsize=11)
    explanation.text(0.05, 0.06, r"Label $y_i$ selects which amplitude is boosted.",
                     transform=explanation.transAxes, va="center", fontsize=12)
    fig.savefig(output / "overview.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    confusion = np.asarray(metrics["confusion_matrix_test"])
    heatmap = ax.imshow(confusion, cmap="Blues", vmin=0)
    if n_classes <= 12:
        for row, column in np.ndindex(confusion.shape):
            ax.text(column, row, str(confusion[row, column]), ha="center", va="center",
                    color="white" if confusion[row, column] > confusion.max() / 2 else "black")
        ax.set(xticks=np.arange(n_classes), yticks=np.arange(n_classes))
    ax.set(xlabel="Predicted class", ylabel="True class",
           title=f"Linear classifier on spectral power · test accuracy {metrics['test_accuracy']:.1%}")
    fig.colorbar(heatmap, ax=ax, label="Examples")
    fig.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-classes", type=int, default=4)
    parser.add_argument("--samples-per-class", type=int, default=200)
    parser.add_argument("--sampling-rate", type=float, default=250)
    parser.add_argument("--duration", type=float, default=5)
    parser.add_argument("--frequency-start", type=float, default=10,
                        help="First class frequency in Hz (default: 10)")
    parser.add_argument("--frequency-spacing", type=float, default=0.075,
                        help="Spacing between class frequencies in Hz (default: 0.075)")
    parser.add_argument("--dominant-amplitude", type=float, default=2)
    parser.add_argument("--background-amplitude", type=float, default=0.5)
    parser.add_argument("--amplitude-jitter", type=float, default=0.1)
    parser.add_argument("--noise-std", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "outputs")
    args = parser.parse_args()
    config = vars(args).copy()
    output = config.pop("output")
    data = make_dataset(**config)
    metrics = evaluate(data, seed=args.seed + 1)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "dataset.npz", **data)
    report = {"config": config, "frequencies_hz": data["frequencies"].tolist(),
              "fft_bin_spacing_hz": args.sampling_rate / data["X"].shape[-1],
              "shape": list(data["X"].shape), **metrics}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_dataset(data, metrics, args.sampling_rate, output)
    print(json.dumps(report, indent=2))
    print(f"Saved dataset, summary, and plots to {output.resolve()}")


if __name__ == "__main__":
    main()
