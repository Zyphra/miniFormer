"""Generate K-class EEG-like signals with adjustable dominant-frequency spacing."""

import argparse
import json
from pathlib import Path

import numpy as np


def frequency_power(signals, sampling_rate, frequencies):
    """Return (N, channels, frequencies) power from (N, channels, time) signals.

    These equal FFT-bin power for bin-aligned frequencies. Closely spaced
    components contribute to each other's projections over a finite window.
    """
    n_times = signals.shape[-1]
    time = np.arange(n_times) / sampling_rate
    basis = np.exp(-2j * np.pi * time[:, None] * np.asarray(frequencies))
    spectrum = signals @ basis
    return 2 * np.abs(spectrum) ** 2 / n_times**2


def channel_model(spatial_length=0.08, noise_spatial_fraction=0.5):
    """Load 19 MNE 10-20 positions and construct class-independent spatial gains.

    spatial_length is in meters. noise_spatial_fraction controls the spatially
    correlated fraction of noise variance, not the total signal correlation.
    """
    if not np.isfinite(spatial_length) or spatial_length <= 0:
        raise ValueError("spatial_length must be finite and positive (meters)")
    if not np.isfinite(noise_spatial_fraction) or not 0 <= noise_spatial_fraction <= 1:
        raise ValueError("noise_spatial_fraction must be between 0 and 1")
    montage = json.loads(Path(__file__).with_name("standard_1020.json").read_text())
    names = np.asarray(montage["channel_names"])
    positions = np.asarray(montage["channel_positions"])
    squared_distances = np.sum((positions[:, None, :] - positions[None, :, :]) ** 2, axis=-1)
    spatial_kernel = np.exp(-squared_distances / (2 * spatial_length**2))
    correlation = ((1 - noise_spatial_fraction) * np.eye(len(names))
                   + noise_spatial_fraction * spatial_kernel)
    # Use the SAME scalp gain for every frequency, preserving the frequency
    # classification task. RMS normalization keeps mean signal power comparable.
    cz = np.flatnonzero(names == "Cz")[0]
    gains = 0.5 + spatial_kernel[:, cz]
    gains /= np.sqrt(np.mean(gains**2))
    return {
        "channel_names": names,
        "channel_positions": positions,
        "channel_gains": gains,
        "noise_correlation": correlation,
        "montage_name": np.asarray(montage["montage"]),
        "coordinate_frame": np.asarray(montage["coordinate_frame"]),
        "coordinate_units": np.asarray(montage["units"]),
    }


def make_dataset(
    n_classes=4,
    samples_per_class=200,
    sampling_rate=256,
    duration=5.0,
    dominant_amplitude=2.0,
    background_amplitude=0.5,
    amplitude_jitter=0.1,
    noise_std=0.5,
    seed=42,
    frequency_start=10.0,
    frequency_spacing=0.075,
    spatial_length=0.08,
    noise_spatial_fraction=0.5,
    boost_channels=None,
):
    """Each class boosts a different frequency; phase and noise vary per sample.

    Returns a dictionary suitable for np.savez_compressed. X has shape
    (examples, 19, time), y has labels 0..K-1, and power is (examples, K).
    Shared sinusoids have fixed spatial gains, plus spatially correlated noise.
    Class k boosts frequency_start + k * frequency_spacing Hz. Close tones
    overlap in power space; separation is not guaranteed even without noise.
    boost_channels names a fixed subset receiving the class-specific boost;
    None boosts every channel. Other channels retain background tones and noise.
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
    spatial = channel_model(spatial_length, noise_spatial_fraction)
    names = spatial["channel_names"]
    selected = names if boost_channels is None else np.asarray(boost_channels)
    assert selected.ndim == 1 and len(selected) > 0, "boost_channels must be a nonempty sequence of channel names"
    assert len(np.unique(selected)) == len(selected), "boost_channels contains duplicate names"
    assert np.isin(selected, names).all(), f"Unknown boost channels: {selected[~np.isin(selected, names)]}"
    boost_mask = np.isin(names, selected)

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
    source_signal = np.zeros((len(labels), n_times))
    for j, frequency in enumerate(frequencies):
        source_signal += amplitudes[:, j, None] * np.sin(
            2 * np.pi * frequency * time[None, :] + phases[:, j, None]
        )
    boost_signal = (dominant_amplitude - background_amplitude) * np.sin(
        2 * np.pi * frequencies[labels, None] * time[None, :]
        + phases[np.arange(len(labels)), labels, None]
    )
    eigenvalues, eigenvectors = np.linalg.eigh(spatial["noise_correlation"])
    # A square root also handles a nearly rank-deficient kernel at fraction=1.
    noise_factor = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))
    n_channels = len(spatial["channel_names"])
    signals = np.empty((len(labels), n_channels, n_times), dtype=np.float32)
    for start in range(0, len(labels), 64):
        stop = min(start + 64, len(labels))
        independent_noise = rng.normal(0, noise_std, size=(stop - start, n_channels, n_times))
        signals[start:stop] = (
            spatial["channel_gains"][None, :, None] * (
                source_signal[start:stop, None, :]
                - (~boost_mask)[None, :, None] * boost_signal[start:stop, None, :]
            )
            + noise_factor @ independent_noise
        )
    channel_power = frequency_power(signals, sampling_rate, frequencies)
    return {
        **spatial,
        "X": signals,
        "y": labels,
        "source_signal": source_signal.astype(np.float32),
        "boost_signal": boost_signal.astype(np.float32),
        "boost_channels": names[boost_mask],
        "boost_channel_mask": boost_mask,
        "channel_power": channel_power,
        "power": channel_power.mean(axis=1),
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
    """Compare unsupervised K-means and a supervised linear SVM on power features.

    Scaling, model fitting, and K-means cluster-to-class matching use training
    data only. Test labels are used solely to score held-out predictions.
    """
    from scipy.optimize import linear_sum_assignment
    from sklearn.cluster import KMeans
    from sklearn.metrics import confusion_matrix
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    train, test = stratified_split(data["y"], seed)
    power, labels = data["power"], data["y"]
    n_classes = len(data["frequencies"])
    scaler = StandardScaler().fit(power[train])
    features = scaler.transform(power)

    kmeans = KMeans(n_clusters=n_classes, n_init=20, random_state=seed)
    kmeans.fit(features[train])
    clusters = kmeans.predict(features)
    counts = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(counts, (clusters[train], labels[train]), 1)
    # Cluster IDs are arbitrary. Choose a one-to-one class mapping that
    # maximizes agreement on training examples, then freeze it for testing.
    cluster_ids, class_ids = linear_sum_assignment(-counts)
    cluster_to_class = np.empty(n_classes, dtype=int)
    cluster_to_class[cluster_ids] = class_ids
    kmeans_predictions = cluster_to_class[clusters]

    svm = SVC(kernel="linear", C=1.0)
    svm.fit(features[train], labels[train])
    svm_predictions = svm.predict(features)

    data.update(train_indices=train, test_indices=test, kmeans_clusters=clusters,
                kmeans_predictions=kmeans_predictions, svm_predictions=svm_predictions)
    results = {
        "kmeans": {"n_clusters": n_classes, "n_init": 20, "random_state": seed,
                   "cluster_to_class": cluster_to_class.tolist()},
        "svm": {"kernel": "linear", "C": 1.0},
    }
    for name, predictions in (("kmeans", kmeans_predictions), ("svm", svm_predictions)):
        results[name].update(
            train_accuracy=float(np.mean(predictions[train] == labels[train])),
            test_accuracy=float(np.mean(predictions[test] == labels[test])),
            confusion_matrix_test=confusion_matrix(
                labels[test], predictions[test], labels=np.arange(n_classes)
            ).tolist(),
        )
    return results


def plot_dataset(data, metrics, sampling_rate, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    boost_mask = data["boost_channel_mask"]
    signals, labels, power = data["X"][:, boost_mask].mean(axis=1), data["y"], data["power"]
    frequencies, time = data["frequencies"], data["time"]
    n_classes = len(frequencies)
    colors = plt.cm.tab10(np.arange(n_classes) % 10)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    n_channels = data["X"].shape[1]
    selected = ", ".join(data["boost_channels"])
    fig.suptitle(f"Synthetic EEG-like signals · {n_channels} channels · {n_classes} classes · {len(labels):,} examples\n"
                 f"Boosted channels (fixed across classes): {selected}", fontsize=14)

    # Cap waveform/spectrum overlays to keep a large-K run readable.
    shown = min(n_classes, 8)
    visible = time < min(1.0, time[-1] + 1 / sampling_rate)
    example_indices = [np.flatnonzero(labels == k)[0] for k in range(shown)]
    offsets = np.arange(shown)[::-1] * 2.5 * np.max(np.abs(signals[example_indices][:, visible]))
    for k, index in enumerate(example_indices):
        axes[0, 0].plot(time[visible], signals[index, visible] + offsets[k], color=colors[k], lw=0.8)
        if not boost_mask.all():
            background = data["X"][index, ~boost_mask].mean(axis=0)
            axes[0, 0].plot(time[visible], background[visible] + offsets[k], color=colors[k], lw=0.8, ls="--", alpha=0.65)
    axes[0, 0].set(yticks=offsets, yticklabels=[f"Class {k}{'*' if k == 0 else ''}: {frequencies[k]:g} Hz" for k in range(shown)],
                   xlabel="Time (s)", title="Channel means · boosted: solid; others: dashed")

    # Zero padding samples the spectrum more densely for display; it does not
    # add frequency resolution or change the features used by the classifier.
    n_fft = 8 * len(time)
    fft_frequencies = np.fft.rfftfreq(n_fft, 1 / sampling_rate)
    display_power = np.zeros((len(labels), len(fft_frequencies)))
    for channel in range(n_channels):
        spectrum = np.fft.rfft(data["X"][:, channel, :], n=n_fft, axis=-1)
        display_power += 2 * np.abs(spectrum) ** 2 / (len(time)**2 * n_channels)
    padding = max(0.5, 0.2 * np.ptp(frequencies), 2 * sampling_rate / len(time))
    for k in range(shown):
        axes[0, 1].plot(fft_frequencies, display_power[labels == k].mean(axis=0),
                            color=colors[k], label=f"Class {k}", alpha=0.85)
        axes[0, 1].axvline(frequencies[k], color=colors[k], ls=":", alpha=0.4)
    axes[0, 1].set(xlim=(max(0, frequencies.min() - padding),
                         min(sampling_rate / 2, frequencies.max() + padding)),
                   xlabel="Frequency (Hz)", ylabel="Mean spectral projection power (a.u.²)",
                   title="Class-average spectra · power averaged across channels")
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
    axes[1, 0].text(0.02, 0.02,
                    f"Linear SVM ({n_classes}D): train {metrics['svm']['train_accuracy']:.1%}, "
                    f"test {metrics['svm']['test_accuracy']:.1%}",
                    transform=axes[1, 0].transAxes, fontsize=9,
                    bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"})
    if n_classes <= 8:
        axes[1, 0].legend(ncols=2, fontsize=8)

    explanation = axes[1, 1]
    explanation.set_axis_off()
    explanation.set_title("Signal generation")
    explanation.text(
        0.5, 0.9, r"$x_{ic}(t) = g_c\,[b_i(t) + m_c\,d_i(t)] + \epsilon_{ic}(t)$",
        transform=explanation.transAxes, ha="center", va="center", fontsize=15,
    )
    explanation.text(
        0.5, 0.73, r"$d_i(t) = \Delta A\,\sin(2\pi f_{y_i}t + \phi_{i,y_i})$",
        transform=explanation.transAxes, ha="center", va="center", fontsize=15,
    )
    definitions = [
        (r"$b_i(t)$", "All background tones, with phase/amplitude jitter"),
        (r"$\Delta A$", "Dominant amplitude minus background amplitude"),
        (r"$m_c$", "1 on selected channels, 0 on other channels"),
        (r"$g_c$", "Fixed channel gain from electrode position"),
        (r"$\epsilon_{ic}(t)$", "Gaussian noise; nearby channels correlated"),
    ]
    for row, (symbol, description) in enumerate(definitions):
        y = 0.56 - 0.1 * row
        explanation.text(0.05, y, symbol, transform=explanation.transAxes,
                         va="center", fontsize=13)
        explanation.text(0.22, y, description, transform=explanation.transAxes,
                         va="center", fontsize=9.5)
    explanation.text(0.05, 0.035, r"Label $y_i$ selects frequency; the channel mask stays fixed.",
                     transform=explanation.transAxes, va="center", fontsize=10)
    fig.savefig(output / "overview.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    fig.suptitle("Held-out predictions · channel-averaged spectral power", fontsize=15)
    vmax = max(np.max(metrics[name]["confusion_matrix_test"]) for name in ("kmeans", "svm"))
    for ax, name, title in zip(axes, ("kmeans", "svm"), ("K-means", "Linear SVM")):
        confusion = np.asarray(metrics[name]["confusion_matrix_test"])
        heatmap = ax.imshow(confusion, cmap="Blues", vmin=0, vmax=vmax)
        if n_classes <= 12:
            for row, column in np.ndindex(confusion.shape):
                ax.text(column, row, str(confusion[row, column]), ha="center", va="center",
                        color="white" if confusion[row, column] > vmax / 2 else "black")
            ax.set(xticks=np.arange(n_classes), yticks=np.arange(n_classes))
        ax.set(xlabel="Predicted class", ylabel="True class",
               title=f"{title} · test accuracy {metrics[name]['test_accuracy']:.1%}")
    fig.colorbar(heatmap, ax=axes, label="Examples")
    fig.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def plot_multichannel(data, output):
    """Show electrode geometry, all channels, and specified/observed correlations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names, positions = data["channel_names"], data["channel_positions"]
    n_channels = len(names)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
    fig.suptitle("19-channel 10–20 system · shared oscillations and spatial noise", fontsize=17)

    positions_cm = 100 * positions
    points = axes[0, 0].scatter(positions_cm[:, 0], positions_cm[:, 1],
                                c=data["channel_gains"], s=120, cmap="viridis",
                                edgecolors=np.where(data["boost_channel_mask"], "black", "none"), linewidths=2)
    for name, (x, y, _) in zip(names, positions_cm):
        axes[0, 0].annotate(name, (x, y), xytext=(0, 8), textcoords="offset points",
                            ha="center", fontsize=9)
    axes[0, 0].set(xlabel="Head x (cm; positive = right)",
                   ylabel="Head y (cm; positive = anterior)",
                   title="Electrode positions · black outline = boosted")
    axes[0, 0].set_aspect("equal")
    axes[0, 0].margins(0.15)
    fig.colorbar(points, ax=axes[0, 0], label="Fixed signal gain")

    index = np.flatnonzero(data["y"] == 0)[0]
    visible = data["time"] < 1.0
    traces = data["X"][index][:, visible]
    offsets = np.arange(n_channels)[::-1] * 2.3 * np.max(np.abs(traces))
    for channel in range(n_channels):
        axes[0, 1].plot(data["time"][visible], traces[channel] + offsets[channel],
                        color=plt.cm.viridis(channel / (n_channels - 1)), lw=0.7)
    axes[0, 1].set(yticks=offsets, yticklabels=names, xlabel="Time (s)",
                   title="One class-0 recording · all 19 channels")
    axes[0, 1].tick_params(axis="y", labelsize=8)

    correlation = np.zeros((n_channels, n_channels))
    for start in range(0, len(data["y"]), 64):
        centered = data["X"][start:start + 64].astype(float)
        centered -= centered.mean(axis=-1, keepdims=True)
        gram = centered @ centered.transpose(0, 2, 1)
        norms = np.sqrt(np.maximum(np.diagonal(gram, axis1=1, axis2=2), 1e-30))
        correlation += (gram / (norms[:, :, None] * norms[:, None, :])).sum(axis=0)
    correlation /= len(data["y"])
    for ax, matrix, title in (
        (axes[1, 0], data["noise_correlation"], "Specified noise correlation"),
        (axes[1, 1], correlation, "Mean within-recording channel correlation"),
    ):
        heatmap = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set(xticks=np.arange(n_channels), xticklabels=names,
               yticks=np.arange(n_channels), yticklabels=names, title=title)
        ax.tick_params(axis="x", labelrotation=90, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        fig.colorbar(heatmap, ax=ax, label="Correlation")
    fig.savefig(output / "multichannel.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-classes", type=int, default=4)
    parser.add_argument("--samples-per-class", type=int, default=200)
    parser.add_argument("--sampling-rate", type=float, default=256)
    parser.add_argument("--duration", type=float, default=5)
    parser.add_argument("--frequency-start", type=float, default=10,
                        help="First class frequency in Hz (default: 10)")
    parser.add_argument("--frequency-spacing", type=float, default=0.075,
                        help="Spacing between class frequencies in Hz (default: 0.075)")
    parser.add_argument("--dominant-amplitude", type=float, default=2)
    parser.add_argument("--background-amplitude", type=float, default=0.5)
    parser.add_argument("--amplitude-jitter", type=float, default=0.1)
    parser.add_argument("--noise-std", type=float, default=0.5)
    parser.add_argument("--spatial-length", type=float, default=0.08,
                        help="Spatial smoothness length in meters (default: 0.08)")
    parser.add_argument("--noise-spatial-fraction", type=float, default=0.5,
                        help="Spatially correlated fraction of noise variance (default: 0.5)")
    parser.add_argument("--boost-channels", nargs="+",
                        help="Channel names receiving the class-specific amplitude boost, fixed across classes (default: all)")
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
              "montage": str(data["montage_name"]),
              "channel_names": data["channel_names"].tolist(),
              "boost_channels": data["boost_channels"].tolist(),
              "coordinate_frame": str(data["coordinate_frame"]),
              "coordinate_units": str(data["coordinate_units"]),
              "fft_bin_spacing_hz": args.sampling_rate / data["X"].shape[-1],
              "features": "Mean of per-channel spectral projection power",
              "channel_power_shape": list(data["channel_power"].shape),
              "classifier_feature_shape": list(data["power"].shape),
              "feature_scaling": "StandardScaler fitted on training power features",
              "shape": list(data["X"].shape), **metrics}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_dataset(data, metrics, args.sampling_rate, output)
    plot_multichannel(data, output)
    print(json.dumps(report, indent=2))
    print(f"Saved dataset, summary, and plots to {output.resolve()}")


if __name__ == "__main__":
    main()
