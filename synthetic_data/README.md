# Synthetic EEG classification sandbox

Each class boosts the amplitude of one frequency. The frequencies are now close
together, making the spectral features overlap over a five-second window.
Spacing controls difficulty before testing whether EEG tokenizers preserve the
same information. These are toy EEG-like signals with arbitrary amplitude units.

## Possible class designs

| Design | Class-specific parameter | Easy feature space | Useful property |
| --- | --- | --- | --- |
| **Dominant frequency (implemented)** | Which sinusoid has the largest amplitude | Fourier projection power at each component frequency | Adjustable spacing; random phase; comparable expected total energy |
| Alpha amplitude levels | A different 10 Hz amplitude for each class | One alpha-power feature | Simplest scalar problem; global amplitude normalization can erase the label |
| Band-power profiles | A vector of delta/theta/alpha/beta amplitudes per class | Four band-power features | Supports many classes with well-spaced prototypes, without adding more frequencies |
| Fixed-phase templates | Different amplitude vectors, identical phases | Raw samples or linear sine projections | Useful if linear separability must hold directly in waveform space |

For alpha levels or power profiles, nearest-prototype classification has affine
scores `2 * prototype @ features - ||prototype||²`. Distinct prototypes are
separable when examples stay sufficiently close to their own prototype.
Unbounded Gaussian noise prevents a strict population-level guarantee in all
these designs.

## Current experiment

`generate.py` implements dominant frequency. Defaults are 4 balanced classes,
200 independent examples per class, one channel, 250 Hz, and 5 seconds:

| Class | 10 Hz amplitude | 10.075 Hz amplitude | 10.150 Hz amplitude | 10.225 Hz amplitude |
| --- | --- | --- | --- | --- |
| 0 | 2.0 | 0.5 | 0.5 | 0.5 |
| 1 | 0.5 | 2.0 | 0.5 | 0.5 |
| 2 | 0.5 | 0.5 | 2.0 | 0.5 |
| 3 | 0.5 | 0.5 | 0.5 | 2.0 |

Each component gets an independent uniform phase and amplitude jitter of ±0.1
per example. Additive Gaussian noise has standard deviation 0.5. All classes
have the same expected total signal power over random phases. Individual windows
can differ in energy because nearby tones interfere.

For any K, class k boosts `frequency_start + k * frequency_spacing` Hz.
Defaults are a start of 10 Hz and spacing of 0.075 Hz. Frequencies are used
exactly as requested, without rounding to FFT bins. They must be positive,
distinct, and below half the sampling rate.

For an example x with T samples, the feature at frequency f[j] is:

```text
p[j] = 2 * |sum_n x[n] * exp(-2πi * f[j] * n / sampling_rate)|² / T²
```

For distinct FFT-aligned tones without noise, this equals `amplitude[j]² / 2`
independently of phase. For closely spaced off-grid tones, components contribute
to each other's projections. These features no longer isolate each component's
power, and phase differences produce overlap even without Gaussian noise.
The power transform is nonlinear; the classifier using those features is linear.

The rule `predict = argmax(p)` is a linear multiclass classifier in this feature
space: its score matrix is the identity. The script checks the **minimum power margin** over every
generated example: correct-class power minus the largest competing power.
A positive margin proves that this particular finite dataset is separated by
the argmax rule. A negative margin only means this particular rule fails on some
examples; it does not establish that no linear classifier could separate them.

As a learned baseline, the script also fits class centroids using only the
training power features. Nearest-centroid prediction is implemented using the
affine scores above. It reports held-out accuracy and saves a confusion matrix.

## How spacing changes difficulty

A five-second window has FFT-bin spacing `1 / duration = 0.2 Hz`. Reducing the
class spacing while retaining distinct aligned bins leaves the original simple
task easy. Sub-bin spacing makes projections overlap. FFT-bin spacing is not an
absolute limit on frequency estimation; longer windows and different features
or classifiers can change these results.

With all other defaults fixed, an exploratory comparison across seeds 42–46
(160 test examples per seed) gave:

| Frequency spacing | Mean test accuracy | Range across seeds |
| --- | --- | --- |
| 0.200 Hz | 100.0% | 100.0–100.0% |
| 0.100 Hz | 98.6% | 96.9–99.4% |
| **0.075 Hz (default)** | **81.4%** | **78.1–85.0%** |
| 0.050 Hz | 64.1% | 61.3–69.4% |

These are results for the centroid baseline, not limits on the best achievable
accuracy. On default seed 42, the centroid classifier scores 78.1%, while the
known `argmax(power)` rule scores 89.4%. This sweep was used to choose an
intermediate difficulty; use fresh seeds for subsequent model comparisons.

## Run

From the repository root, with NumPy and Matplotlib available:

```sh
python3 synthetic_data/generate.py
python3 synthetic_data/generate.py --n-classes 6 --output synthetic_data/outputs/k6
python3 synthetic_data/generate.py --frequency-spacing 0.05 --output synthetic_data/outputs/harder
python3 synthetic_data/generate.py --frequency-spacing 0.2 --output synthetic_data/outputs/easier
```

If needed, install the dependencies in your Python environment with
`python3 -m pip install -r synthetic_data/requirements.txt`.

To recover an FFT-aligned, noise-free separated construction at five seconds:

```sh
python3 synthetic_data/generate.py --frequency-spacing 0.2 --noise-std 0 --output synthetic_data/outputs/clean
```

The default output directory is `synthetic_data/outputs/` (ignored by Git).
Repeating a command with the same seed reproduces the data and overwrites files
in that output directory. Use separate output paths for different experiments.

## Saved data and plots

`dataset.npz` contains:

| Key | Meaning |
| --- | --- |
| `X` | Float32 signals, shape `(N, 1, T)`; defaults to `(800, 1, 1250)` |
| `y` | Integer labels 0 through K−1, shape `(N,)` |
| `power` | Fourier projection power at class frequencies, shape `(N, K)`; close components interfere |
| `train_indices`, `test_indices` | Shuffled stratified approximately 80/20 split; both contain every class |
| `time`, `frequencies` | Sample times and actual component frequencies in Hz |
| `class_amplitudes` | Nominal amplitude table before jitter, shape `(K, K)` |
| `linear_weights`, `linear_bias` | Trained centroid classifier: `argmax(power @ weights.T + bias)` |

```python
import numpy as np

data = np.load("synthetic_data/outputs/dataset.npz")
train = data["train_indices"]
test = data["test_indices"]
X_train, y_train = data["X"][train], data["y"][train]
X_test, y_test = data["X"][test], data["y"][test]
```

You can also call `make_dataset` directly from
`synthetic_data.generate`; it returns the arrays before splitting/evaluation.

`overview.png` shows example waveforms, zoomed class-average spectra, a 2D PCA projection
of held-out power features, and the signal equation with variable definitions. PCA is fitted
on training features; a 2D projection can hide separation present in the full
K-dimensional space. Waveform/spectrum overlays show at most eight classes.
`confusion_matrix.png` shows the learned linear classifier's held-out predictions.
Spectra use zero padding to draw smooth curves; this does not improve frequency
resolution or change the classification features. Dotted lines mark class frequencies.
`summary.json` records generation parameters, frequencies, FFT-bin spacing,
accuracy, and margin.

## Next experiments

- Adjust frequency spacing or window duration to change spectral overlap.
- Increase noise or reduce the dominant/background amplitude gap to control difficulty.
- Add frequency jitter and use integrated band power to test tolerance to peak shifts.
- Add channels with known mixing weights to test spatial information.
- Plot spectrograms after adding time-varying bursts or frequency changes; the
  current stationary sinusoids are more clearly summarized by their spectra.
- Compare spectral-power classification with classification after tokenization.
  Separation in power space does not imply separation after a learned tokenizer.

Each generated window is independent. If later cutting multiple windows from a
shared synthetic recording, split by recording to avoid train/test leakage.
