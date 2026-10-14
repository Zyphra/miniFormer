# Synthetic EEG classification sandbox

Each class boosts the amplitude of one frequency. Nineteen channels share these
oscillations, with smooth spatial gains and correlated noise based on 10–20
electrode positions. Nearby frequencies overlap over a five-second window;
spacing controls difficulty. These are toy EEG-like signals with arbitrary
amplitude units, intended for comparing classifiers and EEG tokenizers.

## Possible class designs

| Design | Class-specific parameter | Easy feature space | Useful property |
| --- | --- | --- | --- |
| **Dominant frequency (implemented)** | Which sinusoid has the largest amplitude | Fourier projection power at each component frequency | Adjustable spacing; random phase; comparable expected total energy |
| Alpha amplitude levels | A different 10 Hz amplitude for each class | One alpha-power feature | Simplest scalar problem; global amplitude normalization can erase the label |
| Band-power profiles | A vector of delta/theta/alpha/beta amplitudes per class | Four band-power features | Supports many classes with well-spaced prototypes, without adding more frequencies |
| Fixed-phase templates | Different amplitude vectors, identical phases | Raw samples or linear sine projections | Useful if linear separability must hold directly in waveform space |

## Current experiment

`generate.py` implements dominant frequency. Defaults are 4 balanced classes,
200 independent recordings per class, 19 channels, 256 Hz, and 5 seconds:

| Class | 10 Hz amplitude | 10.075 Hz amplitude | 10.150 Hz amplitude | 10.225 Hz amplitude |
| --- | --- | --- | --- | --- |
| 0 | 2.0 | 0.5 | 0.5 | 0.5 |
| 1 | 0.5 | 2.0 | 0.5 | 0.5 |
| 2 | 0.5 | 0.5 | 2.0 | 0.5 |
| 3 | 0.5 | 0.5 | 0.5 | 2.0 |

Each component gets an independent uniform phase and amplitude jitter of ±0.1
per recording. These amplitudes and phases are shared across channels within
that recording. Gaussian noise has standard deviation 0.5 at each channel.
All classes have the same expected channel-averaged signal power over random
phases. Individual windows can differ in energy because nearby tones interfere.

For any K, class k boosts `frequency_start + k * frequency_spacing` Hz.
Defaults are a start of 10 Hz and spacing of 0.075 Hz. Frequencies are used
exactly as requested, without rounding to FFT bins. They must be positive,
distinct, and below half the sampling rate.

### Restrict the boost to fixed channels

`--boost-channels C3 Cz C4` applies the class-specific amplitude increase only
to those channels. The same mask is used for every class and recording; there
is no channel shuffling. Other channels keep all background tones, amplitude
jitter, and noise. Omitting the flag preserves the original all-channel data
exactly, including the random draws.

For example, a class-0 recording has nominal 10 Hz amplitude 2.0 on C3/Cz/C4
and 0.5 on all other channels, before the existing spatial gains. The other
frequencies remain at 0.5 everywhere. Class 1 boosts its own frequency on the
same C3/Cz/C4 set, and so on.

```text
d_i(t) = (dominant_amplitude - background_amplitude) * sin(2*pi*f_yi*t + phi_i,yi)
b_i(t) = source_signal_i(t) - d_i(t)
x_ic(t) = g_c * [b_i(t) + boost_channel_mask_c * d_i(t)] + epsilon_ic(t)
```

Selected channels are numerically identical to the original all-channel run
with the same seed. Unselected channels lose only the nominal class boost.
This creates an informative subset, but channel identity still does not define
the label: the boosted frequency does. Unknown, duplicate, or empty channel
lists raise assertions.

## 19-channel spatial model

`standard_1020.json` contains 19 electrodes extracted from MNE 1.11.0's
`standard_1020` template, using `Info.set_montage()` to obtain **head coordinates
in meters**. See the [MNE sensor-location tutorial](https://mne.tools/stable/auto_tutorials/intro/40_sensor_locations.html).
We select this conventional scalp subset from the larger template:

```text
Fp1 Fp2
F7 F3 Fz F4 F8
T7 C3 Cz C4 T8
P7 P3 Pz P4 P8
O1 O2
```

T7/T8/P7/P8 use the modern names for the T3/T4/T5/T6 positions. The saved file
records the exact coordinates, channel order, source, and MNE version. Loading
the generator does not require MNE or a coordinate download.

With the default all-channel boost, generate a clean mixture and distribute it:

```text
s_i(t) = sum_j A_ij * sin(2*pi*f_j*t + phi_ij)
x_ic(t) = g_c * s_i(t) + epsilon_ic(t)
```

The class label still selects the boosted frequency. `g_c` is a fixed gain,
shared by every frequency and every recording. Using electrode position `r_c`:

```text
h_c = 0.5 + exp(-||r_c - r_Cz||² / (2*ell²))
g_c = h_c / sqrt(mean_c(h_c²))
```

This produces a smooth gain profile centered on Cz, with `mean(g_c²) = 1` to
keep average signal power comparable to the single-channel experiment. The
gain is an illustrative spatial profile, not a physiological forward solution.
With all channels boosted, the clean multichannel signal has rank one: every
channel sees a scaled copy of the shared mixture. A subset mask can increase
the clean rank to two (background plus masked boost). Noise makes the observed
recordings differ across channels.

Noise is drawn jointly across channels at each time sample:

```text
epsilon_i(t) ~ Normal(0, sigma² * R)
R_cd = (1-rho) * I[c=d] + rho * exp(-||r_c - r_d||² / (2*ell²))
```

Noise draws are independent over time and recordings. Nearby electrodes have
more correlated noise; distant electrodes have less. `R_cc = 1`, so every
channel retains noise standard deviation `sigma`. A matrix square root of R
mixes independent Gaussian draws to produce this covariance.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `--spatial-length` (`ell`) | 0.08 m | Spatial smoothness of gains and noise; larger means broader sharing |
| `--noise-spatial-fraction` (`rho`) | 0.5 | Fraction of noise variance assigned to the spatial kernel |
| `--noise-std` (`sigma`) | 0.5 | Noise standard deviation at each channel |

`rho` is not the overall channel correlation: shared oscillations correlate
channels even when `rho=0`. To add richer spatial structure later, each sinusoid
could receive its own spatial gain pattern; the current common pattern keeps
the label tied to the original frequency-amplitude task.

## Spectral features

For T samples, compute the power at each component frequency for each channel:

```text
channel_power[i,c,j] = 2 * |sum_n x_ic[n] * exp(-2πi*f_j*n/sampling_rate)|² / T²
power[i,j] = mean_c(channel_power[i,c,j])
```

For distinct FFT-aligned tones without noise, the channel-averaged feature
equals `amplitude[j]² / 2`, because `mean(g_c²)=1`. For closely spaced off-grid tones, components contribute
to each other's projections. These features no longer isolate each component's
power, and phase differences produce overlap even without Gaussian noise.
Both baselines use these K channel-averaged power features. The complete
`(N, 19, T)` signals and `(N, 19, K)` per-channel powers are also saved for models
that explicitly use channels. Shared phase interference remains when averaging
channel powers; averaging can reduce noise variability and change accuracy.

### Why average channel powers?

For the default all-channel boost, every clean channel is `g_c * s_i(t)`. Fourier projection
is linear, so at every tested frequency (including off-grid frequencies):

```text
P_clean[i,c,j] = g_c² * P_source[i,j]
mean_c(P_clean[i,c,j]) = P_source[i,j], since mean_c(g_c²) = 1
```

Pooling therefore preserves the entire clean power vector of the original
experiment. There is no class-specific spatial pattern to identify: the gain
profile is fixed across classes. It also keeps K features per recording (4 by
default), instead of C*K highly correlated features (76 by default).

We compute **power first, then average channels**. Computing power after
averaging waveforms is a different operation. The channel-mean waveform shown
in the overview is just a visualization; classifiers receive mean power.

This is a simple baseline, not an assertion that equal averaging is the optimal
noise estimator. Noise is correlated and channel gains differ, so the channels
do not provide 19 independent measurements. Pooling would discard potentially
useful spatial information if individual frequencies acquired different spatial
patterns. In that extension, flatten `(C, K)` per-channel powers or use a model
with an explicit channel dimension and the saved electrode coordinates.

With `--boost-channels`, pooling all channels includes background-only channels
and can dilute the class contrast. The rank-one argument above no longer
applies. Selected-channel pooling and flattened channel powers are separate
feature comparisons; the default baselines still pool all channels.

A comparison on the default 256 Hz, seed-42 dataset used the same 640 training and
160 test recordings, train-only scaling, and unchanged model settings:

| Channel handling | Features | K-means test accuracy | Linear SVM test accuracy |
| --- | --- | --- | --- |
| Mean per-channel power (default) | 4 | 48.1% | 95.0% |
| Flatten all channel-frequency powers | 76 | 48.8% | 96.9% |

Flattening classified three additional SVM test examples correctly on this
split. The mean remains the compact baseline justified by the common spatial
gain model, not a claim of optimal noisy-data performance. This exploratory
comparison is saved in `outputs/channel_pooling_comparison.json` with the full
configuration; it is separate from the generator's default evaluation.

## K-means and SVM

The script compares two methods on the same stratified 80/20 split:

Each recording and all its channels stay together in one split.

| Method | Training | Settings |
| --- | --- | --- |
| K-means | Unsupervised clustering of training examples | K clusters, 20 initializations, fixed random seed |
| Linear SVM | Supervised classification using training labels | `SVC(kernel="linear", C=1.0)` |

For both methods, `StandardScaler` learns the mean and standard deviation of
each feature from the training split only. The same transformation is applied
to held-out examples. Neither method is trained on the PCA projection.

K-means cluster IDs have no inherent class meaning. After fitting without
labels, we match cluster IDs to classes using a one-to-one assignment that
maximizes agreement with **training labels only** (Hungarian assignment).
This mapping is frozen before predicting and scoring the test examples.
The SVM uses training labels directly. Test labels are used only for scoring.

Both methods report training and test accuracy and a test confusion matrix.
Predictions are saved with the dataset; `summary.json` records model settings
and the K-means cluster-to-class mapping.

## How spacing changes difficulty

A five-second window has FFT-bin spacing `1 / duration = 0.2 Hz`. Reducing the
class spacing while retaining distinct aligned bins leaves the original simple
task easy. Sub-bin spacing makes projections overlap. FFT-bin spacing is not an
absolute limit on frequency estimation; longer windows and different features
or classifiers can change these results.

Change `--frequency-spacing` to explore difficulty. Model performance depends
on the features, method, and random seed; it is not an accuracy ceiling for the
underlying signals. Use fresh seeds for comparisons after choosing parameters.

## Run

From the repository root, with NumPy, Matplotlib, SciPy, and scikit-learn available:

```sh
python3 synthetic_data/sinusoid_amplitude/generate.py
python3 synthetic_data/sinusoid_amplitude/generate.py --n-classes 6 --output synthetic_data/sinusoid_amplitude/outputs/k6
python3 synthetic_data/sinusoid_amplitude/generate.py --frequency-spacing 0.05 --output synthetic_data/sinusoid_amplitude/outputs/harder
python3 synthetic_data/sinusoid_amplitude/generate.py --frequency-spacing 0.2 --output synthetic_data/sinusoid_amplitude/outputs/easier
python3 synthetic_data/sinusoid_amplitude/generate.py --noise-spatial-fraction 0 --output synthetic_data/sinusoid_amplitude/outputs/independent_noise
python3 synthetic_data/sinusoid_amplitude/generate.py --boost-channels C3 Cz C4 --output synthetic_data/sinusoid_amplitude/channel_subset_C3_Cz_C4
```

If needed, install the dependencies in your Python environment with
`python3 -m pip install numpy matplotlib scipy scikit-learn`.

To recover an FFT-aligned, noise-free separated construction at five seconds:

```sh
python3 synthetic_data/sinusoid_amplitude/generate.py --frequency-spacing 0.2 --noise-std 0 --output synthetic_data/sinusoid_amplitude/outputs/clean
```

The default output directory is `synthetic_data/sinusoid_amplitude/outputs/` (ignored by Git).
Repeating a command with the same seed reproduces the data and overwrites files
in that output directory. Use separate output paths for different experiments.

## Saved data and plots

`dataset.npz` contains:

| Key | Meaning |
| --- | --- |
| `X` | Float32 signals, shape `(N, 19, T)`; defaults to `(800, 19, 1280)` |
| `y` | Integer labels 0 through K−1, shape `(N,)` |
| `power` | Channel-averaged Fourier projection power, shape `(N, K)`; used by K-means and SVM |
| `channel_power` | Fourier projection power for every channel, shape `(N, 19, K)` |
| `source_signal` | Clean mixture on boosted channels before gains/noise, shape `(N, T)`; diagnostic ground truth, not a classifier input |
| `boost_signal` | Nominal class-specific boost waveform `(N, T)`; subtract from `source_signal` for unselected channels before applying gains |
| `boost_channels`, `boost_channel_mask` | Resolved selected names and boolean mask `(19,)`, fixed across classes and recordings |
| `channel_names`, `channel_positions` | Electrode names `(19,)` and corresponding 3D positions `(19, 3)` |
| `channel_gains`, `noise_correlation` | Fixed gains `(19,)` and specified noise correlation `(19, 19)` |
| `montage_name`, `coordinate_frame`, `coordinate_units` | `standard_1020`, `head`, and `m` |
| `train_indices`, `test_indices` | Shuffled stratified approximately 80/20 split; both contain every class |
| `time`, `frequencies` | Sample times and actual component frequencies in Hz |
| `class_amplitudes` | Nominal amplitude table before jitter, shape `(K, K)` |
| `kmeans_clusters` | Raw cluster IDs for every example, shape `(N,)` |
| `kmeans_predictions`, `svm_predictions` | Predicted class labels for every example, shape `(N,)`; index with `test_indices` for held-out results |

```python
import numpy as np

data = np.load("synthetic_data/sinusoid_amplitude/outputs/dataset.npz")
train = data["train_indices"]
test = data["test_indices"]
X_train, y_train = data["X"][train], data["y"][train]
X_test, y_test = data["X"][test], data["y"][test]
```

You can also call `make_dataset` directly from
`synthetic_data.sinusoid_amplitude.generate`; it returns the arrays before splitting/evaluation.

`overview.png` shows channel-mean waveforms, class-average spectra (power averaged
across channels), a 2D PCA projection of held-out power features, and the signal
equations with variable definitions. PCA is fitted
on training features; a 2D projection can hide separation present in the full
K-dimensional space. Waveform/spectrum overlays show at most eight classes.
Solid waveform traces average the boosted channels; dashed traces average the
remaining channels when a subset is selected. The title names the boosted
channels. Spectra and PCA retain the all-channel pooled features used by the
baselines. The PCA panel reports the linear SVM's full-feature train/test scores;
the 2D view alone does not establish linear separability.
`confusion_matrix.png` compares K-means and linear SVM held-out predictions side by side.
`multichannel.png` shows the electrode positions colored by gain, all 19 channels
from one class-0 recording, the specified noise correlation, and the measured
within-recording signal correlation averaged across recordings. The correlations
differ because each channel also receives the shared oscillations.
Spectra use zero padding to draw smooth curves; this does not improve frequency
resolution or change the classification features. Dotted lines mark class frequencies.
`summary.json` records generation parameters, montage metadata, frequencies, FFT-bin spacing,
feature scaling, model settings, cluster-to-class mapping, and both models' metrics.

## Next experiments

- Adjust frequency spacing or window duration to change spectral overlap.
- Increase noise or reduce the dominant/background amplitude gap to control difficulty.
- Add frequency jitter and use integrated band power to test tolerance to peak shifts.
- Give different components distinct spatial gains to test spatial information.
- Plot spectrograms after adding time-varying bursts or frequency changes; the
  current stationary sinusoids are more clearly summarized by their spectra.
- Compare spectral-power classification with classification after tokenization.
  Separation in power space does not imply separation after a learned tokenizer.

Each generated window is independent. If later cutting multiple windows from a
shared synthetic recording, split by recording to avoid train/test leakage.
