# Synthetic EEG experiments

Each class boosts a different sinusoid in noisy signals with spatial gains and correlated channel noise.
Defaults: four classes, 800 recordings, 19 channels, 256 Hz, five seconds, and an 80/20 recording split.

| File | Purpose |
| --- | --- |
| [generate.py](generate.py) | Generate data, evaluate spectral SVM/K-means baselines, and plot the dataset. |
| [train_transformer.py](train_transformer.py) | Compare positional normalization using raw waveform patches. |
| [standard_1020.json](standard_1020.json) | Electrode order and head coordinates in meters. |
| [test_generate.py](../../tests/test_generate.py) | Check generation, channel selection, and spatial structure. |
| `outputs/` | Generated arrays, split indices, settings, baseline scores, and plots. |

Run `python -m synthetic_data.sinusoid_amplitude.generate` from the repo root, then the `train_transformer` module; use `--help` for options.
