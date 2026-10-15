# Architecture ablations

Sweep positional encodings, pooling, depth, and dataset size on synthetic EEG waveform patches.
Compare positional normalization on/off across three seeds, with SVM, K-means, and bag-of-tokens baselines.

| File | Purpose |
| --- | --- |
| [run.py](run.py) | Run sweeps with shared data/splits per size and seed; average the results. |
| [baselines.py](baselines.py) | Fit the bag-of-tokens baseline using a codebook learned from training patches. |
| [report.py](report.py) | Build comparison tables and a PDF from saved results; requires `pdflatex`. |
| [test_ablations.py](test_ablations.py) | Check sweep configuration, baselines, result reuse, and reports. |
| `run_*/` | Saved configurations, per-seed datasets/metrics, plots, and `summary.pdf`. |

Run `python -m results.synthetic_runs.sinusoid_amplitude.arch_ablations.run` from the repo root; use `--help` for options. Each run creates a new folder.
