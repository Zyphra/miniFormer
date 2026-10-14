# Architecture ablations

From the repository root:

```sh
python -m pip install torch linear-attention-transformer numpy matplotlib scipy scikit-learn
python -m results.synthetic_runs.sinusoid_amplitude.arch_ablations.run
```

PDF generation requires `pdflatex` with the `geometry`, `booktabs`, and `graphicx`
packages (available in the installed TeX distribution). The runner checks for
`pdflatex` before training and fails if report compilation fails.

The local validation environment is also available:

```sh
results/synthetic_runs/sinusoid_amplitude/arch_ablations/.venv/bin/python -m results.synthetic_runs.sinusoid_amplitude.arch_ablations.run
```

`run.py` wraps the existing transformer experiment and the three baselines.
Each invocation creates a new dated directory here. `--output` must name a new
directory inside `results/synthetic_runs/sinusoid_amplitude/arch_ablations`; existing directories raise an error.
Generated artifacts are ignored by Git; source, tests, and this README are tracked.

| Sweep | Values | Default |
| --- | --- | --- |
| Temporal position | sinusoidal, rotary (RoPE) | sinusoidal |
| Channel position | learned, spatial | spatial |
| Aggregation | CLS_token, mean_pool_final, attn_pool_final, flatten_final | CLS_token |
| Depth | Every integer from 2 to 10 | 4 |
| Dataset size | 0.25×, 0.5×, and every multiple from 1× to 10× | 800 recordings, 200 per class |

Each configuration runs with `normalize=False` and `normalize=True`, including
RoPE. Normalization affects additive temporal/channel embeddings; RoPE rotations
stay unchanged, but its default spatial channel embeddings still change.
Within each normalization pair, initial weights and minibatch order are identical.
Seeds are **42, 43, and 44**. Each seed controls generation, model initialization,
and minibatch order; its split uses seed + 1. All configurations share the dataset
and split within a size/seed pair. Tables and curves show equal-weight means
across seeds, with individual scores retained in each seed's `results.json`.

Only one architecture parameter varies at a time. The default pair is reused
across comparisons: **25 distinct configurations / 50 transformer trainings per
seed**, or 150 trainings before reuse. The training wrapper reads architectural
defaults directly from the classifier constructor, including four layers.
Defaults remain 30 epochs, batch size 32, Adam at 0.001, gradient clipping at
L2 = 1, 64 model dimensions, and 2 heads. Inputs remain 19 channels and ten raw
128-sample patches per channel. `--epochs`, `--batch-size`, `--learning-rate`,
`--seeds`, and `--threads` apply uniformly across the suite.

Dataset sizes change only `samples_per_class` in `make_dataset`. Each size is
regenerated with the same seed and an 80/20 recording split, growing both train
and test sets (160/40, 320/80, then 640/160 through 6,400/1,600). These are not nested subsets or a
fixed-test-set learning curve. Training epochs stay fixed, so larger datasets
receive more optimizer steps. No epoch is selected using test performance.

SVM and K-means reuse `synthetic_data.sinusoid_amplitude.generate.evaluate`: channel-averaged
spectral powers, training-only StandardScaler, linear SVM (`C=1`), and K-means
with 20 initializations. Its cluster-to-class mapping uses training labels only.

Bag-of-tokens uses the existing classifier with multiclass logits. A 128-entry
MiniBatchKMeans codebook is fitted on raw training patches only; held-out patches
are assigned to that frozen codebook. Token counts divided by vocabulary size
feed the classifier, trained with cross-entropy plus its existing L2 penalty
(`C=1`) and the same optimizer/epoch settings as the transformer. Saved curves
report unpenalized cross-entropy. This uses waveform tokens, not LaBraM tokens.
The original binary classifier API still returns one logit by default.

All three baselines run once per dataset size and seed: **36 fits per baseline**
before reuse. The codebook is also refitted once per size/seed. Architecture sweeps share the 1× baseline
results; normalization does not apply to these baselines.

To reuse matching seed-42 results from the original completed study:

```sh
python -m results.synthetic_runs.sinusoid_amplitude.arch_ablations.run \
  --reuse-seed42 results/synthetic_runs/sinusoid_amplitude/arch_ablations/FIRST_ABLATION_RUN
```

This reuses all nine depth configurations and the ten existing datasets and
baseline fits. It validates settings before copying diagnostics and hard-links
the immutable dataset archives. The original dataset-size curve used two layers;
those transformer runs are regenerated at four layers. The 800-recording,
four-layer default is reused from the old depth sweep. Other table variants and
the 200/400-recording experiments are newly trained. `reuse.json` records sources.
Historical launch/training logs retain the commands and paths used at launch.

Each suite saves:

- `config.json`: generation/training settings, experiment references, package versions.
- `seed_42/`, `seed_43/`, `seed_44/`: separate runs and unaveraged results.
- Within each seed, `0.25x/`, `0.5x/`, and `1x/` through `10x/`: datasets,
  generation settings, baseline predictions and metrics.
- Model directories: configuration, histories, loss/accuracy/entropy curves,
  gradients before clipping, and final train/test confusion matrices.
- `baselines/bag_of_tokens/`: codebook centers, token IDs, configuration, and diagnostics.
- Root `results.json`, `comparison.csv`, and five mean final test balanced accuracy plots.
- `summary.pdf` and `summary.tex` at the experiment root: bold ablation titles,
  italic unchanged defaults, and variant-by-normalization tables for temporal
  position (2x2 scores), channel position (2x2), and aggregation (4x2). A second
  page plots balanced accuracy against depth and dataset size, including all
  three baselines. Scores use the final test split; defaults come from saved config.

Rebuild a report from a completed experiment without retraining:

```sh
python -m results.synthetic_runs.sinusoid_amplitude.arch_ablations.report results/synthetic_runs/sinusoid_amplitude/arch_ablations/<run-folder>
```

Vector plots sit beside the PNG plots; LaTeX logs and intermediate files stay in
`report_build/`.

Datasets and plots can occupy several GB. Results average three seeded splits per
size; they do not provide confidence intervals. Failures propagate without
skipping experiments or substituting configurations. Completed transformer pairs
and dataset sizes are saved as the suite advances. No checkpoints are saved.

```sh
python -m unittest results.synthetic_runs.sinusoid_amplitude.arch_ablations.test_ablations
```

Validation runs all architecture variants across three seeds for one epoch
on 20–160 recordings, then exercises reuse. Its artifacts stay in a fresh `validation_*/suite/` folder.
It also checks normalization scope, binary compatibility, multiclass loss,
one-factor configurations, seed averaging, baseline reuse, mismatched reuse rejection,
output containment, and overwrite rejection.
