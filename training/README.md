# Classifier training

`run.py` trains one transformer per token source and seed, preserving labels and subject splits.
Cached sources use `{"name":"dac","cache":"token_cache/dac/depression/checkpoint"}`.
Live sources use `{"name":"dac","tokenizer":"dac","repo":"/path/to/dac","checkpoint":"/path/to/weights.pth"}`.
Live inference runs once per source, keeping tokens in RAM across seeds; its environment needs tokenizer and classifier dependencies.

```sh
python -m training.run --dataset-metadata 'dataset cache/depression/data/cache.json' --sources '{"name":"dac","cache":"token_cache/dac/depression/checkpoint"}' --seeds 33 34 35 --output results/depression/experiment
```

Defaults: `--seeds 33 34 35`, fixed `--data-seed 33`; require at least two distinct seeds. Pass multiple `--sources`; `--example-ids` must retain every class in each split.
Each `<source>/seed_<seed>/` holds config, weights and metrics. Validation balanced accuracy selects the earliest best epoch; test runs once per classifier.
Labels may be stored per window (for events such as seizures) or per recording.
If the EEG cache requests NeuralBench's `ClassificationSampler`, training uses
inverse-frequency sampling with replacement for one dataset-length epoch.
Validation and test always retain all examples in their original order.
`comparison.csv` lists every source/seed; `summary.csv` gives per-source means, sample standard deviations and seed counts after checking completeness. CPU checks: `python -m tests.test_run`.

Keep each experiment's launch script and log inside its output folder, for example
`results/<experiment>/run.sh` and `run.log`.
Keep temporary preflight scripts and logs in `/tmp`, outside the results folders.
The runner accepts an existing output folder containing only `.sh` and `.log` files;
it refuses folders containing results. Launch scripts should use `set -o noclobber`
before redirecting output to their log so rerunning a script cannot overwrite it.

`run_comparisons.py` plots saved checkpoints using a dictionary of panels. Each
panel has a short `note` describing what it represents and `groups` mapping color
groups to checkpoint names and directories. Checkpoint names become the x-axis
labels. Every checkpoint in a group uses
one solid color; the same group name keeps that color across panels. Paths are
relative to the results root and contain `seed_*/metrics.json`. Repeat a path to
reuse a baseline.

```python
from training.run_comparisons import plot

plot({
    "Panel A": {
        "note": "Effect of window size across codebook sizes",
        "groups": {
            "Smaller windows": {"model_a": "experiment/model_a", "model_b": "experiment/model_b"},
            "Default windows": {"model_c": "experiment/model_c", "model_d": "experiment/model_d"},
        },
    },
}, results_root="results", colors={"Smaller windows": "#0072B2", "Default windows": "#D55E00"})
```

For the CLI, save the dictionary as JSON and run
`python -m training.run_comparisons --checkpoints checkpoint_groups.json --results-root results`.
Missing or blank notes prompt for a description of each panel; blank answers
prompt again. Supplied notes skip the prompt. The Python function requires notes
in the dictionary. Notes appear beneath panel titles. Save notes in the input
JSON to reuse them; interactive answers do not modify the input file.

Colors are assigned automatically by group name. Override them with the optional
`colors` dictionary, or `--colors group_colors.json` on the CLI.
Bars show mean test balanced accuracy with sample standard deviation across the
available classifier seeds. The only output is `<results-root>/comparison.png`;
the results root must already exist.
