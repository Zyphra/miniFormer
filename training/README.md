# Classifier training

`run.py` trains one transformer per token source and seed, preserving labels and subject splits.
Cached sources use `{"name":"dac","cache":"token_cache/dac/depression/vq_long_006-best"}`.
Live sources use `{"name":"dac","tokenizer":"dac","repo":"/path/to/dac","checkpoint":"/path/to/weights.pth"}`.
Live inference runs once per source, keeping tokens in RAM across seeds; its environment needs tokenizer and classifier dependencies.

```sh
python -m training.run --dataset-metadata 'dataset cache/depression/data/cache.json' --sources '{"name":"dac","cache":"token_cache/dac/depression/vq_long_006-best"}' --seeds 33 34 35 --output results/depression/comparison-001
```

Defaults: `--seeds 33 34 35`, fixed `--data-seed 33`; require at least two distinct seeds. Pass multiple `--sources`; `--example-ids` must retain every class in each split.
Each `<source>/seed_<seed>/` holds config, weights and metrics. Validation balanced accuracy selects the earliest best epoch; test runs once per classifier.
`comparison.csv` lists every source/seed; `summary.csv` gives per-source means, sample standard deviations and seed counts after checking completeness. CPU checks: `python -m tests.test_run`.
