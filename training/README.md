# Classifier training

`run.py` trains one transformer per named token source, preserving the dataset's labels and subject splits.
Cached sources use `{"name":"dac","cache":"token_cache/dac/depression/vq_long_006-best"}`.
Live sources use `{"name":"dac","tokenizer":"dac","repo":"/path/to/dac","checkpoint":"/path/to/weights.pth"}`.
Live inference reads the EEG cache and keeps tokens in RAM; the Python environment needs both tokenizer and classifier dependencies.

```sh
python -m training.run --dataset-metadata 'dataset cache/depression/data/cache.json' --sources '{"name":"dac","cache":"token_cache/dac/depression/vq_long_006-best"}' --output results/depression/comparison-001
```

Pass multiple JSON sources after `--sources`; use `--example-ids` for an explicit subset, retaining every class in each split.
Validation balanced accuracy selects the earliest best epoch; test runs once. Outputs include config with input hashes, classifier weights and metrics; `comparison.csv` marks completion.
`python -m tests.test_run` checks alignment, cached/live equivalence, checkpoint selection and failure cases on CPU.
