# Tokenization

`dac.py`, `tfm.py`, and `labram.py` load upstream models and apply their preprocessing.
`pipeline.batches(dataset, tokenizer)` streams `(B,C,T)` IDs and example IDs without writing files.
`tokenizer.codebook[ids]` gives `(B,C,T,D)` classifier inputs; example IDs link to cached labels and splits.

To save tokens, run from miniFormer in the tokenizer's Python environment:
```sh
python -m tokenization.cache dac --repo <repo> --checkpoint <weights> --output <new-folder>
```

- Selected DAC checkpoint: `/data/groups/bci/chris/workspace/bci_descript-audio-codec/runs/vq_long_006/best/dac/weights.pth`.
- DAC architecture comes from the checkpoint. Set classifier `codebook_dim=tokenizer.codebook.shape[1]`.
- Verify with `python -m tests.test_pipeline dac --repo <repo> --checkpoint <weights>`; use `tfm` or `labram` for those adapters.
