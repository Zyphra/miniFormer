# miniFormer

Compare classifiers on tokens produced by different EEG models.

## Layout

```text
models/
    bag_of_tokens.py          # Logistic regression on integer token counts
    transformer_classifier.py # Transformer on token embeddings
tokenizers/
    labram/                   # Independent LaBraM Git clone (ignored)
token_cache/
    labram/
        <dataset>/
            <id_name>/        # Cached outputs for this tokenizer and dataset
results/                      # Metrics, plots, and classifier checkpoints
```

Use `token_cache/<tokenizer>/<dataset>/<id_name>/` for each cache. The dataset
and ID directories are added when saving tokens. Tokenizer repositories and
generated cache/result files are local and ignored by Git.

## Classifiers

```python
from models.bag_of_tokens import BagOfTokensClassifier
from models.transformer_classifier import TransformerClassifier
```

The bag-of-tokens model takes integer token IDs shaped `(batch_size, ...)`.
It divides token counts by `vocab_size` and returns one binary logit per example.
Its `loss()` method adds L2 regularization to binary cross-entropy.

The transformer takes embeddings shaped
`(batch_size, channels, coarse_time, codebook_dim)` and returns class logits.
Both use PyTorch; the transformer also requires `linear_attention_transformer`.

### Synthetic EEG transformer

```sh
python -m pip install torch linear-attention-transformer numpy matplotlib
python -m synthetic_data.train_transformer
```

This uses the existing `synthetic_data/outputs/dataset.npz`. Each channel is
split into ten non-overlapping 0.5-second windows at 256 Hz: 128 values per
patch. The classifier projects `(B, 19, 10, 128)` to `(B, 19, 10, 64)`, adds
sinusoidal time and spatial channel embeddings, flattens to 190 tokens, and
prepends CLS. Two attention heads and two layers produce four class logits.

`train()` and `test()` are in `models/transformer_classifier.py`. `train()` runs
one epoch using `zero_grad`, forward, cross-entropy, backward, gradient clipping,
and `optimizer.step`. `test()` uses evaluation mode and disables gradients.
The experiment script handles the epoch loop, data preparation, and plotting.

The two runs use `normalize=False` and `normalize=True`, identical initial
weights, and identical minibatch order. This flag gives each **positional
embedding** unit L2 norm; it does not normalize raw EEG or patch embeddings.
Spatial coordinates are the saved head coordinates in meters, in channel order.

Defaults are 30 epochs, batch size 32, Adam with learning rate 0.001, and global
gradient clipping at L2 norm 1. The saved 640/160 recording split stays intact.
Curves show train/test cross-entropy, balanced accuracy (mean class recall),
and prediction entropy. Both splits are evaluated using fixed end-of-epoch
weights. Gradient plots show each optimizer step and mean per-parameter norms
before clipping; non-finite losses or gradients raise an error. Final confusion
matrices show which classes are being confused. These diagnostics help check
training; they do not prove correctness or establish performance across seeds.
Test curves are descriptive: this script does not select an epoch or tune
hyperparameters using the test split.

Each run saves to a new dated folder under
`results/synthetic_data/sinusoid_amplitude/`. `config.json` is written before
training and records model initialization settings, both normalization variants,
channel coordinates and order, patch settings, optimizer settings, seed, data
path, and runtime details. The folder also contains `metrics.json`,
`learning_curves.png`, `gradients.png`, and `confusion_matrices.png`.
No checkpoints are saved. Use `--output <new-directory>` to choose a run folder;
existing folders are never overwritten. `--epochs`, `--batch-size`,
`--learning-rate`, and `--seed` can be adjusted. This experiment uses raw
waveform patches; no pretrained tokenizer is involved.

## LaBraM

The first tokenizer source is the official [LaBraM repository](https://github.com/935963004/LaBraM).
To set it up in a fresh checkout of miniFormer:

```sh
mkdir -p tokenizers
git clone https://github.com/935963004/LaBraM.git tokenizers/labram
```

The tokenizer is `VQNSP` in `tokenizers/labram/modeling_vqnsp.py`.
`get_codebook_indices()` returns integer token IDs; `get_tokens()` also returns
quantized embeddings. Its checkpoint is `tokenizers/labram/checkpoints/vqnsp.pth`.

Keep each additional tokenizer repository in `tokenizers/<tokenizer>/` and
follow its own dependency and checkpoint instructions. The entire `tokenizers/`
folder is ignored so these repositories retain their own Git history.

Token extraction and cache loading for pretrained tokenizers are still to be
added. The synthetic raw-patch experiment above can already train and evaluate
the transformer.
