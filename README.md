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

This setup provides the classifiers and directory layout. Token extraction,
cache loading, and classifier training/evaluation loops are still to be added.
