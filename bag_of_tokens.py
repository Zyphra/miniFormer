import torch
from torch import nn
from torch.nn import functional as F


class BagOfTokensClassifier(nn.Module):
    """Binary logistic regression on token counts divided by vocab_size.

    Input: torch.long token IDs in [0, vocab_size), shaped (batch_size, ...).
    Output: raw logits shaped (batch_size, 1), for BCEWithLogitsLoss.
    loss() adds L2 regularization to weight; smaller C means a stronger penalty.
    """

    def __init__(self, vocab_size, C=1.0):
        super().__init__()
        if vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if not C > 0:
            raise ValueError("C must be positive")
        self.vocab_size = vocab_size
        self.C = C
        self.weight = nn.Parameter(torch.zeros(vocab_size, 1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, token_ids):
        token_ids = token_ids.flatten(start_dim=1)

        # Count each token ID across all positions in each example.
        counts = self.weight.new_zeros((token_ids.shape[0], self.vocab_size))
        counts.scatter_add_(
            1, token_ids, torch.ones_like(token_ids, dtype=counts.dtype)
        )
        features = counts / self.vocab_size
        return features @ self.weight + self.bias

    def loss(self, logits, targets, n_train):
        """Mean binary cross-entropy + L2, with sklearn's unweighted C scaling.

        targets contains binary labels (0 or 1).
        n_train is the full training-set size, even when using minibatches.
        Only weight is penalized; bias is unpenalized.
        """
        if n_train < 1:
            raise ValueError("n_train must be positive")
        targets = targets.to(logits).reshape_as(logits)
        cross_entropy = F.binary_cross_entropy_with_logits(logits, targets)
        l2 = self.weight.square().sum() / (2 * self.C * n_train)
        return cross_entropy + l2
