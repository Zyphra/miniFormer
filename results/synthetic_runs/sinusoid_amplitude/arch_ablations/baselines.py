"""Baselines on the same recording split as the transformer."""

import json
from functools import partial

import numpy as np
import sklearn
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from torch.utils.data import TensorDataset

from models.bag_of_tokens import BagOfTokensClassifier
from synthetic_data.sinusoid_amplitude.train_transformer import fit, load_data, save_diagnostics


def power_metrics(data, settings):
    """Use the predictions already fitted by synthetic_data.sinusoid_amplitude.generate.evaluate."""
    results = {}
    for name in ("svm", "kmeans"):
        results[name] = {"settings": settings[name]}
        for split in ("train", "test"):
            indices = data[f"{split}_indices"]
            labels, predictions = data["y"][indices], data[f"{name}_predictions"][indices]
            results[name][split] = {
                "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
                "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
                "n_examples": len(indices),
            }
    return results


def run_bag_of_tokens(data_path, output, transformer_config, vocab_size=128):
    """Fit a patch codebook on training recordings only, then fit token counts."""
    training, testing, _, n_classes = load_data(data_path)
    assert n_classes > 2, "This runner uses multiclass logits for shared diagnostics"
    output.mkdir(parents=True, exist_ok=False)
    config = {key: transformer_config[key] for key in (
        "epochs", "batch_size", "seed", "optimizer", "optimizer_settings", "max_grad_norm",
        "n_train", "n_test", "n_classes", "data_path", "torch_version", "threads", "device",
    )}
    codebook = MiniBatchKMeans(n_clusters=vocab_size, batch_size=4096, n_init=3,
                              random_state=config["seed"])
    model = BagOfTokensClassifier(vocab_size=vocab_size, n_classes=n_classes)
    config.update(
        model_class="models.bag_of_tokens.BagOfTokensClassifier", output=str(output),
        vocab_size=vocab_size, C=model.C, parameters=sum(p.numel() for p in model.parameters()),
        tokenizer="MiniBatchKMeans on unnormalized 128-sample training patches only",
        tokenizer_settings=codebook.get_params(), sklearn_version=sklearn.__version__,
    )
    (output / "config.json").write_text(json.dumps(config, indent=2, allow_nan=False) + "\n")
    codebook.fit(training.tensors[0].flatten(0, 2).numpy())
    datasets = []
    for dataset in (training, testing):
        patches, labels = dataset.tensors
        tokens = codebook.predict(patches.flatten(0, 2).numpy()).reshape(patches.shape[:-1])
        datasets.append(TensorDataset(torch.from_numpy(tokens).long(), labels))
    np.savez_compressed(output / "tokens.npz", centers=codebook.cluster_centers_,
                        train=datasets[0].tensors[0].numpy(), test=datasets[1].tensors[0].numpy())
    history = fit(model, *datasets, config, loss_function=partial(model.loss, n_train=len(training)))
    save_diagnostics({"bag_of_tokens": history}, config)
    return {split: history[-1][split] for split in ("train", "test")}
