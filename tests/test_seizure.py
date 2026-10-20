"""CPU checks for window labels and NeuralBench-compatible training sampling."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from training.run import make_loaders, read_labels
from tokenization.labram import resolve_channel_ids


class SeizureTests(unittest.TestCase):
    def test_bipolar_position_fallback(self):
        vocabulary = {"FP1": 1, "F3": 18, "FP1-F7": 121}
        self.assertEqual(resolve_channel_ids(["Fp1-F7", "Fp1-F3", "F3-C3", "Fp1"], vocabulary),
                         {"Fp1-F7": 121, "Fp1-F3": 1, "F3-C3": 18, "Fp1": 1})
        with self.assertRaisesRegex(ValueError, "No LaBraM channel embedding"):
            resolve_channel_ids(["unknown-F3"], vocabulary)

    def test_labels_vary_within_recording(self):
        data = {"label_mapping": {"seizure": 0, "background": 1}, "recordings": [], "windows": []}
        for recording_id, split in enumerate(("train", "val", "test")):
            data["recordings"].append(dict(recording_id=recording_id, subject=str(recording_id), split=split))
            for label in (1, 0):
                data["windows"].append(dict(example_id=len(data["windows"]), recording_id=recording_id, label=label))
        with TemporaryDirectory() as folder:
            path = Path(folder) / "cache.json"
            path.write_text(json.dumps(data))
            _, ids, labels, splits = read_labels(path)
            self.assertEqual(ids.tolist(), list(range(6)))
            self.assertEqual(labels.tolist(), [1, 0, 1, 0, 1, 0])
            self.assertEqual({s: i.tolist() for s, i in splits.items()},
                             {"train": [0, 1], "val": [2, 3], "test": [4, 5]})
            data["recordings"][1]["subject"] = "0"
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(AssertionError, "Subject crosses splits"):
                read_labels(path)

    def test_balanced_sampling_preserves_evaluation(self):
        ids = torch.arange(104).reshape(-1, 1, 1)
        labels = torch.tensor([0] * 2 + [1] * 98 + [0, 1, 0, 1])
        splits = {"train": torch.arange(100), "val": torch.arange(100, 102), "test": torch.arange(102, 104)}
        args = ids, labels, splits, 10, 33, {"name": "ClassificationSampler"}
        first, second = make_loaders(*args), make_loaders(*args)
        weights = first["train"].sampler.weights
        self.assertAlmostEqual(weights[:2].sum().item(), weights[2:].sum().item(), places=5)
        self.assertEqual(first["train"].sampler.num_samples, 100)
        self.assertTrue(first["train"].sampler.replacement)
        for epoch in range(2):
            order = torch.cat([x[:, 0, 0] for x, _ in first["train"]])
            repeat = torch.cat([x[:, 0, 0] for x, _ in second["train"]])
            self.assertTrue(torch.equal(order, repeat))
        for split in ("val", "test"):
            actual = torch.cat([x[:, 0, 0] for x, _ in first[split]])
            self.assertTrue(torch.equal(actual, splits[split]))


if __name__ == "__main__":
    unittest.main()
