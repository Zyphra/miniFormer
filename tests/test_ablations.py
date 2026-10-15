"""Run with python -m unittest tests.test_ablations.

The integration test exercises three seeds on 20..160 recordings, one epoch.
Validation artifacts stay in a new validation_* directory under arch_ablations.
"""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F

from results.synthetic_runs.sinusoid_amplitude.arch_ablations import run
from models.bag_of_tokens import BagOfTokensClassifier
from models.transformer_classifier import TransformerClassifier
from synthetic_data.sinusoid_amplitude.generate import make_dataset


class AblationTests(unittest.TestCase):
    def test_normalization_scope(self):
        torch.manual_seed(42)
        inputs, coordinates = torch.randn(2, 3, 4, 8), torch.randn(3, 3)
        for temporal, channel, affected in (("rotary", None, False), ("rotary", "spatial", True),
                                             ("sinusoidal", None, True)):
            model = TransformerClassifier(4, channels=3, coarse_time=4, codebook_dim=8,
                                          temporal_pos=temporal, channel_pos=channel,
                                          channel_coords=coordinates, normalize=False).eval()
            before = model(inputs)
            model.normalize = True
            self.assertEqual(not torch.equal(before, model(inputs)), affected)

    def test_bag_binary_compatibility_and_multiclass_loss(self):
        tokens = torch.tensor([[0, 1, 1], [2, 2, 0]])
        counts = torch.tensor([[1, 2, 0], [1, 0, 2]]) / 3
        for classes in (2, 4):
            model = BagOfTokensClassifier(3, n_classes=classes)
            with torch.no_grad():
                model.weight.copy_(torch.arange(model.weight.numel()).reshape_as(model.weight) / 10)
            logits = model(tokens)
            torch.testing.assert_close(logits, counts @ model.weight + model.bias)
            targets = torch.tensor([0, classes - 1])
            expected = (F.binary_cross_entropy_with_logits(logits, targets.float()[:, None])
                        if classes == 2 else F.cross_entropy(logits, targets))
            expected += model.weight.square().sum() / 4
            loss = model.loss(logits, targets, n_train=2)
            torch.testing.assert_close(loss, expected)
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertEqual(BagOfTokensClassifier(3).weight.shape, (3, 1))

    def test_full_sweep(self):
        def small_dataset(samples_per_class=20, seed=42, boost_channels=None):
            return make_dataset(samples_per_class=samples_per_class, seed=seed, boost_channels=boost_channels)

        output = Path(tempfile.mkdtemp(prefix="validation_", dir=run.ROOT)) / "suite"
        seeds, sizes = (42, 43, 44), (0.25, 0.5, 1, 2)
        torch.set_num_threads(2)
        with (patch.object(run, "make_dataset", small_dataset),
              patch.object(run, "evaluate", wraps=run.evaluate) as power,
              patch.object(run, "run_bag_of_tokens", wraps=run.run_bag_of_tokens) as bag,
              patch.object(run, "run_experiment", wraps=run.run_experiment) as transformer,
              run.threadpool_limits(limits=2)):
            results = run.run_suite(output, epochs=1, seeds=seeds, sizes=sizes)
        run.save_report(output, note="VALIDATION ONLY: one epoch on reduced datasets; not production results.")
        self.assertEqual((transformer.call_count, power.call_count, bag.call_count), (51, 12, 12))
        self.assertEqual(len(results), 21)
        default = json.loads((output / "seed_42/1x/default/config.json").read_text())
        self.assertEqual(default["depth"], 4)
        runs, comparisons = run.experiment_plan(sizes)
        self.assertNotIn("None", comparisons["channel_pos"])
        for seed in seeds:
            for name, overrides in runs.items():
                config = json.loads((output / f"seed_{seed}/1x" / name / "config.json").read_text())
                for key in run.ABLATIONS:
                    self.assertEqual(config[key], {**default, **overrides}[key])
                self.assertEqual(config["seed"], seed)
                self.assertEqual(config["normalize"], [False, True])
        for parameter, entries in comparisons.items():
            self.assertTrue((output / f"{parameter}.png").is_file())
            for name in entries.values():
                for artifact in ("config.json", "metrics.json", "learning_curves.png", "gradients.png", "confusion_matrices.png"):
                    self.assertTrue((output / "seed_42" / name / artifact).is_file())
        for size in sizes:
            for metrics in results[f"{size}x/baselines"].values():
                self.assertEqual(metrics["train"]["n_examples"], 64 * size)
                self.assertEqual(metrics["test"]["n_examples"], 16 * size)
        individual = [json.loads((output / f"seed_{seed}/results.json").read_text()) for seed in seeds]
        for name, models in results.items():
            for model in models:
                expected = sum(item[name][model]["test"]["balanced_accuracy"] for item in individual) / 3
                self.assertAlmostEqual(models[model]["test"]["balanced_accuracy"], expected)
        with (output / "comparison.csv").open() as file:
            self.assertEqual(len(list(csv.DictReader(file))), 105)
        self.assertTrue((output / "summary.pdf").read_bytes().startswith(b"%PDF-"))
        source = (output / "summary.tex").read_text()
        self.assertEqual(source.count(r"\begin{tabular*}"), 3)
        self.assertIn("80 recordings (64 train / 16 test)", source)
        self.assertIn("3 seeds (42, 43, 44)", source)
        self.assertIn(r"\textbf{Learned vs spatial}", source)
        self.assertNotIn("None &", source)
        self.assertIn(r"\textbf{RoPE vs sinusoidal}", source)
        self.assertIn(r"\texttt{normalize=False} & \texttt{normalize=True}", source)
        self.assertEqual(run.plt.get_fignums(), [])
        with self.assertRaises(FileExistsError):
            run.run_suite(output, epochs=1)
        with self.assertRaises(AssertionError):
            run.run_suite(run.ROOT.parent / "outside", epochs=1)
        suite = json.loads((output / "config.json").read_text())
        with (patch.object(run, "make_dataset", small_dataset),
              patch.object(run, "run_experiment", wraps=run.run_experiment) as transformer,
              run.threadpool_limits(limits=2)):
            reused = run.run_seed(output / "reuse_check", suite, 42, output / "seed_42")
        self.assertEqual(transformer.call_count, 8)
        for name in comparisons["depth"].values():
            self.assertEqual(reused[name], individual[0][name])
        with self.assertRaises(AssertionError):
            run.copy_run(output / "seed_42/1x/default", output / "invalid_copy",
                         output / "seed_42/1x/dataset.npz", {"depth": 2})
        self.assertFalse((output / "invalid_copy").exists())
        with self.assertRaises(AssertionError):
            run.average_results([individual[0], {}])
        print(f"Validation artifacts: {output}", flush=True)


if __name__ == "__main__":
    unittest.main()
