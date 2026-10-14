"""Compare linear classifiers and finite-sample separation on the saved dataset."""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits


def nonseparable_pairs(features, labels):
    """Check whether each class pair admits an affine hyperplane with margin 1."""
    design = np.column_stack((features, np.ones(len(features))))
    failures = []
    for first, second in combinations(np.unique(labels), 2):
        selected = (labels == first) | (labels == second)
        signs = np.where(labels[selected] == first, 1, -1)
        constraints = -signs[:, None] * design[selected]
        solution = linprog(np.zeros(design.shape[1]), A_ub=constraints,
                           b_ub=-np.ones(selected.sum()), bounds=(None, None), method="highs")
        assert solution.status in (0, 2), solution.message
        if solution.success:
            assert np.max(constraints @ solution.x + 1) < 1e-6
        else:
            failures.append([int(first), int(second)])
    return failures


def main():
    root = Path(__file__).resolve().parent
    with np.load(root / "dataset.npz") as data:
        labels, train, test = data["y"], data["train_indices"], data["test_indices"]
        power, mask = data["channel_power"], data["boost_channel_mask"]
    features = {"all_channel_mean": power.mean(axis=1),
                "boosted_channel_mean": power[:, mask].mean(axis=1),
                "all_channels_flattened": power.reshape(len(labels), -1)}
    report = {"svm": {"kernel": "linear", "C": 1.0, "scaling": "StandardScaler fitted on training data"},
              "separation_check": "Pairwise affine hyperplane feasibility on all 800 saved points; "
                                  "uses all labels for geometry only, separately from held-out SVM evaluation.",
              "features": {}}
    with threadpool_limits(limits=2):
        for name, values in features.items():
            scaled = StandardScaler().fit(values[train]).transform(values)
            model = SVC(kernel="linear", C=1.0).fit(scaled[train], labels[train])
            report["features"][name] = {
                "n_features": values.shape[1], "train_accuracy": model.score(scaled[train], labels[train]),
                "test_accuracy": model.score(scaled[test], labels[test]),
                "nonseparable_class_pairs": nonseparable_pairs(scaled, labels),
            }
    (root / "feature_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
