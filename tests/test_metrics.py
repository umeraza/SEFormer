from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seformer.metrics import classification_metrics, softmax_numpy


class MetricTests(unittest.TestCase):
    def test_macro_metrics_and_fixed_class_order(self):
        targets = np.array([0, 0, 1, 1, 2, 2])
        predictions = np.array([0, 1, 1, 1, 2, 0])
        result = classification_metrics(
            targets,
            predictions,
            num_classes=3,
            class_names=["a", "b", "c"],
            average="macro",
        )
        self.assertAlmostEqual(result["accuracy"], 4 / 6)
        self.assertEqual(result["confusion_counts"], [[1, 1, 0], [0, 2, 0], [1, 0, 1]])
        self.assertEqual([row["name"] for row in result["per_class"]], ["a", "b", "c"])
        expected_paper_f1 = (
            2
            * result["macro_precision"]
            * result["macro_recall"]
            / (result["macro_precision"] + result["macro_recall"])
        )
        self.assertAlmostEqual(result["paper_f1"], expected_paper_f1)

    def test_paper_daisee_f1_is_not_conventional_macro_f1(self):
        # Counts reconstructed exactly from the class-wise discussion.
        matrix = np.array(
            [
                [2, 2, 0, 0],
                [0, 64, 16, 4],
                [0, 1, 629, 252],
                [0, 0, 148, 666],
            ]
        )
        targets = []
        predictions = []
        for actual in range(4):
            for predicted in range(4):
                targets.extend([actual] * int(matrix[actual, predicted]))
                predictions.extend([predicted] * int(matrix[actual, predicted]))
        result = classification_metrics(
            targets,
            predictions,
            num_classes=4,
            class_names=["Very Low", "Low", "High", "Very High"],
            average="macro",
        )
        self.assertAlmostEqual(result["accuracy"], 0.7629, places=4)
        self.assertAlmostEqual(result["paper_f1"], 0.7738, places=4)
        self.assertGreater(abs(result["paper_f1"] - result["macro_f1"]), 0.01)

    def test_softmax_rows_sum_to_one(self):
        probabilities = softmax_numpy(np.array([[1.0, 2.0], [1000.0, 1001.0]]))
        np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2))
        np.testing.assert_allclose(probabilities[0], probabilities[1])


if __name__ == "__main__":
    unittest.main()
