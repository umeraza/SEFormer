from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seformer.sampling import sample_frame_indices, temporal_coverage


class SamplingTests(unittest.TestCase):
    def test_center_sampling(self):
        self.assertEqual(
            sample_frame_indices(100, 4, 2, training=False),
            [46, 48, 50, 52],
        )

    def test_short_clip_repeats_last(self):
        self.assertEqual(
            sample_frame_indices(3, 5, 1, training=False),
            [0, 1, 2, 2, 2],
        )

    def test_training_sampling_is_seeded(self):
        first = sample_frame_indices(100, 8, 2, training=True, rng=random.Random(9))
        second = sample_frame_indices(100, 8, 2, training=True, rng=random.Random(9))
        self.assertEqual(first, second)

    def test_stride_changes_coverage_not_tensor_length(self):
        self.assertEqual(temporal_coverage(32, 1), 32)
        self.assertEqual(temporal_coverage(32, 2), 63)


if __name__ == "__main__":
    unittest.main()
