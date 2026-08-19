from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seformer.manifest import load_manifest, manifest_summary


class ManifestTests(unittest.TestCase):
    def test_relative_paths_and_validation_alias(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            media = directory / "clip.npz"
            media.touch()
            manifest_path = directory / "manifest.csv"
            pd.DataFrame(
                [
                    {
                        "path": "clip.npz",
                        "label": 2,
                        "split": "Validation",
                        "sample_id": "sample-1",
                        "subject_id": "person-1",
                    }
                ]
            ).to_csv(manifest_path, index=False)
            frame = load_manifest(manifest_path)
            self.assertEqual(frame.loc[0, "path"], str(media.resolve()))
            self.assertEqual(frame.loc[0, "split"], "val")
            summary = manifest_summary(frame)
            self.assertEqual(summary["samples"], 1)
            self.assertEqual(summary["subjects"], 1)

    def test_missing_required_column(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.csv"
            pd.DataFrame([{"path": "x", "label": 0}]).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_manifest(path)


if __name__ == "__main__":
    unittest.main()
