from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class ModelTests(unittest.TestCase):
    def setUp(self):
        import torch

        from seformer.config import load_config

        self.torch = torch
        self.config = load_config(ROOT / "configs" / "smoke.yaml")

    def test_forward_backward_and_attention(self):
        from seformer.model import build_model

        model = build_model(self.config)
        video = self.torch.randn(2, 3, 8, 32, 32, requires_grad=True)
        output = model(video, return_attention=True)
        self.assertEqual(tuple(output["logits"].shape), (2, 4))
        self.assertIn("pool_global", output["attention"])
        self.assertIn("cvaf_coarse_from_mid", output["attention"])
        output["logits"].sum().backward()
        self.assertIsNotNone(video.grad)

    def test_2d_patchification(self):
        from seformer.model import build_model

        config = {**self.config, "model": {**self.config["model"], "patchification": "2d"}}
        model = build_model(config)
        logits = model(self.torch.randn(1, 3, 8, 32, 32))
        self.assertEqual(tuple(logits.shape), (1, 4))

    def test_wrong_runtime_shape_fails_clearly(self):
        from seformer.model import build_model

        model = build_model(self.config)
        with self.assertRaises(ValueError):
            model(self.torch.randn(1, 3, 7, 32, 32))


if __name__ == "__main__":
    unittest.main()
