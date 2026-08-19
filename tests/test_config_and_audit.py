from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seformer.analysis import analytical_parameter_count, token_accounting
from seformer.config import ConfigError, apply_overrides, load_config, validate_config


class ConfigAndAuditTests(unittest.TestCase):
    def test_daisee_inheritance_and_override(self):
        config = load_config(
            ROOT / "configs" / "daisee.yaml",
            ["data.temporal_stride=2", "training.amp=false"],
        )
        self.assertEqual(config["data"]["frames"], 32)
        self.assertEqual(config["data"]["temporal_stride"], 2)
        self.assertFalse(config["training"]["amp"])
        self.assertEqual(len(config["model"]["views"]), 3)

    def test_paper_token_formula(self):
        config = load_config(ROOT / "configs" / "daisee.yaml")
        accounting = token_accounting(config)
        self.assertEqual(
            [row["patch_tokens"] for row in accounting["views"]],
            [3136, 1568, 784],
        )
        self.assertEqual(accounting["total_patch_tokens"], 5488)
        self.assertEqual(accounting["total_tokens_with_cls"], 5491)

    def test_analytical_parameters_are_near_reported_scale(self):
        config = load_config(ROOT / "configs" / "daisee.yaml")
        total = analytical_parameter_count(config)["total"]
        self.assertGreater(total, 27_000_000)
        self.assertLess(total, 31_000_000)

    def test_invalid_head_divisibility_is_rejected(self):
        config = load_config(ROOT / "configs" / "smoke.yaml")
        invalid = copy.deepcopy(config)
        invalid["model"]["views"][0]["heads"] = 3
        with self.assertRaises(ConfigError):
            validate_config(invalid)

    def test_override_cannot_descend_through_scalar(self):
        with self.assertRaises(ConfigError):
            apply_overrides({"seed": 42}, ["seed.value=3"])


if __name__ == "__main__":
    unittest.main()
