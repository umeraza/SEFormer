#!/usr/bin/env python3
"""Audit paper equations, token claims, and inferred implementation values."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from seformer.analysis import analytical_parameter_count, token_accounting
from seformer.config import load_config
from seformer.sampling import temporal_coverage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument(
        "--strict-known",
        action="store_true",
        help="Fail only if the canonical DAiSEE configuration no longer matches stated paper values",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    tokens = token_accounting(config)
    data = config["data"]
    canonical_schedule = [view["patch_size"] for view in config["model"]["views"]]
    canonical = (
        data["frames"] == 32
        and data["image_size"] == [112, 112]
        and canonical_schedule == [[2, 8, 8], [4, 8, 8], [8, 8, 8]]
    )
    patch_table = []
    for name, temporal, spatial, reported in (
        ("temporal_dense", [1, 4, 8], 8, 1782),
        ("spatial_fine", [2, 4, 8], 4, 2564),
        ("proposed", [2, 4, 8], 8, 1288),
        ("spatial_coarse", [2, 4, 8], 16, 332),
    ):
        computed = sum(
            (data["frames"] // temporal_patch)
            * (data["image_size"][0] // spatial)
            * (data["image_size"][1] // spatial)
            for temporal_patch in temporal
        )
        patch_table.append(
            {
                "schedule": name,
                "computed_from_equation": computed,
                "paper_reported": reported,
                "difference": computed - reported,
            }
        )
    payload = {
        "config": str(Path(args.config).resolve()),
        "input": {
            "sampled_frames": data["frames"],
            "temporal_stride": data["temporal_stride"],
            "source_frame_coverage": temporal_coverage(
                data["frames"], data["temporal_stride"]
            ),
            "image_size": data["image_size"],
        },
        "token_accounting": tokens,
        "paper_patch_table_reported_tokens": 1288 if canonical else None,
        "paper_token_difference": tokens["total_patch_tokens"] - 1288 if canonical else None,
        "patch_schedule_table_audit": patch_table,
        "analytical_parameters": analytical_parameter_count(config),
        "paper_reported_full_parameters": 28_500_000 if canonical else None,
        "known_dataset_discrepancies": {
            "baum1s_manuscript_clips": 1222,
            "baum1_official_uci_clips": 1184,
            "baum1_manuscript_states_described": 12,
            "baum1_official_uci_states": 13,
        },
        "methodology_ambiguities": [
            "CVAF is called bidirectional in the abstract but is unidirectional in the equation.",
            "Metric equations are binary while both tasks are multiclass.",
            "BAUM-1s split assignments and metric averaging are not specified.",
            "View dimensions, heads, training batch size, and augmentation magnitudes are omitted.",
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.strict_known:
        errors = []
        if config["experiment"] == "seformer_daisee" and not canonical:
            errors.append("Canonical DAiSEE config no longer matches the stated input and patch schedule")
        if canonical and tokens["total_patch_tokens"] != 5488:
            errors.append("Patch formula should yield 5488 tokens for the canonical configuration")
        if errors:
            print("\n".join(errors), file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
