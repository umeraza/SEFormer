#!/usr/bin/env python3
"""Aggregate resolved configurations and best validation metrics across runs."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import pandas as pd

from seformer.analysis import analytical_parameter_count, token_accounting
from seformer.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, help="Quoted run-directory glob")
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-test", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric_columns(prefix: str, metrics: dict | None) -> dict:
    if not metrics:
        return {}
    return {
        f"{prefix}_{key}": value
        for key, value in metrics.items()
        if isinstance(value, (int, float, str)) and key not in {"num_samples", "average"}
    }


def main() -> int:
    args = parse_args()
    directories = sorted(Path(path).resolve() for path in glob.glob(args.runs) if Path(path).is_dir())
    if not directories:
        raise SystemExit(f"No run directories matched {args.runs!r}")
    rows = []
    skipped = []
    for directory in directories:
        config_path = directory / "config.resolved.yaml"
        val_path = directory / "best_val_metrics.json"
        test_path = directory / "test_metrics.json"
        if not config_path.is_file() or not val_path.is_file():
            skipped.append({"run": str(directory), "reason": "missing config or best validation metrics"})
            continue
        if args.require_test and not test_path.is_file():
            skipped.append({"run": str(directory), "reason": "missing explicit test metrics"})
            continue
        config = load_config(config_path)
        row = {
            "run": str(directory),
            "experiment": config.get("experiment"),
            "seed": config.get("seed"),
            "dataset": config["data"]["dataset"],
            "views": len(config["model"]["views"]),
            "patchification": config["model"]["patchification"],
            "fusion_enabled": config["model"]["fusion"].get("enabled", True),
            "fusion_direction": config["model"]["fusion"].get("direction"),
            "pooling": config["model"]["pooling"],
            "global_mode": config["model"]["global"]["mode"],
            "global_depth": config["model"]["global"]["depth"],
            "global_dim": config["model"]["global"]["dim"],
            "patch_tokens": token_accounting(config)["total_patch_tokens"],
            "analytical_parameters": analytical_parameter_count(config)["total"],
        }
        row.update(metric_columns("val", read_json(val_path)))
        row.update(metric_columns("test", read_json(test_path) if test_path.is_file() else None))
        rows.append(row)
    if not rows:
        raise SystemExit(f"No complete runs to aggregate. Skipped: {skipped}")
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["dataset", "experiment", "seed"]).to_csv(output, index=False)
    print(json.dumps({"output": str(output), "runs": len(rows), "skipped": skipped}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
