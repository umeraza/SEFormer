#!/usr/bin/env python3
"""Validate media existence, classes, duplicates, and subject leakage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from seformer.manifest import load_manifest, manifest_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--allow-subject-overlap", action="store_true")
    parser.add_argument("--skip-file-check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = load_manifest(args.manifest)
    errors = []
    allowed_splits = {"train", "val", "test"}
    invalid_splits = sorted(set(frame["split"]) - allowed_splits)
    if invalid_splits:
        errors.append(f"Invalid split values: {invalid_splits}")
    missing_splits = sorted(allowed_splits - set(frame["split"]))
    if missing_splits:
        errors.append(f"Missing splits: {missing_splits}")
    if frame["sample_id"].duplicated().any():
        errors.append(f"Duplicate sample IDs: {int(frame['sample_id'].duplicated().sum())}")
    invalid_labels = frame.loc[(frame["label"] < 0) | (frame["label"] >= args.num_classes)]
    if not invalid_labels.empty:
        errors.append(f"Labels outside [0,{args.num_classes - 1}]: {len(invalid_labels)}")
    observed_labels = set(frame["label"].astype(int))
    absent = sorted(set(range(args.num_classes)) - observed_labels)
    if absent:
        errors.append(f"Classes absent from entire manifest: {absent}")
    missing_paths = []
    if not args.skip_file_check:
        missing_paths = [path for path in frame["path"] if not Path(path).is_file()]
        if missing_paths:
            errors.append(f"Missing media paths: {len(missing_paths)}")
    leakage = {}
    if "subject_id" in frame.columns:
        subject_splits = frame.dropna(subset=["subject_id"]).groupby("subject_id")["split"].agg(
            lambda values: sorted(set(values))
        )
        leakage = {str(subject): splits for subject, splits in subject_splits.items() if len(splits) > 1}
        if leakage and not args.allow_subject_overlap:
            errors.append(f"Subjects present in multiple splits: {len(leakage)}")
    summary = manifest_summary(frame)
    summary.update(
        {
            "manifest": str(Path(args.manifest).resolve()),
            "errors": errors,
            "missing_path_examples": missing_paths[:10],
            "subject_leakage_examples": dict(list(leakage.items())[:10]),
            "class_by_split": frame.groupby(["split", "label"])
            .size()
            .unstack(fill_value=0)
            .to_dict("index"),
        }
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
