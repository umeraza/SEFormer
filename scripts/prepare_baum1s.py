#!/usr/bin/env python3
"""Create the six-basic-emotion BAUM-1s manifest from local annotations."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import pandas as pd


CLASS_NAMES = ["Joy", "Fear", "Anger", "Surprise", "Sadness", "Disgust"]
ALIASES = {
    "joy": "Joy",
    "happy": "Joy",
    "happiness": "Joy",
    "fear": "Fear",
    "afraid": "Fear",
    "anger": "Anger",
    "angry": "Anger",
    "surprise": "Surprise",
    "surprised": "Surprise",
    "sad": "Sadness",
    "sadness": "Sadness",
    "disgust": "Disgust",
    "disgusted": "Disgust",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--path-column", default="path")
    parser.add_argument("--label-column", default="emotion")
    parser.add_argument("--subject-column", default="subject_id")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--sample-id-column")
    parser.add_argument("--subject-regex", default=r"(?i)(?:subject|sub|s)[_-]?(\d+)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--relative-paths", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def normalize_label(value: str) -> str | None:
    normalized = re.sub(r"[^a-z]+", "", str(value).lower())
    return ALIASES.get(normalized)


def infer_subject(path: str, pattern: re.Pattern) -> str:
    match = pattern.search(path)
    if not match:
        raise ValueError(
            f"Could not infer subject from {path!r}; provide --subject-column or --subject-regex"
        )
    return match.group(1)


def assign_subject_splits(subjects: list[str], seed: int, train_ratio: float, val_ratio: float):
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("train/val ratios must be positive and sum to less than 1")
    unique = sorted(set(subjects))
    if len(unique) < 3:
        raise ValueError("A generated subject-disjoint split requires at least three subjects")
    random.Random(seed).shuffle(unique)
    train_count = max(1, round(len(unique) * train_ratio))
    val_count = max(1, round(len(unique) * val_ratio))
    if train_count + val_count >= len(unique):
        train_count = len(unique) - 2
        val_count = 1
    mapping = {}
    for subject in unique[:train_count]:
        mapping[subject] = "train"
    for subject in unique[train_count : train_count + val_count]:
        mapping[subject] = "val"
    for subject in unique[train_count + val_count :]:
        mapping[subject] = "test"
    return mapping


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    annotations_path = Path(args.annotations).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    source = pd.read_csv(annotations_path)
    for column in (args.path_column, args.label_column):
        if column not in source.columns:
            raise ValueError(f"Annotation column {column!r} not found")

    rows = []
    missing = []
    subject_pattern = re.compile(args.subject_regex)
    has_subject_column = args.subject_column in source.columns
    for _, item in source.iterrows():
        label_name = normalize_label(item[args.label_column])
        if label_name is None:
            continue
        raw_path = Path(str(item[args.path_column])).expanduser()
        path = raw_path if raw_path.is_absolute() else root / raw_path
        path = path.resolve()
        if not path.is_file():
            missing.append(str(path))
            continue
        subject = (
            str(item[args.subject_column])
            if has_subject_column and pd.notna(item[args.subject_column])
            else infer_subject(str(raw_path), subject_pattern)
        )
        sample_id = (
            str(item[args.sample_id_column])
            if args.sample_id_column and args.sample_id_column in source.columns
            else path.stem
        )
        rows.append(
            {
                "path": str(path.relative_to(output.parent)) if args.relative_paths else str(path),
                "label": CLASS_NAMES.index(label_name),
                "sample_id": sample_id,
                "subject_id": subject,
                "label_name": label_name,
            }
        )
    if missing and not args.allow_missing:
        raise FileNotFoundError(
            f"{len(missing)} annotated files are missing. First examples: {missing[:10]}"
        )
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise ValueError("No six-basic-emotion records remained after filtering")
    split_generated = args.split_column not in source.columns
    if not split_generated:
        # Map split by sample ID/path from the original annotation table.
        split_lookup = {}
        for _, item in source.iterrows():
            raw_path = Path(str(item[args.path_column]))
            sample_id = (
                str(item[args.sample_id_column])
                if args.sample_id_column and args.sample_id_column in source.columns
                else raw_path.stem
            )
            split = str(item[args.split_column]).lower().strip()
            split_lookup[sample_id] = {"validation": "val", "valid": "val"}.get(split, split)
        manifest["split"] = manifest["sample_id"].map(split_lookup)
        invalid = sorted(set(manifest["split"]) - {"train", "val", "test"})
        if invalid:
            raise ValueError(f"Unrecognized split values: {invalid}")
    else:
        mapping = assign_subject_splits(
            manifest["subject_id"].astype(str).tolist(),
            args.seed,
            args.train_ratio,
            args.val_ratio,
        )
        manifest["split"] = manifest["subject_id"].astype(str).map(mapping)
    if manifest["sample_id"].duplicated().any():
        duplicates = manifest.loc[manifest["sample_id"].duplicated(False), "sample_id"].tolist()[:10]
        raise ValueError(f"Duplicate sample IDs: {duplicates}")
    manifest = manifest.sort_values(["split", "subject_id", "sample_id"]).reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)
    summary = {
        "output": str(output),
        "samples": len(manifest),
        "missing": len(missing),
        "class_order": CLASS_NAMES,
        "split_generated": split_generated,
        "split_warning": (
            "Generated subject-disjoint split is exploratory and is not the unspecified paper split."
            if split_generated
            else None
        ),
        "subjects_by_split": manifest.groupby("split")["subject_id"].nunique().to_dict(),
        "labels_by_split": manifest.groupby(["split", "label_name"])
        .size()
        .unstack(fill_value=0)
        .to_dict("index"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
