#!/usr/bin/env python3
"""Create a four-level engagement manifest from an authorized DAiSEE archive."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import pandas as pd


VIDEO_SUFFIXES = {".avi", ".mp4", ".mov", ".mkv"}
SPLIT_ALIASES = {
    "train": ("train",),
    "val": ("validation", "valid", "val"),
    "test": ("test",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Extracted DAiSEE root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--relative-paths", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def split_for_name(name: str) -> str | None:
    normalized = normalize(name)
    for split, aliases in SPLIT_ALIASES.items():
        if any(normalize(alias) in normalized for alias in aliases):
            return split
    return None


def locate_label_files(root: Path) -> dict[str, Path]:
    candidates: dict[str, list[Path]] = defaultdict(list)
    for path in root.rglob("*.csv"):
        split = split_for_name(path.name)
        if split:
            candidates[split].append(path)
    selected = {}
    for split in ("train", "val", "test"):
        matches = sorted(candidates.get(split, []), key=lambda path: (len(path.parts), len(path.name)))
        if not matches:
            raise FileNotFoundError(f"Could not locate a {split} label CSV beneath {root}")
        selected[split] = matches[0]
    return selected


def find_column(columns, alternatives: tuple[str, ...]) -> str:
    by_normalized = {normalize(column): column for column in columns}
    for alternative in alternatives:
        target = normalize(alternative)
        if target in by_normalized:
            return by_normalized[target]
    for normalized, original in by_normalized.items():
        if any(normalize(alternative) in normalized for alternative in alternatives):
            return original
    raise ValueError(f"Could not find any of {alternatives} in columns {list(columns)}")


def build_video_index(root: Path) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    by_name: dict[str, list[Path]] = defaultdict(list)
    by_stem: dict[str, list[Path]] = defaultdict(list)
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
            by_name[path.name.lower()].append(path.resolve())
            by_stem[path.stem.lower()].append(path.resolve())
    if not by_name:
        raise FileNotFoundError(f"No supported videos found beneath {root}")
    return by_name, by_stem


def choose_video(
    clip_value: str,
    split: str,
    by_name: dict[str, list[Path]],
    by_stem: dict[str, list[Path]],
) -> Path | None:
    clip = Path(str(clip_value).strip())
    matches = by_name.get(clip.name.lower(), [])
    if not matches:
        matches = by_stem.get(clip.stem.lower(), [])
    if not matches:
        return None
    split_aliases = SPLIT_ALIASES[split]
    split_matches = [
        path
        for path in matches
        if any(normalize(alias) in normalize("/".join(path.parts)) for alias in split_aliases)
    ]
    pool = split_matches or matches
    if len(pool) > 1:
        raise ValueError(f"Ambiguous video match for {clip_value!r} ({split}): {pool}")
    return pool[0]


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    label_files = locate_label_files(root)
    by_name, by_stem = build_video_index(root)
    rows = []
    missing = []
    for split, label_path in label_files.items():
        labels = pd.read_csv(label_path)
        clip_column = find_column(labels.columns, ("ClipID", "Clip", "Video", "VideoID"))
        engagement_column = find_column(labels.columns, ("Engagement", "Engaged"))
        for _, source in labels.iterrows():
            clip_id = str(source[clip_column]).strip()
            video = choose_video(clip_id, split, by_name, by_stem)
            if video is None:
                missing.append({"split": split, "clip_id": clip_id})
                continue
            label = int(source[engagement_column])
            if not 0 <= label <= 3:
                raise ValueError(f"DAiSEE engagement label must be 0--3: {clip_id} -> {label}")
            subject_id = video.parent.name
            stored_path = str(video.relative_to(output.parent)) if args.relative_paths else str(video)
            rows.append(
                {
                    "path": stored_path,
                    "label": label,
                    "split": split,
                    "sample_id": video.stem,
                    "subject_id": subject_id,
                    "label_name": ["Very Low", "Low", "High", "Very High"][label],
                    "source_label_file": str(label_path),
                }
            )
    if missing and not args.allow_missing:
        preview = missing[:10]
        raise FileNotFoundError(
            f"Could not match {len(missing)} labeled clips. First examples: {preview}. "
            "Use --allow-missing only for a documented partial-data experiment."
        )
    manifest = pd.DataFrame(rows).sort_values(["split", "sample_id"]).reset_index(drop=True)
    duplicates = manifest["sample_id"].duplicated(keep=False)
    if duplicates.any():
        duplicate_ids = manifest.loc[duplicates, "sample_id"].tolist()[:10]
        raise ValueError(f"Duplicate DAiSEE sample IDs detected: {duplicate_ids}")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)
    summary = {
        "output": str(output),
        "samples": len(manifest),
        "missing": len(missing),
        "splits": manifest["split"].value_counts().sort_index().to_dict(),
        "labels_by_split": manifest.groupby(["split", "label"]).size().unstack(fill_value=0).to_dict("index"),
        "label_files": {key: str(value) for key, value in label_files.items()},
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
