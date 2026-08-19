"""Framework-independent manifest loading and summary helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_MANIFEST_COLUMNS = {"path", "label", "split", "sample_id"}


def load_manifest(path: str | Path) -> pd.DataFrame:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    frame = pd.read_csv(manifest_path)
    missing = REQUIRED_MANIFEST_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    frame = frame.copy()
    frame["split"] = frame["split"].astype(str).str.lower().replace({"validation": "val"})
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    frame["sample_id"] = frame["sample_id"].astype(str)
    base = manifest_path.parent

    def resolve_media(value: Any) -> str:
        candidate = Path(str(value)).expanduser()
        if not candidate.is_absolute():
            candidate = base / candidate
        return str(candidate.resolve())

    frame["path"] = frame["path"].map(resolve_media)
    return frame


def manifest_summary(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {
        "samples": int(len(frame)),
        "splits": frame["split"].value_counts().sort_index().astype(int).to_dict(),
        "labels": frame["label"].value_counts().sort_index().astype(int).to_dict(),
        "duplicate_sample_ids": int(frame["sample_id"].duplicated().sum()),
    }
    if "subject_id" in frame.columns:
        subjects = frame["subject_id"].dropna().astype(str)
        result["subjects"] = int(subjects.nunique())
    return result
