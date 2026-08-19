#!/usr/bin/env python3
"""Extract MTCNN-centered face clips to compressed NPZ and write a new manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import numpy as np
import pandas as pd

from seformer.faces import MTCNNFaceCropper
from seformer.manifest import load_manifest


def progress(iterable, **kwargs):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--height", type=int, default=112)
    parser.add_argument("--width", type=int, default=112)
    parser.add_argument("--margin-fraction", type=float, default=0.2)
    parser.add_argument("--min-probability", type=float, default=0.90)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def safe_name(sample_id: str, source_path: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id).strip("._") or "sample"
    digest = hashlib.sha1(source_path.encode("utf-8")).hexdigest()[:10]
    return f"{slug}_{digest}.npz"


def iter_rgb_frames(path: Path):
    if path.suffix.lower() in {".npz", ".npy"}:
        loaded = np.load(path, mmap_mode="r" if path.suffix.lower() == ".npy" else None)
        try:
            array = loaded["frames"] if isinstance(loaded, np.lib.npyio.NpzFile) else loaded
            for frame in array:
                yield np.asarray(frame, dtype=np.uint8)
        finally:
            if isinstance(loaded, np.lib.npyio.NpzFile):
                loaded.close()
        return
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for video preprocessing") from exc
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open {path}")
        while True:
            success, frame = capture.read()
            if not success:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def process_one(path: Path, destination: Path, cropper: MTCNNFaceCropper) -> dict:
    cropper.reset()
    cropped = []
    detected = 0
    probabilities = []
    for frame in iter_rgb_frames(path):
        result = cropper(frame)
        cropped.append(result.frame)
        detected += int(result.detected)
        if result.probability is not None:
            probabilities.append(result.probability)
    if not cropped:
        raise RuntimeError(f"Decoded zero frames from {path}")
    frames = np.stack(cropped, axis=0).astype(np.uint8, copy=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, frames=frames)
    return {
        "source_frames": len(cropped),
        "face_detected_frames": detected,
        "face_detection_rate": detected / len(cropped),
        "mean_face_probability": float(np.mean(probabilities)) if probabilities else None,
    }


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    if args.limit is not None:
        manifest = manifest.iloc[: args.limit].copy()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_manifest = Path(args.output_manifest).expanduser().resolve()
    cropper = MTCNNFaceCropper(
        (args.height, args.width),
        device=args.device,
        margin_fraction=args.margin_fraction,
        min_probability=args.min_probability,
    )
    records = []
    errors = []
    for _, row in progress(manifest.iterrows(), total=len(manifest), desc="face crops"):
        source = Path(row["path"])
        destination = output_dir / str(row["split"]) / safe_name(str(row["sample_id"]), str(source))
        try:
            if destination.exists() and not args.overwrite:
                loaded = np.load(destination)
                try:
                    frame_count = int(loaded["frames"].shape[0])
                finally:
                    loaded.close()
                metadata = {
                    "source_frames": frame_count,
                    "face_detected_frames": None,
                    "face_detection_rate": None,
                    "mean_face_probability": None,
                }
            else:
                metadata = process_one(source, destination, cropper)
            record = row.to_dict()
            record["source_path"] = str(source)
            record["path"] = str(destination)
            record.update(metadata)
            records.append(record)
        except Exception as exc:
            errors.append({"sample_id": str(row["sample_id"]), "path": str(source), "error": str(exc)})
            if not args.continue_on_error:
                raise
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_manifest, index=False)
    error_path = output_manifest.with_suffix(".errors.json")
    with error_path.open("w", encoding="utf-8") as handle:
        json.dump(errors, handle, indent=2)
        handle.write("\n")
    summary = {
        "output_manifest": str(output_manifest),
        "processed": len(records),
        "errors": len(errors),
        "error_log": str(error_path),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
