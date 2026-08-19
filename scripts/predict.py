#!/usr/bin/env python3
"""Predict a single local video with a trained SEFormer checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import numpy as np
import torch

from seformer.checkpoint import load_checkpoint
from seformer.config import load_config
from seformer.data import (
    VideoTensorTransform,
    probe_frame_count,
    read_selected_frames,
    resize_frames,
)
from seformer.faces import MTCNNFaceCropper, crop_clip
from seformer.model import build_model
from seformer.sampling import sample_frame_indices
from seformer.utils import resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--face-detection", action="store_true")
    parser.add_argument("--attention", action="store_true", help="Summarize pooling weights")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def summarize_attention(attention: dict[str, torch.Tensor], top_k: int = 5) -> dict:
    summary = {}
    for name, tensor in attention.items():
        if not name.startswith("pool_"):
            summary[name] = {"shape": list(tensor.shape)}
            continue
        values = tensor[0].detach().float().cpu().numpy()
        indices = np.argsort(values)[::-1][:top_k]
        summary[name] = {
            "shape": list(tensor.shape),
            "top_indices": indices.tolist(),
            "top_weights": values[indices].tolist(),
        }
    return summary


def main() -> int:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    seed_everything(int(config.get("seed", 42)))
    device = resolve_device(config["training"].get("device", "auto"))
    model = build_model(config).to(device)
    load_checkpoint(args.checkpoint, model=model, map_location=device)
    model.eval()

    path = Path(args.video).expanduser().resolve()
    count = probe_frame_count(path)
    indices = sample_frame_indices(
        count,
        config["data"]["frames"],
        config["data"]["temporal_stride"],
        training=False,
    )
    frames = read_selected_frames(path, indices)
    face_metadata = None
    if args.face_detection:
        cropper = MTCNNFaceCropper(
            tuple(config["data"]["image_size"]), device=str(device)
        )
        frames, face_metadata = crop_clip(frames, cropper)
    else:
        frames = resize_frames(frames, tuple(config["data"]["image_size"]))
    tensor = VideoTensorTransform(config["data"], training=False)(frames).unsqueeze(0).to(device)
    with torch.inference_mode():
        output = model(tensor, return_attention=args.attention)
        logits = output["logits"] if isinstance(output, dict) else output
        probabilities = logits.softmax(dim=-1)[0].float().cpu().numpy()
    predicted = int(probabilities.argmax())
    payload = {
        "video": str(path),
        "sampled_indices": indices,
        "class_order": config["data"]["class_names"],
        "predicted_index": predicted,
        "predicted_class": config["data"]["class_names"][predicted],
        "probabilities": {
            name: float(probabilities[index])
            for index, name in enumerate(config["data"]["class_names"])
        },
        "face_preprocessing": face_metadata,
    }
    if isinstance(output, dict):
        payload["attention"] = summarize_attention(output["attention"])
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
