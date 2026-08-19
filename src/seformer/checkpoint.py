"""Checkpoint creation, restoration, and compatibility checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .utils import atomic_torch_save


def checkpoint_payload(
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    config: dict[str, Any],
    epoch: int,
    best_metric: float,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    source_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    return {
        "format_version": 1,
        "epoch": epoch,
        "best_metric": best_metric,
        "metrics": metrics,
        "config": config,
        "model": source_model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
    }


def save_checkpoint(path: str | Path, **kwargs) -> None:
    atomic_torch_save(checkpoint_payload(**kwargs), path)


def load_checkpoint(
    path: str | Path,
    *,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # Compatibility with older supported patch builds.
        checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Unsupported checkpoint structure: {path}")
    model.load_state_dict(checkpoint["model"], strict=strict)
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint
