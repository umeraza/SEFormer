"""Training and inference loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm


@dataclass
class EpochOutput:
    loss: float
    targets: np.ndarray
    logits: np.ndarray
    sample_ids: list[str]
    paths: list[str]


def _logits(model_output: torch.Tensor | dict[str, Any]) -> torch.Tensor:
    return model_output["logits"] if isinstance(model_output, dict) else model_output


def train_one_epoch(
    *,
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer,
    scaler,
    device: torch.device,
    gradient_accumulation: int,
    grad_clip_norm: float | None,
    amp: bool,
    epoch: int,
) -> EpochOutput:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_samples = 0
    all_targets: list[np.ndarray] = []
    all_logits: list[np.ndarray] = []
    all_sample_ids: list[str] = []
    all_paths: list[str] = []
    progress = tqdm(loader, desc=f"train {epoch:03d}", leave=False)

    for step, batch in enumerate(progress):
        videos = batch["video"].to(device, non_blocking=True)
        targets = batch["label"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            logits = _logits(model(videos))
            raw_loss = criterion(logits, targets)
            loss = raw_loss / gradient_accumulation
        scaler.scale(loss).backward()
        should_step = (step + 1) % gradient_accumulation == 0 or step + 1 == len(loader)
        if should_step:
            if grad_clip_norm is not None and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        batch_size = targets.shape[0]
        total_loss += float(raw_loss.detach()) * batch_size
        total_samples += batch_size
        all_targets.append(targets.detach().cpu().numpy())
        all_logits.append(logits.detach().float().cpu().numpy())
        all_sample_ids.extend(str(value) for value in batch["sample_id"])
        all_paths.extend(str(value) for value in batch["path"])
        progress.set_postfix(loss=f"{total_loss / total_samples:.4f}")

    if total_samples == 0:
        raise RuntimeError("Training loader yielded no samples; check drop_last and batch size")
    return EpochOutput(
        loss=total_loss / total_samples,
        targets=np.concatenate(all_targets),
        logits=np.concatenate(all_logits),
        sample_ids=all_sample_ids,
        paths=all_paths,
    )


@torch.inference_mode()
def predict_loader(
    *,
    model: nn.Module,
    loader,
    criterion: nn.Module | None,
    device: torch.device,
    amp: bool,
    description: str,
) -> EpochOutput:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_targets: list[np.ndarray] = []
    all_logits: list[np.ndarray] = []
    all_sample_ids: list[str] = []
    all_paths: list[str] = []
    for batch in tqdm(loader, desc=description, leave=False):
        videos = batch["video"].to(device, non_blocking=True)
        targets = batch["label"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            logits = _logits(model(videos))
            loss = criterion(logits, targets) if criterion is not None else None
        batch_size = targets.shape[0]
        if loss is not None:
            total_loss += float(loss.detach()) * batch_size
        total_samples += batch_size
        all_targets.append(targets.cpu().numpy())
        all_logits.append(logits.float().cpu().numpy())
        all_sample_ids.extend(str(value) for value in batch["sample_id"])
        all_paths.extend(str(value) for value in batch["path"])
    if total_samples == 0:
        raise RuntimeError("Evaluation loader yielded no samples")
    return EpochOutput(
        loss=total_loss / total_samples if criterion is not None else float("nan"),
        targets=np.concatenate(all_targets),
        logits=np.concatenate(all_logits),
        sample_ids=all_sample_ids,
        paths=all_paths,
    )


def criterion_from_config(config: dict[str, Any], device: torch.device) -> nn.Module:
    weights = config["training"].get("class_weights")
    weight_tensor = None
    if weights is not None:
        if len(weights) != config["data"]["num_classes"]:
            raise ValueError("training.class_weights length must equal data.num_classes")
        weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(
        weight=weight_tensor,
        label_smoothing=float(config["training"].get("label_smoothing", 0.0)),
    )
