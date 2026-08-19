#!/usr/bin/env python3
"""Train SEFormer and select checkpoints using validation data only."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import pandas as pd
import torch

from seformer.checkpoint import load_checkpoint, save_checkpoint
from seformer.config import load_config, save_config
from seformer.data import build_loader
from seformer.engine import criterion_from_config, predict_loader, train_one_epoch
from seformer.metrics import classification_metrics, save_evaluation
from seformer.model import build_model
from seformer.utils import environment_snapshot, json_dump, resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML experiment configuration")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable dotted configuration override",
    )
    parser.add_argument("--resume", help="Resume model, optimizer, scheduler, and scaler")
    return parser.parse_args()


def create_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def main() -> int:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    output = Path(config["output"]["dir"]).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    save_config(config, output / "config.resolved.yaml")
    json_dump(environment_snapshot(REPOSITORY_ROOT), output / "environment.json")

    seed_everything(int(config.get("seed", 42)))
    device = resolve_device(config["training"].get("device", "auto"))
    train_loader = build_loader(config, "train")
    val_loader = build_loader(config, "val")
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(config["training"]["epochs"])),
        eta_min=float(config["training"].get("min_learning_rate", 0.0)),
    )
    amp_enabled = bool(config["training"].get("amp", True)) and device.type == "cuda"
    scaler = create_scaler(amp_enabled)
    criterion = criterion_from_config(config, device)

    start_epoch = 1
    mode = config["training"].get("checkpoint_mode", "max")
    best_metric = -math.inf if mode == "max" else math.inf
    if args.resume:
        state = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
        )
        start_epoch = int(state.get("epoch", 0)) + 1
        best_metric = float(state.get("best_metric", best_metric))

    if bool(config["training"].get("compile", False)):
        if not hasattr(torch, "compile"):
            raise RuntimeError("training.compile=true requires torch.compile")
        model = torch.compile(model)

    history: list[dict] = []
    patience = int(config["training"].get("early_stopping_patience", 0))
    epochs_without_improvement = 0
    average = config["evaluation"]["metric_average"]
    class_names = config["data"]["class_names"]
    num_classes = config["data"]["num_classes"]
    selected_name = config["training"].get("checkpoint_metric", f"{average}_f1")

    for epoch in range(start_epoch, int(config["training"]["epochs"]) + 1):
        train_output = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            gradient_accumulation=int(config["training"]["gradient_accumulation"]),
            grad_clip_norm=config["training"].get("grad_clip_norm"),
            amp=amp_enabled,
            epoch=epoch,
        )
        val_output = predict_loader(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            amp=amp_enabled,
            description=f"val {epoch:03d}",
        )
        train_predictions = train_output.logits.argmax(axis=1)
        val_predictions = val_output.logits.argmax(axis=1)
        train_metrics = classification_metrics(
            train_output.targets,
            train_predictions,
            num_classes=num_classes,
            class_names=class_names,
            average=average,
        )
        val_metrics = classification_metrics(
            val_output.targets,
            val_predictions,
            num_classes=num_classes,
            class_names=class_names,
            average=average,
        )
        train_metrics["loss"] = train_output.loss
        val_metrics["loss"] = val_output.loss
        selected = float(val_metrics[selected_name])
        improved = selected > best_metric if mode == "max" else selected < best_metric
        if improved:
            best_metric = selected
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        scheduler.step()
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_output.loss,
            "train_accuracy": train_metrics["accuracy"],
            f"train_{average}_f1": train_metrics[f"{average}_f1"],
            "train_paper_f1": train_metrics["paper_f1"],
            "val_loss": val_output.loss,
            "val_accuracy": val_metrics["accuracy"],
            f"val_{average}_precision": val_metrics[f"{average}_precision"],
            f"val_{average}_recall": val_metrics[f"{average}_recall"],
            f"val_{average}_f1": val_metrics[f"{average}_f1"],
            "val_paper_f1": val_metrics["paper_f1"],
            "best_metric": best_metric,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        print(
            f"epoch={epoch:03d} train_loss={train_output.loss:.4f} "
            f"val_loss={val_output.loss:.4f} val_accuracy={val_metrics['accuracy']:.4f} "
            f"val_{selected_name}={selected:.4f}"
        )

        checkpoint_kwargs = {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scaler": scaler,
            "config": config,
            "epoch": epoch,
            "best_metric": best_metric,
            "metrics": {"train": train_metrics, "val": val_metrics},
        }
        if config["output"].get("save_last", True):
            save_checkpoint(output / "last.pt", **checkpoint_kwargs)
        if improved and config["output"].get("save_best", True):
            save_checkpoint(output / "best.pt", **checkpoint_kwargs)
            save_evaluation(
                output_dir=output,
                prefix="best_val",
                targets=val_output.targets,
                logits=val_output.logits,
                sample_ids=val_output.sample_ids,
                paths=val_output.paths,
                class_names=class_names,
                average=average,
                loss=val_output.loss,
                bootstrap_samples=0,
                seed=int(config.get("seed", 42)),
            )
        json_dump(
            {"epoch": epoch, "best_metric": best_metric, "train": train_metrics, "val": val_metrics},
            output / "latest_metrics.json",
        )
        if patience > 0 and epochs_without_improvement >= patience:
            print(f"Early stopping after {epochs_without_improvement} unimproved epochs")
            break

    print(f"Training complete. Best validation {selected_name}: {best_metric:.6f}")
    print("The test split was not evaluated. Run scripts/evaluate.py explicitly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
