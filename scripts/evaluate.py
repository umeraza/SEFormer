#!/usr/bin/env python3
"""Evaluate a trained SEFormer checkpoint on one explicit split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from seformer.checkpoint import load_checkpoint
from seformer.config import load_config
from seformer.data import build_loader
from seformer.engine import criterion_from_config, predict_loader
from seformer.metrics import save_evaluation
from seformer.model import build_model
from seformer.utils import resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output-dir")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    seed_everything(int(config.get("seed", 42)))
    device = resolve_device(config["training"].get("device", "auto"))
    model = build_model(config).to(device)
    load_checkpoint(args.checkpoint, model=model, map_location=device)
    loader = build_loader(config, args.split)
    criterion = criterion_from_config(config, device)
    output = predict_loader(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        amp=bool(config["training"].get("amp", True)),
        description=args.split,
    )
    destination = Path(args.output_dir or Path(args.checkpoint).resolve().parent)
    metrics = save_evaluation(
        output_dir=destination,
        prefix=args.split,
        targets=output.targets,
        logits=output.logits,
        sample_ids=output.sample_ids,
        paths=output.paths,
        class_names=config["data"]["class_names"],
        average=config["evaluation"]["metric_average"],
        loss=output.loss,
        bootstrap_samples=int(config["evaluation"].get("bootstrap_samples", 0)),
        seed=int(config.get("seed", 42)),
    )
    headline = {key: value for key, value in metrics.items() if isinstance(value, (int, float, str))}
    print(json.dumps(headline, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
