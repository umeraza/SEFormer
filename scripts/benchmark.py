#!/usr/bin/env python3
"""Report token and parameter accounting; optionally time a forward pass."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from seformer.analysis import analytical_parameter_count, token_accounting
from seformer.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--forward", action="store_true", help="Run potentially expensive timing")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    payload = {
        "tokens": token_accounting(config),
        "analytical_parameters": analytical_parameter_count(config),
    }
    try:
        import torch

        from seformer.model import build_model
        from seformer.utils import count_trainable_parameters, resolve_device

        model = build_model(config)
        payload["exact_trainable_parameters"] = count_trainable_parameters(model)
        if args.forward:
            device = resolve_device(args.device)
            model = model.to(device).eval()
            height, width = config["data"]["image_size"]
            sample = torch.randn(
                1,
                config["model"]["in_channels"],
                config["data"]["frames"],
                height,
                width,
                device=device,
            )
            with torch.inference_mode():
                for _ in range(args.warmup):
                    model(sample)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats(device)
                times = []
                for _ in range(args.repeats):
                    start = time.perf_counter()
                    model(sample)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    times.append(1000 * (time.perf_counter() - start))
            payload["forward"] = {
                "device": str(device),
                "batch_size": 1,
                "mean_ms": statistics.mean(times),
                "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
                "repeats": args.repeats,
            }
            if device.type == "cuda":
                payload["forward"]["peak_memory_bytes"] = torch.cuda.max_memory_allocated(device)
    except ImportError as exc:
        payload["exact_trainable_parameters"] = None
        payload["pytorch_unavailable"] = str(exc)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
