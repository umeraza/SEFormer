#!/usr/bin/env python3
"""Run a glob-selected ablation matrix with explicit subprocess isolation."""

from __future__ import annotations

import argparse
import glob
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", required=True, help="Quoted glob, e.g. configs/ablations/*.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matches = sorted(Path(path).resolve() for path in glob.glob(args.configs))
    if not matches:
        raise SystemExit(f"No configuration matched {args.configs!r}")
    failures = []
    for config in matches:
        command = [sys.executable, str(REPOSITORY_ROOT / "scripts" / "train.py"), "--config", str(config)]
        for override in args.overrides:
            command.extend(("--set", override))
        print(" ".join(command), flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(command, cwd=REPOSITORY_ROOT, check=False)
        if result.returncode:
            failures.append((config, result.returncode))
            if not args.continue_on_error:
                return result.returncode
    if failures:
        print("Failed runs:")
        for config, code in failures:
            print(f"  {config}: exit {code}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
