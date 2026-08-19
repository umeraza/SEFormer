"""Deterministic temporal sampling helpers without framework dependencies."""

from __future__ import annotations

import random


def sample_frame_indices(
    total_frames: int,
    frames: int,
    stride: int,
    *,
    training: bool,
    rng: random.Random | None = None,
) -> list[int]:
    """Sample a contiguous strided clip, repeating the final frame if needed."""
    if total_frames <= 0:
        raise ValueError("Cannot sample a video with zero frames")
    if frames <= 0 or stride <= 0:
        raise ValueError("frames and stride must be positive")
    span = (frames - 1) * stride + 1
    max_start = max(total_frames - span, 0)
    generator = rng or random
    if training and max_start > 0:
        start = generator.randint(0, max_start)
    else:
        start = max_start // 2
    return [min(start + index * stride, total_frames - 1) for index in range(frames)]


def temporal_coverage(frames: int, stride: int) -> int:
    if frames <= 0 or stride <= 0:
        raise ValueError("frames and stride must be positive")
    return (frames - 1) * stride + 1
