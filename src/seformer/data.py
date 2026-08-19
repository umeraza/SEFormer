"""Manifest-driven video datasets and data-loader construction."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .manifest import load_manifest
from .sampling import sample_frame_indices


def probe_frame_count(path: str | Path) -> int:
    source = Path(path)
    if source.suffix.lower() in {".npz", ".npy"}:
        loaded = np.load(source, mmap_mode="r" if source.suffix.lower() == ".npy" else None)
        try:
            array = loaded["frames"] if isinstance(loaded, np.lib.npyio.NpzFile) else loaded
            return int(array.shape[0])
        finally:
            if isinstance(loaded, np.lib.npyio.NpzFile):
                loaded.close()

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to decode video files") from exc
    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {source}")
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if count <= 0:
            raise RuntimeError(f"Video reports an invalid frame count: {source}")
        return count
    finally:
        capture.release()


def read_selected_frames(path: str | Path, indices: list[int]) -> np.ndarray:
    """Decode selected frames as uint8 RGB `[T,H,W,C]`."""
    source = Path(path)
    if not indices:
        raise ValueError("At least one frame index is required")
    if source.suffix.lower() in {".npz", ".npy"}:
        loaded = np.load(source, mmap_mode="r" if source.suffix.lower() == ".npy" else None)
        try:
            array = loaded["frames"] if isinstance(loaded, np.lib.npyio.NpzFile) else loaded
            frames = np.asarray(array[indices])
        finally:
            if isinstance(loaded, np.lib.npyio.NpzFile):
                loaded.close()
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"Expected RGB array [T,H,W,3] in {source}; got {frames.shape}")
        return frames.astype(np.uint8, copy=False)

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to decode video files") from exc
    capture = cv2.VideoCapture(str(source))
    cache: dict[int, np.ndarray] = {}
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {source}")
        for index in sorted(set(indices)):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            success, frame = capture.read()
            if not success or frame is None:
                raise RuntimeError(f"Failed to decode frame {index} from {source}")
            cache[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()
    return np.stack([cache[index] for index in indices], axis=0)


def resize_frames(frames: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    height, width = image_size
    if frames.shape[1:3] == (height, width):
        return frames
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to resize video frames") from exc
    return np.stack(
        [cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR) for frame in frames],
        axis=0,
    )


class VideoTensorTransform:
    """Clip-consistent augmentation, scaling, and normalization."""

    def __init__(self, config: dict[str, Any], training: bool) -> None:
        self.training = training
        self.mean = torch.tensor(config["mean"], dtype=torch.float32).view(3, 1, 1, 1)
        self.std = torch.tensor(config["std"], dtype=torch.float32).view(3, 1, 1, 1)
        augmentation = config.get("augmentation", {})
        self.vertical_flip_p = float(augmentation.get("vertical_flip_p", 0.0))
        self.noise_p = float(augmentation.get("gaussian_noise_p", 0.0))
        self.noise_std = float(augmentation.get("gaussian_noise_std", 0.0))

    def __call__(self, frames: np.ndarray) -> torch.Tensor:
        video = torch.from_numpy(np.ascontiguousarray(frames)).permute(3, 0, 1, 2).float()
        video = video / 255.0
        if self.training and self.vertical_flip_p > 0 and torch.rand(()) < self.vertical_flip_p:
            video = video.flip(dims=(2,))
        if self.training and self.noise_p > 0 and torch.rand(()) < self.noise_p:
            video = (video + torch.randn_like(video) * self.noise_std).clamp_(0.0, 1.0)
        return (video - self.mean) / self.std


class ManifestVideoDataset(Dataset):
    def __init__(self, config: dict[str, Any], split: str) -> None:
        self.config = config
        self.split = split.lower().replace("validation", "val")
        frame = load_manifest(config["manifest"])
        self.rows = frame.loc[frame["split"] == self.split].reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(f"Manifest contains no rows for split {self.split!r}")
        invalid = self.rows.loc[
            (self.rows["label"] < 0) | (self.rows["label"] >= config["num_classes"])
        ]
        if not invalid.empty:
            raise ValueError(
                f"Split {self.split!r} has labels outside [0,{config['num_classes'] - 1}]"
            )
        self.training = self.split == "train"
        self.transform = VideoTensorTransform(config, self.training)
        self.image_size = tuple(config["image_size"])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows.iloc[index]
        path = row["path"]
        count = probe_frame_count(path)
        indices = sample_frame_indices(
            count,
            self.config["frames"],
            self.config["temporal_stride"],
            training=self.training,
        )
        frames = resize_frames(read_selected_frames(path, indices), self.image_size)
        return {
            "video": self.transform(frames),
            "label": int(row["label"]),
            "sample_id": str(row["sample_id"]),
            "path": str(path),
        }


class SyntheticVideoDataset(Dataset):
    """Small deterministic dataset for shape and end-to-end smoke tests only."""

    def __init__(self, config: dict[str, Any], split: str, seed: int) -> None:
        self.config = config
        self.split = split
        base_samples = int(config.get("synthetic_samples", 12))
        self.samples = base_samples if split == "train" else max(4, base_samples // 3)
        self.seed = seed + {"train": 0, "val": 10_000, "test": 20_000}.get(split, 30_000)

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> dict[str, Any]:
        generator = torch.Generator().manual_seed(self.seed + index)
        height, width = self.config["image_size"]
        video = torch.randn(
            3,
            self.config["frames"],
            height,
            width,
            generator=generator,
        )
        return {
            "video": video,
            "label": index % self.config["num_classes"],
            "sample_id": f"synthetic-{self.split}-{index:05d}",
            "path": "synthetic://generated",
        }


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataset(config: dict[str, Any], split: str) -> Dataset:
    data = config["data"]
    if data["dataset"] == "synthetic":
        return SyntheticVideoDataset(data, split, int(config.get("seed", 42)))
    return ManifestVideoDataset(data, split)


def _weighted_sampler(dataset: Dataset, num_classes: int) -> WeightedRandomSampler | None:
    if not isinstance(dataset, ManifestVideoDataset):
        return None
    labels = dataset.rows["label"].to_numpy(dtype=int)
    counts = np.bincount(labels, minlength=num_classes)
    if np.any(counts == 0):
        raise ValueError("Weighted sampling requires at least one training example per class")
    weights = 1.0 / counts[labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
    )


def build_loader(config: dict[str, Any], split: str) -> DataLoader:
    dataset = build_dataset(config, split)
    training = split == "train"
    sampler = None
    if training and config["training"].get("weighted_sampler", False):
        sampler = _weighted_sampler(dataset, config["data"]["num_classes"])
    workers = int(config["data"].get("num_workers", 0))
    batch_size = (
        config["training"]["batch_size"]
        if training
        else config["evaluation"].get("batch_size", config["training"]["batch_size"])
    )
    generator = torch.Generator().manual_seed(int(config.get("seed", 42)))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training and sampler is None,
        sampler=sampler,
        num_workers=workers,
        pin_memory=bool(config["data"].get("pin_memory", True)),
        persistent_workers=bool(config["data"].get("persistent_workers", True)) and workers > 0,
        drop_last=training,
        worker_init_fn=seed_worker,
        generator=generator,
    )

