"""MTCNN face localization and deterministic crop fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FaceCropResult:
    frame: np.ndarray
    detected: bool
    box: tuple[int, int, int, int]
    probability: float | None


class MTCNNFaceCropper:
    """Select the highest-confidence face and preserve temporal crop continuity."""

    def __init__(
        self,
        image_size: tuple[int, int] = (112, 112),
        *,
        device: str = "cpu",
        margin_fraction: float = 0.2,
        min_probability: float = 0.90,
    ) -> None:
        try:
            from facenet_pytorch import MTCNN
        except ImportError as exc:
            raise RuntimeError(
                "Face preprocessing requires the optional dependency: pip install '.[face]'"
            ) from exc
        self.detector = MTCNN(keep_all=True, device=device, post_process=False)
        self.image_size = image_size
        self.margin_fraction = margin_fraction
        self.min_probability = min_probability
        self.last_box: tuple[int, int, int, int] | None = None

    @staticmethod
    def _center_box(frame: np.ndarray) -> tuple[int, int, int, int]:
        height, width = frame.shape[:2]
        side = min(height, width)
        left = (width - side) // 2
        top = (height - side) // 2
        return left, top, left + side, top + side

    def _square_and_clip(
        self, box: np.ndarray | tuple[float, float, float, float], frame: np.ndarray
    ) -> tuple[int, int, int, int]:
        height, width = frame.shape[:2]
        left, top, right, bottom = (float(value) for value in box)
        center_x, center_y = (left + right) / 2.0, (top + bottom) / 2.0
        side = max(right - left, bottom - top) * (1.0 + self.margin_fraction)
        side = min(side, float(min(height, width)))
        left = max(0.0, min(center_x - side / 2.0, width - side))
        top = max(0.0, min(center_y - side / 2.0, height - side))
        right, bottom = left + side, top + side
        return int(round(left)), int(round(top)), int(round(right)), int(round(bottom))

    def _detect(self, frame: np.ndarray) -> tuple[tuple[int, int, int, int] | None, float | None]:
        from PIL import Image

        boxes, probabilities = self.detector.detect(Image.fromarray(frame))
        if boxes is None or probabilities is None or len(boxes) == 0:
            return None, None
        valid = np.where(np.asarray(probabilities) >= self.min_probability)[0]
        if len(valid) == 0:
            return None, float(np.nanmax(probabilities))
        selected = int(valid[np.argmax(np.asarray(probabilities)[valid])])
        return self._square_and_clip(boxes[selected], frame), float(probabilities[selected])

    def __call__(self, frame: np.ndarray) -> FaceCropResult:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected RGB HxWx3 frame, observed {frame.shape}")
        box, probability = self._detect(frame)
        detected = box is not None
        if box is None:
            box = self.last_box or self._center_box(frame)
        else:
            self.last_box = box
        left, top, right, bottom = box
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            box = self._center_box(frame)
            left, top, right, bottom = box
            crop = frame[top:bottom, left:right]
            detected = False
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for face-crop resizing") from exc
        target_h, target_w = self.image_size
        resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return FaceCropResult(resized, detected, box, probability)

    def reset(self) -> None:
        self.last_box = None


def crop_clip(frames: np.ndarray, cropper: MTCNNFaceCropper) -> tuple[np.ndarray, dict[str, Any]]:
    cropper.reset()
    outputs: list[np.ndarray] = []
    detected = 0
    probabilities: list[float] = []
    for frame in frames:
        result = cropper(frame)
        outputs.append(result.frame)
        detected += int(result.detected)
        if result.probability is not None:
            probabilities.append(result.probability)
    metadata = {
        "frames": len(outputs),
        "detected_frames": detected,
        "detection_rate": detected / len(outputs) if outputs else 0.0,
        "mean_detection_probability": float(np.mean(probabilities)) if probabilities else None,
    }
    return np.stack(outputs, axis=0), metadata
