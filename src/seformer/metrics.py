"""Multiclass metrics, confidence intervals, and evaluation artifacts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .utils import json_dump


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def classification_metrics(
    targets: list[int] | np.ndarray,
    predictions: list[int] | np.ndarray,
    *,
    num_classes: int,
    class_names: list[str],
    average: str = "macro",
) -> dict[str, Any]:
    targets_array = np.asarray(targets, dtype=np.int64)
    predictions_array = np.asarray(predictions, dtype=np.int64)
    if targets_array.shape != predictions_array.shape:
        raise ValueError("targets and predictions must have identical shape")
    if targets_array.ndim != 1 or targets_array.size == 0:
        raise ValueError("targets and predictions must be non-empty 1D arrays")
    labels = np.arange(num_classes)
    counts = confusion_matrix(targets_array, predictions_array, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        targets_array,
        predictions_array,
        labels=labels,
        average=None,
        zero_division=0,
    )
    avg_precision, avg_recall, avg_f1, _ = precision_recall_fscore_support(
        targets_array,
        predictions_array,
        labels=labels,
        average=average,
        zero_division=0,
    )
    accuracy = float(np.mean(targets_array == predictions_array))
    paper_f1 = (
        float(2 * avg_precision * avg_recall / (avg_precision + avg_recall))
        if avg_precision + avg_recall > 0
        else 0.0
    )
    supported = support > 0
    balanced_accuracy = float(recall[supported].mean()) if np.any(supported) else 0.0
    normalized = np.divide(
        counts,
        counts.sum(axis=1, keepdims=True),
        out=np.zeros_like(counts, dtype=float),
        where=counts.sum(axis=1, keepdims=True) != 0,
    )
    return {
        "num_samples": int(targets_array.size),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        f"{average}_precision": float(avg_precision),
        f"{average}_recall": float(avg_recall),
        f"{average}_f1": float(avg_f1),
        "paper_f1": paper_f1,
        "paper_f1_definition": f"harmonic mean of {average} precision and {average} recall",
        "average": average,
        "per_class": [
            {
                "index": index,
                "name": class_names[index],
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in range(num_classes)
        ],
        "confusion_counts": counts.tolist(),
        "confusion_normalized": normalized.tolist(),
    }


def bootstrap_intervals(
    targets: np.ndarray,
    predictions: np.ndarray,
    *,
    num_classes: int,
    class_names: list[str],
    average: str,
    samples: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, list[float]]:
    if samples <= 0:
        return {}
    rng = np.random.default_rng(seed)
    names = [
        "accuracy",
        f"{average}_precision",
        f"{average}_recall",
        f"{average}_f1",
        "paper_f1",
    ]
    observations = {name: [] for name in names}
    for _ in range(samples):
        indices = rng.integers(0, len(targets), size=len(targets))
        result = classification_metrics(
            targets[indices],
            predictions[indices],
            num_classes=num_classes,
            class_names=class_names,
            average=average,
        )
        for name in names:
            observations[name].append(result[name])
    alpha = (1.0 - confidence) / 2.0
    return {
        name: [
            float(np.quantile(values, alpha)),
            float(np.quantile(values, 1.0 - alpha)),
        ]
        for name, values in observations.items()
    }


def _safe_column(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def save_evaluation(
    *,
    output_dir: str | Path,
    prefix: str,
    targets: np.ndarray,
    logits: np.ndarray,
    sample_ids: list[str],
    paths: list[str],
    class_names: list[str],
    average: str,
    loss: float | None = None,
    bootstrap_samples: int = 0,
    seed: int = 42,
) -> dict[str, Any]:
    import pandas as pd

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    probabilities = softmax_numpy(logits)
    predictions = probabilities.argmax(axis=1)
    result = classification_metrics(
        targets,
        predictions,
        num_classes=len(class_names),
        class_names=class_names,
        average=average,
    )
    if loss is not None:
        result["loss"] = float(loss)
    intervals = bootstrap_intervals(
        targets,
        predictions,
        num_classes=len(class_names),
        class_names=class_names,
        average=average,
        samples=bootstrap_samples,
        seed=seed,
    )
    if intervals:
        result["bootstrap_95_percent_intervals"] = intervals
        result["bootstrap_samples"] = bootstrap_samples

    prediction_frame: dict[str, Any] = {
        "sample_id": sample_ids,
        "path": paths,
        "target": targets,
        "prediction": predictions,
        "target_name": [class_names[int(index)] for index in targets],
        "prediction_name": [class_names[int(index)] for index in predictions],
        "confidence": probabilities.max(axis=1),
    }
    for index, name in enumerate(class_names):
        prediction_frame[f"probability_{_safe_column(name)}"] = probabilities[:, index]
    pd.DataFrame(prediction_frame).to_csv(output / f"{prefix}_predictions.csv", index=False)

    counts = np.asarray(result["confusion_counts"])
    normalized = np.asarray(result["confusion_normalized"])
    pd.DataFrame(counts, index=class_names, columns=class_names).to_csv(
        output / f"{prefix}_confusion_counts.csv", index_label="actual"
    )
    pd.DataFrame(normalized, index=class_names, columns=class_names).to_csv(
        output / f"{prefix}_confusion_normalized.csv", index_label="actual"
    )
    _save_confusion_figure(normalized, class_names, output / f"{prefix}_confusion_normalized.png")
    json_dump(result, output / f"{prefix}_metrics.json")
    return result


def _save_confusion_figure(matrix: np.ndarray, class_names: list[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    size = max(5.0, 0.9 * len(class_names))
    figure, axis = plt.subplots(figsize=(size, size * 0.9))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Row-normalized rate")
    ticks = np.arange(len(class_names))
    axis.set(
        xticks=ticks,
        yticks=ticks,
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted class",
        ylabel="True class",
        title="Row-normalized confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    threshold = 0.5
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                f"{100 * value:.1f}%",
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=8,
            )
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
