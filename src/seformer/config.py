"""Configuration loading, inheritance, overrides, and validation."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a configuration is incomplete or internally inconsistent."""


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries; lists and scalar values are replaced."""
    result = copy.deepcopy(base)
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_recursive(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if resolved in stack:
        chain = " -> ".join(str(item) for item in (*stack, resolved))
        raise ConfigError(f"Circular _base_ configuration chain: {chain}")
    if not resolved.is_file():
        raise FileNotFoundError(f"Configuration not found: {resolved}")

    with resolved.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"Top level of {resolved} must be a mapping")

    base_ref = raw.pop("_base_", None)
    if base_ref is None:
        return raw
    refs = [base_ref] if isinstance(base_ref, str) else list(base_ref)
    merged: dict[str, Any] = {}
    for ref in refs:
        if not isinstance(ref, str):
            raise ConfigError("Every _base_ entry must be a path string")
        merged = deep_merge(merged, _load_recursive(resolved.parent / ref, (*stack, resolved)))
    return deep_merge(merged, raw)


def _set_dotted(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    if any(not key for key in keys):
        raise ConfigError(f"Invalid override key: {dotted_key!r}")
    node: dict[str, Any] = config
    for key in keys[:-1]:
        existing = node.get(key)
        if existing is None:
            node[key] = {}
        elif not isinstance(existing, dict):
            raise ConfigError(f"Cannot descend through non-mapping override key {key!r}")
        node = node[key]
    node[keys[-1]] = value


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    result = copy.deepcopy(config)
    for expression in overrides or []:
        if "=" not in expression:
            raise ConfigError(f"Override must use dotted.key=value syntax: {expression!r}")
        key, raw_value = expression.split("=", 1)
        value = yaml.safe_load(raw_value)
        _set_dotted(result, key.strip(), value)
    return result


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    config = apply_overrides(_load_recursive(Path(path), ()), overrides)
    validate_config(config)
    return config


def save_config(config: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{name} must be a positive integer; received {value!r}")
    return value


def validate_config(config: dict[str, Any]) -> None:
    for section in ("data", "model", "training", "evaluation", "output"):
        if section not in config or not isinstance(config[section], dict):
            raise ConfigError(f"Missing mapping section: {section}")

    data = config["data"]
    frames = _positive_int(data.get("frames"), "data.frames")
    stride = _positive_int(data.get("temporal_stride"), "data.temporal_stride")
    if stride < 1:
        raise ConfigError("data.temporal_stride must be at least 1")
    image_size = data.get("image_size")
    if not isinstance(image_size, list) or len(image_size) != 2:
        raise ConfigError("data.image_size must be [height, width]")
    height, width = (_positive_int(v, "data.image_size") for v in image_size)
    num_classes = _positive_int(data.get("num_classes"), "data.num_classes")
    class_names = data.get("class_names")
    if not isinstance(class_names, list) or len(class_names) != num_classes:
        raise ConfigError("data.class_names length must equal data.num_classes")
    if len(set(str(name) for name in class_names)) != num_classes:
        raise ConfigError("data.class_names must be unique")

    model = config["model"]
    if model.get("patchification") not in {"2d", "3d"}:
        raise ConfigError("model.patchification must be '2d' or '3d'")
    views = model.get("views")
    if not isinstance(views, list) or not views:
        raise ConfigError("model.views must be a non-empty list")
    names: set[str] = set()
    dims: list[int] = []
    for index, view in enumerate(views):
        if not isinstance(view, dict):
            raise ConfigError(f"model.views[{index}] must be a mapping")
        name = str(view.get("name", ""))
        if not name or name in names:
            raise ConfigError("Every model view must have a unique non-empty name")
        names.add(name)
        patch = view.get("patch_size")
        if not isinstance(patch, list) or len(patch) != 3:
            raise ConfigError(f"model.views[{index}].patch_size must be [t,h,w]")
        pt, ph, pw = (_positive_int(v, f"view {name} patch") for v in patch)
        if pt > frames or ph > height or pw > width:
            raise ConfigError(
                f"View {name!r} patch {patch} exceeds input {(frames, height, width)}"
            )
        dim = _positive_int(view.get("embed_dim"), f"view {name} embed_dim")
        heads = _positive_int(view.get("heads"), f"view {name} heads")
        _positive_int(view.get("depth"), f"view {name} depth")
        _positive_int(view.get("mlp_dim"), f"view {name} mlp_dim")
        if dim % heads:
            raise ConfigError(f"View {name!r}: embed_dim {dim} is not divisible by heads {heads}")
        dims.append(dim)

    pooling = model.get("pooling")
    if pooling not in {"sequence", "cls"}:
        raise ConfigError("model.pooling must be 'sequence' or 'cls'")
    if pooling == "cls" and not bool(model.get("use_cls_token", True)):
        raise ConfigError("CLS pooling requires model.use_cls_token=true")

    fusion = model.get("fusion", {})
    direction = fusion.get("direction", "coarse_to_fine")
    if direction not in {"coarse_to_fine", "bidirectional"}:
        raise ConfigError("model.fusion.direction must be coarse_to_fine or bidirectional")
    fusion_heads = _positive_int(fusion.get("heads", 1), "model.fusion.heads")
    if bool(fusion.get("enabled", True)) and len(views) > 1:
        for dim in dims:
            if dim % fusion_heads:
                raise ConfigError(
                    f"View dimension {dim} is not divisible by fusion heads {fusion_heads}"
                )

    global_cfg = model.get("global")
    if not isinstance(global_cfg, dict):
        raise ConfigError("model.global must be a mapping")
    if global_cfg.get("mode") not in {"transformer", "mlp"}:
        raise ConfigError("model.global.mode must be transformer or mlp")
    global_dim = _positive_int(global_cfg.get("dim"), "model.global.dim")
    global_heads = _positive_int(global_cfg.get("heads"), "model.global.heads")
    depth = global_cfg.get("depth")
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
        raise ConfigError("model.global.depth must be a non-negative integer")
    if global_dim % global_heads:
        raise ConfigError("model.global.dim must be divisible by model.global.heads")
    _positive_int(global_cfg.get("mlp_dim"), "model.global.mlp_dim")

    training = config["training"]
    _positive_int(training.get("epochs"), "training.epochs")
    _positive_int(training.get("batch_size"), "training.batch_size")
    _positive_int(training.get("gradient_accumulation"), "training.gradient_accumulation")
    if training.get("optimizer") != "adamw":
        raise ConfigError("Only the paper-specified AdamW optimizer is supported")
    if training.get("scheduler") != "cosine":
        raise ConfigError("Only the paper-specified cosine scheduler is supported")
    for key in ("learning_rate", "weight_decay"):
        value = training.get(key)
        if not isinstance(value, (int, float)) or value < 0:
            raise ConfigError(f"training.{key} must be non-negative")
    smoothing = training.get("label_smoothing", 0.0)
    if not isinstance(smoothing, (int, float)) or not 0 <= smoothing < 1:
        raise ConfigError("training.label_smoothing must lie in [0,1)")

    average = config["evaluation"].get("metric_average", "macro")
    if average not in {"macro", "weighted"}:
        raise ConfigError("evaluation.metric_average must be macro or weighted")


def config_summary(config: dict[str, Any]) -> dict[str, Any]:
    """Return a compact, JSON-serializable experiment summary."""
    return {
        "experiment": config.get("experiment"),
        "seed": config.get("seed"),
        "dataset": config["data"]["dataset"],
        "classes": config["data"]["class_names"],
        "input": {
            "frames": config["data"]["frames"],
            "stride": config["data"]["temporal_stride"],
            "image_size": config["data"]["image_size"],
        },
        "views": config["model"]["views"],
        "fusion": config["model"]["fusion"],
        "global": config["model"]["global"],
    }
