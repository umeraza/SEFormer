"""Pure-Python architecture accounting used by audits and tests."""

from __future__ import annotations

from itertools import pairwise
from math import prod
from typing import Any


PAPER_REPORTED_PATCH_TOKENS = {
    "1,4,8_x_8": 1782,
    "2,4,8_x_4": 2564,
    "2,4,8_x_8": 1288,
    "2,4,8_x_16": 332,
}


def view_patch_grid(
    frames: int, image_size: list[int] | tuple[int, int], patch_size: list[int]
) -> tuple[int, int, int]:
    height, width = image_size
    temporal, patch_h, patch_w = patch_size
    return frames // temporal, height // patch_h, width // patch_w


def token_accounting(config: dict[str, Any]) -> dict[str, Any]:
    data, model = config["data"], config["model"]
    rows: list[dict[str, Any]] = []
    for view in model["views"]:
        grid = view_patch_grid(data["frames"], data["image_size"], view["patch_size"])
        patch_tokens = prod(grid)
        rows.append(
            {
                "name": view["name"],
                "patch_size": list(view["patch_size"]),
                "grid": list(grid),
                "patch_tokens": patch_tokens,
                "tokens_with_cls": patch_tokens + int(bool(model.get("use_cls_token", True))),
            }
        )
    return {
        "views": rows,
        "total_patch_tokens": sum(row["patch_tokens"] for row in rows),
        "total_tokens_with_cls": sum(row["tokens_with_cls"] for row in rows),
    }


def _linear_params(input_dim: int, output_dim: int, bias: bool = True) -> int:
    return input_dim * output_dim + (output_dim if bias else 0)


def _layer_norm_params(dim: int) -> int:
    return 2 * dim


def _self_attention_params(dim: int) -> int:
    # q/k/v projection and output projection, all with bias.
    return 3 * _linear_params(dim, dim) + _linear_params(dim, dim)


def _cross_attention_params(query_dim: int, context_dim: int) -> int:
    # PyTorch MHA with separate q, k, v projections when kdim/vdim differ.
    return (
        _linear_params(query_dim, query_dim)
        + 2 * _linear_params(context_dim, query_dim)
        + _linear_params(query_dim, query_dim)
        + _layer_norm_params(query_dim)
        + _layer_norm_params(context_dim)
    )


def _transformer_block_params(dim: int, mlp_dim: int) -> int:
    return (
        _self_attention_params(dim)
        + 2 * _layer_norm_params(dim)
        + _linear_params(dim, mlp_dim)
        + _linear_params(mlp_dim, dim)
    )


def analytical_parameter_count(config: dict[str, Any]) -> dict[str, int]:
    """Estimate trainable parameters for the repository implementation.

    This mirrors the modules without importing PyTorch, allowing CI and paper
    audits in lightweight environments. The exact instantiated count is reported
    by ``scripts/benchmark.py`` when PyTorch is available.
    """
    data, model = config["data"], config["model"]
    token_info = token_accounting(config)
    use_cls = bool(model.get("use_cls_token", True))
    components: dict[str, int] = {}

    view_total = 0
    for view, tokens in zip(model["views"], token_info["views"], strict=True):
        temporal, patch_h, patch_w = view["patch_size"]
        dim = view["embed_dim"]
        if model["patchification"] == "3d":
            patch_params = dim * model["in_channels"] * temporal * patch_h * patch_w + dim
        else:
            patch_params = dim * model["in_channels"] * patch_h * patch_w + dim
        positional = tokens["tokens_with_cls"] * dim
        cls = dim if use_cls else 0
        encoder = view["depth"] * _transformer_block_params(dim, view["mlp_dim"])
        final_norm = _layer_norm_params(dim)
        pool = _linear_params(dim, 1) if model["pooling"] == "sequence" else 0
        subtotal = patch_params + positional + cls + encoder + final_norm + pool
        components[f"view_{view['name']}"] = subtotal
        view_total += subtotal
    components["views_total"] = view_total

    ordered = sorted(
        zip(model["views"], token_info["views"], strict=True),
        key=lambda item: item[1]["patch_tokens"],
    )
    fusion_total = 0
    if model["fusion"].get("enabled", True) and len(ordered) > 1:
        for (coarse, _), (fine, _) in pairwise(ordered):
            fusion_total += _cross_attention_params(coarse["embed_dim"], fine["embed_dim"])
            if model["fusion"].get("direction") == "bidirectional":
                fusion_total += _cross_attention_params(fine["embed_dim"], coarse["embed_dim"])
    components["fusion"] = fusion_total

    global_cfg = model["global"]
    global_dim = global_cfg["dim"]
    projections = sum(_linear_params(view["embed_dim"], global_dim) for view in model["views"])
    global_pos = len(model["views"]) * global_dim if global_cfg["mode"] == "transformer" else 0
    global_pool = _linear_params(global_dim, 1)
    global_encoder = 0
    if global_cfg["mode"] == "transformer":
        global_encoder = global_cfg["depth"] * _transformer_block_params(
            global_dim, global_cfg["mlp_dim"]
        )
        global_encoder += _layer_norm_params(global_dim)
        classifier_input = global_dim
    else:
        classifier_input = len(model["views"]) * global_dim
        global_pool = 0

    classifier_hidden = model["classifier_hidden"]
    classifier = (
        _layer_norm_params(classifier_input)
        + _linear_params(classifier_input, classifier_hidden)
        + _linear_params(classifier_hidden, data["num_classes"])
    )
    components["global"] = projections + global_pos + global_encoder + global_pool
    components["classifier"] = classifier
    components["total"] = (
        components["views_total"]
        + components["fusion"]
        + components["global"]
        + components["classifier"]
    )
    return components


def format_count(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3f} M"
    if value >= 1_000:
        return f"{value / 1_000:.3f} K"
    return str(value)
