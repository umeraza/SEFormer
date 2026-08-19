"""Complete SEFormer architecture."""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import torch
from torch import nn

from .analysis import token_accounting
from .layers import (
    CrossViewAttentionFusion,
    PatchEmbedding2DTemporalPool,
    PatchEmbedding3D,
    SequencePool,
    TransformerEncoderStack,
)


class ViewEncoder(nn.Module):
    def __init__(
        self,
        *,
        input_shape: tuple[int, int, int],
        in_channels: int,
        patchification: str,
        patch_size: tuple[int, int, int],
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        drop_path: float,
        use_cls_token: bool,
        gradient_checkpointing: bool,
    ) -> None:
        super().__init__()
        frames, height, width = input_shape
        self.expected_input_shape = input_shape
        self.patch_size = patch_size
        self.dim = dim
        self.use_cls_token = use_cls_token
        embedding_type = (
            PatchEmbedding3D if patchification == "3d" else PatchEmbedding2DTemporalPool
        )
        self.patch_embedding = embedding_type(in_channels, dim, patch_size)
        self.num_patches = (
            (frames // patch_size[0])
            * (height // patch_size[1])
            * (width // patch_size[2])
        )
        self.class_token = nn.Parameter(torch.zeros(1, 1, dim)) if use_cls_token else None
        self.position = nn.Parameter(
            torch.zeros(1, self.num_patches + int(use_cls_token), dim)
        )
        self.position_dropout = nn.Dropout(dropout)
        self.encoder = TransformerEncoderStack(
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            max_drop_path=drop_path,
            gradient_checkpointing=gradient_checkpointing,
            final_norm=True,
        )
        nn.init.trunc_normal_(self.position, std=0.02)
        if self.class_token is not None:
            nn.init.trunc_normal_(self.class_token, std=0.02)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        observed = (video.shape[2], video.shape[3], video.shape[4])
        if observed != self.expected_input_shape:
            raise ValueError(
                f"Expected video shape T,H,W={self.expected_input_shape}, observed {observed}"
            )
        tokens = self.patch_embedding(video)
        if tokens.shape[1] != self.num_patches:
            raise RuntimeError(
                f"Patch projection emitted {tokens.shape[1]} tokens; expected {self.num_patches}"
            )
        if self.class_token is not None:
            class_tokens = self.class_token.expand(video.shape[0], -1, -1)
            tokens = torch.cat((class_tokens, tokens), dim=1)
        tokens = self.position_dropout(tokens + self.position)
        return self.encoder(tokens)


class SEFormer(nn.Module):
    """Multi-view spatio-temporal Transformer with CVAF and sequence pooling."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        data, model = config["data"], config["model"]
        input_shape = (data["frames"], *data["image_size"])
        self.use_cls_token = bool(model.get("use_cls_token", True))
        self.pooling_mode = model["pooling"]
        self.pool_include_cls = bool(model.get("pool_include_cls", True))
        self.view_names = [view["name"] for view in model["views"]]
        self.view_definitions = model["views"]

        self.views = nn.ModuleList(
            [
                ViewEncoder(
                    input_shape=input_shape,
                    in_channels=model["in_channels"],
                    patchification=model["patchification"],
                    patch_size=tuple(view["patch_size"]),
                    dim=view["embed_dim"],
                    depth=view["depth"],
                    heads=view["heads"],
                    mlp_dim=view["mlp_dim"],
                    dropout=model["encoder_dropout"],
                    attention_dropout=model["attention_dropout"],
                    drop_path=model["drop_path"],
                    use_cls_token=self.use_cls_token,
                    gradient_checkpointing=bool(model.get("gradient_checkpointing", False)),
                )
                for view in model["views"]
            ]
        )

        token_info = token_accounting(config)["views"]
        ordered_indices = sorted(
            range(len(self.views)), key=lambda index: token_info[index]["patch_tokens"]
        )
        self.fusion_pairs = list(pairwise(ordered_indices))
        self.fusion_enabled = bool(model["fusion"].get("enabled", True)) and bool(
            self.fusion_pairs
        )
        self.fusion_direction = model["fusion"].get("direction", "coarse_to_fine")
        self.fusions = nn.ModuleDict()
        if self.fusion_enabled:
            for coarse_index, fine_index in self.fusion_pairs:
                coarse_dim = model["views"][coarse_index]["embed_dim"]
                fine_dim = model["views"][fine_index]["embed_dim"]
                key = f"{coarse_index}_from_{fine_index}"
                self.fusions[key] = CrossViewAttentionFusion(
                    query_dim=coarse_dim,
                    context_dim=fine_dim,
                    heads=model["fusion"]["heads"],
                    dropout=model["fusion"]["dropout"],
                )
                if self.fusion_direction == "bidirectional":
                    reverse_key = f"{fine_index}_from_{coarse_index}"
                    self.fusions[reverse_key] = CrossViewAttentionFusion(
                        query_dim=fine_dim,
                        context_dim=coarse_dim,
                        heads=model["fusion"]["heads"],
                        dropout=model["fusion"]["dropout"],
                    )

        self.view_poolers = nn.ModuleList(
            [
                SequencePool(view["embed_dim"])
                if self.pooling_mode == "sequence"
                else nn.Identity()
                for view in model["views"]
            ]
        )
        global_cfg = model["global"]
        global_dim = global_cfg["dim"]
        self.global_mode = global_cfg["mode"]
        self.global_projections = nn.ModuleList(
            [nn.Linear(view["embed_dim"], global_dim) for view in model["views"]]
        )
        if self.global_mode == "transformer":
            self.global_position = nn.Parameter(torch.zeros(1, len(self.views), global_dim))
            nn.init.trunc_normal_(self.global_position, std=0.02)
        else:
            self.register_parameter("global_position", None)
        self.global_encoder = TransformerEncoderStack(
            dim=global_dim,
            depth=global_cfg["depth"] if self.global_mode == "transformer" else 0,
            heads=global_cfg["heads"],
            mlp_dim=global_cfg["mlp_dim"],
            dropout=global_cfg["dropout"],
            attention_dropout=global_cfg["attention_dropout"],
            max_drop_path=global_cfg["drop_path"],
            gradient_checkpointing=bool(model.get("gradient_checkpointing", False)),
            final_norm=self.global_mode == "transformer",
        )
        self.global_pool = SequencePool(global_dim) if self.global_mode == "transformer" else None

        if self.global_mode == "transformer":
            classifier_input = global_dim
        else:
            classifier_input = len(self.views) * global_dim
        hidden = model["classifier_hidden"]
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input),
            nn.Linear(classifier_input, hidden),
            nn.GELU(),
            nn.Dropout(model["classifier_dropout"]),
            nn.Linear(hidden, data["num_classes"]),
        )

    def _fuse(
        self, streams: list[torch.Tensor], return_attention: bool
    ) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
        attention: dict[str, torch.Tensor] = {}
        if not self.fusion_enabled:
            return streams, attention
        for coarse_index, fine_index in self.fusion_pairs:
            coarse_original = streams[coarse_index]
            fine_original = streams[fine_index]
            key = f"{coarse_index}_from_{fine_index}"
            coarse_updated, weights = self.fusions[key](
                coarse_original, fine_original, return_weights=return_attention
            )
            streams[coarse_index] = coarse_updated
            if weights is not None:
                attention[f"cvaf_{self.view_names[coarse_index]}_from_{self.view_names[fine_index]}"] = (
                    weights
                )
            if self.fusion_direction == "bidirectional":
                reverse_key = f"{fine_index}_from_{coarse_index}"
                fine_updated, reverse_weights = self.fusions[reverse_key](
                    fine_original, coarse_original, return_weights=return_attention
                )
                streams[fine_index] = fine_updated
                if reverse_weights is not None:
                    attention[
                        f"cvaf_{self.view_names[fine_index]}_from_{self.view_names[coarse_index]}"
                    ] = reverse_weights
        return streams, attention

    def _pool_view(
        self,
        tokens: torch.Tensor,
        pooler: nn.Module,
        *,
        return_attention: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.pooling_mode == "cls":
            return tokens[:, 0], None
        if not isinstance(pooler, SequencePool):
            raise TypeError("Sequence pooling mode requires a SequencePool module")
        pool_tokens = tokens
        if self.use_cls_token and not self.pool_include_cls:
            pool_tokens = pool_tokens[:, 1:]
        if return_attention:
            pooled, weights = pooler(pool_tokens, return_weights=True)
            return pooled, weights
        return pooler(pool_tokens), None

    def forward_features(
        self, video: torch.Tensor, *, return_attention: bool = False
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        streams = [view(video) for view in self.views]
        streams, attention = self._fuse(streams, return_attention)

        pooled_views: list[torch.Tensor] = []
        for name, stream, pooler in zip(
            self.view_names, streams, self.view_poolers, strict=True
        ):
            pooled, weights = self._pool_view(
                stream, pooler, return_attention=return_attention
            )
            pooled_views.append(pooled)
            if weights is not None:
                attention[f"pool_{name}"] = weights

        projected = [
            projection(pooled)
            for projection, pooled in zip(self.global_projections, pooled_views, strict=True)
        ]
        global_tokens = torch.stack(projected, dim=1)
        if self.global_mode == "mlp":
            return global_tokens.flatten(1), attention

        if self.global_position is None or self.global_pool is None:
            raise RuntimeError("Transformer global mode is missing position or pooling parameters")
        global_tokens = self.global_encoder(global_tokens + self.global_position)
        if return_attention:
            representation, weights = self.global_pool(global_tokens, return_weights=True)
            attention["pool_global"] = weights
        else:
            representation = self.global_pool(global_tokens)
        return representation, attention

    def forward(
        self, video: torch.Tensor, *, return_attention: bool = False
    ) -> torch.Tensor | dict[str, Any]:
        representation, attention = self.forward_features(
            video, return_attention=return_attention
        )
        logits = self.classifier(representation)
        if return_attention:
            return {"logits": logits, "attention": attention, "features": representation}
        return logits


def build_model(config: dict[str, Any]) -> SEFormer:
    return SEFormer(config)
