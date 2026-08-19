"""SEFormer tokenization, attention, Transformer, and pooling layers."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class DropPath(nn.Module):
    """Per-sample stochastic depth."""

    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError("DropPath probability must lie in [0,1)")
        self.probability = float(probability)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.probability == 0.0 or not self.training:
            return inputs
        keep = 1.0 - self.probability
        shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)
        mask = inputs.new_empty(shape).bernoulli_(keep)
        return inputs * mask / keep


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class TransformerBlock(nn.Module):
    """Pre-normalized Transformer block matching the manuscript equations."""

    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.attention_output_dropout = nn.Dropout(dropout)
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(inputs)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        inputs = inputs + self.drop_path1(self.attention_output_dropout(attended))
        return inputs + self.drop_path2(self.mlp(self.norm2(inputs)))


class TransformerEncoderStack(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        max_drop_path: float,
        gradient_checkpointing: bool = False,
        final_norm: bool = True,
    ) -> None:
        super().__init__()
        probabilities = torch.linspace(0, max_drop_path, depth).tolist() if depth else []
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    heads=heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    drop_path=probabilities[index],
                )
                for index in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim) if final_norm else nn.Identity()
        self.gradient_checkpointing = gradient_checkpointing

    def _run_block(self, block: Callable, inputs: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training and inputs.requires_grad:
            return checkpoint(block, inputs, use_reentrant=False)
        return block(inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            inputs = self._run_block(block, inputs)
        return self.norm(inputs)


class PatchEmbedding3D(nn.Module):
    """Non-overlapping volumetric tubelet projection."""

    def __init__(self, in_channels: int, dim: int, patch_size: tuple[int, int, int]) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv3d(
            in_channels,
            dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        tokens = self.projection(video)
        return tokens.flatten(2).transpose(1, 2)


class PatchEmbedding2DTemporalPool(nn.Module):
    """Framewise 2D patches followed by non-overlapping temporal mean pooling."""

    def __init__(self, in_channels: int, dim: int, patch_size: tuple[int, int, int]) -> None:
        super().__init__()
        temporal, patch_h, patch_w = patch_size
        self.temporal_group = temporal
        self.spatial_patch = (patch_h, patch_w)
        self.projection = nn.Conv2d(
            in_channels,
            dim,
            kernel_size=self.spatial_patch,
            stride=self.spatial_patch,
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, height, width = video.shape
        frame_batch = video.permute(0, 2, 1, 3, 4).reshape(
            batch * frames, channels, height, width
        )
        patches = self.projection(frame_batch)
        _, dim, grid_h, grid_w = patches.shape
        patches = patches.reshape(batch, frames, dim, grid_h, grid_w)
        usable = (frames // self.temporal_group) * self.temporal_group
        if usable == 0:
            raise ValueError(
                f"Input has {frames} frames, fewer than temporal group {self.temporal_group}"
            )
        patches = patches[:, :usable].reshape(
            batch,
            usable // self.temporal_group,
            self.temporal_group,
            dim,
            grid_h,
            grid_w,
        )
        patches = patches.mean(dim=2)
        return patches.permute(0, 1, 3, 4, 2).reshape(batch, -1, dim)


class SequencePool(nn.Module):
    """Learned softmax weighting over every token in a sequence."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(
        self, tokens: torch.Tensor, *, return_weights: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        weights = self.score(tokens).transpose(1, 2).softmax(dim=-1)
        pooled = torch.bmm(weights, tokens).squeeze(1)
        if return_weights:
            return pooled, weights.squeeze(1)
        return pooled


class CrossViewAttentionFusion(nn.Module):
    """Update a query stream using another view as keys and values."""

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(query_dim)
        self.context_norm = nn.LayerNorm(context_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=heads,
            dropout=dropout,
            kdim=context_dim,
            vdim=context_dim,
            batch_first=True,
        )
        self.output_dropout = nn.Dropout(dropout)

    def forward(
        self,
        query_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        *,
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attended, weights = self.attention(
            self.query_norm(query_tokens),
            self.context_norm(context_tokens),
            self.context_norm(context_tokens),
            need_weights=return_weights,
            average_attn_weights=True,
        )
        return query_tokens + self.output_dropout(attended), weights if return_weights else None
