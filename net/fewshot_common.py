"""Shared utilities for Conv64F few-shot architectures."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from net.backbone.conv64f import Conv64FBackbone


def feature_map_to_tokens(feature_map: torch.Tensor) -> torch.Tensor:
    """Convert `(N, C, H, W)` feature maps to `(N, H*W, C)` token sets."""
    if feature_map.dim() != 4:
        raise ValueError(f"Expected a 4D feature map, got shape={tuple(feature_map.shape)}")
    return feature_map.flatten(2).transpose(1, 2).contiguous()


def merge_support_tokens(support_tokens: torch.Tensor, merge_mode: str = "concat") -> torch.Tensor:
    """Merge support token sets class-wise.

    Args:
        support_tokens: `(Way, Shot, Tokens, Dim)`.
        merge_mode: `concat` preserves every support token; `mean` averages
            aligned tokens across shots and returns `(Way, Tokens, Dim)`.
    """
    if support_tokens.dim() != 4:
        raise ValueError(
            "support_tokens must have shape (Way, Shot, Tokens, Dim), "
            f"got {tuple(support_tokens.shape)}"
        )
    if merge_mode == "concat":
        way_num, shot_num, token_num, dim = support_tokens.shape
        return support_tokens.reshape(way_num, shot_num * token_num, dim)
    if merge_mode == "mean":
        return support_tokens.mean(dim=1)
    raise ValueError(f"Unsupported merge_mode: {merge_mode}")


def pooled_episode_features(feature_map: torch.Tensor) -> torch.Tensor:
    """Global-average-pool a `(N, C, H, W)` tensor to `(N, C)`."""
    if feature_map.dim() != 4:
        raise ValueError(f"Expected a 4D feature map, got shape={tuple(feature_map.shape)}")
    return feature_map.mean(dim=(2, 3))


class BaseConv64FewShotModel(nn.Module):
    """Base class for the new Conv64F few-shot benchmark models."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 64,
        conv64f_pool_last: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.backbone = Conv64FBackbone(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            pool_last=conv64f_pool_last,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images into Conv64F feature maps."""
        return self.backbone(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract global pooled features for visualization utilities."""
        return pooled_episode_features(self.encode(x))

    def get_features(self, images: torch.Tensor) -> torch.Tensor:
        """Return spatial feature maps for external analysis utilities."""
        return self.encode(images)

    @staticmethod
    def validate_episode_inputs(query: torch.Tensor, support: torch.Tensor) -> Tuple[int, int, int, int, int, int]:
        """Validate episode tensor shapes and return `(B, NQ, C, H, W, Way)`."""
        if query.dim() != 5:
            raise ValueError(f"query must have shape (B, NQ, C, H, W), got {tuple(query.shape)}")
        if support.dim() != 6:
            raise ValueError(
                f"support must have shape (B, Way, Shot, C, H, W), got {tuple(support.shape)}"
            )
        bsz, nq, channels, height, width = query.shape
        if support.shape[0] != bsz or support.shape[3:] != (channels, height, width):
            raise ValueError(
                "Support/query episode shapes are inconsistent: "
                f"query={tuple(query.shape)} support={tuple(support.shape)}"
            )
        way_num = support.shape[1]
        return bsz, nq, channels, height, width, way_num
