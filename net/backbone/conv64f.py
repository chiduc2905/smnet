"""Conv64F backbone shared by the new few-shot benchmark models.

The canonical Conv64F stack uses four 3x3 convolution blocks with 64 channels.
For 64x64 inputs and pooling in every block, the output spatial map is 4x4,
which yields 16 tokens for token-level metric learning and episodic scanning.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Conv64FBlock(nn.Module):
    """Single Conv64F block with optional spatial downsampling."""

    def __init__(self, in_channels: int, out_channels: int, use_pool: bool = True) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        ]
        if use_pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Conv64FBackbone(nn.Module):
    """Standard Conv64F feature extractor for few-shot learning.

    Args:
        in_channels: Number of input channels.
        hidden_dim: Number of channels per Conv64F stage. Standard Conv64F uses 64.
        pool_last: Whether to pool after the final block. For 64x64 inputs the
            default `True` gives a 4x4 output feature map.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 64,
        pool_last: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pool_last = pool_last
        self.blocks = nn.Sequential(
            Conv64FBlock(in_channels, hidden_dim, use_pool=True),
            Conv64FBlock(hidden_dim, hidden_dim, use_pool=True),
            Conv64FBlock(hidden_dim, hidden_dim, use_pool=True),
            Conv64FBlock(hidden_dim, hidden_dim, use_pool=pool_last),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)
