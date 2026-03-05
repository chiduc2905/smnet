"""Late single-head attention bridge for lightweight global refinement.

Design:
    x_out = x + gamma_attn * Attn(x) + gamma_local * LocalDW(x)

Where:
    - Attn(x): single-head window attention on normalized features
    - LocalDW(x): depthwise-local branch for short-range compensation
    - gamma_*: learnable residual scales initialized to 0 for stable warm-up
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LateSingleHeadAttentionBridge(nn.Module):
    """Single-head window attention bridge in BCHW format."""

    def __init__(
        self,
        channels: int,
        window_size: int = 4,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.channels = channels
        self.window_size = max(1, int(window_size))

        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.proj = nn.Linear(channels, channels, bias=False)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.scale = channels ** -0.5

        self.local_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
        )

        self.gamma_attn = nn.Parameter(torch.tensor(0.0))
        self.gamma_local = nn.Parameter(torch.tensor(0.0))

    def _window_partition(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int, int, int, int, int]]:
        """Partition BHWC tensor into non-overlapping windows."""
        bsz, h, w, ch = x.shape
        ws = self.window_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws

        if pad_h > 0 or pad_w > 0:
            x = x.permute(0, 3, 1, 2).contiguous()
            x = F.pad(x, (0, pad_w, 0, pad_h))
            x = x.permute(0, 2, 3, 1).contiguous()

        _, hp, wp, _ = x.shape
        x = x.view(bsz, hp // ws, ws, wp // ws, ws, ch)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws * ws, ch)
        meta = (bsz, h, w, hp, wp, pad_h, pad_w)
        return x, meta

    def _window_reverse(self, x: torch.Tensor, meta: Tuple[int, int, int, int, int, int, int]) -> torch.Tensor:
        """Restore window sequence back to BHWC tensor."""
        bsz, h, w, hp, wp, pad_h, pad_w = meta
        ws = self.window_size
        ch = x.shape[-1]

        x = x.view(bsz, hp // ws, wp // ws, ws, ws, ch)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(bsz, hp, wp, ch)
        if pad_h > 0 or pad_w > 0:
            x = x[:, :h, :w, :].contiguous()
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply late bridge to BCHW tensor."""
        bsz, ch, h, w = x.shape
        if ch != self.channels:
            raise ValueError(f"Expected channels={self.channels}, got {ch}")

        x_hwc = x.permute(0, 2, 3, 1).contiguous()
        x_norm = self.norm(x_hwc)

        qkv = self.qkv(x_norm)
        qkv_win, meta = self._window_partition(qkv)  # (B*nw, ws*ws, 3C)
        q, k, v = qkv_win.chunk(3, dim=-1)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        attn_out = torch.matmul(attn, v)
        attn_out = self.proj(attn_out)
        attn_out = self._window_reverse(attn_out, meta)  # (B, H, W, C)
        attn_out = attn_out.permute(0, 3, 1, 2).contiguous()

        local_in = x_norm.permute(0, 3, 1, 2).contiguous()
        local_out = self.local_branch(local_in)

        return x + self.gamma_attn * attn_out + self.gamma_local * local_out


def build_late_attention_bridge(
    channels: int,
    window_size: int = 4,
    attn_dropout: float = 0.0,
) -> LateSingleHeadAttentionBridge:
    """Factory function for LateSingleHeadAttentionBridge."""
    return LateSingleHeadAttentionBridge(
        channels=channels,
        window_size=window_size,
        attn_dropout=attn_dropout,
    )
