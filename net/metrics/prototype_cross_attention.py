"""Prototype-guided cross-attention for few-shot metric learning.

Key upgrades:
1) Weighted prototype refinement (outlier-aware)
2) Multi-prototype per class (K) for intra-class modes
3) Higher-resolution prototype pooling for harder classes
4) Optional prototype detach (off by default)
5) Optional dual-axis tokenization (time/frequency aware)
"""

import math
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeCrossAttention(nn.Module):
    """Prototype-guided cross-attention with lightweight multi-prototype modeling.

    Args:
        channels: Feature channels.
        alpha: Residual weight for attended features.
        t_attn: Temperature factor in attention scaling.
        t_proto: Temperature for weighted prototype aggregation.
        proto_pool_size: Spatial pool size of prototypes.
        num_prototypes: Number of prototypes per class.
        detach_prototypes: If True, stop gradients through prototypes.
        attn_dropout: Dropout on attention weights.
    """

    def __init__(
        self,
        channels: int,
        alpha: float = 0.1,
        t_attn: float = 2.0,
        t_proto: float = 0.7,
        proto_pool_size: int = 12,
        num_prototypes: int = 2,
        detach_prototypes: bool = False,
        attn_dropout: float = 0.1,
        use_axis_proto: bool = True,
        axis_proto_pool: str = "mean",
        axis_proto_mix_init: Sequence[float] = (1.0, 0.5, 0.5),
    ):
        super().__init__()
        self.channels = channels
        self.alpha = alpha
        self.t_attn = t_attn
        self.t_proto = t_proto
        self.proto_pool_size = proto_pool_size
        self.num_prototypes = max(1, int(num_prototypes))
        self.detach_prototypes = detach_prototypes
        self.attn_dropout = attn_dropout
        self.use_axis_proto = bool(use_axis_proto)
        self.axis_proto_pool = axis_proto_pool.lower()
        if self.axis_proto_pool not in {"mean", "max"}:
            raise ValueError(f"axis_proto_pool must be 'mean' or 'max', got {axis_proto_pool}")
        self.scale = 1.0 / (math.sqrt(channels) * t_attn)

        if len(axis_proto_mix_init) != 3:
            raise ValueError("axis_proto_mix_init must contain 3 values for [full, time, freq]")
        self.axis_mix_logits = nn.Parameter(torch.tensor(axis_proto_mix_init, dtype=torch.float32))

    def _compute_weighted_prototype(self, support: torch.Tensor) -> torch.Tensor:
        """Compute weighted mean prototype for one class support set.

        support: (Shot, C, H, W)
        returns: (C, H, W)
        """
        shot, _, _, _ = support.shape
        mu = support.mean(dim=0)

        s_flat = support.contiguous().view(shot, -1)
        mu_flat = mu.contiguous().view(1, -1)
        sim = F.cosine_similarity(s_flat, mu_flat, dim=1)
        w = F.softmax(sim / self.t_proto, dim=0)

        return (w[:, None, None, None] * support).sum(dim=0)

    def _compute_multi_prototypes(self, support: torch.Tensor, k: int) -> torch.Tensor:
        """Compute K prototypes by ranking-support chunking + weighted means."""
        shot = support.shape[0]
        k = max(1, min(k, shot))

        if k == 1:
            return self._compute_weighted_prototype(support).unsqueeze(0)

        flat = support.contiguous().view(shot, -1)
        flat = F.normalize(flat, p=2, dim=1)
        mu = F.normalize(flat.mean(dim=0, keepdim=True), p=2, dim=1)
        sim = torch.matmul(flat, mu.t()).squeeze(1)
        order = torch.argsort(sim, descending=True)

        chunks = torch.chunk(order, k)
        protos = []
        for idx in chunks:
            if idx.numel() == 0:
                continue
            protos.append(self._compute_weighted_prototype(support[idx]))

        if not protos:
            protos.append(self._compute_weighted_prototype(support))

        while len(protos) < k:
            protos.append(protos[-1].clone())

        return torch.stack(protos[:k], dim=0)  # (K, C, H, W)

    def _axis_tokens(self, proto: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create time-axis and frequency-axis prototype tokens.

        Args:
            proto: (K, C, Hp, Wp)
        Returns:
            tokens_time: (C, K*Wp) pooled across frequency axis (Hp)
            tokens_freq: (C, K*Hp) pooled across time axis (Wp)
        """
        if self.axis_proto_pool == "mean":
            proto_time = proto.mean(dim=2)  # (K, C, Wp)
            proto_freq = proto.mean(dim=3)  # (K, C, Hp)
        else:
            proto_time = proto.amax(dim=2)  # (K, C, Wp)
            proto_freq = proto.amax(dim=3)  # (K, C, Hp)

        tokens_time = proto_time.permute(1, 0, 2).contiguous().view(self.channels, -1)
        tokens_freq = proto_freq.permute(1, 0, 2).contiguous().view(self.channels, -1)
        return tokens_time, tokens_freq

    def forward(
        self,
        query_feat: torch.Tensor,
        support_feat: torch.Tensor,
        way_num: int,
        shot_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply prototype-guided cross-attention.

        Args:
            query_feat: (NQ, C, H, W)
            support_feat: (Way*Shot, C, H, W)
            way_num: Number of classes.
            shot_num: Shots per class.

        Returns:
            refined_query: (NQ, Way, C, H, W)
            proto_maps: (Way, K, C, Hp, Wp)
        """
        nq, c, h, w = query_feat.shape
        k_proto = max(1, min(self.num_prototypes, shot_num))

        support_reshaped = support_feat.reshape(way_num, shot_num, c, h, w)

        proto_refined = []
        for cls_idx in range(way_num):
            proto_c = self._compute_multi_prototypes(support_reshaped[cls_idx], k_proto)
            proto_refined.append(proto_c)
        proto_maps = torch.stack(proto_refined, dim=0)  # (Way, K, C, H, W)

        proto_maps = F.adaptive_avg_pool2d(
            proto_maps.reshape(way_num * k_proto, c, h, w),
            output_size=(self.proto_pool_size, self.proto_pool_size),
        ).reshape(way_num, k_proto, c, self.proto_pool_size, self.proto_pool_size)

        if self.detach_prototypes:
            proto_maps = proto_maps.detach()

        refined_queries = []
        for cls_idx in range(way_num):
            proto = proto_maps[cls_idx]  # (K, C, Hp, Wp)
            proto_flat = proto.permute(1, 0, 2, 3).contiguous().view(c, -1)  # (C, K*Hp*Wp)
            proto_t = proto_flat.t()  # (K*Hp*Wp, C)

            q_flat = query_feat.reshape(nq, c, -1)
            q_t = q_flat.permute(0, 2, 1)  # (NQ, H*W, C)

            attn_full = torch.matmul(q_t, proto_flat) * self.scale
            attn_full = F.softmax(attn_full, dim=-1)
            attn_full = F.dropout(attn_full, p=self.attn_dropout, training=self.training)
            attended_full = torch.matmul(attn_full, proto_t)  # (NQ, H*W, C)

            if self.use_axis_proto:
                tokens_time, tokens_freq = self._axis_tokens(proto)
                tokens_time_t = tokens_time.t().contiguous()  # (K*Wp, C)
                tokens_freq_t = tokens_freq.t().contiguous()  # (K*Hp, C)

                attn_time = torch.matmul(q_t, tokens_time) * self.scale
                attn_time = F.softmax(attn_time, dim=-1)
                attn_time = F.dropout(attn_time, p=self.attn_dropout, training=self.training)
                attended_time = torch.matmul(attn_time, tokens_time_t)

                attn_freq = torch.matmul(q_t, tokens_freq) * self.scale
                attn_freq = F.softmax(attn_freq, dim=-1)
                attn_freq = F.dropout(attn_freq, p=self.attn_dropout, training=self.training)
                attended_freq = torch.matmul(attn_freq, tokens_freq_t)

                mix = F.softmax(self.axis_mix_logits, dim=0)
                attended = (
                    mix[0] * attended_full
                    + mix[1] * attended_time
                    + mix[2] * attended_freq
                )
            else:
                attended = attended_full

            attended = attended.permute(0, 2, 1).contiguous().view(nq, c, h, w)

            refined_queries.append(query_feat + self.alpha * attended)

        refined_query = torch.stack(refined_queries, dim=1)  # (NQ, Way, C, H, W)
        return refined_query, proto_maps


def build_prototype_cross_attention(
    channels: int,
    alpha: float = 0.1,
    t_attn: float = 2.0,
    t_proto: float = 0.7,
    proto_pool_size: int = 12,
    num_prototypes: int = 2,
    detach_prototypes: bool = False,
    attn_dropout: float = 0.1,
    use_axis_proto: bool = True,
    axis_proto_pool: str = "mean",
    axis_proto_mix_init: Sequence[float] = (1.0, 0.5, 0.5),
) -> PrototypeCrossAttention:
    """Factory function for PrototypeCrossAttention."""
    return PrototypeCrossAttention(
        channels=channels,
        alpha=alpha,
        t_attn=t_attn,
        t_proto=t_proto,
        proto_pool_size=proto_pool_size,
        num_prototypes=num_prototypes,
        detach_prototypes=detach_prototypes,
        attn_dropout=attn_dropout,
        use_axis_proto=use_axis_proto,
        axis_proto_pool=axis_proto_pool,
        axis_proto_mix_init=axis_proto_mix_init,
    )
