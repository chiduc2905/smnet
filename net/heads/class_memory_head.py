"""Heads for class-memory few-shot models."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.metrics.sliced_wasserstein import SlicedWassersteinDistance, merge_support_tokens_by_class


class ClassMemoryReadoutHead(nn.Module):
    """Turn class-conditioned query readouts into few-shot logits."""

    def __init__(self, dim: int, temperature: float = 16.0) -> None:
        super().__init__()
        self.temperature = temperature
        self.fusion = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, query_readouts: torch.Tensor, class_memories: torch.Tensor) -> torch.Tensor:
        memory_expanded = class_memories.unsqueeze(0).expand(query_readouts.shape[0], -1, -1)
        fused = self.fusion(torch.cat([query_readouts, memory_expanded], dim=-1))
        fused = self.norm(fused + query_readouts)
        return self.temperature * torch.einsum(
            "nwd,wd->nw",
            F.normalize(fused, p=2, dim=-1),
            F.normalize(class_memories, p=2, dim=-1),
        )


class AuxiliaryTokenSWAlignment(nn.Module):
    """Auxiliary SW score between refined query tokens and class token sets."""

    def __init__(
        self,
        sw_distance: SlicedWassersteinDistance,
        merge_mode: str = "concat",
        score_scale: float = 8.0,
    ) -> None:
        super().__init__()
        self.sw_distance = sw_distance
        self.merge_mode = merge_mode
        self.score_scale = score_scale

    def forward(self, query_tokens: torch.Tensor, support_tokens: torch.Tensor) -> torch.Tensor:
        support_merged = merge_support_tokens_by_class(support_tokens, merge_mode=self.merge_mode)
        if query_tokens.dim() == 3:
            query_tokens = query_tokens.unsqueeze(1).expand(-1, support_merged.shape[0], -1, -1)
        if query_tokens.dim() != 4:
            raise ValueError(
                "query_tokens must have shape (NQ, Tokens, Dim) or (NQ, Way, Tokens, Dim), "
                f"got {tuple(query_tokens.shape)}"
            )
        support_expanded = support_merged.unsqueeze(0).expand(query_tokens.shape[0], -1, -1, -1)
        return -self.score_scale * self.sw_distance(query_tokens, support_expanded, reduction="none")


class PermutationAwareClassReadout(ClassMemoryReadoutHead):
    """Class-memory readout with a class-specific consistency bias."""

    def forward(
        self,
        query_readouts: torch.Tensor,
        class_memories: torch.Tensor,
        consistency_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scores = super().forward(query_readouts, class_memories)
        if consistency_bias is not None:
            scores = scores + consistency_bias.unsqueeze(0)
        return scores
