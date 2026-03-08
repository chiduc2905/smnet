"""SW-centric heads for token-level few-shot metric learning."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.metrics.sliced_wasserstein import SlicedWassersteinDistance, merge_support_tokens_by_class


class TokenSetProjector(nn.Module):
    """Project token sets to a shared metric space."""

    def __init__(self, input_dim: int, output_dim: int | None = None) -> None:
        super().__init__()
        output_dim = output_dim or input_dim
        self.proj = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(tokens))


class GlobalFeatureDistanceHead(nn.Module):
    """Compute class scores from pooled global support/query features."""

    def __init__(self, metric: str = "cosine", temperature: float = 16.0) -> None:
        super().__init__()
        if metric not in {"cosine", "sqeuclidean"}:
            raise ValueError(f"Unsupported metric: {metric}")
        self.metric = metric
        self.temperature = temperature

    def forward(self, query_global: torch.Tensor, support_global: torch.Tensor) -> torch.Tensor:
        prototypes = support_global.mean(dim=1)
        if self.metric == "cosine":
            q = F.normalize(query_global, p=2, dim=-1)
            p = F.normalize(prototypes, p=2, dim=-1)
            return self.temperature * torch.matmul(q, p.transpose(0, 1))
        diff = query_global.unsqueeze(1) - prototypes.unsqueeze(0)
        return -diff.square().mean(dim=-1)


class TokenSWClassificationHead(nn.Module):
    """Classify queries by token-level SW with optional global feature scoring."""

    def __init__(
        self,
        sw_distance: SlicedWassersteinDistance,
        merge_mode: str = "concat",
        token_scale: float = 16.0,
        global_head: GlobalFeatureDistanceHead | None = None,
        global_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.sw_distance = sw_distance
        self.merge_mode = merge_mode
        self.token_scale = token_scale
        self.global_head = global_head
        self.global_weight = global_weight

    def forward(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        query_global: torch.Tensor | None = None,
        support_global: torch.Tensor | None = None,
    ) -> torch.Tensor:
        support_merged = merge_support_tokens_by_class(support_tokens, merge_mode=self.merge_mode)
        query_expanded = query_tokens.unsqueeze(1).expand(-1, support_merged.shape[0], -1, -1)
        support_expanded = support_merged.unsqueeze(0).expand(query_tokens.shape[0], -1, -1, -1)
        token_scores = -self.token_scale * self.sw_distance(
            query_expanded,
            support_expanded,
            reduction="none",
        )

        if self.global_head is None or query_global is None or support_global is None:
            return token_scores
        global_scores = self.global_head(query_global, support_global)
        return token_scores + self.global_weight * global_scores
