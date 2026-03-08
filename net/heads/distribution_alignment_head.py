"""Distribution-aware heads for episodic selective scan and hierarchical models."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.embeddings.episodic_role_embedding import EpisodicRoleEmbedding
from net.metrics.sliced_wasserstein import SlicedWassersteinDistance, merge_support_tokens_by_class
from net.ssm.episodic_selective_scan import EpisodicSelectiveScanBlock


class ClassConditionalQueryReadout(nn.Module):
    """Read query tokens with class-conditioned episodic scan metadata."""

    def __init__(
        self,
        dim: int,
        state_dim: int,
        max_positions: int = 32,
        max_classes: int = 32,
        use_role_embedding: bool = True,
        use_boundary_gate: bool = True,
    ) -> None:
        super().__init__()
        self.use_role_embedding = use_role_embedding
        self.role_embedding = (
            EpisodicRoleEmbedding(dim, max_positions=max_positions, max_classes=max_classes)
            if use_role_embedding
            else None
        )
        self.scan = EpisodicSelectiveScanBlock(dim, state_dim, use_boundary_gate=use_boundary_gate)
        self.state_init = nn.Linear(dim, state_dim)
        self.class_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        query_tokens: torch.Tensor,
        class_state: torch.Tensor,
        class_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query_tokens.dim() != 3:
            raise ValueError(
                "query_tokens must have shape (NQ, Tokens, Dim), "
                f"got {tuple(query_tokens.shape)}"
            )
        nq, token_num, dim = query_tokens.shape
        device = query_tokens.device
        if class_state.dim() == 1:
            class_state = class_state.unsqueeze(0).expand(nq, -1)

        if self.use_role_embedding and self.role_embedding is not None:
            role_ids = torch.ones(nq, token_num, dtype=torch.long, device=device)
            phase_ids = torch.ones(nq, token_num, dtype=torch.long, device=device)
            position_ids = torch.arange(token_num, device=device).view(1, token_num).expand(nq, -1)
            class_ids = torch.full((nq, token_num), class_index, dtype=torch.long, device=device)
            boundary_ids = torch.zeros(nq, token_num, dtype=torch.long, device=device)
            boundary_ids[:, 0] = 1
            metadata = self.role_embedding(role_ids, phase_ids, position_ids, class_ids, boundary_ids)
        else:
            metadata = torch.zeros(nq, token_num, dim, device=device, dtype=query_tokens.dtype)
            boundary_ids = torch.zeros(nq, token_num, dtype=torch.long, device=device)
            boundary_ids[:, 0] = 1

        refined, final_state = self.scan(
            query_tokens,
            metadata=metadata,
            boundary_flags=boundary_ids,
            initial_state=self.state_init(class_state),
        )
        readout = self.norm(self.class_proj(class_state) + refined.mean(dim=1))
        readout = self.norm(readout + self.class_proj(class_state))
        return readout, refined


class DistributionAlignmentSWHead(nn.Module):
    """SW alignment head for episodic selective scan outputs."""

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
        support_expanded = support_merged.unsqueeze(0).expand(query_tokens.shape[0], -1, -1, -1)
        return -self.score_scale * self.sw_distance(query_tokens, support_expanded, reduction="none")


class AuxiliarySWConsistencyHead(DistributionAlignmentSWHead):
    """SW head used as a secondary consistency/alignment score."""


class HierarchicalQueryMatcher(nn.Module):
    """Match query shot descriptors against hierarchical class memories."""

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

    def forward(self, query_embeddings: torch.Tensor, class_memories: torch.Tensor) -> torch.Tensor:
        query_expanded = query_embeddings.unsqueeze(1).expand(-1, class_memories.shape[0], -1)
        class_expanded = class_memories.unsqueeze(0).expand(query_embeddings.shape[0], -1, -1)
        fused = self.fusion(torch.cat([query_expanded, class_expanded], dim=-1))
        fused = self.norm(fused + query_expanded)
        return self.temperature * torch.einsum(
            "nwd,wd->nw",
            F.normalize(fused, p=2, dim=-1),
            F.normalize(class_memories, p=2, dim=-1),
        )


class HierarchicalSWMetricHead(DistributionAlignmentSWHead):
    """SW metric head for hierarchical token distributions."""
