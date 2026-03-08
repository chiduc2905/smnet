"""Role and phase embeddings for episodic selective scan models."""

from __future__ import annotations

import torch
import torch.nn as nn


class EpisodicRoleEmbedding(nn.Module):
    """Embed episodic semantics such as role, phase, position, and class id."""

    def __init__(self, dim: int, max_positions: int = 32, max_classes: int = 32) -> None:
        super().__init__()
        self.role_embedding = nn.Embedding(2, dim)
        self.phase_embedding = nn.Embedding(2, dim)
        self.position_embedding = nn.Embedding(max_positions, dim)
        self.class_embedding = nn.Embedding(max_classes, dim)
        self.boundary_embedding = nn.Embedding(2, dim)

    def forward(
        self,
        role_ids: torch.Tensor,
        phase_ids: torch.Tensor,
        position_ids: torch.Tensor,
        class_ids: torch.Tensor,
        boundary_ids: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.role_embedding(role_ids)
            + self.phase_embedding(phase_ids)
            + self.position_embedding(position_ids)
            + self.class_embedding(class_ids)
            + self.boundary_embedding(boundary_ids)
        )
