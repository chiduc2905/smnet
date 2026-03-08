"""Hierarchical SSM modules for token-to-shot and shot-to-class encoding."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from net.ssm.common import SelectiveStateSpaceCell, run_selective_scan


class TokenLevelSSMEncoder(nn.Module):
    """Encode image tokens into refined tokens and shot descriptors."""

    def __init__(
        self,
        dim: int,
        state_dim: int,
        depth: int = 1,
        max_tokens: int = 32,
    ) -> None:
        super().__init__()
        self.position_embedding = nn.Embedding(max_tokens, dim)
        self.cells = nn.ModuleList([SelectiveStateSpaceCell(dim, state_dim) for _ in range(depth)])
        self.state_to_shot = nn.ModuleList([nn.Linear(state_dim, dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if tokens.dim() != 3:
            raise ValueError(f"tokens must have shape (Batch, Tokens, Dim), got {tuple(tokens.shape)}")
        position_ids = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = tokens + self.position_embedding(position_ids).unsqueeze(0)
        shot_state = None
        for layer_idx, cell in enumerate(self.cells):
            hidden, shot_state = run_selective_scan(cell, hidden)
            hidden = hidden + self.state_to_shot[layer_idx](shot_state).unsqueeze(1)
        shot_embedding = self.norm(hidden.mean(dim=1) + self.state_to_shot[-1](shot_state))
        return hidden, shot_embedding


class ShotLevelMemorySSM(nn.Module):
    """Aggregate shot descriptors within a class."""

    def __init__(self, dim: int, state_dim: int, depth: int = 1) -> None:
        super().__init__()
        self.cells = nn.ModuleList([SelectiveStateSpaceCell(dim, state_dim) for _ in range(depth)])
        self.state_to_class = nn.ModuleList([nn.Linear(state_dim, dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, shot_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if shot_embeddings.dim() == 2:
            shot_embeddings = shot_embeddings.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False
        hidden = shot_embeddings
        class_state = None
        for layer_idx, cell in enumerate(self.cells):
            hidden, state = run_selective_scan(cell, hidden)
            class_state = self.state_to_class[layer_idx](state)
            hidden = hidden + class_state.unsqueeze(1)
        class_repr = self.norm(class_state + hidden.mean(dim=1))
        if squeeze_batch:
            return class_repr.squeeze(0), hidden.squeeze(0)
        return class_repr, hidden


class HierarchicalClassAggregator(nn.Module):
    """Fuse shot-level summaries with the class memory state."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fusion = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, shot_embeddings: torch.Tensor, class_memory: torch.Tensor) -> torch.Tensor:
        shot_summary = shot_embeddings.mean(dim=-2)
        fused = self.fusion(torch.cat([shot_summary, class_memory], dim=-1))
        return self.norm(fused + class_memory)
