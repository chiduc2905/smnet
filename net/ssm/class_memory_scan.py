"""Class-memory write/read SSM modules for few-shot learning."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from net.ssm.common import SelectiveStateSpaceCell, run_selective_scan


class ShotDescriptorEncoder(nn.Module):
    """Encode per-shot token sets into compact support descriptors."""

    def __init__(self, token_dim: int, descriptor_dim: int | None = None) -> None:
        super().__init__()
        descriptor_dim = descriptor_dim or token_dim
        self.token_score = nn.Linear(token_dim, 1)
        self.token_proj = nn.Linear(token_dim, descriptor_dim)
        self.global_proj = nn.Linear(token_dim, descriptor_dim)
        self.norm = nn.LayerNorm(descriptor_dim)

    def forward(
        self,
        shot_tokens: torch.Tensor,
        shot_globals: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if shot_tokens.dim() < 3:
            raise ValueError(
                "shot_tokens must have shape (..., Tokens, Dim), "
                f"got {tuple(shot_tokens.shape)}"
            )
        weights = torch.softmax(self.token_score(shot_tokens).squeeze(-1), dim=-1)
        token_summary = torch.einsum("...t,...td->...d", weights, self.token_proj(shot_tokens))
        if shot_globals is not None:
            token_summary = token_summary + self.global_proj(shot_globals)
        return self.norm(token_summary)


class ClassMemoryWriteSSM(nn.Module):
    """Scan support shot descriptors into a class-specific memory state."""

    def __init__(self, dim: int, state_dim: int, depth: int = 1) -> None:
        super().__init__()
        self.cells = nn.ModuleList([SelectiveStateSpaceCell(dim, state_dim) for _ in range(depth)])
        self.state_to_memory = nn.ModuleList([nn.Linear(state_dim, dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, shot_descriptors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        squeeze_batch = False
        if shot_descriptors.dim() == 2:
            shot_descriptors = shot_descriptors.unsqueeze(0)
            squeeze_batch = True
        if shot_descriptors.dim() != 3:
            raise ValueError(
                "shot_descriptors must have shape (Batch, Shot, Dim) or (Shot, Dim), "
                f"got {tuple(shot_descriptors.shape)}"
            )

        hidden = shot_descriptors
        final_memory = None
        for layer_idx, cell in enumerate(self.cells):
            hidden, final_state = run_selective_scan(cell, hidden)
            final_memory = self.state_to_memory[layer_idx](final_state)
            hidden = hidden + final_memory.unsqueeze(1)

        memory = self.norm(final_memory + hidden[:, -1])
        if squeeze_batch:
            return memory.squeeze(0), hidden.squeeze(0)
        return memory, hidden


class ClassMemoryReadSSM(nn.Module):
    """Read query evidence from a class memory without concatenating support/query."""

    def __init__(self, dim: int, state_dim: int, depth: int = 1) -> None:
        super().__init__()
        self.cells = nn.ModuleList([SelectiveStateSpaceCell(dim, state_dim) for _ in range(depth)])
        self.state_init = nn.ModuleList([nn.Linear(dim, state_dim) for _ in range(depth)])
        self.memory_condition = nn.Linear(dim, dim)
        self.query_condition = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        query_tokens: torch.Tensor,
        class_memory: torch.Tensor,
        query_global: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if query_tokens.dim() != 3:
            raise ValueError(
                "query_tokens must have shape (NQ, Tokens, Dim), "
                f"got {tuple(query_tokens.shape)}"
            )
        if class_memory.dim() == 1:
            class_memory = class_memory.unsqueeze(0).expand(query_tokens.shape[0], -1)
        elif class_memory.dim() != 2:
            raise ValueError(f"class_memory must have shape (Dim,) or (NQ, Dim), got {tuple(class_memory.shape)}")

        hidden = query_tokens
        conditioning = self.memory_condition(class_memory)
        if query_global is not None:
            conditioning = conditioning + self.query_condition(query_global)

        for layer_idx, cell in enumerate(self.cells):
            init_state = self.state_init[layer_idx](class_memory)
            cond_seq = conditioning.unsqueeze(1).expand(-1, hidden.shape[1], -1)
            hidden, _ = run_selective_scan(
                cell,
                hidden,
                conditioning=cond_seq,
                initial_state=init_state,
            )

        readout = self.norm(hidden.mean(dim=1) + class_memory)
        if query_global is not None:
            readout = self.norm(readout + query_global)
        return readout, hidden
