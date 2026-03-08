"""Few-shot-specific selective scan modules with role and boundary awareness."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from net.embeddings.episodic_role_embedding import EpisodicRoleEmbedding
from net.ssm.common import SelectiveStateSpaceCell


class BoundaryAwareStateGate(nn.Module):
    """Boundary-conditioned carry gate for episodic state transitions."""

    def __init__(self, dim: int, state_dim: int) -> None:
        super().__init__()
        self.input_proj = nn.Linear(dim, state_dim)
        self.meta_proj = nn.Linear(dim, state_dim)
        self.boundary_bias = nn.Parameter(torch.full((state_dim,), -1.5))

    def forward(
        self,
        x_t: torch.Tensor,
        meta_t: torch.Tensor,
        boundary_t: torch.Tensor,
    ) -> torch.Tensor:
        gate = torch.sigmoid(self.input_proj(x_t) + self.meta_proj(meta_t))
        boundary = boundary_t.to(dtype=gate.dtype).unsqueeze(-1)
        reset_gate = torch.sigmoid(self.boundary_bias).view(*([1] * (gate.dim() - 1)), -1)
        return gate * (1.0 - boundary) + reset_gate * boundary


class EpisodicSelectiveScanBlock(nn.Module):
    """Selective scan block that consumes role/phase/boundary metadata."""

    def __init__(self, dim: int, state_dim: int, use_boundary_gate: bool = True) -> None:
        super().__init__()
        self.cell = SelectiveStateSpaceCell(dim, state_dim)
        self.boundary_gate = BoundaryAwareStateGate(dim, state_dim) if use_boundary_gate else None

    def forward(
        self,
        inputs: torch.Tensor,
        metadata: torch.Tensor,
        boundary_flags: torch.Tensor,
        initial_state: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if inputs.dim() != 3 or metadata.shape != inputs.shape:
            raise ValueError(
                "inputs and metadata must both have shape (Batch, Seq, Dim), "
                f"got inputs={tuple(inputs.shape)} metadata={tuple(metadata.shape)}"
            )
        state = initial_state
        if state is None:
            state = self.cell.init_state((inputs.shape[0],), device=inputs.device, dtype=inputs.dtype)

        outputs = []
        for step_idx in range(inputs.shape[1]):
            carry_scale = None
            if self.boundary_gate is not None:
                carry_scale = self.boundary_gate(
                    inputs[:, step_idx],
                    metadata[:, step_idx],
                    boundary_flags[:, step_idx],
                )
            out_t, state = self.cell(
                inputs[:, step_idx],
                state,
                conditioning=metadata[:, step_idx],
                carry_scale=carry_scale,
            )
            outputs.append(out_t)
        return torch.stack(outputs, dim=1), state


class SupportSetStateEncoder(nn.Module):
    """Encode support shots into class-conditioned episodic states."""

    def __init__(
        self,
        dim: int,
        state_dim: int,
        max_positions: int = 16,
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
        self.state_to_class = nn.Linear(state_dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, shot_descriptors: torch.Tensor, class_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if shot_descriptors.dim() == 2:
            shot_descriptors = shot_descriptors.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        batch_size, shot_num, dim = shot_descriptors.shape
        device = shot_descriptors.device
        if self.use_role_embedding and self.role_embedding is not None:
            role_ids = torch.zeros(batch_size, shot_num, dtype=torch.long, device=device)
            phase_ids = torch.zeros(batch_size, shot_num, dtype=torch.long, device=device)
            position_ids = torch.arange(shot_num, device=device).view(1, shot_num).expand(batch_size, -1)
            class_ids = torch.full((batch_size, shot_num), class_index, dtype=torch.long, device=device)
            boundary_ids = torch.zeros(batch_size, shot_num, dtype=torch.long, device=device)
            boundary_ids[:, 0] = 1
            metadata = self.role_embedding(role_ids, phase_ids, position_ids, class_ids, boundary_ids)
        else:
            metadata = torch.zeros(batch_size, shot_num, dim, device=device, dtype=shot_descriptors.dtype)
            boundary_ids = torch.zeros(batch_size, shot_num, dtype=torch.long, device=device)
            boundary_ids[:, 0] = 1

        refined, final_state = self.scan(
            shot_descriptors,
            metadata,
            boundary_flags=boundary_ids,
        )
        class_state = self.norm(self.state_to_class(final_state) + refined[:, -1])
        if squeeze_batch:
            return class_state.squeeze(0), refined.squeeze(0)
        return class_state, refined
