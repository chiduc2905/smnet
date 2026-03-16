"""Replay-controlled query reader for transport-prior few-shot models."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from net.metrics.sliced_wasserstein import SlicedWassersteinDistance
from net.ssm.common import SelectiveStateSpaceCell


class ReplayQueryReader(nn.Module):
    """Read query tokens while repeatedly re-anchoring the state to a class prior."""

    def __init__(
        self,
        dim: int,
        state_dim: int,
        num_projections: int = 32,
        p: float = 2.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.cell = SelectiveStateSpaceCell(dim, state_dim)
        self.prior_to_state = nn.Linear(dim, state_dim)
        self.prior_to_condition = nn.Linear(dim, dim)
        self.query_global_proj = nn.Linear(dim, dim)
        self.prior_key = nn.Linear(dim, dim, bias=False)
        self.prior_value = nn.Linear(dim, dim, bias=False)
        self.carry_gate = nn.Linear(dim * 3 + 2, state_dim)
        self.replay_gate = nn.Linear(dim * 3 + 2, 1)
        self.replay_state = nn.Linear(dim, state_dim)
        self.replay_token = nn.Linear(dim, dim)
        self.output_norm = nn.LayerNorm(dim)
        self.readout_fusion = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.readout_norm = nn.LayerNorm(dim)
        self.sw_distance = SlicedWassersteinDistance(
            num_projections=num_projections,
            p=p,
            reduction="none",
            normalize_inputs=True,
        )
        self.scale = dim ** -0.5

    def _expand_prior(self, query_tokens: torch.Tensor, prior_atoms: torch.Tensor) -> torch.Tensor:
        if prior_atoms.dim() == 2:
            return prior_atoms.unsqueeze(0).expand(query_tokens.shape[0], -1, -1)
        if prior_atoms.dim() == 3 and prior_atoms.shape[0] == query_tokens.shape[0]:
            return prior_atoms
        raise ValueError(
            "prior_atoms must have shape (Atoms, Dim) or (NQ, Atoms, Dim), "
            f"got {tuple(prior_atoms.shape)}"
        )

    def _expand_confidence(self, query_tokens: torch.Tensor, confidence: torch.Tensor | float | None) -> torch.Tensor:
        if confidence is None:
            return torch.zeros(query_tokens.shape[0], 1, device=query_tokens.device, dtype=query_tokens.dtype)
        if isinstance(confidence, float):
            return torch.full(
                (query_tokens.shape[0], 1),
                float(confidence),
                device=query_tokens.device,
                dtype=query_tokens.dtype,
            )
        if confidence.dim() == 0:
            return confidence.view(1, 1).expand(query_tokens.shape[0], -1).to(query_tokens.dtype)
        if confidence.dim() == 1 and confidence.shape[0] == query_tokens.shape[0]:
            return confidence.unsqueeze(-1).to(query_tokens.dtype)
        raise ValueError(
            "transport confidence must be scalar or (NQ,), "
            f"got {tuple(confidence.shape)}"
        )

    def _prior_context(self, x_t: torch.Tensor, prior_atoms: torch.Tensor) -> torch.Tensor:
        attention = torch.matmul(x_t.unsqueeze(1), self.prior_key(prior_atoms).transpose(1, 2)) * self.scale
        attention = torch.softmax(attention, dim=-1)
        return torch.matmul(attention, self.prior_value(prior_atoms)).squeeze(1)

    def forward(
        self,
        query_tokens: torch.Tensor,
        prior_atoms: torch.Tensor,
        query_global: torch.Tensor | None = None,
        transport_confidence: torch.Tensor | float | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if query_tokens.dim() != 3:
            raise ValueError(
                "query_tokens must have shape (NQ, Tokens, Dim), "
                f"got {tuple(query_tokens.shape)}"
            )
        prior_atoms = self._expand_prior(query_tokens, prior_atoms)
        prior_summary = prior_atoms.mean(dim=1)
        confidence = self._expand_confidence(query_tokens, transport_confidence)
        state = self.prior_to_state(prior_summary) * (1.0 + confidence)

        if query_global is not None:
            query_bias = self.query_global_proj(query_global)
        else:
            query_bias = torch.zeros_like(prior_summary)

        outputs = []
        replay_gates = []
        prefix_sw = []
        for step_idx in range(query_tokens.shape[1]):
            x_t = query_tokens[:, step_idx]
            prior_ctx = self._prior_context(x_t, prior_atoms)
            prefix_tokens = query_tokens[:, : step_idx + 1]
            sw_residual = self.sw_distance(prefix_tokens, prior_atoms, reduction="none")
            gate_input = torch.cat([x_t, prior_ctx, prior_summary, sw_residual.unsqueeze(-1), confidence], dim=-1)
            carry_scale = torch.sigmoid(self.carry_gate(gate_input))

            conditioning = self.prior_to_condition(prior_ctx + prior_summary + query_bias)
            out_t, state = self.cell(
                x_t,
                state,
                conditioning=conditioning,
                carry_scale=carry_scale,
            )

            replay_gate = torch.sigmoid(self.replay_gate(gate_input))
            state = state + replay_gate * self.replay_state(prior_summary)
            out_t = self.output_norm(out_t + replay_gate * self.replay_token(prior_ctx))

            outputs.append(out_t)
            replay_gates.append(replay_gate.squeeze(-1))
            prefix_sw.append(sw_residual)

        refined_query = torch.stack(outputs, dim=1)
        readout = self.readout_fusion(torch.cat([refined_query.mean(dim=1), prior_summary], dim=-1))
        readout = self.readout_norm(readout + refined_query.mean(dim=1))
        if query_global is not None:
            readout = self.readout_norm(readout + query_global)
        return readout, refined_query, {
            "replay_gates": torch.stack(replay_gates, dim=1),
            "prefix_sw": torch.stack(prefix_sw, dim=1),
        }
