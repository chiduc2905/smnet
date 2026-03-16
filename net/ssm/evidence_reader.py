"""Selective state-space evidence reader for TEM-Mamba."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from net.ssm.common import SelectiveStateSpaceCell, run_selective_scan


class TransportEvidenceReader(nn.Module):
    """Integrate classwise transport evidence with a shared selective scan."""

    def __init__(
        self,
        dim: int,
        state_dim: int = 16,
        depth: int = 1,
        readout_mode: str = "final",
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be positive")
        if readout_mode not in {"final", "mean"}:
            raise ValueError(f"Unsupported readout_mode: {readout_mode}")
        self.readout_mode = readout_mode
        self.layers = nn.ModuleList(
            SelectiveStateSpaceCell(dim, state_dim) for _ in range(depth)
        )
        self.readout_norm = nn.LayerNorm(dim)

    def forward(self, evidence_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, List[torch.Tensor]]]:
        if evidence_tokens.dim() != 3:
            raise ValueError(
                f"evidence_tokens must have shape (Batch, Tokens, Dim), got {tuple(evidence_tokens.shape)}"
            )

        hidden = evidence_tokens
        states: List[torch.Tensor] = []
        for layer in self.layers:
            hidden, state = run_selective_scan(layer, hidden)
            states.append(state)

        if self.readout_mode == "final":
            readout = hidden[:, -1]
        else:
            readout = hidden.mean(dim=1)
        readout = self.readout_norm(readout)
        return readout, hidden, {"layer_states": states}
