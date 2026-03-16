"""Classification head for transport-evidence readouts."""

from __future__ import annotations

import torch
import torch.nn as nn


class TransportEvidenceClassificationHead(nn.Module):
    """Shared scalar scorer for classwise evidence readouts."""

    def __init__(self, dim: int, temperature: float = 16.0) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.norm = nn.LayerNorm(dim)
        self.scorer = nn.Linear(dim, 1)

    def forward(self, class_readouts: torch.Tensor) -> torch.Tensor:
        if class_readouts.dim() != 3:
            raise ValueError(
                "class_readouts must have shape (Queries, Way, Dim), "
                f"got {tuple(class_readouts.shape)}"
            )
        logits = self.scorer(self.norm(class_readouts)).squeeze(-1)
        return self.temperature * logits
