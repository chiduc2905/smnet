"""Trajectory-level transport scoring head for transport-prior few-shot models."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.metrics.sliced_wasserstein import SlicedWassersteinDistance


class TrajectoryTransportHead(nn.Module):
    """Score each class using final-state cosine and trajectory-to-prior transport."""

    def __init__(
        self,
        dim: int,
        temperature: float = 16.0,
        transport_weight: float = 8.0,
        confidence_weight: float = 0.5,
        num_projections: int = 32,
        p: float = 2.0,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.transport_weight = float(transport_weight)
        self.confidence_weight = float(confidence_weight)
        self.fusion = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)
        self.sw_distance = SlicedWassersteinDistance(
            num_projections=num_projections,
            p=p,
            reduction="none",
            normalize_inputs=True,
        )

    def forward(
        self,
        query_readouts: torch.Tensor,
        query_trajectories: torch.Tensor,
        prior_atoms: torch.Tensor,
        transport_confidence: torch.Tensor | None = None,
        return_components: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if query_readouts.dim() != 3:
            raise ValueError(
                "query_readouts must have shape (NQ, Way, Dim), "
                f"got {tuple(query_readouts.shape)}"
            )
        if query_trajectories.dim() != 4:
            raise ValueError(
                "query_trajectories must have shape (NQ, Way, Tokens, Dim), "
                f"got {tuple(query_trajectories.shape)}"
            )
        if prior_atoms.dim() != 3:
            raise ValueError(
                "prior_atoms must have shape (Way, Atoms, Dim), "
                f"got {tuple(prior_atoms.shape)}"
            )

        prior_summary = prior_atoms.mean(dim=1)
        prior_expanded = prior_summary.unsqueeze(0).expand(query_readouts.shape[0], -1, -1)
        fused = self.fusion(torch.cat([query_readouts, prior_expanded], dim=-1))
        fused = self.norm(fused + query_readouts)
        cosine = self.temperature * torch.einsum(
            "nwd,wd->nw",
            F.normalize(fused, p=2, dim=-1),
            F.normalize(prior_summary, p=2, dim=-1),
        )

        prior_atoms_expanded = prior_atoms.unsqueeze(0).expand(query_trajectories.shape[0], -1, -1, -1)
        transport = self.sw_distance(query_trajectories, prior_atoms_expanded, reduction="none")

        if transport_confidence is None:
            confidence_bias = torch.zeros_like(cosine)
        else:
            confidence_bias = transport_confidence.view(1, -1).expand_as(cosine)

        logits = cosine - self.transport_weight * transport + self.confidence_weight * confidence_bias
        if not return_components:
            return logits
        return logits, {
            "cosine_logits": cosine,
            "transport_logits": -self.transport_weight * transport,
            "confidence_bias": self.confidence_weight * confidence_bias,
        }
