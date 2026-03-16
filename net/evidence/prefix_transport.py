"""Prefix transport evidence builder for TEM-Mamba."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.metrics.sliced_wasserstein import SlicedWassersteinDistance


class PrefixTransportEvidenceBuilder(nn.Module):
    """Convert support-query transport into a classwise evidence sequence."""

    def __init__(
        self,
        token_dim: int,
        evidence_dim: int | None = None,
        num_projections: int = 64,
        p: float = 2.0,
        normalize_inputs: bool = True,
        use_transport_metrics: bool = True,
        use_delta: bool = True,
        use_support_context: bool = True,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.evidence_dim = evidence_dim or token_dim
        self.use_transport_metrics = bool(use_transport_metrics)
        self.use_delta = bool(use_delta)
        self.use_support_context = bool(use_support_context)
        if not self.use_transport_metrics and not self.use_support_context:
            raise ValueError("TEM-Mamba needs support context or transport metrics to stay class-conditional")
        self.sw_distance = SlicedWassersteinDistance(
            num_projections=num_projections,
            p=p,
            reduction="none",
            normalize_inputs=normalize_inputs,
        )

        self.query_proj = nn.Linear(token_dim, self.evidence_dim)
        self.context_proj = nn.Linear(token_dim, self.evidence_dim) if self.use_support_context else None
        if self.use_transport_metrics:
            scalar_dim = 2 if self.use_delta else 1
            self.transport_proj = nn.Sequential(
                nn.Linear(scalar_dim, self.evidence_dim),
                nn.GELU(),
                nn.Linear(self.evidence_dim, self.evidence_dim),
            )
        else:
            self.transport_proj = None
        self.mix_proj = nn.Linear(self.evidence_dim, self.evidence_dim)
        self.output_norm = nn.LayerNorm(self.evidence_dim)

    def _compute_prefix_transport(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute classwise prefix SW distances shaped `(NQ, Way, Tokens)`."""
        num_query, token_count, dim = query_tokens.shape
        way_num, support_token_count, support_dim = support_tokens.shape
        if support_dim != dim:
            raise ValueError(
                f"Support/query dims must match, got query={dim} support={support_dim}"
            )

        support_expanded = support_tokens.unsqueeze(0).expand(num_query, -1, -1, -1)
        support_expanded = support_expanded.reshape(num_query * way_num, support_token_count, dim)

        prefix_distances = []
        for step_idx in range(token_count):
            prefix = query_tokens[:, : step_idx + 1, :]
            prefix = prefix.unsqueeze(1).expand(-1, way_num, -1, -1)
            prefix = prefix.reshape(num_query * way_num, step_idx + 1, dim)
            distances_t = self.sw_distance(prefix, support_expanded, reduction="none")
            prefix_distances.append(distances_t.reshape(num_query, way_num))
        return torch.stack(prefix_distances, dim=-1)

    def forward(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if query_tokens.dim() != 3:
            raise ValueError(
                f"query_tokens must have shape (NQ, Tokens, Dim), got {tuple(query_tokens.shape)}"
            )
        if support_tokens.dim() != 3:
            raise ValueError(
                f"support_tokens must have shape (Way, Tokens, Dim), got {tuple(support_tokens.shape)}"
            )

        num_query, token_count, _ = query_tokens.shape
        way_num = support_tokens.shape[0]
        support_summary = support_tokens.mean(dim=1)
        if self.use_transport_metrics:
            prefix_transport = self._compute_prefix_transport(query_tokens, support_tokens)
            previous = torch.cat([prefix_transport[:, :, :1], prefix_transport[:, :, :-1]], dim=-1)
            transport_delta = previous - prefix_transport
            transport_delta[:, :, 0] = 0.0
        else:
            prefix_transport = query_tokens.new_zeros(num_query, way_num, token_count)
            transport_delta = query_tokens.new_zeros(num_query, way_num, token_count)

        evidence = self.query_proj(query_tokens).unsqueeze(1).expand(-1, way_num, -1, -1)
        if self.context_proj is not None:
            context = self.context_proj(support_summary).unsqueeze(0).unsqueeze(2)
            evidence = evidence + context.expand(num_query, -1, token_count, -1)

        if self.transport_proj is not None:
            scalar_features = [prefix_transport.unsqueeze(-1)]
            if self.use_delta:
                scalar_features.append(transport_delta.unsqueeze(-1))
            scalar_features_t = torch.cat(scalar_features, dim=-1)
            evidence = evidence + self.transport_proj(scalar_features_t)
        evidence = self.output_norm(self.mix_proj(F.gelu(evidence)))

        return evidence, {
            "prefix_transport": prefix_transport,
            "transport_delta": transport_delta,
            "support_summary": support_summary,
        }
