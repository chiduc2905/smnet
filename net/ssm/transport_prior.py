"""Transport-based class prior calibration for few-shot support sets."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from net.bank.base_token_bank import BaseTokenBank
from net.metrics.sliced_wasserstein import SlicedWassersteinDistance


class TransportPriorCalibrator(nn.Module):
    """Build a calibrated class prior from raw support atoms and a learned prior bank."""

    def __init__(
        self,
        dim: int,
        num_prior_atoms: int = 4,
        bank_size: int = 16,
        bank_atoms_per_entry: int = 4,
        topk: int = 4,
        num_projections: int = 32,
        p: float = 2.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_prior_atoms = int(num_prior_atoms)
        self.topk = int(topk)
        self.bank = BaseTokenBank(dim, num_entries=bank_size, atoms_per_entry=bank_atoms_per_entry)
        self.prior_queries = nn.Parameter(torch.randn(self.num_prior_atoms, dim) * 0.02)
        self.support_key = nn.Linear(dim, dim, bias=False)
        self.support_value = nn.Linear(dim, dim, bias=False)
        self.bank_key = nn.Linear(dim, dim, bias=False)
        self.bank_value = nn.Linear(dim, dim, bias=False)
        self.alpha_gate = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )
        self.confidence_gate = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )
        self.prior_norm = nn.LayerNorm(dim)
        self.sw_distance = SlicedWassersteinDistance(
            num_projections=num_projections,
            p=p,
            reduction="none",
            normalize_inputs=True,
        )
        self.scale = dim ** -0.5

    def _cross_attend(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        attention = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        attention = torch.softmax(attention, dim=-1)
        return torch.matmul(attention, values)

    def forward(self, class_atoms: torch.Tensor) -> Dict[str, torch.Tensor]:
        if class_atoms.dim() != 3:
            raise ValueError(
                "class_atoms must have shape (Classes, Tokens, Dim), "
                f"got {tuple(class_atoms.shape)}"
            )
        class_summary = class_atoms.mean(dim=1)
        retrieved_atoms, retrieval_weights, retrieval_indices, top_scores, reliability = self.bank.retrieve(
            class_summary,
            topk=self.topk,
        )

        bank_weights = retrieval_weights * reliability
        bank_weights = bank_weights / bank_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        weighted_bank_atoms = (retrieved_atoms * bank_weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        bank_summary = weighted_bank_atoms.mean(dim=1)

        queries = self.prior_queries.unsqueeze(0).expand(class_atoms.shape[0], -1, -1)
        support_prior = self._cross_attend(
            queries,
            self.support_key(class_atoms),
            self.support_value(class_atoms),
        )
        bank_prior = self._cross_attend(
            queries,
            self.bank_key(weighted_bank_atoms),
            self.bank_value(weighted_bank_atoms),
        )

        alpha_input = torch.cat([class_summary, bank_summary, class_summary - bank_summary], dim=-1)
        alpha = torch.sigmoid(self.alpha_gate(alpha_input))
        prior_atoms = alpha.unsqueeze(-1) * support_prior + (1.0 - alpha).unsqueeze(-1) * bank_prior
        prior_atoms = self.prior_norm(prior_atoms + queries)

        raw_bank_sw = self.sw_distance(class_atoms, weighted_bank_atoms, reduction="none")
        prior_support_sw = self.sw_distance(prior_atoms, class_atoms, reduction="none")
        confidence_input = torch.stack(
            [
                top_scores.mean(dim=-1),
                -raw_bank_sw,
                -prior_support_sw,
            ],
            dim=-1,
        )
        transport_confidence = torch.sigmoid(self.confidence_gate(confidence_input)).squeeze(-1)

        return {
            "prior_atoms": prior_atoms,
            "support_prior_atoms": support_prior,
            "bank_prior_atoms": bank_prior,
            "weighted_bank_atoms": weighted_bank_atoms,
            "transport_confidence": transport_confidence,
            "retrieval_weights": retrieval_weights,
            "retrieval_indices": retrieval_indices,
            "raw_bank_sw": raw_bank_sw,
            "prior_support_sw": prior_support_sw,
        }
