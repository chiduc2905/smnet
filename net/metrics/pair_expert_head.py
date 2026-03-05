"""Pair Expert Correction head for confusing class pairs.

This module adds lightweight binary experts to refine logits near confusing
boundaries without increasing backbone capacity.
"""
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairExpertCorrectionHead(nn.Module):
    """Lightweight expert-based logit correction for predefined class pairs.

    Args:
        embed_dim: Input embedding dimension.
        pairs: Sequence of (class_a, class_b). Positive target is class_b.
        delta_max: Max correction magnitude after tanh.
        conf_threshold: Apply correction when max probability is below threshold.
    """

    def __init__(
        self,
        embed_dim: int,
        pairs: Sequence[Tuple[int, int]] = ((0, 3), (1, 2)),
        delta_max: float = 0.5,
        conf_threshold: float = 0.60,
    ):
        super().__init__()
        self.pairs = tuple((int(a), int(b)) for a, b in pairs)
        self.delta_max = float(delta_max)
        self.conf_threshold = float(conf_threshold)

        hidden = max(8, embed_dim // 4)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dim, hidden, bias=False),
                    nn.GELU(),
                    nn.Linear(hidden, 1, bias=True),
                )
                for _ in self.pairs
            ]
        )

    def forward(
        self,
        query_embed: torch.Tensor,
        scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """Apply expert corrections.

        Args:
            query_embed: (N, D) normalized query embeddings.
            scores: (N, Way) logits before correction.

        Returns:
            corrected_scores: (N, Way)
            aux: dict with expert_logits and gate masks per pair.
        """
        corrected = scores.clone()
        probs = torch.softmax(scores, dim=1)
        max_prob = probs.max(dim=1).values
        top2_idx = probs.topk(k=min(2, probs.shape[1]), dim=1).indices

        expert_logits: List[torch.Tensor] = []
        gate_masks: List[torch.Tensor] = []

        for expert, (class_a, class_b) in zip(self.experts, self.pairs):
            logits = expert(query_embed).squeeze(-1)  # (N,)
            delta = self.delta_max * torch.tanh(logits)  # bounded correction

            if top2_idx.shape[1] == 1:
                top2_mask = (top2_idx[:, 0] == class_a) | (top2_idx[:, 0] == class_b)
            else:
                top2_mask = (
                    ((top2_idx[:, 0] == class_a) | (top2_idx[:, 0] == class_b))
                    & ((top2_idx[:, 1] == class_a) | (top2_idx[:, 1] == class_b))
                )
            uncertain_mask = max_prob < self.conf_threshold
            gate = top2_mask | uncertain_mask

            if class_a < corrected.shape[1] and class_b < corrected.shape[1]:
                corrected[gate, class_a] = corrected[gate, class_a] - delta[gate]
                corrected[gate, class_b] = corrected[gate, class_b] + delta[gate]

            expert_logits.append(logits)
            gate_masks.append(gate)

        aux = {
            "expert_logits": expert_logits,
            "gates": gate_masks,
            "pairs": self.pairs,
        }
        return corrected, aux

    def compute_loss(self, expert_logits: Iterable[torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        """Binary expert supervision on samples that belong to each pair."""
        losses = []
        for logits, (class_a, class_b) in zip(expert_logits, self.pairs):
            mask = (targets == class_a) | (targets == class_b)
            if mask.any():
                pair_targets = (targets[mask] == class_b).float()
                losses.append(F.binary_cross_entropy_with_logits(logits[mask], pair_targets))

        if not losses:
            return targets.new_zeros((), dtype=torch.float32)
        return torch.stack(losses).mean()

