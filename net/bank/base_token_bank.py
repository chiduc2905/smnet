"""Learnable token bank used for transport prior calibration."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseTokenBank(nn.Module):
    """Retrieve a small set of learnable prior atom banks for each support class."""

    def __init__(
        self,
        dim: int,
        num_entries: int = 16,
        atoms_per_entry: int = 4,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_entries = int(num_entries)
        self.atoms_per_entry = int(atoms_per_entry)
        self.bank_atoms = nn.Parameter(torch.randn(self.num_entries, self.atoms_per_entry, dim) * 0.02)
        self.bank_strength = nn.Parameter(torch.zeros(self.num_entries))
        self.atom_norm = nn.LayerNorm(dim)

    def retrieve(
        self,
        query_vectors: torch.Tensor,
        topk: int = 4,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if query_vectors.dim() == 1:
            query_vectors = query_vectors.unsqueeze(0)
            squeeze_batch = True
        elif query_vectors.dim() == 2:
            squeeze_batch = False
        else:
            raise ValueError(
                "query_vectors must have shape (Dim,) or (Batch, Dim), "
                f"got {tuple(query_vectors.shape)}"
            )

        topk = max(1, min(int(topk), self.num_entries))
        bank_atoms = self.atom_norm(self.bank_atoms)
        bank_summaries = F.normalize(bank_atoms.mean(dim=1), p=2, dim=-1)
        query_norm = F.normalize(query_vectors, p=2, dim=-1)

        similarities = torch.matmul(query_norm, bank_summaries.transpose(0, 1))
        top_scores, top_indices = torch.topk(similarities, k=topk, dim=-1)
        retrieval_weights = torch.softmax(top_scores, dim=-1)

        gathered = bank_atoms.index_select(0, top_indices.reshape(-1))
        gathered = gathered.view(query_vectors.shape[0], topk, self.atoms_per_entry, self.dim)
        reliability = torch.sigmoid(self.bank_strength.index_select(0, top_indices.reshape(-1)))
        reliability = reliability.view(query_vectors.shape[0], topk)

        if squeeze_batch:
            return (
                gathered.squeeze(0),
                retrieval_weights.squeeze(0),
                top_indices.squeeze(0),
                top_scores.squeeze(0),
                reliability.squeeze(0),
            )
        return gathered, retrieval_weights, top_indices, top_scores, reliability
