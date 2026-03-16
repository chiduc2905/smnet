"""Support atomization for transport-prior few-shot reasoning."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.ssm.common import SelectiveStateSpaceCell, run_selective_scan


class SupportAtomizer(nn.Module):
    """Turn one support token set into a small set of support atoms."""

    def __init__(
        self,
        dim: int,
        num_atoms: int = 4,
        state_dim: int = 16,
        max_tokens: int = 64,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_atoms = int(num_atoms)
        self.position_embedding = nn.Embedding(max_tokens, dim)
        self.local_dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.local_pw = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.scan_cell = SelectiveStateSpaceCell(dim, state_dim)
        self.atom_queries = nn.Parameter(torch.randn(self.num_atoms, dim) * 0.02)
        self.token_norm = nn.LayerNorm(dim)
        self.atom_norm = nn.LayerNorm(dim)
        self.confidence_head = nn.Linear(dim, 1)
        self.scale = dim ** -0.5

    def _local_mix(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, token_num, dim = tokens.shape
        side = int(math.isqrt(token_num))
        if side * side != token_num:
            return tokens
        x = tokens.transpose(1, 2).reshape(batch_size, dim, side, side)
        x = x + self.local_pw(self.local_dw(x))
        return x.flatten(2).transpose(1, 2).contiguous()

    def forward(self, support_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if support_tokens.dim() < 3:
            raise ValueError(
                "support_tokens must have shape (..., Tokens, Dim), "
                f"got {tuple(support_tokens.shape)}"
            )
        leading_shape = support_tokens.shape[:-2]
        token_num = support_tokens.shape[-2]
        dim = support_tokens.shape[-1]
        flat_tokens = support_tokens.reshape(-1, token_num, dim)

        pos_ids = torch.arange(token_num, device=flat_tokens.device)
        hidden = flat_tokens + self.position_embedding(pos_ids).unsqueeze(0)
        hidden = self._local_mix(hidden)
        hidden, _ = run_selective_scan(self.scan_cell, hidden)
        hidden = self.token_norm(hidden + flat_tokens)

        queries = self.atom_queries.unsqueeze(0).expand(flat_tokens.shape[0], -1, -1)
        attention = torch.matmul(queries, hidden.transpose(1, 2)) * self.scale
        attention = torch.softmax(attention, dim=-1)
        atoms = torch.matmul(attention, hidden)
        atoms = self.atom_norm(atoms + queries)
        confidence = torch.sigmoid(self.confidence_head(atoms).squeeze(-1))

        atoms = atoms.reshape(*leading_shape, self.num_atoms, dim)
        confidence = confidence.reshape(*leading_shape, self.num_atoms)
        return atoms, confidence
