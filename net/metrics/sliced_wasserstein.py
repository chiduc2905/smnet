"""Reusable sliced Wasserstein distance for token-level few-shot matching."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.fewshot_common import feature_map_to_tokens, merge_support_tokens


def prepare_token_distribution(tokens: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    """Prepare a token distribution tensor for SW computation.

    Accepted shapes:
        - `(..., Tokens, Dim)` token sets
        - `(..., Dim, H, W)` feature maps, which are flattened to tokens

    Returns:
        Tensor shaped `(..., Tokens, Dim)`.
    """
    if tokens.dim() < 3:
        raise ValueError(f"Expected at least 3 dimensions, got shape={tuple(tokens.shape)}")
    if tokens.dim() == 4 and tokens.shape[1] > tokens.shape[2] and tokens.shape[1] > tokens.shape[3]:
        tokens = feature_map_to_tokens(tokens)
    if normalize:
        tokens = F.normalize(tokens, p=2, dim=-1)
    return tokens


def merge_support_tokens_by_class(support_tokens: torch.Tensor, merge_mode: str = "concat") -> torch.Tensor:
    """Merge support tokens from `(Way, Shot, Tokens, Dim)` to class-level sets."""
    return merge_support_tokens(support_tokens, merge_mode=merge_mode)


class SlicedWassersteinDistance(nn.Module):
    """Compute sliced Wasserstein distances between token sets.

    Expected shapes:
        - `query_tokens`: `(..., Tq, D)`
        - `support_tokens`: `(..., Ts, D)`

    The leading dimensions must match. The module returns a tensor shaped
    `(...)` when `reduction="none"`, otherwise a scalar reduced over the
    leading dimensions.
    """

    def __init__(
        self,
        num_projections: int = 64,
        p: float = 2.0,
        reduction: str = "mean",
        normalize_inputs: bool = True,
        projection_seed: int = 7,
    ) -> None:
        super().__init__()
        if num_projections <= 0:
            raise ValueError("num_projections must be positive")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.num_projections = int(num_projections)
        self.p = float(p)
        self.reduction = reduction
        self.normalize_inputs = bool(normalize_inputs)
        self.projection_seed = int(projection_seed)
        self.register_buffer("_projection_bank", torch.empty(0), persistent=False)
        self._projection_dim: Optional[int] = None

    def _build_projection_bank(self, feature_dim: int) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.projection_seed)
        bank = torch.randn(feature_dim, self.num_projections, generator=generator)
        bank = F.normalize(bank, p=2, dim=0)
        self._projection_dim = feature_dim
        self._projection_bank = bank
        return bank

    def _get_projection_bank(self, feature_dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._projection_dim != feature_dim or self._projection_bank.numel() == 0:
            self._build_projection_bank(feature_dim)
        return self._projection_bank.to(device=device, dtype=dtype)

    @staticmethod
    def _resample_sorted_values(sorted_values: torch.Tensor, target_tokens: int) -> torch.Tensor:
        """Resample sorted projection values to a shared token count."""
        current_tokens = sorted_values.shape[-2]
        if current_tokens == target_tokens:
            return sorted_values
        if current_tokens == 1:
            return sorted_values.expand(*sorted_values.shape[:-2], target_tokens, sorted_values.shape[-1])

        device = sorted_values.device
        dtype = sorted_values.dtype
        grid = torch.linspace(0, current_tokens - 1, target_tokens, device=device, dtype=dtype)
        left = grid.floor().long()
        right = grid.ceil().long()
        alpha = (grid - left.to(dtype)).view(*([1] * (sorted_values.dim() - 2)), target_tokens, 1)
        left_vals = torch.index_select(sorted_values, dim=-2, index=left)
        right_vals = torch.index_select(sorted_values, dim=-2, index=right)
        return left_vals * (1.0 - alpha) + right_vals * alpha

    def forward(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        reduction: Optional[str] = None,
    ) -> torch.Tensor:
        reduction = reduction or self.reduction
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unsupported reduction: {reduction}")

        query_tokens = prepare_token_distribution(query_tokens, normalize=self.normalize_inputs)
        support_tokens = prepare_token_distribution(support_tokens, normalize=self.normalize_inputs)

        if query_tokens.shape[:-2] != support_tokens.shape[:-2]:
            raise ValueError(
                "Leading dimensions must match for SW computation: "
                f"query={tuple(query_tokens.shape)} support={tuple(support_tokens.shape)}"
            )
        if query_tokens.shape[-1] != support_tokens.shape[-1]:
            raise ValueError(
                "Token feature dimensions must match: "
                f"query={query_tokens.shape[-1]} support={support_tokens.shape[-1]}"
            )

        feature_dim = query_tokens.shape[-1]
        projections = self._get_projection_bank(feature_dim, query_tokens.device, query_tokens.dtype)
        query_proj = torch.matmul(query_tokens, projections)
        support_proj = torch.matmul(support_tokens, projections)

        query_sorted, _ = torch.sort(query_proj, dim=-2)
        support_sorted, _ = torch.sort(support_proj, dim=-2)

        target_tokens = max(query_sorted.shape[-2], support_sorted.shape[-2])
        query_sorted = self._resample_sorted_values(query_sorted, target_tokens)
        support_sorted = self._resample_sorted_values(support_sorted, target_tokens)

        diff = torch.abs(query_sorted - support_sorted).pow(self.p)
        projected = diff.mean(dim=-2)
        if self.p != 1.0:
            projected = projected.clamp_min(0.0).pow(1.0 / self.p)
        distances = projected.mean(dim=-1)

        if reduction == "none":
            return distances
        if reduction == "sum":
            return distances.sum()
        return distances.mean()
