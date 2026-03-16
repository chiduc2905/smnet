"""2D token serialization utilities for transport-evidence models."""

from __future__ import annotations

from typing import Iterable, Tuple

import torch


SUPPORTED_SERIALIZATION_ORDERS = (
    "row_major",
    "row_major_reverse",
    "column_major",
    "column_major_reverse",
)


def _parse_order_names(order_names: str | Iterable[str] | None) -> Tuple[str, ...]:
    if order_names is None:
        return SUPPORTED_SERIALIZATION_ORDERS
    if isinstance(order_names, str):
        parsed = tuple(part.strip() for part in order_names.split(",") if part.strip())
    else:
        parsed = tuple(str(part).strip() for part in order_names if str(part).strip())
    if not parsed:
        raise ValueError("At least one serialization order must be provided")
    invalid = [order for order in parsed if order not in SUPPORTED_SERIALIZATION_ORDERS]
    if invalid:
        raise ValueError(
            f"Unsupported serialization orders: {invalid}. "
            f"Supported={SUPPORTED_SERIALIZATION_ORDERS}"
        )
    return parsed


class TokenGridSerializer:
    """Serialize a `(H, W)` token grid using a fixed set of generic scan orders."""

    def __init__(self, order_names: str | Iterable[str] | None = None) -> None:
        self.order_names = _parse_order_names(order_names)

    @staticmethod
    def _build_index(order_name: str, height: int, width: int, device: torch.device) -> torch.Tensor:
        base = torch.arange(height * width, device=device).reshape(height, width)
        if order_name == "row_major":
            return base.reshape(-1)
        if order_name == "row_major_reverse":
            return torch.flip(base.reshape(-1), dims=[0])
        if order_name == "column_major":
            return base.transpose(0, 1).reshape(-1)
        if order_name == "column_major_reverse":
            return torch.flip(base.transpose(0, 1).reshape(-1), dims=[0])
        raise ValueError(f"Unsupported serialization order: {order_name}")

    def serialize(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Return serialized tokens stacked as `(Orders, ..., Tokens, Dim)`."""
        if tokens.dim() < 3:
            raise ValueError(f"Expected tokens with shape (..., Tokens, Dim), got {tuple(tokens.shape)}")
        token_count = tokens.shape[-2]
        if token_count != height * width:
            raise ValueError(
                f"Token count {token_count} does not match spatial grid {height}x{width}"
            )
        serialized = []
        for order_name in self.order_names:
            index = self._build_index(order_name, height, width, tokens.device)
            serialized.append(tokens.index_select(dim=-2, index=index))
        return torch.stack(serialized, dim=0)
