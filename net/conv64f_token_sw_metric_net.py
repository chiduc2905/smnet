"""Conv64F token-distribution metric baseline using sliced Wasserstein distance."""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from net.fewshot_common import BaseConv64FewShotModel, feature_map_to_tokens, pooled_episode_features
from net.heads.token_sw_head import (
    GlobalFeatureDistanceHead,
    TokenSetProjector,
    TokenSWClassificationHead,
)
from net.metrics.sliced_wasserstein import SlicedWassersteinDistance


class Conv64FTokenSWMetricNet(BaseConv64FewShotModel):
    """Token-level SW baseline on top of a shared Conv64F backbone.

    SW is the primary metric. Global pooled distance is optional and controlled
    by `token_metric_mode`.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 64,
        token_dim: int | None = None,
        temperature: float = 16.0,
        conv64f_pool_last: bool = True,
        sw_num_projections: int = 64,
        sw_p: float = 2.0,
        sw_normalize: bool = True,
        token_merge_mode: str = "concat",
        token_metric_mode: str = "token_only",
        global_metric: str = "cosine",
        global_metric_weight: float = 1.0,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            conv64f_pool_last=conv64f_pool_last,
        )
        if token_metric_mode not in {"token_only", "token_plus_global"}:
            raise ValueError(f"Unsupported token_metric_mode: {token_metric_mode}")
        token_dim = token_dim or hidden_dim
        self.token_metric_mode = token_metric_mode
        self.token_projector = TokenSetProjector(hidden_dim, token_dim)
        self.sw_distance = SlicedWassersteinDistance(
            num_projections=sw_num_projections,
            p=sw_p,
            reduction="none",
            normalize_inputs=sw_normalize,
        )
        global_head = None
        if token_metric_mode == "token_plus_global":
            global_head = GlobalFeatureDistanceHead(metric=global_metric, temperature=temperature)
        self.classification_head = TokenSWClassificationHead(
            sw_distance=self.sw_distance,
            merge_mode=token_merge_mode,
            token_scale=temperature,
            global_head=global_head,
            global_weight=global_metric_weight,
        )

    def _encode_episode(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        way_num, shot_num = support.shape[:2]
        q_features = self.encode(query)
        s_features = self.encode(support.reshape(way_num * shot_num, *support.shape[-3:]))
        q_tokens = self.token_projector(feature_map_to_tokens(q_features))
        s_tokens = self.token_projector(feature_map_to_tokens(s_features)).reshape(
            way_num,
            shot_num,
            -1,
            q_tokens.shape[-1],
        )
        q_global = pooled_episode_features(q_features)
        s_global = pooled_episode_features(s_features).reshape(way_num, shot_num, -1)
        return q_tokens, s_tokens, q_global, s_global

    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bsz, _, _, _, _, _ = self.validate_episode_inputs(query, support)
        all_scores = []
        aux_payload = {"episode_scores": []}
        for batch_idx in range(bsz):
            q_tokens, s_tokens, q_global, s_global = self._encode_episode(query[batch_idx], support[batch_idx])
            logits = self.classification_head(
                q_tokens,
                s_tokens,
                query_global=q_global,
                support_global=s_global,
            )
            all_scores.append(logits)
            aux_payload["episode_scores"].append(logits.detach())

        scores = torch.cat(all_scores, dim=0)
        if not return_aux:
            return scores
        aux_payload["episode_scores"] = torch.cat(aux_payload["episode_scores"], dim=0)
        return scores, aux_payload
