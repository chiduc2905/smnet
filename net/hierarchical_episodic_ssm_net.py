"""Hierarchical episodic SSM model for token-to-shot and shot-to-class reasoning."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from net.fewshot_common import BaseConv64FewShotModel, feature_map_to_tokens
from net.heads.distribution_alignment_head import HierarchicalQueryMatcher, HierarchicalSWMetricHead
from net.heads.token_sw_head import TokenSetProjector
from net.metrics.sliced_wasserstein import SlicedWassersteinDistance
from net.ssm.hierarchical_ssm import HierarchicalClassAggregator, ShotLevelMemorySSM, TokenLevelSSMEncoder


class HierarchicalEpisodicSSMNet(BaseConv64FewShotModel):
    """Hierarchical few-shot model with token-level and shot-level SSM stages."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 64,
        token_dim: int | None = None,
        ssm_state_dim: int = 16,
        temperature: float = 16.0,
        conv64f_pool_last: bool = True,
        use_sw: bool = True,
        sw_weight: float = 0.25,
        sw_num_projections: int = 64,
        sw_p: float = 2.0,
        sw_normalize: bool = True,
        token_merge_mode: str = "concat",
        hierarchical_token_depth: int = 1,
        hierarchical_shot_depth: int = 1,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            conv64f_pool_last=conv64f_pool_last,
        )
        token_dim = token_dim or hidden_dim
        self.use_sw = use_sw
        self.sw_weight = sw_weight
        self.token_projector = TokenSetProjector(hidden_dim, token_dim)
        self.token_encoder = TokenLevelSSMEncoder(
            dim=token_dim,
            state_dim=ssm_state_dim,
            depth=hierarchical_token_depth,
            max_tokens=64,
        )
        self.shot_encoder = ShotLevelMemorySSM(
            dim=token_dim,
            state_dim=ssm_state_dim,
            depth=hierarchical_shot_depth,
        )
        self.class_aggregator = HierarchicalClassAggregator(token_dim)
        self.query_matcher = HierarchicalQueryMatcher(token_dim, temperature=temperature)
        self.support_token_adapter = nn.Linear(token_dim, token_dim)
        self.support_token_norm = nn.LayerNorm(token_dim)
        self.sw_head = HierarchicalSWMetricHead(
            SlicedWassersteinDistance(
                num_projections=sw_num_projections,
                p=sw_p,
                reduction="none",
                normalize_inputs=sw_normalize,
            ),
            merge_mode=token_merge_mode,
            score_scale=temperature,
        )

    def _condition_support_tokens(self, support_tokens: torch.Tensor, class_repr: torch.Tensor) -> torch.Tensor:
        context = self.support_token_adapter(class_repr).view(1, 1, -1)
        return self.support_token_norm(support_tokens + context)

    def _encode_episode(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        return q_tokens, s_tokens

    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bsz, _, _, _, _, _ = self.validate_episode_inputs(query, support)
        all_scores = []
        aux_payload = {"matcher_logits": [], "sw_logits": []}
        for batch_idx in range(bsz):
            q_tokens, s_tokens = self._encode_episode(query[batch_idx], support[batch_idx])
            refined_q_tokens, q_embeddings = self.token_encoder(q_tokens)
            support_flat = s_tokens.reshape(-1, s_tokens.shape[-2], s_tokens.shape[-1])
            refined_s_flat, s_embeddings_flat = self.token_encoder(support_flat)
            refined_s_tokens = refined_s_flat.reshape(*s_tokens.shape)
            s_embeddings = s_embeddings_flat.reshape(s_tokens.shape[0], s_tokens.shape[1], -1)

            class_reprs = []
            conditioned_support = []
            for class_idx in range(s_tokens.shape[0]):
                class_memory, refined_shots = self.shot_encoder(s_embeddings[class_idx])
                class_repr = self.class_aggregator(refined_shots, class_memory)
                class_reprs.append(class_repr)
                conditioned_support.append(
                    self._condition_support_tokens(refined_s_tokens[class_idx], class_repr)
                )

            class_reprs_t = torch.stack(class_reprs, dim=0)
            conditioned_support_t = torch.stack(conditioned_support, dim=0)
            matcher_logits = self.query_matcher(q_embeddings, class_reprs_t)
            logits = matcher_logits
            sw_logits = torch.zeros_like(matcher_logits)
            if self.use_sw:
                refined_q_expanded = refined_q_tokens.unsqueeze(1).expand(-1, class_reprs_t.shape[0], -1, -1)
                sw_logits = self.sw_head(refined_q_expanded, conditioned_support_t)
                logits = logits + self.sw_weight * sw_logits

            all_scores.append(logits)
            aux_payload["matcher_logits"].append(matcher_logits.detach())
            aux_payload["sw_logits"].append(sw_logits.detach())

        scores = torch.cat(all_scores, dim=0)
        if not return_aux:
            return scores
        return scores, {key: torch.cat(value, dim=0) for key, value in aux_payload.items()}
