"""Few-shot selective scan model with explicit episodic semantics."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from net.fewshot_common import BaseConv64FewShotModel, feature_map_to_tokens, pooled_episode_features
from net.heads.class_memory_head import ClassMemoryReadoutHead
from net.heads.distribution_alignment_head import (
    ClassConditionalQueryReadout,
    DistributionAlignmentSWHead,
)
from net.heads.token_sw_head import TokenSetProjector
from net.metrics.sliced_wasserstein import SlicedWassersteinDistance
from net.ssm.class_memory_scan import ShotDescriptorEncoder
from net.ssm.episodic_selective_scan import SupportSetStateEncoder


class EpisodicSelectiveScanMambaNet(BaseConv64FewShotModel):
    """Role-aware and boundary-aware selective scan for few-shot episodes.

    Support shots are scanned per class in a write phase. Queries are scanned
    again in a class-conditioned read phase. SW acts as a distribution-aware
    alignment score between refined query tokens and refined support tokens.
    """

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
        use_role_embedding: bool = True,
        use_boundary_gate: bool = True,
        max_episode_positions: int = 32,
        max_way_num: int = 32,
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
        self.shot_encoder = ShotDescriptorEncoder(token_dim, token_dim)
        self.support_state_encoder = SupportSetStateEncoder(
            dim=token_dim,
            state_dim=ssm_state_dim,
            max_positions=max_episode_positions,
            max_classes=max_way_num,
            use_role_embedding=use_role_embedding,
            use_boundary_gate=use_boundary_gate,
        )
        self.query_readout = ClassConditionalQueryReadout(
            dim=token_dim,
            state_dim=ssm_state_dim,
            max_positions=max_episode_positions,
            max_classes=max_way_num,
            use_role_embedding=use_role_embedding,
            use_boundary_gate=use_boundary_gate,
        )
        self.readout_head = ClassMemoryReadoutHead(token_dim, temperature=temperature)
        self.support_token_adapter = nn.Linear(token_dim, token_dim)
        self.support_token_norm = nn.LayerNorm(token_dim)
        self.sw_alignment = DistributionAlignmentSWHead(
            SlicedWassersteinDistance(
                num_projections=sw_num_projections,
                p=sw_p,
                reduction="none",
                normalize_inputs=sw_normalize,
            ),
            merge_mode=token_merge_mode,
            score_scale=temperature,
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

    def _refine_support_tokens(
        self,
        support_tokens: torch.Tensor,
        refined_shot_states: torch.Tensor,
    ) -> torch.Tensor:
        context = self.support_token_adapter(refined_shot_states).unsqueeze(1)
        return self.support_token_norm(support_tokens + context)

    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bsz, _, _, _, _, _ = self.validate_episode_inputs(query, support)
        all_scores = []
        aux_payload = {
            "scan_logits": [],
            "sw_logits": [],
        }
        for batch_idx in range(bsz):
            q_tokens, s_tokens, _, s_global = self._encode_episode(query[batch_idx], support[batch_idx])
            shot_descriptors = self.shot_encoder(s_tokens, s_global)

            class_states = []
            query_readouts = []
            refined_queries = []
            refined_support = []
            for class_idx in range(s_tokens.shape[0]):
                class_state, refined_shots = self.support_state_encoder(
                    shot_descriptors[class_idx],
                    class_index=class_idx,
                )
                readout, refined_query = self.query_readout(
                    q_tokens,
                    class_state,
                    class_index=class_idx,
                )
                class_states.append(class_state)
                query_readouts.append(readout)
                refined_queries.append(refined_query)
                refined_support.append(self._refine_support_tokens(s_tokens[class_idx], refined_shots))

            class_states_t = torch.stack(class_states, dim=0)
            query_readouts_t = torch.stack(query_readouts, dim=1)
            refined_queries_t = torch.stack(refined_queries, dim=1)
            refined_support_t = torch.stack(refined_support, dim=0)

            scan_logits = self.readout_head(query_readouts_t, class_states_t)
            logits = scan_logits
            sw_logits = torch.zeros_like(scan_logits)
            if self.use_sw:
                sw_logits = self.sw_alignment(refined_queries_t, refined_support_t)
                logits = logits + self.sw_weight * sw_logits

            all_scores.append(logits)
            aux_payload["scan_logits"].append(scan_logits.detach())
            aux_payload["sw_logits"].append(sw_logits.detach())

        scores = torch.cat(all_scores, dim=0)
        if not return_aux:
            return scores
        return scores, {key: torch.cat(value, dim=0) for key, value in aux_payload.items()}
