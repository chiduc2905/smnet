"""Transport Evidence Mamba network for few-shot classification."""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from net.evidence.grid_serializer import TokenGridSerializer
from net.evidence.prefix_transport import PrefixTransportEvidenceBuilder
from net.fewshot_common import BaseConv64FewShotModel, feature_map_to_tokens, merge_support_tokens
from net.heads.token_sw_head import TokenSetProjector
from net.heads.transport_evidence_head import TransportEvidenceClassificationHead
from net.ssm.evidence_reader import TransportEvidenceReader


class TransportEvidenceMambaNet(BaseConv64FewShotModel):
    """Few-shot classifier that integrates support-query transport evidence."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 64,
        token_dim: int | None = None,
        evidence_dim: int | None = None,
        ssm_state_dim: int = 16,
        ssm_depth: int = 1,
        temperature: float = 16.0,
        conv64f_pool_last: bool = True,
        sw_num_projections: int = 64,
        sw_p: float = 2.0,
        sw_normalize: bool = True,
        use_transport_metrics: bool = True,
        token_merge_mode: str = "concat",
        serialization_orders: str | tuple[str, ...] = (
            "row_major",
            "row_major_reverse",
            "column_major",
            "column_major_reverse",
        ),
        use_delta: bool = True,
        use_support_context: bool = True,
        readout_mode: str = "final",
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            conv64f_pool_last=conv64f_pool_last,
        )
        token_dim = token_dim or hidden_dim
        evidence_dim = evidence_dim or token_dim
        self.token_merge_mode = token_merge_mode
        self.serializer = TokenGridSerializer(serialization_orders)
        self.token_projector = TokenSetProjector(hidden_dim, token_dim)
        self.evidence_builder = PrefixTransportEvidenceBuilder(
            token_dim=token_dim,
            evidence_dim=evidence_dim,
            num_projections=sw_num_projections,
            p=sw_p,
            normalize_inputs=sw_normalize,
            use_transport_metrics=use_transport_metrics,
            use_delta=use_delta,
            use_support_context=use_support_context,
        )
        self.evidence_reader = TransportEvidenceReader(
            dim=evidence_dim,
            state_dim=ssm_state_dim,
            depth=ssm_depth,
            readout_mode=readout_mode,
        )
        self.classification_head = TransportEvidenceClassificationHead(
            dim=evidence_dim,
            temperature=temperature,
        )

    def _encode_episode(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        way_num, shot_num = support.shape[:2]
        q_features = self.encode(query)
        s_features = self.encode(support.reshape(way_num * shot_num, *support.shape[-3:]))
        spatial_hw = tuple(q_features.shape[-2:])
        q_tokens = self.token_projector(feature_map_to_tokens(q_features))
        s_tokens = self.token_projector(feature_map_to_tokens(s_features)).reshape(
            way_num,
            shot_num,
            -1,
            q_tokens.shape[-1],
        )
        return q_tokens, s_tokens, spatial_hw

    def _score_episode(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        spatial_hw: Tuple[int, int],
        return_aux: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor | tuple[str, ...]]]:
        support_sets = merge_support_tokens(support_tokens, merge_mode=self.token_merge_mode)
        serialized_queries = self.serializer.serialize(query_tokens, *spatial_hw)

        order_logits = []
        order_readouts = []
        order_hidden = []
        order_transport = []
        order_delta = []

        for order_idx in range(serialized_queries.shape[0]):
            evidence_tokens, evidence_aux = self.evidence_builder(
                serialized_queries[order_idx],
                support_sets,
            )
            num_query, way_num, token_count, evidence_dim = evidence_tokens.shape
            readout, hidden, _ = self.evidence_reader(
                evidence_tokens.reshape(num_query * way_num, token_count, evidence_dim)
            )
            readout = readout.reshape(num_query, way_num, evidence_dim)
            hidden = hidden.reshape(num_query, way_num, token_count, evidence_dim)
            logits = self.classification_head(readout)

            order_logits.append(logits)
            if return_aux:
                order_readouts.append(readout)
                order_hidden.append(hidden)
                order_transport.append(evidence_aux["prefix_transport"])
                order_delta.append(evidence_aux["transport_delta"])

        stacked_logits = torch.stack(order_logits, dim=0)
        fused_logits = stacked_logits.mean(dim=0)
        if not return_aux:
            return fused_logits
        return fused_logits, {
            "order_names": self.serializer.order_names,
            "order_logits": stacked_logits,
            "order_readouts": torch.stack(order_readouts, dim=0),
            "order_hidden": torch.stack(order_hidden, dim=0),
            "prefix_transport": torch.stack(order_transport, dim=0),
            "transport_delta": torch.stack(order_delta, dim=0),
        }

    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor | tuple[str, ...]]]:
        bsz, _, _, _, _, _ = self.validate_episode_inputs(query, support)
        all_scores = []
        aux_payload = {
            "order_logits": [],
            "order_readouts": [],
            "order_hidden": [],
            "prefix_transport": [],
            "transport_delta": [],
        }

        for batch_idx in range(bsz):
            q_tokens, s_tokens, spatial_hw = self._encode_episode(query[batch_idx], support[batch_idx])
            result = self._score_episode(q_tokens, s_tokens, spatial_hw, return_aux=return_aux)
            if return_aux:
                logits, episode_aux = result
                for key in aux_payload:
                    aux_payload[key].append(episode_aux[key])
            else:
                logits = result
            all_scores.append(logits)

        scores = torch.cat(all_scores, dim=0)
        if not return_aux:
            return scores
        aux = {key: torch.stack(value, dim=0) for key, value in aux_payload.items()}
        aux["order_names"] = self.serializer.order_names
        return scores, aux
