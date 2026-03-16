"""Transport-Prior Replay Mamba network for low-shot scalogram classification."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from net.atoms.support_atomizer import SupportAtomizer
from net.fewshot_common import BaseConv64FewShotModel, feature_map_to_tokens, pooled_episode_features
from net.heads.token_sw_head import TokenSetProjector
from net.heads.trajectory_transport_head import TrajectoryTransportHead
from net.ssm.replay_query_reader import ReplayQueryReader
from net.ssm.transport_prior import TransportPriorCalibrator


class TransportPriorReplayMambaNet(BaseConv64FewShotModel):
    """Few-shot classifier with transport-calibrated support priors and replay query reading."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 64,
        token_dim: int | None = None,
        ssm_state_dim: int = 16,
        temperature: float = 16.0,
        conv64f_pool_last: bool = True,
        num_support_atoms: int = 4,
        num_prior_atoms: int = 4,
        prior_bank_size: int = 16,
        prior_bank_atoms_per_entry: int = 4,
        prior_bank_topk: int = 4,
        sw_num_projections: int = 64,
        sw_p: float = 2.0,
        trajectory_transport_weight: float = 8.0,
        confidence_logit_weight: float = 0.5,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            conv64f_pool_last=conv64f_pool_last,
        )
        token_dim = token_dim or hidden_dim
        self.token_projector = TokenSetProjector(hidden_dim, token_dim)
        self.support_atomizer = SupportAtomizer(
            dim=token_dim,
            num_atoms=num_support_atoms,
            state_dim=ssm_state_dim,
        )
        self.prior_calibrator = TransportPriorCalibrator(
            dim=token_dim,
            num_prior_atoms=num_prior_atoms,
            bank_size=prior_bank_size,
            bank_atoms_per_entry=prior_bank_atoms_per_entry,
            topk=prior_bank_topk,
            num_projections=sw_num_projections,
            p=sw_p,
        )
        self.query_reader = ReplayQueryReader(
            dim=token_dim,
            state_dim=ssm_state_dim,
            num_projections=sw_num_projections,
            p=sw_p,
        )
        self.trajectory_head = TrajectoryTransportHead(
            dim=token_dim,
            temperature=temperature,
            transport_weight=trajectory_transport_weight,
            confidence_weight=confidence_logit_weight,
            num_projections=sw_num_projections,
            p=sw_p,
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
        aux_payload = {
            "transport_confidence": [],
            "raw_bank_sw": [],
            "prior_support_sw": [],
            "mean_replay_gate": [],
            "mean_prefix_sw": [],
            "cosine_logits": [],
            "transport_logits": [],
        }

        for batch_idx in range(bsz):
            q_tokens, s_tokens, q_global, _ = self._encode_episode(query[batch_idx], support[batch_idx])
            support_atoms, atom_confidence = self.support_atomizer(s_tokens)
            merged_atoms = support_atoms.reshape(s_tokens.shape[0], -1, support_atoms.shape[-1])
            prior_payload = self.prior_calibrator(merged_atoms)
            prior_atoms = prior_payload["prior_atoms"]
            transport_confidence = prior_payload["transport_confidence"]

            query_readouts = []
            query_trajectories = []
            replay_gate_means = []
            prefix_sw_means = []
            for class_idx in range(s_tokens.shape[0]):
                readout, refined_query, query_aux = self.query_reader(
                    q_tokens,
                    prior_atoms[class_idx],
                    query_global=q_global,
                    transport_confidence=transport_confidence[class_idx],
                )
                query_readouts.append(readout)
                query_trajectories.append(refined_query)
                replay_gate_means.append(query_aux["replay_gates"].mean(dim=1))
                prefix_sw_means.append(query_aux["prefix_sw"].mean(dim=1))

            query_readouts_t = torch.stack(query_readouts, dim=1)
            query_trajectories_t = torch.stack(query_trajectories, dim=1)
            logits, logit_components = self.trajectory_head(
                query_readouts_t,
                query_trajectories_t,
                prior_atoms,
                transport_confidence=transport_confidence,
                return_components=True,
            )
            all_scores.append(logits)

            aux_payload["transport_confidence"].append(transport_confidence.unsqueeze(0))
            aux_payload["raw_bank_sw"].append(prior_payload["raw_bank_sw"].unsqueeze(0))
            aux_payload["prior_support_sw"].append(prior_payload["prior_support_sw"].unsqueeze(0))
            aux_payload["mean_replay_gate"].append(torch.stack(replay_gate_means, dim=1))
            aux_payload["mean_prefix_sw"].append(torch.stack(prefix_sw_means, dim=1))
            aux_payload["cosine_logits"].append(logit_components["cosine_logits"].detach())
            aux_payload["transport_logits"].append(logit_components["transport_logits"].detach())

            _ = atom_confidence  # reserved for future confidence-aware atom regularization

        scores = torch.cat(all_scores, dim=0)
        if not return_aux:
            return scores
        return scores, {key: torch.cat(value, dim=0) for key, value in aux_payload.items()}
