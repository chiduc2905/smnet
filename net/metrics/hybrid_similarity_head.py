"""Cosine similarity head with Uncertainty-Aware Prototype Scoring (UAPS)."""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingProjection(nn.Module):
    """Meta-Baseline style bottleneck projection."""

    def __init__(self, in_dim: int, proj_dim: Optional[int] = None):
        super().__init__()
        self.proj_dim = proj_dim if proj_dim is not None else in_dim // 2
        self.proj = nn.Linear(in_dim, self.proj_dim, bias=False)
        self.norm = nn.LayerNorm(self.proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        z = self.norm(z)
        z = F.normalize(z, p=2, dim=-1)
        return z


class CosineSimilarityHead(nn.Module):
    """Cosine + UAPS scoring.

    score_c = tau * cos(z_q, z_mu_c) - beta * mean((z_q - z_mu_c)^2 / (var_c + eps))
    """

    def __init__(
        self,
        in_dim: int,
        proj_dim: Optional[int] = None,
        temperature: float = 16.0,
        use_projection: bool = True,
        beta_maha: float = 0.25,
        uaps_eps: float = 1e-4,
        num_classes: int = 4,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.use_projection = use_projection
        self.temperature = float(temperature)
        self.beta_maha = float(beta_maha)
        self.uaps_eps = float(uaps_eps)

        if use_projection:
            self.proj_dim = proj_dim if proj_dim is not None else in_dim // 2
            self.embedding_proj = EmbeddingProjection(in_dim, self.proj_dim)
        else:
            self.proj_dim = in_dim
            self.embedding_proj = None

        # Fallback variance when shot == 1 (or unstable class variance).
        self.global_logvar = nn.Parameter(torch.zeros(max(1, int(num_classes)), self.proj_dim))

    def project(self, feat: torch.Tensor) -> torch.Tensor:
        if self.use_projection and self.embedding_proj is not None:
            return self.embedding_proj(feat)
        return F.normalize(feat, p=2, dim=-1)

    def compute_prototypes(self, z_support: torch.Tensor, way_num: int, shot_num: int) -> torch.Tensor:
        z_support = z_support.view(way_num, shot_num, -1)
        prototypes = z_support.mean(dim=1)
        return F.normalize(prototypes, p=2, dim=-1)

    def _fallback_var(self, class_idx: int, device: torch.device) -> torch.Tensor:
        if class_idx < self.global_logvar.shape[0]:
            return torch.exp(self.global_logvar[class_idx]).to(device)
        return torch.exp(self.global_logvar.mean(dim=0)).to(device)

    def score_with_uaps(
        self,
        z_query: torch.Tensor,
        z_proto: torch.Tensor,
        z_support: Optional[torch.Tensor],
        class_idx: int,
    ) -> torch.Tensor:
        """Class-conditional score with variance-aware distance penalty."""
        cosine = torch.mm(z_query, z_proto.t()).squeeze(1)

        fallback_var = self._fallback_var(class_idx, z_query.device)
        if z_support is not None and z_support.shape[0] > 1:
            class_var = z_support.var(dim=0, unbiased=False)
            class_var = class_var + 0.1 * fallback_var
        else:
            class_var = fallback_var

        class_var = torch.clamp(class_var, min=self.uaps_eps)
        diff = z_query - z_proto
        maha = ((diff * diff) / class_var.unsqueeze(0)).mean(dim=1)
        return self.temperature * cosine - self.beta_maha * maha

    def forward(self, z_query: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        s_cos = torch.mm(z_query, prototypes.t())
        return self.temperature * s_cos

    def forward_episode(
        self,
        query_vectors: torch.Tensor,
        support_shots: torch.Tensor,
        prototype_vectors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Score one episode with UAPS.

        Args:
            query_vectors: (NQ, Way, C)
            support_shots: (Way, Shot, C)
            prototype_vectors: Optional (Way, C), overrides support mean as center.
        """
        nq, way_num, _ = query_vectors.shape
        scores = []
        for class_idx in range(way_num):
            q_c = query_vectors[:, class_idx, :]  # (NQ, C)
            s_c = support_shots[class_idx]  # (Shot, C)

            z_q = self.project(q_c)
            z_support = self.project(s_c)

            if prototype_vectors is not None:
                z_proto = self.project(prototype_vectors[class_idx : class_idx + 1, :])
            else:
                z_proto = F.normalize(z_support.mean(dim=0, keepdim=True), p=2, dim=-1)

            score_c = self.score_with_uaps(
                z_query=z_q,
                z_proto=z_proto,
                z_support=z_support,
                class_idx=class_idx,
            ).view(nq, 1)
            scores.append(score_c)

        return torch.cat(scores, dim=1)

    def forward_with_support(
        self,
        feat_query: torch.Tensor,
        feat_support: torch.Tensor,
        way_num: int,
        shot_num: int,
    ) -> torch.Tensor:
        z_query = self.project(feat_query)
        z_support = self.project(feat_support).view(way_num, shot_num, -1)
        prototypes = F.normalize(z_support.mean(dim=1), p=2, dim=-1)

        score_list = []
        for c in range(way_num):
            z_proto = prototypes[c : c + 1, :]
            score_c = self.score_with_uaps(
                z_query=z_query,
                z_proto=z_proto,
                z_support=z_support[c],
                class_idx=c,
            ).view(-1, 1)
            score_list.append(score_c)
        return torch.cat(score_list, dim=1)


HybridSimilarityHead = CosineSimilarityHead


def build_cosine_similarity_head(
    in_dim: int,
    proj_dim: Optional[int] = None,
    temperature: float = 16.0,
    beta_maha: float = 0.25,
    uaps_eps: float = 1e-4,
    num_classes: int = 4,
) -> CosineSimilarityHead:
    return CosineSimilarityHead(
        in_dim=in_dim,
        proj_dim=proj_dim,
        temperature=temperature,
        beta_maha=beta_maha,
        uaps_eps=uaps_eps,
        num_classes=num_classes,
    )


build_hybrid_similarity_head = build_cosine_similarity_head
