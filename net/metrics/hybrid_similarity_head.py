"""Hybrid Similarity Head: Cosine + Relation Delta Correction.

Combines:
- Cosine similarity as the main decision metric (preserves geometry)
- Lightweight relation delta head for non-linear correction (learns decision boundaries)

Design principles:
- Cosine dominant (lambda ~0.2-0.3 for delta)
- No deep relation modules (just 2-layer MLP)
- Pairwise comparison features (product + abs_diff), not concatenation
- No BatchNorm in head (uses LayerNorm for stability)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class EmbeddingProjection(nn.Module):
    """Meta-Baseline style bottleneck projection.
    
    Projects features to lower dimension for better generalization:
        Linear(C → C//2) → LayerNorm → L2 normalize
    
    Args:
        in_dim: Input feature dimension
        proj_dim: Projection dimension (default: in_dim // 2)
    """
    
    def __init__(self, in_dim: int, proj_dim: Optional[int] = None):
        super().__init__()
        self.proj_dim = proj_dim if proj_dim is not None else in_dim // 2
        
        self.proj = nn.Linear(in_dim, self.proj_dim, bias=False)
        self.norm = nn.LayerNorm(self.proj_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and normalize features.
        
        Args:
            x: (B, C) input features
            
        Returns:
            z: (B, proj_dim) projected, L2-normalized features
        """
        z = self.proj(x)
        z = self.norm(z)
        z = F.normalize(z, p=2, dim=-1)
        return z


class RelationDeltaHead(nn.Module):
    """Lightweight relation correction head.
    
    Learns a scalar correction delta based on pairwise comparison features:
        pair_feat = [z_query * proto, abs(z_query - proto)]
        delta = MLP(pair_feat)
    
    Uses element-wise product and absolute difference (NOT concatenation)
    for better feature interaction modeling.
    
    Args:
        embed_dim: Embedding dimension (C//2 from projection)
        hidden_dim: Hidden dimension in MLP (default: embed_dim // 2)
    """
    
    def __init__(self, embed_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        self.hidden_dim = hidden_dim if hidden_dim is not None else embed_dim // 2
        
        # Input: concat of [product, abs_diff] = 2 * embed_dim
        input_dim = embed_dim * 2
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 1)  # Output: scalar delta
        )
        
        # CRITICAL: Initialize so delta starts near 0
        # This prevents delta from disrupting cosine at training start
        self._init_weights()
    
    def _init_weights(self):
        """Initialize MLP so delta output starts near 0.
        
        Last layer: weight → small, bias → 0
        This ensures cosine is dominant at training start.
        """
        # Get the last Linear layer
        last_linear = self.mlp[-1]
        nn.init.normal_(last_linear.weight, mean=0.0, std=0.01)
        if last_linear.bias is not None:
            nn.init.zeros_(last_linear.bias)
    
    def forward(self, z_query: torch.Tensor, proto: torch.Tensor) -> torch.Tensor:
        """Compute relation delta for query-prototype pair.
        
        Args:
            z_query: (NQ, D) query embeddings
            proto: (D,) or (Way, D) prototype embeddings
            
        Returns:
            delta: (NQ,) or (NQ, Way) scalar correction values
        """
        # Handle single prototype vs multiple prototypes
        if proto.dim() == 1:
            proto = proto.unsqueeze(0)  # (1, D)
        
        # Broadcast for (NQ, Way, D) comparison
        if proto.dim() == 2 and z_query.dim() == 2:
            # z_query: (NQ, D), proto: (Way, D)
            # Expand to (NQ, Way, D) for pairwise comparison
            NQ, D = z_query.shape
            Way = proto.shape[0]
            
            z_query_exp = z_query.unsqueeze(1).expand(NQ, Way, D)  # (NQ, Way, D)
            proto_exp = proto.unsqueeze(0).expand(NQ, Way, D)      # (NQ, Way, D)
            
            # Pairwise comparison features
            product = z_query_exp * proto_exp          # Element-wise product
            abs_diff = torch.abs(z_query_exp - proto_exp)  # Absolute difference
            
            pair_feat = torch.cat([product, abs_diff], dim=-1)  # (NQ, Way, 2*D)
            
            # Flatten for MLP, then reshape
            pair_feat_flat = pair_feat.view(NQ * Way, -1)  # (NQ*Way, 2*D)
            delta_flat = self.mlp(pair_feat_flat)           # (NQ*Way, 1)
            delta = delta_flat.view(NQ, Way)                # (NQ, Way)
        else:
            raise ValueError(f"Unexpected shapes: z_query={z_query.shape}, proto={proto.shape}")
        
        return delta


class HybridSimilarityHead(nn.Module):
    """Hybrid Similarity Head: Cosine + Relation Delta.
    
    Combines cosine similarity (main score) with a lightweight relation delta
    correction. Cosine provides geometric distance, delta learns decision
    non-linearity for class boundaries.
    
    Final score: score_c = tau * cosine(z_q, proto_c) + lambda * delta(z_q, proto_c)
    
    Args:
        in_dim: Input feature dimension from backbone
        proj_dim: Projection dimension (default: in_dim // 2, ignored if use_projection=False)
        temperature: Cosine scaling factor tau (default: 16.0)
        delta_lambda: Weight for relation delta (default: 0.25)
        use_projection: Whether to use embedding projection (default: True)
    """
    
    def __init__(
        self,
        in_dim: int,
        proj_dim: Optional[int] = None,
        temperature: float = 16.0,
        delta_lambda: float = 0.25,
        use_projection: bool = True
    ):
        super().__init__()
        
        self.in_dim = in_dim
        self.use_projection = use_projection
        self.temperature = temperature
        self.delta_lambda = delta_lambda
        
        if use_projection:
            self.proj_dim = proj_dim if proj_dim is not None else in_dim // 2
            # 2️⃣ Embedding Projection (Meta-Baseline style)
            self.embedding_proj = EmbeddingProjection(in_dim, self.proj_dim)
            embed_dim = self.proj_dim
        else:
            # No projection, use full dimension
            self.proj_dim = in_dim
            self.embedding_proj = None
            embed_dim = in_dim
        
        # 5️⃣ Relation Delta Head
        self.relation_delta = RelationDeltaHead(embed_dim)
    
    def project(self, feat: torch.Tensor) -> torch.Tensor:
        """Project features through embedding projection.
        
        Args:
            feat: (B, C) GAP features from backbone
            
        Returns:
            z: (B, proj_dim) projected, L2-normalized embeddings
        """
        if self.use_projection and self.embedding_proj is not None:
            return self.embedding_proj(feat)
        else:
            # No projection, just L2 normalize
            return F.normalize(feat, p=2, dim=-1)
    
    def compute_prototypes(self, z_support: torch.Tensor, way_num: int, shot_num: int) -> torch.Tensor:
        """Compute class prototypes from support embeddings.
        
        Args:
            z_support: (Way*Shot, D) support embeddings
            way_num: Number of classes
            shot_num: Number of shots per class
            
        Returns:
            prototypes: (Way, D) L2-normalized class prototypes
        """
        # Reshape to (Way, Shot, D)
        z_support = z_support.view(way_num, shot_num, -1)
        
        # 3️⃣ Simple mean prototype
        prototypes = z_support.mean(dim=1)  # (Way, D)
        
        # Re-normalize after mean
        prototypes = F.normalize(prototypes, p=2, dim=-1)
        
        return prototypes
    
    def forward(
        self,
        z_query: torch.Tensor,
        prototypes: torch.Tensor
    ) -> torch.Tensor:
        """Compute hybrid similarity scores.
        
        Args:
            z_query: (NQ, D) query embeddings (already projected)
            prototypes: (Way, D) class prototypes (already projected)
            
        Returns:
            scores: (NQ, Way) final similarity scores
        """
        # 4️⃣ Cosine Similarity (Main Score)
        # Since both are L2 normalized: cosine = dot product
        s_cos = torch.mm(z_query, prototypes.t())  # (NQ, Way)
        s_cos = self.temperature * s_cos
        
        # 5️⃣ Relation Delta
        delta = self.relation_delta(z_query, prototypes)  # (NQ, Way)
        
        # 6️⃣ Final Score: cosine + lambda * delta
        scores = s_cos + self.delta_lambda * delta
        
        return scores
    
    def forward_with_support(
        self,
        feat_query: torch.Tensor,
        feat_support: torch.Tensor,
        way_num: int,
        shot_num: int
    ) -> torch.Tensor:
        """Full forward: project, prototype, score.
        
        Convenience method that handles the full pipeline.
        
        Args:
            feat_query: (NQ, C) query GAP features (raw, not projected)
            feat_support: (Way*Shot, C) support GAP features (raw)
            way_num: Number of classes
            shot_num: Shots per class
            
        Returns:
            scores: (NQ, Way) final similarity scores
        """
        # Project both query and support
        z_query = self.project(feat_query)       # (NQ, D)
        z_support = self.project(feat_support)   # (Way*Shot, D)
        
        # Compute prototypes
        prototypes = self.compute_prototypes(z_support, way_num, shot_num)  # (Way, D)
        
        # Compute hybrid scores
        scores = self.forward(z_query, prototypes)  # (NQ, Way)
        
        return scores


def build_hybrid_similarity_head(
    in_dim: int,
    proj_dim: Optional[int] = None,
    temperature: float = 16.0,
    delta_lambda: float = 0.25
) -> HybridSimilarityHead:
    """Factory function for HybridSimilarityHead.
    
    Args:
        in_dim: Input feature dimension from backbone
        proj_dim: Projection dimension (default: in_dim // 2)
        temperature: Cosine scaling tau (default: 16.0)
        delta_lambda: Delta weight lambda (default: 0.25)
        
    Returns:
        Configured HybridSimilarityHead
    """
    return HybridSimilarityHead(
        in_dim=in_dim,
        proj_dim=proj_dim,
        temperature=temperature,
        delta_lambda=delta_lambda
    )
