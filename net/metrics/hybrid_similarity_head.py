"""Cosine Similarity Head with Bottleneck Projection.

Pipeline:
    GAP (from backbone) → Bottleneck (Linear → LayerNorm) → L2 Normalize → Cosine Similarity

Design principles:
- Simple and effective: Cosine similarity with temperature scaling
- Meta-Baseline style bottleneck projection for better generalization
- No complex relation modules (removed for better accuracy)
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


class CosineSimilarityHead(nn.Module):
    """Cosine Similarity Head with Bottleneck Projection.
    
    Simple and effective similarity computation:
        score = tau * cosine(z_query, prototype)
    
    Pipeline:
        GAP → Bottleneck (Linear → LayerNorm) → L2 → Cosine
    
    Args:
        in_dim: Input feature dimension from backbone
        proj_dim: Projection dimension (default: in_dim // 2)
        temperature: Cosine scaling factor tau (default: 16.0)
        use_projection: Whether to use embedding projection (default: True)
    """
    
    def __init__(
        self,
        in_dim: int,
        proj_dim: Optional[int] = None,
        temperature: float = 16.0,
        use_projection: bool = True
    ):
        super().__init__()
        
        self.in_dim = in_dim
        self.use_projection = use_projection
        self.temperature = temperature
        
        if use_projection:
            self.proj_dim = proj_dim if proj_dim is not None else in_dim // 2
            # Embedding Projection (Meta-Baseline style bottleneck)
            self.embedding_proj = EmbeddingProjection(in_dim, self.proj_dim)
        else:
            # No projection, use full dimension
            self.proj_dim = in_dim
            self.embedding_proj = None
    
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
        
        # Simple mean prototype
        prototypes = z_support.mean(dim=1)  # (Way, D)
        
        # Re-normalize after mean
        prototypes = F.normalize(prototypes, p=2, dim=-1)
        
        return prototypes
    
    def forward(
        self,
        z_query: torch.Tensor,
        prototypes: torch.Tensor
    ) -> torch.Tensor:
        """Compute cosine similarity scores.
        
        Args:
            z_query: (NQ, D) query embeddings (already projected)
            prototypes: (Way, D) class prototypes (already projected)
            
        Returns:
            scores: (NQ, Way) similarity scores
        """
        # Cosine Similarity: since both are L2 normalized, cosine = dot product
        s_cos = torch.mm(z_query, prototypes.t())  # (NQ, Way)
        scores = self.temperature * s_cos
        
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
            scores: (NQ, Way) similarity scores
        """
        # Project both query and support
        z_query = self.project(feat_query)       # (NQ, D)
        z_support = self.project(feat_support)   # (Way*Shot, D)
        
        # Compute prototypes
        prototypes = self.compute_prototypes(z_support, way_num, shot_num)  # (Way, D)
        
        # Compute cosine scores
        scores = self.forward(z_query, prototypes)  # (NQ, Way)
        
        return scores


# Alias for backward compatibility
HybridSimilarityHead = CosineSimilarityHead


def build_cosine_similarity_head(
    in_dim: int,
    proj_dim: Optional[int] = None,
    temperature: float = 16.0
) -> CosineSimilarityHead:
    """Factory function for CosineSimilarityHead.
    
    Args:
        in_dim: Input feature dimension from backbone
        proj_dim: Projection dimension (default: in_dim // 2)
        temperature: Cosine scaling tau (default: 16.0)
        
    Returns:
        Configured CosineSimilarityHead
    """
    return CosineSimilarityHead(
        in_dim=in_dim,
        proj_dim=proj_dim,
        temperature=temperature
    )


# Alias for backward compatibility
build_hybrid_similarity_head = build_cosine_similarity_head
