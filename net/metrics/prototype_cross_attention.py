"""Prototype-Guided Cross-Attention Module (SOTA Version).

Implements Prototype Refinement + Safe Cross-Attention for Few-Shot Learning.

Key improvements over simple mean prototype:
1. Weighted mean prototype refinement - outlier rejection via cosine similarity weighting
2. Spatial pooling - noise reduction by downsampling prototype maps
3. Attention dropout - regularization during training

Pipeline:
    Support features (Way*Shot, C, H, W)
        → Weighted Mean (outlier rejection) → Prototype maps (Way, C, H, W)
        → Spatial Pooling (8×8) → Reduced proto maps (Way, C, Hp, Wp)
        → DETACH
    
    Query features (NQ, C, H, W)
        → Cross-attention with detached prototypes
        → Residual update: Q = Q + alpha * attended
        → Refined query (NQ, Way, C, H, W)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple


class PrototypeCrossAttention(nn.Module):
    """SOTA Prototype-Guided Cross-Attention with Outlier Rejection.
    
    Improvements:
    1. Weighted mean prototype - samples closer to mean get higher weight
    2. Spatial pooling - reduce noise in prototype maps
    3. Attention dropout - regularization
    
    Args:
        channels: Feature channel dimension
        alpha: Residual weight (default: 0.1)
        t_attn: Temperature for attention softmax (default: 2.0)
        t_proto: Temperature for prototype weighting (default: 0.7)
        proto_pool_size: Spatial pooling size for prototypes (default: 8)
        attn_dropout: Dropout rate for attention weights (default: 0.1)
    """
    
    def __init__(
        self, 
        channels: int, 
        alpha: float = 0.1, 
        t_attn: float = 2.0,
        t_proto: float = 0.7,
        proto_pool_size: int = 8,
        attn_dropout: float = 0.1
    ):
        super().__init__()
        self.channels = channels
        self.alpha = alpha
        self.t_attn = t_attn
        self.t_proto = t_proto
        self.proto_pool_size = proto_pool_size
        self.attn_dropout = attn_dropout
        self.scale = 1.0 / (math.sqrt(channels) * t_attn)
    
    def _compute_weighted_prototype(
        self, 
        support: torch.Tensor,  # (Shot, C, H, W)
    ) -> torch.Tensor:
        """Compute weighted mean prototype with outlier rejection.
        
        Uses cosine similarity to initial mean as weights.
        Samples closer to mean get higher weight, outliers get lower weight.
        
        Args:
            support: (Shot, C, H, W) support features for one class
            
        Returns:
            proto: (C, H, W) weighted mean prototype
        """
        Shot, C, H, W = support.shape
        
        # Step 1: Compute initial mean
        mu = support.mean(dim=0)  # (C, H, W)
        
        # Step 2: Compute cosine similarity of each sample to mean
        # Use contiguous() because support may be a slice (non-contiguous)
        S_flat = support.contiguous().view(Shot, -1)  # (Shot, C*H*W)
        mu_flat = mu.contiguous().view(1, -1)          # (1, C*H*W)
        
        sim = F.cosine_similarity(S_flat, mu_flat, dim=1)  # (Shot,)
        
        # Step 3: Softmax with temperature to get weights
        # Lower T_proto → more weight on high-similarity samples
        w = F.softmax(sim / self.t_proto, dim=0)  # (Shot,)
        
        # Step 4: Weighted sum
        proto = (w[:, None, None, None] * support).sum(dim=0)  # (C, H, W)
        
        return proto
    
    def forward(
        self,
        query_feat: torch.Tensor,
        support_feat: torch.Tensor,
        way_num: int,
        shot_num: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply prototype-guided cross-attention with SOTA refinement.
        
        Args:
            query_feat: (NQ, C, H, W) query feature maps
            support_feat: (Way*Shot, C, H, W) support feature maps
            way_num: Number of classes
            shot_num: Number of shots per class
            
        Returns:
            refined_query: (NQ, Way, C, H, W) refined query for each class
            proto_maps: (Way, C, Hp, Wp) prototype feature maps (detached, pooled)
        """
        NQ, C, H, W = query_feat.shape
        
        # ============================================================
        # Step 1: Reshape support → per-class
        # ============================================================
        support_reshaped = support_feat.view(way_num, shot_num, C, H, W)
        
        # ============================================================
        # Step 2: Weighted Mean Prototype Refinement (Outlier Rejection)
        # ============================================================
        proto_refined = []
        for c in range(way_num):
            proto_c = self._compute_weighted_prototype(support_reshaped[c])
            proto_refined.append(proto_c)
        
        proto_maps = torch.stack(proto_refined, dim=0)  # (Way, C, H, W)
        
        # ============================================================
        # Step 3: Spatial Pooling (Noise Reduction)
        # ============================================================
        proto_maps = F.adaptive_avg_pool2d(
            proto_maps, 
            output_size=(self.proto_pool_size, self.proto_pool_size)
        )  # (Way, C, Hp, Wp)
        
        Hp, Wp = self.proto_pool_size, self.proto_pool_size
        
        # ============================================================
        # Step 4: DETACH (Mandatory for Few-Shot)
        # ============================================================
        proto_maps = proto_maps.detach()
        
        # ============================================================
        # Step 5: Safe Cross-Attention (Query → Prototype)
        # ============================================================
        refined_queries = []
        
        for c in range(way_num):
            proto = proto_maps[c]                   # (C, Hp, Wp)
            proto_flat = proto.view(C, -1)          # (C, Hp*Wp)
            proto_t = proto_flat.t()                # (Hp*Wp, C)
            
            # Query flatten
            Q_flat = query_feat.view(NQ, C, -1)     # (NQ, C, H*W)
            Q_t = Q_flat.permute(0, 2, 1)           # (NQ, H*W, C)
            
            # Attention scores
            attn = torch.matmul(Q_t, proto_flat) * self.scale  # (NQ, H*W, Hp*Wp)
            attn = F.softmax(attn, dim=-1)
            
            # Dropout (only during training)
            attn = F.dropout(attn, p=self.attn_dropout, training=self.training)
            
            # Attend to prototype
            attended = torch.matmul(attn, proto_t)  # (NQ, H*W, C)
            attended = attended.permute(0, 2, 1)    # (NQ, C, H*W)
            attended = attended.view(NQ, C, H, W)   # (NQ, C, H, W)
            
            # Residual update
            Q_refined = query_feat + self.alpha * attended
            refined_queries.append(Q_refined)
        
        # Stack: (NQ, Way, C, H, W)
        refined_query = torch.stack(refined_queries, dim=1)
        
        return refined_query, proto_maps


def build_prototype_cross_attention(
    channels: int, 
    alpha: float = 0.1,
    t_attn: float = 2.0,
    t_proto: float = 0.7,
    proto_pool_size: int = 8,
    attn_dropout: float = 0.1
) -> PrototypeCrossAttention:
    """Factory function for PrototypeCrossAttention.
    
    Args:
        channels: Feature channel dimension
        alpha: Residual weight (default: 0.1)
        t_attn: Temperature for attention softmax (default: 2.0)
        t_proto: Temperature for prototype weighting (default: 0.7)
        proto_pool_size: Spatial pooling size for prototypes (default: 8)
        attn_dropout: Dropout rate for attention weights (default: 0.1)
        
    Returns:
        Configured PrototypeCrossAttention module
    """
    return PrototypeCrossAttention(
        channels=channels, 
        alpha=alpha, 
        t_attn=t_attn,
        t_proto=t_proto,
        proto_pool_size=proto_pool_size,
        attn_dropout=attn_dropout
    )
