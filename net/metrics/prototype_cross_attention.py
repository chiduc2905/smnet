"""Prototype-Guided Cross-Attention Module.

Lightweight cross-attention between query and class prototype feature maps.
Applied between UnifiedAttention and GAP for spatial feature refinement.

Pipeline:
    Query features (B, C, H, W) + Support features (Way*Shot, C, H, W)
    → Prototype maps (Way, C, H, W) by averaging shots [DETACHED]
    → Cross-attention: Q' = softmax( (Q·Pᵀ) / (√C · T_attn) ) · P
    → Residual update: Q = Q + alpha * Q'
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple


class PrototypeCrossAttention(nn.Module):
    """Lightweight prototype-guided cross-attention.
    
    Refines query features by attending to class prototype feature maps.
    Prototype maps are DETACHED to prevent gradient flow to support features.
    
    For each class c:
        1. Compute prototype map P_c = mean(support_c, dim=shot).detach()
        2. Cross-attention: Q' = softmax( (Q · P_c^T) / (√C · T_attn) ) · P_c
        3. Residual: Q = Q + alpha * Q'
    
    Args:
        channels: Feature channel dimension
        alpha: Residual weight (default: 0.1)
        t_attn: Temperature for attention softmax (default: 2.0)
    """
    
    def __init__(self, channels: int, alpha: float = 0.1, t_attn: float = 2.0):
        super().__init__()
        self.channels = channels
        self.alpha = alpha
        self.t_attn = t_attn
        self.scale = 1.0 / (math.sqrt(channels) * t_attn)
    
    def forward(
        self,
        query_feat: torch.Tensor,
        support_feat: torch.Tensor,
        way_num: int,
        shot_num: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply prototype-guided cross-attention.
        
        Args:
            query_feat: (NQ, C, H, W) query feature maps
            support_feat: (Way*Shot, C, H, W) support feature maps
            way_num: Number of classes
            shot_num: Number of shots per class
            
        Returns:
            refined_query: (NQ, Way, C, H, W) refined query for each class
            proto_maps: (Way, C, H, W) prototype feature maps (detached)
        """
        NQ, C, H, W = query_feat.shape
        
        # ============================================================
        # Step 1: Compute prototype feature maps (DETACHED)
        # ============================================================
        # Reshape support: (Way*Shot, C, H, W) → (Way, Shot, C, H, W)
        support_reshaped = support_feat.view(way_num, shot_num, C, H, W)
        
        # Average over shots: (Way, C, H, W) and DETACH
        proto_maps = support_reshaped.mean(dim=1)
        proto_maps = proto_maps.detach()  # 📌 Detach immediately after creation
        
        # ============================================================
        # Step 2: Cross-attention for each query-prototype pair
        # ============================================================
        # Q' = softmax( (Q · P^T) / (√C · T_attn) ) · P
        # Query is NOT detached - gradients flow through query
        
        refined_queries = []
        
        for c in range(way_num):
            # Get class prototype: (C, H, W) - already detached
            proto_c = proto_maps[c]  # (C, H, W)
            
            # Flatten spatial: (C, H*W)
            proto_flat = proto_c.view(C, H * W)  # (C, H*W)
            query_flat = query_feat.view(NQ, C, H * W)  # (NQ, C, H*W)
            
            # Transpose for attention computation
            query_t = query_flat.permute(0, 2, 1)  # (NQ, H*W, C)
            proto_t = proto_flat.permute(1, 0)     # (H*W, C)
            
            # Attention scores with temperature: (NQ, H*W_q, H*W_p)
            # scale = 1 / (√C · T_attn)
            attn = torch.matmul(query_t, proto_flat) * self.scale  # (NQ, H*W, H*W)
            attn = F.softmax(attn, dim=-1)  # (NQ, H*W, H*W)
            
            # Attend to prototype: (NQ, H*W, C)
            attended = torch.matmul(attn, proto_t)  # (NQ, H*W, C)
            
            # Back to (NQ, C, H*W) and reshape
            attended = attended.permute(0, 2, 1)  # (NQ, C, H*W)
            attended = attended.view(NQ, C, H, W)  # (NQ, C, H, W)
            
            # Residual update (query NOT detached)
            refined_c = query_feat + self.alpha * attended  # (NQ, C, H, W)
            refined_queries.append(refined_c)
        
        # Stack: (Way, NQ, C, H, W) → (NQ, Way, C, H, W)
        refined_query = torch.stack(refined_queries, dim=0)  # (Way, NQ, C, H, W)
        refined_query = refined_query.permute(1, 0, 2, 3, 4)  # (NQ, Way, C, H, W)
        
        return refined_query, proto_maps


def build_prototype_cross_attention(
    channels: int, 
    alpha: float = 0.1,
    t_attn: float = 2.0
) -> PrototypeCrossAttention:
    """Factory function for PrototypeCrossAttention.
    
    Args:
        channels: Feature channel dimension
        alpha: Residual weight (default: 0.1)
        t_attn: Temperature for attention softmax (default: 2.0)
        
    Returns:
        Configured PrototypeCrossAttention module
    """
    return PrototypeCrossAttention(channels=channels, alpha=alpha, t_attn=t_attn)
