"""M6: Channel Metric Attention (CMA).

Purpose: Refine slot channels based on similarity to class prototype.
Channels that align well with the class prototype are weighted higher.

Input:
    - slots: (B, K, C) slot descriptors
    - class_proto: (C,) class prototype vector (from support)
    - slot_weights: Optional (B, K) from M5 SCA

Output:
    - refined_slots: (B, K, C) channel-refined slots
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ChannelMetricAttention(nn.Module):
    """Channel Metric Attention for channel-wise slot refinement.
    
    Refines each slot by weighting channels based on their alignment
    with the class prototype.
    
    Args:
        dim: Feature dimension C
        temperature: Temperature for channel softmax (default: 1.0)
        use_residual: Whether to add residual connection (default: True)
    """
    
    def __init__(
        self,
        dim: int,
        temperature: float = 1.0,
        use_residual: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.temperature = temperature
        self.use_residual = use_residual
        
        # Learnable temperature
        self.temp_scale = nn.Parameter(torch.tensor(temperature))
        
        # Optional residual scaling
        if use_residual:
            self.residual_scale = nn.Parameter(torch.tensor(0.5))
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(dim)
    
    def forward(
        self,
        slots: torch.Tensor,
        class_proto: torch.Tensor,
        slot_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Refine slots via channel-wise attention with class prototype.
        
        Args:
            slots: (B, K, C) slot descriptors
            class_proto: (C,) class prototype vector
            slot_weights: Optional (B, K) slot importance from SCA
            
        Returns:
            refined_slots: (B, K, C) channel-refined slot descriptors
        """
        B, K, C = slots.shape
        
        # Normalize class prototype
        class_proto_norm = F.normalize(class_proto, dim=-1)  # (C,)
        
        refined_slots = []
        
        for k in range(K):
            s = slots[:, k, :]  # (B, C)
            
            # Channel-wise competition: element-wise product with prototype
            # Higher values = channel aligns with class
            channel_scores = s * class_proto_norm.unsqueeze(0)  # (B, C)
            
            # Softmax over channels for attention weights
            beta = F.softmax(channel_scores / self.temp_scale, dim=-1)  # (B, C)
            
            # Scale slot by channel importance
            s_refined = beta * s  # (B, C)
            
            # Residual connection (optional)
            if self.use_residual:
                s_refined = s + self.residual_scale * s_refined
            
            # Apply slot weight from SCA if provided
            if slot_weights is not None:
                s_refined = s_refined * slot_weights[:, k:k+1]  # (B, C)
            
            refined_slots.append(s_refined)
        
        # Stack: (B, K, C)
        refined_slots = torch.stack(refined_slots, dim=1)
        
        # Normalize for stability
        refined_slots = self.norm(refined_slots)
        
        return refined_slots


def compute_class_prototype(
    patches: torch.Tensor,
    attn: Optional[torch.Tensor] = None,
    slot_weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute class prototype from support patches (utility function).
    
    Args:
        patches: (Shot, N, C) patches from support set
        attn: Optional (Shot, K, N) attention weights
        slot_weights: Optional (Shot, K) slot existence weights
        
    Returns:
        proto: (C,) class prototype vector (L2 normalized)
    """
    Shot, N, C = patches.shape
    
    if attn is not None and slot_weights is not None:
        # Weighted by slot attention and slot existence
        # pixel_importance = max over slots, weighted by slot existence
        weighted_attn = attn * slot_weights.unsqueeze(-1)  # (Shot, K, N)
        pixel_importance = weighted_attn.max(dim=1)[0]  # (Shot, N)
        pixel_importance = pixel_importance / (pixel_importance.sum(dim=-1, keepdim=True) + 1e-8)
        
        weighted_patches = patches * pixel_importance.unsqueeze(-1)  # (Shot, N, C)
        proto = weighted_patches.sum(dim=(0, 1))  # (C,)
    else:
        # Simple mean
        proto = patches.mean(dim=(0, 1))  # (C,)
    
    # L2 normalize
    proto = F.normalize(proto, dim=-1)
    
    return proto
