"""UnifiedSpatialChannelAttention: The ONLY multiplicative attention module.

This module is the ONLY place where MULTIPLY (gating) is allowed.
Applied AFTER DualBranchFusion for feature selection.

Reference:
    - MULTIPLY for modules that SELECT or REWEIGHT existing features (attention)
    - All other modules use ADD (residual)
"""
import torch
import torch.nn as nn
import math

# Reuse ECAPlus from dual_branch_fusion
from net.backbone.dual_branch_fusion import ECAPlus


class UnifiedSpatialChannelAttention(nn.Module):
    """Unified Attention Block (MULTIPLY for selection).
    
    This is the ONLY multiplicative attention in the pipeline.
    Applied after DualBranchFusion for feature selection.
    
    Design:
        - Channel: ECAPlus (reused from dual_branch_fusion)
        - Spatial: Gated DWConv (3×3)
        - Order: Channel → Spatial
        - Combo: Y = X ⊙ A_ch ⊙ A_sp
    
    Architecture:
        Input X
            ↓
        ┌─────────────────────────────────────────────┐
        │ Channel Attention: ECA++                    │
        │   Y1 = X ⊙ Sigmoid(Conv1D(GAP(X)))         │
        └─────────────────────────────────────────────┘
            ↓
        ┌─────────────────────────────────────────────┐
        │ Spatial Attention: Gated DWConv            │
        │   Y2 = Y1 ⊙ Sigmoid(DWConv3x3(Y1))         │
        └─────────────────────────────────────────────┘
            ↓
        Output Y2
    
    Args:
        channels: Number of input channels
        spatial_kernel: Kernel size for spatial DWConv (default: 3)
    """
    
    def __init__(self, channels: int, spatial_kernel: int = 3):
        super().__init__()
        
        # Channel attention: reuse ECAPlus
        self.channel_attn = ECAPlus(channels)
        
        # Spatial attention: Gated depthwise + sigmoid
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(
                channels, channels,
                kernel_size=spatial_kernel,
                padding=spatial_kernel // 2,
                groups=channels,  # Depthwise
                bias=False
            ),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input features
            
        Returns:
            (B, C, H, W) attention-weighted features
        """
        # Channel selection (ECA++ - MUL)
        x = self.channel_attn(x)
        
        # Spatial selection (MUL)
        x = x * self.spatial_attn(x)
        
        return x
