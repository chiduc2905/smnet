"""UnifiedSpatialChannelAttention: Residual Gating for Feature Selection.

FIXED: Changed from multiplicative (X ⊙ A_ch ⊙ A_sp) to residual gating:
    X_out = X + X ⊙ A_ch + X ⊙ A_sp

This prevents gradient collapse from cascaded multiplications.
"""
import torch
import torch.nn as nn
import math

# Reuse ECAPlus from dual_branch_fusion
from net.backbone.dual_branch_fusion import ECAPlus


class UnifiedSpatialChannelAttention(nn.Module):
    """Unified Attention Block with RESIDUAL GATING.
    
    IMPORTANT: Uses residual gating instead of multiplicative:
        X_out = X + X ⊙ A_ch + X ⊙ A_sp
    
    This allows:
        - Strong gradient flow through X (identity)
        - Additive contributions from channel/spatial attention
        - No cascaded multiplications that collapse gradients
    
    Args:
        channels: Number of input channels
        spatial_kernel: Kernel size for spatial DWConv (default: 3)
    """
    
    def __init__(self, channels: int, spatial_kernel: int = 3):
        super().__init__()
        
        # Channel attention: ECAPlus returns A_ch (attention weights after sigmoid)
        # We need raw attention weights, so create custom version
        self.channel_gap = nn.AdaptiveAvgPool2d(1)
        # Adaptive kernel size for ECA
        t = int(abs(math.log2(channels) / 2 + 0.5))
        k = t if t % 2 else t + 1
        k = max(k, 3)
        self.channel_conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.channel_sigmoid = nn.Sigmoid()
        
        # Spatial attention: DWConv → Sigmoid
        self.spatial_conv = nn.Conv2d(
            channels, channels,
            kernel_size=spatial_kernel,
            padding=spatial_kernel // 2,
            groups=channels,  # Depthwise
            bias=False
        )
        self.spatial_sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input features
            
        Returns:
            (B, C, H, W) attention-weighted features with residual
        """
        B, C, H, W = x.shape
        
        # Channel attention weights A_ch
        y = self.channel_gap(x).view(B, C, 1).transpose(1, 2)  # (B, 1, C)
        y = self.channel_conv1d(y)  # (B, 1, C)
        A_ch = self.channel_sigmoid(y).transpose(1, 2).view(B, C, 1, 1)  # (B, C, 1, 1)
        
        # Spatial attention weights A_sp
        A_sp = self.spatial_sigmoid(self.spatial_conv(x))  # (B, C, H, W)
        
        # RESIDUAL GATING: X_out = X + X⊙A_ch + X⊙A_sp
        x_out = x + x * A_ch + x * A_sp
        
        return x_out

