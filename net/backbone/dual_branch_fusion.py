"""Dual-Branch Local-Global Feature Extraction with Anchor-Guided Residual Fusion.

This module implements a parallel dual-branch architecture:
- Branch 1 (Local): ConvMixer++ for local/mid-range spatial and channel interactions
- Branch 2 (Global): SS2D/Mamba for global pixel-level context

The original input X serves as an anchor for residual fusion, preserving
feature geometry for downstream slot grouping and covariance estimation.

Reference Architecture:
    X (anchor) ──────────────────────────────────────────┐
         │                                               │
         ├──→ [Local Branch (ConvMixer++)] ──→ F_local   │
         │                                               │
         └──→ [Global Branch (SS2D)]       ──→ F_global  │
                                                         │
         Concat(F_local, F_global) → Proj → α·Proj + X ──┘
                                                         │
                                              Y = Norm(fused)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


# =============================================================================
# Branch 1: Local-Mid Spatial & Channel Branch (ConvMixer++ Branch)
# =============================================================================

class SELite(nn.Module):
    """Lightweight Squeeze-and-Excitation for channel interaction.
    
    Minimal SE block with reduced bottleneck for efficiency.
    """
    
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        reduced = max(channels // reduction, 4)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        y = self.pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y


class SpatialGate(nn.Module):
    """Lightweight spatial gating via sigmoid-activated depthwise convolution."""
    
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, 
                      padding=kernel_size // 2, groups=channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class LocalMidBranch(nn.Module):
    """Local-Mid Spatial & Channel Branch (ConvMixer++ style).
    
    Focuses on local and mid-range spatial interactions without global attention.
    
    Components:
        1. Depthwise Conv (k=3) for local spatial modeling
        2. Dilated Depthwise Conv (k=3, d=2) for mid-range spatial interaction
        3. Lightweight spatial gating (sigmoid-activated DW conv)
        4. Pointwise (1×1) conv for channel mixing
        5. SE-Lite for channel interaction
        6. SiLU activations throughout
    
    Args:
        channels: Number of input/output channels
        dilation: Dilation rate for mid-range conv (default: 2)
    """
    
    def __init__(self, channels: int, dilation: int = 2):
        super().__init__()
        
        # Local spatial modeling (k=3, standard)
        self.local_dw = nn.Conv2d(
            channels, channels, kernel_size=3, 
            padding=1, groups=channels, bias=False
        )
        self.local_bn = nn.BatchNorm2d(channels)
        
        # Mid-range spatial modeling (k=3, dilated)
        self.mid_dw = nn.Conv2d(
            channels, channels, kernel_size=3,
            padding=dilation, dilation=dilation, groups=channels, bias=False
        )
        self.mid_bn = nn.BatchNorm2d(channels)
        
        # Spatial gating
        self.spatial_gate = SpatialGate(channels, kernel_size=3)
        
        # Channel mixing (pointwise)
        self.channel_mix = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
        
        # Lightweight channel attention
        self.se = SELite(channels, reduction=8)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input features
            
        Returns:
            F_local: (B, C, H, W) local-mid range features
        """
        # Local spatial
        local = F.silu(self.local_bn(self.local_dw(x)))
        
        # Mid-range spatial (dilated)
        mid = F.silu(self.mid_bn(self.mid_dw(local)))
        
        # Combine local and mid features
        combined = local + mid
        
        # Spatial gating
        gated = self.spatial_gate(combined)
        
        # Channel mixing
        mixed = self.channel_mix(gated)
        
        # Channel attention
        out = self.se(mixed)
        
        return out


# =============================================================================
# Branch 2: Global Pixel-Level Branch (SS2D Branch)
# =============================================================================

class GlobalSS2DBranch(nn.Module):
    """Global Pixel-Level Branch using SS2D (4-way Mamba scanning).
    
    Captures long-range dependencies through 4 independent directional scans.
    
    Components:
        1. 1×1 conv projection with normalization before SS2D
        2. Four INDEPENDENT SS2D scans (no weight sharing):
           - Left-to-right (row-major)
           - Right-to-left (row-major reversed)
           - Top-to-bottom (column-major)
           - Bottom-to-top (column-major reversed)
        3. Gated fusion of directional outputs
        4. Skip connection to prevent over-smoothing
        5. No non-linear activation after SS2D
    
    Args:
        d_model: Model dimension (channels)
        d_state: SSM state dimension (default: 16)
        d_conv: Local convolution width in Mamba (default: 4)
        expand: Expansion factor (default: 1)
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        
        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba-ssm package is required. Install with: pip install mamba-ssm"
            )
        
        # Pre-SS2D projection with normalization
        self.pre_norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner, bias=False)
        
        # 4 INDEPENDENT Mamba modules (no weight sharing)
        self.mamba_lr = Mamba(d_model=self.d_inner, d_state=d_state, d_conv=d_conv, expand=1)
        self.mamba_rl = Mamba(d_model=self.d_inner, d_state=d_state, d_conv=d_conv, expand=1)
        self.mamba_tb = Mamba(d_model=self.d_inner, d_state=d_state, d_conv=d_conv, expand=1)
        self.mamba_bt = Mamba(d_model=self.d_inner, d_state=d_state, d_conv=d_conv, expand=1)
        
        # Fusion gate (learnable weighted combination)
        self.fusion_gate = nn.Parameter(torch.ones(4) / 4)
        
        # Output projection (no activation after)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # Skip gate to prevent over-smoothing
        self.skip_gate = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input features
            
        Returns:
            F_global: (B, C, H, W) global context features
        """
        B, C, H, W = x.shape
        L = H * W
        
        # Flatten: (B, C, H, W) -> (B, L, C)
        x_seq = rearrange(x, 'b c h w -> b (h w) c')
        
        # Save for skip connection
        skip = x_seq
        
        # Pre-normalize and project
        x_norm = self.pre_norm(x_seq)
        x_proj = self.in_proj(x_norm)  # (B, L, d_inner)
        
        # === 4 Independent Directional Scans ===
        
        # 1. Left-to-Right (row-major)
        y_lr = self.mamba_lr(x_proj)
        
        # 2. Right-to-Left (row-major reversed)
        x_rl = torch.flip(x_proj, dims=[1])
        y_rl = self.mamba_rl(x_rl)
        y_rl = torch.flip(y_rl, dims=[1])  # Flip back
        
        # 3. Top-to-Bottom (column-major)
        x_tb = x_proj.view(B, H, W, -1).permute(0, 2, 1, 3).contiguous().view(B, L, -1)
        y_tb = self.mamba_tb(x_tb)
        y_tb = y_tb.view(B, W, H, -1).permute(0, 2, 1, 3).contiguous().view(B, L, -1)
        
        # 4. Bottom-to-Top (column-major reversed)
        x_bt = torch.flip(x_tb, dims=[1])
        y_bt = self.mamba_bt(x_bt)
        y_bt = torch.flip(y_bt, dims=[1])
        y_bt = y_bt.view(B, W, H, -1).permute(0, 2, 1, 3).contiguous().view(B, L, -1)
        
        # === Gated Fusion ===
        gate = F.softmax(self.fusion_gate, dim=0)
        y_fused = gate[0] * y_lr + gate[1] * y_rl + gate[2] * y_tb + gate[3] * y_bt
        
        # Output projection (NO activation after SS2D)
        out = self.out_proj(y_fused)
        
        # Skip connection with learnable gate
        out = self.skip_gate * out + (1 - self.skip_gate) * skip
        
        # Reshape back: (B, L, C) -> (B, C, H, W)
        out = rearrange(out, 'b (h w) c -> b c h w', h=H, w=W)
        
        return out


# =============================================================================
# Anchor-Guided Dual-Branch Fusion Module
# =============================================================================

class DualBranchFusion(nn.Module):
    """Dual-Branch Local-Global Feature Extraction with Anchor-Guided Residual Fusion.
    
    This module processes input features through two parallel branches:
    - Local Branch: ConvMixer++ for local/mid-range spatial and channel interactions
    - Global Branch: SS2D for global pixel-level context
    
    The original input X serves as an ANCHOR for residual fusion, preserving
    feature geometry critical for downstream slot grouping and covariance estimation.
    
    Fusion Strategy:
        F_cat = Concat(F_local, F_global)  # (B, 2C, H, W)
        F_proj = Conv1x1(F_cat)            # (B, C, H, W)
        Y = Norm(X + α · F_proj)           # Anchor-guided residual
    
    Why Local and Global are Complementary:
        - Local branch captures fine-grained textures, edges, and local patterns
        - Global branch captures long-range dependencies and holistic structure
        - Neither can fully substitute the other:
          * Local alone misses global context → poor class discrimination
          * Global alone loses fine details → poor intra-class precision
        - Fusion combines both for robust few-shot representations
    
    Role of Anchor-Guided Residual:
        - Preserves original feature geometry (critical for slot attention)
        - Prevents information loss during branch processing
        - Learnable α balances contribution of branch features
        - Linear residual path enables smooth gradient flow
    
    Args:
        channels: Number of input/output channels
        d_state: SSM state dimension for global branch (default: 16)
        dilation: Dilation rate for local branch mid-range conv (default: 2)
    """
    
    def __init__(
        self,
        channels: int,
        d_state: int = 16,
        dilation: int = 2
    ):
        super().__init__()
        
        self.channels = channels
        
        # Branch 1: Local-Mid Spatial & Channel
        self.local_branch = LocalMidBranch(channels, dilation=dilation)
        
        # Branch 2: Global SS2D
        self.global_branch = GlobalSS2DBranch(
            d_model=channels,
            d_state=d_state,
            d_conv=4,
            expand=1
        )
        
        # Fusion: Concat (2C) -> Proj (C)
        self.fusion_proj = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
        
        # Learnable scaling parameter α
        self.alpha = nn.Parameter(torch.tensor(0.1))
        
        # Final normalization
        self.norm = nn.LayerNorm(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input features (from ConvMixer)
            
        Returns:
            y: (B, C, H, W) fused features with anchor-guided residual
        """
        B, C, H, W = x.shape
        
        # === ANCHOR: Keep original input (NO processing) ===
        anchor = x
        
        # === BRANCH 1: Local-Mid Processing ===
        f_local = self.local_branch(x)  # (B, C, H, W)
        
        # === BRANCH 2: Global Processing ===
        f_global = self.global_branch(x)  # (B, C, H, W)
        
        # === FUSION ===
        # Concatenate along channel dimension
        f_cat = torch.cat([f_local, f_global], dim=1)  # (B, 2C, H, W)
        
        # Project back to C channels
        f_proj = self.fusion_proj(f_cat)  # (B, C, H, W)
        
        # === ANCHOR-GUIDED RESIDUAL ===
        # Y = Norm(X + α · Proj(F_cat))
        fused = anchor + self.alpha * f_proj
        
        # Apply LayerNorm (need to reshape for LN)
        fused = rearrange(fused, 'b c h w -> b h w c')
        fused = self.norm(fused)
        fused = rearrange(fused, 'b h w c -> b c h w')
        
        return fused
    
    def get_branch_outputs(self, x: torch.Tensor) -> dict:
        """Get individual branch outputs for visualization/analysis.
        
        Args:
            x: (B, C, H, W) input features
            
        Returns:
            dict with 'anchor', 'local', 'global', 'fused' tensors
        """
        anchor = x
        f_local = self.local_branch(x)
        f_global = self.global_branch(x)
        
        f_cat = torch.cat([f_local, f_global], dim=1)
        f_proj = self.fusion_proj(f_cat)
        fused = anchor + self.alpha * f_proj
        
        fused = rearrange(fused, 'b c h w -> b h w c')
        fused = self.norm(fused)
        fused = rearrange(fused, 'b h w c -> b c h w')
        
        return {
            'anchor': anchor,
            'local': f_local,
            'global': f_global,
            'fused': fused,
            'alpha': self.alpha.item()
        }


# =============================================================================
# Convenience Factory
# =============================================================================

def build_dual_branch_fusion(
    channels: int = 64,
    d_state: int = 16,
    dilation: int = 2
) -> DualBranchFusion:
    """Factory function to build DualBranchFusion module.
    
    Args:
        channels: Feature dimension (default: 64)
        d_state: Mamba state dimension (default: 16)
        dilation: Dilation for local branch (default: 2)
        
    Returns:
        Configured DualBranchFusion instance
    """
    return DualBranchFusion(
        channels=channels,
        d_state=d_state,
        dilation=dilation
    )
