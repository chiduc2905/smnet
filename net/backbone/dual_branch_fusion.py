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
        1. Depthwise Conv (k=3) + BN + GELU for local spatial modeling
        2. Dilated Depthwise Conv (k=3, d=2) + BN + GELU for mid-range spatial
        3. Lightweight spatial gating (sigmoid-activated DW conv)
        4. Pointwise (1×1) conv + BN + GELU for channel mixing
        5. SE-Lite for channel interaction
    
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
        
        # Channel mixing (pointwise) with BN + GELU
        self.channel_mix = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
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
        # Local spatial: DWConv + BN + GELU
        local = F.gelu(self.local_bn(self.local_dw(x)))
        
        # Mid-range spatial (dilated): DWConv + BN + GELU
        mid = F.gelu(self.mid_bn(self.mid_dw(local)))
        
        # Combine local and mid features
        combined = local + mid
        
        # Spatial gating
        gated = self.spatial_gate(combined)
        
        # Channel mixing (1x1 Conv + BN + GELU)
        mixed = self.channel_mix(gated)
        
        # Channel attention
        out = self.se(mixed)
        
        return out


# =============================================================================
# Branch 2: Vision State Space (VSS) Block
# =============================================================================

class VSSBlock(nn.Module):
    """Vision State Space (VSS) Block for global feature extraction.
    
    Standard VSS Block architecture from vision Mamba papers:
    
        Input X
            ↓
        ┌─────────────────────────────────────────────┐
        │ Pre-Normalization: X1 = LN(X)               │
        └─────────────────────────────────────────────┘
            ↓
        ┌─────────────────────────────────────────────┐
        │ SS2D Block:                                 │
        │   Linear (1×1 conv) → DWConv(k=3) → SiLU   │
        │   → SS2D (4-way scan) → LN → Linear (1×1)   │
        └─────────────────────────────────────────────┘
            ↓
        ┌─────────────────────────────────────────────┐
        │ First Residual: X2 = X + SS2D_Block(X1)     │
        └─────────────────────────────────────────────┘
            ↓
        ┌─────────────────────────────────────────────┐
        │ FFN: LN → Linear → SiLU → Linear            │
        └─────────────────────────────────────────────┘
            ↓
        ┌─────────────────────────────────────────────┐
        │ Second Residual: Y = X2 + FFN(LN(X2))       │
        └─────────────────────────────────────────────┘
            ↓
        Output Y
    
    Key Design:
        - NO gating branch (pure sequential)
        - Two residual connections (after SS2D, after FFN)
        - LayerNorm throughout (not BatchNorm)
        - Linear residual paths (no activation on residual)
        - 4-way SS2D with independent Mamba modules (no weight sharing)
    
    Args:
        d_model: Model dimension (channels)
        d_state: SSM state dimension (default: 4)
        d_conv: Local convolution width in Mamba (default: 4)
        expand: Expansion factor for SS2D inner dimension (default: 2)
        ffn_expand: FFN expansion factor (default: 4)
        dw_kernel: Depthwise conv kernel size (default: 3)
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 4,
        d_conv: int = 4,
        expand: int = 2,
        ffn_expand: int = 4,
        dw_kernel: int = 3
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.ffn_hidden = int(ffn_expand * d_model)
        
        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba-ssm package is required. Install with: pip install mamba-ssm"
            )
        
        # =============================================
        # Pre-Normalization
        # =============================================
        self.norm1 = nn.LayerNorm(d_model)
        
        # =============================================
        # SS2D Block
        # =============================================
        # Linear (1×1 conv equivalent)
        self.in_proj = nn.Linear(d_model, self.d_inner, bias=False)
        
        # Depthwise Conv (k=3)
        self.dw_conv = nn.Conv2d(
            self.d_inner, self.d_inner,
            kernel_size=dw_kernel,
            padding=dw_kernel // 2,
            groups=self.d_inner,
            bias=False
        )
        
        # 4 INDEPENDENT Mamba modules (no weight sharing)
        self.mamba_lr = Mamba(d_model=self.d_inner, d_state=d_state, d_conv=d_conv, expand=1)
        self.mamba_rl = Mamba(d_model=self.d_inner, d_state=d_state, d_conv=d_conv, expand=1)
        self.mamba_tb = Mamba(d_model=self.d_inner, d_state=d_state, d_conv=d_conv, expand=1)
        self.mamba_bt = Mamba(d_model=self.d_inner, d_state=d_state, d_conv=d_conv, expand=1)
        
        # Post-SS2D LayerNorm
        self.norm_ss2d = nn.LayerNorm(self.d_inner)
        
        # Output Linear (1×1 conv equivalent)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # =============================================
        # Lightweight FFN Block: C → C → C (no expansion)
        # =============================================
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.GELU(),
            nn.Linear(d_model, d_model, bias=False)
        )
    
    def _ss2d_forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Apply 4-way SS2D scanning (no weight sharing).
        
        Args:
            x: (B, L, d_inner) input sequence
            H, W: spatial dimensions
            
        Returns:
            (B, L, d_inner) output after averaged 4-way scanning
        """
        B, L, D = x.shape
        
        # 1. Left-to-Right (row-major)
        y_lr = self.mamba_lr(x)
        
        # 2. Right-to-Left (row-major reversed)
        x_rl = torch.flip(x, dims=[1])
        y_rl = self.mamba_rl(x_rl)
        y_rl = torch.flip(y_rl, dims=[1])
        
        # 3. Top-to-Bottom (column-major)
        x_tb = x.view(B, H, W, D).permute(0, 2, 1, 3).contiguous().view(B, L, D)
        y_tb = self.mamba_tb(x_tb)
        y_tb = y_tb.view(B, W, H, D).permute(0, 2, 1, 3).contiguous().view(B, L, D)
        
        # 4. Bottom-to-Top (column-major reversed)
        x_bt = torch.flip(x_tb, dims=[1])
        y_bt = self.mamba_bt(x_bt)
        y_bt = torch.flip(y_bt, dims=[1])
        y_bt = y_bt.view(B, W, H, D).permute(0, 2, 1, 3).contiguous().view(B, L, D)
        
        # Average fusion (standard in VSS)
        y_fused = (y_lr + y_rl + y_tb + y_bt) / 4.0
        
        return y_fused
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input features
            
        Returns:
            y: (B, C, H, W) output features (same resolution)
        """
        B, C, H, W = x.shape
        L = H * W
        
        # Flatten: (B, C, H, W) -> (B, H, W, C) -> (B, L, C)
        x_flat = x.permute(0, 2, 3, 1).reshape(B, L, C)
        
        # =============================================
        # Stage 1: Pre-LN → SS2D Block → Residual
        # =============================================
        
        # Pre-Normalization: X1 = LN(X)
        x1 = self.norm1(x_flat)
        
        # SS2D Block:
        # 1. Linear projection
        h = self.in_proj(x1)  # (B, L, d_inner)
        
        # 2. DWConv (need spatial reshape)
        h_2d = h.reshape(B, H, W, self.d_inner).permute(0, 3, 1, 2)  # (B, d_inner, H, W)
        h_2d = self.dw_conv(h_2d)
        h = h_2d.permute(0, 2, 3, 1).reshape(B, L, self.d_inner)  # (B, L, d_inner)
        
        # 3. SiLU activation
        h = F.silu(h)
        
        # 4. SS2D (4-way scan)
        h = self._ss2d_forward(h, H, W)
        
        # 5. LayerNorm
        h = self.norm_ss2d(h)
        
        # 6. Linear projection back
        h = self.out_proj(h)  # (B, L, C)
        
        # First Residual: X2 = X + SS2D_Block(X1)
        x2 = x_flat + h
        
        # =============================================
        # Stage 2: LN → FFN → Residual
        # =============================================
        
        # FFN: LN → Linear → SiLU → Linear
        h2 = self.ffn(self.norm2(x2))
        
        # Second Residual: Y = X2 + FFN(LN(X2))
        y = x2 + h2
        
        # Reshape back: (B, L, C) -> (B, C, H, W)
        y = y.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        return y


# Alias for backward compatibility
GlobalSS2DBranch = VSSBlock


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
        
        # Branch 2: Global VSS Block (standard vision Mamba)
        self.global_branch = VSSBlock(
            d_model=channels,
            d_state=d_state,
            d_conv=4,
            expand=2,
            ffn_expand=4
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
