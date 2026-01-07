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
# ResidualChannelMix - Simple conv1x1 channel mixing (ConvNeXt-style)
# =============================================================================

class ResidualChannelMix(nn.Module):
    """Residual 1×1 Channel Mixing (NO attention, NO gating).
    
    Simple channel interaction following ConvNeXt/MetaFormer design:
        Y = X + α · Mix(X)
        Mix(X) = SiLU(GroupNorm(Conv1×1(X)))
    
    Why this is the BEST choice:
        - True channel mixing (not attention)
        - No memory state (unlike Mamba)
        - No gating/sigmoid (no conflict with ECA++)
        - Very stable for few-shot
    
    Args:
        channels: Number of input/output channels
        alpha_init: Initial residual scaling (default: 0.1)
    """
    
    def __init__(self, channels: int, alpha_init: float = 0.1):
        super().__init__()
        
        self.mix = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(8, channels // 4), num_channels=channels),
            nn.SiLU(inplace=True)
        )
        
        # Learnable residual scaling
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Y = X + α · Mix(X)"""
        return x + self.alpha * self.mix(x)


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


class ECAPlus(nn.Module):
    """Enhanced Channel Attention (ECA++) using 1D convolution.
    
    More efficient than SE block - uses 1D conv instead of FC layers.
    Adaptive kernel size based on channel count.
    """
    
    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        # Adaptive kernel size: k = |log2(C)/gamma + b/gamma|_odd
        t = int(abs(math.log2(channels) / gamma + b / gamma))
        k = t if t % 2 else t + 1  # Ensure odd
        k = max(k, 3)  # Minimum kernel size 3
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # GAP: (B, C, 1, 1) -> (B, C, 1) -> (B, 1, C)
        y = self.gap(x).view(B, C, 1).transpose(1, 2)  # (B, 1, C)
        # Conv1D across channels
        y = self.conv1d(y)  # (B, 1, C)
        y = self.sigmoid(y).transpose(1, 2).view(B, C, 1, 1)  # (B, C, 1, 1)
        return x * y


class LocalAGLKABranch(nn.Module):
    """Attention-Guided Large Kernel Attention (AG-LKA) Branch.
    
    Replaces ConvMixer++ with a more powerful large-kernel attention mechanism.
    
    Architecture (following user pseudocode):
        X1 = Norm(X)
        
        # Local stem
        X2 = DWConv3x3(X1) → SiLU
        
        # Large-kernel spatial (multi-scale)
        S = DWConv5x5(X2) + DWConv7x7_dilated(X2)
        
        # Spatial gate
        g = Sigmoid(Linear(GAP(X2)))
        S = S * g
        
        # Channel interaction (ECA++)
        c = Sigmoid(Conv1D(GAP(S)))
        F = S * c
        
        # Residual
        Y = X + α·F
    
    Args:
        channels: Number of input/output channels
        dilation: Dilation for 7x7 conv (default: 2)
    """
    
    def __init__(self, channels: int, dilation: int = 2, **kwargs):
        super().__init__()
        
        self.channels = channels
        
        # Pre-normalization
        self.norm = nn.LayerNorm(channels)
        
        # Local stem: DWConv 3x3 + SiLU
        self.local_stem = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, 
                      padding=1, groups=channels, bias=False),
            nn.SiLU(inplace=True)
        )
        
        # Large-kernel spatial: DWConv 5x5
        self.lk_conv5 = nn.Conv2d(
            channels, channels, kernel_size=5,
            padding=2, groups=channels, bias=False
        )
        
        # Large-kernel spatial: DWConv 7x7 dilated
        # Effective RF: 7 + (7-1)*(dilation-1) = 7 + 6*(2-1) = 13 with dilation=2
        self.lk_conv7_dilated = nn.Conv2d(
            channels, channels, kernel_size=7,
            padding=3 * dilation, dilation=dilation, 
            groups=channels, bias=False
        )
        
        # === Parallel Asymmetric Branch (NEW) ===
        # DWConv(1×9) for temporal/time-axis awareness
        self.asym_temporal = nn.Conv2d(
            channels, channels, kernel_size=(1, 9),
            padding=(0, 4), groups=channels, bias=False
        )
        # DWConv(7×1) for frequency-axis awareness
        self.asym_frequency = nn.Conv2d(
            channels, channels, kernel_size=(7, 1),
            padding=(3, 0), groups=channels, bias=False
        )
        # NOTE: α, β removed for debugging - using simple addition
        
        # Spatial gate: Sigmoid(Linear(GAP))
        self.spatial_gap = nn.AdaptiveAvgPool2d(1)
        self.spatial_gate_fc = nn.Sequential(
            nn.Linear(channels, channels, bias=False),
            nn.Sigmoid()
        )
        
        # Channel mixing: Residual Conv1x1 (ConvNeXt-style)
        # Simple: Y = X + α·SiLU(GroupNorm(Conv1x1(X)))
        # No Mamba, no gating, pure channel interaction
        self.channel_mix = ResidualChannelMix(channels, alpha_init=0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input features
            
        Returns:
            (B, C, H, W) output features with AG-LKA + Asymmetric Branches
        """
        B, C, H, W = x.shape
        
        # === Pre-Normalization ===
        # Need to reshape for LayerNorm: (B, C, H, W) -> (B, H, W, C)
        x1 = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x1 = self.norm(x1)
        x1 = x1.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # === Local Stem ===
        x2 = self.local_stem(x1)  # DWConv3x3 + SiLU
        
        # === Symmetric Branch: Large-Kernel Spatial (Multi-scale) ===
        x_sym = self.lk_conv5(x2) + self.lk_conv7_dilated(x2)  # DWConv5x5 + DWConv7x7_d
        
        # === Asymmetric Branches (NEW) ===
        x_t = self.asym_temporal(x2)   # DWConv(1×9) - temporal RF
        x_f = self.asym_frequency(x2)  # DWConv(7×1) - frequency RF
        
        # === Parallel Fusion: SIMPLE ADDITION (no α, β for debugging) ===
        s = x_sym + x_t + x_f
        
        # === Spatial Gate ===
        # g = Sigmoid(Linear(GAP(X2)))
        g = self.spatial_gap(x2).view(B, C)  # (B, C)
        g = self.spatial_gate_fc(g).view(B, C, 1, 1)  # (B, C, 1, 1)
        s = s * g  # Spatial gating
        
        # === Channel Mixing (Residual Conv1x1) ===
        f = self.channel_mix(s)  # Y = S + α·Mix(S), no external residual needed
        
        # === Final Residual (from input) ===
        y = x + 0.1 * f  # Weak residual from original input
        
        return y


# Keep old name as alias for backward compatibility
LocalMidBranch = LocalAGLKABranch


# =============================================================================
# Branch 2: Vision State Space (VSS) Block - Lightweight for 64x64 Few-Shot
# =============================================================================

class VSSBlock(nn.Module):
    """Lightweight VSS Block optimized for 64×64 few-shot learning.
    
    Architecture:
        Input X
            ↓
        ┌─────────────────────────────────────────────┐
        │ Pre-Normalization: X1 = LN(X)               │
        └─────────────────────────────────────────────┘
            ↓
        ┌─────────────────────────────────────────────┐
        │ SS2D Block:                                 │
        │   Linear (C→C) → DWConv(k=3) → SiLU        │
        │   → SS2D (4-way scan) → LN → Linear (C→C)   │
        └─────────────────────────────────────────────┘
            ↓
        ┌─────────────────────────────────────────────┐
        │ Residual: Y = X + SS2D_Block(X1)            │
        └─────────────────────────────────────────────┘
            ↓
        Output Y
    
    Key Design (optimized for few-shot):
        - expand=1: No channel expansion (C→C, not C→2C)
        - d_conv=3: Smaller kernel for short sequences
        - NO FFN: Removed to prevent overfitting
        - Single residual connection
        - 4-way SS2D with independent Mamba modules
    
    Args:
        d_model: Model dimension (channels)
        d_state: SSM state dimension (default: 4)
        d_conv: Local convolution width in Mamba (default: 3)
        dw_kernel: Depthwise conv kernel size (default: 3)
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 4,
        d_conv: int = 3,
        dw_kernel: int = 3,
        **kwargs  # Ignore unused args like expand, ffn_expand
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_inner = d_model  # expand=1, no expansion
        
        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba-ssm package is required. Install with: pip install mamba-ssm"
            )
        
        # =============================================
        # Pre-Normalization
        # =============================================
        self.norm1 = nn.LayerNorm(d_model)
        
        # =============================================
        # SS2D Block (Lightweight: C → C)
        # =============================================
        # Linear (no expansion)
        self.in_proj = nn.Linear(d_model, self.d_inner, bias=False)
        
        # Depthwise Conv (k=3)
        self.dw_conv = nn.Conv2d(
            self.d_inner, self.d_inner,
            kernel_size=dw_kernel,
            padding=dw_kernel // 2,
            groups=self.d_inner,
            bias=False
        )
        
        # Shared Mamba module for all 4 directions (Weight Sharing)
        self.mamba = Mamba(
            d_model=self.d_inner,
            d_state=d_state,
            d_conv=d_conv,
            expand=1
        )
        
        # Post-SS2D LayerNorm
        self.norm_ss2d = nn.LayerNorm(self.d_inner)
        
        # Output Linear
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # NO FFN - removed for few-shot learning
    
    def _ss2d_forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Apply 4-way SS2D scanning with SHARED weights.
        
        Args:
            x: (B, L, d_inner) input sequence
            H, W: spatial dimensions
            
        Returns:
            (B, L, d_inner) output after averaged 4-way scanning
        """
        B, L, D = x.shape
        
        # 1. Left-to-Right (row-major)
        y_lr = self.mamba(x)
        
        # 2. Right-to-Left (row-major reversed)
        x_rl = torch.flip(x, dims=[1])
        y_rl = self.mamba(x_rl)
        y_rl = torch.flip(y_rl, dims=[1])
        
        # 3. Top-to-Bottom (column-major)
        x_tb = x.view(B, H, W, D).permute(0, 2, 1, 3).contiguous().view(B, L, D)
        y_tb = self.mamba(x_tb)
        y_tb = y_tb.view(B, W, H, D).permute(0, 2, 1, 3).contiguous().view(B, L, D)
        
        # 4. Bottom-to-Top (column-major reversed)
        x_bt = torch.flip(x_tb, dims=[1])
        y_bt = self.mamba(x_bt)
        y_bt = torch.flip(y_bt, dims=[1])
        y_bt = y_bt.view(B, W, H, D).permute(0, 2, 1, 3).contiguous().view(B, L, D)
        
        # Average fusion
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
        # Pre-LN → SS2D Block → Residual
        # =============================================
        
        # Pre-Normalization: X1 = LN(X)
        x1 = self.norm1(x_flat)
        
        # SS2D Block:
        # 1. Linear projection (no expansion: C → C)
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
        
        # Residual: Y = X + SS2D_Block(X1)
        y = x_flat + h
        
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
    
    Args:
        channels: Number of input/output channels
        d_state: SSM state dimension for global branch (default: 4)
        dilation: Dilation rate for local branch mid-range conv (default: 2)
        mode: Ablation mode - 'both', 'local_only', 'global_only' (default: 'both')
    """
    
    def __init__(
        self,
        channels: int,
        d_state: int = 4,
        dilation: int = 2,
        mode: str = 'both'
    ):
        super().__init__()
        
        self.channels = channels
        self.mode = mode
        
        # Branch 1: Local-Mid Spatial & Channel (uses ChannelMamba)
        if mode in ['both', 'local_only']:
            self.local_branch = LocalMidBranch(channels, dilation=dilation)
        else:
            self.local_branch = None
        
        # Branch 2: Lightweight VSS Block (optimized for few-shot)
        if mode in ['both', 'global_only']:
            self.global_branch = VSSBlock(
                d_model=channels,
                d_state=d_state,
                d_conv=3  # Smaller kernel for short sequences
            )
        else:
            self.global_branch = None
        
        # Fusion: depends on mode
        if mode == 'both':
            self.fusion_proj = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
        elif mode in ['local_only', 'global_only']:
            # Single branch: no fusion needed, just identity-like
            self.fusion_proj = None
        
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
        
        if self.mode == 'both':
            # === BRANCH 1: Local-Mid Processing ===
            f_local = self.local_branch(x)  # (B, C, H, W)
            
            # === BRANCH 2: Global Processing ===
            f_global = self.global_branch(x)  # (B, C, H, W)
            
            # === FUSION ===
            f_cat = torch.cat([f_local, f_global], dim=1)  # (B, 2C, H, W)
            f_proj = self.fusion_proj(f_cat)  # (B, C, H, W)
            
        elif self.mode == 'local_only':
            f_proj = self.local_branch(x) - x  # Get residual of local branch
            
        elif self.mode == 'global_only':
            f_proj = self.global_branch(x) - x  # Get residual of global branch
        
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
