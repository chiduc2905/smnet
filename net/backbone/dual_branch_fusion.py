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
from einops import rearrange, repeat
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
    
    Multi-scale local-to-mid-range spatial feature extraction with asymmetric
    convolutions for scalogram processing (time × frequency axes).
    
    Architecture (optimized for 128×128 input):
        X1 = Norm(X)
        
        # Local stem
        X2 = DWConv3x3(X1) → SiLU
        
        # Multi-scale 5×5 with different dilations
        # Effective RF: 5 (dil1), 9 (dil2), 13 (dil3)
        S = DWConv5x5_d1(X2) + DWConv5x5_d2(X2) + DWConv5x5_d3(X2)
        
        # Asymmetric branches for time-frequency awareness
        A = DWConv1×7(X2) + DWConv5×1(X2)
        
        # Fusion
        F = S + A
        
        # Spatial gate
        g = Sigmoid(Linear(GAP(X2)))
        F = F * g
        
        # Channel mixing
        Y = ChannelMix(F)
        
        # Residual
        Output = X + α·Y
    
    Args:
        channels: Number of input/output channels
    """
    
    def __init__(self, channels: int, **kwargs):
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
        
        # === Multi-scale 5×5 with dilations ===
        # Dilation 1: Effective RF = 5, captures local details
        self.lk_conv5_d1 = nn.Conv2d(
            channels, channels, kernel_size=5,
            padding=2, dilation=1,
            groups=channels, bias=False
        )
        
        # Dilation 2: Effective RF = 9, captures mid-range patterns
        self.lk_conv5_d2 = nn.Conv2d(
            channels, channels, kernel_size=5,
            padding=4, dilation=2,
            groups=channels, bias=False
        )
        
        # Dilation 3: Effective RF = 13, captures larger context
        self.lk_conv5_d3 = nn.Conv2d(
            channels, channels, kernel_size=5,
            padding=6, dilation=3,
            groups=channels, bias=False
        )
        
        # === Asymmetric Branches for Scalogram ===
        # DWConv(1×7) for temporal/time-axis awareness (horizontal)
        self.asym_temporal = nn.Conv2d(
            channels, channels, kernel_size=(1, 7),
            padding=(0, 3), groups=channels, bias=False
        )
        
        # DWConv(5×1) for frequency-axis awareness (vertical)
        self.asym_frequency = nn.Conv2d(
            channels, channels, kernel_size=(5, 1),
            padding=(2, 0), groups=channels, bias=False
        )
        
        # Spatial gate: Sigmoid(Linear(GAP))
        self.spatial_gap = nn.AdaptiveAvgPool2d(1)
        self.spatial_gate_fc = nn.Sequential(
            nn.Linear(channels, channels, bias=False),
            nn.Sigmoid()
        )
        
        # === Normalization + Activation after branch sums ===
        # Multi-scale norm: LayerNorm + SiLU after DWConv5×5_d1 + d2 + d3
        self.ms_norm = nn.LayerNorm(channels)
        
        # Asymmetric norm: LayerNorm + SiLU after DWConv1×7 + DWConv5×1
        self.asym_norm = nn.LayerNorm(channels)
        
        # Channel mixing: Residual Conv1x1 (ConvNeXt-style)
        self.channel_mix = ResidualChannelMix(channels, alpha_init=0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input features
            
        Returns:
            (B, C, H, W) output features with multi-scale + asymmetric
        """
        B, C, H, W = x.shape
        
        # === Pre-Normalization ===
        x1 = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x1 = self.norm(x1)
        x1 = x1.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # === Local Stem ===
        x2 = self.local_stem(x1)  # DWConv3x3 + SiLU
        
        # === Multi-scale 5×5 Branch ===
        # Sum → LN → SiLU
        x_ms = self.lk_conv5_d1(x2) + self.lk_conv5_d2(x2) + self.lk_conv5_d3(x2)
        x_ms = x_ms.permute(0, 2, 3, 1)  # (B, H, W, C)
        x_ms = self.ms_norm(x_ms)
        x_ms = x_ms.permute(0, 3, 1, 2)  # (B, C, H, W)
        x_ms = F.silu(x_ms)
        
        # === Asymmetric Branches ===
        # Sum → LN → SiLU
        x_asym = self.asym_temporal(x2) + self.asym_frequency(x2)
        x_asym = x_asym.permute(0, 2, 3, 1)  # (B, H, W, C)
        x_asym = self.asym_norm(x_asym)
        x_asym = x_asym.permute(0, 3, 1, 2)  # (B, C, H, W)
        x_asym = F.silu(x_asym)
        
        # === Parallel Fusion ===
        s = x_ms + x_asym
        
        # === Spatial Gate ===
        g = self.spatial_gap(x2).view(B, C)  # (B, C)
        g = self.spatial_gate_fc(g).view(B, C, 1, 1)  # (B, C, 1, 1)
        s = s * g  # Spatial gating
        
        # === Channel Mixing ===
        f = self.channel_mix(s)  # Y = S + α·Mix(S)
        
        # === Final Residual ===
        y = x + 0.1 * f
        
        return y


# Keep old name as alias for backward compatibility
LocalMidBranch = LocalAGLKABranch


# =============================================================================
# Branch 2: Vision State Space (VSS) Block - Lightweight for 64x64 Few-Shot
# =============================================================================

class SS2D(nn.Module):
    """Selective Scan 2D - MambaU-Lite style with 4 separate weight sets.
    
    Architecture matching the VSS Block diagram:
        Input (B, H, W, C)
            ↓
        Linear (C → 2*d_inner)  →  Split into x, z
            ↓                         ↓
        DWConv → SiLU               SiLU
            ↓                         │
        SS2D (4-way scan)             │
            ↓                         │
        LayerNorm                     │
            ↓                         │
            └──── × (multiply) ───────┘
                      ↓
                Linear (d_inner → C)
                      ↓
                    Output
    
    Key: Uses 4 SEPARATE weight sets for 4 scanning directions (K=4).
    
    Args:
        d_model: Model dimension (channels)
        d_state: SSM state dimension (default: 16)
        d_conv: Local convolution width (default: 3)
        expand: Expansion factor for d_inner (default: 2)
        dropout: Dropout rate (default: 0.0)
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 8,       # Few-shot optimized (was 16)
        d_conv: int = 3,
        expand: int = 1,        # Few-shot optimized (was 2)
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        dropout: float = 0.0,
        conv_bias: bool = True,
        bias: bool = False,
        **kwargs
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        
        # Input projection: C → 2*d_inner (for gating)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        
        # Depthwise Conv2D
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )
        self.act = nn.SiLU()
        
        # =============================================
        # 4 SEPARATE weight sets for 4 directions (K=4)
        # =============================================
        # x_proj: Projects to (dt_rank + d_state*2) for each direction
        x_proj_list = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
            for _ in range(4)
        ]
        self.x_proj_weight = nn.Parameter(
            torch.stack([t.weight for t in x_proj_list], dim=0)  # (K=4, dt_rank+2*d_state, d_inner)
        )
        
        # dt_projs: Projects dt_rank → d_inner for each direction
        dt_projs_list = [
            self._dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, 
                          dt_min, dt_max, dt_init_floor)
            for _ in range(4)
        ]
        self.dt_projs_weight = nn.Parameter(
            torch.stack([t.weight for t in dt_projs_list], dim=0)  # (K=4, d_inner, dt_rank)
        )
        self.dt_projs_bias = nn.Parameter(
            torch.stack([t.bias for t in dt_projs_list], dim=0)  # (K=4, d_inner)
        )
        
        # A_logs: SSM state matrix (log form) - K=4 copies
        self.A_logs = self._A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        
        # Ds: Skip connection parameter - K=4 copies
        self.Ds = self._D_init(self.d_inner, copies=4, merge=True)
        
        # Output normalization and projection
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
    
    @staticmethod
    def _dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", 
                 dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        """Initialize dt projection layer."""
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        
        return dt_proj
    
    @staticmethod
    def _A_log_init(d_state, d_inner, copies=1, merge=True):
        """Initialize A in log form (S4D real initialization)."""
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log
    
    @staticmethod
    def _D_init(d_inner, copies=1, merge=True):
        """Initialize D (skip parameter)."""
        D = torch.ones(d_inner)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D
    
    def forward_core(self, x: torch.Tensor):
        """Apply 4-way SS2D scanning with separate weights.
        
        Args:
            x: (B, d_inner, H, W) input after DWConv + SiLU
            
        Returns:
            y: (B, d_inner, H, W) output from 4-way scan fusion
        """
        try:
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        except ImportError:
            raise ImportError("mamba_ssm.ops.selective_scan_interface is required")
        
        B, C, H, W = x.shape
        L = H * W
        K = 4
        
        # Create 4 scanning directions
        # x_hwwh: (B, 2, d_inner, L) - row-major and col-major
        x_hwwh = torch.stack([
            x.view(B, -1, L),  # row-major (H, W) -> L
            torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)  # col-major (W, H) -> L
        ], dim=1)
        
        # xs: (B, K=4, d_inner, L) - forward and reversed for both
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)
        
        # Project to get dt, B, C for SSM
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        
        # Prepare for selective scan
        xs = xs.float().view(B, -1, L)  # (B, K*d_inner, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (B, K*d_inner, L)
        Bs = Bs.float().view(B, K, -1, L)  # (B, K, d_state, L)
        Cs = Cs.float().view(B, K, -1, L)  # (B, K, d_state, L)
        Ds = self.Ds.float().view(-1)  # (K*d_inner,)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (K*d_inner, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (K*d_inner,)
        
        # Selective scan
        out_y = selective_scan_fn(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        
        # Reverse and transpose back
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        
        # Fuse 4 directions: sum
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        
        return y
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, C) input features
            
        Returns:
            y: (B, H, W, C) output features
        """
        B, H, W, C = x.shape
        
        # 1. Input projection → Split (Gating)
        xz = self.in_proj(x)  # (B, H, W, 2*d_inner)
        x_feat, z = xz.chunk(2, dim=-1)  # Each (B, H, W, d_inner)
        
        # 2. Branch 1: DWConv → SiLU → SS2D → LN
        x_feat = x_feat.permute(0, 3, 1, 2).contiguous()  # (B, d_inner, H, W)
        x_feat = self.act(self.conv2d(x_feat))  # DWConv + SiLU
        
        # SS2D (4-way scan)
        y = self.forward_core(x_feat)  # (B, H, W, d_inner)
        y = self.out_norm(y)  # LayerNorm
        
        # 3. Branch 2: Gating
        # 4. Merge: y * SiLU(z)
        y = y * F.silu(z)
        
        # 5. Output projection
        out = self.out_proj(y)  # (B, H, W, C)
        out = self.dropout(out)
        
        return out


class VSSBlock(nn.Module):
    """VSS Block - MambaU-Lite style.
    
    Architecture (matching the diagram):
        Input X (B, C, H, W)
            ↓
        LN (LayerNorm)
            ↓
        SS2D Module (contains: Linear→Split→DW→SiLU→Scan→LN→×Gate→Linear)
            ↓
        Residual: Y = X + SS2D(LN(X))
            ↓
        Output Y
    
    Args:
        d_model: Model dimension (channels)
        d_state: SSM state dimension (default: 16)
        d_conv: Local convolution width (default: 3)
        expand: Expansion factor (default: 2)
        drop_path: Drop path rate (default: 0.0)
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 8,       # Few-shot optimized
        d_conv: int = 3,
        expand: int = 1,        # Few-shot optimized
        drop_path: float = 0.1, # Regularization for few-shot
        **kwargs
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Pre-normalization
        self.ln_1 = nn.LayerNorm(d_model)
        
        # SS2D with gating (contains the full VSS architecture)
        self.self_attention = SS2D(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            **kwargs
        )
        
        # Drop path for regularization
        from timm.layers import DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input features
            
        Returns:
            y: (B, C, H, W) output features
        """
        B, C, H, W = x.shape
        
        # Reshape: (B, C, H, W) -> (B, H, W, C)
        x_hwc = x.permute(0, 2, 3, 1).contiguous()
        
        # Residual: Y = X + DropPath(SS2D(LN(X)))
        y = x_hwc + self.drop_path(self.self_attention(self.ln_1(x_hwc)))
        
        # Reshape back: (B, H, W, C) -> (B, C, H, W)
        y = y.permute(0, 3, 1, 2).contiguous()
        
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
        d_state: int = 8,  # Few-shot optimized
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
        
        # Branch 2: MambaU-Lite style VSS Block
        if mode in ['both', 'global_only']:
            self.global_branch = VSSBlock(
                d_model=channels,
                d_state=d_state,  # Default 16 from function param
                d_conv=3,
                expand=1  # Few-shot optimized (was 2)
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
