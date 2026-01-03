"""SS2D (Selective Scan 2D) for global spatial modeling.

Reference: VMamba: Visual State Space Model (Liu et al., 2024)
https://arxiv.org/abs/2401.10166

Implements 4-way scanning for 2D-aware state space modeling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba-ssm not installed. SS2D will not be available.")


class SS2D(nn.Module):
    """Selective Scan 2D - Global spatial modeling via 4-way Mamba scanning.
    
    Scans the 2D feature map in 4 directions to capture global spatial context
    while preserving 2D structure awareness:
        1. Top-left → Bottom-right (row-major)
        2. Bottom-right → Top-left (reverse row-major)
        3. Top-right → Bottom-left (column-major)
        4. Bottom-left → Top-right (reverse column-major)
    
    Architecture:
        - Input projection with LayerNorm
        - 4-way parallel Mamba scanning
        - Fusion via concatenation + Linear projection
        - Output projection with residual connection
    
    Args:
        d_model: Model dimension (channel dimension)
        d_state: SSM state dimension (default: 16)
        d_conv: Local convolution width (default: 4)
        expand: Expansion factor for inner dimension (default: 2)
        dropout: Dropout rate (default: 0.0)
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.d_model = d_model
        
        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba-ssm package is required for SS2D. "
                "Install with: pip install mamba-ssm"
            )
        
        # Input normalization and projection
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_model)
        
        # 4 Mamba modules for 4-way scanning
        self.mamba_forward_row = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.mamba_backward_row = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.mamba_forward_col = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.mamba_backward_col = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        # Fusion layer: concatenate 4 directions then project
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
        
        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout)
        )
        
        # Final normalization
        self.out_norm = nn.LayerNorm(d_model)
    
    def _scan_row_major(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Scan in row-major order (top-left to bottom-right)."""
        # x: (B, H*W, C) - already in row-major order
        return self.mamba_forward_row(x)
    
    def _scan_row_major_reverse(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Scan in reverse row-major order (bottom-right to top-left)."""
        x_rev = torch.flip(x, dims=[1])
        out = self.mamba_backward_row(x_rev)
        return torch.flip(out, dims=[1])
    
    def _scan_col_major(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Scan in column-major order (top-right to bottom-left)."""
        B, N, C = x.shape
        # Reshape to (B, H, W, C), transpose to column-major, flatten
        x_2d = x.view(B, H, W, C)
        x_col = x_2d.permute(0, 2, 1, 3).contiguous()  # (B, W, H, C)
        x_col = x_col.view(B, N, C)
        
        out = self.mamba_forward_col(x_col)
        
        # Convert back to row-major
        out_2d = out.view(B, W, H, C)
        out_row = out_2d.permute(0, 2, 1, 3).contiguous()
        return out_row.view(B, N, C)
    
    def _scan_col_major_reverse(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Scan in reverse column-major order (bottom-left to top-right)."""
        B, N, C = x.shape
        # Reshape to (B, H, W, C), transpose to column-major, reverse, flatten
        x_2d = x.view(B, H, W, C)
        x_col = x_2d.permute(0, 2, 1, 3).contiguous()  # (B, W, H, C)
        x_col = x_col.view(B, N, C)
        x_col_rev = torch.flip(x_col, dims=[1])
        
        out = self.mamba_backward_col(x_col_rev)
        out = torch.flip(out, dims=[1])
        
        # Convert back to row-major
        out_2d = out.view(B, W, H, C)
        out_row = out_2d.permute(0, 2, 1, 3).contiguous()
        return out_row.view(B, N, C)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input feature maps
            
        Returns:
            (B, C, H, W) output feature maps with global spatial context
        """
        B, C, H, W = x.shape
        
        # Flatten to sequence: (B, C, H, W) -> (B, H*W, C)
        x_seq = rearrange(x, 'b c h w -> b (h w) c')
        
        # Input projection with normalization
        x_seq = self.norm(x_seq)
        x_proj = self.in_proj(x_seq)
        
        # 4-way parallel scanning
        out_fwd_row = self._scan_row_major(x_proj, H, W)
        out_bwd_row = self._scan_row_major_reverse(x_proj, H, W)
        out_fwd_col = self._scan_col_major(x_proj, H, W)
        out_bwd_col = self._scan_col_major_reverse(x_proj, H, W)
        
        # Fuse all 4 directions
        fused = torch.cat([out_fwd_row, out_bwd_row, out_fwd_col, out_bwd_col], dim=-1)
        fused = self.fusion(fused)
        
        # Output projection
        out = self.out_proj(fused)
        out = self.out_norm(out)
        
        # Residual connection
        out = out + x_seq
        
        # Reshape back to spatial: (B, H*W, C) -> (B, C, H, W)
        out = rearrange(out, 'b (h w) c -> b c h w', h=H, w=W)
        
        return out


class SS2DBlock(nn.Module):
    """SS2D block with pre-normalization and feed-forward network.
    
    Architecture:
        - LayerNorm -> SS2D -> Residual
        - LayerNorm -> FFN -> Residual
    
    Args:
        d_model: Model dimension
        d_state: SSM state dimension
        d_conv: Convolution width
        expand: Expansion factor
        ffn_expand: FFN expansion factor (default: 4)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        ffn_expand: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()
        
        # SS2D layer
        self.ss2d = SS2D(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout
        )
        
        # Feed-forward network
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_expand),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_expand, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input feature maps
            
        Returns:
            (B, C, H, W) output feature maps
        """
        # SS2D with residual (already includes residual)
        x = self.ss2d(x)
        
        # FFN with residual
        B, C, H, W = x.shape
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        x_flat = x_flat + self.ffn(self.norm_ffn(x_flat))
        x = rearrange(x_flat, 'b (h w) c -> b c h w', h=H, w=W)
        
        return x


# Backward compatibility alias
PixelMamba = SS2D
