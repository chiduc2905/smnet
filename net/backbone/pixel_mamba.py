"""SS2D (Selective Scan 2D) for global spatial modeling.

Reference: VMamba: Visual State Space Model (Liu et al., 2024)
https://arxiv.org/abs/2401.10166

Implements 4-way scanning with SHARED weights for parameter efficiency.
Based on few-shot-mamba implementation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math

try:
    from mamba_ssm import Mamba
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba-ssm not installed. SS2D will not be available.")


class SS2D(nn.Module):
    """Selective Scan 2D - Global spatial modeling via 4-way Mamba scanning.
    
    Uses 4 separate Mamba modules for each scanning direction (row, row-rev, col, col-rev).
    
    4 Scanning directions:
        1. Row-major (left to right)
        2. Row-major reversed (right to left)
        3. Column-major (top to bottom)
        4. Column-major reversed (bottom to top)
    
    Args:
        d_model: Model dimension (channel dimension)
        d_state: SSM state dimension (default: 16)
        d_conv: Local convolution width (default: 4)
        expand: Expansion factor for inner dimension (default: 1)
        dropout: Dropout rate (default: 0.0)
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        
        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba-ssm package is required for SS2D. "
                "Install with: pip install mamba-ssm"
            )
        
        # Input normalization and projection
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # 4 SEPARATE Mamba modules for 4 directions (Original SS2D design)
        # This increases parameters but allows learning direction-specific features
        self.mambas = nn.ModuleList([
            Mamba(
                d_model=self.d_inner,
                d_state=d_state,
                d_conv=d_conv,
                expand=1  # Already expanded by in_proj
            ) for _ in range(4)
        ])
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Final normalization
        self.out_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input feature maps
            
        Returns:
            (B, C, H, W) output feature maps with global spatial context
        """
        B, C, H, W = x.shape
        L = H * W
        
        # Flatten to sequence: (B, C, H, W) -> (B, H*W, C)
        x_seq = rearrange(x, 'b c h w -> b (h w) c')
        
        # Save residual
        residual = x_seq
        
        # Input projection with normalization
        x_seq = self.norm(x_seq)
        xz = self.in_proj(x_seq)  # (B, L, d_inner*2)
        x_proj, z = xz.chunk(2, dim=-1)  # Each (B, L, d_inner)
        
        # Create 4 scanning directions
        # 1. Row-major: keep as is
        x_row = x_proj
        
        # 2. Row-major reversed
        x_row_rev = torch.flip(x_proj, dims=[1])
        
        # 3. Column-major: reshape, transpose, flatten
        x_2d = x_proj.view(B, H, W, -1)
        x_col = x_2d.permute(0, 2, 1, 3).contiguous().view(B, L, -1)
        
        # 4. Column-major reversed
        x_col_rev = torch.flip(x_col, dims=[1])
        
        # Apply SEPARATE Mamba to each direction
        y_row = self.mambas[0](x_row)
        y_row_rev = self.mambas[1](x_row_rev)
        y_col = self.mambas[2](x_col)
        y_col_rev = self.mambas[3](x_col_rev)
        
        # Reverse the reversed directions back
        y_row_rev = torch.flip(y_row_rev, dims=[1])
        
        # Convert column-major back to row-major
        y_col = y_col.view(B, W, H, -1).permute(0, 2, 1, 3).contiguous().view(B, L, -1)
        
        y_col_rev = torch.flip(y_col_rev, dims=[1])
        y_col_rev = y_col_rev.view(B, W, H, -1).permute(0, 2, 1, 3).contiguous().view(B, L, -1)
        
        # Fuse all 4 directions (simple average)
        y = (y_row + y_row_rev + y_col + y_col_rev) / 4
        
        # Gate with z
        y = y * F.silu(z)
        
        # Output projection
        out = self.out_proj(y)
        out = self.dropout(out)
        out = self.out_norm(out)
        
        # Residual connection
        out = out + residual
        
        # Reshape back to spatial: (B, H*W, C) -> (B, C, H, W)
        out = rearrange(out, 'b (h w) c -> b c h w', h=H, w=W)
        
        return out


# Backward compatibility alias
PixelMamba = SS2D
