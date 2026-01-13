"""USCMambaNet Feature Extractor: Core feature extraction modules.

Provides basic feature extraction components:
    1. PatchEmbed2D: Overlapping patch embedding (stride=1)
    2. PatchMerging2D: Swin-style hierarchical patch merging
    3. SpatialDownsample: Deprecated, use PatchMerging2D instead
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple


# =============================================================================
# Patch Embedding (Overlapping, stride=1)
# =============================================================================

class PatchEmbed2D(nn.Module):
    """Overlapping Patch Embedding with stride=1.
    
    Projects input image to embedding dimension while preserving spatial resolution.
    Uses kernel_size=3, stride=1, padding='same' for overlapping patches.
    
    Args:
        in_channels: Number of input image channels (default: 3 for RGB)
        embed_dim: Embedding dimension (default: 64)
        kernel_size: Convolution kernel size (default: 3)
        norm_layer: Normalization layer (default: LayerNorm)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 64,
        kernel_size: int = 3,
        norm_layer: nn.Module = nn.LayerNorm
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Overlapping projection: stride=1, same padding
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2  # 'same' padding
        )
        
        # Normalization (applied in channel dimension)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, H, W) input images
            
        Returns:
            (B, embed_dim, H, W) embedded features
        """
        # Project: (B, C_in, H, W) -> (B, embed_dim, H, W)
        x = self.proj(x)
        
        # Apply LayerNorm if present
        if self.norm is not None:
            # (B, C, H, W) -> (B, H, W, C) -> norm -> (B, H, W, C) -> (B, C, H, W)
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
            x = x.permute(0, 3, 1, 2)
        
        return x


# =============================================================================
# Patch Merging (Swin-style hierarchical downsampling)
# =============================================================================

class PatchMerging2D(nn.Module):
    """Swin Transformer-style Patch Merging for hierarchical spatial reduction.
    
    Merges 2×2 adjacent patches into one, reducing spatial dimensions by 2x
    while increasing channel dimension to 2C (after linear projection).
    
    Process:
        1. Extract 4 sub-patches from 2×2 regions
        2. Concatenate along channel dimension: C → 4C
        3. Apply LayerNorm
        4. Linear projection: 4C → 2C
    
    Spatial change: (B, C, H, W) → (B, 2C, H/2, W/2)
    
    Args:
        dim: Input channel dimension
        norm_layer: Normalization layer (default: LayerNorm)
    """
    
    def __init__(self, dim: int, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        
        # Linear projection: 4C → 2C
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        
        # Normalization before reduction
        self.norm = norm_layer(4 * dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input features (NOTE: channel-first format)
            
        Returns:
            (B, 2C, H/2, W/2) merged features
        """
        B, C, H, W = x.shape
        
        # Ensure H and W are even
        assert H % 2 == 0 and W % 2 == 0, f"H ({H}) and W ({W}) must be divisible by 2"
        
        # Convert to channel-last for patch merging: (B, C, H, W) → (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        
        # Extract 4 corners of each 2×2 patch
        x0 = x[:, 0::2, 0::2, :]  # (B, H/2, W/2, C) - top-left
        x1 = x[:, 1::2, 0::2, :]  # (B, H/2, W/2, C) - bottom-left
        x2 = x[:, 0::2, 1::2, :]  # (B, H/2, W/2, C) - top-right
        x3 = x[:, 1::2, 1::2, :]  # (B, H/2, W/2, C) - bottom-right
        
        # Concatenate: (B, H/2, W/2, 4C)
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        
        # Normalize
        x = self.norm(x)
        
        # Linear projection: 4C → 2C
        x = self.reduction(x)  # (B, H/2, W/2, 2C)
        
        # Convert back to channel-first: (B, H/2, W/2, 2C) → (B, 2C, H/2, W/2)
        x = x.permute(0, 3, 1, 2)
        
        return x


class SpatialDownsample(nn.Module):
    """DEPRECATED: Use PatchMerging2D instead.
    
    Spatial downsampling using strided convolution + pooling.
    Kept for backward compatibility.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        scale_factor: int = 4
    ):
        super().__init__()
        
        if out_channels is None:
            out_channels = in_channels
        
        self.scale_factor = scale_factor
        
        if scale_factor == 4:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.GELU(),
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU()
            )
        elif scale_factor == 2:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU()
            )
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.AdaptiveAvgPool2d(64 // scale_factor)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(x)
