"""ConvMixer-based local spatial feature extraction.

Reference: Patches Are All You Need? (Trockman & Kolter, 2022)
https://arxiv.org/abs/2201.09792
"""
import torch
import torch.nn as nn


class ConvMixerBlock(nn.Module):
    """Single ConvMixer block with depthwise separable convolution.
    
    Architecture:
        - Depthwise Conv (spatial mixing with large kernel)
        - GELU + BatchNorm
        - Pointwise Conv (channel mixing)
        - GELU + BatchNorm
        - Residual connection around depthwise conv
    
    Args:
        dim: Number of channels
        kernel_size: Kernel size for depthwise convolution (default: 9)
    """
    
    def __init__(self, dim: int, kernel_size: int = 9):
        super().__init__()
        
        # Depthwise convolution (spatial mixing)
        self.depthwise = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, groups=dim, padding=kernel_size // 2),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )
        
        # Pointwise convolution (channel mixing)
        self.pointwise = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input feature maps
            
        Returns:
            (B, C, H, W) output feature maps
        """
        # Residual around depthwise conv
        x = x + self.depthwise(x)
        x = self.pointwise(x)
        return x


class ConvMixerEncoder(nn.Module):
    """ConvMixer encoder for local spatial feature extraction.
    
    Applies patch embedding followed by multiple ConvMixer blocks.
    
    Args:
        in_channels: Number of input channels (default: 3 for RGB)
        dim: Hidden dimension (default: 256)
        depth: Number of ConvMixer blocks (default: 4)
        kernel_size: Kernel size for depthwise convolution (default: 9)
        patch_size: Initial patch embedding size (default: 4)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        dim: int = 256,
        depth: int = 4,
        kernel_size: int = 9,
        patch_size: int = 4
    ):
        super().__init__()
        
        self.dim = dim
        self.patch_size = patch_size
        
        # Patch embedding: non-overlapping patches
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )
        
        # ConvMixer blocks
        self.blocks = nn.Sequential(*[
            ConvMixerBlock(dim, kernel_size) for _ in range(depth)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, H, W) input images
            
        Returns:
            (B, dim, H', W') feature maps where H' = H/patch_size, W' = W/patch_size
        """
        x = self.patch_embed(x)
        x = self.blocks(x)
        return x
