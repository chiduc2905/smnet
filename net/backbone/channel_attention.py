"""Lightweight channel attention module.

Reference: Squeeze-and-Excitation Networks (Hu et al., CVPR 2018)
https://arxiv.org/abs/1709.01507
"""
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention.
    
    Recalibrates channel-wise feature responses by explicitly modeling
    interdependencies between channels.
    
    Architecture:
        - Global Average Pooling: (B, C, H, W) -> (B, C, 1, 1)
        - FC -> ReLU -> FC -> Sigmoid: channel attention weights
        - Channel-wise rescaling
    
    Args:
        in_channels: Number of input channels
        reduction: Channel reduction ratio (default: 16)
    """
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        
        self.in_channels = in_channels
        reduced_channels = max(in_channels // reduction, 8)
        
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input feature maps
            
        Returns:
            (B, C, H, W) channel-recalibrated feature maps
        """
        B, C, H, W = x.shape
        
        # Squeeze: global average pooling
        y = self.squeeze(x).view(B, C)
        
        # Excitation: channel attention weights
        y = self.excitation(y).view(B, C, 1, 1)
        
        # Scale
        return x * y.expand_as(x)
