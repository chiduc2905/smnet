"""SMNet Feature Extractor: Unified slot-based feature extraction.

Combines ConvMixer, Channel Attention, SS2D (4-way Mamba), Slot Attention, 
and Slot Mamba into a complete feature extraction pipeline for SMNet.
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple

from .convmixer import ConvMixerEncoder
from .channel_attention import ChannelAttention
from .pixel_mamba import SS2D
from .slot_attention import SlotAttention
from .slot_mamba import SlotMamba


class SlotFeatureExtractor(nn.Module):
    """Complete feature extractor combining all backbone modules.
    
    Pipeline:
        1. ConvMixer: Local spatial extraction
        2. ChannelAttention: Local channel interaction
        3. PixelMamba: Pixel-level global context
        4. SlotAttention: Semantic grouping into slots
        5. SlotMamba: Slot-level global reasoning
    
    Args:
        in_channels: Number of input channels (default: 3 for RGB)
        hidden_dim: Hidden dimension throughout the network (default: 256)
        num_slots: Maximum number of semantic slots K (default: 4)
        convmixer_depth: Number of ConvMixer blocks (default: 4)
        convmixer_kernel: ConvMixer kernel size (default: 9)
        patch_size: Initial patch embedding size (default: 4)
        slot_iters: Slot attention refinement iterations (default: 3)
        slot_mamba_layers: Number of SlotMamba layers (default: 2)
        learnable_slots: Whether slot count is learnable (default: True)
        channel_reduction: Channel attention reduction ratio (default: 16)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 256,
        num_slots: int = 4,
        convmixer_depth: int = 4,
        convmixer_kernel: int = 9,
        patch_size: int = 4,
        slot_iters: int = 3,
        slot_mamba_layers: int = 2,
        learnable_slots: bool = True,
        channel_reduction: int = 16
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_slots = num_slots
        self.learnable_slots = learnable_slots
        
        # Stage 1: ConvMixer for local spatial extraction
        self.convmixer = ConvMixerEncoder(
            in_channels=in_channels,
            dim=hidden_dim,
            depth=convmixer_depth,
            kernel_size=convmixer_kernel,
            patch_size=patch_size
        )
        
        # Stage 2: Channel attention for local channel interaction
        self.channel_attn = ChannelAttention(
            in_channels=hidden_dim,
            reduction=channel_reduction
        )
        
        # Stage 3: SS2D (4-way Mamba) for global spatial context
        self.pixel_mamba = SS2D(
            d_model=hidden_dim,
            d_state=16,
            d_conv=4,
            expand=2,
            dropout=0.0
        )
        
        # Stage 4: Slot Attention for semantic grouping
        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            dim=hidden_dim,
            iters=slot_iters,
            learnable_slots=learnable_slots
        )
        
        # Stage 5: Slot Mamba for slot-level global reasoning
        self.slot_mamba = SlotMamba(
            d_model=hidden_dim,
            d_state=16,
            d_conv=4,
            expand=2,
            num_layers=slot_mamba_layers
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_intermediates: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (B, C_in, H, W) input images
            return_intermediates: Whether to return intermediate features
            
        Returns:
            slots: (B, K, hidden_dim) slot descriptors
            slot_weights: (B, K) slot existence weights (if learnable_slots)
        """
        # Stage 1: Local spatial extraction
        features = self.convmixer(x)  # (B, hidden_dim, H', W')
        
        # Stage 2: Channel attention
        features = self.channel_attn(features)  # (B, hidden_dim, H', W')
        
        # Stage 3: Pixel-level global context
        features = self.pixel_mamba(features)  # (B, hidden_dim, H', W')
        
        # Stage 4: Slot attention
        slots, slot_weights = self.slot_attention(features)  # (B, K, hidden_dim)
        
        # Stage 5: Slot-level global reasoning
        slots = self.slot_mamba(slots)  # (B, K, hidden_dim)
        
        return slots, slot_weights
    
    def extract_weighted_slots(
        self,
        x: torch.Tensor,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """Extract slots weighted by existence scores.
        
        Args:
            x: (B, C_in, H, W) input images
            threshold: Threshold for slot activation
            
        Returns:
            (B, K, hidden_dim) weighted slot descriptors
        """
        slots, slot_weights = self.forward(x)
        
        if slot_weights is not None:
            return slots * slot_weights.unsqueeze(-1)
        
        return slots


def build_slot_extractor(
    image_size: int = 224,
    num_classes: int = 4,
    **kwargs
) -> SlotFeatureExtractor:
    """Factory function to build slot feature extractor.
    
    Args:
        image_size: Input image size (default: 224)
        num_classes: Number of classes (used as hint for num_slots)
        **kwargs: Additional arguments for SlotFeatureExtractor
        
    Returns:
        Configured SlotFeatureExtractor instance
    """
    # Default num_slots to num_classes if not specified
    if 'num_slots' not in kwargs:
        kwargs['num_slots'] = num_classes
    
    return SlotFeatureExtractor(**kwargs)
