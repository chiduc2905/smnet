"""SMNet Feature Extractor: Unified slot-based feature extraction.

Pipeline (v3):
    1. PatchEmbed2D: Overlapping patch embedding (stride=1)
    2. Downsample: Optional spatial reduction for efficiency
    3. DualBranchFusion: Parallel local-global feature extraction
    4. Slot Attention: Semantic grouping into K slots
    5. Slot Mamba: Slot-level global reasoning

Changes from v2:
    - Replaced ConvMixer with simple PatchEmbed2D (stride=1, overlapping)
    - Added configurable downsampling for spatial reduction
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple

from .dual_branch_fusion import DualBranchFusion
from .slot_attention import SlotAttention
from .slot_mamba import SlotMamba


# =============================================================================
# Patch Embedding (Overlapping, stride=1)
# =============================================================================

class PatchEmbed2D(nn.Module):
    """Overlapping Patch Embedding with stride=1.
    
    Projects input image to embedding dimension while preserving spatial resolution.
    Uses kernel_size=3, stride=1, padding='same' for overlapping patches.
    
    Args:
        in_channels: Number of input image channels (default: 1)
        embed_dim: Embedding dimension (default: 64)
        kernel_size: Convolution kernel size (default: 3)
        norm_layer: Normalization layer (default: LayerNorm)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
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


class SpatialDownsample(nn.Module):
    """Spatial downsampling using strided convolution + pooling.
    
    Reduces spatial resolution by a given factor while increasing channel capacity.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels (default: same as input)
        scale_factor: Downsampling factor (default: 4, e.g., 64->16)
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
            # Two-stage downsampling: /2 -> /2
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.GELU(),
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU()
            )
        elif scale_factor == 2:
            # Single-stage downsampling
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU()
            )
        else:
            # Adaptive pooling for other factors
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.AdaptiveAvgPool2d(64 // scale_factor)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(x)


# =============================================================================
# Main Feature Extractor
# =============================================================================

class SlotFeatureExtractor(nn.Module):
    """Complete feature extractor with PatchEmbed2D and DualBranchFusion.
    
    Pipeline (v3):
        1. PatchEmbed2D: Overlapping patch embedding (stride=1)
        2. SpatialDownsample: Reduce spatial size (64 -> 16)
        3. DualBranchFusion: Parallel local-global with anchor-guided fusion
        4. SlotAttention: Semantic grouping into slots
        5. SlotMamba: Slot-level global reasoning
    
    Args:
        in_channels: Number of input channels (default: 1 for grayscale)
        hidden_dim: Hidden dimension throughout the network (default: 64)
        num_slots: Maximum number of semantic slots K (default: 4)
        patch_kernel: Patch embedding kernel size (default: 3)
        downsample_factor: Spatial downsampling factor (default: 4, 64->16)
        slot_iters: Slot attention refinement iterations (default: 3)
        slot_mamba_layers: Number of SlotMamba layers (default: 1)
        learnable_slots: Whether slot count is learnable (default: True)
        dual_branch_dilation: Dilation for local branch mid-range conv (default: 2)
        d_state: Mamba state dimension (default: 16)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 64,
        num_slots: int = 4,
        patch_kernel: int = 3,
        downsample_factor: int = 4,
        slot_iters: int = 3,
        slot_mamba_layers: int = 1,
        learnable_slots: bool = True,
        dual_branch_dilation: int = 2,
        d_state: int = 16
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_slots = num_slots
        self.learnable_slots = learnable_slots
        self.downsample_factor = downsample_factor
        
        # Stage 1: Overlapping Patch Embedding (stride=1)
        self.patch_embed = PatchEmbed2D(
            in_channels=in_channels,
            embed_dim=hidden_dim,
            kernel_size=patch_kernel,
            norm_layer=nn.LayerNorm
        )
        
        # Stage 2: Spatial Downsampling (64×64 -> 16×16)
        self.downsample = SpatialDownsample(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            scale_factor=downsample_factor
        )
        
        # Stage 3: DualBranchFusion (Local + Global with anchor residual)
        self.dual_branch = DualBranchFusion(
            channels=hidden_dim,
            d_state=d_state,
            dilation=dual_branch_dilation
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
            d_state=d_state,
            d_conv=4,
            expand=1,
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
        # Stage 1: Overlapping patch embedding
        features = self.patch_embed(x)  # (B, hidden_dim, H, W) - same spatial size
        
        # Stage 2: Spatial downsampling
        features = self.downsample(features)  # (B, hidden_dim, H/factor, W/factor)
        
        # Stage 3: Dual-branch local-global fusion
        features = self.dual_branch(features)  # (B, hidden_dim, H', W')
        
        # Stage 4: Slot attention - semantic grouping
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
    
    def get_branch_analysis(self, x: torch.Tensor) -> dict:
        """Get detailed branch outputs for visualization/analysis.
        
        Args:
            x: (B, C_in, H, W) input images
            
        Returns:
            dict with all intermediate outputs
        """
        # Stage 1
        patch_features = self.patch_embed(x)
        
        # Stage 2
        down_features = self.downsample(patch_features)
        
        # Stage 3 - detailed
        branch_outputs = self.dual_branch.get_branch_outputs(down_features)
        
        # Stage 4 & 5
        slots, slot_weights = self.slot_attention(branch_outputs['fused'])
        slots = self.slot_mamba(slots)
        
        return {
            'patch_embed_out': patch_features,
            'downsample_out': down_features,
            'local_branch': branch_outputs['local'],
            'global_branch': branch_outputs['global'],
            'fused': branch_outputs['fused'],
            'alpha': branch_outputs['alpha'],
            'slots': slots,
            'slot_weights': slot_weights
        }


def build_slot_extractor(
    image_size: int = 64,
    num_classes: int = 4,
    **kwargs
) -> SlotFeatureExtractor:
    """Factory function to build slot feature extractor.
    
    Args:
        image_size: Input image size (default: 64)
        num_classes: Number of classes (used as hint for num_slots)
        **kwargs: Additional arguments for SlotFeatureExtractor
        
    Returns:
        Configured SlotFeatureExtractor instance
    """
    # Default num_slots to num_classes if not specified
    if 'num_slots' not in kwargs:
        kwargs['num_slots'] = num_classes
    
    return SlotFeatureExtractor(**kwargs)
