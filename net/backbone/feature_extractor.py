"""SMNet Feature Extractor: Unified slot-based feature extraction.

Pipeline (v4 - 224×224 input):
    1. PatchEmbed2D: Overlapping patch embedding (stride=1)
    2. PatchMerging2D: Swin-style hierarchical patch merging (224→112→56→28)
    3. DualBranchFusion: Parallel local-global feature extraction
    4. Slot Attention: Semantic grouping into K slots
    5. Slot Mamba: Slot-level global reasoning

Changes from v3:
    - Input size changed from 64×64 to 224×224 (grayscale)
    - Replaced SpatialDownsample with PatchMerging2D (Swin-style)
    - Hierarchical channel expansion: C → 2C → 4C → 8C
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


# =============================================================================
# Main Feature Extractor
# =============================================================================

class SlotFeatureExtractor(nn.Module):
    """Complete feature extractor with PatchEmbed2D and PatchMerging2D.
    
    Pipeline (v4 - 224×224 input):
        1. PatchEmbed2D: Overlapping patch embedding (1, 224, 224) → (C, 224, 224)
        2. PatchMerging2D stages: Hierarchical spatial reduction
           - Stage 1: (C, 224, 224) → (2C, 112, 112)
           - Stage 2: (2C, 112, 112) → (4C, 56, 56)
           - Stage 3: (4C, 56, 56) → (8C, 28, 28)
        3. Channel projection: 8C → hidden_dim (for lightweight slot attention)
        4. DualBranchFusion: Parallel local-global with anchor-guided fusion
        5. SlotAttention: Semantic grouping into slots
        6. SlotMamba: Slot-level global reasoning
    
    Tensor sizes for default config (C=32, hidden_dim=64):
        Input:       (B, 1, 224, 224)
        PatchEmbed:  (B, 32, 224, 224)   [50,176 tokens × 32 dim]
        Merge1:      (B, 64, 112, 112)   [12,544 tokens × 64 dim]
        Merge2:      (B, 128, 56, 56)    [3,136 tokens × 128 dim]
        Merge3:      (B, 256, 28, 28)    [784 tokens × 256 dim]
        Proj:        (B, 64, 28, 28)     [784 tokens × 64 dim]
        DualBranch:  (B, 64, 28, 28)
        Slots:       (B, K, 64)
    
    Args:
        in_channels: Number of input channels (default: 1 for grayscale)
        base_dim: Base embedding dimension (default: 32, doubles each stage)
        hidden_dim: Final hidden dimension for slots (default: 64)
        num_slots: Maximum number of semantic slots K (default: 4)
        patch_kernel: Patch embedding kernel size (default: 3)
        num_merging_stages: Number of PatchMerging2D stages (default: 3)
        slot_iters: Slot attention refinement iterations (default: 3)
        slot_mamba_layers: Number of SlotMamba layers (default: 1)
        learnable_slots: Whether slot count is learnable (default: True)
        dual_branch_dilation: Dilation for local branch mid-range conv (default: 2)
        d_state: Mamba state dimension (default: 16)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        base_dim: int = 32,
        hidden_dim: int = 64,
        num_slots: int = 4,
        patch_kernel: int = 3,
        num_merging_stages: int = 3,
        slot_iters: int = 3,
        slot_mamba_layers: int = 1,
        learnable_slots: bool = True,
        dual_branch_dilation: int = 2,
        d_state: int = 16
    ):
        super().__init__()
        
        self.base_dim = base_dim
        self.hidden_dim = hidden_dim
        self.num_slots = num_slots
        self.learnable_slots = learnable_slots
        self.num_merging_stages = num_merging_stages
        
        # Stage 1: Overlapping Patch Embedding (stride=1)
        # (B, 1, 224, 224) → (B, base_dim, 224, 224)
        self.patch_embed = PatchEmbed2D(
            in_channels=in_channels,
            embed_dim=base_dim,
            kernel_size=patch_kernel,
            norm_layer=nn.LayerNorm
        )
        
        # Stage 2: Hierarchical PatchMerging2D stages
        # Each stage: spatial /2, channels ×2
        self.merging_stages = nn.ModuleList()
        current_dim = base_dim
        
        for i in range(num_merging_stages):
            self.merging_stages.append(
                PatchMerging2D(dim=current_dim, norm_layer=nn.LayerNorm)
            )
            current_dim = current_dim * 2  # Channels double each stage
        
        # Final channel dimension after merging stages
        # For 3 stages with base_dim=32: 32 → 64 → 128 → 256
        self.final_merge_dim = current_dim
        
        # Stage 3: Channel Projection (reduce to hidden_dim for slot attention)
        # 256 → 64 (if base_dim=32, num_merging_stages=3)
        self.channel_proj = nn.Sequential(
            nn.Conv2d(self.final_merge_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )
        
        # Stage 4: DualBranchFusion (Local + Global with anchor residual)
        self.dual_branch = DualBranchFusion(
            channels=hidden_dim,
            d_state=d_state,
            dilation=dual_branch_dilation
        )
        
        # Stage 5: Slot Attention for semantic grouping
        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            dim=hidden_dim,
            iters=slot_iters,
            learnable_slots=learnable_slots
        )
        
        # Stage 6: Slot Mamba for slot-level global reasoning
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
            x: (B, C_in, H, W) input images (expected: B, 1, 224, 224)
            return_intermediates: Whether to return intermediate features
            
        Returns:
            slots: (B, K, hidden_dim) slot descriptors
            slot_weights: (B, K) slot existence weights (if learnable_slots)
        """
        # Stage 1: Overlapping patch embedding
        # (B, 1, 224, 224) → (B, base_dim, 224, 224)
        features = self.patch_embed(x)
        
        # Stage 2: Hierarchical patch merging
        # (B, 32, 224, 224) → (B, 64, 112, 112) → (B, 128, 56, 56) → (B, 256, 28, 28)
        for merge_layer in self.merging_stages:
            features = merge_layer(features)
        
        # Stage 3: Channel projection
        # (B, 256, 28, 28) → (B, 64, 28, 28)
        features = self.channel_proj(features)
        
        # Stage 4: Dual-branch local-global fusion
        features = self.dual_branch(features)  # (B, hidden_dim, 28, 28)
        
        # Stage 5: Slot attention - semantic grouping
        slots, slot_weights = self.slot_attention(features)  # (B, K, hidden_dim)
        
        # Stage 6: Slot-level global reasoning
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
        # Stage 1: Patch embedding
        patch_features = self.patch_embed(x)
        
        # Stage 2: Hierarchical merging
        merge_outputs = [patch_features]
        features = patch_features
        for merge_layer in self.merging_stages:
            features = merge_layer(features)
            merge_outputs.append(features)
        
        # Stage 3: Channel projection
        proj_features = self.channel_proj(features)
        
        # Stage 4: Dual-branch - detailed
        branch_outputs = self.dual_branch.get_branch_outputs(proj_features)
        
        # Stage 5 & 6: Slots
        slots, slot_weights = self.slot_attention(branch_outputs['fused'])
        slots = self.slot_mamba(slots)
        
        return {
            'patch_embed_out': patch_features,
            'merge_stage_outputs': merge_outputs,
            'channel_proj_out': proj_features,
            'local_branch': branch_outputs['local'],
            'global_branch': branch_outputs['global'],
            'fused': branch_outputs['fused'],
            'alpha': branch_outputs['alpha'],
            'slots': slots,
            'slot_weights': slot_weights
        }


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
    
    Note:
        For 224×224 input with default settings (base_dim=32, num_merging_stages=3):
        - Spatial: 224 → 112 → 56 → 28
        - Channels: 32 → 64 → 128 → 256 → 64 (after projection)
    """
    # Default num_slots to num_classes if not specified
    if 'num_slots' not in kwargs:
        kwargs['num_slots'] = num_classes
    
    return SlotFeatureExtractor(**kwargs)
