"""SMNet Backbone: Feature extraction modules.

Provides all components for the USCMambaNet feature extractor.

Module hierarchy:
    M1: PatchEmbed2D       - Overlapping patch embedding
    M2: PatchMerging2D     - Hierarchical downsampling
    M3: DualBranchFusion   - Local + Global branches
    M4: UnifiedSpatialChannelAttention - Unified attention module
"""

from .convmixer import ConvMixerBlock, ConvMixerEncoder
from .channel_attention import ChannelAttention
from .pixel_mamba import SS2D, PixelMamba  # SS2D = 4-way scanning with shared weights
from .dual_branch_fusion import DualBranchFusion, LocalMidBranch, GlobalSS2DBranch
from .feature_extractor import PatchEmbed2D, PatchMerging2D, SpatialDownsample
from .unified_attention import UnifiedSpatialChannelAttention

__all__ = [
    # Legacy modules (still available)
    'ConvMixerBlock',
    'ConvMixerEncoder',
    'ChannelAttention',
    'SS2D',
    'PixelMamba',
    # M1-M2: Patch processing
    'PatchEmbed2D',
    'PatchMerging2D',
    'SpatialDownsample',
    # M3: Dual Branch Fusion
    'DualBranchFusion',
    'LocalMidBranch',
    'GlobalSS2DBranch',
    # M4: Unified Attention
    'UnifiedSpatialChannelAttention',
]
