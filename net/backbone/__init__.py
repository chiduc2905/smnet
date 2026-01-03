"""SMNet Backbone: Feature extraction modules.

Provides all components for the SMNet feature extractor.
"""

from .convmixer import ConvMixerBlock, ConvMixerEncoder
from .channel_attention import ChannelAttention
from .pixel_mamba import SS2D, PixelMamba  # SS2D = 4-way scanning with shared weights
from .slot_attention import SlotAttention
from .slot_mamba import SlotMamba
from .dual_branch_fusion import DualBranchFusion, LocalMidBranch, GlobalSS2DBranch
from .feature_extractor import SlotFeatureExtractor, PatchEmbed2D, SpatialDownsample

__all__ = [
    # Legacy modules (still available)
    'ConvMixerBlock',
    'ConvMixerEncoder',
    'ChannelAttention',
    'SS2D',
    'PixelMamba',
    # Core modules
    'PatchEmbed2D',
    'SpatialDownsample',
    'DualBranchFusion',
    'LocalMidBranch',
    'GlobalSS2DBranch',
    'SlotAttention',
    'SlotMamba',
    'SlotFeatureExtractor',
]

