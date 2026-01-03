"""SMNet Backbone: Feature extraction modules.

Provides all components for the SMNet feature extractor.
"""

from .convmixer import ConvMixerBlock, ConvMixerEncoder
from .channel_attention import ChannelAttention
from .pixel_mamba import SS2D, PixelMamba  # SS2D = 4-way scanning with shared weights
from .slot_attention import SlotAttention  # Mamba-based Slot Attention
from .slot_mamba import SlotMamba
from .dual_branch_fusion import DualBranchFusion, LocalMidBranch, GlobalSS2DBranch
from .feature_extractor import SlotFeatureExtractor, PatchEmbed2D, PatchMerging2D, SpatialDownsample
from .class_aware_inference import ClassAwareInferenceHead, SlotAttentionWithPatches

__all__ = [
    # Legacy modules (still available)
    'ConvMixerBlock',
    'ConvMixerEncoder',
    'ChannelAttention',
    'SS2D',
    'PixelMamba',
    # Core modules
    'PatchEmbed2D',
    'PatchMerging2D',
    'SpatialDownsample',
    'DualBranchFusion',
    'LocalMidBranch',
    'GlobalSS2DBranch',
    'SlotAttention',  # Mamba-based
    'SlotMamba',
    'SlotFeatureExtractor',
    # Class-Aware Inference
    'ClassAwareInferenceHead',
    'SlotAttentionWithPatches',
]
