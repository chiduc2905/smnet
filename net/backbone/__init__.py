"""SMNet Backbone: Feature extraction modules.

Provides all components for the SMNet feature extractor.

Module hierarchy:
    M1: PatchEmbed2D       - Overlapping patch embedding
    M2: PatchMerging2D     - Hierarchical downsampling
    M3: DualBranchFusion   - Local + Global branches
    M4: SlotAttention      - Semantic grouping with Mamba
    M5: SlotCovarianceAttention (SCA) - Slot importance via covariance
    M6: ChannelMetricAttention (CMA) - Channel-wise slot refinement
    M7: ClassAwareInferenceHead - Few-shot classification
"""

from .convmixer import ConvMixerBlock, ConvMixerEncoder
from .channel_attention import ChannelAttention
from .pixel_mamba import SS2D, PixelMamba  # SS2D = 4-way scanning with shared weights
from .slot_attention import SlotAttention  # Mamba-based Slot Attention
from .slot_mamba import SlotMamba
from .dual_branch_fusion import DualBranchFusion, LocalMidBranch, GlobalSS2DBranch
from .feature_extractor import SlotFeatureExtractor, PatchEmbed2D, PatchMerging2D, SpatialDownsample
from .class_aware_inference import ClassAwareInferenceHead, SlotAttentionWithPatches

# M5 & M6: Slot Refinement modules (NEW)
from .slot_covariance_attention import SlotCovarianceAttention, compute_class_covariance
from .channel_metric_attention import ChannelMetricAttention, compute_class_prototype

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
    # M4: Slot Attention
    'SlotAttention',  # Mamba-based
    'SlotMamba',
    'SlotFeatureExtractor',
    # M5: Slot Covariance Attention (NEW)
    'SlotCovarianceAttention',
    'compute_class_covariance',
    # M6: Channel Metric Attention (NEW)
    'ChannelMetricAttention',
    'compute_class_prototype',
    # M7: Class-Aware Inference
    'ClassAwareInferenceHead',
    'SlotAttentionWithPatches',
]
