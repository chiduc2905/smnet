"""SMNet Backbone: Feature extraction modules.

Provides all components for the SMNet feature extractor.
"""

from .convmixer import ConvMixerBlock, ConvMixerEncoder
from .channel_attention import ChannelAttention
from .pixel_mamba import SS2D, SS2DBlock, PixelMamba  # SS2D = 4-way scanning
from .slot_attention import SlotAttention
from .slot_mamba import SlotMamba
from .feature_extractor import SlotFeatureExtractor

__all__ = [
    'ConvMixerBlock',
    'ConvMixerEncoder',
    'ChannelAttention',
    'SS2D',
    'SS2DBlock',
    'PixelMamba',  # Alias for SS2D
    'SlotAttention',
    'SlotMamba',
    'SlotFeatureExtractor',
]
