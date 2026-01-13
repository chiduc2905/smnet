"""Neural network modules for few-shot learning.

Provides reusable similarity/distance metrics extracted from baseline models,
backbone feature extraction modules, and few-shot networks.
"""

from . import metrics
from . import backbone
from .usc_mamba_net import USCMambaNet, build_usc_mamba_net

__all__ = [
    'metrics',
    'backbone',
    'USCMambaNet',
    'build_usc_mamba_net',
]
