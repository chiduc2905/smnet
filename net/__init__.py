"""Neural network modules for few-shot learning.

Provides reusable similarity/distance metrics extracted from baseline models,
backbone feature extraction modules, and few-shot networks.
"""

from . import metrics
from . import backbone
from .usc_mamba_net import USCMambaNet, build_usc_mamba_net
from .conv64f_token_sw_metric_net import Conv64FTokenSWMetricNet
from .class_memory_scan_mamba_net import ClassMemoryScanMambaNet
from .episodic_selective_scan_mamba_net import EpisodicSelectiveScanMambaNet
from .permutation_robust_class_memory_mamba_net import PermutationRobustClassMemoryMambaNet
from .hierarchical_episodic_ssm_net import HierarchicalEpisodicSSMNet
from .model_factory import MODEL_REGISTRY, MODEL_METADATA, build_model_from_args, get_model_choices

__all__ = [
    'metrics',
    'backbone',
    'USCMambaNet',
    'build_usc_mamba_net',
    'Conv64FTokenSWMetricNet',
    'ClassMemoryScanMambaNet',
    'EpisodicSelectiveScanMambaNet',
    'PermutationRobustClassMemoryMambaNet',
    'HierarchicalEpisodicSSMNet',
    'MODEL_REGISTRY',
    'MODEL_METADATA',
    'build_model_from_args',
    'get_model_choices',
]
