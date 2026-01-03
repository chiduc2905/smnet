"""Neural network modules for few-shot learning.

Provides reusable similarity/distance metrics extracted from baseline models,
backbone feature extraction modules, and few-shot networks.
"""

from . import metrics
from . import backbone
from .slot_fewshot import SMNet
from .class_aware_fewshot import ClassAwareSMNet, build_class_aware_smnet

__all__ = [
    'metrics',
    'backbone',
    'SMNet',
    'ClassAwareSMNet',
    'build_class_aware_smnet',
]
