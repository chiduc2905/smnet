"""Neural network modules for few-shot learning.

Provides reusable similarity/distance metrics extracted from baseline models,
and backbone feature extraction modules.
"""

from . import metrics
from . import backbone

__all__ = ['metrics', 'backbone']
