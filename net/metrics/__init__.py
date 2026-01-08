"""Similarity metrics for few-shot learning.

Extracted from baseline models for reuse in proposed architectures.
"""

from .euclidean import EuclideanDistance, squared_euclidean_distance
from .cosine import CosineBlock, cosine_similarity
from .covariance import CovaBlock, SlotCovarianceBlock
from .relation import RelationBlock
from .local_knn import LocalKNN, local_knn_score
from .emd import SinkhornDistance, sinkhorn_distance
from .learned_distance import LearnedDistance
from .transformer import SetTransformer
from .prototype_refinement import (
    refine_prototype,
    refine_prototype_weighted,
    compute_class_prototypes
)
from .class_conditional_cosine import (
    ClassConditionalCosine,
    ClassConditionalCosineWithPatches
)

__all__ = [
    # Distance/Similarity functions
    'EuclideanDistance',
    'squared_euclidean_distance',
    'CosineBlock',
    'cosine_similarity',
    'CovaBlock',
    'SlotCovarianceBlock',
    'RelationBlock',
    'LocalKNN',
    'local_knn_score',
    'SinkhornDistance',
    'sinkhorn_distance',
    'LearnedDistance',
    'SetTransformer',
    # Prototype refinement
    'refine_prototype',
    'refine_prototype_weighted',
    'compute_class_prototypes',
    # Class-conditional cosine
    'ClassConditionalCosine',
    'ClassConditionalCosineWithPatches',
]
