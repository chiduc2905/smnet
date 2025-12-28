"""Euclidean distance metric for few-shot learning.

Extracted from: Prototypical Networks (Snell et al., NIPS 2017)
"""
import torch
import torch.nn as nn


def squared_euclidean_distance(query: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    """Compute squared Euclidean distance between query and prototypes.
    
    Args:
        query: (B, NQ, D) or (NQ, D) query embeddings
        prototypes: (B, Way, D) or (Way, D) prototype embeddings
        
    Returns:
        distances: (B, NQ, Way) or (NQ, Way) squared distances
    """
    # Use torch.cdist for efficient pairwise distance computation
    # cdist returns L2 distance, so we square it
    if query.dim() == 2:
        # (NQ, D) and (Way, D) -> (NQ, Way)
        return torch.cdist(query.unsqueeze(0), prototypes.unsqueeze(0)).pow(2).squeeze(0)
    else:
        # (B, NQ, D) and (B, Way, D) -> (B, NQ, Way)
        return torch.cdist(query, prototypes).pow(2)


class EuclideanDistance(nn.Module):
    """Euclidean distance module for prototype-based classification.
    
    Returns negative squared distances (higher = more similar).
    """
    
    def __init__(self):
        super(EuclideanDistance, self).__init__()
    
    def forward(self, query: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """Compute negative squared Euclidean distance.
        
        Args:
            query: (B, NQ, D) or (NQ, D) query embeddings
            prototypes: (B, Way, D) or (Way, D) prototype embeddings
            
        Returns:
            scores: (B, NQ, Way) or (NQ, Way) negative distances (similarity scores)
        """
        dists = squared_euclidean_distance(query, prototypes)
        return -dists  # Negative so higher = more similar
