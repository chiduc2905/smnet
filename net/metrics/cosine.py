"""Cosine similarity metric for few-shot learning.

Used by: CosineNet, MatchingNet, Baseline++
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_similarity(query: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity between query and prototypes.
    
    Args:
        query: (B, NQ, D) or (NQ, D) query embeddings
        prototypes: (B, Way, D) or (Way, D) prototype embeddings
        
    Returns:
        similarity: (B, NQ, Way) or (NQ, Way) cosine similarities in [-1, 1]
    """
    # Normalize
    q_norm = F.normalize(query, p=2, dim=-1)
    p_norm = F.normalize(prototypes, p=2, dim=-1)
    
    if query.dim() == 2:
        # (NQ, D) @ (D, Way) -> (NQ, Way)
        return torch.mm(q_norm, p_norm.t())
    else:
        # (B, NQ, D) @ (B, D, Way) -> (B, NQ, Way)
        return torch.bmm(q_norm, p_norm.transpose(1, 2))


class CosineBlock(nn.Module):
    """Cosine similarity module for prototype-based classification."""
    
    def __init__(self, temperature: float = 1.0, learnable_temp: bool = False):
        """Initialize CosineBlock.
        
        Args:
            temperature: Scaling factor for similarity (higher = sharper)
            learnable_temp: If True, temperature is a learnable parameter
        """
        super(CosineBlock, self).__init__()
        
        if learnable_temp:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.register_buffer('temperature', torch.tensor(temperature))
    
    def forward(self, query: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """Compute scaled cosine similarity.
        
        Args:
            query: (B, NQ, D) or (NQ, D) query embeddings
            prototypes: (B, Way, D) or (Way, D) prototype embeddings
            
        Returns:
            scores: (B, NQ, Way) or (NQ, Way) scaled cosine similarities
        """
        sim = cosine_similarity(query, prototypes)
        return sim * self.temperature
