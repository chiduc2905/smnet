"""Learned distance metric for few-shot learning.

Extracted from: SiameseNet (Koch et al., ICML-W 2015)
"Siamese Neural Networks for One-shot Image Recognition"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedDistance(nn.Module):
    """Learned distance module using MLP on feature differences.
    
    Takes L1 distance (absolute difference) of embeddings as input
    and learns to output a similarity score.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        """Initialize LearnedDistance.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
        """
        super(LearnedDistance, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, query: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        """Compute learned similarity score.
        
        Args:
            query: (B, D) query embeddings
            support: (B, D) support embeddings
            
        Returns:
            scores: (B, 1) similarity scores in [0, 1]
        """
        # L1 distance as input
        diff = torch.abs(query - support)
        return self.net(diff)
    
    def forward_pairs(self, query: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        """Compute similarity for all query-support pairs.
        
        Args:
            query: (NQ, D) query embeddings
            support: (Way, D) support embeddings (prototypes)
            
        Returns:
            scores: (NQ, Way) similarity matrix
        """
        NQ, D = query.size()
        Way = support.size(0)
        
        # Expand for pairwise comparison
        # query: (NQ, D) -> (NQ, Way, D)
        # support: (Way, D) -> (NQ, Way, D)
        q_exp = query.unsqueeze(1).expand(-1, Way, -1)
        s_exp = support.unsqueeze(0).expand(NQ, -1, -1)
        
        # L1 distance
        diff = torch.abs(q_exp - s_exp)  # (NQ, Way, D)
        diff = diff.view(-1, D)  # (NQ*Way, D)
        
        # Compute similarity
        sims = self.net(diff)  # (NQ*Way, 1)
        scores = sims.view(NQ, Way)  # (NQ, Way)
        
        return scores
