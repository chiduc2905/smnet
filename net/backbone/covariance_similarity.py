"""Covariance-based Metric Similarity for Few-Shot Classification.

Formula:
    Σ = (X - mean(X)) @ (X - mean(X))^T / (N-1)   # (C, C) covariance
    Y = Σ @ Q                                       # (C, d)
    F = (Q * Y).sum(dim=0)                          # (d,) - efficient diag
    score = mean(F) * logit_scale

CRITICAL:
    - NO L2-normalize on query (destroys second-order info)
    - NO full matrix (1024x1024) - use efficient formula
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CovarianceSimilarity(nn.Module):
    """Covariance-based similarity metric (CORRECTED VERSION).
    
    Pipeline (per class):
        1. Compute covariance matrix Σ from support patches
        2. Y = Σ @ Q (NO L2-normalize!)
        3. F = (Q * Y).sum(dim=0) - efficient diag computation
        4. score = mean(F) * logit_scale
    
    Args:
        logit_scale: Initial logit scale for boosting CE gradient (default: 10.0)
        eps: Small constant for numerical stability (default: 1e-8)
    """
    
    def __init__(self, logit_scale: float = 10.0, eps: float = 1e-8):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(logit_scale))
        self.eps = eps
    
    def compute_covariance(self, support_features: torch.Tensor) -> torch.Tensor:
        """Compute covariance matrix from support features.
        
        Args:
            support_features: (Shot, C, H, W) support feature maps
            
        Returns:
            cov: (C, C) covariance matrix
        """
        Shot, C, H, W = support_features.shape
        d = H * W
        
        # Reshape: (Shot, C, H, W) -> (Shot, C, d) -> mean -> (C, d)
        X = support_features.reshape(Shot, C, d)  # (Shot, C, d)
        X = X.mean(dim=0)  # (C, d) - average over shots
        
        # Center the features along spatial dimension
        X = X - X.mean(dim=1, keepdim=True)  # (C, d)
        
        # Covariance: Σ = X @ X^T / (d-1)
        cov = torch.mm(X, X.t()) / (d - 1 + self.eps)  # (C, C)
        
        return cov
    
    def forward(
        self,
        query_features: torch.Tensor,
        support_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute covariance-based similarity score.
        
        Args:
            query_features: (NQ, C, H, W) query feature maps
            support_features: (Shot, C, H, W) support feature maps for ONE class
            
        Returns:
            scores: (NQ,) similarity scores for this class
        """
        NQ, C, H, W = query_features.shape
        d = H * W
        
        # Step 1: Compute covariance matrix Σ from support
        cov = self.compute_covariance(support_features)  # (C, C)
        
        # Step 2: For each query, compute F efficiently (NO L2-normalize!)
        scores = []
        
        for i in range(NQ):
            # Q: (C, d) - NO normalize!
            Q = query_features[i].view(C, d)  # (C, d)
            
            # Efficient diagonal computation:
            # diag(Q^T Σ Q) = sum_c (Q * (Σ @ Q))
            Y = torch.mm(cov, Q)  # (C, d)
            F_i = (Q * Y).sum(dim=0)  # (d,) - this is diag(Q^T Σ Q)
            
            # Mean over spatial locations
            score_i = F_i.mean()  # scalar
            scores.append(score_i)
        
        scores = torch.stack(scores)  # (NQ,)
        
        # Apply learnable logit scale
        scores = scores * self.logit_scale
        
        return scores
