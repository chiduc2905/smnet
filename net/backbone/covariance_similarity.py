"""Covariance-based Metric Similarity for Few-Shot Classification.

Formula:
    Σ = (X - mean(X)) @ (X - mean(X))^T / (N-1)   # (C, C) covariance
    Y = Σ @ Q                                       # (C, d)
    F = (Q * Y).sum(dim=0)                          # (d,) - efficient diag
    score = mean(F) * logit_scale

CRITICAL:
    - NO L2-normalize on query
    - Center SUPPORT only, NOT query
    - Vectorized implementation (no loop)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CovarianceSimilarity(nn.Module):
    """Covariance-based similarity metric (VECTORIZED VERSION).
    
    Pipeline (per class):
        1. Compute covariance matrix Σ from support patches (centered)
        2. Y = Σ @ Q_all (VECTORIZED, no loop)
        3. F = (Q * Y).sum(dim=1) - efficient diag
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
        
        # Center the features along spatial dimension (SUPPORT ONLY!)
        X = X - X.mean(dim=1, keepdim=True)  # (C, d)
        
        # Covariance: Σ = X @ X^T / (d-1)
        cov = torch.mm(X, X.t()) / (d - 1 + self.eps)  # (C, C)
        
        return cov
    
    def forward(
        self,
        query_features: torch.Tensor,
        support_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute covariance-based similarity score (VECTORIZED).
        
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
        
        # Step 2: Vectorized query processing (NO loop, NO center query)
        # Q_all: (NQ, C, d) - NO normalize, NO center!
        Q_all = query_features.reshape(NQ, C, d)  # (NQ, C, d)
        
        # Y_all = Σ @ Q_all for all queries at once
        # cov: (C, C), Q_all: (NQ, C, d)
        # Use einsum for batched matmul: (C, C) @ (NQ, C, d) -> (NQ, C, d)
        Y_all = torch.einsum('cc,ncd->ncd', cov, Q_all)  # (NQ, C, d)
        
        # F_all = (Q * Y).sum(dim=1) - this is diag(Q^T Σ Q) for each query
        F_all = (Q_all * Y_all).sum(dim=1)  # (NQ, d)
        
        # Mean over spatial locations
        scores = F_all.mean(dim=1)  # (NQ,)
        
        # Apply learnable logit scale
        scores = scores * self.logit_scale
        
        return scores
