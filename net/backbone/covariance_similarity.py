"""Covariance-based Metric Similarity for Few-Shot Classification.

Formula:
    Σ = (X - mean(X)) @ (X - mean(X))^T / (N-1)   # (C, C) covariance
    Q_norm = Q / ||Q||_2 per channel              # L2-normalize query
    F = (Q_norm * (Σ @ Q_norm)).sum(dim=0)        # efficient diag
    score = mean(F)

CRITICAL:
    - L2-normalize query per channel (bounds output)
    - Center SUPPORT only, NOT query
    - Vectorized implementation (no loop)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CovarianceSimilarity(nn.Module):
    """Covariance-based similarity metric with L2-normalized query.
    
    Pipeline (per class):
        1. Compute covariance matrix Σ from support patches (centered)
        2. L2-normalize query per channel (CRITICAL for bounded output!)
        3. F = (Q_norm * (Σ @ Q_norm)).sum(dim=1) - efficient diag
        4. score = mean(F)
    
    Args:
        eps: Small constant for numerical stability (default: 1e-8)
    """
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    
    def compute_covariance(self, support_features: torch.Tensor) -> torch.Tensor:
        """Compute covariance matrix from support features.
        
        Args:
            support_features: (Shot, C, H, W) support feature maps
            
        Returns:
            cov: (C, C) covariance matrix
        """
        B, C, h, w = support_features.shape
        
        # Reshape: (Shot, C, H, W) -> (C, Shot*H*W)
        support = support_features.permute(1, 0, 2, 3).contiguous().view(C, -1)
        
        # Center
        mean = torch.mean(support, dim=1, keepdim=True)
        centered = support - mean
        
        # Covariance: (C, C)
        N = h * w * B
        cov = centered @ centered.t() / (N - 1 + self.eps)
        
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
        
        # Step 2: Vectorized query processing with L2-norm per channel
        # Q_all: (NQ, C, d)
        Q_all = query_features.reshape(NQ, C, d)
        
        # L2-normalize per channel (CRITICAL - like reference!)
        # norm shape: (NQ, C, 1)
        Q_norm = torch.norm(Q_all, p=2, dim=2, keepdim=True)  # (NQ, C, 1)
        Q_all = Q_all / (Q_norm + self.eps)  # (NQ, C, d), each channel has ||.|| = 1
        
        # Y_all = Σ @ Q_all for all queries at once
        # Use einsum: (C, C) @ (NQ, C, d) -> (NQ, C, d)
        Y_all = torch.einsum('cc,ncd->ncd', cov, Q_all)  # (NQ, C, d)
        
        # F_all = (Q * Y).sum(dim=1) - this is diag(Q^T Σ Q) for each query
        F_all = (Q_all * Y_all).sum(dim=1)  # (NQ, d)
        
        # Mean over spatial locations
        scores = F_all.mean(dim=1)  # (NQ,)
        
        return scores
