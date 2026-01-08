"""Covariance-based Metric Similarity for Few-Shot Classification.

CORRECT APPROACH (proven to work):
    (A) Spatial pooling BEFORE covariance: 32×32 → 8×8
    (B) Energy score WITHOUT learnable: score = F.mean()
    (C) Z-score normalization across classes

NO Conv1D, NO learnable aggregation!
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CovarianceSimilarity(nn.Module):
    """Covariance-based similarity with spatial pooling and z-score.
    
    Pipeline:
        1. AvgPool2d(4×4) to reduce spatial: 32×32 → 8×8 (d=64)
        2. Compute regularized covariance Σ from support
        3. F = (Q * (Σ @ Q)).sum(dim=0) - efficient diag
        4. score = F.mean() - energy-based, no learnable
        5. Z-score normalize across classes
    
    Args:
        pool_size: Pooling kernel size (default: 4)
        reg_lambda: Regularization for covariance (default: 0.1)
        eps: Small constant for numerical stability (default: 1e-8)
    """
    
    def __init__(self, pool_size: int = 4, reg_lambda: float = 0.1, eps: float = 1e-8):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=pool_size, stride=pool_size)
        self.reg_lambda = reg_lambda
        self.eps = eps
    
    def compute_covariance(self, support_features: torch.Tensor) -> torch.Tensor:
        """Compute REGULARIZED covariance matrix from pooled support features.
        
        Args:
            support_features: (Shot, C, H, W) pooled support feature maps
            
        Returns:
            cov: (C, C) regularized covariance matrix
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
        
        # REGULARIZE: Σ + λI (prevents eigenvalue explosion)
        cov = cov + self.reg_lambda * torch.eye(C, device=cov.device)
        
        return cov
    
    def compute_score(self, query_sam: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        """Compute energy score = mean(diag(Q^T Σ Q)).
        
        Args:
            query_sam: (C, d) query features (NO normalize!)
            cov: (C, C) covariance matrix
            
        Returns:
            score: scalar energy score
        """
        # Efficient formula for diag(Q^T @ Σ @ Q)
        # Y = Σ @ Q: (C, C) @ (C, d) -> (C, d)
        Y = cov @ query_sam  # (C, d)
        
        # F = (Q * Y).sum(dim=0): element-wise multiply then sum over channels
        F = (query_sam * Y).sum(dim=0)  # (d,)
        
        # Energy score = mean(F) - NO learnable!
        score = F.mean()  # scalar
        
        return score
    
    def forward(
        self,
        query_features: torch.Tensor,
        support_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute covariance-based similarity score for ONE class.
        
        Args:
            query_features: (NQ, C, H, W) query feature maps
            support_features: (Shot, C, H, W) support for ONE class
            
        Returns:
            scores: (NQ,) similarity scores for this class
        """
        NQ, C, H, W = query_features.shape
        
        # Step 1: Spatial pooling to reduce variance (32×32 → 8×8)
        q_pooled = self.pool(query_features)  # (NQ, C, H/4, W/4)
        s_pooled = self.pool(support_features)  # (Shot, C, H/4, W/4)
        
        _, _, h, w = q_pooled.shape
        d = h * w  # Should be 64 for 32→8
        
        # Step 2: Compute REGULARIZED covariance Σ from support
        cov = self.compute_covariance(s_pooled)  # (C, C)
        
        # Step 3: For each query, compute energy score
        scores = []
        for i in range(NQ):
            query_sam = q_pooled[i].view(C, -1)  # (C, d) - NO normalize!
            score_i = self.compute_score(query_sam, cov)  # scalar
            scores.append(score_i)
        
        scores = torch.stack(scores)  # (NQ,)
        
        return scores


def normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    """Z-score normalize scores across classes.
    
    Args:
        scores: (NQ, Way) raw scores
        
    Returns:
        normalized: (NQ, Way) z-score normalized
    """
    mean = scores.mean(dim=1, keepdim=True)
    std = scores.std(dim=1, keepdim=True) + 1e-8
    return (scores - mean) / std
