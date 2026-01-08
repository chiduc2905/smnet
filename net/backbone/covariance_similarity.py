"""Covariance-based Metric Similarity for Few-Shot Classification.

Matching reference implementation from few-shot-mamba:
    - Covariance: Σ = centered @ centered^T / (N-1)
    - L2-normalize query per channel: query = query / ||query||
    - F = diag(Q^T Σ Q)
    - score = mean(F)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CovarianceSimilarity(nn.Module):
    """Covariance-based similarity metric matching reference exactly.
    
    Args:
        eps: Small constant for numerical stability (default: 1e-8)
    """
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    
    def compute_covariance(self, support_features: torch.Tensor) -> torch.Tensor:
        """Compute covariance matrix from support features.
        
        Matches reference:
            support = permute(1,0,2,3).view(C, -1)
            support = support - mean
            cov = support @ support^T / (N-1)
        """
        B, C, h, w = support_features.shape
        
        # Reshape: (Shot, C, H, W) -> (C, Shot*H*W) - SAME as reference
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
        
        Matches reference exactly:
            query = query.view(C, -1)
            query = query / ||query||_2 (per channel)
            temp = query^T @ cov @ query  # (d, d)
            F = diag(temp)  # (d,)
            score = mean(F)
        """
        NQ, C, H, W = query_features.shape
        d = H * W
        
        # Step 1: Compute covariance matrix Σ from support
        cov = self.compute_covariance(support_features)  # (C, C)
        
        # Step 2: Process each query (matching reference loop)
        scores = []
        
        for i in range(NQ):
            query_sam = query_features[i]  # (C, H, W)
            query_sam = query_sam.view(C, -1)  # (C, d)
            
            # L2-normalize per channel - EXACTLY like reference
            query_sam_norm = torch.norm(query_sam, p=2, dim=1, keepdim=True)  # (C, 1)
            query_sam = query_sam / (query_sam_norm + self.eps)  # (C, d)
            
            # Compute Q^T @ Σ @ Q and take diagonal - EXACTLY like reference
            # temp = query_sam^T @ cov @ query_sam  # (d, C) @ (C, C) @ (C, d) -> (d, d)
            temp = query_sam.t() @ cov @ query_sam  # (d, d)
            F_i = temp.diag()  # (d,)
            
            # Mean over spatial locations
            score_i = F_i.mean()  # scalar
            scores.append(score_i)
        
        scores = torch.stack(scores)  # (NQ,)
        
        return scores
