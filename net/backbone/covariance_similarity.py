"""Covariance-based Metric Similarity for Few-Shot Classification.

KEY PRINCIPLES:
    - NO L2-normalize query/support (preserves magnitude for second-order stats)
    - Covariance is feature extractor → need learnable aggregation (Conv1d)
    - Regularize covariance: Σ + λI for stability

Formula:
    Σ = (X - mean(X)) @ (X - mean(X))^T / (N-1) + λI   # regularized cov
    Y = Σ @ Q                                            # (C, d)
    F = (Q * Y).sum(dim=0)                               # (d,) efficient diag
    logits = Conv1d(F_all)                               # learnable aggregation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CovarianceSimilarity(nn.Module):
    """Covariance-based similarity with learnable Conv1d aggregation.
    
    Pipeline:
        1. Compute regularized covariance Σ from support
        2. For each query: F = (Q * (Σ @ Q)).sum(dim=0)  # efficient diag
        3. Stack F for all classes: (Way * d,)
        4. Conv1d aggregation to get logits: (Way,)
    
    Args:
        d: Spatial dimension H*W (default: 1024 for 32x32 features)
        reg_lambda: Regularization for covariance (default: 0.1)
        eps: Small constant for numerical stability (default: 1e-8)
    """
    
    def __init__(self, d: int = 1024, reg_lambda: float = 0.1, eps: float = 1e-8):
        super().__init__()
        self.d = d
        self.reg_lambda = reg_lambda
        self.eps = eps
        
        # Learnable aggregation: Conv1d(stride=d) to get per-class logits
        self.agg = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=d,
            stride=d,
            bias=True
        )
    
    def compute_covariance(self, support_features: torch.Tensor) -> torch.Tensor:
        """Compute REGULARIZED covariance matrix from support features.
        
        Args:
            support_features: (Shot, C, H, W) support feature maps
            
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
        
        # REGULARIZE: Σ + λI (CRITICAL for stability!)
        cov = cov + self.reg_lambda * torch.eye(C, device=cov.device)
        
        return cov
    
    def compute_F(self, query_sam: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        """Compute F = diag(Q^T Σ Q) efficiently.
        
        Args:
            query_sam: (C, d) query features (NO normalize!)
            cov: (C, C) covariance matrix
            
        Returns:
            F: (d,) diagonal elements
        """
        # Efficient formula (equivalent to diag(Q^T @ Σ @ Q))
        # Y = Σ @ Q: (C, C) @ (C, d) -> (C, d)
        Y = cov @ query_sam  # (C, d)
        
        # F = (Q * Y).sum(dim=0): element-wise multiply then sum over channels
        F = (query_sam * Y).sum(dim=0)  # (d,)
        
        return F
    
    def forward(
        self,
        query_features: torch.Tensor,
        support_features: torch.Tensor,
        way_num: int = None
    ) -> torch.Tensor:
        """Compute F for a SINGLE class (called multiple times, once per class).
        
        This method is called per-class by USCMambaNet.forward().
        
        Args:
            query_features: (NQ, C, H, W) query feature maps
            support_features: (Shot, C, H, W) support for ONE class
            
        Returns:
            F_all: (NQ, d) F values for each query (NOT logits yet!)
        """
        NQ, C, H, W = query_features.shape
        d = H * W
        
        # Update d if different from init (lazy initialization)
        if d != self.d:
            self.d = d
            self.agg = nn.Conv1d(1, 1, kernel_size=d, stride=d, bias=True).to(query_features.device)
        
        # Step 1: Compute REGULARIZED covariance Σ from support
        cov = self.compute_covariance(support_features)  # (C, C)
        
        # Step 2: For each query, compute F (NO L2-normalize!)
        F_list = []
        for i in range(NQ):
            query_sam = query_features[i].view(C, -1)  # (C, d) - NO normalize!
            F_i = self.compute_F(query_sam, cov)  # (d,)
            F_list.append(F_i)
        
        F_all = torch.stack(F_list, dim=0)  # (NQ, d)
        
        return F_all
    
    def aggregate_to_logits(self, F_per_class: list) -> torch.Tensor:
        """Aggregate F values from all classes to logits using Conv1d.
        
        Args:
            F_per_class: List of (NQ, d) tensors, one per class
            
        Returns:
            logits: (NQ, Way) classification logits
        """
        # Stack: list of (NQ, d) -> (NQ, Way, d)
        F_stacked = torch.stack(F_per_class, dim=1)  # (NQ, Way, d)
        NQ, Way, d = F_stacked.shape
        
        # Reshape for Conv1d: (NQ, 1, Way*d)
        F_flat = F_stacked.view(NQ, 1, -1)  # (NQ, 1, Way*d)
        
        # Apply learnable aggregation
        logits = self.agg(F_flat)  # (NQ, 1, Way)
        logits = logits.squeeze(1)  # (NQ, Way)
        
        return logits
