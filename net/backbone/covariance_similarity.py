"""Covariance-based Metric Similarity for Few-Shot Classification.

Matching reference implementation from few-shot-mamba:
    1. L2-normalize query per channel (CRITICAL!)
    2. Compute covariance Σ from centered support
    3. F = diag(Q_norm^T Σ Q_norm)
    4. Conv1d classifier to aggregate (stride = h*w)

This ensures bounded output and controlled loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CovarianceSimilarity(nn.Module):
    """Covariance-based similarity metric (matching reference).
    
    Pipeline:
        1. Compute covariance matrix Σ from support (centered)
        2. L2-normalize query per channel (like reference!)
        3. F = diag(Q_norm^T Σ Q_norm)
        4. Conv1d classifier to get per-class scores
    
    Args:
        h: Height of feature map (default: 32)
        w: Width of feature map (default: 32)
        way_num: Number of classes (default: 4)
        eps: Small constant for numerical stability (default: 1e-8)
    """
    
    def __init__(self, h: int = 32, w: int = 32, way_num: int = 4, eps: float = 1e-8):
        super().__init__()
        self.h = h
        self.w = w
        self.way_num = way_num
        self.eps = eps
        
        # Classifier like reference: Conv1d(stride=h*w) to aggregate per-class
        self.classifier = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(1, 1, kernel_size=h*w, stride=h*w, bias=True)
        )
    
    def compute_covariance(self, support_features: torch.Tensor) -> torch.Tensor:
        """Compute covariance matrix from support features.
        
        Matches reference exactly:
            support = support.permute(1,0,2,3).view(C, -1)
            support = support - mean
            cov = support @ support^T / (N-1)
        
        Args:
            support_features: (Shot, C, H, W) support feature maps
            
        Returns:
            cov: (C, C) covariance matrix
        """
        B, C, h, w = support_features.shape
        
        # Reshape like reference: (Shot, C, H, W) -> (C, Shot*H*W)
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
        support_list: list
    ) -> torch.Tensor:
        """Compute covariance-based similarity (matching reference).
        
        Args:
            query_features: (NQ, C, H, W) query feature maps
            support_list: List of (Shot, C, H, W) support features, one per class
            
        Returns:
            scores: (NQ, Way) similarity scores
        """
        NQ, C, H, W = query_features.shape
        d = H * W
        Way = len(support_list)
        
        # Compute covariance for each class
        cov_list = [self.compute_covariance(s) for s in support_list]
        
        # Process each query
        all_F = []
        
        for i in range(NQ):
            query_sam = query_features[i]  # (C, H, W)
            query_sam = query_sam.view(C, -1)  # (C, d)
            
            # L2-normalize per channel (CRITICAL - like reference!)
            # Reference: query_sam_norm = torch.norm(query_sam, 2, 1, True)
            query_norm = torch.norm(query_sam, p=2, dim=1, keepdim=True)  # (C, 1)
            query_sam = query_sam / (query_norm + self.eps)  # (C, d), each row has ||.|| = 1
            
            # For each class, compute F = diag(Q^T Σ Q)
            F_concat = []
            for j, cov in enumerate(cov_list):
                # Q^T @ Σ @ Q: (d, C) @ (C, C) @ (C, d) -> (d, d)
                temp = query_sam.t() @ cov @ query_sam  # (d, d)
                F_j = temp.diag()  # (d,)
                F_concat.append(F_j)
            
            # Concatenate all class F: (Way * d,)
            F_all = torch.cat(F_concat, dim=0)  # (Way * d,)
            all_F.append(F_all.unsqueeze(0))  # (1, Way * d)
        
        # Stack all queries: (NQ, Way * d)
        all_F = torch.cat(all_F, dim=0)  # (NQ, Way*d)
        
        # Apply classifier: Conv1d(stride=h*w) to get (NQ, Way)
        # Input: (NQ, 1, Way*d), Output: (NQ, 1, Way)
        scores = self.classifier(all_F.unsqueeze(1))  # (NQ, 1, Way)
        scores = scores.squeeze(1)  # (NQ, Way)
        
        return scores
