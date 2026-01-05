"""M5: Slot Covariance Attention (SCA).

Purpose: Weight slots by how well their patch covariance matches class covariance.
Slots with similar second-order statistics to the class are weighted higher.

Input:
    - slots: (B, K, C) slot descriptors
    - attn: (B, K, N) attention matrix (slot → patch soft assignment)
    - patches: (B, N, C) patch tokens
    - class_cov: (C, C) class covariance matrix (from support)

Output:
    - alpha: (B, K) slot importance weights (sum to 1)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SlotCovarianceAttention(nn.Module):
    """Slot Covariance Attention for slot importance weighting.
    
    Computes importance of each slot by comparing the covariance structure
    of its assigned patches with the target class covariance.
    
    Args:
        dim: Feature dimension C
        temperature: Temperature for softmax (default: 1.0)
        eps: Epsilon for numerical stability
    """
    
    def __init__(
        self,
        dim: int,
        temperature: float = 1.0,
        eps: float = 1e-8
    ):
        super().__init__()
        self.dim = dim
        self.temperature = temperature
        self.eps = eps
        
        # Optional learnable temperature
        self.temp_scale = nn.Parameter(torch.tensor(temperature))
    
    def compute_weighted_covariance(
        self,
        patches: torch.Tensor,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """Compute weighted covariance matrix from patches.
        
        Args:
            patches: (B, N, C) patch tokens
            weights: (B, N) attention weights for this slot
            
        Returns:
            cov: (B, C, C) covariance matrix
        """
        B, N, C = patches.shape
        
        # Normalize weights to sum to 1
        weights = weights / (weights.sum(dim=-1, keepdim=True) + self.eps)  # (B, N)
        
        # Weighted mean: (B, C)
        mean = torch.einsum('bn,bnc->bc', weights, patches)
        
        # Center patches
        centered = patches - mean.unsqueeze(1)  # (B, N, C)
        
        # Weighted covariance: (B, C, C)
        # cov = sum_n w_n * (x_n - mean) @ (x_n - mean)^T
        weighted_centered = centered * weights.unsqueeze(-1)  # (B, N, C)
        cov = torch.einsum('bnc,bnd->bcd', weighted_centered, centered)  # (B, C, C)
        
        return cov
    
    def frobenius_similarity(
        self,
        cov1: torch.Tensor,
        cov2: torch.Tensor
    ) -> torch.Tensor:
        """Compute Frobenius inner product similarity between covariance matrices.
        
        Args:
            cov1: (B, C, C) or (C, C) first covariance
            cov2: (C, C) second covariance (class covariance)
            
        Returns:
            similarity: (B,) similarity scores
        """
        # Normalize covariances by Frobenius norm
        cov1_norm = cov1 / (torch.norm(cov1, p='fro', dim=(-2, -1), keepdim=True) + self.eps)
        cov2_norm = cov2 / (torch.norm(cov2, p='fro') + self.eps)
        
        # Frobenius inner product: trace(A^T @ B)
        if cov1.dim() == 3:
            # cov1: (B, C, C), cov2: (C, C)
            sim = torch.einsum('bcd,cd->b', cov1_norm, cov2_norm)
        else:
            sim = torch.einsum('cd,cd->', cov1_norm, cov2_norm)
        
        return sim
    
    def forward(
        self,
        slots: torch.Tensor,
        attn: torch.Tensor,
        patches: torch.Tensor,
        class_cov: torch.Tensor
    ) -> torch.Tensor:
        """Compute slot importance weights via covariance matching.
        
        Args:
            slots: (B, K, C) slot descriptors (not used directly, kept for interface)
            attn: (B, K, N) attention matrix
            patches: (B, N, C) patch tokens
            class_cov: (C, C) class covariance matrix
            
        Returns:
            alpha: (B, K) slot importance weights
        """
        B, K, N = attn.shape
        C = patches.shape[-1]
        
        slot_scores = []
        
        for k in range(K):
            # Soft assignment weights for slot k
            weights_k = attn[:, k, :]  # (B, N)
            
            # Compute covariance of patches assigned to slot k
            cov_k = self.compute_weighted_covariance(patches, weights_k)  # (B, C, C)
            
            # Compare with class covariance
            score_k = self.frobenius_similarity(cov_k, class_cov)  # (B,)
            slot_scores.append(score_k)
        
        # Stack and apply softmax for competition
        scores = torch.stack(slot_scores, dim=-1)  # (B, K)
        alpha = F.softmax(scores / self.temp_scale, dim=-1)  # (B, K)
        
        return alpha


def compute_class_covariance(
    patches: torch.Tensor,
    attn: Optional[torch.Tensor] = None,
    eps: float = 1e-8
) -> torch.Tensor:
    """Compute covariance matrix from patches (utility function).
    
    Args:
        patches: (Shot, N, C) or (B, N, C) patches from support set
        attn: Optional (Shot, K, N) attention for weighting
        eps: Numerical stability
        
    Returns:
        cov: (C, C) class covariance matrix
    """
    if patches.dim() == 3:
        # Flatten shot and spatial: (Shot, N, C) -> (Shot*N, C)
        all_patches = patches.reshape(-1, patches.shape[-1])
    else:
        all_patches = patches
    
    # Compute mean
    mean = all_patches.mean(dim=0)  # (C,)
    
    # Center
    centered = all_patches - mean.unsqueeze(0)  # (M, C)
    
    # Covariance
    M = centered.shape[0]
    cov = (centered.T @ centered) / (M - 1 + eps)  # (C, C)
    
    return cov


def compute_class_prototype(
    patches: torch.Tensor
) -> torch.Tensor:
    """Compute class prototype (mean vector) from patches.
    
    Args:
        patches: (Shot, N, C) patches from support set
        
    Returns:
        proto: (C,) class prototype vector
    """
    if patches.dim() == 3:
        # Flatten shot and spatial: (Shot, N, C) -> (Shot*N, C)
        all_patches = patches.reshape(-1, patches.shape[-1])
    else:
        all_patches = patches
    
    # Mean over all patches
    proto = all_patches.mean(dim=0)  # (C,)
    
    return proto
