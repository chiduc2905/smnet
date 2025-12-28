"""Earth Mover's Distance (EMD) / Sinkhorn distance for few-shot learning.

Extracted from: DeepEMD (Zhang et al., CVPR 2020)
"DeepEMD: Few-Shot Image Classification with Differentiable Earth Mover's Distance"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinkhorn_distance(cost: torch.Tensor, n_iters: int = 10, 
                      reg: float = 0.1) -> torch.Tensor:
    """Approximate EMD using Sinkhorn algorithm.
    
    The Sinkhorn algorithm provides a differentiable approximation to 
    optimal transport (EMD) by adding entropy regularization.
    
    Args:
        cost: (B, N, M) cost matrix
        n_iters: Number of Sinkhorn iterations
        reg: Entropy regularization (smaller = closer to true EMD)
        
    Returns:
        distance: (B,) approximate EMD values
    """
    B, N, M = cost.size()
    
    # Uniform marginals
    mu = torch.ones(B, N, device=cost.device) / N
    nu = torch.ones(B, M, device=cost.device) / M
    
    # Gibbs kernel
    K = torch.exp(-cost / reg)
    
    # Sinkhorn iterations
    u = torch.ones_like(mu)
    for _ in range(n_iters):
        v = nu / (torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + 1e-8)
        u = mu / (torch.bmm(K, v.unsqueeze(-1)).squeeze(-1) + 1e-8)
    
    # Compute transport plan
    P = u.unsqueeze(-1) * K * v.unsqueeze(1)
    
    # Compute distance
    distance = (P * cost).sum(dim=(1, 2))
    
    return distance


class SinkhornDistance(nn.Module):
    """Sinkhorn distance module for optimal transport matching.
    
    Computes differentiable approximation to Earth Mover's Distance.
    """
    
    def __init__(self, n_iters: int = 10, reg: float = 0.1):
        """Initialize SinkhornDistance.
        
        Args:
            n_iters: Number of Sinkhorn iterations
            reg: Entropy regularization
        """
        super(SinkhornDistance, self).__init__()
        self.n_iters = n_iters
        self.reg = reg
    
    def forward(self, query_local: torch.Tensor, 
                support_local: torch.Tensor) -> torch.Tensor:
        """Compute Sinkhorn distance between local descriptors.
        
        Args:
            query_local: (B, D, N_q) normalized query local descriptors
            support_local: (B, D, N_s) normalized support local descriptors
            
        Returns:
            distance: (B,) Sinkhorn distances
        """
        # Compute cost matrix (cosine distance)
        # query: (B, D, N_q), support: (B, D, N_s)
        # sim: (B, N_q, N_s)
        sim = torch.bmm(query_local.transpose(1, 2), support_local)
        cost = 1 - sim  # Cosine distance
        
        return sinkhorn_distance(cost, self.n_iters, self.reg)


class BestMatchDistance(nn.Module):
    """Simplified EMD using best-match aggregation.
    
    For each query descriptor, find best matching support descriptor
    and average the distances. Faster but less accurate than Sinkhorn.
    """
    
    def __init__(self):
        super(BestMatchDistance, self).__init__()
    
    def forward(self, query_local: torch.Tensor, 
                support_local: torch.Tensor) -> torch.Tensor:
        """Compute best-match distance.
        
        Args:
            query_local: (B, D, N_q) normalized query local descriptors
            support_local: (B, D, N_s) normalized support local descriptors
            
        Returns:
            similarity: (B,) average best-match similarities (negative = distance)
        """
        # Cosine similarity: (B, N_q, N_s)
        sim = torch.bmm(query_local.transpose(1, 2), support_local)
        
        # Best match for each query descriptor
        best_sim, _ = sim.max(dim=2)  # (B, N_q)
        
        # Average best matches
        return best_sim.mean(dim=1)
