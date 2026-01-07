"""SimplePatchSimilarity: Non-learnable position-wise cosine similarity.

Computes similarity between query and support features using simple,
non-learnable operations: position-wise cosine similarity + mean/topk aggregation.

Reference:
    Support feature maps  Fs ∈ R[C×H×W]
    Query feature maps    Fq ∈ R[C×H×W]
    
    1. Flatten spatial → patches
       Fs → {s₁,…,s_N}, Fq → {q₁,…,q_N}
    
    2. Compute cosine similarity per patch (position-wise)
       sim_i = cosine(q_i, s_i)
    
    3. Aggregate similarities (NON-learnable)
       score = mean(sim_i)  or  top-k mean
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplePatchSimilarity(nn.Module):
    """Simple Position-wise Patch Similarity (NON-learnable).
    
    Pipeline:
        1. Flatten: (B, C, H, W) → (B, N, C)
        2. L2-normalize patches
        3. Position-wise cosine: sim_i = cos(q_i, s_i)
        4. Aggregate: mean or top-k mean (no learnable params)
    
    Args:
        aggregation: 'mean' or 'topk' (default: 'mean')
        topk_ratio: Ratio of top-k patches to use (default: 0.5)
        temperature: Temperature scaling for scores (default: 1.0)
    """
    
    def __init__(
        self,
        aggregation: str = 'mean',
        topk_ratio: float = 0.5,
        temperature: float = 1.0
    ):
        super().__init__()
        self.aggregation = aggregation
        self.topk_ratio = topk_ratio
        self.temperature = temperature
    
    def forward(
        self,
        query_features: torch.Tensor,
        support_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute similarity score between query and support.
        
        Args:
            query_features: (NQ, C, H, W) query feature maps
            support_features: (Shot, C, H, W) support feature maps for ONE class
            
        Returns:
            scores: (NQ,) similarity scores
        """
        NQ, C, H, W = query_features.shape
        Shot = support_features.shape[0]
        N = H * W
        
        # 1. Flatten spatial → patches: (B, C, H, W) → (B, N, C)
        q_patches = query_features.flatten(2).transpose(1, 2)  # (NQ, N, C)
        s_patches = support_features.flatten(2).transpose(1, 2)  # (Shot, N, C)
        
        # Average support across shots → prototype per patch position
        # CRITICAL: Detach support to prevent gradient flowing through support
        s_proto = s_patches.mean(dim=0).detach()  # (N, C) - detached!
        
        # 2. L2-normalize
        q_norm = F.normalize(q_patches, dim=-1)  # (NQ, N, C)
        s_norm = F.normalize(s_proto, dim=-1)    # (N, C)
        
        # 3. Position-wise cosine similarity: sim_i = cos(q_i, s_i)
        # For each query patch at position i, compare with support patch at same position
        sim = (q_norm * s_norm.unsqueeze(0)).sum(dim=-1)  # (NQ, N)
        
        # 4. Aggregate (NON-learnable)
        if self.aggregation == 'mean':
            scores = sim.mean(dim=-1)  # (NQ,)
        elif self.aggregation == 'topk':
            k = max(1, int(N * self.topk_ratio))
            topk_sim = sim.topk(k, dim=-1)[0]  # (NQ, k)
            scores = topk_sim.mean(dim=-1)     # (NQ,)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        # Temperature scaling
        scores = scores / self.temperature
        
        return scores


class AllPairsSimilarity(nn.Module):
    """All-pairs patch similarity with TOP-K aggregation.
    
    Computes NxN similarity matrix between all query and support patches,
    then aggregates using TOP-K (NOT max!) per query patch.
    
    Pipeline:
        1. Flatten: (B, C, H, W) → (B, N, C)
        2. L2-normalize
        3. All-pairs cosine: sim[i,j] = cos(q_i, s_j), shape (NQ, N, N)
        4. TOP-K per query patch: top_sim[i] = topk_j(sim[i,j]), shape (NQ, N, k)
        5. Mean over patches: score = mean(mean_k(top_sim)), shape (NQ,)
    
    Args:
        temperature: Temperature scaling for scores (default: 1.0)
        topk_ratio: Ratio of top-k per query patch (default: 0.2)
    """
    
    def __init__(self, temperature: float = 1.0, topk_ratio: float = 0.2):
        super().__init__()
        self.temperature = temperature
        self.topk_ratio = topk_ratio
        # FIX 2: Learnable logit scale for CE gradient boost
        self.logit_scale = nn.Parameter(torch.tensor(10.0))
    
    def forward(
        self,
        query_features: torch.Tensor,
        support_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute all-pairs similarity score with TOP-K.
        
        Args:
            query_features: (NQ, C, H, W) query feature maps
            support_features: (Shot, C, H, W) support feature maps for ONE class
            
        Returns:
            scores: (NQ,) similarity scores
        """
        NQ, C, H, W = query_features.shape
        N = H * W
        
        # Flatten
        q_patches = query_features.flatten(2).transpose(1, 2)  # (NQ, N, C)
        s_patches = support_features.flatten(2).transpose(1, 2)  # (Shot, N, C)
        
        # Average support → prototype (DETACHED!)
        s_proto = s_patches.mean(dim=0).detach()  # (N, C)
        
        # L2-normalize
        q_norm = F.normalize(q_patches, dim=-1)  # (NQ, N, C)
        s_norm = F.normalize(s_proto, dim=-1)    # (N, C)
        
        # All-pairs cosine similarity: (NQ, N, N)
        sim_matrix = torch.einsum('qnc,mc->qnm', q_norm, s_norm)
        
        # TOP-K per query patch (NOT max!)
        # For each of N query patches, get top-k similarities to support patches
        k = max(1, int(N * self.topk_ratio))
        topk_per_patch = sim_matrix.topk(k, dim=-1)[0]  # (NQ, N, k)
        
        # Mean over k, then mean over N patches
        mean_per_patch = topk_per_patch.mean(dim=-1)  # (NQ, N)
        scores = mean_per_patch.mean(dim=-1)  # (NQ,)
        
        # FIX 2: Apply learnable logit scale (target std ≈ 3-6)
        scores = scores * self.logit_scale / self.temperature
        
        return scores
