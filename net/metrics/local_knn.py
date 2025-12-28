"""Local k-NN similarity metric for few-shot learning.

Extracted from: DN4 (Li et al., CVPR 2019)
"Revisiting Local Descriptor based Image-to-Class Measure for Few-shot Learning"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def local_knn_score(query_local: torch.Tensor, support_local: torch.Tensor, 
                    k: int = 3) -> torch.Tensor:
    """Compute local k-NN similarity score.
    
    For each query local descriptor, find k-nearest neighbors in support
    descriptors and sum the similarities.
    
    Args:
        query_local: (N_q, D) normalized query local descriptors
        support_local: (N_s, D) normalized support local descriptors
        k: Number of nearest neighbors
        
    Returns:
        score: Scalar similarity score
    """
    # Cosine similarity: (N_q, N_s)
    sim = torch.mm(query_local, support_local.t())
    
    # Top-k per query descriptor
    k_actual = min(k, sim.size(1))
    topk_sim, _ = sim.topk(k_actual, dim=1)  # (N_q, k)
    
    # Sum all top-k matches
    return topk_sim.sum()


class LocalKNN(nn.Module):
    """Local k-NN module for few-shot classification.
    
    Uses local descriptors (spatial feature map positions) and k-NN matching.
    Well-suited for texture/pattern recognition.
    """
    
    def __init__(self, k_neighbors: int = 3):
        """Initialize LocalKNN.
        
        Args:
            k_neighbors: Number of nearest neighbors for voting
        """
        super(LocalKNN, self).__init__()
        self.k = k_neighbors
    
    def forward(self, query_features: torch.Tensor, 
                support_features: torch.Tensor) -> torch.Tensor:
        """Compute local k-NN similarity scores.
        
        Args:
            query_features: (B, D, h, w) query feature maps
            support_features: (B, Way, D, h*w) or (B, Way, Shot, D, h*w) support features
                             (should be flattened spatial dims)
            
        Returns:
            scores: (B, Way) similarity scores per class
        """
        B, D, h, w = query_features.size()
        
        # Flatten query spatial dims and normalize
        q_local = query_features.view(B, D, -1).permute(0, 2, 1)  # (B, h*w, D)
        q_local = F.normalize(q_local, p=2, dim=-1)
        
        # Handle different support shapes
        if support_features.dim() == 4:
            # (B, Way, D, N_s)
            Way = support_features.size(1)
            s_local = support_features.permute(0, 1, 3, 2)  # (B, Way, N_s, D)
            s_local = F.normalize(s_local, p=2, dim=-1)
        else:
            # (B, Way, Shot, D, N_s) -> average over shots
            Way = support_features.size(1)
            s_local = support_features.mean(dim=2).permute(0, 1, 3, 2)  # (B, Way, N_s, D)
            s_local = F.normalize(s_local, p=2, dim=-1)
        
        # Compute scores
        scores = []
        for b in range(B):
            class_scores = []
            for c in range(Way):
                score = local_knn_score(q_local[b], s_local[b, c], self.k)
                class_scores.append(score)
            scores.append(torch.stack(class_scores))
        
        return torch.stack(scores)  # (B, Way)
