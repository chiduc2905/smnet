"""SMNet (Slot Mamba Network) for few-shot learning.

Architecture v3:
- PatchEmbed2D: Overlapping patch embedding (stride=1)
- SpatialDownsample: Reduce spatial size for efficiency
- DualBranchFusion: Parallel local-global with anchor-guided fusion
  - Local Branch: ConvMixer++ (DW, Dilated DW, SE-Lite)
  - Global Branch: SS2D (4-way Mamba)
- Slot Attention: Semantic grouping into K slots
- Slot Mamba: Inter-slot reasoning
- Slot Covariance: Similarity computation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from net.backbone import SlotFeatureExtractor
from net.metrics import SlotCovarianceBlock


class SMNet(nn.Module):
    """SMNet: Slot Mamba Network for few-shot classification.
    
    Architecture:
        1. Shared SlotFeatureExtractor for support and query
        2. SlotCovarianceBlock for metric-based classification
    
    Args:
        in_channels: Input image channels (default: 1)
        hidden_dim: Hidden dimension (default: 64)
        num_slots: Number of semantic slots K (default: 4)
        learnable_slots: Whether slot count is learnable (default: True)
        regularization: Covariance regularization (default: 1e-3)
        device: Device to use
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 64,
        num_slots: int = 4,
        learnable_slots: bool = True,
        regularization: float = 1e-3,
        device: str = 'cuda'
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_slots = num_slots
        self.device = device
        
        # Shared Feature Extractor (v3 pipeline)
        self.encoder = SlotFeatureExtractor(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_slots=num_slots,
            learnable_slots=learnable_slots,
            # New v3 parameters
            patch_kernel=3,           # PatchEmbed2D kernel size
            downsample_factor=4,      # 64×64 → 16×16
            dual_branch_dilation=2,   # Local branch dilation
            d_state=16,               # Mamba state dimension
            slot_iters=3,
            slot_mamba_layers=1
        )
        
        # Slot Covariance Similarity
        self.similarity = SlotCovarianceBlock(
            regularization=regularization,
            use_mahalanobis=True,
            normalize_slots=True
        )
        
        self.to(device)
    
    def forward(self, query: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        """Compute few-shot classification scores.
        
        Args:
            query: (B, NQ, C, H, W) query images
            support: (B, Way, Shot, C, H, W) support images
            
        Returns:
            scores: (B*NQ, Way) similarity scores
        """
        B, NQ, C, H, W = query.shape
        B_s, Way, Shot, C_s, H_s, W_s = support.shape
        
        scores_list = []
        
        for b in range(B):
            # Extract query slots
            q_b = query[b]  # (NQ, C, H, W)
            q_slots, q_weights = self.encoder(q_b)  # (NQ, K, hidden_dim), (NQ, K)
            
            # Extract support slots per class
            support_slots = []
            support_weights = []
            
            for w in range(Way):
                s_class = support[b, w]  # (Shot, C, H, W)
                s_slots, s_weights = self.encoder(s_class)  # (Shot, K, hidden_dim)
                support_slots.append(s_slots)
                support_weights.append(s_weights)
            
            # Compute covariance-based similarity
            # q_slots: (NQ, K, hidden_dim)
            # support_slots: List of (Shot, K, hidden_dim) per class
            scores = self.similarity(
                q_slots, 
                support_slots,
                q_weights,
                support_weights
            )  # (NQ, Way)
            
            scores_list.append(scores)
        
        # Concatenate all batches
        all_scores = torch.cat(scores_list, dim=0)  # (B*NQ, Way)
        
        return all_scores
    
    def get_slot_features(self, images: torch.Tensor) -> tuple:
        """Extract slot features for visualization.
        
        Args:
            images: (B, C, H, W) input images
            
        Returns:
            slots: (B, K, hidden_dim) slot descriptors
            slot_weights: (B, K) slot existence weights
        """
        return self.encoder(images)

