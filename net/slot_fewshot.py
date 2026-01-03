"""SMNet (Slot Mamba Network) for few-shot learning.

Integrates:
- ConvMixer for local spatial extraction
- SS2D (4-way Mamba) for global spatial context
- Channel Attention for local channel interaction
- Slot Attention for semantic grouping (learnable K slots)
- Slot Mamba for inter-slot reasoning
- Slot Covariance for similarity computation
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
        in_channels: Input image channels (default: 3)
        hidden_dim: Hidden dimension (default: 256)
        num_slots: Number of semantic slots K (default: 4)
        learnable_slots: Whether slot count is learnable (default: True)
        regularization: Covariance regularization (default: 1e-3)
        device: Device to use
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 256,
        num_slots: int = 4,
        learnable_slots: bool = True,
        regularization: float = 1e-3,
        device: str = 'cuda'
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_slots = num_slots
        self.device = device
        
        # Shared Feature Extractor
        self.encoder = SlotFeatureExtractor(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_slots=num_slots,
            learnable_slots=learnable_slots,
            convmixer_depth=4,
            convmixer_kernel=9,
            patch_size=4,
            slot_iters=3,
            slot_mamba_layers=2
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


class SMNetLight(nn.Module):
    """SMNet Lightweight version with fewer parameters.
    
    Reduces:
    - ConvMixer depth: 4 -> 2
    - Slot Mamba layers: 2 -> 1
    - Hidden dim: 256 -> 128
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 128,
        num_slots: int = 4,
        device: str = 'cuda'
    ):
        super().__init__()
        
        self.encoder = SlotFeatureExtractor(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_slots=num_slots,
            learnable_slots=True,
            convmixer_depth=2,
            convmixer_kernel=7,
            patch_size=4,
            slot_iters=2,
            slot_mamba_layers=1
        )
        
        self.similarity = SlotCovarianceBlock(
            regularization=1e-3,
            use_mahalanobis=True,
            normalize_slots=True
        )
        
        self.to(device)
    
    def forward(self, query: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        """Same as SlotCovarianceNet.forward()"""
        B, NQ, C, H, W = query.shape
        B_s, Way, Shot, C_s, H_s, W_s = support.shape
        
        scores_list = []
        
        for b in range(B):
            q_b = query[b]
            q_slots, q_weights = self.encoder(q_b)
            
            support_slots = []
            support_weights = []
            
            for w in range(Way):
                s_class = support[b, w]
                s_slots, s_weights = self.encoder(s_class)
                support_slots.append(s_slots)
                support_weights.append(s_weights)
            
            scores = self.similarity(q_slots, support_slots, q_weights, support_weights)
            scores_list.append(scores)
        
        return torch.cat(scores_list, dim=0)
