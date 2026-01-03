"""SMNet (Slot Mamba Network) for few-shot learning.

Architecture (with Class-Aware Inference):
- PatchEmbed2D: Overlapping patch embedding (stride=1)
- PatchMerging2D: Swin-style hierarchical patch merging
- DualBranchFusion: Parallel local-global with anchor-guided fusion
  - Local Branch: ConvMixer++
  - Global Branch: SS2D (4-way Mamba)
- Slot Attention (Mamba): Semantic grouping into K slots
- Slot Mamba: Inter-slot reasoning
- Class-Aware Inference: Slot-based class-conditioned patch refinement

Pipeline:
    Input:       (B, 1, H, W)
    PatchEmbed:  (B, 32, H, W)
    Merge1-3:    (B, 256, H', W')
    Proj:        (B, 64, H', W')
    DualBranch:  (B, 64, H', W')
    Slots:       (B, K, 64)
    ClassAware:  (NQ, Way) similarity scores
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from einops import rearrange

from net.backbone import SlotFeatureExtractor
from net.backbone.class_aware_inference import ClassAwareInferenceHead


class SMNet(nn.Module):
    """SMNet: Slot Mamba Network with Class-Aware Inference for few-shot classification.
    
    Architecture:
        1. Shared SlotFeatureExtractor for support and query
        2. ClassAwareInferenceHead for slot-based classification
    
    Args:
        in_channels: Input image channels (default: 1 for grayscale)
        base_dim: Base embedding dimension (default: 32, doubles each merge stage)
        hidden_dim: Final hidden dimension for slots (default: 64)
        num_slots: Number of semantic slots K (default: 4)
        num_merging_stages: Number of PatchMerging2D stages (default: 3)
        learnable_slots: Whether slot count is learnable (default: True)
        temperature: Temperature for similarity scaling (default: 0.1)
        lambda_init: Initial value for residual scaling λ (default: 0.5)
        device: Device to use
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        base_dim: int = 32,
        hidden_dim: int = 64,
        num_slots: int = 4,
        num_merging_stages: int = 3,
        learnable_slots: bool = True,
        regularization: float = 1e-3,  # kept for backward compatibility
        temperature: float = 0.1,
        lambda_init: float = 0.5,
        device: str = 'cuda'
    ):
        super().__init__()
        
        self.base_dim = base_dim
        self.hidden_dim = hidden_dim
        self.num_slots = num_slots
        self.device = device
        
        # Shared Feature Extractor (Mamba-based Slot Attention)
        self.encoder = SlotFeatureExtractor(
            in_channels=in_channels,
            base_dim=base_dim,
            hidden_dim=hidden_dim,
            num_slots=num_slots,
            learnable_slots=learnable_slots,
            patch_kernel=3,
            num_merging_stages=num_merging_stages,
            dual_branch_dilation=2,
            d_state=16,
            slot_iters=3,
            slot_mamba_layers=1
        )
        
        # Class-Aware Inference Head
        self.inference_head = ClassAwareInferenceHead(
            dim=hidden_dim,
            temperature=temperature,
            lambda_init=lambda_init,
            learnable_lambda=True,
            use_mlp_similarity=False
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
            # === Extract Query Features ===
            q_b = query[b]  # (NQ, C, H, W)
            q_slots, q_weights, q_features, q_attn = self.encoder(q_b)
            # q_slots: (NQ, K, hidden_dim)
            # q_weights: (NQ, K)
            # q_features: (NQ, hidden_dim, H', W')
            # q_attn: (NQ, K, H'*W')
            
            # Convert features to patch format for ClassAwareInferenceHead
            NQ_q, C_f, H_f, W_f = q_features.shape
            q_patches = rearrange(q_features, 'b c h w -> b (h w) c')  # (NQ, N, C)
            
            # === Extract Support Features per Class ===
            support_patches_list = []
            support_slots_list = []
            support_attn_list = []
            support_weights_list = []
            
            for w in range(Way):
                s_class = support[b, w]  # (Shot, C, H, W)
                s_slots, s_weights, s_features, s_attn = self.encoder(s_class)
                # s_slots: (Shot, K, hidden_dim)
                # s_features: (Shot, hidden_dim, H', W')
                # s_attn: (Shot, K, H'*W')
                
                # Convert to patch format
                s_patches = rearrange(s_features, 'b c h w -> b (h w) c')  # (Shot, N, C)
                
                support_patches_list.append(s_patches)
                support_slots_list.append(s_slots)
                support_attn_list.append(s_attn)
                support_weights_list.append(s_weights)
            
            # === Class-Aware Inference ===
            scores = self.inference_head(
                query_patches=q_patches,
                query_slots=q_slots,
                query_attn=q_attn,
                support_patches_list=support_patches_list,
                support_slots_list=support_slots_list,
                support_attn_list=support_attn_list,
                support_weights_list=support_weights_list
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
        slots, slot_weights, _, _ = self.encoder(images)
        return slots, slot_weights
