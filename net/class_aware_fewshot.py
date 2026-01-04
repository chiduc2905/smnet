"""Class-Aware SMNet for Few-Shot Learning.

Implements a slot-based, class-aware few-shot network that combines:
- Shared encoder (PatchEmbed2D → PatchMerging2D → DualBranchFusion)
- Slot Attention with Mamba (replaces GRU with Mamba for slot updates)
- Class-Aware Inference Head with patch-to-patch similarity

Architecture v6 (RGB 64×64 Class-Aware with Mamba):
    Input:       (B, 3, 64, 64)    ← RGB
    PatchEmbed:  (B, 32, 64, 64)
    Merge1:      (B, 64, 32, 32)
    Merge2:      (B, 128, 16, 16)  ← N = 256 patches
    Proj:        (B, 64, 16, 16)
    DualBranch:  (B, 64, 16, 16)
    
    SlotAttentionMamba (Mamba replaces GRU):
        - Slots: (B, K, 64)
        - Attn:  (B, K, 256)
        - Patches: (B, 256, 64)
    
    Class-Aware Inference:
        - Class Embeddings from support slots
        - Slot filtering by class similarity
        - Patch refinement with class-conditioned mask
        - Patch-to-patch similarity matrix for classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
from einops import rearrange

from net.backbone.feature_extractor import (
    PatchEmbed2D,
    PatchMerging2D
)
from net.backbone.dual_branch_fusion import DualBranchFusion
from net.backbone.slot_attention import SlotAttention
from net.backbone.class_aware_inference import (
    ClassAwareInferenceHead,
    SlotAttentionWithPatches
)


class ClassAwareSMNet(nn.Module):
    """Class-Aware SMNet for few-shot classification with Mamba-based Slot Attention.
    
    This network uses:
        - SlotAttentionMamba: Mamba SSM replaces GRU for slot updates
        - ClassAwareInferenceHead: Patch-to-patch similarity matrix
        - No separate SlotMamba module (Mamba is integrated into SlotAttention)
    
    Key features:
        1. Mamba-based slot updates for better temporal modeling
        2. Patch-to-patch similarity for fine-grained matching
        3. Class-conditioned patch refinement
    
    Args:
        in_channels: Input image channels (default: 3 for RGB)
        base_dim: Base embedding dimension (default: 32)
        hidden_dim: Final hidden dimension (default: 64)
        num_slots: Number of semantic slots K (default: 4)
        num_merging_stages: Number of PatchMerging2D stages (default: 3)
        slot_iters: Slot attention iterations (default: 3)
        d_state: Mamba state dimension (default: 16)
        temperature: Temperature for similarity (default: 0.1)
        lambda_init: Initial residual scaling (default: 0.5)
        device: Device to use
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        base_dim: int = 32,
        hidden_dim: int = 64,
        num_slots: int = 4,
        num_merging_stages: int = 2,
        slot_iters: int = 3,
        d_state: int = 16,
        temperature: float = 0.1,
        lambda_init: float = 0.5,
        device: str = 'cuda'
    ):
        super().__init__()
        
        self.base_dim = base_dim
        self.hidden_dim = hidden_dim
        self.num_slots = num_slots
        self.device = device
        
        # ============================================================
        # STAGE 1: Patch Embedding
        # ============================================================
        self.patch_embed = PatchEmbed2D(
            in_channels=in_channels,
            embed_dim=base_dim,
            kernel_size=3,
            norm_layer=nn.LayerNorm
        )
        
        # ============================================================
        # STAGE 2: Hierarchical Patch Merging
        # ============================================================
        self.merging_stages = nn.ModuleList()
        current_dim = base_dim
        
        for i in range(num_merging_stages):
            self.merging_stages.append(
                PatchMerging2D(dim=current_dim, norm_layer=nn.LayerNorm)
            )
            current_dim = current_dim * 2
        
        self.final_merge_dim = current_dim
        
        # ============================================================
        # STAGE 3: Channel Projection
        # ============================================================
        self.channel_proj = nn.Sequential(
            nn.Conv2d(self.final_merge_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )
        
        # ============================================================
        # STAGE 4: Dual Branch Fusion
        # ============================================================
        self.dual_branch = DualBranchFusion(
            channels=hidden_dim,
            d_state=d_state,
            dilation=2
        )
        
        # ============================================================
        # STAGE 5: Slot Attention with Mamba (replaces GRU + SlotMamba)
        # ============================================================
        slot_attention = SlotAttention(
            num_slots=num_slots,
            dim=hidden_dim,
            iters=slot_iters,
            d_state=d_state,
            d_conv=4,
            learnable_slots=True
        )
        self.slot_module = SlotAttentionWithPatches(slot_attention, hidden_dim)
        
        # NOTE: SlotMamba is removed - Mamba is now integrated into SlotAttention
        
        # ============================================================
        # STAGE 6: Class-Aware Inference Head
        # ============================================================
        self.inference_head = ClassAwareInferenceHead(
            dim=hidden_dim,
            temperature=temperature,
            lambda_init=lambda_init,
            learnable_lambda=True,
            use_mlp_similarity=False
        )
        
        self.to(device)
    
    def encode(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode images to slots, attention, and patches.
        
        Args:
            x: (B, C, H, W) input images
            
        Returns:
            slots: (B, K, hidden_dim) slot descriptors
            slot_weights: (B, K) slot existence weights
            attn: (B, K, N) slot attention matrix
            patches: (B, N, hidden_dim) patch tokens
        """
        # Stage 1: Patch embedding
        features = self.patch_embed(x)
        
        # Stage 2: Hierarchical merging
        for merge_layer in self.merging_stages:
            features = merge_layer(features)
        
        # Stage 3: Channel projection
        features = self.channel_proj(features)
        
        # Stage 4: Dual branch fusion
        features = self.dual_branch(features)
        
        # Stage 5: Slot attention with Mamba (Mamba replaces GRU for slot updates)
        slots, slot_weights, attn, patches = self.slot_module(features)
        
        # No separate SlotMamba - Mamba is integrated into SlotAttentionMamba
        
        return slots, slot_weights, attn, patches
    
    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor
    ) -> torch.Tensor:
        """Compute few-shot classification scores using class-aware inference.
        
        Args:
            query: (B, NQ, C, H, W) query images
            support: (B, Way, Shot, C, H, W) support images
            
        Returns:
            scores: (B*NQ, Way) similarity scores
        """
        B, NQ, C, H, W = query.shape
        B_s, Way, Shot, C_s, H_s, W_s = support.shape
        
        all_scores = []
        
        for b in range(B):
            # === Encode Query ===
            q_b = query[b]  # (NQ, C, H, W)
            q_slots, q_weights, q_attn, q_patches = self.encode(q_b)
            # q_slots: (NQ, K, hidden_dim)
            # q_attn: (NQ, K, N)
            # q_patches: (NQ, N, hidden_dim)
            
            # === Encode Support per class ===
            support_slots_list = []
            support_attn_list = []
            support_patches_list = []
            support_weights_list = []
            
            for w in range(Way):
                s_class = support[b, w]  # (Shot, C, H, W)
                s_slots, s_weights, s_attn, s_patches = self.encode(s_class)
                # s_slots: (Shot, K, hidden_dim)
                # s_attn: (Shot, K, N)
                # s_patches: (Shot, N, hidden_dim)
                
                support_slots_list.append(s_slots)
                support_attn_list.append(s_attn)
                support_patches_list.append(s_patches)
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
            
            all_scores.append(scores)
        
        # Concatenate all batches
        all_scores = torch.cat(all_scores, dim=0)  # (B*NQ, Way)
        
        return all_scores
    
    def get_slot_features(
        self,
        images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract features for visualization.
        
        Args:
            images: (B, C, H, W) input images
            
        Returns:
            slots: (B, K, hidden_dim)
            slot_weights: (B, K)
            attn: (B, K, N)
            patches: (B, N, hidden_dim)
        """
        return self.encode(images)
    
    def get_attention_maps(
        self,
        images: torch.Tensor,
        spatial_size: Tuple[int, int] = (16, 16)
    ) -> torch.Tensor:
        """Get slot attention maps reshaped to spatial dimensions.
        
        Args:
            images: (B, C, H, W) input images
            spatial_size: (H, W) spatial dimensions after merging
            
        Returns:
            attn_maps: (B, K, H, W) spatial attention maps per slot
        """
        slots, slot_weights, attn, patches = self.encode(images)
        H, W = spatial_size
        
        # Reshape attention: (B, K, N) -> (B, K, H, W)
        attn_maps = rearrange(attn, 'b k (h w) -> b k h w', h=H, w=W)
        
        return attn_maps


def build_class_aware_smnet(
    num_classes: int = 4,
    num_slots: int = None,
    **kwargs
) -> ClassAwareSMNet:
    """Factory function for ClassAwareSMNet.
    
    Args:
        num_classes: Number of classes (Way)
        num_slots: Number of slots (default: num_classes)
        **kwargs: Additional arguments
        
    Returns:
        Configured ClassAwareSMNet
    """
    if num_slots is None:
        num_slots = num_classes
    
    return ClassAwareSMNet(num_slots=num_slots, **kwargs)
