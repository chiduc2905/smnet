"""SMNet (Slot Mamba Network) for few-shot learning.

Architecture (RGB 64×64 with SAFF Module):
- PatchEmbed2D: Overlapping patch embedding (stride=1)
- PatchMerging2D: Swin-style hierarchical patch merging (64→32→16)
- DualBranchFusion: Parallel local-global with anchor-guided fusion
  - Local Branch: AG-LKA (Attention-Guided Large Kernel Attention)
  - Global Branch: VSS Block (4-way Mamba)
- SAFF Module (from paper arXiv:2508.09699):
  - Slot Attention: Semantic grouping into K slots
  - Class Token: Class-agnostic embedding
  - Slot-Class Filtering: Compare slots with class token
  - Attention Mask: Apply filtered attention to patches
  - Refined Patches: Weighted masking
  - Similarity Matrix: Cross support/query patch similarity
  - Conv1D + MLP → Final Score

Pipeline:
    Input:       (B, 3, 64, 64)    ← RGB
    PatchEmbed:  (B, 32, 64, 64)
    Merge1-2:    (B, 128, 16, 16)  ← N = 256 patches
    DualBranch:  (B, 128, 16, 16)
    SAFF:        (NQ, Way) similarity scores
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from einops import rearrange

from net.backbone.feature_extractor import PatchEmbed2D, PatchMerging2D
from net.backbone.dual_branch_fusion import DualBranchFusion
from net.backbone.saff import SAFFModule


class SMNet(nn.Module):
    """SMNet: Slot Mamba Network with SAFF for few-shot classification.
    
    Architecture:
        1. Backbone: PatchEmbed → PatchMerging → DualBranch
        2. SAFF Module for slot-based classification
    
    Args:
        in_channels: Input image channels (default: 3 for RGB)
        base_dim: Base embedding dimension (default: 32)
        num_slots: Number of semantic slots K (default: 5)
        slot_iters: Number of slot attention iterations (default: 5)
        num_merging_stages: Number of PatchMerging2D stages (default: 2)
        lambda_init: Initial value for residual scaling λ (default: 2.0)
        mask_threshold: Threshold for binary masking (default: None for weighted)
        device: Device to use
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        base_dim: int = 32,
        num_slots: int = 5,
        slot_iters: int = 5,
        num_merging_stages: int = 2,
        lambda_init: float = 1.0,
        mask_threshold: Optional[float] = None,
        device: str = 'cuda',
        **kwargs  # For backward compatibility and ablation config
    ):
        super().__init__()
        
        self.base_dim = base_dim
        self.num_slots = num_slots
        self.device = device
        
        # === Ablation Config ===
        ablation_type = kwargs.get('ablation_type', None)
        ablation_mode = kwargs.get('ablation_mode', None)
        
        # SAFF ablation: 'with' (default) or 'without'
        self.use_saff = not (ablation_type == 'saff' and ablation_mode == 'without')
        
        # DualBranch ablation: 'both' (default), 'local_only', 'global_only'
        dual_branch_mode = 'both'
        if ablation_type == 'dual_branch':
            dual_branch_mode = ablation_mode
        
        # === Backbone ===
        
        # Stage 1: Overlapping Patch Embedding
        # (B, 3, 64, 64) → (B, base_dim, 64, 64)
        self.patch_embed = PatchEmbed2D(
            in_channels=in_channels,
            embed_dim=base_dim,
            kernel_size=3,
            norm_layer=nn.LayerNorm
        )
        
        # Stage 2: Hierarchical PatchMerging2D
        # Each stage: spatial /2, channels ×2
        self.merging_stages = nn.ModuleList()
        current_dim = base_dim
        
        for i in range(num_merging_stages):
            self.merging_stages.append(
                PatchMerging2D(dim=current_dim, norm_layer=nn.LayerNorm)
            )
            current_dim = current_dim * 2
        
        # Final channel dimension (128 for 2 merge stages with base_dim=32)
        self.hidden_dim = current_dim
        
        # Number of patches after merging
        # 64→32→16, so num_patches = 16×16 = 256
        self.num_patches = (64 // (2 ** num_merging_stages)) ** 2
        
        # Stage 3: DualBranchFusion with ablation mode
        self.dual_branch = DualBranchFusion(
            channels=self.hidden_dim,
            d_state=16,
            dilation=2,
            mode=dual_branch_mode  # 'both', 'local_only', 'global_only'
        )
        
        # === SAFF Module (conditional) ===
        temperature = kwargs.get('temperature', 0.5)
        
        if self.use_saff:
            self.saff = SAFFModule(
                dim=self.hidden_dim,
                num_slots=num_slots,
                num_iters=slot_iters,
                num_patches=self.num_patches,
                lambda_init=lambda_init,
                temperature=temperature,
                debug=True
            )
        else:
            # Simple cosine similarity head when SAFF disabled
            self.saff = None
            self.temperature = temperature
        
        self.to(device)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patch features using backbone.
        
        Args:
            x: (B, C, H, W) input images
            
        Returns:
            patches: (B, N, hidden_dim) patch embeddings
        """
        # Stage 1: Patch embedding
        features = self.patch_embed(x)  # (B, 32, 64, 64)
        
        # Stage 2: Hierarchical merging
        for merge in self.merging_stages:
            features = merge(features)  # (B, 64, 32, 32) → (B, 128, 16, 16)
        
        # Stage 3: DualBranch fusion
        features = self.dual_branch(features)  # (B, 128, 16, 16)
        
        # Convert to patches: (B, C, H, W) → (B, N, C)
        patches = rearrange(features, 'b c h w -> b (h w) c')
        
        # Debug logging for feature extractor (periodic)
        if self.training and hasattr(self, '_feat_debug_counter'):
            self._feat_debug_counter += 1
            if self._feat_debug_counter % 100 == 1:
                with torch.no_grad():
                    # Patch statistics
                    patch_norm = patches.norm(dim=-1)  # (B, N)
                    patch_mean = patches.mean()
                    patch_std = patches.std()
                    print(f"[DEBUG-FEAT] patches: shape={patches.shape}, "
                          f"mean={patch_mean.item():.4f}, std={patch_std.item():.4f}, "
                          f"norm_range=[{patch_norm.min().item():.2f}, {patch_norm.max().item():.2f}]")
        elif self.training and not hasattr(self, '_feat_debug_counter'):
            self._feat_debug_counter = 0
        
        return patches
    
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
            q_patches = self.extract_features(q_b)  # (NQ, N, hidden_dim)
            
            # === Extract Support Features per Class ===
            support_patches_list = []
            
            for w in range(Way):
                s_class = support[b, w]  # (Shot, C, H, W)
                s_patches = self.extract_features(s_class)  # (Shot, N, hidden_dim)
                support_patches_list.append(s_patches)
            
            # === Classification ===
            if self.use_saff:
                # SAFF Forward
                scores = self.saff(
                    query_patches=q_patches,
                    support_patches_list=support_patches_list
                )  # (NQ, Way)
            else:
                # Simple cosine similarity (SAFF disabled ablation)
                scores = self._simple_cosine_forward(q_patches, support_patches_list)
            
            scores_list.append(scores)
        
        # Concatenate all batches
        all_scores = torch.cat(scores_list, dim=0)  # (B*NQ, Way)
        
        return all_scores
    
    def _simple_cosine_forward(
        self, 
        query_patches: torch.Tensor, 
        support_patches_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """Simple prototype-based cosine similarity (when SAFF disabled).
        
        Args:
            query_patches: (NQ, N, C) query patch embeddings
            support_patches_list: List of (Shot, N, C) support patches per class
            
        Returns:
            scores: (NQ, Way) similarity scores
        """
        NQ, N, C = query_patches.shape
        Way = len(support_patches_list)
        
        # Global average pooling for query: (NQ, N, C) -> (NQ, C)
        q_proto = query_patches.mean(dim=1)  # (NQ, C)
        q_norm = F.normalize(q_proto, dim=-1)  # (NQ, C)
        
        scores = []
        for w in range(Way):
            s_patches = support_patches_list[w]  # (Shot, N, C)
            # Global average pooling: (Shot, N, C) -> (Shot, C) -> (C,)
            s_proto = s_patches.mean(dim=(0, 1))  # (C,)
            s_norm = F.normalize(s_proto, dim=-1)  # (C,)
            
            # Cosine similarity: (NQ, C) @ (C,) -> (NQ,)
            sim = torch.einsum('qc,c->q', q_norm, s_norm)
            # Temperature scaling
            sim = sim / self.temperature
            scores.append(sim)
        
        # Stack: (NQ, Way)
        scores = torch.stack(scores, dim=1)
        return scores
    
    def get_slot_features(self, images: torch.Tensor) -> tuple:
        """Extract slot features for visualization.
        
        Args:
            images: (B, C, H, W) input images
            
        Returns:
            slots: (B, K, hidden_dim) slot descriptors
            attn: (B, K, N) attention weights
        """
        patches = self.extract_features(images)
        refined, slots, attn, mask = self.saff.encode(patches)
        return slots, attn
