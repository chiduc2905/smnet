"""SMNet (Slot Mamba Network) for few-shot learning.

Architecture (RGB 64×64 with Dual-Path Residual):
- Backbone: PatchEmbed + PatchMerging (M1→M2)
- Path A: GAP + cosine (similarity-only, no learnable params)
- Path B: DualBranch + SAFF (slot-based refinement)
- Fusion: score = s_A + λ * s_B (λ learnable)

Pipeline:
    Input:       (B, 3, 64, 64)    ← RGB
    PatchEmbed:  (B, 32, 64, 64)   ← M1
    Merge1-2:    (B, 128, 16, 16)  ← M2 (256 patches)
    
    Path A: GAP(M2) → normalize → cosine → s_A
    Path B: DualBranch(M2) → SAFF → cosine → s_B
    
    Fusion: s_A + λ * s_B
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
    """SMNet: Slot Mamba Network with Dual-Path Residual for few-shot classification.
    
    Architecture:
        Path A: Backbone (M2) → GAP → cosine similarity (anchor)
        Path B: Backbone (M2) → DualBranch → SAFF → similarity (refiner)
        Fusion: score = s_A + λ * s_B
    
    Args:
        in_channels: Input image channels (default: 3 for RGB)
        base_dim: Base embedding dimension (default: 32)
        num_slots: Number of semantic slots K (default: 5)
        slot_iters: Number of slot attention iterations (default: 5)
        num_merging_stages: Number of PatchMerging2D stages (default: 2)
        lambda_init: Initial value for residual scaling λ (default: 0.3)
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
        lambda_init: float = 0.3,  # Lower init for residual path
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
        
        # === Backbone (M1 + M2) ===
        
        # Stage 1: Overlapping Patch Embedding (M1)
        # (B, 3, 64, 64) → (B, base_dim, 64, 64)
        self.patch_embed = PatchEmbed2D(
            in_channels=in_channels,
            embed_dim=base_dim,
            kernel_size=3,
            norm_layer=nn.LayerNorm
        )
        
        # Stage 2: Hierarchical PatchMerging2D (M2)
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
        
        # === Path B: DualBranch + SAFF ===
        self.dual_branch = DualBranchFusion(
            channels=self.hidden_dim,
            d_state=16,
            dilation=2,
            mode=dual_branch_mode  # 'both', 'local_only', 'global_only'
        )
        
        # Temperature for Path A cosine similarity
        temperature = kwargs.get('temperature', 0.5)
        self.temperature = temperature
        
        if self.use_saff:
            self.saff = SAFFModule(
                dim=self.hidden_dim,
                num_slots=num_slots,
                num_iters=slot_iters,
                num_patches=self.num_patches,
                lambda_init=1.0,  # Internal SAFF lambda
                temperature=temperature
            )
        else:
            self.saff = None
        
        # === Dual-Path Fusion ===
        # Learnable residual lambda: score = s_A + λ * s_B
        self.residual_lambda = nn.Parameter(torch.tensor(lambda_init))
        
        self.to(device)
    
    def extract_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from backbone (M1 + M2) BEFORE DualBranch.
        
        This is used for Path A (similarity-only path).
        
        Args:
            x: (B, C, H, W) input images
            
        Returns:
            features: (B, C, H', W') feature maps after M2
        """
        # Stage 1: Patch embedding (M1)
        features = self.patch_embed(x)  # (B, 32, 64, 64)
        
        # Stage 2: Hierarchical merging (M2)
        for merge in self.merging_stages:
            features = merge(features)  # (B, 64, 32, 32) → (B, 128, 16, 16)
        
        return features  # (B, 128, 16, 16)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patch features using full backbone + DualBranch.
        
        This is used for Path B (SAFF path).
        
        Args:
            x: (B, C, H, W) input images
            
        Returns:
            patches: (B, N, hidden_dim) patch embeddings
        """
        # Backbone: M1 + M2
        features = self.extract_backbone_features(x)  # (B, 128, 16, 16)
        
        # DualBranch fusion
        features = self.dual_branch(features)  # (B, 128, 16, 16)
        
        # Convert to patches: (B, C, H, W) → (B, N, C)
        patches = rearrange(features, 'b c h w -> b (h w) c')
        
        return patches
    
    def _compute_path_a_scores(
        self,
        query_features: torch.Tensor,
        support_features_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """Compute Path A scores: simple GAP + cosine similarity.
        
        NO learnable parameters - pure similarity anchor.
        
        Args:
            query_features: (NQ, C, H, W) query features from M2
            support_features_list: List of (Shot, C, H, W) support features per class
            
        Returns:
            scores: (NQ, Way) similarity scores
        """
        NQ = query_features.shape[0]
        Way = len(support_features_list)
        
        # GAP for query: (NQ, C, H, W) → (NQ, C)
        q_gap = query_features.mean(dim=(2, 3))  # (NQ, C)
        q_norm = F.normalize(q_gap, dim=-1)  # (NQ, C)
        
        scores = []
        for w in range(Way):
            s_features = support_features_list[w]  # (Shot, C, H, W)
            
            # GAP for support: (Shot, C, H, W) → (Shot, C) → (C,)
            s_gap = s_features.mean(dim=(2, 3))  # (Shot, C)
            s_proto = s_gap.mean(dim=0)  # (C,) - class prototype
            s_norm = F.normalize(s_proto, dim=-1)  # (C,)
            
            # Cosine similarity: (NQ, C) @ (C,) → (NQ,)
            sim = torch.einsum('qc,c->q', q_norm, s_norm)
            # Temperature scaling
            sim = sim / self.temperature
            scores.append(sim)
        
        # Stack: (NQ, Way)
        scores = torch.stack(scores, dim=1)
        return scores
    
    def forward(self, query: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        """Compute few-shot classification scores using Dual-Path.
        
        Path A: Backbone → GAP → cosine (anchor)
        Path B: Backbone → DualBranch → SAFF (refiner)
        Fusion: score = s_A + λ * s_B
        
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
            
            # Path A: Backbone features (before DualBranch)
            q_backbone = self.extract_backbone_features(q_b)  # (NQ, 128, 16, 16)
            
            # Path B: Full features (with DualBranch)
            q_patches = self.extract_features(q_b)  # (NQ, N, hidden_dim)
            
            # === Extract Support Features per Class ===
            support_backbone_list = []  # For Path A
            support_patches_list = []    # For Path B
            
            for w in range(Way):
                s_class = support[b, w]  # (Shot, C, H, W)
                
                # Path A: Backbone features
                s_backbone = self.extract_backbone_features(s_class)  # (Shot, 128, 16, 16)
                support_backbone_list.append(s_backbone)
                
                # Path B: Full features
                s_patches = self.extract_features(s_class)  # (Shot, N, hidden_dim)
                support_patches_list.append(s_patches)
            
            # === Path A: Similarity-only (anchor) ===
            s_A = self._compute_path_a_scores(q_backbone, support_backbone_list)  # (NQ, Way)
            
            # === Path B: SAFF (refiner) ===
            if self.use_saff:
                s_B = self.saff(
                    query_patches=q_patches,
                    support_patches_list=support_patches_list
                )  # (NQ, Way)
            else:
                # If SAFF disabled, use simple cosine (same as Path A but with DualBranch features)
                s_B = self._simple_cosine_forward(q_patches, support_patches_list)
            
            # === Fusion: s_A + λ * s_B ===
            scores = s_A + self.residual_lambda * s_B  # (NQ, Way)
            
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
        if self.saff is not None:
            # Need to provide dummy class stats for encode
            from net.backbone.slot_covariance_attention import compute_class_covariance, compute_class_prototype
            class_cov = compute_class_covariance(patches)
            class_proto = compute_class_prototype(patches)
            refined, slots, alpha = self.saff.encode(patches, class_cov, class_proto)
            return slots, self.saff.slot_attention(patches)[1]
        return None, None
