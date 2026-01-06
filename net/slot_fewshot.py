"""SMNet (Slot Mamba Network) for few-shot learning.

Architecture (RGB 64×64 with Dual-Path Residual):
- Backbone: PatchEmbed + PatchMerging (M1→M2)
- Path A: GAP + cosine (similarity-only, NO gradient through backbone)
- Path B: DualBranch + SAFF (slot-based refinement)
- Fusion: score = s_A + λ * s_B (λ learnable)

CRITICAL FIXES:
1. Path A uses .detach() to stop gradient through backbone
2. SAFF applies mask to SIMILARITY, not to FEATURES
3. Scale normalization for s_B before fusion
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
        Path A: Backbone (M2).detach() → GAP → cosine similarity (anchor, NO gradient)
        Path B: Backbone (M2) → DualBranch → SAFF → similarity (refiner)
        Fusion: score = s_A + λ * s_B
    
    Args:
        in_channels: Input image channels (default: 3 for RGB)
        base_dim: Base embedding dimension (default: 32)
        num_slots: Number of semantic slots K (default: 5)
        slot_iters: Number of slot attention iterations (default: 5)
        num_merging_stages: Number of PatchMerging2D stages (default: 2)
        lambda_init: Initial value for residual scaling λ (default: 0.3)
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
        
        # Debug mode for logging s_A vs s_B scales
        self.debug_mode = kwargs.get('debug_mode', False)
        self._debug_counter = 0
        
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
        self.patch_embed = PatchEmbed2D(
            in_channels=in_channels,
            embed_dim=base_dim,
            kernel_size=3,
            norm_layer=nn.LayerNorm
        )
        
        # Stage 2: Hierarchical PatchMerging2D (M2)
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
        self.num_patches = (64 // (2 ** num_merging_stages)) ** 2
        
        # === Path B: DualBranch + SAFF ===
        self.dual_branch = DualBranchFusion(
            channels=self.hidden_dim,
            d_state=16,
            dilation=2,
            mode=dual_branch_mode
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
                lambda_init=1.0,
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
        
        🔥 CRITICAL: Features are DETACHED - no gradient through backbone!
        
        Args:
            query_features: (NQ, C, H, W) query features from M2 (DETACHED)
            support_features_list: List of (Shot, C, H, W) support features per class (DETACHED)
            
        Returns:
            scores: (NQ, Way) similarity scores (no gradient to backbone)
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
        
        Path A: Backbone.detach() → GAP → cosine (anchor, NO gradient)
        Path B: Backbone → DualBranch → SAFF (refiner)
        Fusion: score = s_A + λ * normalize(s_B)
        
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
            
            # Backbone features (shared computation)
            q_backbone = self.extract_backbone_features(q_b)  # (NQ, 128, 16, 16)
            
            # 🔥 FIX 1: DETACH for Path A - no gradient through backbone!
            q_backbone_detached = q_backbone.detach()
            
            # Path B: Full features (with DualBranch) - gradient flows here
            q_dual = self.dual_branch(q_backbone)  # (NQ, 128, 16, 16)
            q_patches = rearrange(q_dual, 'b c h w -> b (h w) c')  # (NQ, N, C)
            
            # === Extract Support Features per Class ===
            support_backbone_list = []  # For Path A (detached)
            support_patches_list = []    # For Path B
            
            for w in range(Way):
                s_class = support[b, w]  # (Shot, C, H, W)
                
                # Backbone features
                s_backbone = self.extract_backbone_features(s_class)  # (Shot, 128, 16, 16)
                
                # 🔥 FIX 1: DETACH for Path A
                support_backbone_list.append(s_backbone.detach())
                
                # Path B: Full features
                s_dual = self.dual_branch(s_backbone)
                s_patches = rearrange(s_dual, 'b c h w -> b (h w) c')
                support_patches_list.append(s_patches)
            
            # === Path A: Similarity-only (anchor, NO gradient) ===
            s_A = self._compute_path_a_scores(q_backbone_detached, support_backbone_list)  # (NQ, Way)
            
            # === Path B: SAFF (refiner) ===
            if self.use_saff:
                s_B = self.saff(
                    query_patches=q_patches,
                    support_patches_list=support_patches_list
                )  # (NQ, Way)
            else:
                s_B = self._simple_cosine_forward(q_patches, support_patches_list)
            
            # 🔥 FIX 2: Normalize s_B to prevent scale domination
            s_B_normalized = s_B / (s_B.std() + 1e-6)
            
            # === Debug logging ===
            if self.training and self.debug_mode:
                self._debug_counter += 1
                if self._debug_counter % 50 == 1:
                    print(f"[DEBUG] |s_A|={s_A.abs().mean().item():.4f}, "
                          f"|s_B|={s_B.abs().mean().item():.4f}, "
                          f"|s_B_norm|={s_B_normalized.abs().mean().item():.4f}, "
                          f"λ={self.residual_lambda.item():.4f}")
            
            # === Fusion: s_A + λ * s_B_normalized ===
            scores = s_A + self.residual_lambda * s_B_normalized  # (NQ, Way)
            
            scores_list.append(scores)
        
        # Concatenate all batches
        all_scores = torch.cat(scores_list, dim=0)  # (B*NQ, Way)
        
        return all_scores
    
    def _simple_cosine_forward(
        self, 
        query_patches: torch.Tensor, 
        support_patches_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """Simple prototype-based cosine similarity (when SAFF disabled)."""
        NQ, N, C = query_patches.shape
        Way = len(support_patches_list)
        
        q_proto = query_patches.mean(dim=1)
        q_norm = F.normalize(q_proto, dim=-1)
        
        scores = []
        for w in range(Way):
            s_patches = support_patches_list[w]
            s_proto = s_patches.mean(dim=(0, 1))
            s_norm = F.normalize(s_proto, dim=-1)
            
            sim = torch.einsum('qc,c->q', q_norm, s_norm)
            sim = sim / self.temperature
            scores.append(sim)
        
        scores = torch.stack(scores, dim=1)
        return scores
    
    def get_slot_features(self, images: torch.Tensor) -> tuple:
        """Extract slot features for visualization."""
        patches = self.extract_features(images)
        if self.saff is not None:
            from net.backbone.slot_covariance_attention import compute_class_covariance, compute_class_prototype
            class_cov = compute_class_covariance(patches)
            class_proto = compute_class_prototype(patches)
            refined, slots, alpha = self.saff.encode(patches, class_cov, class_proto)
            return slots, self.saff.slot_attention(patches)[1]
        return None, None
