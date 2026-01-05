"""SAFF Module with SCA + CMA Integration.

Architecture (combining SAFF + SCA + CMA):
    1. SlotAttention: Extract K slots from patches
    2. SCA (SlotCovarianceAttention): Compute slot priority α_k via covariance matching
    3. SAFF-style Patch Refinement: mask = Σ_k attn[k,i] * α_k
    4. CMA (ChannelMetricAttention): Channel-level slot refinement  
    5. Similarity: Cosine + MLP → Score

Key design:
    - SAFF handles patch-level refinement
    - SCA handles slot-level priority (replaces simple cosine filtering)
    - CMA handles channel-level refinement (after SAFF, not before)
    
Based on: "Slot Attention-based Feature Filtering for Few-Shot Learning"
arXiv:2508.09699, CVPR 2025 Workshop
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional

from net.backbone.slot_covariance_attention import (
    SlotCovarianceAttention, compute_class_covariance, compute_class_prototype
)
from net.backbone.channel_metric_attention import ChannelMetricAttention

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


class SAFFSlotAttention(nn.Module):
    """Slot Attention module for SAFF using Mamba.
    
    Iteratively refines slots to bind to different parts of input patches.
    Uses Mamba for slot update and interaction.
    
    Args:
        dim: Feature dimension
        num_slots: Number of slots K
        num_iters: Number of refinement iterations
        eps: Epsilon for numerical stability
    """
    
    def __init__(
        self,
        dim: int,
        num_slots: int = 5,
        num_iters: int = 5,
        eps: float = 1e-8
    ):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.num_iters = num_iters
        self.eps = eps
        
        if not MAMBA_AVAILABLE:
            raise ImportError("mamba-ssm package is required for SAFFSlotAttention")
        
        # Slot initialization (learnable)
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.ones(1, 1, dim) * 0.1)
        
        # Layer norms
        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        
        # Attention projections
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        
        # Mamba for slot update (replaces GRU)
        # Processes sequence of K slots
        self.mamba = Mamba(
            d_model=dim,
            d_state=16,
            d_conv=4,
            expand=1  # Lightweight
        )
        self.norm_mamba = nn.LayerNorm(dim)
        
        self.scale = dim ** -0.5
    
    def forward(self, patches: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            patches: (B, N, C) patch embeddings
            
        Returns:
            slots: (B, K, C) slot descriptors
            attn: (B, K, N) attention weights
        """
        B, N, C = patches.shape
        
        # Initialize slots (with learnable parameters + noise)
        slots = self.slots_mu.expand(B, self.num_slots, -1)
        slots = slots + self.slots_sigma * torch.randn_like(slots)
        
        # Normalize inputs
        patches_norm = self.norm_input(patches)
        
        # Keys and values from patches
        k = self.to_k(patches_norm)  # (B, N, C)
        v = self.to_v(patches_norm)  # (B, N, C)
        
        attn = None
        
        for _ in range(self.num_iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            
            # Query from slots
            q = self.to_q(slots)  # (B, K, C)
            
            # Attention: slots query patches
            dots = torch.einsum('bkc,bnc->bkn', q, k) * self.scale
            
            # Softmax over slots (competition for patches)
            attn = F.softmax(dots, dim=1)  # (B, K, N)
            
            # Weighted mean of values
            attn_weights = attn / (attn.sum(dim=-1, keepdim=True) + self.eps)
            updates = torch.einsum('bkn,bnc->bkc', attn_weights, v)
            
            # Mamba update: Refine slots based on updates
            # 1. Residual addition of updates (Gradient-like step)
            slots = slots_prev + updates
            
            # 2. Mamba mixing (Slot interaction and refinement)
            # Normalize before Mamba
            slots = slots + self.mamba(self.norm_mamba(slots))
        
        return slots, attn


class SAFFModule(nn.Module):
    """SAFF Module with SCA + CMA Integration.
    
    Complete pipeline:
        SlotAttention → SCA → Patch Refinement → CMA → Similarity
    
    Args:
        dim: Feature dimension
        num_slots: Number of slots K (default: 5)
        num_iters: Slot attention iterations (default: 5)
        num_patches: Number of patches N (default: 256)
        lambda_init: Residual scaling for patch refinement (default: 2.0)
    """
    
    def __init__(
        self,
        dim: int,
        num_slots: int = 5,
        num_iters: int = 5,
        num_patches: int = 256,
        lambda_init: float = 2.0,
        temperature: float = 0.1,  # Temperature for similarity scaling
        debug: bool = True  # Enable debug logging
    ):
        super().__init__()
        
        self.dim = dim
        self.num_slots = num_slots
        self.num_patches = num_patches
        self.temperature = temperature
        self.debug = debug
        self._debug_counter = 0  # For periodic debug logging
        
        # Step 1: Slot Attention (Mamba)
        self.slot_attention = SAFFSlotAttention(
            dim=dim,
            num_slots=num_slots,
            num_iters=num_iters
        )
        
        # Step 2: SCA (Slot Covariance Attention) - slot priority
        self.sca = SlotCovarianceAttention(dim=dim, temperature=1.0)
        
        # Step 3: Patch refinement scaling λ
        self.lambda_scale = nn.Parameter(torch.tensor(lambda_init))
        self.patch_norm = nn.LayerNorm(dim)
        
        # Step 4: CMA (Channel Metric Attention) - channel refinement
        self.cma = ChannelMetricAttention(dim=dim, temperature=1.0, use_residual=True)
        
        # Step 5: Similarity Classifier (Conv1d based)
        # Matches architecture in few-shot-mamba/net/new_proposed.py
        # Input: (NQ, 1, N) -> Output (NQ, 1, 1) -> Squeeze
        self.classifier1 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.5),  # Default dropout p=0.5 as in reference
            nn.Conv1d(1, 1, kernel_size=num_patches, stride=num_patches, bias=True)
        )
    
    def compute_attention_mask(
        self,
        attn: torch.Tensor,
        alpha: torch.Tensor
    ) -> torch.Tensor:
        """Compute combined attention mask from slot attention and slot priority.
        
        SAFF-style: mask_i = Σ_k attn[k,i] * α_k
        
        Args:
            attn: (B, K, N) slot attention weights
            alpha: (B, K) slot priority from SCA
            
        Returns:
            mask: (B, N) patch-level attention mask
        """
        # alpha: (B, K) → (B, K, 1)
        alpha_expanded = alpha.unsqueeze(-1)  # (B, K, 1)
        
        # mask_i = Σ_k attn[k,i] * α_k
        # (B, K, N) * (B, K, 1) → sum over K → (B, N)
        mask = (attn * alpha_expanded).sum(dim=1)  # (B, N)
        
        # Normalize to [0, 1]
        mask = mask / (mask.max(dim=-1, keepdim=True)[0] + 1e-8)
        
        return mask
    
    def refine_patches(
        self,
        patches: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """SAFF-style patch refinement.
        
        refined_patch = patch + λ * (mask * patch)
        
        Args:
            patches: (B, N, C) original patches
            mask: (B, N) attention mask
            
        Returns:
            refined: (B, N, C) refined patches
        """
        masked = mask.unsqueeze(-1) * patches  # (B, N, C)
        refined = patches + self.lambda_scale * masked
        refined = self.patch_norm(refined)
        return refined
    
    def compute_similarity(
        self,
        query_patches: torch.Tensor,
        support_patches: torch.Tensor
    ) -> torch.Tensor:
        """Compute patch-to-patch similarity.
        
        Args:
            query_patches: (NQ, N, C) refined query patches
            support_patches: (Shot, N, C) refined support patches
            
        Returns:
            similarity: (NQ,) scores
        """
        NQ, N, C = query_patches.shape
        
        # Normalize
        q_norm = F.normalize(query_patches, dim=-1)
        s_norm = F.normalize(support_patches, dim=-1)
        
        # Average support across shots
        s_avg = s_norm.mean(dim=0)  # (N, C)
        
        # Similarity matrix: (NQ, N, N)
        sim_matrix = torch.einsum('qnc,mc->qnm', q_norm, s_avg)
        
        # Temperature scaling - amplify gradients
        sim_matrix = sim_matrix / self.temperature
        
        # Max over support patches for each query patch
        max_sim = sim_matrix.max(dim=-1)[0]  # (NQ, N)
        
        # Debug logging (every 100 calls)
        if self.debug and self.training:
            self._debug_counter += 1
            if self._debug_counter % 100 == 1:
                print(f"[DEBUG] max_sim: range=[{max_sim.min().item():.3f}, {max_sim.max().item():.3f}], "
                      f"mean={max_sim.mean().item():.3f}, std={max_sim.std().item():.4f}")
        
        # Classifier1 (Conv1d-based scoring)
        # Input: (NQ, N) -> (NQ, 1, N) for Conv1d
        max_sim = max_sim.unsqueeze(1)
        score = self.classifier1(max_sim)  # (NQ, 1, 1)
        score = score.squeeze(-1).squeeze(-1)  # (NQ,)
        
        return score
    
    def encode(
        self,
        patches: torch.Tensor,
        class_cov: torch.Tensor,
        class_proto: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode patches through full SAFF + SCA + CMA pipeline.
        
        Args:
            patches: (B, N, C) patch embeddings
            class_cov: (C, C) class covariance for SCA
            class_proto: (C,) class prototype for CMA
            
        Returns:
            refined_patches: (B, N, C)
            refined_slots: (B, K, C)
            alpha: (B, K) slot priorities
        """
        # Step 1: Slot Attention
        slots, attn = self.slot_attention(patches)  # (B, K, C), (B, K, N)
        
        # Step 2: SCA - compute slot priority α_k
        alpha = self.sca(slots, attn, patches, class_cov)  # (B, K)
        
        # Step 3: SAFF-style Patch Refinement
        mask = self.compute_attention_mask(attn, alpha)  # (B, N)
        refined_patches = self.refine_patches(patches, mask)  # (B, N, C)
        
        # Step 4: CMA - channel-level slot refinement
        refined_slots = self.cma(slots, class_proto, slot_weights=alpha)  # (B, K, C)
        
        return refined_patches, refined_slots, alpha
    
    def forward(
        self,
        query_patches: torch.Tensor,
        support_patches_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """Full SAFF forward pass.
        
        Args:
            query_patches: (NQ, N, C) query patch embeddings
            support_patches_list: List of (Shot, N, C) support patches per class
            
        Returns:
            scores: (NQ, M) similarity scores, M = number of classes
        """
        NQ, N, C = query_patches.shape
        M = len(support_patches_list)
        
        scores = []
        
        for m in range(M):
            support_patches = support_patches_list[m]  # (Shot, N, C)
            
            # Compute class statistics from support
            class_cov = compute_class_covariance(support_patches)  # (C, C)
            class_proto = compute_class_prototype(support_patches)  # (C,)
            
            # Encode support
            s_refined, s_slots, s_alpha = self.encode(
                support_patches, class_cov, class_proto
            )
            
            # Encode query (using same class statistics)
            q_refined, q_slots, q_alpha = self.encode(
                query_patches, class_cov, class_proto
            )
            
            # Compute similarity
            sim = self.compute_similarity(q_refined, s_refined)  # (NQ,)
            scores.append(sim)
        
        # Stack: (NQ, M)
        scores = torch.stack(scores, dim=1)
        
        return scores
