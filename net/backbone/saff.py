"""SAFF Module with Similarity-based Masking.

CRITICAL FIX: Mask is applied to SIMILARITY, NOT to FEATURES.
This prevents SAFF from being a "hidden classifier" that overfits.

Architecture:
    1. SlotAttention: Extract K slots from patches
    2. SCA (SlotCovarianceAttention): Compute slot priority α_k
    3. Compute raw patch-to-patch similarity
    4. Apply mask to SIMILARITY (MatchingNet-style refinement)
    5. Conv1D → Final Score
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
        self.mamba = Mamba(
            d_model=dim,
            d_state=16,
            d_conv=4,
            expand=1
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
        
        # Initialize slots
        slots = self.slots_mu.expand(B, self.num_slots, -1)
        slots = slots + self.slots_sigma * torch.randn_like(slots)
        
        # Normalize inputs
        patches_norm = self.norm_input(patches)
        
        # Keys and values from patches
        k = self.to_k(patches_norm)
        v = self.to_v(patches_norm)
        
        attn = None
        
        for _ in range(self.num_iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            
            q = self.to_q(slots)
            
            # Attention: slots query patches
            dots = torch.einsum('bkc,bnc->bkn', q, k) * self.scale
            
            # Softmax over slots (competition for patches)
            attn = F.softmax(dots, dim=1)
            
            # Weighted mean of values
            attn_weights = attn / (attn.sum(dim=-1, keepdim=True) + self.eps)
            updates = torch.einsum('bkn,bnc->bkc', attn_weights, v)
            
            # Mamba update
            slots = slots_prev + updates
            slots = slots + self.mamba(self.norm_mamba(slots))
        
        return slots, attn


class SAFFModule(nn.Module):
    """SAFF Module with Similarity-based Masking.
    
    🔥 CRITICAL FIX: Mask is applied to SIMILARITY, NOT to FEATURES.
    
    Pipeline:
        1. SlotAttention → slots, attn
        2. SCA → α_k (slot priority)
        3. Compute attention mask from attn * α
        4. Compute raw patch-to-patch cosine similarity
        5. Apply mask to similarity: masked_sim = mask * sim
        6. Aggregate → score
    """
    
    def __init__(
        self,
        dim: int,
        num_slots: int = 5,
        num_iters: int = 5,
        num_patches: int = 256,
        lambda_init: float = 1.0,
        temperature: float = 0.5
    ):
        super().__init__()
        
        self.dim = dim
        self.num_slots = num_slots
        self.num_patches = num_patches
        self.temperature = temperature
        
        # Step 1: Slot Attention (Mamba)
        self.slot_attention = SAFFSlotAttention(
            dim=dim,
            num_slots=num_slots,
            num_iters=num_iters
        )
        
        # Step 2: SCA (Slot Covariance Attention) - slot priority
        self.sca = SlotCovarianceAttention(dim=dim, temperature=1.5)
        
        # Step 3: CMA (Channel Metric Attention) - optional channel refinement
        self.cma = ChannelMetricAttention(dim=dim, temperature=1.5, use_residual=True)
        
        # Similarity aggregation (Conv1d based)
        self.classifier1 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.5),
            nn.Conv1d(1, 1, kernel_size=num_patches, stride=num_patches, bias=True)
        )
    
    def compute_attention_mask(
        self,
        attn: torch.Tensor,
        alpha: torch.Tensor
    ) -> torch.Tensor:
        """Compute combined attention mask from slot attention and slot priority.
        
        mask_i = Σ_k attn[k,i] * α_k
        
        Args:
            attn: (B, K, N) slot attention weights
            alpha: (B, K) slot priority from SCA
            
        Returns:
            mask: (B, N) patch-level attention mask
        """
        alpha_expanded = alpha.unsqueeze(-1)  # (B, K, 1)
        mask = (attn * alpha_expanded).sum(dim=1)  # (B, N)
        
        # Normalize to [0, 1]
        mask = mask / (mask.max(dim=-1, keepdim=True)[0] + 1e-8)
        
        return mask
    
    def compute_masked_similarity(
        self,
        query_patches: torch.Tensor,
        support_patches: torch.Tensor,
        query_mask: torch.Tensor,
        support_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute similarity with mask applied to SIMILARITY, not features.
        
        🔥 CRITICAL: MatchingNet-style refinement
        
        sim_patch = cosine(q_patch, s_patch)
        scored_sim = mask * sim_patch
        score = aggregate(scored_sim)
        
        Args:
            query_patches: (NQ, N, C) query patches (RAW, not refined)
            support_patches: (Shot, N, C) support patches (RAW, not refined)
            query_mask: (NQ, N) attention mask for query
            support_mask: (Shot, N) attention mask for support
            
        Returns:
            score: (NQ,) similarity scores
        """
        NQ, N, C = query_patches.shape
        Shot = support_patches.shape[0]
        
        # Normalize patches for cosine similarity
        q_norm = F.normalize(query_patches, dim=-1)  # (NQ, N, C)
        s_norm = F.normalize(support_patches, dim=-1)  # (Shot, N, C)
        
        # Average support across shots
        s_avg = s_norm.mean(dim=0)  # (N, C)
        s_mask_avg = support_mask.mean(dim=0)  # (N,)
        
        # Patch-to-patch similarity: (NQ, N, N)
        # sim[q, i, j] = cosine(q_patch_i, s_patch_j)
        sim_matrix = torch.einsum('qic,jc->qij', q_norm, s_avg)
        
        # Temperature scaling
        sim_matrix = sim_matrix / self.temperature
        
        # Apply masks to similarity
        # Combined mask: query_mask * support_mask^T
        # mask_combined[q, i, j] = query_mask[q, i] * support_mask[j]
        mask_combined = torch.einsum('qi,j->qij', query_mask, s_mask_avg)  # (NQ, N, N)
        
        # Masked similarity
        masked_sim = sim_matrix * mask_combined  # (NQ, N, N)
        
        # Max over support patches for each query patch
        max_sim = masked_sim.max(dim=-1)[0]  # (NQ, N)
        
        # Aggregate with classifier
        max_sim = max_sim.unsqueeze(1)  # (NQ, 1, N)
        score = self.classifier1(max_sim)  # (NQ, 1, 1)
        score = score.squeeze(-1).squeeze(-1)  # (NQ,)
        
        return score
    
    def forward(
        self,
        query_patches: torch.Tensor,
        support_patches_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """Full SAFF forward pass with similarity-based masking.
        
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
            class_cov = compute_class_covariance(support_patches)
            class_proto = compute_class_prototype(support_patches)
            
            # === Query: Slot Attention + SCA → mask ===
            q_slots, q_attn = self.slot_attention(query_patches)  # (NQ, K, C), (NQ, K, N)
            q_alpha = self.sca(q_slots, q_attn, query_patches, class_cov)  # (NQ, K)
            q_mask = self.compute_attention_mask(q_attn, q_alpha)  # (NQ, N)
            
            # === Support: Slot Attention + SCA → mask ===
            s_slots, s_attn = self.slot_attention(support_patches)  # (Shot, K, C), (Shot, K, N)
            s_alpha = self.sca(s_slots, s_attn, support_patches, class_cov)  # (Shot, K)
            s_mask = self.compute_attention_mask(s_attn, s_alpha)  # (Shot, N)
            
            # === Compute similarity with mask on SIMILARITY ===
            # NO feature refinement - mask is applied to similarity!
            sim = self.compute_masked_similarity(
                query_patches,   # RAW patches
                support_patches, # RAW patches
                q_mask,
                s_mask
            )  # (NQ,)
            
            scores.append(sim)
        
        # Stack: (NQ, M)
        scores = torch.stack(scores, dim=1)
        
        return scores
    
    def encode(
        self,
        patches: torch.Tensor,
        class_cov: torch.Tensor,
        class_proto: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode patches (for backward compatibility / visualization)."""
        slots, attn = self.slot_attention(patches)
        alpha = self.sca(slots, attn, patches, class_cov)
        mask = self.compute_attention_mask(attn, alpha)
        
        # Return mask as "refined" placeholder (not actually refining features)
        return patches, slots, alpha
