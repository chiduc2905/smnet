"""Class-Aware Slot-Based Few-Shot Inference Head.

Implements a slot-based, class-aware few-shot inference head following the design
in recent slot-attention-based few-shot learning papers.

Pipeline:
    1. Patch Tokenization: F → {p₁, p₂, ..., pₙ}, N = H×W
    2. Slot Attention: Semantic grouping with attention matrix A ∈ R^(N×K)
    3. Class Embedding: Aggregate support slots per class
    4. Slot Filtering: Cosine similarity weighting
    5. Slot-to-Patch Mask: Project filtered slots to patch space
    6. Patch Refinement: Class-conditioned residual refinement
    7. Similarity: Cosine similarity for classification
    8. Prediction: Highest similarity class

Reference Architecture:
    Input patches (from DualBranchFusion output)
         ↓
    ┌─────────────────────────────────────────────┐
    │ Slot Attention → Slots (K×C) + Attn (N×K)  │
    └─────────────────────────────────────────────┘
         ↓
    ┌─────────────────────────────────────────────┐
    │ For each class m:                           │
    │   1. Compute class embedding cₘ             │
    │   2. Slot filtering: αₖ = cos(sₖ, cₘ)       │
    │   3. Patch mask: mᵢ = Σₖ Aᵢₖ · αₖ          │
    │   4. Refined patches: p̃ᵢ = pᵢ + λ·mᵢ·pᵢ    │
    └─────────────────────────────────────────────┘
         ↓
    ┌─────────────────────────────────────────────┐
    │ Cosine similarity → Logits → Prediction     │
    └─────────────────────────────────────────────┘
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from einops import rearrange


class ClassAwareInferenceHead(nn.Module):
    """Class-Aware Inference Head for slot-based few-shot learning.
    
    Takes patch tokens and slot attention outputs, applies class-conditioned
    refinement, and computes similarity scores for few-shot classification.
    
    Args:
        dim: Feature dimension (channels)
        temperature: Temperature for cosine similarity (default: 0.1)
        lambda_init: Initial value for residual scaling λ (default: 0.5)
        learnable_lambda: Whether λ is learnable (default: True)
        use_mlp_similarity: Whether to use MLP on similarity scores (default: False)
    """
    
    def __init__(
        self,
        dim: int,
        temperature: float = 0.1,
        lambda_init: float = 0.5,
        learnable_lambda: bool = True,
        use_mlp_similarity: bool = False
    ):
        super().__init__()
        
        self.dim = dim
        self.temperature = temperature
        
        # Learnable residual scaling factor λ
        if learnable_lambda:
            self.lambda_scale = nn.Parameter(torch.tensor(lambda_init))
        else:
            self.register_buffer('lambda_scale', torch.tensor(lambda_init))
        
        # Optional MLP for similarity refinement
        self.use_mlp_similarity = use_mlp_similarity
        if use_mlp_similarity:
            self.similarity_mlp = nn.Sequential(
                nn.Linear(1, dim // 4),
                nn.ReLU(),
                nn.Linear(dim // 4, 1)
            )
        
        # Layer norm for patch refinement
        self.patch_norm = nn.LayerNorm(dim)
    
    def compute_class_embeddings(
        self,
        support_slots: List[torch.Tensor],
        support_weights: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """Compute class embeddings from support slots.
        
        Step 3: For each class, aggregate slot representations from support samples.
        
        Args:
            support_slots: List of (Shot, K, C) slot tensors, one per class
            support_weights: Optional list of (Shot, K) slot weights
            
        Returns:
            class_embeddings: (M, C) where M is number of classes
        """
        class_embeddings = []
        
        for m, slots in enumerate(support_slots):
            # slots: (Shot, K, C)
            Shot, K, C = slots.shape
            
            if support_weights is not None and support_weights[m] is not None:
                # Weight slots by existence scores
                weights = support_weights[m]  # (Shot, K)
                weighted_slots = slots * weights.unsqueeze(-1)  # (Shot, K, C)
                # Sum over slots and shots, normalize
                class_emb = weighted_slots.sum(dim=(0, 1)) / (weights.sum() + 1e-8)
            else:
                # Simple mean over slots and shots
                class_emb = slots.mean(dim=(0, 1))  # (C,)
            
            class_embeddings.append(class_emb)
        
        return torch.stack(class_embeddings, dim=0)  # (M, C)
    
    def filter_slots_by_class(
        self,
        slots: torch.Tensor,
        class_embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Filter slots via class similarity.
        
        Step 4: Compute cosine similarity between each slot and class embedding.
        
        Args:
            slots: (B, K, C) slot descriptors
            class_embedding: (C,) class embedding vector
            
        Returns:
            filtered_slots: (B, K, C) weighted slots
            alpha: (B, K) similarity weights
        """
        # Normalize for cosine similarity
        slots_norm = F.normalize(slots, dim=-1)  # (B, K, C)
        class_norm = F.normalize(class_embedding.unsqueeze(0), dim=-1)  # (1, C)
        
        # Cosine similarity: (B, K)
        alpha = torch.einsum('bkc,c->bk', slots_norm, class_norm.squeeze(0))
        
        # Apply softmax for soft weighting (optional, can also use raw similarity)
        alpha_soft = F.softmax(alpha / self.temperature, dim=-1)
        
        # Weight slots: s̃ₖ = αₖ · sₖ
        filtered_slots = alpha_soft.unsqueeze(-1) * slots  # (B, K, C)
        
        return filtered_slots, alpha_soft
    
    def compute_patch_mask(
        self,
        attn: torch.Tensor,
        alpha: torch.Tensor
    ) -> torch.Tensor:
        """Compute class-conditioned patch mask.
        
        Step 5: Project filtered slots back to patch space.
        mᵢ = Σₖ Aᵢₖ · αₖ
        
        Args:
            attn: (B, K, N) attention matrix from slot attention
            alpha: (B, K) slot-class similarity weights
            
        Returns:
            mask: (B, N) patch-level mask
        """
        # attn: (B, K, N), alpha: (B, K)
        # mᵢ = Σₖ Aᵢₖ · αₖ
        # Transpose attn to (B, N, K) for einsum
        attn_t = attn.transpose(1, 2)  # (B, N, K)
        
        mask = torch.einsum('bnk,bk->bn', attn_t, alpha)  # (B, N)
        
        # Normalize mask to [0, 1]
        mask = mask / (mask.max(dim=-1, keepdim=True)[0] + 1e-8)
        
        return mask
    
    def refine_patches(
        self,
        patches: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply class-conditioned patch refinement.
        
        Step 6: pᵢ_refined = pᵢ + λ · mᵢ · pᵢ
        
        Args:
            patches: (B, N, C) patch tokens
            mask: (B, N) class-conditioned mask
            
        Returns:
            refined_patches: (B, N, C) refined patch tokens
        """
        # p̃ᵢ = mᵢ · pᵢ
        masked_patches = mask.unsqueeze(-1) * patches  # (B, N, C)
        
        # Residual refinement: pᵢ_refined = pᵢ + λ · p̃ᵢ
        refined = patches + self.lambda_scale * masked_patches
        
        # Apply layer norm for stability
        refined = self.patch_norm(refined)
        
        return refined
    
    def compute_similarity(
        self,
        query_patches: torch.Tensor,
        support_patches: torch.Tensor
    ) -> torch.Tensor:
        """Compute patch-to-patch similarity between query and support.
        
        Step 7: Compute cosine similarity matrix between all pairs of patches,
        then aggregate to get final similarity score.
        
        Following the paper diagram:
            - Compute NxN' similarity matrix between refined patches
            - Aggregate to scalar score (mean of max matching)
        
        Args:
            query_patches: (NQ, N, C) refined query patches
            support_patches: (Shot, N, C) refined support patches for one class
            
        Returns:
            similarity: (NQ,) similarity score per query
        """
        NQ, N, C = query_patches.shape
        Shot = support_patches.shape[0]
        
        # Normalize patches for cosine similarity
        query_norm = F.normalize(query_patches, dim=-1)  # (NQ, N, C)
        support_norm = F.normalize(support_patches, dim=-1)  # (Shot, N, C)
        
        # Average support patches across shots first
        support_avg = support_norm.mean(dim=0)  # (N, C) - prototype per patch position
        
        # Compute patch-to-patch similarity matrix
        # query: (NQ, N, C), support: (N, C)
        # sim_matrix: (NQ, N, N) - for each query, similarity between its patches and support patches
        sim_matrix = torch.einsum('qnc,mc->qnm', query_norm, support_avg)  # (NQ, N, N)
        
        # Aggregation strategy: Hungarian matching or soft matching
        # Using soft matching: for each query patch, take max over support patches
        max_sim_per_query_patch = sim_matrix.max(dim=-1)[0]  # (NQ, N)
        
        # Also get max over query patches for each support
        max_sim_per_support_patch = sim_matrix.max(dim=1)[0]  # (NQ, N)
        
        # Symmetric matching score: average of both directions
        query_to_support = max_sim_per_query_patch.mean(dim=-1)  # (NQ,)
        support_to_query = max_sim_per_support_patch.mean(dim=-1)  # (NQ,)
        
        # Final similarity: average of bidirectional matching
        sim = (query_to_support + support_to_query) / 2.0  # (NQ,)
        
        if self.use_mlp_similarity:
            sim = self.similarity_mlp(sim.unsqueeze(-1)).squeeze(-1)
        
        return sim
    
    def forward(
        self,
        query_patches: torch.Tensor,
        query_slots: torch.Tensor,
        query_attn: torch.Tensor,
        support_patches_list: List[torch.Tensor],
        support_slots_list: List[torch.Tensor],
        support_attn_list: List[torch.Tensor],
        support_weights_list: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """Full class-aware few-shot inference.
        
        Args:
            query_patches: (NQ, N, C) query patch tokens
            query_slots: (NQ, K, C) query slot descriptors
            query_attn: (NQ, K, N) query attention matrix
            support_patches_list: List of (Shot, N, C) support patches per class
            support_slots_list: List of (Shot, K, C) support slots per class
            support_attn_list: List of (Shot, K, N) support attention per class
            support_weights_list: Optional list of (Shot, K) slot weights per class
            
        Returns:
            scores: (NQ, M) similarity scores, M = number of classes
        """
        NQ = query_patches.shape[0]
        M = len(support_slots_list)  # Number of classes (Way)
        
        # Step 3: Compute class embeddings
        class_embeddings = self.compute_class_embeddings(
            support_slots_list, support_weights_list
        )  # (M, C)
        
        scores = []
        
        for m in range(M):
            class_emb = class_embeddings[m]  # (C,)
            
            # === Process Query ===
            # Step 4: Filter query slots by class similarity
            _, alpha_q = self.filter_slots_by_class(query_slots, class_emb)
            
            # Step 5: Compute query patch mask
            mask_q = self.compute_patch_mask(query_attn, alpha_q)
            
            # Step 6: Refine query patches
            refined_query = self.refine_patches(query_patches, mask_q)
            
            # === Process Support for class m ===
            support_patches_m = support_patches_list[m]  # (Shot, N, C)
            support_slots_m = support_slots_list[m]  # (Shot, K, C)
            support_attn_m = support_attn_list[m]  # (Shot, K, N)
            
            # Step 4: Filter support slots
            _, alpha_s = self.filter_slots_by_class(support_slots_m, class_emb)
            
            # Step 5: Compute support patch mask
            mask_s = self.compute_patch_mask(support_attn_m, alpha_s)
            
            # Step 6: Refine support patches
            refined_support = self.refine_patches(support_patches_m, mask_s)
            
            # Step 7: Compute similarity
            sim = self.compute_similarity(refined_query, refined_support)  # (NQ,)
            
            scores.append(sim)
        
        # Stack scores: (NQ, M)
        scores = torch.stack(scores, dim=1)
        
        # Scale by temperature for better gradients
        scores = scores / self.temperature
        
        return scores


class SlotAttentionWithPatches(nn.Module):
    """Wrapper around SlotAttention that also returns patch tokens.
    
    This module ensures we have access to both the slots and the original
    patch tokens needed for class-aware inference.
    
    Args:
        slot_attention: Existing SlotAttention module
        hidden_dim: Feature dimension
    """
    
    def __init__(self, slot_attention: nn.Module, hidden_dim: int):
        super().__init__()
        self.slot_attention = slot_attention
        self.norm_patches = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Args:
            features: (B, C, H, W) feature maps from dual branch fusion
            
        Returns:
            slots: (B, K, C) slot descriptors
            slot_weights: (B, K) slot existence weights
            attn: (B, K, N) attention matrix
            patches: (B, N, C) normalized patch tokens
        """
        B, C, H, W = features.shape
        N = H * W
        
        # Extract patch tokens (before slot attention internal norm)
        patches = rearrange(features, 'b c h w -> b (h w) c')  # (B, N, C)
        patches = self.norm_patches(patches)
        
        # Run slot attention with attention output
        slots, slot_weights, attn = self.slot_attention(features, return_attn=True)
        
        return slots, slot_weights, attn, patches
