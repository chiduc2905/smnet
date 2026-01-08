"""Class-Conditional Cosine Decision Module for Few-Shot Learning.

Implements lightweight class-conditional feature reweighting to enhance
class discrimination without modifying the backbone or base similarity metric.

The key insight is that different classes may rely on different feature
channels for discrimination. By learning a class-specific channel-wise
gating function, we can emphasize the most discriminative channels for
each class, improving separation between visually similar classes.

Architecture:
    - Gating function g(·): Linear(C → C) + Sigmoid
    - Reweighted query: q'_c = q ⊙ a_c (element-wise multiplication)
    - Re-normalized for cosine similarity

Usage:
    module = ClassConditionalCosine(feature_dim=128)
    scores = module(query, prototypes)

Training:
    - Use standard CrossEntropyLoss
    - Only g(·) parameters are learnable
    - Backbone remains frozen
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ClassConditionalCosine(nn.Module):
    """Class-Conditional Cosine Similarity Module.
    
    Enhances class discrimination by learning class-specific channel weights
    that emphasize the most discriminative features for each class.
    
    HOW THIS INCREASES CLASS-SPECIFIC DISCRIMINATION:
    
    1. CHANNEL SELECTION: Each class prototype is passed through a gating
       function that outputs channel-wise weights in [0, 1]. High weights
       indicate channels that are important for distinguishing this class.
       
    2. ADAPTIVE EMPHASIS: When comparing a query to class c, the query
       features are reweighted using class c's learned channel weights.
       This emphasizes features that are discriminative for class c.
       
    3. CROSS-CLASS CONTRAST: Since each class has different weights,
       the same query gets different emphasis when compared to different
       classes, maximizing the contrast between the true class and others.
    
    Example intuition:
        - Class A (corona PD): emphasizes high-frequency texture channels
        - Class B (void PD): emphasizes shape/contour channels
        - Query from Class A: when compared to A, texture channels boosted,
          increasing similarity; when compared to B, shape channels boosted,
          likely decreasing similarity → better discrimination
    
    Args:
        feature_dim: Dimension of feature vectors (C)
        temperature: Scaling factor for cosine similarity (default: 1.0)
        bias: Whether to use bias in the gating function (default: True)
    """
    
    def __init__(
        self,
        feature_dim: int,
        temperature: float = 1.0,
        bias: bool = True
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.temperature = temperature
        
        # === Gating Function g(·) ===
        # Minimal parameter count: just C*C + C (if bias) = C*(C+1)
        # For C=128: 128*129 = 16,512 parameters
        #
        # Architecture choice:
        # - Linear: learns channel correlations from prototype
        # - Sigmoid: outputs weights in [0, 1], interpretable as importance
        #
        # Why not deeper? 
        # - Minimal parameters is a constraint
        # - Single layer is sufficient for channel reweighting
        # - Avoids overfitting in few-shot setting
        self.gate = nn.Sequential(
            nn.Linear(feature_dim, feature_dim, bias=bias),
            nn.Sigmoid()
        )
        
        # Initialize gate to output near-uniform weights initially
        # This ensures the module doesn't disrupt pretrained representations
        self._init_weights()
    
    def _init_weights(self):
        """Initialize gate to output ~0.5 (uniform) initially."""
        # Small weight initialization means sigmoid outputs ~0.5
        nn.init.xavier_uniform_(self.gate[0].weight, gain=0.1)
        if self.gate[0].bias is not None:
            # Bias of 0 + small weights → sigmoid(~0) ≈ 0.5
            nn.init.zeros_(self.gate[0].bias)
    
    def compute_gate(self, prototype: torch.Tensor) -> torch.Tensor:
        """Compute channel-wise gating weights from class prototype.
        
        Args:
            prototype: (C,) or (Way, C) L2-normalized class prototype(s)
            
        Returns:
            gate_weights: Same shape as input, values in [0, 1]
        """
        return self.gate(prototype)
    
    def forward(
        self,
        query: torch.Tensor,
        prototypes: torch.Tensor,
        return_gates: bool = False
    ) -> torch.Tensor:
        """Compute class-conditional cosine similarity scores.
        
        For each class c:
            1. Compute gating weights: a_c = g(p_c) ∈ [0,1]^C
            2. Reweight query: q'_c = q ⊙ a_c
            3. Re-normalize: q'_c = q'_c / ||q'_c||
            4. Compute score: score_c = cosine(q'_c, p_c)
        
        Args:
            query: (NQ, C) L2-normalized query features
            prototypes: (Way, C) L2-normalized class prototypes
            return_gates: If True, also return gating weights for visualization
            
        Returns:
            scores: (NQ, Way) similarity scores (higher = more similar)
            Optional[gates]: (Way, C) gating weights if return_gates=True
        """
        NQ, C = query.shape
        Way, _ = prototypes.shape
        
        # === Step 1: Compute gating weights for each class ===
        # gate_weights: (Way, C), each row is channel importance for that class
        gate_weights = self.gate(prototypes)  # (Way, C)
        
        # === Step 2-4: Compute scores for all query-class pairs ===
        scores = []
        
        for w in range(Way):
            # Get class-specific gating weights
            a_c = gate_weights[w]  # (C,)
            
            # Reweight query features with class-specific emphasis
            # Channels with high a_c get emphasized, low a_c get suppressed
            q_reweighted = query * a_c.unsqueeze(0)  # (NQ, C)
            
            # Re-normalize to ensure fair cosine comparison
            # Without this, different gating patterns would have different norms
            q_reweighted = F.normalize(q_reweighted, p=2, dim=-1)  # (NQ, C)
            
            # Get class prototype (already L2-normalized)
            p_c = prototypes[w]  # (C,)
            
            # Cosine similarity: dot product of normalized vectors
            # (NQ, C) @ (C,) -> (NQ,)
            sim = torch.mv(q_reweighted, p_c)  # (NQ,)
            
            # Logit scaling: logit = cosine * τ (τ ∈ [10, 30] recommended)
            sim = sim * self.temperature
            
            scores.append(sim)
        
        # Stack scores: (Way, NQ) -> (NQ, Way)
        scores = torch.stack(scores, dim=1)  # (NQ, Way)
        
        if return_gates:
            return scores, gate_weights
        return scores
    
    def forward_efficient(
        self,
        query: torch.Tensor,
        prototypes: torch.Tensor
    ) -> torch.Tensor:
        """Efficient batched forward (alternative implementation).
        
        This version uses broadcasting for efficiency but may use more memory.
        Use for small Way (number of classes).
        
        Args:
            query: (NQ, C) L2-normalized query features
            prototypes: (Way, C) L2-normalized class prototypes
            
        Returns:
            scores: (NQ, Way) similarity scores
        """
        NQ, C = query.shape
        Way, _ = prototypes.shape
        
        # Compute all gating weights at once
        gate_weights = self.gate(prototypes)  # (Way, C)
        
        # Expand query for broadcasting: (NQ, 1, C)
        query_exp = query.unsqueeze(1)  # (NQ, 1, C)
        
        # Expand gates for broadcasting: (1, Way, C)
        gates_exp = gate_weights.unsqueeze(0)  # (1, Way, C)
        
        # Reweight query for each class: (NQ, Way, C)
        q_reweighted = query_exp * gates_exp  # (NQ, Way, C)
        
        # Re-normalize each reweighted query
        q_reweighted = F.normalize(q_reweighted, p=2, dim=-1)  # (NQ, Way, C)
        
        # Expand prototypes: (1, Way, C)
        prototypes_exp = prototypes.unsqueeze(0)  # (1, Way, C)
        
        # Dot product for cosine similarity
        # (NQ, Way, C) * (1, Way, C) -> sum over C -> (NQ, Way)
        scores = (q_reweighted * prototypes_exp).sum(dim=-1)  # (NQ, Way)
        
        # Logit scaling: logit = cosine * τ
        scores = scores * self.temperature
        
        return scores


class ClassConditionalCosineWithPatches(nn.Module):
    """Extended version for patch-based few-shot learning.
    
    This version handles (NQ, N, C) patch embeddings by first pooling
    to (NQ, C) before applying class-conditional cosine similarity.
    
    Args:
        feature_dim: Dimension of feature vectors (C)
        temperature: Scaling factor for cosine similarity
        pooling: 'mean' or 'max' for spatial pooling strategy
    """
    
    def __init__(
        self,
        feature_dim: int,
        temperature: float = 1.0,
        pooling: str = 'mean'
    ):
        super().__init__()
        
        self.cc_cosine = ClassConditionalCosine(
            feature_dim=feature_dim,
            temperature=temperature
        )
        self.pooling = pooling
    
    def pool_features(self, x: torch.Tensor) -> torch.Tensor:
        """Pool patch features to single vector.
        
        Args:
            x: (B, N, C) patch embeddings
            
        Returns:
            pooled: (B, C) pooled features
        """
        if self.pooling == 'mean':
            return x.mean(dim=1)
        elif self.pooling == 'max':
            return x.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
    
    def forward(
        self,
        query_patches: torch.Tensor,
        support_patches_list: list
    ) -> torch.Tensor:
        """Compute scores from patch embeddings.
        
        Args:
            query_patches: (NQ, N, C) query patch embeddings
            support_patches_list: List of (Shot, N, C) support patches per class
            
        Returns:
            scores: (NQ, Way) similarity scores
        """
        # Pool query patches
        query = self.pool_features(query_patches)  # (NQ, C)
        query = F.normalize(query, p=2, dim=-1)
        
        # Compute prototypes from support
        prototypes = []
        for support_patches in support_patches_list:
            # (Shot, N, C) -> (Shot, C) -> (C,)
            support = self.pool_features(support_patches)  # (Shot, C)
            proto = support.mean(dim=0)  # (C,) mean over shots
            proto = F.normalize(proto, p=2, dim=-1)
            prototypes.append(proto)
        
        prototypes = torch.stack(prototypes, dim=0)  # (Way, C)
        
        # Apply class-conditional cosine
        return self.cc_cosine(query, prototypes)
