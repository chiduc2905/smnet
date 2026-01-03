"""Covariance-based similarity metrics for few-shot learning.

Includes:
- CovaBlock: Original covariance metric (Li et al., AAAI 2019)
- SlotCovarianceBlock: Slot-based covariance for semantic slot representations
"""
import torch
import torch.nn as nn
from typing import List, Optional, Tuple


class CovaBlock(nn.Module):
    """Covariance-based similarity module (Vectorized).
    
    Computes similarity based on covariance matrices of support features,
    capturing distribution-level information rather than just point estimates.
    
    Reference: CovaMNet (Li et al., AAAI 2019)
    """
    
    def __init__(self):
        super(CovaBlock, self).__init__()

    def cal_covariance(self, support_features):
        """
        Args:
            support_features: (K, Shot, C, h, w) or (B, K, Shot, C, h, w)
            # Note: SMNet main.py currently passes (B, Way, Shot, C, H, W)
            # But we might want to process by reshaping.
            # Let's assume input is (Ns, C, h, w) where Ns is standard
            # Actually, let's match the MixMamba input format: (K, Shot, C, h, w)
        """
        # We need to handle the input shape carefully.
        # If input is list, stack it.
        if isinstance(support_features, list):
             support_features = torch.stack(support_features)
             
        if support_features.dim() == 6: # (B, K, Shot, C, H, W)
             # Reshape to (B*K, Shot, C, H, W) or handle per batch
             # For simplicity, let's assume we call this per episode or reshape outside.
             # But this module is usually called inside forward.
             pass
        
        # Implementation from MixMamba (optimized)
        # Expected input: (K, Shot, C, h, w)
        if support_features.dim() != 5:
            # Try to handle (Way*Shot, C, H, W) -> (Way, Shot, ...)
             raise ValueError(f"Expected 5D input (K, Shot, C, h, w), got {support_features.shape}")

        K, Shot, C, h, w = support_features.size()
        
        # Reshape to (K, C, N) where N = Shot*h*w
        support_flat = support_features.permute(0, 2, 1, 3, 4).contiguous().view(K, C, -1)
        
        # Mean centering
        mean_support = torch.mean(support_flat, dim=2, keepdim=True)
        support_centered = support_flat - mean_support
        
        # Compute covariance: (K, C, N) @ (K, N, C) -> (K, C, C)
        covariance_matrix = torch.bmm(support_centered, support_centered.transpose(1, 2))
        
        # Normalize
        covariance_matrix = torch.div(covariance_matrix, Shot * h * w - 1)
        return covariance_matrix

    def cal_similarity(self, query_features, covariance_matrices):
        """
        Args:
            query_features: (B, C, h, w)
            covariance_matrices: (K, C, C)
        Returns:
            similarity: (B, K*h*w) flattened map
        """
        B, C, h, w = query_features.size()
        K = covariance_matrices.size(0)
        
        # Reshape query: (B, C, L) where L = h*w
        query_flat = query_features.view(B, C, -1)
        
        # Normalize query (B, C, L)
        query_norm = torch.norm(query_flat, p=2, dim=1, keepdim=True) # (B, 1, L)
        query_normalized = query_flat / (query_norm + 1e-8)
        
        # Compute Mahalanobis-like distance: q.T @ M @ q
        # output (B, K, L)
        similarity_map = torch.einsum('bcl, kcd, bdl -> bkl', query_normalized, covariance_matrices, query_normalized)
        
        # Flatten to (B, K*h*w)
        similarity_flat = similarity_map.reshape(B, -1)
        
        return similarity_flat

    def forward(self, query_features, support_features):
        """
        Args:
            query_features: (B, C, h, w)
            support_features: (K, Shot, C, h, w)
        """
        covariance_matrices = self.cal_covariance(support_features)
        similarity = self.cal_similarity(query_features, covariance_matrices)
        return similarity


class SlotCovarianceBlock(nn.Module):
    """Slot-based covariance similarity for few-shot learning.
    
    Computes similarity using second-order statistics (covariance matrices)
    over slot representations for robust few-shot classification.
    
    Key Design Principles:
        1. Feature extraction shared between support and query
        2. No classifier head - pure metric-based inference
        3. Covariance captures distribution shape, not just mean
        4. Robust to few-shot regimes and domain shift
    
    Args:
        regularization: Ridge regularization for covariance inversion (default: 1e-3)
        use_mahalanobis: If True, use Mahalanobis distance; else bilinear (default: True)
        normalize_slots: Whether to L2-normalize slots before computation (default: True)
    """
    
    def __init__(
        self,
        regularization: float = 1e-3,
        use_mahalanobis: bool = True,
        normalize_slots: bool = True
    ):
        super().__init__()
        
        self.regularization = regularization
        self.use_mahalanobis = use_mahalanobis
        self.normalize_slots = normalize_slots
    
    def aggregate_class_slots(
        self,
        slots_list: List[torch.Tensor],
        weights_list: Optional[List[torch.Tensor]] = None
    ) -> List[torch.Tensor]:
        """Aggregate slots across all shots for each class.
        
        Args:
            slots_list: List of (n_shots, K, C) slot tensors, one per class
            weights_list: Optional list of (n_shots, K) slot weights
            
        Returns:
            List of (n_shots * K, C) aggregated slot matrices per class
        """
        aggregated = []
        
        for i, slots in enumerate(slots_list):
            n_shots, K, C = slots.shape
            
            # Reshape to (n_shots * K, C)
            class_slots = slots.reshape(-1, C)
            
            # Apply slot weights if provided
            if weights_list is not None and weights_list[i] is not None:
                weights = weights_list[i].reshape(-1)  # (n_shots * K,)
                # Weight each slot by its existence score
                class_slots = class_slots * weights.unsqueeze(-1)
            
            aggregated.append(class_slots)
        
        return aggregated
    
    def compute_slot_covariance(
        self,
        slots: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute covariance matrix from slot representations.
        
        Args:
            slots: (N, C) aggregated slots where N = n_shots * K
            weights: Optional (N,) slot weights for weighted covariance
            
        Returns:
            (C, C) regularized covariance matrix
        """
        N, C = slots.shape
        
        # L2 normalize if specified
        if self.normalize_slots:
            slots = torch.nn.functional.normalize(slots, p=2, dim=-1)
        
        # Compute weighted mean
        if weights is not None:
            weights_sum = weights.sum() + 1e-8
            mean = (slots * weights.unsqueeze(-1)).sum(dim=0) / weights_sum
            # Weighted centering
            centered = (slots - mean) * weights.unsqueeze(-1).sqrt()
            effective_n = weights_sum
        else:
            mean = slots.mean(dim=0)
            centered = slots - mean
            effective_n = N
        
        # Compute covariance: Σ = X^T X / (N - 1)
        covariance = centered.T @ centered / (effective_n - 1 + 1e-8)
        
        # Ridge regularization: Σ_reg = Σ + λI
        identity = torch.eye(C, device=slots.device, dtype=slots.dtype)
        covariance = covariance + self.regularization * identity
        
        return covariance
    
    def compute_class_covariances(
        self,
        support_slots: List[torch.Tensor],
        support_weights: Optional[List[torch.Tensor]] = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Compute covariance matrices for each class.
        
        Args:
            support_slots: List of (n_shots, K, C) support slot tensors per class
            support_weights: Optional list of (n_shots, K) slot weights
            
        Returns:
            covariances: List of (C, C) covariance matrices
            inv_covariances: List of (C, C) inverse covariance matrices
        """
        # Aggregate slots per class
        aggregated = self.aggregate_class_slots(support_slots, support_weights)
        
        covariances = []
        inv_covariances = []
        
        for class_slots in aggregated:
            # Compute covariance
            cov = self.compute_slot_covariance(class_slots)
            covariances.append(cov)
            
            # Compute inverse for Mahalanobis distance
            try:
                inv_cov = torch.linalg.inv(cov)
            except RuntimeError:
                # Fallback to pseudo-inverse if singular
                inv_cov = torch.linalg.pinv(cov)
            
            inv_covariances.append(inv_cov)
        
        return covariances, inv_covariances
    
    def compute_similarity(
        self,
        query_slots: torch.Tensor,
        class_inv_covariances: List[torch.Tensor],
        query_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute similarity between query slots and class covariances.
        
        Uses Mahalanobis-style scoring:
            score(q, c) = Σ_k w_k * s_k^T @ Σ_c^{-1} @ s_k
        
        Args:
            query_slots: (B, K, C) query slot descriptors
            class_inv_covariances: List of (C, C) inverse covariance matrices
            query_weights: Optional (B, K) query slot weights
            
        Returns:
            (B, num_classes) similarity scores
        """
        B, K, C = query_slots.shape
        num_classes = len(class_inv_covariances)
        
        # L2 normalize query slots
        if self.normalize_slots:
            query_slots = torch.nn.functional.normalize(query_slots, p=2, dim=-1)
        
        # Initialize similarity scores
        scores = torch.zeros(B, num_classes, device=query_slots.device, dtype=query_slots.dtype)
        
        for b in range(B):
            for c, inv_cov in enumerate(class_inv_covariances):
                # For each slot, compute s^T @ Σ^{-1} @ s
                slot_scores = torch.zeros(K, device=query_slots.device)
                
                for k in range(K):
                    s = query_slots[b, k]  # (C,)
                    
                    if self.use_mahalanobis:
                        # Mahalanobis: s^T @ Σ^{-1} @ s
                        slot_scores[k] = s @ inv_cov @ s
                    else:
                        # Bilinear: s^T @ Σ @ s (without inversion)
                        slot_scores[k] = s @ class_inv_covariances[c] @ s
                
                # Aggregate slot scores (weighted if provided)
                if query_weights is not None:
                    scores[b, c] = (slot_scores * query_weights[b]).sum()
                else:
                    scores[b, c] = slot_scores.sum()
        
        return scores
    
    def forward(
        self,
        query_slots: torch.Tensor,
        support_slots: List[torch.Tensor],
        query_weights: Optional[torch.Tensor] = None,
        support_weights: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """Compute slot covariance-based similarity.
        
        Args:
            query_slots: (B, K, C) query slot descriptors
            support_slots: List of (n_shots, K, C) support slots per class
            query_weights: Optional (B, K) query slot existence weights
            support_weights: Optional list of (n_shots, K) support slot weights
            
        Returns:
            (B, num_classes) similarity scores for classification
            
        Note:
            Mahalanobis distance is NEGATED to convert to similarity:
            - Lower distance → Higher similarity → Higher score
            - This is required for CrossEntropy loss compatibility
        """
        # Compute class-level covariance matrices
        _, inv_covariances = self.compute_class_covariances(
            support_slots, support_weights
        )
        
        # Compute distance scores
        distances = self.compute_similarity(
            query_slots, inv_covariances, query_weights
        )
        
        # CRITICAL: Negate distances to convert to similarities
        # Lower distance = more similar = higher score
        scores = -distances
        
        return scores
    
    def predict(
        self,
        query_slots: torch.Tensor,
        support_slots: List[torch.Tensor],
        query_weights: Optional[torch.Tensor] = None,
        support_weights: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """Predict class labels for queries.
        
        Args:
            query_slots: (B, K, C) query slot descriptors
            support_slots: List of (n_shots, K, C) support slots per class
            query_weights: Optional (B, K) query slot weights
            support_weights: Optional list of (n_shots, K) support slot weights
            
        Returns:
            (B,) predicted class indices
        """
        scores = self.forward(
            query_slots, support_slots, query_weights, support_weights
        )
        return scores.argmax(dim=-1)
