"""Covariance-based similarity metrics for few-shot learning.

Includes:
- CovaBlock: Original covariance metric (Li et al., AAAI 2019)
- SlotCovarianceBlock: Slot-based covariance for semantic slot representations
"""
import torch
import torch.nn as nn
from typing import List, Optional, Tuple


class CovaBlock(nn.Module):
    """Covariance-based similarity module.
    
    Computes similarity based on covariance matrices of support features,
    capturing distribution-level information rather than just point estimates.
    
    Reference: CovaMNet (Li et al., AAAI 2019)
    "Distribution Consistency Based Covariance Metric Networks for Few-Shot Learning"
    """
    
    def __init__(self):
        super(CovaBlock, self).__init__()

    def cal_covariance(self, input_features: list) -> list:
        """Calculate covariance matrices for each class.
        
        Args:
            input_features: List of (B, C, h, w) feature tensors, one per class
            
        Returns:
            List of (C, C) covariance matrices
        """
        CovaMatrix_list = []
        for i in range(len(input_features)):
            support_set_sam = input_features[i]
            B, C, h, w = support_set_sam.size()

            support_set_sam = support_set_sam.permute(1, 0, 2, 3)
            support_set_sam = support_set_sam.contiguous().view(C, -1)
            mean_support = torch.mean(support_set_sam, 1, True)
            support_set_sam = support_set_sam - mean_support

            covariance_matrix = support_set_sam @ torch.transpose(support_set_sam, 0, 1)
            covariance_matrix = torch.div(covariance_matrix, h * w * B - 1)
            CovaMatrix_list.append(covariance_matrix)
        return CovaMatrix_list

    def cal_similarity(self, query_features: torch.Tensor, CovaMatrix_list: list) -> torch.Tensor:
        """Calculate similarity between query and class covariance matrices.
        
        Args:
            query_features: (B, C, h, w) query feature maps
            CovaMatrix_list: List of (C, C) covariance matrices
            
        Returns:
            (B, num_classes * h * w) similarity scores
        """
        B, C, h, w = query_features.size()
        Cova_Sim = []

        for i in range(B):
            query_sam = query_features[i]
            query_sam = query_sam.view(C, -1)
            query_sam_norm = torch.norm(query_sam, 2, 1, True)
            query_sam = query_sam / query_sam_norm

            mea_sim = torch.zeros(1, len(CovaMatrix_list) * h * w).to(query_sam.device)

            for j in range(len(CovaMatrix_list)):
                temp_dis = torch.transpose(query_sam, 0, 1) @ CovaMatrix_list[j] @ query_sam
                mea_sim[0, j * h * w:(j + 1) * h * w] = temp_dis.diag()

            Cova_Sim.append(mea_sim.view(1, -1))

        Cova_Sim = torch.cat(Cova_Sim, 0)
        return Cova_Sim

    def forward(self, query_features: torch.Tensor, support_features: list) -> torch.Tensor:
        """Compute covariance-based similarity.
        
        Args:
            query_features: (B, C, h, w) query feature maps
            support_features: List of (B, C, h, w) support features per class
            
        Returns:
            (B, num_classes * h * w) similarity scores
        """
        CovaMatrix_list = self.cal_covariance(support_features)
        Cova_Sim = self.cal_similarity(query_features, CovaMatrix_list)
        return Cova_Sim


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
        """
        # Compute class-level covariance matrices
        _, inv_covariances = self.compute_class_covariances(
            support_slots, support_weights
        )
        
        # Compute similarity scores
        scores = self.compute_similarity(
            query_slots, inv_covariances, query_weights
        )
        
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
