"""Prototype Refinement Module for Few-Shot Learning.

Implements outlier-robust prototype computation by discarding support samples
that are far from the initial prototype, reducing variance in the prototype
and stabilizing decision boundaries between visually similar classes.

Usage:
    refined_proto = refine_prototype(support_feats)

Key Benefits:
    - Reduces prototype variance by removing noisy/outlier samples
    - Stabilizes decision boundaries between similar classes
    - Deterministic per episode (no learnable parameters)
    - Maintains L2 normalization for consistent cosine similarity

Reference:
    This implements OPTION A (outlier removal) from the prototype refinement
    specification, chosen for its simplicity and stability.
"""
import torch
import torch.nn.functional as F


def refine_prototype(
    support_feats: torch.Tensor,
    outlier_fraction: float = 0.2
) -> torch.Tensor:
    """Refine class prototype by removing outlier support samples.
    
    This function computes an outlier-robust prototype by:
    1. Computing initial prototype as mean of all support features
    2. Measuring cosine distance from each sample to the prototype
    3. Discarding the top 20% farthest (most outlier) samples
    4. Recomputing prototype from remaining samples
    5. L2-normalizing the final prototype
    
    WHY THIS IMPROVES ROBUSTNESS:
    - Outlier samples (e.g., noisy images, atypical views) pull the prototype
      away from the true class center, causing decision boundary shifts
    - By removing distant samples, the prototype better represents the
      "core" samples of the class, leading to more stable classification
    - Especially important for visually similar classes where boundaries
      are tight and small prototype shifts cause misclassification
    
    Args:
        support_feats: (K, C) L2-normalized support features for one class
                       K = number of support samples (Shot)
                       C = feature dimension
        outlier_fraction: Fraction of farthest samples to discard (default: 0.2)
                         Set to 0.0 to disable outlier removal
                       
    Returns:
        refined_proto: (C,) L2-normalized refined prototype
        
    Example:
        >>> support = torch.randn(5, 128)  # 5-shot, 128-dim features
        >>> support = F.normalize(support, p=2, dim=-1)
        >>> proto = refine_prototype(support)
        >>> print(proto.shape)  # torch.Size([128])
    """
    K, C = support_feats.shape
    
    # Edge case: single sample, return it directly
    if K == 1:
        return F.normalize(support_feats.squeeze(0), p=2, dim=-1)
    
    # Step 1: Compute initial prototype (simple mean)
    # This gives us a reference point to measure distances from
    initial_proto = support_feats.mean(dim=0)  # (C,)
    initial_proto = F.normalize(initial_proto, p=2, dim=-1)  # (C,)
    
    # Step 2: Compute cosine distance from each sample to initial prototype
    # cosine_distance = 1 - cosine_similarity
    # Higher distance = sample is farther from the prototype = potential outlier
    # 
    # Using cosine distance instead of Euclidean because:
    # - Features are L2-normalized, so cosine distance is more meaningful
    # - Cosine similarity is what we use for final classification
    similarities = torch.mv(support_feats, initial_proto)  # (K,)
    distances = 1.0 - similarities  # (K,), range [0, 2]
    
    # Step 3: Determine how many samples to discard
    num_to_discard = int(K * outlier_fraction)
    num_to_keep = max(1, K - num_to_discard)  # Keep at least 1 sample
    
    # Edge case: if we're keeping all samples, just return normalized mean
    if num_to_keep >= K:
        refined_proto = support_feats.mean(dim=0)
        return F.normalize(refined_proto, p=2, dim=-1)
    
    # Step 4: Find indices of samples to KEEP (lowest distances)
    # We use topk with largest=False to get the closest samples
    _, keep_indices = torch.topk(distances, k=num_to_keep, largest=False)
    
    # Step 5: Recompute prototype from kept samples only
    # These samples are closer to the initial prototype, representing
    # the "core" of the class distribution
    kept_feats = support_feats[keep_indices]  # (num_to_keep, C)
    refined_proto = kept_feats.mean(dim=0)  # (C,)
    
    # Step 6: Final L2 normalization
    # Ensures prototype remains in the same normalized feature space
    # as the original features for consistent cosine similarity
    refined_proto = F.normalize(refined_proto, p=2, dim=-1)
    
    return refined_proto


def refine_prototype_weighted(
    support_feats: torch.Tensor,
    tau: float = 0.5
) -> torch.Tensor:
    """Alternative: Refine class prototype using weighted averaging (OPTION B).
    
    Instead of hard outlier removal, this uses soft weighting where
    samples closer to the initial prototype contribute more to the final prototype.
    
    Weight formula: w_i = exp(-d_i / τ) where d_i is cosine distance
    
    WHY THIS IMPROVES ROBUSTNESS:
    - Soft weighting is more stable than hard cutoffs
    - Samples far from prototype get low weights but still contribute
    - Temperature τ controls the sharpness of weighting
      - Low τ (e.g., 0.1): very sharp, almost like hard cutoff
      - High τ (e.g., 1.0): gentle weighting, almost like mean
    
    Args:
        support_feats: (K, C) L2-normalized support features for one class
        tau: Temperature parameter (default: 0.5)
             Lower = more aggressive downweighting of outliers
             Higher = more uniform weighting
             
    Returns:
        refined_proto: (C,) L2-normalized refined prototype
    """
    K, C = support_feats.shape
    
    # Edge case: single sample
    if K == 1:
        return F.normalize(support_feats.squeeze(0), p=2, dim=-1)
    
    # Step 1: Initial prototype
    initial_proto = support_feats.mean(dim=0)  # (C,)
    initial_proto = F.normalize(initial_proto, p=2, dim=-1)
    
    # Step 2: Compute cosine distances
    similarities = torch.mv(support_feats, initial_proto)  # (K,)
    distances = 1.0 - similarities  # (K,)
    
    # Step 3: Compute weights via exponential of negative distance
    # exp(-d/τ): higher for closer samples, lower for farther
    weights = torch.exp(-distances / tau)  # (K,)
    
    # Step 4: Normalize weights to sum to 1
    weights = weights / weights.sum()  # (K,)
    
    # Step 5: Weighted average
    # refined_proto = Σ w_i * s_i
    refined_proto = torch.einsum('k,kc->c', weights, support_feats)  # (C,)
    
    # Step 6: L2 normalize
    refined_proto = F.normalize(refined_proto, p=2, dim=-1)
    
    return refined_proto


def compute_class_prototypes(
    support_patches_list: list,
    use_weighted: bool = False,
    outlier_fraction: float = 0.2,
    tau: float = 0.5
) -> torch.Tensor:
    """Compute refined prototypes for all classes.
    
    A convenience function for few-shot learning that processes
    support features for multiple classes and returns stacked prototypes.
    
    Args:
        support_patches_list: List of (Shot, N, C) or (Shot, C) support features
                             One tensor per class
        use_weighted: If True, use weighted averaging (OPTION B)
                     If False, use outlier removal (OPTION A, recommended)
        outlier_fraction: For OPTION A, fraction to discard
        tau: For OPTION B, temperature parameter
        
    Returns:
        prototypes: (Way, C) L2-normalized prototypes for all classes
    """
    prototypes = []
    
    for support_feats in support_patches_list:
        # If support has patch dimension, average over patches first
        if support_feats.dim() == 3:
            # (Shot, N, C) -> (Shot, C) via global average pooling
            support_feats = support_feats.mean(dim=1)
        
        # L2 normalize before refinement
        support_feats = F.normalize(support_feats, p=2, dim=-1)
        
        # Apply refinement
        if use_weighted:
            proto = refine_prototype_weighted(support_feats, tau=tau)
        else:
            proto = refine_prototype(support_feats, outlier_fraction=outlier_fraction)
        
        prototypes.append(proto)
    
    # Stack: (Way, C)
    return torch.stack(prototypes, dim=0)
