"""USCMambaNet: Unified Spatial-Channel Mamba Network.

Implements a clean few-shot learning architecture following ADD-vs-MULTIPLY design:
- ADD: Feature extraction (encoder, dual-branch fusion, channel mixing)
- MUL: Feature selection (unified attention ONLY)
- Similarity: Non-learnable position-wise cosine + mean/topk

This model removes all slot-based mechanisms for simpler, more interpretable matching.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange

from net.backbone.feature_extractor import PatchEmbed2D, PatchMerging2D
from net.backbone.dual_branch_fusion import DualBranchFusion
from net.backbone.unified_attention import UnifiedSpatialChannelAttention
from net.backbone.simple_similarity import SimplePatchSimilarity, AllPairsSimilarity


class USCMambaNet(nn.Module):
    """Unified Spatial-Channel Mamba Network (U-SCMambaNet).
    
    Uses ADD-vs-MULTIPLY design principles:
    - ADD: Feature extraction (encoder, fusion, channel mixing)
    - MUL: Feature selection (unified attention after fusion ONLY)
    - Similarity: Non-learnable position-wise cosine + mean/topk
    
    Architecture:
        Input → Encoder → DualBranchFusion → UnifiedAttention → SimpleSimilarity
        
        Encoder:
            PatchEmbed2D → PatchMerging2D (×N) → ChannelProjection
        
        Feature Extraction (ADD-based):
            DualBranchFusion: Local (AG-LKA + ChannelMamba) + Global (SS2D)
        
        Feature Selection (MUL-based):
            UnifiedSpatialChannelAttention: ECA++ (channel) + DWConv (spatial)
        
        Similarity (Non-learnable):
            SimplePatchSimilarity: Position-wise cosine + mean/topk aggregation
    
    Args:
        in_channels: Input channels (default: 3)
        base_dim: Base embedding dim (default: 32)
        hidden_dim: Hidden dim (default: 64)
        num_merging_stages: Patch merging stages (default: 2)
        d_state: Mamba state dimension (default: 4)
        aggregation: 'mean' or 'topk' for similarity (default: 'topk')
        topk_ratio: Ratio for top-k aggregation (default: 0.2)
        similarity_mode: 'position' or 'allpairs' (default: 'position')
        temperature: Temperature for similarity scaling (default: 0.2)
        device: Device to use
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        base_dim: int = 32,
        hidden_dim: int = 64,
        num_merging_stages: int = 2,
        d_state: int = 4,
        aggregation: str = 'topk',
        topk_ratio: float = 0.2,
        similarity_mode: str = 'position',
        temperature: float = 0.2,
        device: str = 'cuda'
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.device = device
        
        # ============================================================
        # STAGE 1: Patch Embedding
        # ============================================================
        self.patch_embed = PatchEmbed2D(
            in_channels=in_channels,
            embed_dim=base_dim,
            kernel_size=3,
            norm_layer=nn.LayerNorm
        )
        
        # ============================================================
        # STAGE 2: Hierarchical Patch Merging
        # ============================================================
        self.merging_stages = nn.ModuleList()
        current_dim = base_dim
        
        for i in range(num_merging_stages):
            self.merging_stages.append(
                PatchMerging2D(dim=current_dim, norm_layer=nn.LayerNorm)
            )
            current_dim = current_dim * 2
        
        final_merge_dim = current_dim
        
        # ============================================================
        # STAGE 3: Channel Projection (lightweight, with residual)
        # ============================================================
        # Use GroupNorm instead of BatchNorm (more stable for small batch)
        # Linear projection only, no nonlinearity (reduce "heat")
        self.channel_proj_conv = nn.Conv2d(final_merge_dim, hidden_dim, kernel_size=1, bias=False)
        self.channel_proj_norm = nn.GroupNorm(num_groups=8, num_channels=hidden_dim)
        
        # ============================================================
        # STAGE 4: Feature Extraction (ADD-based)
        # ============================================================
        self.dual_branch = DualBranchFusion(
            channels=hidden_dim,
            d_state=d_state,
            dilation=2
        )
        
        # ============================================================
        # STAGE 5: Feature Selection (MUL-based) - ONLY MUL HERE
        # ============================================================
        self.unified_attention = UnifiedSpatialChannelAttention(hidden_dim)
        
        # ============================================================
        # STAGE 6: Similarity (NON-learnable)
        # ============================================================
        if similarity_mode == 'position':
            self.similarity = SimplePatchSimilarity(
                aggregation=aggregation,
                topk_ratio=topk_ratio,
                temperature=temperature
            )
        elif similarity_mode == 'allpairs':
            self.similarity = AllPairsSimilarity(temperature=temperature)
        else:
            raise ValueError(f"Unknown similarity_mode: {similarity_mode}")
        
        self.to(device)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to feature maps.
        
        Args:
            x: (B, C, H, W) input images
            
        Returns:
            features: (B, hidden_dim, H', W') encoded features
        """
        # Stage 1: Patch embedding
        f = self.patch_embed(x)
        
        # Stage 2: Hierarchical merging
        for merge in self.merging_stages:
            f = merge(f)
        
        # Stage 3: Channel projection (residual-friendly)
        # Linear proj + GroupNorm, NO activation (reduce nonlinearity)
        f_proj = self.channel_proj_conv(f)
        f_proj = self.channel_proj_norm(f_proj)
        # Skip connection if dims differ, just use proj
        f = f_proj
        
        # Stage 4: Dual branch fusion (ADD-based)
        f = self.dual_branch(f)
        
        # Stage 5: Unified attention (MUL-based)
        f = self.unified_attention(f)
        
        return f
    
    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor
    ) -> torch.Tensor:
        """Few-shot classification.
        
        Args:
            query: (B, NQ, C, H, W) query images
            support: (B, Way, Shot, C, H, W) support images
            
        Returns:
            scores: (B*NQ, Way) similarity scores
        """
        B, NQ, C, H, W = query.shape
        Way = support.shape[1]
        Shot = support.shape[2]
        
        all_scores = []
        
        for b in range(B):
            # Encode queries: (NQ, C, H, W) -> (NQ, hidden, H', W')
            q_features = self.encode(query[b])
            
            scores_per_class = []
            for w in range(Way):
                # Encode support for class w: (Shot, C, H, W) -> (Shot, hidden, H', W')
                s_features = self.encode(support[b, w])
                
                # Compute similarity (non-learnable)
                sim = self.similarity(q_features, s_features)  # (NQ,)
                scores_per_class.append(sim)
            
            # Stack: (NQ, Way)
            scores_per_class = torch.stack(scores_per_class, dim=1)
            all_scores.append(scores_per_class)
        
        return torch.cat(all_scores, dim=0)  # (B*NQ, Way)
    
    def get_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features for visualization.
        
        Args:
            images: (B, C, H, W) input images
            
        Returns:
            features: (B, hidden_dim, H', W') feature maps
        """
        return self.encode(images)


def build_usc_mamba_net(
    aggregation: str = 'topk',
    **kwargs
) -> USCMambaNet:
    """Factory function for USCMambaNet.
    
    Args:
        aggregation: 'mean' or 'topk' (default: 'topk')
        **kwargs: Additional arguments
        
    Returns:
        Configured USCMambaNet
    """
    return USCMambaNet(aggregation=aggregation, **kwargs)


# Aliases for backward compatibility
SlotFreeSMNet = USCMambaNet
build_slot_free_smnet = build_usc_mamba_net
