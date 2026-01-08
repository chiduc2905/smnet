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
from net.backbone.covariance_similarity import CovarianceSimilarity


class ConvBlock(nn.Module):
    """Conv3×3 block with residual: Y = Proj(X) + GELU(BN(Conv(X))).
    
    Used to replace first PatchMerging to preserve spatial resolution.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
        # Residual projection if channels differ
        self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x) + self.act(self.bn(self.conv(x)))
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
        aggregation: str = 'mean',
        topk_ratio: float = 1.0,
        similarity_mode: str = 'covariance',  # NEW: covariance-based similarity
        temperature: float = 1.0,
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
        # STAGE 2: ConvBlocks (preserve spatial, increase channels)
        # (B, 32, 64, 64) → (B, 64, 64, 64)
        # ============================================================
        self.conv_blocks = nn.Sequential(
            ConvBlock(base_dim, base_dim * 2),      # 32 → 64
            ConvBlock(base_dim * 2, base_dim * 2),  # 64 → 64
        )
        
        # ============================================================
        # STAGE 3: Single PatchMerging (spatial /2, channels ×2)
        # (B, 64, 64, 64) → (B, 128, 32, 32)
        # ============================================================
        self.patch_merge = PatchMerging2D(dim=base_dim * 2, norm_layer=nn.LayerNorm)
        
        final_merge_dim = base_dim * 4  # 128
        
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
        self.similarity_mode = similarity_mode  # Store for forward()
        
        if similarity_mode == 'position':
            self.similarity = SimplePatchSimilarity(
                aggregation=aggregation,
                topk_ratio=topk_ratio,
                temperature=temperature
            )
        elif similarity_mode == 'allpairs':
            self.similarity = AllPairsSimilarity(temperature=temperature)
        elif similarity_mode == 'covariance':
            # Parameters will be set in forward based on feature size
            self.similarity = None
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
        # Stage 1: Patch embedding (B, 3, 64, 64) → (B, 32, 64, 64)
        f = self.patch_embed(x)
        
        # Stage 2: ConvBlocks (preserve spatial) (B, 32, 64, 64) → (B, 64, 64, 64)
        f = self.conv_blocks(f)
        
        # Stage 3: Single PatchMerging (B, 64, 64, 64) → (B, 128, 32, 32)
        f = self.patch_merge(f)
        
        # Stage 4: Channel projection (residual-friendly)
        # Linear proj + GroupNorm, NO activation (reduce nonlinearity)
        f_proj = self.channel_proj_conv(f)
        f_proj = self.channel_proj_norm(f_proj)
        # Skip connection if dims differ, just use proj
        f = f_proj
        
        # Stage 5: Dual branch fusion (ADD-based)
        f = self.dual_branch(f)
        
        # Stage 6: Unified attention (MUL-based)
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
            
            if self.similarity_mode == 'covariance':
                # Encode all support classes
                support_list = []
                for w in range(Way):
                    s_features = self.encode(support[b, w])  # (Shot, hidden, H', W')
                    support_list.append(s_features)
                
                # Initialize CovarianceSimilarity with correct feature size if needed
                _, _, Hf, Wf = q_features.shape
                if self.similarity is None:
                    self.similarity = CovarianceSimilarity(h=Hf, w=Wf, way_num=Way).to(q_features.device)
                
                # Get scores directly: (NQ, Way)
                scores_per_class = self.similarity(q_features, support_list)
            else:
                # Original per-class loop for other similarity modes
                scores_per_class = []
                for w in range(Way):
                    s_features = self.encode(support[b, w])
                    sim = self.similarity(q_features, s_features)  # (NQ,)
                    scores_per_class.append(sim)
                scores_per_class = torch.stack(scores_per_class, dim=1)  # (NQ, Way)
            
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
