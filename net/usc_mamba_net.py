"""USCMambaNet: Unified Spatial-Channel Mamba Network.

Implements a clean few-shot learning architecture following ADD-vs-MULTIPLY design:
- ADD: Feature extraction (encoder, dual-branch fusion, channel mixing)
- MUL: Feature selection (unified attention ONLY)
- Similarity: Refined prototype + class-conditional cosine

Pipeline after Stage 6 (UnifiedAttention):
    Support Features → refine_prototype (non-learnable, remove outliers)
                     → ClassConditionalCosine (learnable gating)
                     → Final Scores
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange

from net.backbone.feature_extractor import PatchEmbed2D, PatchMerging2D
from net.backbone.dual_branch_fusion import DualBranchFusion
from net.backbone.unified_attention import UnifiedSpatialChannelAttention
from net.metrics.prototype_refinement import refine_prototype
from net.metrics.class_conditional_cosine import ClassConditionalCosine


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
    - Prototype: refine_prototype (non-learnable, outlier removal)
    - Similarity: ClassConditionalCosine (learnable channel gating)
    
    Architecture:
        Input → Encoder → DualBranchFusion → UnifiedAttention
              → refine_prototype → ClassConditionalCosine → Scores
        
        Encoder:
            PatchEmbed2D → ConvBlocks → PatchMerging2D → ChannelProjection
        
        Feature Extraction (ADD-based):
            DualBranchFusion: Local (AG-LKA) + Global (VSS/Mamba)
        
        Feature Selection (MUL-based):
            UnifiedSpatialChannelAttention: ECA++ (channel) + DWConv (spatial)
        
        Prototype Refinement (Non-learnable):
            refine_prototype: Remove top 20% outlier samples
        
        Similarity (Learnable, lightweight):
            ClassConditionalCosine: g(proto) = Linear + Sigmoid → channel weights
    
    Args:
        in_channels: Input channels (default: 3)
        base_dim: Base embedding dim (default: 32)
        hidden_dim: Hidden dim (default: 64)
        num_merging_stages: Patch merging stages (default: 2)
        d_state: Mamba state dimension (default: 4)
        temperature: Temperature for cosine similarity (default: 1.0)
        outlier_fraction: Fraction of outliers to remove (default: 0.2)
        device: Device to use
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        base_dim: int = 32,
        hidden_dim: int = 64,
        num_merging_stages: int = 2,
        d_state: int = 4,
        temperature: float = 1.0,
        outlier_fraction: float = 0.2,
        device: str = 'cuda',
        **kwargs  # For backward compatibility
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.device = device
        self.temperature = temperature
        self.outlier_fraction = outlier_fraction
        
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
        # STAGE 4: Channel Projection (lightweight)
        # ============================================================
        self.channel_proj_conv = nn.Conv2d(final_merge_dim, hidden_dim, kernel_size=1, bias=False)
        self.channel_proj_norm = nn.GroupNorm(num_groups=8, num_channels=hidden_dim)
        
        # ============================================================
        # STAGE 5: Feature Extraction (ADD-based)
        # ============================================================
        self.dual_branch = DualBranchFusion(
            channels=hidden_dim,
            d_state=d_state,
            dilation=2
        )
        
        # ============================================================
        # STAGE 6: Feature Selection (MUL-based)
        # ============================================================
        self.unified_attention = UnifiedSpatialChannelAttention(hidden_dim)
        
        # ============================================================
        # STAGE 7: Class-Conditional Cosine (LEARNABLE, lightweight)
        # Only Linear(C→C) + Sigmoid = ~4K params for hidden_dim=64
        # ============================================================
        self.cc_cosine = ClassConditionalCosine(
            feature_dim=hidden_dim,
            temperature=temperature
        )
        
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
        """Few-shot classification with shot-dependent prototype handling.
        
        Pipeline:
            1. Encode query and support images → GAP → L2 normalize
            2. Prototype Construction (shot-dependent):
               - 1-shot: Use single support embedding directly (no refinement)
               - 5-shot: Apply refine_prototype to remove outliers
            3. ClassConditionalCosine: Compute class-conditional scores (learnable)
        
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
            # ============================================================
            # Step 1: Encode query images → GAP → L2 normalize
            # ============================================================
            q_features = self.encode(query[b])  # (NQ, hidden, H', W')
            q_vectors = q_features.mean(dim=[2, 3])  # GAP: (NQ, hidden)
            q_vectors = F.normalize(q_vectors, p=2, dim=-1)  # L2 normalize
            
            # ============================================================
            # Step 2: Encode support and compute prototypes (shot-dependent)
            # ============================================================
            prototypes = []
            for w in range(Way):
                # Encode support for class w
                s_features = self.encode(support[b, w])  # (Shot, hidden, H', W')
                s_vectors = s_features.mean(dim=[2, 3])  # GAP: (Shot, hidden)
                s_vectors = F.normalize(s_vectors, p=2, dim=-1)  # L2 normalize
                
                # === SHOT-DEPENDENT PROTOTYPE CONSTRUCTION ===
                if Shot >= 5:
                    # 5-shot: Apply prototype refinement to reduce outlier influence
                    # This stabilizes class prototypes for visually similar classes
                    # by removing the top 20% farthest support samples
                    proto = refine_prototype(
                        s_vectors,
                        outlier_fraction=self.outlier_fraction
                    )  # (hidden,)
                else:
                    # 1-shot: No refinement possible with single sample
                    # Use the single support embedding directly as prototype
                    # ClassConditionalCosine will handle discrimination
                    proto = s_vectors.squeeze(0)  # (hidden,)
                    proto = F.normalize(proto, p=2, dim=-1)  # Ensure L2 normalized
                
                prototypes.append(proto)
            
            # Stack prototypes: (Way, hidden)
            prototypes = torch.stack(prototypes, dim=0)
            
            # ============================================================
            # Step 3: Class-Conditional Cosine Similarity (learnable)
            # ============================================================
            # For BOTH 1-shot and 5-shot:
            # - g(proto_c) = Linear + Sigmoid → class-specific channel weights
            # - score_c = cosine(q ⊙ g(proto_c), proto_c)
            # This is the ONLY learnable component in Stage 7
            # Improves decision robustness by learning which channels matter
            # for each class, increasing effective cosine margin
            scores = self.cc_cosine(q_vectors, prototypes)  # (NQ, Way)
            
            all_scores.append(scores)
        
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
