"""USCMambaNet: Unified Spatial-Channel Mamba Network.

Implements a clean few-shot learning architecture following ADD-vs-MULTIPLY design:
- ADD: Feature extraction (encoder, dual-branch fusion, channel mixing)
- MUL: Feature selection (unified attention ONLY)
- Similarity: Simple GAP → L2 Norm → Cosine Similarity × temperature

Pipeline after Stage 6 (UnifiedAttention):
    Features → GAP → L2 Normalize → Cosine Similarity × temperature → Final Scores
    
Note: ClassConditionalCosine and refine_prototype temporarily removed for ablation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange

from net.backbone.feature_extractor import PatchEmbed2D, PatchMerging2D
from net.backbone.dual_branch_fusion import DualBranchFusion
from net.backbone.unified_attention import UnifiedSpatialChannelAttention
# Temporarily disabled for ablation:
# from net.metrics.prototype_refinement import refine_prototype
# from net.metrics.class_conditional_cosine import ClassConditionalCosine


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
    - Similarity: Simple GAP → L2 Norm → Cosine × temperature
    
    Note: ClassConditionalCosine and refine_prototype temporarily disabled for ablation.
    
    Architecture:
        Input → Encoder → DualBranchFusion → UnifiedAttention
              → GAP → L2 Norm → Cosine Similarity × temperature → Scores
        
        Encoder:
            PatchEmbed2D → ConvBlocks → PatchMerging2D → ChannelProjection
        
        Feature Extraction (ADD-based):
            DualBranchFusion: Local (AG-LKA) + Global (VSS/Mamba)
        
        Feature Selection (MUL-based):
            UnifiedSpatialChannelAttention: ECA++ (channel) + DWConv (spatial)
        
        Similarity (Non-learnable, simple):
            Cosine Similarity: scaled by temperature
    
    Args:
        in_channels: Input channels (default: 3)
        base_dim: Base embedding dim (default: 32)
        hidden_dim: Hidden dim (default: 64)
        num_merging_stages: Patch merging stages (default: 2)
        d_state: Mamba state dimension (default: 4)
        temperature: Temperature for cosine similarity (default: 1.0)
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
        device: str = 'cuda',
        **kwargs  # For backward compatibility (outlier_fraction ignored)
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.device = device
        self.temperature = temperature
        
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
        # (B, 32, 128, 128) → (B, 64, 128, 128)
        # ============================================================
        self.conv_blocks = nn.Sequential(
            ConvBlock(base_dim, base_dim * 2),      # 32 → 64
            ConvBlock(base_dim * 2, base_dim * 2),  # 64 → 64
        )
        
        # ============================================================
        # STAGE 3: Single PatchMerging (spatial /2, channels ×2)
        # (B, 64, 128, 128) → (B, 128, 64, 64)
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
        # STAGE 7: Simple Cosine Similarity (NON-learnable)
        # Simple: GAP → L2 Norm → Cosine × temperature
        # ClassConditionalCosine temporarily disabled for ablation
        # ============================================================
        # No learnable components needed for simple cosine
        
        self.to(device)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to feature maps.
        
        Args:
            x: (B, C, H, W) input images
            
        Returns:
            features: (B, hidden_dim, H', W') encoded features
        """
        # Stage 1: Patch embedding (B, 3, 128, 128) → (B, 32, 128, 128)
        f = self.patch_embed(x)
        
        # Stage 2: ConvBlocks (preserve spatial) (B, 32, 128, 128) → (B, 64, 128, 128)
        f = self.conv_blocks(f)
        
        # Stage 3: Single PatchMerging (B, 64, 128, 128) → (B, 128, 64, 64)
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
        """Few-shot classification with simple cosine similarity.
        
        Pipeline (simplified for ablation):
            1. Encode query and support images → GAP → L2 normalize
            2. Prototype: Simple mean of support embeddings (no refinement)
            3. Cosine Similarity × temperature (no class-conditional gating)
        
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
            # Step 2: Encode support and compute prototypes (simple mean)
            # ============================================================
            prototypes = []
            for w in range(Way):
                # Encode support for class w
                s_features = self.encode(support[b, w])  # (Shot, hidden, H', W')
                s_vectors = s_features.mean(dim=[2, 3])  # GAP: (Shot, hidden)
                s_vectors = F.normalize(s_vectors, p=2, dim=-1)  # L2 normalize
                
                # === SIMPLE MEAN PROTOTYPE (no refinement) ===
                # Just average all support samples for this class
                proto = s_vectors.mean(dim=0)  # (hidden,)
                proto = F.normalize(proto, p=2, dim=-1)  # Re-normalize after mean
                
                prototypes.append(proto)
            
            # Stack prototypes: (Way, hidden)
            prototypes = torch.stack(prototypes, dim=0)
            
            # ============================================================
            # Step 3: Simple Cosine Similarity × temperature
            # ============================================================
            # cosine(q, proto) = q @ proto.T (since both L2 normalized)
            # scores = (NQ, Way)
            scores = torch.mm(q_vectors, prototypes.t()) * self.temperature
            
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
