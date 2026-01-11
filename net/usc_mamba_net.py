"""USCMambaNet: Unified Spatial-Channel Mamba Network.

Implements a clean few-shot learning architecture following ADD-vs-MULTIPLY design:
- ADD: Feature extraction (encoder, dual-branch fusion, channel mixing)
- MUL: Feature selection (unified attention ONLY)
- Prototype Cross-Attention: Query refinement before GAP
- Similarity: Cosine Similarity with Bottleneck

Pipeline after Stage 6 (UnifiedAttention):
    Features → PrototypeCrossAttention → GAP → CosineSimilarityHead → Final Scores
    
CosineSimilarityHead:
    - Bottleneck (Linear → LayerNorm)
    - L2 Normalize
    - Cosine Similarity × temperature
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange

from net.backbone.feature_extractor import PatchEmbed2D, PatchMerging2D
from net.backbone.dual_branch_fusion import DualBranchFusion
from net.backbone.unified_attention import UnifiedSpatialChannelAttention
from net.metrics.hybrid_similarity_head import CosineSimilarityHead
from net.metrics.prototype_cross_attention import PrototypeCrossAttention


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
    - Cross-Attention: Prototype-guided query refinement
    - Similarity: Cosine with Bottleneck
    
    Architecture:
        Input → Encoder → DualBranchFusion → UnifiedAttention
              → PrototypeCrossAttention → GAP → CosineSimilarityHead → Scores
        
        Encoder:
            PatchEmbed2D → ConvBlocks → PatchMerging2D → ChannelProjection
        
        Feature Extraction (ADD-based):
            DualBranchFusion: Local (AG-LKA) + Global (VSS/Mamba)
        
        Feature Selection (MUL-based):
            UnifiedSpatialChannelAttention: ECA++ (channel) + DWConv (spatial)
        
        Query Refinement:
            PrototypeCrossAttention: Q' = softmax(Q·Pᵀ/√C)·P, Q = Q + α*Q'
        
        Similarity:
            CosineSimilarityHead: Bottleneck → L2 → Cosine
    
    Args:
        in_channels: Input channels (default: 3)
        base_dim: Base embedding dim (default: 32)
        hidden_dim: Hidden dim (default: 64)
        num_merging_stages: Patch merging stages (default: 2)
        d_state: Mamba state dimension (default: 4)
        temperature: Temperature for cosine similarity (default: 16.0)
        cross_attn_alpha: Residual weight for cross-attention (default: 0.1)
        use_projection: Whether to use bottleneck projection (default: True)
        device: Device to use
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        base_dim: int = 32,
        hidden_dim: int = 64,
        num_merging_stages: int = 2,
        d_state: int = 4,
        temperature: float = 16.0,
        cross_attn_alpha: float = 0.1,
        use_projection: bool = True,
        device: str = 'cuda',
        **kwargs  # For backward compatibility (delta_lambda, etc.)
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.device = device
        self.temperature = temperature
        self.use_projection = use_projection
        
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
        # STAGE 7: Prototype Cross-Attention (Query Refinement)
        # ============================================================
        self.proto_cross_attn = PrototypeCrossAttention(
            channels=hidden_dim,
            alpha=cross_attn_alpha
        )
        
        # ============================================================
        # STAGE 8: Cosine Similarity Head (Bottleneck → L2 → Cosine)
        # ============================================================
        self.similarity_head = CosineSimilarityHead(
            in_dim=hidden_dim,
            proj_dim=hidden_dim // 2,
            temperature=temperature,
            use_projection=use_projection
        )
        
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
        """Few-shot classification with Prototype Cross-Attention and Cosine Similarity.
        
        Pipeline:
            1. Encode query and support images
            2. PrototypeCrossAttention: Refine query with prototype maps
            3. GAP on refined queries
            4. CosineSimilarityHead: Bottleneck → L2 → Cosine
        
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
            # Step 1: Encode features (keep spatial)
            # ============================================================
            # Query: (NQ, C, H, W) → (NQ, hidden, H', W')
            q_features = self.encode(query[b])  # (NQ, hidden, H', W')
            
            # Support: (Way*Shot, C, H, W) → (Way*Shot, hidden, H', W')
            s_flat = support[b].view(Way * Shot, C, H, W)
            s_features = self.encode(s_flat)  # (Way*Shot, hidden, H', W')
            
            # ============================================================
            # Step 2: Prototype Cross-Attention (refine query)
            # ============================================================
            # Q' = softmax(Q·Pᵀ/√C)·P, Q = Q + α*Q'
            # refined_query: (NQ, Way, hidden, H', W')
            refined_query, proto_maps = self.proto_cross_attn(
                query_feat=q_features,
                support_feat=s_features,
                way_num=Way,
                shot_num=Shot
            )
            
            # ============================================================
            # Step 3: GAP on refined queries
            # ============================================================
            # (NQ, Way, hidden, H', W') → (NQ, Way, hidden)
            q_vectors = refined_query.mean(dim=[3, 4])  # GAP
            
            # Support vectors from prototype maps: (Way, hidden, H', W') → (Way, hidden)
            s_vectors = proto_maps.mean(dim=[2, 3])  # GAP
            
            # ============================================================
            # Step 4: Cosine Similarity for each query-prototype pair
            # ============================================================
            # Project and compute scores
            # q_vectors: (NQ, Way, hidden), s_vectors: (Way, hidden)
            
            # Project query vectors for each class
            scores_list = []
            for c in range(Way):
                q_c = q_vectors[:, c, :]  # (NQ, hidden) - query refined for class c
                s_c = s_vectors[c:c+1, :]  # (1, hidden) - prototype for class c
                
                # Project both
                z_q = self.similarity_head.project(q_c)  # (NQ, D)
                z_s = self.similarity_head.project(s_c)  # (1, D)
                z_s = F.normalize(z_s, p=2, dim=-1)  # Re-normalize
                
                # Cosine similarity
                score_c = torch.mm(z_q, z_s.t())  # (NQ, 1)
                score_c = self.similarity_head.temperature * score_c
                scores_list.append(score_c)
            
            # Stack: (NQ, Way)
            scores = torch.cat(scores_list, dim=1)
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
