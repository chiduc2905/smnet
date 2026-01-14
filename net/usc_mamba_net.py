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

Ablation Flags:
    - dualpath_mode: 'both', 'local_only', 'global_only', or 'none'
    - use_unified_attention: Enable/disable unified multi-scale attention
    - use_cross_attention: Enable/disable prototype cross-attention
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
        dualpath_mode: 'both', 'local_only', 'global_only', or 'none' (default: 'both')
        use_unified_attention: Enable unified multi-scale attention (default: True)
        use_cross_attention: Enable prototype cross-attention (default: True)
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
        # Ablation flags
        dualpath_mode: str = 'both',  # 'both', 'local_only', 'global_only', 'none'
        use_unified_attention: bool = True,
        use_cross_attention: bool = True,
        device: str = 'cuda',
        **kwargs  # For backward compatibility
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.device = device
        self.temperature = temperature
        self.use_projection = use_projection
        
        # Ablation flags
        self.dualpath_mode = dualpath_mode
        self.use_unified_attention = use_unified_attention
        self.use_cross_attention = use_cross_attention
        
        # ============================================================
        # STAGES 1-2: Lightweight Conv Stem (2 blocks)
        # Input: (B, 3, 128, 128) → Output: (B, 64, 32, 32)
        # Preserves spatial resolution for Mamba/LKA
        # ============================================================
        def conv_block(in_ch, out_ch):
            """Standard conv block: Conv → BN → SiLU → MaxPool"""
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)  # /2
            )
        
        self.backbone = nn.Sequential(
            conv_block(in_channels, hidden_dim),      # 3 → 64, 128 → 64
            conv_block(hidden_dim, hidden_dim),       # 64 → 64, 64 → 32
        )
        # Final output: (B, 64, 32, 32) with 2 MaxPools (128 / 4 = 32)
        
        # ============================================================
        # STAGE 5: Feature Extraction (ADD-based) - ABLATION: dualpath_mode
        # ============================================================
        if self.dualpath_mode != 'none':
            self.dual_branch = DualBranchFusion(
                channels=hidden_dim,
                d_state=d_state,
                dilation=2,
                mode=self.dualpath_mode  # 'both', 'local_only', or 'global_only'
            )
        else:
            # No dual branch processing - identity
            self.dual_branch = nn.Identity()
        
        # ============================================================
        # STAGE 6: Feature Selection (MUL-based) - ABLATION: use_unified_attention
        # ============================================================
        if self.use_unified_attention:
            self.unified_attention = UnifiedSpatialChannelAttention(hidden_dim)
        else:
            # Simple identity - no attention
            self.unified_attention = nn.Identity()
        
        # ============================================================
        # STAGE 7: Prototype Cross-Attention - ABLATION: use_cross_attention
        # ============================================================
        if self.use_cross_attention:
            self.proto_cross_attn = PrototypeCrossAttention(
                channels=hidden_dim,
                alpha=cross_attn_alpha
            )
        else:
            self.proto_cross_attn = None
        
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
        # Stages 1-2: Conv stem (B, 3, 128, 128) → (B, 64, 32, 32)
        f = self.backbone(x)
        
        # Stage 5: Dual branch fusion (ADD-based) - conditional
        f = self.dual_branch(f)
        
        # Stage 6: Unified attention (MUL-based) - conditional
        f = self.unified_attention(f)
        
        return f
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features for external use (e.g., t-SNE, center loss).
        
        Args:
            x: (B, C, H, W) input images
            
        Returns:
            features: (B, hidden_dim) pooled features
        """
        f = self.encode(x)  # (B, hidden_dim, H', W')
        f = f.mean(dim=[2, 3])  # GAP → (B, hidden_dim)
        return f
    
    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor
    ) -> torch.Tensor:
        """Few-shot classification with Prototype Cross-Attention and Cosine Similarity.
        
        Pipeline:
            1. Encode query and support images
            2. PrototypeCrossAttention: Refine query with prototype maps (if enabled)
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
            # Step 2: Prototype Cross-Attention (refine query) - conditional
            # ============================================================
            if self.use_cross_attention and self.proto_cross_attn is not None:
                # Q' = softmax(Q·Pᵀ/√C)·P, Q = Q + α*Q'
                # refined_query: (NQ, Way, hidden, H', W')
                refined_query, proto_maps = self.proto_cross_attn(
                    query_feat=q_features,
                    support_feat=s_features,
                    way_num=Way,
                    shot_num=Shot
                )
                
                # (NQ, Way, hidden, H', W') → (NQ, Way, hidden)
                q_vectors = refined_query.mean(dim=[3, 4])  # GAP
                
                # Support vectors from prototype maps: (Way, hidden, H', W') → (Way, hidden)
                s_vectors = proto_maps.mean(dim=[2, 3])  # GAP
            else:
                # No cross-attention: direct GAP on features
                # Query: (NQ, hidden, H', W') → (NQ, hidden)
                q_gap = q_features.mean(dim=[2, 3])  # (NQ, hidden)
                
                # Support: (Way*Shot, hidden, H', W') → (Way, hidden)
                s_gap = s_features.mean(dim=[2, 3])  # (Way*Shot, hidden)
                s_vectors = s_gap.view(Way, Shot, -1).mean(dim=1)  # (Way, hidden)
                
                # Expand q_vectors for each class
                q_vectors = q_gap.unsqueeze(1).expand(-1, Way, -1)  # (NQ, Way, hidden)
            
            # ============================================================
            # Step 3: Cosine Similarity for each query-prototype pair
            # ============================================================
            scores_list = []
            for c in range(Way):
                q_c = q_vectors[:, c, :]  # (NQ, hidden)
                s_c = s_vectors[c:c+1, :]  # (1, hidden)
                
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
    return USCMambaNet(**kwargs)
