"""USCMambaNet: Unified Spatial-Channel Mamba Network.

Implements a clean few-shot learning architecture following ADD-vs-MULTIPLY design:
- ADD: Feature extraction (encoder, dual-branch fusion, channel mixing)
- MUL: Feature selection (late attention bridge ONLY by default)
- Similarity: Cosine Similarity with Bottleneck

Pipeline:
    Features → GAP → CosineSimilarityHead → Final Scores
    
CosineSimilarityHead:
    - Bottleneck (Linear → LayerNorm)
    - L2 Normalize
    - Cosine Similarity × temperature

Ablation Flags:
    - dualpath_mode: 'both', 'local_only', 'global_only', or 'none'
    - use_unified_attention: Enable/disable unified multi-scale attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from net.backbone.dual_branch_fusion import DualBranchFusion
from net.backbone.late_attention_bridge import LateSingleHeadAttentionBridge
from net.backbone.unified_attention import UnifiedSpatialChannelAttention
from net.metrics.hybrid_similarity_head import CosineSimilarityHead
from net.metrics.pair_expert_head import PairExpertCorrectionHead


class ConvStem64(nn.Module):
    """Legacy conv stem (same structure as old stem logic)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()

        def conv_block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.SiLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

        self.stem = nn.Sequential(
            conv_block(in_ch, out_ch),
            conv_block(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class USCMambaNet(nn.Module):
    """Unified Spatial-Channel Mamba Network (U-SCMambaNet).
    
    Uses ADD-vs-MULTIPLY design principles:
    - ADD: Feature extraction (encoder, fusion, channel mixing)
    - MUL: Feature selection (late attention bridge by default)
    - Similarity: Cosine with Bottleneck
    
    Architecture:
        Input → Encoder → DualBranchFusion → (UnifiedAttention optional)
              → LateAttentionBridge → GAP → CosineSimilarityHead → Scores
        
        Encoder:
            PatchEmbed2D → ConvBlocks → PatchMerging2D → ChannelProjection
        
        Feature Extraction (ADD-based):
            DualBranchFusion: Local (AG-LKA) + Global (VSS/Mamba)
        
        Feature Selection (MUL-based):
            UnifiedSpatialChannelAttention: ECA++ (channel) + DWConv (spatial)
        
        Similarity:
            CosineSimilarityHead: Bottleneck → L2 → Cosine
    
    Args:
        in_channels: Input channels (default: 3)
        base_dim: Base embedding dim (default: 32)
        hidden_dim: Hidden dim (default: 64)
        num_merging_stages: Patch merging stages (default: 2)
        d_state: Mamba state dimension (default: 8)
        global_expand: Expansion factor in VSS global branch (default: 2)
        temperature: Temperature for cosine similarity (default: 16.0)
        cross_attn_alpha: Residual weight for cross-attention (default: 0.1)
        use_projection: Whether to use bottleneck projection (default: True)
        dualpath_mode: 'both', 'local_only', 'global_only', or 'none' (default: 'both')
        use_unified_attention: Enable unified multi-scale attention (default: False)
        use_cross_attention: Deprecated, kept for backward compatibility
        device: Device to use
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        base_dim: int = 32,
        hidden_dim: int = 64,
        num_merging_stages: int = 2,
        d_state: int = 8,
        global_expand: int = 2,
        temperature: float = 16.0,
        beta_maha: float = 0.25,
        uaps_eps: float = 1e-4,
        cross_attn_alpha: float = 0.1,
        proto_pool_size: int = 12,
        num_prototypes: int = 2,
        detach_prototypes: bool = False,
        use_axis_proto: bool = False,
        axis_proto_pool: str = 'mean',
        axis_proto_mix_init: Tuple[float, float, float] = (1.0, 0.5, 0.5),
        use_late_attention: bool = True,
        late_attn_window: int = 4,
        late_attn_dropout: float = 0.0,
        similarity_proj_dim: Optional[int] = None,
        way_num: int = 4,
        use_pair_expert: bool = False,
        pair_conf_threshold: float = 0.60,
        pair_delta_max: float = 0.5,
        use_ms_global: bool = True,
        ms_downsample: int = 2,
        atrous_rate: int = 2,
        use_projection: bool = True,
        # Ablation flags
        dualpath_mode: str = 'both',  # 'both', 'local_only', 'global_only', 'none'
        use_unified_attention: bool = False,
        use_cross_attention: bool = False,
        device: str = 'cuda',
        **kwargs  # For backward compatibility
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.device = device
        self.temperature = temperature
        self.use_projection = use_projection
        self.way_num = way_num
        
        # Ablation flags
        self.dualpath_mode = dualpath_mode
        self.use_unified_attention = use_unified_attention
        # Cross-attention removed from main architecture (kept args for compatibility).
        self.use_cross_attention = False
        
        # ============================================================
        # STAGES 1-2: Conv stem for 64x64 input
        # Input: (B, 3, 64, 64) → Output: (B, hidden_dim, 16, 16)
        # ============================================================
        self.backbone = ConvStem64(in_channels, hidden_dim)
        
        # ============================================================
        # STAGE 5: Feature Extraction (ADD-based) - ABLATION: dualpath_mode
        # ============================================================
        if self.dualpath_mode != 'none':
            self.dual_branch = DualBranchFusion(
                channels=hidden_dim,
                d_state=d_state,
                expand=global_expand,
                dilation=2,
                mode=self.dualpath_mode,  # 'both', 'local_only', or 'global_only'
                use_ms_global=use_ms_global,
                ms_downsample=ms_downsample,
                atrous_rate=atrous_rate,
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
        # STAGE 6.5: Late Single-Head Attention Bridge
        # ============================================================
        self.use_late_attention = bool(use_late_attention)
        if self.use_late_attention:
            self.late_attention = LateSingleHeadAttentionBridge(
                channels=hidden_dim,
                window_size=late_attn_window,
                attn_dropout=late_attn_dropout,
            )
        else:
            self.late_attention = nn.Identity()
        
        # ============================================================
        # STAGE 7: Prototype Cross-Attention removed (architecture simplification)
        # Keep placeholder for backward compatibility with old checkpoints/code paths.
        self.proto_cross_attn = None
        
        # ============================================================
        # STAGE 8: Cosine Similarity Head (Bottleneck → L2 → Cosine)
        # ============================================================
        self.similarity_head = CosineSimilarityHead(
            in_dim=hidden_dim,
            proj_dim=similarity_proj_dim if similarity_proj_dim is not None else hidden_dim,
            temperature=temperature,
            beta_maha=beta_maha,
            uaps_eps=uaps_eps,
            num_classes=way_num,
            use_projection=use_projection
        )

        # Pair expert refinement for known confusing boundaries.
        if way_num >= 4:
            pair_defs = ((0, 3), (1, 2))
        elif way_num == 2:
            pair_defs = ((0, 1),)
        else:
            pair_defs = ()

        self.use_pair_expert = bool(use_pair_expert and len(pair_defs) > 0)
        if self.use_pair_expert:
            self.pair_expert = PairExpertCorrectionHead(
                embed_dim=self.similarity_head.proj_dim,
                pairs=pair_defs,
                delta_max=pair_delta_max,
                conf_threshold=pair_conf_threshold,
            )
        else:
            self.pair_expert = None
        
        self.to(device)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to feature maps.
        
        Args:
            x: (B, C, H, W) input images
            
        Returns:
            features: (B, hidden_dim, H', W') encoded features
        """
        # Stages 1-2: Conv stem (B, 3, 64, 64) → (B, hidden_dim, 16, 16)
        f = self.backbone(x)
        
        # Stage 5: Dual branch fusion (ADD-based) - conditional
        f = self.dual_branch(f)
        
        # Stage 6: Unified attention (MUL-based) - conditional
        f = self.unified_attention(f)

        # Stage 6.5: Late attention bridge
        f = self.late_attention(f)
        
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
        support: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor:
        """Few-shot classification with simplified late-attention architecture.
        
        Pipeline:
            1. Encode query and support images
            2. GAP on query/support features
            3. CosineSimilarityHead: Bottleneck → L2 → Cosine
        
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
        all_raw_scores = []
        pair_logits_collect = None
        pair_gates_collect = None
        
        for b in range(B):
            # ============================================================
            # Step 1: Encode features (keep spatial)
            # ============================================================
            # Query: (NQ, C, H, W) → (NQ, hidden, H', W')
            q_features = self.encode(query[b])  # (NQ, hidden, H', W')
            
            # Support: (Way*Shot, C, H, W) → (Way*Shot, hidden, H', W')
            s_flat = support[b].view(Way * Shot, C, H, W)
            s_features = self.encode(s_flat)  # (Way*Shot, hidden, H', W')
            s_gap = s_features.mean(dim=[2, 3]).view(Way, Shot, -1)  # (Way, Shot, hidden)
            
            # ============================================================
            # Step 2: Direct prototype matching (cross-attention removed)
            # ============================================================
            q_gap = q_features.mean(dim=[2, 3])  # (NQ, hidden)
            s_vectors = s_gap.mean(dim=1)  # (Way, hidden)
            q_vectors = q_gap.unsqueeze(1).expand(-1, Way, -1)  # (NQ, Way, hidden)
            
            # ============================================================
            # Step 3: Cosine + UAPS scoring
            # ============================================================
            raw_scores = self.similarity_head.forward_episode(
                query_vectors=q_vectors,
                support_shots=s_gap,
                prototype_vectors=s_vectors,
            )
            scores = raw_scores

            # ============================================================
            # Step 4: Pair Expert correction (optional)
            # ============================================================
            if self.use_pair_expert and self.pair_expert is not None:
                q_base = q_features.mean(dim=[2, 3])  # (NQ, hidden)
                z_base = self.similarity_head.project(q_base)  # (NQ, D)
                scores, pair_aux = self.pair_expert(z_base, scores)

                if pair_logits_collect is None:
                    pair_logits_collect = [[] for _ in pair_aux["expert_logits"]]
                    pair_gates_collect = [[] for _ in pair_aux["gates"]]
                for i in range(len(pair_aux["expert_logits"])):
                    pair_logits_collect[i].append(pair_aux["expert_logits"][i])
                    pair_gates_collect[i].append(pair_aux["gates"][i])

            all_raw_scores.append(raw_scores)
            all_scores.append(scores)

        scores_out = torch.cat(all_scores, dim=0)  # (B*NQ, Way)
        if not return_aux:
            return scores_out

        aux = {
            "raw_scores": torch.cat(all_raw_scores, dim=0),
            "expert_logits": None,
            "expert_gates": None,
        }
        if pair_logits_collect is not None:
            aux["expert_logits"] = [torch.cat(v, dim=0) for v in pair_logits_collect]
            aux["expert_gates"] = [torch.cat(v, dim=0) for v in pair_gates_collect]
        return scores_out, aux

    def compute_pair_expert_loss(self, aux: Optional[dict], targets: torch.Tensor) -> Optional[torch.Tensor]:
        """Compute auxiliary BCE loss for pair experts."""
        if not self.use_pair_expert or self.pair_expert is None or aux is None:
            return None
        expert_logits = aux.get("expert_logits")
        if expert_logits is None:
            return None
        return self.pair_expert.compute_loss(expert_logits, targets)
    
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
