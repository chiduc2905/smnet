"""Ablation Study 2: Slot Refinement Modules (M5 + M6).

Tests the contribution of slot refinement modules:
    - None (baseline, no SCA or CMA)
    - SCA only (SlotCovarianceAttention)
    - CMA only (ChannelMetricAttention)
    - Both (SCA + CMA)

Expected result: Both >= SCA only, Both >= CMA only, All >= None
"""
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum


class SlotRefinementMode(Enum):
    """Slot refinement ablation modes."""
    NONE = "none"           # No refinement (baseline)
    SCA_ONLY = "sca_only"   # Only SlotCovarianceAttention
    CMA_ONLY = "cma_only"   # Only ChannelMetricAttention
    BOTH = "both"           # SCA + CMA (full pipeline)


@dataclass
class SlotRefinementAblationConfig:
    """Configuration for slot refinement ablation."""
    mode: SlotRefinementMode = SlotRefinementMode.BOTH
    
    # Model hyperparameters
    dim: int = 128
    num_slots: int = 4
    sca_temperature: float = 1.0
    cma_temperature: float = 1.0
    cma_use_residual: bool = True
    
    def __post_init__(self):
        if isinstance(self.mode, str):
            self.mode = SlotRefinementMode(self.mode)
    
    @property
    def use_sca(self) -> bool:
        return self.mode in [SlotRefinementMode.SCA_ONLY, SlotRefinementMode.BOTH]
    
    @property
    def use_cma(self) -> bool:
        return self.mode in [SlotRefinementMode.CMA_ONLY, SlotRefinementMode.BOTH]


class SlotRefinementAblation(nn.Module):
    """Slot Refinement module with ablation support.
    
    Combines M5 (SCA) and M6 (CMA) with configurable ablation.
    
    Args:
        config: SlotRefinementAblationConfig with mode selection
    """
    
    def __init__(self, config: SlotRefinementAblationConfig):
        super().__init__()
        self.config = config
        self.mode = config.mode
        
        # Import modules
        from net.backbone.slot_covariance_attention import SlotCovarianceAttention
        from net.backbone.channel_metric_attention import ChannelMetricAttention
        
        # Initialize based on mode
        self.sca = None
        self.cma = None
        
        if config.use_sca:
            self.sca = SlotCovarianceAttention(
                dim=config.dim,
                temperature=config.sca_temperature
            )
        
        if config.use_cma:
            self.cma = ChannelMetricAttention(
                dim=config.dim,
                temperature=config.cma_temperature,
                use_residual=config.cma_use_residual
            )
    
    def forward(
        self,
        slots: torch.Tensor,
        attn: torch.Tensor,
        patches: torch.Tensor,
        class_cov: torch.Tensor,
        class_proto: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply slot refinement based on configuration.
        
        Args:
            slots: (B, K, C) slot descriptors
            attn: (B, K, N) attention matrix
            patches: (B, N, C) patch tokens
            class_cov: (C, C) class covariance matrix
            class_proto: (C,) class prototype vector
            
        Returns:
            refined_slots: (B, K, C) refined slot descriptors
            alpha: (B, K) slot importance weights (or None if SCA disabled)
        """
        alpha = None
        refined_slots = slots
        
        # M5: Slot Covariance Attention
        if self.sca is not None:
            alpha = self.sca(slots, attn, patches, class_cov)  # (B, K)
        
        # M6: Channel Metric Attention
        if self.cma is not None:
            refined_slots = self.cma(slots, class_proto, slot_weights=alpha)  # (B, K, C)
        elif alpha is not None:
            # If only SCA, apply alpha weights to slots
            refined_slots = slots * alpha.unsqueeze(-1)  # (B, K, C)
        
        return refined_slots, alpha


# ============================================================================
# Ablation Runner
# ============================================================================

def get_ablation_configs() -> dict:
    """Get all ablation configurations for slot refinement."""
    return {
        "none": SlotRefinementAblationConfig(mode=SlotRefinementMode.NONE),
        "sca_only": SlotRefinementAblationConfig(mode=SlotRefinementMode.SCA_ONLY),
        "cma_only": SlotRefinementAblationConfig(mode=SlotRefinementMode.CMA_ONLY),
        "both": SlotRefinementAblationConfig(mode=SlotRefinementMode.BOTH),
    }


def run_ablation_test():
    """Quick test of ablation configurations."""
    print("=" * 60)
    print("Slot Refinement (SCA + CMA) Ablation Test")
    print("=" * 60)
    
    # Test inputs
    B, K, N, C = 2, 4, 256, 128
    slots = torch.randn(B, K, C)
    attn = torch.softmax(torch.randn(B, K, N), dim=1)
    patches = torch.randn(B, N, C)
    class_cov = torch.eye(C) + 0.1 * torch.randn(C, C)
    class_cov = (class_cov + class_cov.T) / 2  # Symmetric
    class_proto = torch.randn(C)
    class_proto = class_proto / class_proto.norm()
    
    configs = get_ablation_configs()
    
    for name, config in configs.items():
        print(f"\n[{name.upper()}] Mode: {config.mode.value}")
        print(f"  use_sca: {config.use_sca}, use_cma: {config.use_cma}")
        
        try:
            model = SlotRefinementAblation(config)
            model.eval()
            
            with torch.no_grad():
                refined_slots, alpha = model(
                    slots, attn, patches, class_cov, class_proto
                )
            
            params = sum(p.numel() for p in model.parameters())
            print(f"  Input slots:    {tuple(slots.shape)}")
            print(f"  Refined slots:  {tuple(refined_slots.shape)}")
            print(f"  Alpha:          {tuple(alpha.shape) if alpha is not None else 'None'}")
            print(f"  Params:         {params:,}")
            print(f"  ✓ OK")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_ablation_test()
