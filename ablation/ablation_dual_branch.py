"""Ablation Study 1: DualBranchFusion Components.

Tests the contribution of local and global branches in DualBranchFusion:
    - Local only (ConvMixer branch)
    - Global only (SS2D/Mamba branch)  
    - Both branches (full DualBranchFusion)

Expected result: Both > Local only, Both > Global only
"""
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Literal, Optional
from enum import Enum


class DualBranchMode(Enum):
    """DualBranchFusion ablation modes."""
    LOCAL_ONLY = "local_only"      # Only ConvMixer branch
    GLOBAL_ONLY = "global_only"    # Only SS2D/Mamba branch
    BOTH = "both"                  # Full fusion (default)


@dataclass
class DualBranchAblationConfig:
    """Configuration for DualBranchFusion ablation."""
    mode: DualBranchMode = DualBranchMode.BOTH
    
    # Model hyperparameters
    channels: int = 128
    d_state: int = 16
    dilation: int = 2
    
    def __post_init__(self):
        if isinstance(self.mode, str):
            self.mode = DualBranchMode(self.mode)


class DualBranchFusionAblation(nn.Module):
    """DualBranchFusion with ablation support.
    
    Wraps DualBranchFusion to enable testing individual branches.
    
    Args:
        config: DualBranchAblationConfig with mode selection
    """
    
    def __init__(self, config: DualBranchAblationConfig):
        super().__init__()
        self.config = config
        self.mode = config.mode
        
        # Import here to avoid circular imports
        from net.backbone.dual_branch_fusion import (
            LocalBranch, 
            GlobalBranch,
            DualBranchFusion
        )
        
        if config.mode == DualBranchMode.LOCAL_ONLY:
            self.branch = LocalBranch(
                channels=config.channels,
                dilation=config.dilation
            )
            self._forward_fn = self._forward_local
            
        elif config.mode == DualBranchMode.GLOBAL_ONLY:
            self.branch = GlobalBranch(
                channels=config.channels,
                d_state=config.d_state
            )
            self._forward_fn = self._forward_global
            
        else:  # BOTH
            self.dual_branch = DualBranchFusion(
                channels=config.channels,
                d_state=config.d_state,
                dilation=config.dilation
            )
            self._forward_fn = self._forward_both
    
    def _forward_local(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with local branch only."""
        return self.branch(x)
    
    def _forward_global(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with global branch only."""
        return self.branch(x)
    
    def _forward_both(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with full dual branch fusion."""
        return self.dual_branch(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_fn(x)


# ============================================================================
# Ablation Runner
# ============================================================================

def get_ablation_configs() -> dict:
    """Get all ablation configurations for DualBranch."""
    return {
        "local_only": DualBranchAblationConfig(mode=DualBranchMode.LOCAL_ONLY),
        "global_only": DualBranchAblationConfig(mode=DualBranchMode.GLOBAL_ONLY),
        "both": DualBranchAblationConfig(mode=DualBranchMode.BOTH),
    }


def run_ablation_test():
    """Quick test of ablation configurations."""
    print("=" * 60)
    print("DualBranchFusion Ablation Test")
    print("=" * 60)
    
    # Test input
    B, C, H, W = 2, 128, 16, 16
    x = torch.randn(B, C, H, W)
    
    configs = get_ablation_configs()
    
    for name, config in configs.items():
        print(f"\n[{name.upper()}] Mode: {config.mode.value}")
        
        try:
            model = DualBranchFusionAblation(config)
            model.eval()
            
            with torch.no_grad():
                out = model(x)
            
            params = sum(p.numel() for p in model.parameters())
            print(f"  Input:  {tuple(x.shape)}")
            print(f"  Output: {tuple(out.shape)}")
            print(f"  Params: {params:,}")
            print(f"  ✓ OK")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_ablation_test()
