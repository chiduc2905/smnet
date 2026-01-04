"""Ablation Study 3: Slot Attention Module.

Tests the contribution of SlotAttention:
    - Without SlotAttention (direct patch features)
    - With SlotAttention (semantic grouping)

Expected result: With SlotAttention > Without SlotAttention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum
from einops import rearrange


class SlotAttentionMode(Enum):
    """SlotAttention ablation modes."""
    WITHOUT = "without"     # No slot attention, use patches directly
    WITH = "with"           # Full slot attention (default)


@dataclass
class SlotAttentionAblationConfig:
    """Configuration for SlotAttention ablation."""
    mode: SlotAttentionMode = SlotAttentionMode.WITH
    
    # Model hyperparameters  
    dim: int = 128
    num_slots: int = 4
    slot_iters: int = 3
    d_state: int = 16
    d_conv: int = 4
    learnable_slots: bool = True
    
    def __post_init__(self):
        if isinstance(self.mode, str):
            self.mode = SlotAttentionMode(self.mode)
    
    @property
    def use_slot_attention(self) -> bool:
        return self.mode == SlotAttentionMode.WITH


class NoSlotAttention(nn.Module):
    """Bypass module when SlotAttention is disabled.
    
    Creates pseudo-slots by splitting patches into K groups.
    """
    
    def __init__(self, dim: int, num_slots: int = 4):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        
        # Project patches to slot dimension
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
    
    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Create pseudo-slots from patch features.
        
        Args:
            x: (B, C, H, W) feature maps
            return_attn: Whether to return (fake) attention maps
            
        Returns:
            slots: (B, K, C) pseudo-slots (mean pooled regions)
            slot_weights: (B, K) all ones
            attn: (B, K, N) uniform attention (if return_attn)
        """
        B, C, H, W = x.shape
        N = H * W
        K = self.num_slots
        
        # Flatten to patches
        patches = rearrange(x, 'b c h w -> b (h w) c')  # (B, N, C)
        patches = self.norm(self.proj(patches))
        
        # Create pseudo-slots by splitting spatially
        # Divide patches into K groups
        patches_per_slot = N // K
        
        slots = []
        attn_list = []
        
        for k in range(K):
            start_idx = k * patches_per_slot
            end_idx = start_idx + patches_per_slot if k < K - 1 else N
            
            # Mean pool patches in this region
            slot_k = patches[:, start_idx:end_idx, :].mean(dim=1)  # (B, C)
            slots.append(slot_k)
            
            # Fake attention: uniform within region
            attn_k = torch.zeros(B, N, device=x.device, dtype=x.dtype)
            attn_k[:, start_idx:end_idx] = 1.0 / (end_idx - start_idx)
            attn_list.append(attn_k)
        
        slots = torch.stack(slots, dim=1)  # (B, K, C)
        slot_weights = torch.ones(B, K, device=x.device, dtype=x.dtype)
        
        if return_attn:
            attn = torch.stack(attn_list, dim=1)  # (B, K, N)
            return slots, slot_weights, attn
        
        return slots, slot_weights


class SlotAttentionAblation(nn.Module):
    """SlotAttention with ablation support.
    
    Can switch between full SlotAttention and a bypass mode.
    
    Args:
        config: SlotAttentionAblationConfig with mode selection
    """
    
    def __init__(self, config: SlotAttentionAblationConfig):
        super().__init__()
        self.config = config
        self.mode = config.mode
        
        if config.use_slot_attention:
            from net.backbone.slot_attention import SlotAttention
            self.module = SlotAttention(
                num_slots=config.num_slots,
                dim=config.dim,
                iters=config.slot_iters,
                d_state=config.d_state,
                d_conv=config.d_conv,
                learnable_slots=config.learnable_slots
            )
        else:
            self.module = NoSlotAttention(
                dim=config.dim,
                num_slots=config.num_slots
            )
    
    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward through slot attention or bypass.
        
        Args:
            x: (B, C, H, W) feature maps
            return_attn: Whether to return attention maps
            
        Returns:
            slots: (B, K, C) slot descriptors
            slot_weights: (B, K) slot existence weights
            attn: (B, K, N) attention maps (if return_attn)
        """
        return self.module(x, return_attn=return_attn)


# ============================================================================
# Ablation Runner
# ============================================================================

def get_ablation_configs() -> dict:
    """Get all ablation configurations for SlotAttention."""
    return {
        "without_slot_attention": SlotAttentionAblationConfig(mode=SlotAttentionMode.WITHOUT),
        "with_slot_attention": SlotAttentionAblationConfig(mode=SlotAttentionMode.WITH),
    }


def run_ablation_test():
    """Quick test of ablation configurations."""
    print("=" * 60)
    print("SlotAttention Ablation Test")
    print("=" * 60)
    
    # Test input
    B, C, H, W = 2, 128, 16, 16
    x = torch.randn(B, C, H, W)
    
    configs = get_ablation_configs()
    
    for name, config in configs.items():
        print(f"\n[{name.upper()}] Mode: {config.mode.value}")
        print(f"  use_slot_attention: {config.use_slot_attention}")
        
        try:
            model = SlotAttentionAblation(config)
            model.eval()
            
            with torch.no_grad():
                slots, slot_weights, attn = model(x, return_attn=True)
            
            params = sum(p.numel() for p in model.parameters())
            print(f"  Input:        {tuple(x.shape)}")
            print(f"  Slots:        {tuple(slots.shape)}")
            print(f"  Slot weights: {tuple(slot_weights.shape)}")
            print(f"  Attention:    {tuple(attn.shape)}")
            print(f"  Params:       {params:,}")
            print(f"  ✓ OK")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_ablation_test()
