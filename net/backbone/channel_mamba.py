"""ChannelMamba: Mamba-based Channel Mixing Module (ADD-based).

Uses Mamba SSM to model channel interactions WITHOUT multiplicative gating.
This follows the ADD design principle: Y = X + α · Mix(X)

Reference:
    - Feature extraction modules use RESIDUAL ADDITION (not attention/gating)
"""
import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


class ChannelMamba(nn.Module):
    """Mamba-based Channel Receptive Field (Mixing, NOT Attention).
    
    Uses Mamba SSM to model channel interactions without multiplicative gating.
    
    Pipeline:
        1. GAP: (B, C, H, W) → (B, C, 1, 1) → (B, C)
        2. Expand to (B, 1, C) for Mamba sequence processing
        3. Mamba: channel sequence modeling
        4. Linear: (B, C) → (B, C, 1, 1)
        5. ADD residual: Y = X + α · Mix
    
    Design Constraints:
        - Operates on channel dimension only (1D sequence of C tokens)
        - Uses Global Average Pooling to obtain channel sequence
        - Lightweight Mamba: d_state=4, expand=1, single layer, no FFN
        - Integration: Y = X + α · ChannelMix(X)
    
    Args:
        channels: Number of input channels
        d_state: SSM state dimension (default: 4)
        d_conv: Local convolution width in Mamba (default: 3)
        alpha_init: Initial value for residual scaling (default: 0.1)
    """
    
    def __init__(
        self,
        channels: int,
        d_state: int = 4,
        d_conv: int = 3,
        alpha_init: float = 0.1
    ):
        super().__init__()
        
        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba-ssm package is required. Install with: pip install mamba-ssm"
            )
        
        self.channels = channels
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Mamba for channel sequence modeling
        # Lightweight: expand=1, d_state=4
        self.mamba = Mamba(
            d_model=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=1  # No expansion for lightweight
        )
        
        # Output projection
        self.proj = nn.Linear(channels, channels, bias=False)
        
        # Learnable residual scaling
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input features
            
        Returns:
            (B, C, H, W) output features with channel mixing
        """
        B, C, H, W = x.shape
        
        # GAP: (B, C, H, W) → (B, C, 1, 1) → (B, C)
        gap = self.gap(x).view(B, C)
        
        # Mamba expects (B, L, D) → treat as (B, 1, C)
        gap_seq = gap.unsqueeze(1)  # (B, 1, C)
        
        # Channel sequence modeling
        mix = self.mamba(gap_seq).squeeze(1)  # (B, C)
        
        # Project back
        mix = self.proj(mix).view(B, C, 1, 1)  # (B, C, 1, 1)
        
        # ADD residual: Y = X + α · Mix
        return x + self.alpha * mix
