"""Slot-level global reasoning via Mamba state-space module.

Processes slot descriptors as a sequence to capture inter-slot dependencies.
"""
import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


class SlotMamba(nn.Module):
    """Mamba state-space model for slot-level global reasoning.
    
    Processes slot descriptors as a sequence to model inter-slot
    dependencies and capture global semantic relationships.
    
    Args:
        d_model: Slot dimension
        d_state: SSM state dimension (default: 16)
        d_conv: Local convolution width (default: 4)
        expand: Expansion factor (default: 2)
        num_layers: Number of Mamba layers (default: 2)
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 1,
        num_layers: int = 1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba-ssm package is required for SlotMamba. "
                "Install with: pip install mamba-ssm"
            )
        
        # Stack of Mamba layers with layer norms
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(num_layers):
            self.layers.append(
                Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand
                )
            )
            self.norms.append(nn.LayerNorm(d_model))
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slots: (B, K, C) slot descriptors
            
        Returns:
            (B, K, C) refined slot descriptors with global context
        """
        x = slots
        
        for mamba, norm in zip(self.layers, self.norms):
            # Pre-norm residual
            residual = x
            x = norm(x)
            x = mamba(x)
            x = x + residual
        
        x = self.final_norm(x)
        
        return x
