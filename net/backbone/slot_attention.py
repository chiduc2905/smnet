"""Slot Attention with Mamba for semantic grouping.

Reference: Object-Centric Learning with Slot Attention (Locatello et al., NeurIPS 2020)
https://arxiv.org/abs/2006.15055

Uses Mamba SSM for slot updates instead of GRU.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba_ssm not available. SlotAttention will use fallback linear update.")


class SlotAttention(nn.Module):
    """Slot Attention with Mamba for slot updates.
    
    Groups pixel features into K semantic slot descriptors using
    iterative attention refinement with Mamba SSM for temporal modeling
    across iterations.
    
    Architecture:
        - Initialize K slot vectors (learnable)
        - For T iterations:
            - Compute attention: slots -> pixels (softmax over slots)
            - Weighted aggregation: pixels -> slots
            - Mamba update on slots
            - MLP refinement
        - Optional: Predict slot existence scores
    
    Args:
        num_slots: Maximum number of slot descriptors K (default: 4)
        dim: Feature dimension
        iters: Number of refinement iterations (default: 3)
        hidden_dim: MLP hidden dimension (default: dim * 2)
        d_state: Mamba state dimension (default: 16)
        d_conv: Mamba convolution width (default: 4)
        eps: Epsilon for numerical stability (default: 1e-8)
        learnable_slots: If True, predict slot existence scores (default: True)
    """
    
    def __init__(
        self,
        num_slots: int = 4,
        dim: int = 256,
        iters: int = 3,
        hidden_dim: int = None,
        d_state: int = 16,
        d_conv: int = 4,
        eps: float = 1e-8,
        learnable_slots: bool = True
    ):
        super().__init__()
        
        self.num_slots = num_slots
        self.dim = dim
        self.iters = iters
        self.eps = eps
        self.learnable_slots = learnable_slots
        
        hidden_dim = hidden_dim or dim * 2
        
        # Slot initialization (learnable mean and std)
        self.slot_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, dim))
        
        # Layer norms
        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)
        
        # Attention projections
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        
        # Mamba for slot updates
        if MAMBA_AVAILABLE:
            self.slot_mamba = Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=1  # No expansion for efficiency
            )
        else:
            # Fallback to simple linear update if Mamba not available
            self.slot_mamba = None
            self.linear_update = nn.Linear(dim * 2, dim)
        
        # MLP for slot refinement
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
        
        # Learnable slot existence predictor
        if learnable_slots:
            self.slot_existence = nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.GELU(),
                nn.Linear(dim // 2, 1),
                nn.Sigmoid()
            )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with xavier uniform."""
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
    
    def _mamba_update(
        self,
        updates: torch.Tensor,
        slots_prev: torch.Tensor
    ) -> torch.Tensor:
        """Update slots using Mamba SSM.
        
        Args:
            updates: (B, K, C) aggregated updates from attention
            slots_prev: (B, K, C) previous slot states
            
        Returns:
            (B, K, C) updated slots
        """
        B, K, C = updates.shape
        
        if self.slot_mamba is not None:
            # Concatenate previous slots and updates as sequence
            seq = torch.cat([slots_prev, updates], dim=1)  # (B, 2K, C)
            
            # Apply Mamba
            out = self.slot_mamba(seq)  # (B, 2K, C)
            
            # Take the second half (corresponding to updates position)
            slots = out[:, K:, :]  # (B, K, C)
            
            # Residual connection
            slots = slots + slots_prev
        else:
            # Fallback: simple linear combination
            combined = torch.cat([slots_prev, updates], dim=-1)  # (B, K, 2C)
            slots = self.linear_update(combined) + slots_prev
        
        return slots
    
    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False
    ) -> tuple:
        """
        Args:
            x: (B, C, H, W) input feature maps
            return_attn: Whether to return attention maps
            
        Returns:
            slots: (B, K, C) slot descriptors
            slot_weights: (B, K) slot existence weights (if learnable_slots=True)
            attn: (B, K, H*W) attention maps (if return_attn=True)
        """
        B, C, H, W = x.shape
        N = H * W
        
        # Flatten spatial dimensions: (B, C, H, W) -> (B, N, C)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm_input(x)
        
        # Initialize slots with learnable distribution
        slots = self.slot_mu + self.slot_log_sigma.exp() * torch.randn(
            B, self.num_slots, self.dim, device=x.device, dtype=x.dtype
        )
        
        # Iterative attention refinement
        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            
            # Attention: Q from slots, K and V from inputs
            q = self.to_q(slots)  # (B, K, C)
            k = self.to_k(x)       # (B, N, C)
            v = self.to_v(x)       # (B, N, C)
            
            # Attention scores: (B, K, N)
            scale = self.dim ** -0.5
            attn = torch.einsum('bkc,bnc->bkn', q, k) * scale
            
            # Softmax over slots (competition for pixels)
            attn = F.softmax(attn, dim=1)
            
            # Normalize to sum to 1 over pixels (for weighted mean)
            attn_weights = attn / (attn.sum(dim=-1, keepdim=True) + self.eps)
            
            # Weighted aggregation: (B, K, C)
            updates = torch.einsum('bkn,bnc->bkc', attn_weights, v)
            
            # Mamba update
            slots = self._mamba_update(updates, slots_prev)
            
            # MLP refinement with residual
            slots = slots + self.mlp(self.norm_mlp(slots))
        
        # Compute slot existence weights
        slot_weights = None
        if self.learnable_slots:
            slot_weights = self.slot_existence(slots).squeeze(-1)  # (B, K)
        
        if return_attn:
            return slots, slot_weights, attn
        
        return slots, slot_weights
    
    def get_effective_slots(
        self,
        slots: torch.Tensor,
        slot_weights: torch.Tensor,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """Get slots weighted by their existence scores."""
        if slot_weights is None:
            return slots
        return slots * slot_weights.unsqueeze(-1)
