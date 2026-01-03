"""Slot Attention for semantic grouping.

Reference: Object-Centric Learning with Slot Attention (Locatello et al., NeurIPS 2020)
https://arxiv.org/abs/2006.15055

Extended with learnable number of slots via slot existence prediction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SlotAttention(nn.Module):
    """Slot Attention module for grouping features into semantic slots.
    
    Groups pixel features into K semantic slot descriptors using
    iterative attention refinement. Supports learnable number of active slots.
    
    Architecture:
        - Initialize K slot vectors (learnable)
        - For T iterations:
            - Compute attention: slots -> pixels (softmax over slots)
            - Weighted aggregation: pixels -> slots
            - GRU update on slots
            - MLP refinement
        - Optional: Predict slot existence scores for dynamic slot count
    
    Args:
        num_slots: Maximum number of slot descriptors K (default: 4)
        dim: Feature dimension
        iters: Number of refinement iterations (default: 3)
        hidden_dim: MLP hidden dimension (default: dim * 2)
        eps: Epsilon for numerical stability (default: 1e-8)
        learnable_slots: If True, predict slot existence scores (default: True)
    """
    
    def __init__(
        self,
        num_slots: int = 4,
        dim: int = 256,
        iters: int = 3,
        hidden_dim: int = None,
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
        
        # GRU for slot updates
        self.gru = nn.GRUCell(dim, dim)
        
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
        """Initialize weights with truncated normal."""
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
    
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
        # Sample from N(mu, sigma) for each batch
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
            
            # GRU update
            slots = self.gru(
                updates.reshape(-1, self.dim),
                slots_prev.reshape(-1, self.dim)
            ).reshape(B, self.num_slots, self.dim)
            
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
        """Get slots weighted by their existence scores.
        
        Args:
            slots: (B, K, C) slot descriptors
            slot_weights: (B, K) slot existence weights
            threshold: Threshold for slot activation (default: 0.5)
            
        Returns:
            (B, K, C) weighted slot descriptors
        """
        if slot_weights is None:
            return slots
        
        # Apply soft weighting
        return slots * slot_weights.unsqueeze(-1)
