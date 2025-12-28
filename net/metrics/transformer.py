"""Transformer-based embedding adaptation for few-shot learning.

Extracted from: FEAT (Ye et al., ICLR 2021)
"FEAT: Few-Shot Embedding Adaptation with Transformer"
"""
import torch
import torch.nn as nn


class SetTransformer(nn.Module):
    """Set Transformer for embedding adaptation.
    
    Takes a set of embeddings and adapts them based on the entire set context.
    Used to make embeddings task-adaptive.
    """
    
    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1):
        """Initialize SetTransformer.
        
        Args:
            dim: Feature dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(SetTransformer, self).__init__()
        
        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention adaptation.
        
        Args:
            x: (B, N, D) embeddings where N is set size
            
        Returns:
            (B, N, D) adapted embeddings
        """
        # Self-attention with residual
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x


class CrossAttention(nn.Module):
    """Cross-attention module for query-support interaction.
    
    Allows query embeddings to attend to support embeddings.
    """
    
    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1):
        """Initialize CrossAttention.
        
        Args:
            dim: Feature dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(CrossAttention, self).__init__()
        
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, query: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        """Apply cross-attention.
        
        Args:
            query: (B, NQ, D) query embeddings
            support: (B, NS, D) support embeddings
            
        Returns:
            (B, NQ, D) adapted query embeddings
        """
        attn_out, _ = self.attn(query, support, support)
        return self.norm(query + attn_out)


class EmbeddingAdapter(nn.Module):
    """Combined embedding adapter with self and cross attention.
    
    Full FEAT-style adaptation pipeline.
    """
    
    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1):
        """Initialize EmbeddingAdapter.
        
        Args:
            dim: Feature dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(EmbeddingAdapter, self).__init__()
        
        # Self-attention on support set
        self.support_self_attn = SetTransformer(dim, n_heads, dropout)
        
        # Cross-attention from query to support
        self.query_cross_attn = CrossAttention(dim, n_heads, dropout)
    
    def forward(self, query: torch.Tensor, support: torch.Tensor) -> tuple:
        """Adapt embeddings based on task context.
        
        Args:
            query: (B, NQ, D) query embeddings
            support: (B, NS, D) support embeddings
            
        Returns:
            adapted_query: (B, NQ, D) adapted query embeddings
            adapted_support: (B, NS, D) adapted support embeddings
        """
        # Adapt support via self-attention
        adapted_support = self.support_self_attn(support)
        
        # Adapt query via cross-attention to support
        adapted_query = self.query_cross_attn(query, adapted_support)
        
        return adapted_query, adapted_support
