"""ArcFace and CosFace: SOTA Angular Margin Losses for Closed-Set Embedding.

References:
- ArcFace: https://arxiv.org/abs/1801.07698 (Additive Angular Margin)
- CosFace: https://arxiv.org/abs/1801.09414 (Additive Cosine Margin)

Key principle:
- Embedding and classifier weights are L2 normalized
- logit = s * cos(θ)
- Margin is applied ONLY to target class to enforce separation

CosFace: cos(θ_y) ← cos(θ_y) - m
ArcFace: cos(θ_y) ← cos(θ_y + m)  # stronger angular penalty

Usage:
    # Training
    logits = arcface(feat, labels)
    loss = F.cross_entropy(logits, labels)
    
    # Inference (NO margin, just cosine classifier)
    W = F.normalize(arcface.weight)
    logits = torch.matmul(feat, W.t()) * arcface.s
    pred = logits.argmax(dim=1)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ArcFace(nn.Module):
    """ArcFace: Additive Angular Margin Loss.
    
    Applies angular margin to target class:
        cos(θ + m) instead of cos(θ)
    
    This is stronger than CosFace because margin is in angle space.
    
    Args:
        in_features: Embedding dimension
        out_features: Number of classes
        scale: Logit scaling factor s (default: 30.0)
        margin: Angular margin m in radians (default: 0.5)
    """
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int,
        scale: float = 30.0, 
        margin: float = 0.5
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = scale
        self.m = margin
        
        # Classifier weights (will be L2 normalized in forward)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        # Precompute cos/sin of margin for efficiency
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        
        # Threshold for numeric stability
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
    
    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute ArcFace logits.
        
        Args:
            x: (B, in_features) embeddings (will be L2-normalized)
            labels: (B,) class labels
            
        Returns:
            logits: (B, out_features) scaled logits with angular margin
        """
        # L2 normalize embedding and weights
        x = F.normalize(x, p=2, dim=1)
        W = F.normalize(self.weight, p=2, dim=1)
        
        # Cosine similarity: (B, num_classes)
        cosine = F.linear(x, W)
        
        # Compute sin from cos
        sine = torch.sqrt(1.0 - torch.clamp(cosine ** 2, 0, 1))
        
        # cos(θ + m) = cos(θ)cos(m) - sin(θ)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # Numeric stability: when cos(θ) < cos(π - m), use linear approximation
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # One-hot encoding for target class
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        
        # Apply margin ONLY to target class
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        # Scale
        logits = logits * self.s
        
        return logits
    
    def inference(self, x: torch.Tensor) -> torch.Tensor:
        """Pure cosine inference WITHOUT margin (for val/test).
        
        Args:
            x: (B, in_features) embeddings
            
        Returns:
            logits: (B, out_features) scaled cosine logits
        """
        x = F.normalize(x, p=2, dim=1)
        W = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(x, W)
        return cosine * self.s


class CosFace(nn.Module):
    """CosFace: Additive Cosine Margin Loss.
    
    Simpler than ArcFace - subtracts margin directly from cosine:
        cos(θ) - m instead of cos(θ)
    
    Easier to tune and more stable during training.
    
    Args:
        in_features: Embedding dimension
        out_features: Number of classes
        scale: Logit scaling factor s (default: 30.0)
        margin: Cosine margin m (default: 0.35)
    """
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int,
        scale: float = 30.0, 
        margin: float = 0.35
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = scale
        self.m = margin
        
        # Classifier weights
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute CosFace logits.
        
        Args:
            x: (B, in_features) embeddings (will be L2-normalized)
            labels: (B,) class labels
            
        Returns:
            logits: (B, out_features) scaled logits with cosine margin
        """
        # L2 normalize
        x = F.normalize(x, p=2, dim=1)
        W = F.normalize(self.weight, p=2, dim=1)
        
        # Cosine similarity
        cosine = F.linear(x, W)  # (B, num_classes)
        
        # One-hot for target class
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        
        # Subtract margin from target class logit
        logits = cosine - one_hot * self.m
        
        # Scale
        logits = logits * self.s
        
        return logits
    
    def inference(self, x: torch.Tensor) -> torch.Tensor:
        """Pure cosine inference WITHOUT margin (for val/test).
        
        Args:
            x: (B, in_features) embeddings
            
        Returns:
            logits: (B, out_features) scaled cosine logits
        """
        x = F.normalize(x, p=2, dim=1)
        W = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(x, W)
        return cosine * self.s


# Factory functions
def build_arcface(in_features: int, out_features: int, scale: float = 30.0, margin: float = 0.5) -> ArcFace:
    """Build ArcFace module with recommended defaults."""
    return ArcFace(in_features, out_features, scale, margin)


def build_cosface(in_features: int, out_features: int, scale: float = 30.0, margin: float = 0.35) -> CosFace:
    """Build CosFace module with recommended defaults."""
    return CosFace(in_features, out_features, scale, margin)
