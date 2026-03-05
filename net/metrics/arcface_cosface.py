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
        margin: float = 0.5,
        class_margins: torch.Tensor = None,
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

        if class_margins is not None:
            cm = torch.as_tensor(class_margins, dtype=torch.float32).view(-1)
            if cm.numel() != out_features:
                raise ValueError(f"class_margins size={cm.numel()} must match out_features={out_features}")
            self.register_buffer("class_margins", cm)
        else:
            self.class_margins = None
    
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
        
        # Target cosine
        idx = torch.arange(cosine.size(0), device=cosine.device)
        target_cos = cosine[idx, labels]
        target_sin = torch.sqrt(1.0 - torch.clamp(target_cos ** 2, 0, 1))

        if self.class_margins is not None:
            margin = self.class_margins[labels]
            cos_m = torch.cos(margin)
            sin_m = torch.sin(margin)
            th = torch.cos(math.pi - margin)
            mm = torch.sin(math.pi - margin) * margin
        else:
            cos_m = torch.full_like(target_cos, self.cos_m)
            sin_m = torch.full_like(target_cos, self.sin_m)
            th = torch.full_like(target_cos, self.th)
            mm = torch.full_like(target_cos, self.mm)

        phi_target = target_cos * cos_m - target_sin * sin_m
        phi_target = torch.where(target_cos > th, phi_target, target_cos - mm)

        logits = cosine.clone()
        logits[idx, labels] = phi_target
        
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
        margin: float = 0.35,
        class_margins: torch.Tensor = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = scale
        self.m = margin
        
        # Classifier weights
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        if class_margins is not None:
            cm = torch.as_tensor(class_margins, dtype=torch.float32).view(-1)
            if cm.numel() != out_features:
                raise ValueError(f"class_margins size={cm.numel()} must match out_features={out_features}")
            self.register_buffer("class_margins", cm)
        else:
            self.class_margins = None
    
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
        
        logits = cosine.clone()
        idx = torch.arange(cosine.size(0), device=cosine.device)
        if self.class_margins is not None:
            logits[idx, labels] = logits[idx, labels] - self.class_margins[labels]
        else:
            logits[idx, labels] = logits[idx, labels] - self.m
        
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
