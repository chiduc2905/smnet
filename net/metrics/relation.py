"""Learned relation metric for few-shot learning.

Extracted from: RelationNet (Sung et al., CVPR 2018)
"Learning to Compare: Relation Network for Few-Shot Learning"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationBlock(nn.Module):
    """Learned relation module that compares feature pairs.
    
    Takes concatenated feature pairs and outputs relation scores in [0,1].
    Input is typically the concatenation of query and support features.
    """
    
    def __init__(self, input_channels: int = 128, hidden_size: int = 8):
        """Initialize RelationBlock.
        
        Args:
            input_channels: Number of input channels (2 * feature_channels from concat)
            hidden_size: Hidden dimension for FC layers
        """
        super(RelationBlock, self).__init__()
        
        # Relation module: learns to compare concatenated features
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # FC layers
        # Assuming input is 4x4, after 2 maxpools: 4->2->1
        self.fc1 = nn.Linear(64 * 1 * 1, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute relation score for concatenated feature pairs.
        
        Args:
            x: (B, input_channels, H, W) concatenated feature pairs
            
        Returns:
            scores: (B, 1) relation scores in [0, 1]
        """
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)  # Flatten
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))  # Sigmoid for [0,1]
        return out


class RelationMLP(nn.Module):
    """Simpler MLP-based relation module for flattened features."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        """Initialize RelationMLP.
        
        Args:
            input_dim: Input dimension (2 * feature_dim from concat)
            hidden_dim: Hidden layer dimension
        """
        super(RelationMLP, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute relation score.
        
        Args:
            x: (B, input_dim) concatenated features
            
        Returns:
            scores: (B, 1) relation scores in [0, 1]
        """
        return self.net(x)
