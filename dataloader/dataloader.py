"""Episodic sampler for N-way K-shot few-shot learning."""
import torch
from torch.utils.data import Dataset


class FewshotDataset(Dataset):
    """
    N-way K-shot episode generator.
    
    Each episode contains:
    - Support set: way_num classes × shot_num samples
    - Query set: way_num classes × query_num samples
    
    Labels are episode-relative: 0, 1, ..., way_num-1
    """
    
    def __init__(self, data, labels, episode_num, way_num, shot_num, query_num, seed=None):
        """
        Args:
            data: Tensor (N, C, H, W)
            labels: Tensor (N,) with class labels 0..way_num-1
            episode_num: Number of episodes
            way_num: Classes per episode
            shot_num: Support samples per class
            query_num: Query samples per class
            seed: Random seed for reproducibility
        """
        self.data = data
        self.labels = labels
        self.episode_num = episode_num
        self.way_num = way_num
        self.shot_num = shot_num
        self.query_num = query_num
        self.seed = seed if seed is not None else 0
        
        # Pre-compute indices for each class
        self.class_indices = {}
        for c in range(way_num):
            self.class_indices[c] = (labels == c).nonzero(as_tuple=True)[0]
        
        # Validate data availability
        self._validate()
    
    def _validate(self):
        """Check if enough samples exist for requested shot/query."""
        required = self.shot_num + self.query_num
        for c in range(self.way_num):
            available = len(self.class_indices[c])
            if available < required:
                print(f"Warning: Class {c} has {available} samples, need {required}")

    def __len__(self):
        return self.episode_num

    def __getitem__(self, index):
        """
        Generate one episode.
        
        Returns:
            query_images: (way_num * query_num, C, H, W)
            query_targets: (way_num * query_num,) with labels 0..way_num-1
            support_images: (way_num * shot_num, C, H, W)
            support_targets: (way_num * shot_num,) with labels 0..way_num-1
        """
        gen = torch.Generator()
        gen.manual_seed(self.seed * 10000 + index)
        
        support_images, support_targets = [], []
        query_images, query_targets = [], []
        
        for class_id in range(self.way_num):
            indices = self.class_indices[class_id]
            perm = torch.randperm(len(indices), generator=gen)
            shuffled = indices[perm]
            
            # Split into support and query
            s_idx = shuffled[:self.shot_num]
            q_idx = shuffled[self.shot_num:self.shot_num + self.query_num]
            
            support_images.append(self.data[s_idx])
            query_images.append(self.data[q_idx])
            
            # Episode-relative labels
            support_targets.append(torch.full((len(s_idx),), class_id, dtype=torch.long))
            query_targets.append(torch.full((len(q_idx),), class_id, dtype=torch.long))
        
        return (torch.cat(query_images), torch.cat(query_targets),
                torch.cat(support_images), torch.cat(support_targets))
