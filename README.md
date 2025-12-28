# Similarity Metrics for Few-Shot Learning

A collection of reusable similarity/distance metrics extracted from state-of-the-art few-shot learning models.

## Overview

This repository provides **8 standalone similarity metrics** that can be integrated into any few-shot learning architecture:

| Metric | Description | Source Paper |
|--------|-------------|--------------|
| **Euclidean** | Squared Euclidean distance | ProtoNet (NIPS 2017) |
| **Cosine** | Cosine similarity with optional temperature | MatchingNet, Baseline++ |
| **Covariance** | Distribution-based covariance metric | CovaMNet (AAAI 2019) |
| **Relation** | Learned CNN/MLP comparator | RelationNet (CVPR 2018) |
| **Local k-NN** | Top-k local descriptor matching | DN4 (CVPR 2019) |
| **EMD/Sinkhorn** | Optimal transport distance | DeepEMD (CVPR 2020) |
| **Learned Distance** | MLP on L1 difference | SiameseNet (ICML-W 2015) |
| **Transformer** | Self-attention embedding adaptation | FEAT (ICLR 2021) |

## Installation

```bash
pip install torch
```

## Usage

```python
import torch
from net.metrics import (
    EuclideanDistance,
    CosineBlock,
    CovaBlock,
    RelationBlock,
    LocalKNN,
    SinkhornDistance,
    LearnedDistance,
    SetTransformer
)

# Example: Euclidean distance
euclidean = EuclideanDistance()
query = torch.randn(10, 64)      # 10 queries, 64-dim
prototypes = torch.randn(4, 64)  # 4 classes
scores = euclidean(query, prototypes)  # (10, 4)

# Example: Cosine similarity with learnable temperature
cosine = CosineBlock(temperature=10.0, learnable_temp=True)
scores = cosine(query, prototypes)

# Example: Transformer adaptation
transformer = SetTransformer(dim=64, n_heads=4)
embeddings = torch.randn(1, 14, 64)  # batch, set_size, dim
adapted = transformer(embeddings)
```

## Structure

```
net/
├── __init__.py
├── utils.py              # Weight initialization utilities
└── metrics/
    ├── __init__.py
    ├── euclidean.py      # Euclidean distance
    ├── cosine.py         # Cosine similarity
    ├── covariance.py     # Covariance-based similarity
    ├── relation.py       # Learned relation module
    ├── local_knn.py      # Local descriptor k-NN
    ├── emd.py            # Earth Mover's Distance
    ├── learned_distance.py  # Learned distance (Siamese)
    └── transformer.py    # Transformer adaptation
```

## License

MIT License
