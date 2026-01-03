# SMNet: Slot Mamba Network for Few-Shot Learning

A few-shot classification framework combining ConvMixer, Mamba state-space models, Slot Attention, and covariance-based metric learning.

## Architecture Overview

![SMNet Architecture](docs/slot_covariance_architecture.png)

### Pipeline

1. **ConvMixer Encoder** - Local spatial feature extraction with depthwise separable convolutions
2. **Channel Attention** - SE-style lightweight channel interaction
3. **SS2D (4-way Mamba)** - Global spatial context via 4-directional scanning
4. **Slot Attention** - Groups features into K semantic slot descriptors (learnable K=4)
5. **Slot Mamba** - Inter-slot reasoning via state-space sequence modeling
6. **Slot Covariance Similarity** - Second-order metric using covariance matrices

### Key Design Principles

- **Shared Weights**: Same feature extractor for support and query ensures consistent embedding space
- **No Classifier Head**: Pure metric-based inference using covariance similarity
- **Second-Order Statistics**: Captures distribution shape, not just mean
- **Learnable Slots**: Number of active slots is predicted per image

## Installation

```bash
pip install torch>=1.12.0 einops>=0.6.0
pip install mamba-ssm  # Requires CUDA
```

## Training

### Quick Start

```bash
# 1-shot 4-way training
python main.py --model smnet --shot_num 1 --way_num 4 --dataset_path ./scalogram

# 5-shot training
python main.py --model smnet --shot_num 5 --way_num 4 --dataset_path ./scalogram

# 10-shot training
python main.py --model smnet --shot_num 10 --way_num 4 --dataset_path ./scalogram

# Lightweight model (faster)
python main.py --model smnet_light --shot_num 1 --way_num 4
```

### Full Training Options

```bash
python main.py \
    --model smnet \
    --dataset_path ./scalogram \
    --dataset_name minh \
    --shot_num 1 \
    --way_num 4 \
    --num_slots 4 \
    --hidden_dim 256 \
    --num_epochs 100 \
    --lr 1e-4 \
    --training_samples 800 \
    --episode_num_train 100 \
    --episode_num_val 200 \
    --episode_num_test 300 \
    --project smnet
```

### Run All Experiments

```bash
# Run 1-shot, 5-shot, 10-shot with all sample sizes
python run_all_experiments.py --project smnet --dataset_path /path/to/dataset
```

### Test Only

```bash
python main.py --mode test --weights checkpoints/smnet_1shot_best.pth
```

## Usage (API)

### Feature Extraction

```python
from net.backbone import SlotFeatureExtractor

# Create extractor with K=4 learnable slots
extractor = SlotFeatureExtractor(
    hidden_dim=256,
    num_slots=4,
    learnable_slots=True
)

# Extract slot descriptors
images = torch.randn(5, 3, 224, 224)
slots, slot_weights = extractor(images)  # (5, 4, 256), (5, 4)
```

### SMNet Model

```python
from net.slot_fewshot import SMNet

# Create SMNet model
model = SMNet(
    hidden_dim=256,
    num_slots=4,
    learnable_slots=True,
    device='cuda'
)

# Forward pass
support = torch.randn(1, 5, 5, 3, 64, 64)  # (B, Way, Shot, C, H, W)
query = torch.randn(1, 25, 3, 64, 64)       # (B, NQ, C, H, W)
scores = model(query, support)              # (25, 5)
predictions = scores.argmax(dim=-1)
```

## Structure

```
smnet/
├── main.py                       # Training script
├── run_all_experiments.py        # Run all experiments
├── dataset.py                    # Dataset loader
├── dataloader/
│   └── dataloader.py             # Episodic sampler
├── function/
│   └── function.py               # Losses, visualization
├── net/
│   ├── __init__.py
│   ├── slot_fewshot.py           # SMNet and SMNetLight models
│   ├── utils.py                  # Weight initialization
│   ├── backbone/
│   │   ├── convmixer.py          # ConvMixer encoder
│   │   ├── channel_attention.py  # SE-style attention
│   │   ├── pixel_mamba.py        # SS2D (4-way Mamba)
│   │   ├── slot_attention.py     # Slot Attention with learnable K
│   │   ├── slot_mamba.py         # Slot-level Mamba SSM
│   │   └── feature_extractor.py  # Unified pipeline
│   └── metrics/
│       ├── covariance.py         # SlotCovarianceBlock
│       ├── euclidean.py
│       └── cosine.py
├── checkpoints/                  # Saved models
├── results/                      # Results and plots
└── docs/                         # Documentation
```

## Model Variants

| Model | Hidden Dim | Expand | Mamba Layers | Parameters |
|-------|-----------|--------|--------------|------------|
| **SMNet** | 96 | 1 | 1 | ~500K |
| **SMNet-Light** | 48 | 1 | 1 | ~150K |

## Theory

### Why Shared Weights?

1. **Consistent Feature Space**: Support and query must be in same embedding space
2. **Meta-Learning**: Encoder learns transferable representations
3. **Fair Comparison**: Identical transformations for valid similarity

### Why Slot Covariance?

| Method | Captures | Limitation |
|--------|----------|------------|
| Averaging | First-order (mean) | Ignores variance |
| Cosine | Direction only | Ignores magnitude |
| **Slot Covariance** | Second-order (covariance) | Captures distribution shape |

### Prediction Formula

$$\hat{y} = \arg\max_{c} \sum_{k=1}^{K} w_k \cdot s_q^{(k)^\top} \Sigma_c^{-1} s_q^{(k)}$$

Where:
- $s_q^{(k)}$ = k-th query slot descriptor
- $w_k$ = slot existence weight
- $\Sigma_c$ = class covariance matrix

## Citation

If you use SMNet in your research, please cite:

```bibtex
@article{smnet2024,
  title={SMNet: Slot Mamba Network for Few-Shot Learning},
  author={Your Name},
  year={2024}
}
```

## License

MIT License
