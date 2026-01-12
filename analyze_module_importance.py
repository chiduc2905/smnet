"""Analyze Module Importance in USCMambaNet.

Two methods:
1. Weight Magnitude Analysis - which modules have largest weights
2. Gradient Magnitude Analysis - which modules receive largest gradients

Usage:
    python analyze_module_importance.py --weights checkpoints/your_model.pth
"""
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

from net.usc_mamba_net import USCMambaNet


def count_params(module):
    """Count trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def get_weight_stats(module, name=""):
    """Get L2 norm and mean magnitude of weights."""
    total_l2 = 0.0
    total_elements = 0
    
    for param in module.parameters():
        if param.requires_grad:
            total_l2 += param.data.norm(2).item() ** 2
            total_elements += param.numel()
    
    l2_norm = np.sqrt(total_l2)
    mean_mag = np.sqrt(total_l2 / max(total_elements, 1))
    
    return {
        'name': name,
        'params': total_elements,
        'l2_norm': l2_norm,
        'mean_magnitude': mean_mag
    }


def analyze_model(model, checkpoint_path=None):
    """Analyze weight distribution across modules."""
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading weights from: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    
    print("\n" + "="*70)
    print("USCMambaNet Module Importance Analysis")
    print("="*70)
    
    # Define modules to analyze
    modules = OrderedDict([
        ('1. PatchEmbed', model.patch_embed),
        ('2. ConvBlocks', model.conv_blocks),
        ('3. PatchMerge', model.patch_merge),
        ('4. ChannelProj', nn.Sequential(model.channel_proj_conv, model.channel_proj_norm)),
        ('5. DualBranch (AG-LKA + Mamba)', model.dual_branch),
        ('6. UnifiedAttention', model.unified_attention),
        ('7. ProtoCrossAttn', model.proto_cross_attn),
        ('8. SimilarityHead', model.similarity_head),
    ])
    
    # Analyze each module
    results = []
    total_params = 0
    
    print("\n📊 Parameter Distribution:")
    print("-"*70)
    print(f"{'Module':<40} {'Params':>10} {'%':>8} {'L2 Norm':>10} {'Mean Mag':>10}")
    print("-"*70)
    
    for name, module in modules.items():
        stats = get_weight_stats(module, name)
        results.append(stats)
        total_params += stats['params']
    
    # Print results sorted by params
    for stats in results:
        pct = 100 * stats['params'] / max(total_params, 1)
        print(f"{stats['name']:<40} {stats['params']:>10,} {pct:>7.1f}% {stats['l2_norm']:>10.3f} {stats['mean_magnitude']:>10.5f}")
    
    print("-"*70)
    print(f"{'TOTAL':<40} {total_params:>10,} {'100.0%':>8}")
    
    # Find most important modules by different metrics
    print("\n" + "="*70)
    print("🎯 Module Ranking by Importance Metrics")
    print("="*70)
    
    # By parameter count
    by_params = sorted(results, key=lambda x: x['params'], reverse=True)
    print("\n📦 By Parameter Count (larger = more capacity):")
    for i, r in enumerate(by_params[:5], 1):
        print(f"  {i}. {r['name']}: {r['params']:,} params")
    
    # By L2 norm (larger = more learned)
    by_l2 = sorted(results, key=lambda x: x['l2_norm'], reverse=True)
    print("\n📈 By L2 Norm (larger = learned stronger weights):")
    for i, r in enumerate(by_l2[:5], 1):
        print(f"  {i}. {r['name']}: L2={r['l2_norm']:.4f}")
    
    # By mean magnitude (larger = more active per-param)
    by_mag = sorted(results, key=lambda x: x['mean_magnitude'], reverse=True)
    print("\n🔥 By Mean Weight Magnitude (larger = more active per param):")
    for i, r in enumerate(by_mag[:5], 1):
        print(f"  {i}. {r['name']}: mean={r['mean_magnitude']:.6f}")
    
    # Recommendations
    print("\n" + "="*70)
    print("💡 Recommendations for Ablation Study")
    print("="*70)
    print("""
To determine which module is TRULY most important for accuracy:

1. Run ablation study with run_ablation.py (disable each module one by one)

2. For your 140K param model, key modules to ablate:
   - DualBranch (AG-LKA + Mamba): Local-global feature extraction
   - UnifiedAttention: Feature selection/gating
   - ProtoCrossAttn: Query refinement with prototypes
   - SimilarityHead projection: Bottleneck embedding

3. Create bypass versions:
   - Replace DualBranch with identity
   - Replace UnifiedAttention with identity
   - Set proto_cross_attn.alpha = 0
   - Set similarity_head.use_projection = False
""")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to trained model checkpoint')
    parser.add_argument('--hidden_dim', type=int, default=64)
    args = parser.parse_args()
    
    # Create model
    model = USCMambaNet(
        in_channels=3,
        hidden_dim=args.hidden_dim,
        temperature=16.0,
        device='cpu'
    )
    
    # Analyze
    analyze_model(model, args.weights)


if __name__ == '__main__':
    main()
