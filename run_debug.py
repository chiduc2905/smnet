#!/usr/bin/env python
"""Comprehensive Debug Script for USCMambaNet.

This script covers ALL debug levels A-F:
    A. Mục tiêu: Kiểm tra model có học không
    B. Level 1: Gradient norm, Feature stats
    C. Level 2: Overfit 1 episode
    D. Level 3: Isolate từng module
    E. Level 4: Logit stats, Label check
    F. Quick test sống/chết

Usage:
    # Level 1: Check gradient flow
    python run_debug.py --level 1
    
    # Level 2: Overfit 1 episode (CRITICAL)
    python run_debug.py --level 2
    
    # Level 3: Ablation (disable modules)
    python run_debug.py --level 3 --disable_attention
    python run_debug.py --level 3 --disable_asymmetric
    python run_debug.py --level 3 --disable_vss
    
    # Level 4: Logit analysis
    python run_debug.py --level 4
    
    # Full debug (all levels) - saves report to file
    python run_debug.py --level all --save_report
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
import io

from dataset import load_dataset
from dataloader.dataloader import FewshotDataset
from function.function import ContrastiveLoss, seed_func
from function.debug_utils import print_grad_norm, print_logit_stats, stat

# Model
from net.usc_mamba_net import USCMambaNet


# Global report buffer
REPORT_BUFFER = []


def log(msg=""):
    """Print and optionally save to report buffer."""
    print(msg)
    REPORT_BUFFER.append(msg)


def get_args():
    parser = argparse.ArgumentParser(description='USCMambaNet Debug Script')
    
    # Debug level
    parser.add_argument('--level', type=str, default='all',
                        choices=['1', '2', '3', '4', 'all'],
                        help='Debug level: 1=gradient, 2=overfit, 3=ablation, 4=logit, all=full')
    
    # Ablation flags (Level 3)
    parser.add_argument('--disable_attention', action='store_true',
                        help='Disable UnifiedSpatialChannelAttention')
    parser.add_argument('--disable_asymmetric', action='store_true',
                        help='Disable Asymmetric RF (keep AG-LKA only)')
    parser.add_argument('--disable_vss', action='store_true',
                        help='Disable VSS Block (keep Conv+LKA only)')
    
    # Basic config
    parser.add_argument('--dataset_path', type=str, default='/mnt/disk2/nhatnc/res/scalogram_fewshot/proposed_model/smnet/scalogram_official')
    parser.add_argument('--shot_num', type=int, default=5)
    parser.add_argument('--way_num', type=int, default=3)
    parser.add_argument('--query_num', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of iterations for overfit test')
    
    # Report
    parser.add_argument('--save_report', action='store_true',
                        help='Save debug report to file for AI analysis')
    
    return parser.parse_args()


class DebugUSCMambaNet(USCMambaNet):
    """USCMambaNet with debug mode and ablation flags."""
    
    def __init__(self, disable_attention=False, disable_asymmetric=False, 
                 disable_vss=False, print_stats=False, **kwargs):
        super().__init__(**kwargs)
        self.disable_attention = disable_attention
        self.disable_asymmetric = disable_asymmetric
        self.disable_vss = disable_vss
        self.print_stats = print_stats
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode with optional stats printing and ablation."""
        # Stage 1: Patch embedding
        f = self.patch_embed(x)
        if self.print_stats:
            stat(f, "1. PatchEmbed")
        
        # Stage 2: ConvBlocks
        f = self.conv_blocks(f)
        if self.print_stats:
            stat(f, "2. ConvBlocks")
        
        # Stage 3: Single PatchMerging
        f = self.patch_merge(f)
        if self.print_stats:
            stat(f, "3. PatchMerge")
        
        # Stage 4: Channel projection
        f_proj = self.channel_proj_conv(f)
        f_proj = self.channel_proj_norm(f_proj)
        f = f_proj
        if self.print_stats:
            stat(f, "4. ChannelProj")
        
        # Stage 5: DualBranchFusion (with ablation)
        if self.disable_vss:
            # Only use local branch (AG-LKA)
            f = self.dual_branch.local_branch(f)
            if self.print_stats:
                stat(f, "5. LocalOnly (no VSS)")
        else:
            f = self.dual_branch(f)
            if self.print_stats:
                stat(f, "5. DualBranch")
        
        # Stage 6: UnifiedAttention (with ablation)
        if self.disable_attention:
            # Skip attention, just pass through
            if self.print_stats:
                stat(f, "6. SKIPPED (no Attn)")
        else:
            f = self.unified_attention(f)
            if self.print_stats:
                stat(f, "6. UnifiedAttn")
        
        return f


def load_single_episode(args):
    """Load a single fixed episode for overfit testing."""
    print("\n📦 Loading dataset...")
    dataset = load_dataset(
        args.dataset_path,
        image_size=args.image_size
    )
    
    # Convert to torch tensors (FewshotDataset expects tensors)
    train_X = torch.from_numpy(dataset.X_train).float()
    train_y = torch.from_numpy(dataset.y_train).long()
    
    print(f"  Train: {train_X.shape}")
    
    # Create single episode
    seed_func(args.seed)
    fewshot_dataset = FewshotDataset(
        train_X, train_y,
        episode_num=1,
        way_num=args.way_num,
        shot_num=args.shot_num,
        query_num=args.query_num,
        seed=args.seed
    )
    
    from torch.utils.data import DataLoader
    loader = DataLoader(fewshot_dataset, batch_size=1, shuffle=False)
    
    # Get single episode
    # FewshotDataset returns: (query, q_labels, support, s_labels)
    for query, q_labels, support, s_labels in loader:
        return support, s_labels, query, q_labels


def debug_level_1(args):
    """Level 1: Check gradient flow and feature statistics."""
    print("\n" + "="*60)
    print("🔬 DEBUG LEVEL 1: Gradient Flow & Feature Stats")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model with stats printing
    model = DebugUSCMambaNet(
        print_stats=True,
        disable_attention=args.disable_attention,
        disable_asymmetric=args.disable_asymmetric,
        disable_vss=args.disable_vss
    ).to(device)
    
    criterion = ContrastiveLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Load single episode
    support, _, query, q_labels = load_single_episode(args)
    
    B = query.shape[0]
    C, H, W = query.shape[2], query.shape[3], query.shape[4]
    support = support.view(B, args.way_num, args.shot_num, C, H, W).to(device)
    query = query.to(device)
    targets = q_labels.view(-1).to(device)
    
    print("\n📊 Feature Statistics (single forward pass):")
    print("-"*60)
    
    # Forward
    model.train()
    optimizer.zero_grad()
    scores = model(query, support)
    loss = criterion(scores, targets)
    loss.backward()
    
    print("\n📈 Gradient Norms:")
    print_grad_norm(model, epoch=0, step=0, print_every=1)
    
    print("\n📉 Logit Stats:")
    print_logit_stats(scores, step=0, print_every=1)


def debug_level_2(args):
    """Level 2: Overfit 1 episode (CRITICAL TEST)."""
    print("\n" + "="*60)
    print("🔥 DEBUG LEVEL 2: Overfit 1 Episode (CRITICAL)")
    print("="*60)
    print(f"Target: acc → 100%, loss → 0 in {args.iterations} iterations")
    print("-"*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = DebugUSCMambaNet(
        print_stats=False,
        disable_attention=args.disable_attention,
        disable_asymmetric=args.disable_asymmetric,
        disable_vss=args.disable_vss
    ).to(device)
    
    criterion = ContrastiveLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Load single episode (FIXED - no re-sampling)
    support, _, query, q_labels = load_single_episode(args)
    
    B = query.shape[0]
    C, H, W = query.shape[2], query.shape[3], query.shape[4]
    support = support.view(B, args.way_num, args.shot_num, C, H, W).to(device)
    query = query.to(device)
    targets = q_labels.view(-1).to(device)
    
    print(f"\n📦 Episode: {args.way_num}-way {args.shot_num}-shot")
    print(f"   Support: {support.shape}, Query: {query.shape}, Targets: {targets.shape}")
    print(f"   Labels: {targets.tolist()}")
    
    # Overfit loop
    model.train()
    pbar = tqdm(range(args.iterations), desc="Overfitting")
    
    for i in pbar:
        optimizer.zero_grad()
        scores = model(query, support)
        loss = criterion(scores, targets)
        loss.backward()
        optimizer.step()
        
        # Accuracy
        preds = scores.argmax(dim=1)
        acc = (preds == targets).float().mean().item()
        
        pbar.set_postfix(loss=f'{loss.item():.4f}', acc=f'{acc*100:.1f}%')
        
        # Early success
        if acc == 1.0 and loss.item() < 0.01:
            print(f"\n✅ SUCCESS! Overfit achieved at iteration {i+1}")
            print(f"   Loss: {loss.item():.6f}, Acc: {acc*100:.1f}%")
            return True
    
    # Final check
    print(f"\n❌ FAILED to overfit after {args.iterations} iterations")
    print(f"   Final Loss: {loss.item():.4f}, Final Acc: {acc*100:.1f}%")
    print("\n⚠️ Possible issues:")
    print("   - Similarity / loss logic is WRONG")
    print("   - Label mapping mismatch")
    print("   - Feature collapse")
    return False


def debug_level_3(args):
    """Level 3: Ablation - disable modules one by one."""
    print("\n" + "="*60)
    print("🔧 DEBUG LEVEL 3: Module Ablation")
    print("="*60)
    
    ablation_name = []
    if args.disable_attention:
        ablation_name.append("NO ATTENTION")
    if args.disable_asymmetric:
        ablation_name.append("NO ASYMMETRIC")
    if args.disable_vss:
        ablation_name.append("NO VSS")
    
    if not ablation_name:
        print("⚠️ No ablation flags set. Use:")
        print("   --disable_attention")
        print("   --disable_asymmetric")
        print("   --disable_vss")
        return
    
    print(f"Ablation: {', '.join(ablation_name)}")
    print("-"*60)
    
    # Run overfit test with ablation
    debug_level_2(args)


def debug_level_4(args):
    """Level 4: Detailed logit analysis and label verification."""
    print("\n" + "="*60)
    print("📊 DEBUG LEVEL 4: Logit Analysis & Label Verification")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = DebugUSCMambaNet(print_stats=False).to(device)
    criterion = ContrastiveLoss().to(device)
    
    # Load episode
    support, s_labels, query, q_labels = load_single_episode(args)
    
    B = query.shape[0]
    C, H, W = query.shape[2], query.shape[3], query.shape[4]
    support = support.view(B, args.way_num, args.shot_num, C, H, W).to(device)
    query = query.to(device)
    targets = q_labels.view(-1).to(device)
    
    print("\n🏷️ Label Verification:")
    print(f"   Support labels: {s_labels.view(-1).tolist()}")
    print(f"   Query labels: {q_labels.view(-1).tolist()}")
    print(f"   Targets: {targets.tolist()}")
    print(f"   Unique classes: {sorted(targets.unique().tolist())}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        scores = model(query, support)
    
    print("\n📈 Logit Analysis (before softmax):")
    print(f"   Shape: {scores.shape}")
    print(f"   Min: {scores.min().item():.4f}")
    print(f"   Max: {scores.max().item():.4f}")
    print(f"   Mean: {scores.mean().item():.4f}")
    print(f"   Std: {scores.std().item():.4f}")
    
    print("\n📋 Per-query logits:")
    for i in range(min(5, scores.shape[0])):
        logits = scores[i].cpu().numpy()
        pred = logits.argmax()
        target = targets[i].item()
        status = "✅" if pred == target else "❌"
        print(f"   Query {i}: {logits} → pred={pred}, target={target} {status}")
    
    # Check for issues
    if scores.std().item() < 0.1:
        print("\n⚠️ WARNING: Logits too similar (std < 0.1)")
        print("   → Similarity is NOT discriminating between classes!")
    
    if abs(scores.max().item() - scores.min().item()) < 0.1:
        print("\n⚠️ WARNING: Logits flat (range < 0.1)")
        print("   → No class separation!")


def run_all_levels(args):
    """Run all debug levels."""
    print("\n" + "="*60)
    print("🔬 RUNNING ALL DEBUG LEVELS")
    print("="*60)
    
    debug_level_1(args)
    success = debug_level_2(args)
    debug_level_4(args)
    
    print("\n" + "="*60)
    print("📋 SUMMARY")
    print("="*60)
    
    if success:
        print("✅ Model CAN overfit → architecture logic is correct")
        print("   Next: Train with full data")
    else:
        print("❌ Model CANNOT overfit → CRITICAL BUG in architecture/loss")
        print("\n   Try these ablations:")
        print("   python run_debug.py --level 3 --disable_attention")
        print("   python run_debug.py --level 3 --disable_vss")
    
    return success


def save_debug_report(args, overfit_success):
    """Save structured debug report for AI analysis."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"debug_report_{timestamp}.md"
    
    # Get model info
    model = DebugUSCMambaNet(print_stats=False)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    report = f"""# USCMambaNet Debug Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Command:** `python run_debug.py --level {args.level} --save_report`

---

## 1. Model Configuration

| Parameter | Value |
|-----------|-------|
| Model | USCMambaNet |
| Input | RGB {args.image_size}×{args.image_size} |
| Total Params | {total_params:,} |
| Trainable Params | {trainable_params:,} |
| Few-shot | {args.way_num}-way {args.shot_num}-shot |
| Query/Class | {args.query_num} |
| Learning Rate | {args.lr} |
| Seed | {args.seed} |

---

## 2. Current Similarity Configuration (DEBUG MODE)

| Setting | Value | Note |
|---------|-------|------|
| **similarity_mode** | `allpairs` | FIX 1: soft alignment |
| **aggregation** | `mean` | 100% patches |
| **topk_ratio** | `1.0` | all patches |
| **temperature** | `1.0` | softer scores |
| **logit_scale** | `10.0` (learnable) | FIX 2: boost CE gradient |

### AllPairs Similarity Formula:
```
S_ij = cos(q_i, s_j)           # all query-support patch pairs
score_i = mean_k topk(S_ij)    # top-k per query patch
final = mean_i(score_i) × logit_scale
```

---

## 3. Architecture Summary

```
Input (B, 3, 64, 64)
    ↓
PatchEmbed2D → (B, 32, 64, 64)
    ↓
ConvBlock ×2 → (B, 64, 64, 64)
    ↓
PatchMerging2D → (B, 128, 32, 32)
    ↓
ChannelProj → (B, 64, 32, 32)
    ↓
DualBranchFusion → (B, 64, 32, 32)
  ├── LocalAGLKABranch (DWConv5×5 + DWConv7×7_d2 + Asym1×9 + Asym7×1)
  └── VSSBlock (4-way SS2D Mamba)
    ↓
UnifiedSpatialChannelAttention → (B, 64, 32, 32)
  - RESIDUAL GATING: X + X⊙A_ch + X⊙A_sp
    ↓
AllPairsSimilarity → (NQ, Way)
  - all-pairs cosine + top-k + logit_scale
```

---

## 4. Debug Results

### Level 2: Overfit Test (CRITICAL)

| Metric | Result |
|--------|--------|
| Overfit Success | {'✅ YES' if overfit_success else '❌ NO'} |
| Iterations | {args.iterations} |
| Target | acc=100%, loss→0 |

"""
    
    if not overfit_success:
        report += """
### ⚠️ DIAGNOSIS: Model CANNOT Overfit

**Possible Issues:**
1. **Similarity Logic Bug** - SimplePatchSimilarity không phân biệt được class
2. **Feature Collapse** - std < 0.3 ở một stage nào đó
3. **Gradient Vanishing** - grad ≈ 0 ở nhiều layer
4. **Label Mismatch** - support/query labels không khớp

**Recommended Actions:**
```bash
# Thử tắt UnifiedAttention
python run_debug.py --level 3 --disable_attention

# Thử tắt VSS (chỉ dùng Conv+LKA)
python run_debug.py --level 3 --disable_vss
```
"""
    else:
        report += """
### ✅ DIAGNOSIS: Model CAN Overfit

Model có khả năng học. Nếu training vẫn không tốt:
1. Kiểm tra data augmentation
2. Tăng epochs
3. Tinh chỉnh hyperparameters (lr, weight_decay)
"""

    report += """
---

## 4. Key Hyperparameters to Check

| Setting | Current | Recommended Range |
|---------|---------|-------------------|
| aggregation | mean | mean (for debug) |
| topk_ratio | 1.0 | 0.2-1.0 |
| temperature | 1.0 | 0.2-1.0 |
| alpha_residual | 0.1 | 0.05-0.2 |

---

## 5. Console Output Log

```
"""
    report += "\n".join(REPORT_BUFFER[-100:])  # Last 100 lines
    report += """
```

---

## 6. Next Steps for AI Analysis

If providing this report to an AI:
1. Share the full report
2. Share relevant code files if needed:
   - `net/usc_mamba_net.py`
   - `net/backbone/dual_branch_fusion.py`
   - `net/backbone/simple_similarity.py`
3. Ask specific questions about the debug results
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📄 Report saved to: {report_path}")
    return report_path


if __name__ == '__main__':
    args = get_args()
    seed_func(args.seed)
    
    overfit_success = False
    
    if args.level == '1':
        debug_level_1(args)
    elif args.level == '2':
        overfit_success = debug_level_2(args)
    elif args.level == '3':
        debug_level_3(args)
    elif args.level == '4':
        debug_level_4(args)
    else:
        overfit_success = run_all_levels(args)
    
    # Save report if requested
    if args.save_report:
        save_debug_report(args, overfit_success)

