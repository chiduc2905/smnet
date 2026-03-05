"""USCMambaNet (Unified Spatial-Channel Mamba Network) - Training and Evaluation.

This script trains and evaluates USCMambaNet which uses:
- PatchEmbed + PatchMerging for hierarchical feature extraction
- DualBranchFusion (AG-LKA + SS2D) for local-global features
- UnifiedSpatialChannelAttention for feature selection
- SimplePatchSimilarity for non-learnable cosine matching
"""
import os
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter, defaultdict
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import wandb

# FLOPs calculation
try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("Warning: thop not installed. Run 'pip install thop' for FLOPs calculation.")

from dataset import load_dataset
from dataloader.dataloader import FewshotDataset
from function.function import (
    seed_func,
    plot_confusion_matrix, plot_tsne, plot_umap, plot_training_curves
)
from function.debug_utils import print_grad_norm, print_logit_stats, set_debug_mode, is_debug_mode

# Model
from net.usc_mamba_net import USCMambaNet


# =============================================================================
# Configuration
# =============================================================================

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='USCMambaNet Few-shot Learning')
    
    # Paths
    parser.add_argument('--dataset_path', type=str,
                        default='/mnt/disk2/nhatnc/res/scalogram_fewshot/proposed_model/smnet/scalogram_27_1')
    parser.add_argument('--path_weights', type=str, default='checkpoints/')
    parser.add_argument('--path_results', type=str, default='results/')
    parser.add_argument('--weights', type=str, default=None, help='Checkpoint for testing')
    parser.add_argument('--dataset_name', type=str, default='knee_aug_split',
                        help='Dataset name for checkpoint naming')
    
    # Model
    parser.add_argument('--model', type=str, default='uscmamba', 
                        choices=['uscmamba'])
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension for feature extractor')
    parser.add_argument('--d_state', type=int, default=8,
                        help='Mamba/VSS state dimension')
    parser.add_argument('--global_expand', type=int, default=2,
                        help='Expansion factor for global VSS branch')
    parser.add_argument('--proto_pool_size', type=int, default=12,
                        help='Prototype spatial pooling size')
    parser.add_argument('--num_prototypes', type=int, default=2,
                        help='Number of prototypes per class in cross-attention')
    parser.add_argument('--detach_prototypes', action='store_true',
                        help='Detach prototype maps in cross-attention (default: False)')
    parser.add_argument('--similarity_proj_dim', type=int, default=None,
                        help='Projection dim for similarity head (default: hidden_dim)')
    
    # Few-shot settings
    parser.add_argument('--way_num', type=int, default=4)
    parser.add_argument('--shot_num', type=int, default=1)
    parser.add_argument('--query_num', type=int, default=None,
                        help='Legacy: set same queries per class for train/val/test')
    parser.add_argument('--query_num_train', type=int, default=1, help='Queries per class for training episodes')
    parser.add_argument('--query_num_val', type=int, default=1, help='Queries per class for validation episodes')
    parser.add_argument('--query_num_test', type=int, default=1, help='Queries per class for test episodes')
    parser.add_argument('--selected_classes', type=str, default=None,
                        help='Comma-separated class indices to use (e.g. "0,1" for first 2 classes). If None, use all classes.')
    parser.add_argument('--image_size', type=int, default=64,
                        help='Input image size (default: 64)')
    
    # Ablation control (for experiments from run_all_experiments.py)
    parser.add_argument('--dualpath_mode', type=str, default='both',
                        choices=['local_only', 'global_only', 'both'],
                        help='DualPath mode: local_only, global_only, or both (default: both)')
    parser.add_argument('--use_unified_attention', type=str, default='true',
                        choices=['true', 'false'],
                        help='Use Unified Spatial-Channel Attention (default: true)')
    parser.add_argument('--use_cross_attention', type=str, default='true',
                        choices=['true', 'false'],
                        help='Use Prototype Cross-Attention (default: true)')
    parser.add_argument('--use_ms_global', type=str, default='true',
                        choices=['true', 'false'],
                        help='Enable multi-scale shared-weight global branch')
    parser.add_argument('--ms_downsample', type=int, default=2,
                        help='Downsample ratio for multi-scale global branch')
    parser.add_argument('--atrous_rate', type=int, default=2,
                        help='Dilation rate for atrous companion branch')
    parser.add_argument('--use_late_attention', type=str, default='true',
                        choices=['true', 'false'],
                        help='Enable late single-head attention bridge')
    parser.add_argument('--late_attn_window', type=int, default=4,
                        help='Window size for late single-head attention')
    parser.add_argument('--late_attn_dropout', type=float, default=0.0,
                        help='Attention dropout for late bridge')
    parser.add_argument('--use_axis_proto', type=str, default='true',
                        choices=['true', 'false'],
                        help='Enable dual-axis prototype tokenization')
    parser.add_argument('--axis_proto_pool', type=str, default='mean',
                        choices=['mean', 'max'],
                        help='Pooling mode for axis prototype tokens')
    parser.add_argument('--axis_proto_mix_init', type=str, default='1.0,0.5,0.5',
                        help='Initial mix logits for [full,time,freq] proto branches')
    
    # Training
    parser.add_argument('--training_samples', type=int, default=None,
                        help='Total training samples (must be divisible by way_num, e.g. 32=8/class for 4-way)')
    parser.add_argument('--episode_num_train', type=int, default=200)
    parser.add_argument('--episode_num_val', type=int, default=300)
    parser.add_argument('--episode_num_test', type=int, default=300)
    parser.add_argument('--num_epochs', type=int, default=100)

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3, help='Base learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-5, help='Min LR for cosine')
    parser.add_argument('--start_lr', type=float, default=1e-5, help='Start LR for warmup')
    parser.add_argument('--warmup_iters', type=int, default=500, help='Warmup iterations')
    parser.add_argument('--temperature', type=float, default=16.0,
                        help='Cosine similarity temperature τ (recommended: 16-20)')
    parser.add_argument('--beta_maha', type=float, default=0.25,
                        help='UAPS variance-aware penalty weight')
    parser.add_argument('--uaps_eps', type=float, default=1e-4,
                        help='UAPS numerical epsilon')
    parser.add_argument('--cross_attn_alpha', type=float, default=0.3,
                        help='Prototype Cross-Attention residual weight (0.1-0.5)')
    parser.add_argument('--use_pair_expert', type=str, default='false',
                        choices=['true', 'false'],
                        help='Use Pair Expert Correction head')
    parser.add_argument('--lambda_pair_expert', type=float, default=0.0,
                        help='Weight for pair expert BCE loss')
    parser.add_argument('--lambda_pair', type=float, default=0.0,
                        help='Weight for pair-adaptive hinge margin loss')
    parser.add_argument('--pair_margin', type=float, default=0.20,
                        help='Margin for pair-adaptive hinge loss')
    parser.add_argument('--class_margins', type=str, default='0.45,0.35,0.30,0.45',
                        help='Comma-separated per-class margins for ArcFace/CosFace')
    parser.add_argument('--hard_mining_ratio', type=float, default=0.0,
                        help='Ratio of hard-sample injection into training episodes')
    parser.add_argument('--delta_lambda', type=float, default=0.35,
                        help='Weight for relation delta correction (recommended: 0.3-0.4)')
    parser.add_argument('--no_projection', action='store_true',
                        help='Disable embedding projection in similarity head (for debugging)')
    parser.add_argument('--grad_clip', type=float, default=2.0,
                        help='Gradient clipping max norm')
    parser.add_argument('--eta_min', type=float, default=1e-5,
                        help='Min LR for CosineAnnealingLR')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--seed', type=int, default=42)
    
    # Loss
    parser.add_argument('--lambda_center', type=float, default=0.0, 
                        help='Weight for Center Loss (default: 0.01)')
    parser.add_argument('--margin_type', type=str, default='none',
                        choices=['none', 'cosface', 'arcface'],
                        help='Margin loss type: none (CE only), cosface, or arcface')
    parser.add_argument('--margin', type=float, default=0.3,
                        help='Margin value for CosFace/ArcFace (default: 0.3)')
    parser.add_argument('--margin_scale', type=float, default=20.0,
                        help='Scale factor s for margin loss (default: 20.0)')
    parser.add_argument('--lambda_margin', type=float, default=0.0,
                        help='Weight for margin loss regularizer (recommended: 0.05-0.2)')
    
    # Debug
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode: print gradients, feature stats, logits')
    # Mode
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--save_misclf_report', action='store_true',
                        help='Save per-file misclassification report from test episodes')
    parser.add_argument('--misclf_topk', type=int, default=30,
                        help='Top-K most frequently misclassified files to save')
    
    # WandB
    parser.add_argument('--project', type=str, default='uscmamba',
                        help='WandB project name')
    
    return parser.parse_args()


def get_model(args):
    """Initialize model based on args."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert string args to boolean for ablation flags
    use_unified = args.use_unified_attention.lower() == 'true'
    use_cross = args.use_cross_attention.lower() == 'true'
    use_pair_expert = args.use_pair_expert.lower() == 'true'
    use_ms_global = args.use_ms_global.lower() == 'true'
    use_late_attention = args.use_late_attention.lower() == 'true'
    use_axis_proto = args.use_axis_proto.lower() == 'true'

    axis_mix_parts = [p.strip() for p in args.axis_proto_mix_init.split(',')]
    if len(axis_mix_parts) != 3:
        raise ValueError("--axis_proto_mix_init must have exactly 3 comma-separated values")
    axis_proto_mix_init = tuple(float(v) for v in axis_mix_parts)
    
    model = USCMambaNet(
        in_channels=3,  # RGB input
        hidden_dim=args.hidden_dim,
        d_state=args.d_state,
        global_expand=args.global_expand,
        temperature=args.temperature,
        beta_maha=args.beta_maha,
        uaps_eps=args.uaps_eps,
        cross_attn_alpha=args.cross_attn_alpha,
        proto_pool_size=args.proto_pool_size,
        num_prototypes=args.num_prototypes,
        detach_prototypes=args.detach_prototypes,
        use_axis_proto=use_axis_proto,
        axis_proto_pool=args.axis_proto_pool,
        axis_proto_mix_init=axis_proto_mix_init,
        use_late_attention=use_late_attention,
        late_attn_window=args.late_attn_window,
        late_attn_dropout=args.late_attn_dropout,
        similarity_proj_dim=args.similarity_proj_dim,
        delta_lambda=args.delta_lambda,
        way_num=args.way_num,
        use_pair_expert=use_pair_expert,
        use_ms_global=use_ms_global,
        ms_downsample=args.ms_downsample,
        atrous_rate=args.atrous_rate,
        use_projection=not args.no_projection,
        dualpath_mode=args.dualpath_mode,
        use_unified_attention=use_unified,
        use_cross_attention=use_cross,
        device=str(device)
    )
    
    # Print ablation config
    print(f"\nModel Config:")
    print(f"  dualpath_mode: {args.dualpath_mode}")
    print(f"  use_unified_attention: {use_unified}")
    print(f"  use_cross_attention: {use_cross}")
    print(f"  use_ms_global: {use_ms_global} (downsample={args.ms_downsample}, atrous={args.atrous_rate})")
    print(f"  use_late_attention: {use_late_attention} (window={args.late_attn_window})")
    print(f"  use_axis_proto: {use_axis_proto} ({args.axis_proto_pool}, mix={axis_proto_mix_init})")
    print(f"  use_pair_expert: {use_pair_expert}")
    
    return model.to(device)


# =============================================================================
# Training
# =============================================================================


def train_loop(net, train_X, train_y, val_X, val_y, args):
    """Train with pure CE objective (architecture-focused protocol)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.eta_min,
    )

    history = {
        "train_acc": [],
        "val_acc": [],
        "train_loss": [],
        "val_loss": [],
    }
    best_acc = 0.0

    for epoch in range(1, args.num_epochs + 1):
        train_seed = args.seed + epoch
        train_ds = FewshotDataset(
            train_X,
            train_y,
            args.episode_num_train,
            args.way_num,
            args.shot_num,
            args.query_num_train,
            seed=train_seed,
        )
        train_gen = torch.Generator()
        train_gen.manual_seed(train_seed)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            generator=train_gen,
        )

        net.train()
        total_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}")
        for step, (query, q_labels, support, s_labels) in enumerate(pbar):
            optimizer.zero_grad()

            B = query.shape[0]
            C, H, W = query.shape[2], query.shape[3], query.shape[4]

            support = support.view(B, args.way_num, args.shot_num, C, H, W).to(device)
            query = query.to(device)
            targets = q_labels.view(-1).to(device)

            scores = net(query, support)
            loss = F.cross_entropy(scores, targets)

            with torch.no_grad():
                preds = scores.argmax(dim=1)
                train_correct += (preds == targets).sum().item()
                train_total += targets.size(0)

            loss.backward()

            if args.debug and step == 0:
                print_grad_norm(net, epoch, step, print_every=1)
                print_logit_stats(scores, step, print_every=1)

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)

            optimizer.step()
            total_loss += loss.item()
            current_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}")

        scheduler.step()
        train_acc = train_correct / train_total if train_total > 0 else 0.0

        val_seed = args.seed + epoch
        val_ds = FewshotDataset(
            val_X,
            val_y,
            args.episode_num_val,
            args.way_num,
            args.shot_num,
            args.query_num_val,
            seed=val_seed,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            worker_init_fn=lambda w: seed_func(val_seed + w),
        )

        val_acc, val_loss = evaluate(net, val_loader, args)
        avg_loss = total_loss / len(train_loader)

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_loss"].append(avg_loss)
        history["val_loss"].append(val_loss if val_loss else 0.0)

        train_val_gap = train_acc - val_acc
        print(
            f"Epoch {epoch}: Loss={avg_loss:.4f}, Train={train_acc:.4f}, "
            f"Val={val_acc:.4f} (gap={train_val_gap:+.4f})"
        )

        wandb.log(
            {
                "epoch": epoch,
                "loss/train": avg_loss,
                "loss/val": val_loss,
                "accuracy/train": train_acc,
                "accuracy/val": val_acc,
                "train_val_gap": train_val_gap,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        if val_acc > best_acc:
            best_acc = val_acc
            print(f"  → New best val: {val_acc:.4f}")
            wandb.run.summary["best_val_acc"] = best_acc

            samples_suffix = f"{args.training_samples}samples" if args.training_samples else "all"
            final_model_filename = f"{args.dataset_name}_{args.model}_{samples_suffix}_{args.shot_num}shot_final.pth"
            final_path = os.path.join(args.path_weights, final_model_filename)
            torch.save(net.state_dict(), final_path)
            print(f"  Saved best model to {final_path}")

    samples_str = f"{args.training_samples}samples" if args.training_samples else "allsamples"
    curves_path = os.path.join(
        args.path_results,
        f"training_{args.dataset_name}_{args.model}_{samples_str}_{args.shot_num}shot",
    )
    plot_training_curves(history, curves_path)

    if os.path.exists(f"{curves_path}_curves.png"):
        wandb.log({"training_curves": wandb.Image(f"{curves_path}_curves.png")})

    print(f"Best Validation Accuracy: {best_acc:.4f}")
    return best_acc, history


def evaluate(net, loader, args):
    """Compute accuracy and loss on loader."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.eval()
    correct, total = 0, 0
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for query, q_labels, support, s_labels in loader:
            B = query.shape[0]
            C, H, W = query.shape[2], query.shape[3], query.shape[4]
            
            shot_num = support.shape[1] // args.way_num
            
            support = support.view(B, args.way_num, shot_num, C, H, W).to(device)
            query = query.to(device)
            targets = q_labels.view(-1).to(device)
            
            scores = net(query, support)
            preds = scores.argmax(dim=1)
            
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
            # Use F.cross_entropy directly
            loss = F.cross_entropy(scores, targets)
            total_loss += loss.item()
            num_batches += 1
    
    acc = correct / total if total > 0 else 0
    avg_loss = total_loss / num_batches if num_batches > 0 else None
    
    return acc, avg_loss


# =============================================================================
# Testing
# =============================================================================

def calculate_p_value(acc, baseline, n):
    """Z-test for proportion significance."""
    from scipy.stats import norm
    if n <= 0:
        return 1.0
    z = (acc - baseline) / np.sqrt(baseline * (1 - baseline) / n)
    return 2 * norm.sf(abs(z))


def test_final(net, loader, args, test_X=None, test_y=None, test_file_paths=None):
    """Final evaluation with detailed metrics.
    
    Args:
        test_X: Full test set for visualization (to avoid duplicate samples in t-SNE/UMAP)
        test_y: Full test labels
    """
    import time
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_episodes = len(loader)
    
    print(f"\n{'='*60}")
    print(f"Final Test: USCMambaNet | {args.dataset_name} | {args.shot_num}-shot")
    print(f"{num_episodes} episodes × {args.way_num} classes × {args.query_num_test} query")
    print('='*60)
    
    net.eval()
    all_preds, all_targets = [], []
    episode_accuracies = []
    episode_times = []
    query_seen = Counter()
    query_mis = Counter()
    query_pair = defaultdict(Counter)
    
    # ====================================================================
    # Accuracy metrics: Use episodes (this is correct for few-shot eval)
    # ====================================================================
    with torch.no_grad():
        for batch in tqdm(loader, desc='Testing'):
            if len(batch) == 6:
                query, q_labels, support, s_labels, q_indices, _ = batch
                q_indices_np = q_indices.view(-1).cpu().numpy()
            else:
                query, q_labels, support, s_labels = batch
                q_indices_np = None

            start_time = time.perf_counter()
            
            B, NQ, C, H, W = query.shape
            
            support = support.view(B, args.way_num, args.shot_num, C, H, W).to(device)
            query = query.to(device)
            targets = q_labels.view(-1).to(device)
            
            scores = net(query, support)
            preds = scores.argmax(dim=1)
            
            end_time = time.perf_counter()
            episode_time_ms = (end_time - start_time) * 1000
            episode_times.append(episode_time_ms)
            
            episode_correct = (preds == targets).float().mean().item()
            episode_accuracies.append(episode_correct)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            if q_indices_np is not None:
                preds_np = preds.cpu().numpy()
                targets_np = targets.cpu().numpy()
                for idx_i, true_i, pred_i in zip(q_indices_np, targets_np, preds_np):
                    idx_i = int(idx_i)
                    true_i = int(true_i)
                    pred_i = int(pred_i)
                    query_seen[idx_i] += 1
                    if pred_i != true_i:
                        query_mis[idx_i] += 1
                        query_pair[idx_i][(true_i, pred_i)] += 1
    
    # Metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    episode_accuracies = np.array(episode_accuracies)
    episode_times = np.array(episode_times)
    
    acc_mean = episode_accuracies.mean()
    acc_std = episode_accuracies.std()
    acc_worst = episode_accuracies.min()
    acc_best = episode_accuracies.max()
    
    # 95% Confidence Interval: Mean ± 1.96 * (std / sqrt(n))
    n_episodes = len(episode_accuracies)
    acc_ci95 = 1.96 * acc_std / np.sqrt(n_episodes)
    
    time_mean = episode_times.mean()
    time_std = episode_times.std()
    
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, 
        labels=list(range(args.way_num)),
        average='macro', 
        zero_division=0
    )
    p_val = calculate_p_value(acc_mean, 1.0/args.way_num, len(all_targets))
    
    # Print results
    print(f"\n{'='*60}")
    print("ACCURACY METRICS")
    print('='*60)
    print(f"  Mean Accuracy : {acc_mean*100:.2f} ± {acc_ci95*100:.2f}% (95% CI)")
    print(f"  Std Deviation : {acc_std*100:.2f}%")
    print(f"  Worst-case    : {acc_worst*100:.2f}%")
    print(f"  Best-case     : {acc_best*100:.2f}%")
    print(f"  Precision     : {prec:.4f}")
    print(f"  Recall        : {rec:.4f}")
    print(f"  F1-Score      : {f1:.4f}")
    print(f"  p-value       : {p_val:.2e}")
    print(f"\nInference Time  : {time_mean:.2f} ± {time_std:.2f} ms/episode")
    
    # Log to WandB
    wandb.log({
        "test_accuracy_mean": acc_mean,
        "test_accuracy_std": acc_std,
        "test_accuracy_ci95": acc_ci95,
        "test_accuracy_worst": acc_worst,
        "test_accuracy_best": acc_best,
        "test_precision": prec,
        "test_recall": rec,
        "test_f1": f1,
        "inference_time_mean_ms": time_mean,
    })
    
    wandb.run.summary["test_accuracy_mean"] = acc_mean
    wandb.run.summary["test_accuracy_ci95"] = acc_ci95
    
    # Plots
    samples_str = f"_{args.training_samples}samples" if args.training_samples else "_allsamples"
    
    cm_base = os.path.join(args.path_results, 
                           f"confusion_matrix_{args.dataset_name}_{args.model}_{samples_str.strip('_')}_{args.shot_num}shot")
    plot_confusion_matrix(all_targets, all_preds, args.way_num, cm_base, class_names=args.class_names)
    
    if os.path.exists(f"{cm_base}_2col.png"):
        wandb.log({"confusion_matrix": wandb.Image(f"{cm_base}_2col.png")})

    if args.save_misclf_report and query_seen:
        pair_totals = Counter()
        true_totals = Counter()
        for t, p in zip(all_targets.tolist(), all_preds.tolist()):
            true_totals[int(t)] += 1
            if int(t) != int(p):
                pair_totals[(int(t), int(p))] += 1

        pair_path = os.path.join(
            args.path_results,
            f"misclass_pairs_{args.dataset_name}_{args.model}_{samples_str.strip('_')}_{args.shot_num}shot.txt",
        )
        with open(pair_path, "w") as f:
            f.write("Most common confusion pairs (True -> Pred)\n")
            f.write("-" * 70 + "\n")
            for (t, p), cnt in pair_totals.most_common():
                denom = max(1, true_totals[t])
                rate = cnt / denom
                t_name = args.class_names[t] if t < len(args.class_names) else f"Class{t}"
                p_name = args.class_names[p] if p < len(args.class_names) else f"Class{p}"
                f.write(f"{t_name} -> {p_name}: {cnt} ({rate:.2%} of true {t_name})\n")

        rows = []
        for idx_i, seen in query_seen.items():
            mis = query_mis.get(idx_i, 0)
            if mis <= 0:
                continue
            top_pair, top_pair_cnt = query_pair[idx_i].most_common(1)[0]
            true_i, pred_i = top_pair
            true_name = args.class_names[true_i] if true_i < len(args.class_names) else f"Class{true_i}"
            pred_name = args.class_names[pred_i] if pred_i < len(args.class_names) else f"Class{pred_i}"
            file_path = f"index_{idx_i}"
            if test_file_paths is not None and 0 <= idx_i < len(test_file_paths):
                file_path = test_file_paths[idx_i]

            rows.append({
                "file_index": idx_i,
                "file_path": file_path,
                "mis_count": mis,
                "seen_count": seen,
                "mis_rate": mis / max(1, seen),
                "top_confusion_true": true_name,
                "top_confusion_pred": pred_name,
                "top_confusion_count": top_pair_cnt,
            })

        rows.sort(key=lambda x: (-x["mis_count"], -x["mis_rate"], x["file_index"]))
        if args.misclf_topk > 0:
            rows = rows[:args.misclf_topk]

        mis_path = os.path.join(
            args.path_results,
            f"misclassified_files_{args.dataset_name}_{args.model}_{samples_str.strip('_')}_{args.shot_num}shot.csv",
        )
        with open(mis_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "file_index",
                    "file_path",
                    "mis_count",
                    "seen_count",
                    "mis_rate",
                    "top_confusion_true",
                    "top_confusion_pred",
                    "top_confusion_count",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        print(f"Saved confusion-pair summary: {pair_path}")
        print(f"Saved per-file misclassification report: {mis_path}")
    
    # ====================================================================
    # t-SNE/UMAP: Use UNIQUE test samples (not episode duplicates!)
    # ====================================================================
    # Problem with old approach:
    # - 300 episodes × 15 queries = 4500 samples
    # - But test set only has a finite number of unique samples per class
    # - Each sample appears ~30 times → creates artificial "lumps" in t-SNE
    #
    # Solution: Extract features from full test set once (150 unique samples)
    # ====================================================================
    
    if test_X is not None and test_y is not None:
        print(f"\n{'='*60}")
        print(f"Extracting features for t-SNE/UMAP from UNIQUE test samples")
        print(f"Test set size: {len(test_X)} samples ({len(test_X)//args.way_num}/class)")
        print('='*60)
        
        with torch.no_grad():
            # Extract features from unique test samples
            test_X_device = test_X.to(device)
            test_y_np = test_y.cpu().numpy()
            
            # Batch processing for memory efficiency
            batch_size = 32
            all_features = []
            
            for i in range(0, len(test_X), batch_size):
                batch_X = test_X_device[i:i+batch_size]
                
                # Extract backbone features (same as in episodes)
                features = net.encode(batch_X)  # (N, hidden_dim, H', W')
                feat_backbone = features.mean(dim=(2, 3))  # GAP: (N, hidden_dim)
                feat_backbone = F.normalize(feat_backbone, p=2, dim=-1)  # L2 normalize
                
                all_features.append(feat_backbone.cpu().numpy())
            
            features = np.vstack(all_features)
            
            print(f"Extracted {len(features)} unique features (shape: {features.shape})")
            
            # 1. t-SNE
            tsne_path = os.path.join(args.path_results, 
                                         f"tsne_{args.dataset_name}_{args.model}_{samples_str.strip('_')}_{args.shot_num}shot")
            plot_tsne(features, test_y_np, args.way_num, tsne_path, class_names=args.class_names)
            
            if os.path.exists(f"{tsne_path}_tsne.png"):
                wandb.log({"tsne_plot": wandb.Image(f"{tsne_path}_tsne.png")})
    
            # 2. UMAP
            umap_path = os.path.join(args.path_results, 
                                     f"umap_{args.dataset_name}_{args.model}_{samples_str.strip('_')}_{args.shot_num}shot")
            plot_umap(features, test_y_np, args.way_num, umap_path, class_names=args.class_names)
            
            if os.path.exists(f"{umap_path}_umap.png"):
                wandb.log({"umap_plot": wandb.Image(f"{umap_path}_umap.png")})
    else:
        print("\n⚠️  Warning: test_X/test_y not provided, skipping t-SNE/UMAP (would have duplicates)")
    
    
    # Save results to file
    txt_path = os.path.join(args.path_results, 
                            f"results_{args.dataset_name}_{args.model}_{samples_str.strip('_')}_{args.shot_num}shot.txt")
    with open(txt_path, 'w') as f:
        f.write(f"Model: SMNet ({args.model})\n")
        f.write(f"Dataset: {args.dataset_name}\n")
        f.write(f"Shot: {args.shot_num}\n")
        f.write(f"Training Samples: {args.training_samples if args.training_samples else 'All'}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy : {acc_mean:.4f} ± {acc_std:.4f}\n")
        f.write(f"Worst-case : {acc_worst:.4f}\n")
        f.write(f"Best-case : {acc_best:.4f}\n")
        f.write(f"Precision : {prec:.4f}\n")
        f.write(f"Recall : {rec:.4f}\n")
        f.write(f"F1-Score : {f1:.4f}\n")
        f.write(f"Inference Time: {time_mean:.2f} ± {time_std:.2f} ms/episode\n")
    print(f"Results saved to {txt_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    args = get_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Legacy compatibility: one query_num for all splits
    if args.query_num is not None:
        args.query_num_train = args.query_num
        args.query_num_val = args.query_num
        args.query_num_test = args.query_num
    
    print(f"\n{'='*60}")
    print("USCMambaNet: Unified Spatial-Channel Mamba Network")
    print('='*60)
    print(f"Config: {args.model} | {args.shot_num}-shot | {args.num_epochs} epochs | Device: {args.device}")
    print(f"Architecture: PatchEmbed → ConvBlocks → PatchMerge → DualBranch(AG-LKA+SS2D) → UnifiedAttn → SimpleSimilarity")
    print(f"Dataset: {args.dataset_path}")
    
    # Initialize WandB
    samples_str = f"{args.training_samples}samples" if args.training_samples else "all"
    run_name = f"uscmamba_{args.dataset_name}_{samples_str}_{args.shot_num}shot"
    
    config = vars(args).copy()
    config['architecture'] = 'USCMambaNet (Unified Spatial-Channel Mamba Network)'
    
    wandb.init(project=args.project, config=config, name=run_name, group=f"uscmamba_{args.dataset_name}", job_type=args.mode)
    
    # Set seed BEFORE anything else for full reproducibility
    seed_func(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.makedirs(args.path_weights, exist_ok=True)
    os.makedirs(args.path_results, exist_ok=True)
    
    # Load dataset
    dataset = load_dataset(args.dataset_path, image_size=args.image_size)
    
    def to_tensor(X, y):
        X = torch.from_numpy(X.astype(np.float32))
        y = torch.from_numpy(y).long()
        return X, y
    
    train_X, train_y = to_tensor(dataset.X_train, dataset.y_train)
    val_X, val_y = to_tensor(dataset.X_val, dataset.y_val)
    test_X, test_y = to_tensor(dataset.X_test, dataset.y_test)
    train_file_paths = [p for p, _ in getattr(dataset, "train_files", [])] if hasattr(dataset, "train_files") else None
    val_file_paths = [p for p, _ in getattr(dataset, "val_files", [])] if hasattr(dataset, "val_files") else None
    test_file_paths = [p for p, _ in getattr(dataset, "test_files", [])] if hasattr(dataset, "test_files") else None
    
    # ============================================================
    # Filter to selected classes if specified
    # ============================================================
    # Default class names from dataset metadata (canonical 4-class split)
    pretty_map = {
        'surface': 'Surface',
        'internal': 'Internal',
        'corona': 'Corona',
        'notpd': 'NotPD',
        'nopd': 'NotPD',
    }
    dataset_classes = list(getattr(dataset, 'classes', []))
    if dataset_classes:
        ALL_CLASS_NAMES = [pretty_map.get(c.lower(), c) for c in dataset_classes]
    else:
        n_classes = int(len(torch.unique(train_y)))
        ALL_CLASS_NAMES = [f'Class{i}' for i in range(n_classes)]
    
    selected = None
    if args.selected_classes:
        selected = [int(c.strip()) for c in args.selected_classes.split(',')]
        if any(c < 0 or c >= len(ALL_CLASS_NAMES) for c in selected):
            raise ValueError(f"selected_classes={selected} out of range for classes={ALL_CLASS_NAMES}")
        print(f"\n⚠️ Using only selected classes: {selected}")
        
        # Store actual class names for this run (for t-SNE, confusion matrix)
        args.class_names = [ALL_CLASS_NAMES[i] for i in selected]
        print(f"   Class names: {args.class_names}")
        
        # Update way_num to match selected classes
        args.way_num = len(selected)
        print(f"   way_num updated to {args.way_num}")
        
        def filter_classes(X, y, selected_classes, file_paths=None):
            """Filter data to only include selected classes and remap labels."""
            mask = torch.zeros(len(y), dtype=torch.bool)
            for c in selected_classes:
                mask |= (y == c)
            
            X_filtered = X[mask]
            y_filtered = y[mask]
            
            # Remap labels to 0, 1, 2, ... (contiguous)
            label_map = {old: new for new, old in enumerate(selected_classes)}
            y_remapped = torch.tensor([label_map[yi.item()] for yi in y_filtered])

            filtered_paths = None
            if file_paths is not None and len(file_paths) == len(y):
                filtered_paths = [fp for fp, keep in zip(file_paths, mask.tolist()) if keep]

            return X_filtered, y_remapped, filtered_paths
        
        train_X, train_y, train_file_paths = filter_classes(train_X, train_y, selected, train_file_paths)
        val_X, val_y, val_file_paths = filter_classes(val_X, val_y, selected, val_file_paths)
        test_X, test_y, test_file_paths = filter_classes(test_X, test_y, selected, test_file_paths)
        
        print(f"   Train: {len(train_X)}, Val: {len(val_X)}, Test: {len(test_X)}")
    else:
        # Use all classes
        args.class_names = ALL_CLASS_NAMES
        if args.way_num != len(args.class_names):
            print(f"ℹ️ way_num={args.way_num} does not match dataset classes={len(args.class_names)}. "
                  f"Using way_num={len(args.class_names)}.")
            args.way_num = len(args.class_names)

    wandb.config.update(
        {
            "way_num": args.way_num,
            "query_num_train": args.query_num_train,
            "query_num_val": args.query_num_val,
            "query_num_test": args.query_num_test,
        },
        allow_val_change=True,
    )
    
    # Limit training samples if specified
    if args.training_samples:
        if args.training_samples % args.way_num != 0:
            raise ValueError(
                f"training_samples ({args.training_samples}) must be divisible by way_num ({args.way_num}) "
                "for balanced class sampling."
            )
        per_class = args.training_samples // args.way_num
        X_list, y_list = [], []
        
        for c in range(args.way_num):
            idx = (train_y == c).nonzero(as_tuple=True)[0]
            if len(idx) < per_class:
                raise ValueError(f"Class {c}: need {per_class}, have {len(idx)}")
            
            g = torch.Generator().manual_seed(args.seed)
            perm = torch.randperm(len(idx), generator=g)[:per_class]
            X_list.append(train_X[idx[perm]])
            y_list.append(train_y[idx[perm]])
        
        train_X = torch.cat(X_list)
        train_y = torch.cat(y_list)
        print(f"Using {args.training_samples} training samples ({per_class}/class)")
    
    # Note: Training dataset is created INSIDE train_loop with epoch-dependent seed
    # This ensures different episodes each epoch but reproducible across experiments
    
    test_ds = FewshotDataset(
        test_X,
        test_y,
        args.episode_num_test,
        args.way_num,
        args.shot_num,
        args.query_num_test,
        args.seed,
        return_indices=args.save_misclf_report,
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    # Initialize Model
    net = get_model(args)
    
    # Log model parameters
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"\nModel Parameters: {total_params:,} (trainable: {trainable_params:,})")
    wandb.log({"model/total_parameters": total_params, "model/trainable_parameters": trainable_params})
    
    if args.mode == 'train':
        best_acc, history = train_loop(net, train_X, train_y, val_X, val_y, args)
        
        # Load BEST checkpoint for testing
        samples_suffix = f'{args.training_samples}samples' if args.training_samples else 'all'
        path = os.path.join(args.path_weights, f'{args.dataset_name}_{args.model}_{samples_suffix}_{args.shot_num}shot_final.pth')
        print(f'Testing with BEST checkpoint: {path}')
        net.load_state_dict(torch.load(path))
        test_final(net, test_loader, args, test_X=test_X, test_y=test_y, test_file_paths=test_file_paths)
        
    else:  # Test only
        if args.weights:
            net.load_state_dict(torch.load(args.weights))
            test_final(net, test_loader, args, test_X=test_X, test_y=test_y, test_file_paths=test_file_paths)
        else:
            print("Error: Please specify --weights for test mode")
    
    wandb.finish()


if __name__ == '__main__':
    main()
