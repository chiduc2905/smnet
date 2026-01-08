"""USCMambaNet (Unified Spatial-Channel Mamba Network) - Training and Evaluation.

This script trains and evaluates USCMambaNet which uses:
- PatchEmbed + PatchMerging for hierarchical feature extraction
- DualBranchFusion (AG-LKA + SS2D) for local-global features
- UnifiedSpatialChannelAttention for feature selection
- SimplePatchSimilarity for non-learnable cosine matching
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
    ContrastiveLoss, MarginContrastiveLoss, CenterLoss, seed_func,
    plot_confusion_matrix, plot_tsne, plot_training_curves
)
from function.debug_utils import print_grad_norm, print_logit_stats, set_debug_mode, is_debug_mode

# Model
from net.usc_mamba_net import USCMambaNet


# =============================================================================
# Configuration
# =============================================================================

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='SMNet (Slot Mamba Network) Few-shot Learning')
    
    # Paths
    parser.add_argument('--dataset_path', type=str, default='/mnt/disk2/nhatnc/res/scalogram_fewshot/proposed_model/smnet/scalogram_v2_split')
    parser.add_argument('--path_weights', type=str, default='checkpoints/')
    parser.add_argument('--path_results', type=str, default='results/')
    parser.add_argument('--weights', type=str, default=None, help='Checkpoint for testing')
    parser.add_argument('--dataset_name', type=str, default='scalogram',
                        help='Dataset name for checkpoint naming')
    
    # Model
    parser.add_argument('--model', type=str, default='uscmamba', 
                        choices=['uscmamba'])
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension for feature extractor')
    
    # Few-shot settings
    parser.add_argument('--way_num', type=int, default=4)
    parser.add_argument('--shot_num', type=int, default=1)
    parser.add_argument('--query_num', type=int, default=5, help='Queries per class (same for train/val/test)')
    parser.add_argument('--selected_classes', type=str, default=None,
                        help='Comma-separated class indices to use (e.g. "0,1" for first 2 classes). If None, use all classes.')
    parser.add_argument('--image_size', type=int, default=128,
                        help='Input image size (default: 128)')
    
    # Training
    parser.add_argument('--training_samples', type=int, default=None, 
                        help='Total training samples (e.g. 30=10/class)')
    parser.add_argument('--episode_num_train', type=int, default=100)
    parser.add_argument('--episode_num_val', type=int, default=150)
    parser.add_argument('--episode_num_test', type=int, default=150)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3, help='Base learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-5, help='Min LR for cosine')
    parser.add_argument('--start_lr', type=float, default=1e-5, help='Start LR for warmup')
    parser.add_argument('--warmup_iters', type=int, default=500, help='Warmup iterations')
    parser.add_argument('--temperature', type=float, default=10.0,
                        help='Logit scale τ (logit = cosine * τ, recommended: 10-30)')
    parser.add_argument('--outlier_fraction', type=float, default=0.2,
                        help='Fraction of outliers to remove in 5-shot (0.0 to disable)')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping max norm')
    parser.add_argument('--eta_min', type=float, default=1e-5,
                        help='Min LR for CosineAnnealingLR')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--seed', type=int, default=42)
    
    # Loss
    parser.add_argument('--lambda_center', type=float, default=0.1, 
                        help='Weight for Center Loss (default: 0.1)')
    parser.add_argument('--margin', type=float, default=0.2,
                        help='CosFace margin for correct class (default: 0.2). Set to 0 to disable.')
    
    # Debug
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode: print gradients, feature stats, logits')
    # Mode
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    
    # WandB
    parser.add_argument('--project', type=str, default='smnet',
                        help='WandB project name')
    
    return parser.parse_args()


def get_model(args):
    """Initialize model based on args."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = USCMambaNet(
        in_channels=3,  # RGB input
        hidden_dim=args.hidden_dim,
        temperature=args.temperature,
        outlier_fraction=args.outlier_fraction,
        device=str(device)
    )
    
    return model.to(device)


# =============================================================================
# Training
# =============================================================================

def train_loop(net, train_loader, val_X, val_y, args):
    """Train with CosineAnnealingLR + Warmup (per-iteration LR adjustment)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Loss: Simple CrossEntropy (no margin, no contrastive, no cosine)
    print("Using F.cross_entropy (standard CE loss)")
    
    # Calculate feature dimension dynamically
    # USCMambaNet.encode() returns (B, hidden_dim, H', W')
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, args.image_size, args.image_size).to(device)
        dummy_features = net.encode(dummy_input)  # (1, C, H', W')
        feat_dim = dummy_features.shape[1]  # hidden_dim (channel dim)
        
    criterion_center = CenterLoss(num_classes=args.way_num, feat_dim=feat_dim, device=device)
    
    # Optimizer with weight decay
    optimizer = optim.AdamW([
        {'params': net.parameters()},
        {'params': criterion_center.parameters()}
    ], lr=args.lr, weight_decay=args.weight_decay)
    
    # CosineAnnealingLR Scheduler (sync with main branch)
    # LR decays from lr to eta_min over T_max epochs following cosine curve
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.eta_min
    )
    
    # Training history
    history = {
        'train_acc': [],
        'val_acc': [],
        'train_loss': [],
        'val_loss': []
    }
    
    best_acc = 0.0
    
    for epoch in range(1, args.num_epochs + 1):
        net.train()
        total_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.num_epochs}')
        for query, q_labels, support, s_labels in pbar:
            optimizer.zero_grad()
            
            B = query.shape[0]
            C, H, W = query.shape[2], query.shape[3], query.shape[4]
            
            support = support.view(B, args.way_num, args.shot_num, C, H, W).to(device)
            query = query.to(device)
            targets = q_labels.view(-1).to(device)
            
            # Forward
            scores = net(query, support)
            
            # Track training accuracy on same episodes (not re-sampled)
            with torch.no_grad():
                preds = scores.argmax(dim=1)
                train_correct += (preds == targets).sum().item()
                train_total += targets.size(0)
            
            # Main Loss: Standard CrossEntropy (no margin, no contrastive)
            loss_main = F.cross_entropy(scores, targets)
            
            # Center Loss (optional)
            if args.lambda_center > 0:
                q_flat = query.view(-1, C, H, W)
                feat_maps = net.encode(q_flat)  # (NQ, C, H', W')
                features = feat_maps.mean(dim=(2, 3))  # Global avg pool: (NQ, C)
                features = F.normalize(features, p=2, dim=1)
                loss_center = criterion_center(features, targets)
                loss = loss_main + args.lambda_center * loss_center
            else:
                loss = loss_main
            
            loss.backward()
            
            # DEBUG: Print gradient norms and logit stats
            if args.debug and step == 0:  # Print once per epoch
                print_grad_norm(net, epoch, step, print_every=1)
                print_logit_stats(scores, step, print_every=1)
            
            # Gradient clipping
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            
            optimizer.step()
            
            total_loss += loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(loss=f'{loss.item():.4f}', lr=f'{current_lr:.2e}')
        
        # Step scheduler at end of epoch
        scheduler.step()
        
        # Training accuracy from same episodes (not re-sampled!)
        train_acc = train_correct / train_total if train_total > 0 else 0

        
        val_ds = FewshotDataset(val_X, val_y, args.episode_num_val,
                                args.way_num, args.shot_num, args.query_num, args.seed + epoch)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
        
        val_acc, val_loss = evaluate(net, val_loader, args)
        avg_loss = total_loss / len(train_loader)
        
        # Track history
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_loss'].append(avg_loss)
        history['val_loss'].append(val_loss if val_loss else 0.0)
        
        train_val_gap = train_acc - val_acc
        
        print(f'Epoch {epoch}: Loss={avg_loss:.4f}, Train={train_acc:.4f}, Val={val_acc:.4f} (gap={train_val_gap:+.4f})')
        
        # Log to WandB
        wandb.log({
            "epoch": epoch,
            "loss/train": avg_loss,
            "loss/val": val_loss,
            "accuracy/train": train_acc,
            "accuracy/val": val_acc,
            "train_val_gap": train_val_gap,
            "lr": optimizer.param_groups[0]['lr']
        })
        
        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            samples_suffix = f'{args.training_samples}samples' if args.training_samples else 'all'
            model_filename = f'{args.dataset_name}_{args.model}_{samples_suffix}_{args.shot_num}shot_best.pth'
            path = os.path.join(args.path_weights, model_filename)
            torch.save(net.state_dict(), path)
            print(f'  → Best model saved ({val_acc:.4f})')
            wandb.run.summary["best_val_acc"] = best_acc
    
    # Plot training curves
    samples_str = f"{args.training_samples}samples" if args.training_samples else "allsamples"
    curves_path = os.path.join(args.path_results, 
                               f"training_{args.dataset_name}_{args.model}_{samples_str}_{args.shot_num}shot")
    plot_training_curves(history, curves_path)
    
    if os.path.exists(f"{curves_path}_curves.png"):
        wandb.log({"training_curves": wandb.Image(f"{curves_path}_curves.png")})
    
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


def test_final(net, loader, args):
    """Final evaluation with detailed metrics."""
    import time
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_episodes = len(loader)
    
    print(f"\n{'='*60}")
    print(f"Final Test: USCMambaNet | {args.dataset_name} | {args.shot_num}-shot")
    print(f"{num_episodes} episodes × {args.way_num} classes × {args.query_num} query")
    print('='*60)
    
    net.eval()
    all_preds, all_targets, all_features = [], [], []
    episode_accuracies = []
    episode_times = []
    
    with torch.no_grad():
        for query, q_labels, support, s_labels in tqdm(loader, desc='Testing'):
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
            
            # Extract features for t-SNE using encode (returns B,C,H,W)
            q_flat = query.view(-1, C, H, W)
            features = net.encode(q_flat)  # (NQ, hidden_dim, H', W')
            feat = features.mean(dim=(2, 3))  # Global avg pool: (NQ, hidden_dim)
            all_features.append(feat.cpu().numpy())
    
    # Metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    episode_accuracies = np.array(episode_accuracies)
    episode_times = np.array(episode_times)
    
    acc_mean = episode_accuracies.mean()
    acc_std = episode_accuracies.std()
    acc_worst = episode_accuracies.min()
    acc_best = episode_accuracies.max()
    
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
    print(f"  Mean Accuracy : {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"  Worst-case    : {acc_worst:.4f}")
    print(f"  Best-case     : {acc_best:.4f}")
    print(f"  Precision     : {prec:.4f}")
    print(f"  Recall        : {rec:.4f}")
    print(f"  F1-Score      : {f1:.4f}")
    print(f"  p-value       : {p_val:.2e}")
    print(f"\nInference Time  : {time_mean:.2f} ± {time_std:.2f} ms/episode")
    
    # Log to WandB
    wandb.log({
        "test_accuracy_mean": acc_mean,
        "test_accuracy_std": acc_std,
        "test_accuracy_worst": acc_worst,
        "test_accuracy_best": acc_best,
        "test_precision": prec,
        "test_recall": rec,
        "test_f1": f1,
        "inference_time_mean_ms": time_mean,
    })
    
    wandb.run.summary["test_accuracy_mean"] = acc_mean
    wandb.run.summary["test_accuracy_std"] = acc_std
    
    # Plots
    samples_str = f"_{args.training_samples}samples" if args.training_samples else "_allsamples"
    
    cm_base = os.path.join(args.path_results, 
                           f"confusion_matrix_{args.dataset_name}_{args.model}_{samples_str.strip('_')}_{args.shot_num}shot")
    plot_confusion_matrix(all_targets, all_preds, args.way_num, cm_base)
    
    if os.path.exists(f"{cm_base}_2col.png"):
        wandb.log({"confusion_matrix": wandb.Image(f"{cm_base}_2col.png")})
    
    if all_features:
        features = np.vstack(all_features)
        tsne_base = os.path.join(args.path_results, 
                                 f"tsne_{args.dataset_name}_{args.model}_{samples_str.strip('_')}_{args.shot_num}shot")
        plot_tsne(features, all_targets, args.way_num, tsne_base)
        
        if os.path.exists(f"{tsne_base}_2col.png"):
            wandb.log({"tsne_plot": wandb.Image(f"{tsne_base}_2col.png")})
    
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
    
    seed_func(args.seed)
    
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
    
    # ============================================================
    # Filter to selected classes if specified
    # ============================================================
    if args.selected_classes:
        selected = [int(c.strip()) for c in args.selected_classes.split(',')]
        print(f"\n⚠️ Using only selected classes: {selected}")
        
        # Update way_num to match selected classes
        args.way_num = len(selected)
        print(f"   way_num updated to {args.way_num}")
        
        def filter_classes(X, y, selected_classes):
            """Filter data to only include selected classes and remap labels."""
            mask = torch.zeros(len(y), dtype=torch.bool)
            for c in selected_classes:
                mask |= (y == c)
            
            X_filtered = X[mask]
            y_filtered = y[mask]
            
            # Remap labels to 0, 1, 2, ... (contiguous)
            label_map = {old: new for new, old in enumerate(selected_classes)}
            y_remapped = torch.tensor([label_map[yi.item()] for yi in y_filtered])
            
            return X_filtered, y_remapped
        
        train_X, train_y = filter_classes(train_X, train_y, selected)
        val_X, val_y = filter_classes(val_X, val_y, selected)
        test_X, test_y = filter_classes(test_X, test_y, selected)
        
        print(f"   Train: {len(train_X)}, Val: {len(val_X)}, Test: {len(test_X)}")
    
    # Limit training samples if specified
    if args.training_samples:
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
    
    # Create data loaders
    # Training: 1 query per class (reduces variance, faster training)
    train_ds = FewshotDataset(train_X, train_y, args.episode_num_train,
                              args.way_num, args.shot_num, 1, args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    
    test_ds = FewshotDataset(test_X, test_y, args.episode_num_test,
                             args.way_num, args.shot_num, args.query_num, args.seed)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    # Initialize Model
    net = get_model(args)
    
    # Log model parameters
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"\nModel Parameters: {total_params:,} (trainable: {trainable_params:,})")
    wandb.log({"model/total_parameters": total_params, "model/trainable_parameters": trainable_params})
    
    if args.mode == 'train':
        best_acc, history = train_loop(net, train_loader, val_X, val_y, args)
        
        # Load best model for testing
        samples_suffix = f'{args.training_samples}samples' if args.training_samples else 'all'
        path = os.path.join(args.path_weights, f'{args.dataset_name}_{args.model}_{samples_suffix}_{args.shot_num}shot_best.pth')
        net.load_state_dict(torch.load(path))
        test_final(net, test_loader, args)
        
    else:  # Test only
        if args.weights:
            net.load_state_dict(torch.load(args.weights))
            test_final(net, test_loader, args)
        else:
            print("Error: Please specify --weights for test mode")
    
    wandb.finish()


if __name__ == '__main__':
    main()
