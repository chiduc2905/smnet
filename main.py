"""SMNet (Slot Mamba Network) - Training and Evaluation.

This script trains and evaluates SMNet which uses:
- ConvMixer for local spatial extraction
- SS2D (4-way Mamba) for global spatial context
- Slot Attention for semantic grouping
- Slot Mamba for inter-slot reasoning
- Covariance-based similarity for classification
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
    ContrastiveLoss, CenterLoss, seed_func,
    plot_confusion_matrix, plot_tsne, plot_training_curves
)

# Model
from net.slot_fewshot import SMNet


# =============================================================================
# Configuration
# =============================================================================

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='SMNet (Slot Mamba Network) Few-shot Learning')
    
    # Paths
    parser.add_argument('--dataset_path', type=str, default='./scalogram_minh/')
    parser.add_argument('--path_weights', type=str, default='checkpoints/')
    parser.add_argument('--path_results', type=str, default='results/')
    parser.add_argument('--weights', type=str, default=None, help='Checkpoint for testing')
    parser.add_argument('--dataset_name', type=str, default='scalogram',
                        help='Dataset name for checkpoint naming')
    
    # Model
    parser.add_argument('--model', type=str, default='smnet', 
                        choices=['smnet'])
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension for feature extractor')
    parser.add_argument('--num_slots', type=int, default=5,
                        help='Number of semantic slots K (paper: 5)')
    parser.add_argument('--slot_iters', type=int, default=5,
                        help='Slot attention iterations (paper: 5)')
    parser.add_argument('--lambda_init', type=float, default=1.0,
                        help='Lambda for class-aware refinement (paper: 2.0)')
    
    # Few-shot settings
    parser.add_argument('--way_num', type=int, default=4)
    parser.add_argument('--shot_num', type=int, default=1)
    parser.add_argument('--query_num', type=int, default=5, help='Queries per class (same for train/val/test)')
    parser.add_argument('--image_size', type=int, default=64,
                        help='Input image size')
    
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
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Temperature for similarity scaling (higher=softer)')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping max norm')
    parser.add_argument('--step_size', type=int, default=10,
                        help='StepLR step size (epochs)')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='StepLR gamma (LR multiplier)')
    parser.add_argument('--seed', type=int, default=42)
    
    # Loss
    parser.add_argument('--lambda_center', type=float, default=0.0, 
                        help='Weight for Center Loss (default: 0.0, disabled)')
    
    # Mode
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    
    # WandB
    parser.add_argument('--project', type=str, default='smnet',
                        help='WandB project name')
    
    return parser.parse_args()


def get_model(args):
    """Initialize model based on args."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SMNet(
        in_channels=3,  # RGB input
        num_slots=args.num_slots,
        slot_iters=args.slot_iters,  # SAFF paper: 5 iterations
        learnable_slots=True,
        regularization=1e-3,
        temperature=args.temperature,
        lambda_init=args.lambda_init,  # SAFF paper: lambda=2.0
        device=str(device)
    )
    
    return model.to(device)


# =============================================================================
# Training
# =============================================================================

def train_loop(net, train_loader, val_X, val_y, args):
    """Train with CosineAnnealingLR + Warmup (per-iteration LR adjustment)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Loss functions
    criterion_main = ContrastiveLoss().to(device)
    
    # Calculate feature dimension dynamically
    # SMNet uses extract_features which returns (B, N_patches, hidden_dim)
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, args.image_size, args.image_size).to(device)
        dummy_patches = net.extract_features(dummy_input)  # (1, N, C)
        feat_dim = dummy_patches.shape[-1]  # hidden_dim
        
    criterion_center = CenterLoss(num_classes=args.way_num, feat_dim=feat_dim, device=device)
    
    # Optimizer
    optimizer = optim.Adam([
        {'params': net.parameters()},
        {'params': criterion_center.parameters()}
    ], lr=args.lr)
    
    # StepLR Scheduler - simple step decay
    # LR = 0.001, gamma = 0.1, step_size = 10 epochs
    scheduler = lr_scheduler.StepLR(
        optimizer, 
        step_size=args.step_size,  # Decay every 10 epochs
        gamma=args.gamma  # Multiply by 0.1
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
            
            # Main Loss (CrossEntropy via ContrastiveLoss)
            loss_main = criterion_main(scores, targets)
            
            # Center Loss (optional)
            if args.lambda_center > 0:
                q_flat = query.view(-1, C, H, W)
                patches = net.extract_features(q_flat)  # (NQ, N, C)
                features = patches.mean(dim=1)  # Global avg pool: (NQ, C)
                features = F.normalize(features, p=2, dim=1)
                loss_center = criterion_center(features, targets)
                loss = loss_main + args.lambda_center * loss_center
            else:
                loss = loss_main
            
            loss.backward()
            
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
        
        val_acc, val_loss = evaluate(net, val_loader, args, criterion_main)
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


def evaluate(net, loader, args, criterion_main=None):
    """Compute accuracy and optionally loss on loader."""
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
            
            if criterion_main is not None:
                loss = criterion_main(scores, targets)
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
    print(f"Final Test: SMNet | {args.dataset_name} | {args.shot_num}-shot")
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
            
            # Extract features for t-SNE
            q_flat = query.view(-1, C, H, W)
            patches = net.extract_features(q_flat)  # (NQ, N, C)
            feat = patches.mean(dim=1)  # Global avg pool: (NQ, C)
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
    print("SMNet: Slot Mamba Network")
    print('='*60)
    print(f"Config: {args.model} | {args.shot_num}-shot | {args.num_epochs} epochs | Device: {args.device}")
    print(f"Architecture: ConvMixer → SS2D → ChannelAttn → SlotAttn({args.num_slots}) → SlotMamba → Covariance")
    print(f"Dataset: {args.dataset_path}")
    
    # Initialize WandB
    samples_str = f"{args.training_samples}samples" if args.training_samples else "all"
    run_name = f"smnet_{args.dataset_name}_{samples_str}_{args.shot_num}shot"
    
    config = vars(args).copy()
    config['architecture'] = 'SMNet (Slot Mamba Network)'
    
    wandb.init(project=args.project, config=config, name=run_name, group=f"smnet_{args.dataset_name}", job_type=args.mode)
    
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
    train_ds = FewshotDataset(train_X, train_y, args.episode_num_train,
                              args.way_num, args.shot_num, args.query_num, args.seed)
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
