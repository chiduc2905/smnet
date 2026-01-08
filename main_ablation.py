"""SMNet Ablation Training Script.

Simple wrapper that calls main.py with ablation mode flags.
For now, runs standard SMNet training - ablation configs will be added to SMNet model.
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

from dataset import load_dataset
from dataloader.dataloader import FewshotDataset
from function.function import (
    ContrastiveLoss, CenterLoss, seed_func,
    plot_confusion_matrix, plot_tsne, plot_training_curves
)

# Model - uses standard SMNet for now
from net.slot_fewshot import SMNet


# =============================================================================
# Configuration
# =============================================================================

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='SMNet Ablation Training')
    
    # Paths
    parser.add_argument('--dataset_path', type=str, default='/mnt/disk2/nhatnc/res/scalogram_fewshot/proposed_model/smnet/scalogram_v2_split')
    parser.add_argument('--path_weights', type=str, default='checkpoints/')
    parser.add_argument('--path_results', type=str, default='results/')
    parser.add_argument('--dataset_name', type=str, default='minh')
    
    # Ablation settings
    parser.add_argument('--ablation_type', type=str, required=True,
                        choices=['dual_branch', 'slot_refinement', 'slot_attention', 'saff'])
    parser.add_argument('--ablation_mode', type=str, required=True,
                        help='Mode within ablation type')
    
    # Model
    parser.add_argument('--num_slots', type=int, default=5)
    parser.add_argument('--slot_iters', type=int, default=5)
    parser.add_argument('--lambda_init', type=float, default=2.0)
    
    # Few-shot settings
    parser.add_argument('--way_num', type=int, default=4)
    parser.add_argument('--shot_num', type=int, default=1)
    parser.add_argument('--query_num', type=int, default=5)
    parser.add_argument('--image_size', type=int, default=64)
    
    # Training
    parser.add_argument('--training_samples', type=int, default=None)
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
                        help='Temperature for similarity scaling')
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--step_size', type=int, default=10,
                        help='StepLR step size (epochs)')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='StepLR gamma (LR multiplier)')
    parser.add_argument('--seed', type=int, default=42)
    
    # WandB
    parser.add_argument('--project', type=str, default='smnet-ablation')
    
    return parser.parse_args()


def get_model(args):
    """Initialize SMNet model with ablation config."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Ablation config passed to SMNet
    ablation_config = {
        'ablation_type': args.ablation_type,
        'ablation_mode': args.ablation_mode,
    }
    
    model = SMNet(
        in_channels=3,  # RGB
        num_slots=args.num_slots,
        slot_iters=args.slot_iters,
        lambda_init=args.lambda_init,
        temperature=args.temperature,
        device=str(device),
        # Ablation config (SMNet will handle)
        **ablation_config
    )
    
    return model.to(device)


# =============================================================================
# Training
# =============================================================================

def train_loop(net, train_loader, val_X, val_y, args):
    """Train with CosineAnnealingLR + Warmup (per-iteration LR adjustment)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    criterion_main = ContrastiveLoss().to(device)
    
    # Calculate feature dimension
    # SMNet uses extract_features which returns (B, N_patches, hidden_dim)
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, args.image_size, args.image_size).to(device)
        dummy_patches = net.extract_features(dummy_input)  # (1, N, C)
        feat_dim = dummy_patches.shape[-1]  # hidden_dim
        
    criterion_center = CenterLoss(num_classes=args.way_num, feat_dim=feat_dim, device=device)
    
    optimizer = optim.Adam([
        {'params': net.parameters()},
        {'params': criterion_center.parameters()}
    ], lr=args.lr)
    
    # StepLR Scheduler - simple step decay
    scheduler = lr_scheduler.StepLR(
        optimizer, 
        step_size=args.step_size,
        gamma=args.gamma
    )
    
    history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}
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
            
            scores = net(query, support)
            
            with torch.no_grad():
                preds = scores.argmax(dim=1)
                train_correct += (preds == targets).sum().item()
                train_total += targets.size(0)
            
            loss = criterion_main(scores, targets)
            loss.backward()
            
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            
            optimizer.step()
            total_loss += loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(loss=f'{loss.item():.4f}', lr=f'{current_lr:.2e}')
        
        # Step scheduler at end of epoch
        scheduler.step()
        
        train_acc = train_correct / train_total if train_total > 0 else 0
        
        val_ds = FewshotDataset(val_X, val_y, args.episode_num_val,
                                args.way_num, args.shot_num, args.query_num, args.seed + epoch)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
        
        val_acc, val_loss = evaluate(net, val_loader, args, criterion_main)
        avg_loss = total_loss / len(train_loader)
        
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_loss'].append(avg_loss)
        history['val_loss'].append(val_loss if val_loss else 0.0)
        
        print(f'Epoch {epoch}: Loss={avg_loss:.4f}, Train={train_acc:.4f}, Val={val_acc:.4f}')
        
        wandb.log({
            "epoch": epoch,
            "loss/train": avg_loss,
            "loss/val": val_loss,
            "accuracy/train": train_acc,
            "accuracy/val": val_acc,
            "lr": optimizer.param_groups[0]['lr']
        })
        
        if val_acc > best_acc:
            best_acc = val_acc
            samples_suffix = f'{args.training_samples}samples' if args.training_samples else 'all'
            ablation_suffix = f'{args.ablation_type}_{args.ablation_mode}'
            model_filename = f'{args.dataset_name}_{ablation_suffix}_{samples_suffix}_{args.shot_num}shot_best.pth'
            path = os.path.join(args.path_weights, model_filename)
            torch.save(net.state_dict(), path)
            print(f'  → Best model saved ({val_acc:.4f})')
            wandb.run.summary["best_val_acc"] = best_acc
    
    return best_acc, history


def evaluate(net, loader, args, criterion_main=None):
    """Compute accuracy and loss."""
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


def test_final(net, loader, args):
    """Final evaluation with metrics."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*60}")
    print(f"Final Test: {args.ablation_type}_{args.ablation_mode} | {args.shot_num}-shot")
    print('='*60)
    
    net.eval()
    all_preds, all_targets = [], []
    episode_accuracies = []
    
    with torch.no_grad():
        for query, q_labels, support, s_labels in tqdm(loader, desc='Testing'):
            B, NQ, C, H, W = query.shape
            
            support = support.view(B, args.way_num, args.shot_num, C, H, W).to(device)
            query = query.to(device)
            targets = q_labels.view(-1).to(device)
            
            scores = net(query, support)
            preds = scores.argmax(dim=1)
            
            episode_correct = (preds == targets).float().mean().item()
            episode_accuracies.append(episode_correct)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    episode_accuracies = np.array(episode_accuracies)
    
    acc_mean = episode_accuracies.mean()
    acc_std = episode_accuracies.std()
    
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, 
        labels=list(range(args.way_num)),
        average='macro', 
        zero_division=0
    )
    
    print(f"\n{'='*60}")
    print(f"Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    print('='*60)
    
    wandb.log({
        "test_accuracy_mean": acc_mean,
        "test_accuracy_std": acc_std,
        "test_precision": prec,
        "test_recall": rec,
        "test_f1": f1,
    })
    
    wandb.run.summary["test_accuracy_mean"] = acc_mean
    wandb.run.summary["test_accuracy_std"] = acc_std
    
    # Save results
    samples_str = f"{args.training_samples}samples" if args.training_samples else "all"
    ablation_str = f"{args.ablation_type}_{args.ablation_mode}"
    txt_path = os.path.join(args.path_results, 
                            f"results_{args.dataset_name}_{ablation_str}_{samples_str}_{args.shot_num}shot.txt")
    with open(txt_path, 'w') as f:
        f.write(f"Ablation: {args.ablation_type} - {args.ablation_mode}\n")
        f.write(f"Dataset: {args.dataset_name}\n")
        f.write(f"Shot: {args.shot_num}\n")
        f.write(f"Training Samples: {args.training_samples if args.training_samples else 'All'}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy: {acc_mean:.4f} ± {acc_std:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall: {rec:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
    print(f"Results saved to {txt_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    args = get_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*60}")
    print(f"SMNet Ablation: {args.ablation_type} - {args.ablation_mode}")
    print('='*60)
    print(f"Config: {args.shot_num}-shot | {args.num_epochs} epochs | Device: {args.device}")
    print(f"Training samples: {args.training_samples}")
    print(f"NOTE: Ablation config is logged but architecture runs standard SMNet for now")
    
    # Initialize WandB
    samples_str = f"{args.training_samples}samples" if args.training_samples else "all"
    run_name = f"{args.ablation_type}_{args.ablation_mode}_{samples_str}_{args.shot_num}shot"
    
    config = vars(args).copy()
    wandb.init(project=args.project, config=config, name=run_name, 
               group=f"ablation_{args.ablation_type}", job_type="train")
    
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
    
    # Limit training samples
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
    
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Model Parameters: {total_params:,}")
    wandb.log({"model/total_parameters": total_params})
    
    # Train
    best_acc, history = train_loop(net, train_loader, val_X, val_y, args)
    
    # Load best and test
    samples_suffix = f'{args.training_samples}samples' if args.training_samples else 'all'
    ablation_suffix = f'{args.ablation_type}_{args.ablation_mode}'
    path = os.path.join(args.path_weights, f'{args.dataset_name}_{ablation_suffix}_{samples_suffix}_{args.shot_num}shot_best.pth')
    net.load_state_dict(torch.load(path))
    test_final(net, test_loader, args)
    
    wandb.finish()


if __name__ == '__main__':
    main()
