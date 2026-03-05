"""USCMambaNet Ablation Training Script (architecture-focused, CE-only).

Supported ablation groups:
    - dualpath: local_only / global_only / both
    - global_context: without_ms / with_ms
    - attention_stack: none / unified_only / late_only / both
    - prototype: no_cross / cross_no_axis / cross_axis

Results are saved to results/ folder in format:
    results_{dataset}_{ablation_type}_{mode}_{samples}samples_{shot}shot.txt
"""
import os
import argparse
import numpy as np
import torch
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
    seed_func,
)

# Model
from net.usc_mamba_net import USCMambaNet


# =============================================================================
# Configuration
# =============================================================================

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='USCMambaNet Ablation Training')
    
    # Paths
    parser.add_argument('--dataset_path', type=str, 
                        default='/mnt/disk2/nhatnc/res/scalogram_fewshot/proposed_model/smnet/scalogram_knee_augmented_split')
    parser.add_argument('--path_weights', type=str, default='checkpoints/')
    parser.add_argument('--path_results', type=str, default='results/')
    parser.add_argument('--dataset_name', type=str, default='knee_aug_split')
    
    # Ablation settings
    parser.add_argument('--ablation_type', type=str, required=True,
                        choices=['dualpath', 'global_context', 'attention_stack', 'prototype'],
                        help='Type of ablation study')
    parser.add_argument('--ablation_mode', type=str, required=True,
                        help='Mode depends on type (see run_ablation.py)')
    
    # Few-shot settings
    parser.add_argument('--way_num', type=int, default=4)
    parser.add_argument('--shot_num', type=int, default=1)
    parser.add_argument('--query_num', type=int, default=None,
                        help='Legacy: set same queries per class for train/val/test')
    parser.add_argument('--query_num_train', type=int, default=1)
    parser.add_argument('--query_num_val', type=int, default=1)
    parser.add_argument('--query_num_test', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--d_state', type=int, default=8)
    parser.add_argument('--global_expand', type=int, default=2)
    parser.add_argument('--proto_pool_size', type=int, default=12)
    parser.add_argument('--num_prototypes', type=int, default=2)
    parser.add_argument('--detach_prototypes', action='store_true')
    parser.add_argument('--similarity_proj_dim', type=int, default=None)
    
    # Training
    parser.add_argument('--training_samples', type=int, default=None)
    parser.add_argument('--episode_num_train', type=int, default=200)
    parser.add_argument('--episode_num_val', type=int, default=300)
    parser.add_argument('--episode_num_test', type=int, default=300)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3, help='Base learning rate')
    parser.add_argument('--eta_min', type=float, default=1e-5, help='Min LR for cosine')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--grad_clip', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--temperature', type=float, default=16.0,
                        help='Cosine similarity temperature (same as main.py)')
    parser.add_argument('--beta_maha', type=float, default=0.25,
                        help='UAPS variance-aware penalty weight')
    parser.add_argument('--uaps_eps', type=float, default=1e-4,
                        help='UAPS epsilon')
    parser.add_argument('--cross_attn_alpha', type=float, default=0.3,
                        help='Prototype Cross-Attention residual weight (same as main.py)')
    parser.add_argument('--ms_downsample', type=int, default=2,
                        help='Downsample ratio for multi-scale global branch')
    parser.add_argument('--atrous_rate', type=int, default=2,
                        help='Dilation for atrous branch')
    parser.add_argument('--late_attn_window', type=int, default=4,
                        help='Window size for late attention bridge')
    parser.add_argument('--late_attn_dropout', type=float, default=0.0,
                        help='Dropout for late attention bridge')
    parser.add_argument('--axis_proto_pool', type=str, default='mean',
                        choices=['mean', 'max'],
                        help='Pooling for axis prototype tokens')
    parser.add_argument('--axis_proto_mix_init', type=str, default='1.0,0.5,0.5',
                        help='Initial mix logits [full,time,freq] for axis proto')
    
    # WandB
    parser.add_argument('--project', type=str, default='uscmamba-ablation')
    
    return parser.parse_args()


def get_ablation_config(ablation_type: str, ablation_mode: str) -> dict:
    """Get model configuration based on ablation type and mode.
    
    Returns a dict of flags to pass to USCMambaNet.
    """
    # Default full model (architecture-only, no pair expert)
    config = {
        'dualpath_mode': 'both',
        'use_ms_global': True,
        'use_unified_attention': True,
        'use_late_attention': True,
        'use_cross_attention': True,
        'use_axis_proto': True,
        'use_pair_expert': False,
    }
    
    if ablation_type == 'dualpath':
        if ablation_mode not in {'local_only', 'global_only', 'both'}:
            raise ValueError("dualpath mode must be one of: local_only, global_only, both")
        config['dualpath_mode'] = ablation_mode
    elif ablation_type == 'global_context':
        if ablation_mode == 'without_ms':
            config['use_ms_global'] = False
        elif ablation_mode == 'with_ms':
            config['use_ms_global'] = True
        else:
            raise ValueError("global_context mode must be: without_ms, with_ms")
    elif ablation_type == 'attention_stack':
        if ablation_mode == 'none':
            config['use_unified_attention'] = False
            config['use_late_attention'] = False
        elif ablation_mode == 'unified_only':
            config['use_unified_attention'] = True
            config['use_late_attention'] = False
        elif ablation_mode == 'late_only':
            config['use_unified_attention'] = False
            config['use_late_attention'] = True
        elif ablation_mode == 'both':
            config['use_unified_attention'] = True
            config['use_late_attention'] = True
        else:
            raise ValueError("attention_stack mode must be: none, unified_only, late_only, both")
    elif ablation_type == 'prototype':
        if ablation_mode == 'no_cross':
            config['use_cross_attention'] = False
            config['use_axis_proto'] = False
        elif ablation_mode == 'cross_no_axis':
            config['use_cross_attention'] = True
            config['use_axis_proto'] = False
        elif ablation_mode == 'cross_axis':
            config['use_cross_attention'] = True
            config['use_axis_proto'] = True
        else:
            raise ValueError("prototype mode must be: no_cross, cross_no_axis, cross_axis")
    else:
        raise ValueError(f"Unsupported ablation type: {ablation_type}")

    if not config['use_cross_attention']:
        config['use_axis_proto'] = False
    
    return config


def get_model(args):
    """Initialize USCMambaNet model with ablation config."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get ablation-specific configuration
    ablation_config = get_ablation_config(args.ablation_type, args.ablation_mode)

    axis_mix_parts = [p.strip() for p in args.axis_proto_mix_init.split(',')]
    if len(axis_mix_parts) != 3:
        raise ValueError("--axis_proto_mix_init must have exactly 3 comma-separated values")
    axis_proto_mix_init = tuple(float(v) for v in axis_mix_parts)
    
    print(f"\nAblation Config: {args.ablation_type} = {args.ablation_mode}")
    print(f"  dualpath_mode: {ablation_config['dualpath_mode']}")
    print(f"  use_ms_global: {ablation_config['use_ms_global']} (downsample={args.ms_downsample}, atrous={args.atrous_rate})")
    print(f"  use_unified_attention: {ablation_config['use_unified_attention']}")
    print(f"  use_late_attention: {ablation_config['use_late_attention']} (window={args.late_attn_window})")
    print(f"  use_cross_attention: {ablation_config['use_cross_attention']}")
    print(f"  use_axis_proto: {ablation_config['use_axis_proto']} ({args.axis_proto_pool}, mix={axis_proto_mix_init})")
    
    model = USCMambaNet(
        in_channels=3,
        way_num=args.way_num,
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
        ms_downsample=args.ms_downsample,
        atrous_rate=args.atrous_rate,
        late_attn_window=args.late_attn_window,
        late_attn_dropout=args.late_attn_dropout,
        axis_proto_pool=args.axis_proto_pool,
        axis_proto_mix_init=axis_proto_mix_init,
        similarity_proj_dim=args.similarity_proj_dim,
        **ablation_config
    )
    
    return model.to(device)


# =============================================================================
# Training
# =============================================================================

def train_loop(net, train_X, train_y, val_X, val_y, args):
    """Train with CE-only objective under episodic protocol."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    optimizer = optim.AdamW(
        net.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Cosine Annealing Scheduler
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.num_epochs,
        eta_min=args.eta_min
    )
    
    history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}
    best_acc = 0.0
    
    for epoch in range(1, args.num_epochs + 1):
        train_seed = args.seed + epoch
        train_ds = FewshotDataset(train_X, train_y, args.episode_num_train,
                                  args.way_num, args.shot_num, args.query_num_train, train_seed)
        train_gen = torch.Generator()
        train_gen.manual_seed(train_seed)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  generator=train_gen)
        
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

            loss = F.cross_entropy(scores, targets)
            loss.backward()
            
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            
            optimizer.step()
            total_loss += loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(loss=f'{loss.item():.4f}', lr=f'{current_lr:.2e}')
        
        scheduler.step()
        
        train_acc = train_correct / train_total if train_total > 0 else 0
        val_seed = args.seed + epoch
        val_ds = FewshotDataset(val_X, val_y, args.episode_num_val,
                                args.way_num, args.shot_num, args.query_num_val, val_seed)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
        
        val_acc, val_loss = evaluate(net, val_loader, args)
        avg_loss = total_loss / len(train_loader)
        
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_loss'].append(avg_loss)
        history['val_loss'].append(val_loss if val_loss else 0.0)
        
        print(f'Epoch {epoch}: Loss={avg_loss:.4f}, Train={train_acc:.4f}, Val={val_acc:.4f}')
        
        wandb.log({
            "epoch": epoch,
            "loss/train": avg_loss,
                "loss/val": val_loss if val_loss is not None else 0.0,
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


def evaluate(net, loader, args):
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

            loss = F.cross_entropy(scores, targets)
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
    
    # Save results to file
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
    
    return acc_mean, acc_std


# =============================================================================
# Main
# =============================================================================

def main():
    args = get_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.query_num is not None:
        args.query_num_train = args.query_num
        args.query_num_val = args.query_num
        args.query_num_test = args.query_num
    
    print(f"\n{'='*60}")
    print(f"USCMambaNet Ablation: {args.ablation_type} - {args.ablation_mode}")
    print('='*60)
    print(f"Config: {args.shot_num}-shot | {args.num_epochs} epochs | Device: {args.device}")
    print(f"Training samples: {args.training_samples}")
    print("NOTE: CE-only objective (no center/margin/pair/hard-mining)")
    
    # Initialize WandB
    samples_str = f"{args.training_samples}samples" if args.training_samples else "all"
    run_name = f"{args.ablation_type}_{args.ablation_mode}_{samples_str}_{args.shot_num}shot"
    
    config = vars(args).copy()
    wandb.init(project=args.project, config=config, name=run_name, 
               group=f"ablation_{args.ablation_type}", job_type="train")
    
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

    dataset_classes = list(getattr(dataset, 'classes', []))
    if dataset_classes and args.way_num != len(dataset_classes):
        print(f"ℹ️ way_num={args.way_num} does not match dataset classes={len(dataset_classes)}. "
              f"Using way_num={len(dataset_classes)}.")
        args.way_num = len(dataset_classes)
    
    # Limit training samples
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
    
    test_ds = FewshotDataset(test_X, test_y, args.episode_num_test,
                             args.way_num, args.shot_num, args.query_num_test, args.seed)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    # Initialize Model
    net = get_model(args)
    
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Model Parameters: {total_params:,}")
    wandb.log({"model/total_parameters": total_params})
    
    # Train
    best_acc, history = train_loop(net, train_X, train_y, val_X, val_y, args)
    
    # Load best and test
    samples_suffix = f'{args.training_samples}samples' if args.training_samples else 'all'
    ablation_suffix = f'{args.ablation_type}_{args.ablation_mode}'
    path = os.path.join(args.path_weights, f'{args.dataset_name}_{ablation_suffix}_{samples_suffix}_{args.shot_num}shot_best.pth')
    net.load_state_dict(torch.load(path))
    test_final(net, test_loader, args)
    
    wandb.finish()


if __name__ == '__main__':
    main()
