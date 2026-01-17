"""Hyperparameter Search for USCMambaNet.

Proper few-shot protocol:
- Fix training protocol (epochs, lr, episodes)
- For each config: train full epochs, use FINAL checkpoint
- Evaluate on val episodes (random, not fixed)
- Compare configs by mean val acc ± CI
- Choose config with high mean AND low CI (stable)
"""
import subprocess
import sys
import argparse
import os
import re
import itertools
from datetime import datetime
import csv


def get_args():
    parser = argparse.ArgumentParser(description='Hyperparameter Search for USCMambaNet')
    parser.add_argument('--project', type=str, default='uscmamba-hpsearch', help='WandB project name')
    parser.add_argument('--dataset_path', type=str, 
                        default='/mnt/disk2/nhatnc/res/scalogram_fewshot/proposed_model/smnet/scalogram_official',
                        help='Path to dataset')
    parser.add_argument('--dataset_name', type=str, default='minh', help='Dataset name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--training_samples', type=int, default=30, help='Training samples (30, 60, 150)')
    parser.add_argument('--shot_num', type=int, default=1, help='Shot number (1 or 5)')
    parser.add_argument('--num_epochs', type=int, default=100, help='Fixed number of epochs')
    parser.add_argument('--search_mode', type=str, default='margin', 
                        choices=['margin', 'attention', 'temperature', 'architecture', 
                                 'optimizer', 'loss_weight', 'regularization', 'full', 'all'],
                        help='Which hyperparameters to search')
    return parser.parse_args()


# =============================================================================
# Hyperparameter Search Spaces
# =============================================================================

# Margin Loss Parameters (ArcFace/CosFace)
MARGIN_SEARCH = {
    'margin_type': ['arcface', 'cosface', 'none'],
    'lambda_margin': [0.05, 0.1, 0.2],
    'margin_scale': [10.0, 15.0, 20.0],
    'margin': [0.2, 0.3, 0.4],
}

# Attention Parameters  
ATTENTION_SEARCH = {
    'cross_attn_alpha': [0.05, 0.1, 0.2, 0.3],
    'use_unified_attention': ['true', 'false'],
    'use_cross_attention': ['true', 'false'],
}

# Temperature Parameters
TEMPERATURE_SEARCH = {
    'temperature': [8.0, 12.0, 16.0, 20.0],
    'delta_lambda': [0.15, 0.25, 0.35],
}

# Model Architecture Parameters
ARCHITECTURE_SEARCH = {
    'hidden_dim': [32, 64, 128],  # Feature dimension
    'dualpath_mode': ['both', 'local_only', 'global_only'],  # Dual-branch mode
}

# Optimizer / Regularization Parameters
OPTIMIZER_SEARCH = {
    'weight_decay': [1e-4, 5e-4, 1e-3],  # L2 regularization
    'grad_clip': [0.5, 1.0, 2.0],  # Gradient clipping
    'lr': [5e-4, 1e-3, 2e-3],  # Learning rate
}

# Loss Weight Parameters
LOSS_WEIGHT_SEARCH = {
    'lambda_center': [0.0, 0.01, 0.05, 0.1],  # Center loss weight
    'lambda_margin': [0.0, 0.05, 0.1, 0.2],  # Margin loss weight
}

# Regularization Search (focused)
REGULARIZATION_SEARCH = {
    'weight_decay': [1e-4, 5e-4, 1e-3],
    'grad_clip': [0.5, 1.0],
    'lambda_center': [0.0, 0.01, 0.05],
}

# Full Search (reduced grid to avoid explosion)
FULL_SEARCH = {
    'margin_type': ['arcface', 'none'],
    'lambda_margin': [0.1],
    'margin_scale': [15.0],
    'temperature': [12.0, 16.0],
    'cross_attn_alpha': [0.05, 0.1],
}

# Comprehensive Search (all key params, reduced values)
ALL_SEARCH = {
    'hidden_dim': [64],  # Keep fixed for speed
    'temperature': [12.0, 16.0],
    'cross_attn_alpha': [0.05, 0.1],
    'delta_lambda': [0.2, 0.3],
    'margin_type': ['arcface', 'none'],
    'lambda_margin': [0.1],
    'weight_decay': [5e-4],
    'lambda_center': [0.0, 0.01],
}


def get_search_space(mode):
    """Get hyperparameter search space based on mode."""
    if mode == 'margin':
        return MARGIN_SEARCH
    elif mode == 'attention':
        return ATTENTION_SEARCH
    elif mode == 'temperature':
        return TEMPERATURE_SEARCH
    elif mode == 'architecture':
        return ARCHITECTURE_SEARCH
    elif mode == 'optimizer':
        return OPTIMIZER_SEARCH
    elif mode == 'loss_weight':
        return LOSS_WEIGHT_SEARCH
    elif mode == 'regularization':
        return REGULARIZATION_SEARCH
    elif mode == 'full':
        return FULL_SEARCH
    elif mode == 'all':
        return ALL_SEARCH
    else:
        raise ValueError(f"Unknown search mode: {mode}")


def generate_configs(search_space):
    """Generate all combinations from search space."""
    keys = list(search_space.keys())
    values = list(search_space.values())
    
    configs = []
    for combo in itertools.product(*values):
        config = dict(zip(keys, combo))
        
        # Skip invalid combinations
        if config.get('margin_type') == 'none':
            # If no margin loss, skip margin params
            config['lambda_margin'] = 0.0
            config['margin_scale'] = 1.0
            config['margin'] = 0.0
        
        # Avoid duplicates
        if config not in configs:
            configs.append(config)
    
    return configs


def run_experiment(config, base_args):
    """Run single experiment with given config."""
    print(f"\n{'='*60}")
    print(f"Config: {config}")
    print('='*60)
    
    cmd = [
        sys.executable, 'main.py',
        '--model', 'uscmamba',
        '--shot_num', str(base_args.shot_num),
        '--way_num', '3',
        '--query_num', '5',
        '--image_size', '128',
        '--mode', 'train',
        '--project', base_args.project,
        '--dataset_path', base_args.dataset_path,
        '--dataset_name', base_args.dataset_name,
        '--training_samples', str(base_args.training_samples),
        '--num_epochs', str(base_args.num_epochs),
        '--lr', '1e-3',
        '--eta_min', '1e-5',
        '--weight_decay', '5e-4',
        '--grad_clip', '0.5',
        '--episode_num_train', '200',
        '--episode_num_val', '300',
        '--episode_num_test', '600',  # More test episodes for stable CI
        '--seed', str(base_args.seed),
    ]
    
    # Add config-specific arguments
    for key, value in config.items():
        cmd.extend([f'--{key}', str(value)])
    
    try:
        # Don't capture output to avoid buffer overflows with large logs
        # Let it stream to stdout so user sees progress (and knows it's not frozen)
        subprocess.run(cmd, check=True) 
        return True, ""
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return False, str(e)


def parse_results(base_args, config_id):
    """Parse results from result file."""
    samples_str = f"{base_args.training_samples}samples"
    result_file = os.path.join(
        'results',
        f"results_{base_args.dataset_name}_uscmamba_{samples_str}_{base_args.shot_num}shot.txt"
    )
    
    if not os.path.exists(result_file):
        return None
    
    with open(result_file, 'r') as f:
        content = f.read()
    
    # Parse metrics
    acc_match = re.search(r'Accuracy\s*:\s*([\d.]+)\s*±\s*([\d.]+)', content)
    if acc_match:
        mean_acc = float(acc_match.group(1))
        std_acc = float(acc_match.group(2))
    else:
        return None
    
    # Parse 95% CI if available
    ci_match = re.search(r'Mean Accuracy\s*:\s*[\d.]+\s*±\s*([\d.]+)%\s*\(95%', content)
    ci95 = float(ci_match.group(1)) / 100 if ci_match else std_acc * 1.96
    
    return {
        'mean_acc': mean_acc,
        'std_acc': std_acc,
        'ci95': ci95,
    }


def save_results_csv(all_results, output_path):
    """Save all results to CSV for easy comparison."""
    if not all_results:
        return
    
    # Get all keys from configs
    config_keys = set()
    for r in all_results:
        config_keys.update(r['config'].keys())
    config_keys = sorted(config_keys)
    
    fieldnames = ['config_id'] + config_keys + ['mean_acc', 'std_acc', 'ci95', 'status']
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for r in all_results:
            row = {'config_id': r['config_id'], 'status': r['status']}
            row.update(r['config'])
            if r['metrics']:
                row.update(r['metrics'])
            writer.writerow(row)
    
    print(f"\nResults saved to: {output_path}")


def print_summary(all_results):
    """Print summary of best configs."""
    print("\n" + "="*60)
    print("HYPERPARAMETER SEARCH SUMMARY")
    print("="*60)
    
    # Filter successful results
    successful = [r for r in all_results if r['status'] == 'success' and r['metrics']]
    
    if not successful:
        print("No successful experiments!")
        return
    
    # Sort by mean accuracy (descending)
    successful.sort(key=lambda x: x['metrics']['mean_acc'], reverse=True)
    
    print(f"\nTotal configs tested: {len(all_results)}")
    print(f"Successful: {len(successful)}")
    
    print("\n--- TOP 5 CONFIGS (by mean accuracy) ---")
    for i, r in enumerate(successful[:5], 1):
        m = r['metrics']
        print(f"\n{i}. Config {r['config_id']}: {m['mean_acc']*100:.2f} ± {m['ci95']*100:.2f}%")
        for k, v in r['config'].items():
            if v != 0.0 and v != 1.0 and v != 'none':
                print(f"   {k}: {v}")
    
    # Best by mean acc + low CI (stability score)
    print("\n--- BEST STABLE CONFIG ---")
    # Score = mean_acc - 0.5 * ci95 (penalize high variance)
    for r in successful:
        r['stability_score'] = r['metrics']['mean_acc'] - 0.5 * r['metrics']['ci95']
    
    successful.sort(key=lambda x: x['stability_score'], reverse=True)
    best = successful[0]
    m = best['metrics']
    print(f"Config {best['config_id']}: {m['mean_acc']*100:.2f} ± {m['ci95']*100:.2f}%")
    print("Parameters:")
    for k, v in best['config'].items():
        if v != 0.0 and v != 1.0:
            print(f"  --{k} {v}")


def main():
    args = get_args()
    
    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('hyperparam_results', exist_ok=True)
    
    # Get search space
    search_space = get_search_space(args.search_mode)
    configs = generate_configs(search_space)
    
    print("="*60)
    print("USCMambaNet Hyperparameter Search")
    print("="*60)
    print(f"Search mode: {args.search_mode}")
    print(f"Total configs: {len(configs)}")
    print(f"Training samples: {args.training_samples}")
    print(f"Shot: {args.shot_num}")
    print(f"Epochs: {args.num_epochs} (FIXED)")
    print(f"Seed: {args.seed}")
    print("="*60)
    
    # Run all experiments
    all_results = []
    
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}]")
        
        success, output = run_experiment(config, args)
        
        if success:
            metrics = parse_results(args, i)
            status = 'success' if metrics else 'parse_error'
        else:
            metrics = None
            status = 'failed'
        
        all_results.append({
            'config_id': i,
            'config': config,
            'metrics': metrics,
            'status': status,
        })
        
        # Save intermediate results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(
            'hyperparam_results',
            f"hpsearch_{args.search_mode}_{args.training_samples}samples_{args.shot_num}shot.csv"
        )
        save_results_csv(all_results, csv_path)
    
    # Print summary
    print_summary(all_results)
    
    print("\n" + "="*60)
    print("Hyperparameter search completed!")
    print("="*60)


if __name__ == '__main__':
    main()
