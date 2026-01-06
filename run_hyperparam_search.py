"""Hyperparameter Grid Search for SMNet.

Protocol following Chen et al. 2019, Baseline++, Dhillon et al. 2020:

PHASE A: Hyperparameter Selection (using VALIDATION)
    - Fix epochs = 100
    - Grid search over hyperparameters
    - Validate AFTER full training (epoch 100)
    - Choose config with highest val acc

PHASE B: Final Training & Testing
    - Train with chosen hyperparameters
    - Test at epoch 100 (blind)
    - Report mean ± std

Usage:
    # Phase A: Grid search
    python run_hyperparam_search.py --phase A --grid temperature
    
    # Phase B: Final test with chosen hyperparams
    python run_hyperparam_search.py --phase B --temperature 0.5 --lr 1e-3
"""
import os
import sys
import argparse
import subprocess
import json
from dataclasses import dataclass, field
from typing import List, Dict
from datetime import datetime

# =============================================================================
# Configuration
# =============================================================================

@dataclass 
class HPOConfig:
    """Hyperparameter optimization config."""
    
    # Dataset
    dataset_path: str = './scalogram_minh'
    dataset_name: str = 'minh'
    
    # Fixed settings
    way_num: int = 4
    shot_num: int = 1
    query_num: int = 5
    image_size: int = 64
    num_epochs: int = 100  # FIXED
    
    # Episodes
    episode_num_train: int = 100
    episode_num_val: int = 150
    episode_num_test: int = 150
    
    # Training samples
    training_samples: int = 600  # Use enough for HPO
    
    # WandB
    project: str = 'smnet-hpo'


# Hyperparameter grids (priority order)
GRIDS = {
    # Priority 1: Temperature (MOST IMPORTANT for metric learning)
    'temperature': [0.1, 0.3, 0.5, 1.0],
    
    # Priority 2: Learning rate (lr_init)
    'lr': [1e-4, 3e-4, 1e-3],
    
    # Priority 3: Weight decay
    'weight_decay': [1e-5, 1e-4, 5e-4],
    
    # Priority 4: Lambda init (only if still overfitting)
    'lambda_init': [0.3, 0.5, 1.0],
    
    # Priority 5: eta_min (optional, usually fixed)
    'eta_min': [1e-6, 1e-5],
}

# Default values (baseline)
DEFAULTS = {
    'temperature': 0.5,
    'lr': 1e-3,
    'eta_min': 1e-5,
    'weight_decay': 1e-4,
    'lambda_init': 1.0,
}


# =============================================================================
# Phase A: Grid Search
# =============================================================================

def run_single_config(config: HPOConfig, hyperparams: Dict) -> Dict:
    """Run training + validation for a single hyperparameter config.
    
    Returns:
        Dict with 'val_acc', 'val_std', 'train_acc'
    """
    # Build command
    cmd = [
        sys.executable, 'main.py',
        '--mode', 'train',
        '--shot_num', str(config.shot_num),
        '--way_num', str(config.way_num),
        '--query_num', str(config.query_num),
        '--image_size', str(config.image_size),
        '--dataset_path', config.dataset_path,
        '--dataset_name', config.dataset_name,
        '--project', config.project,
        '--num_epochs', str(config.num_epochs),
        '--episode_num_train', str(config.episode_num_train),
        '--episode_num_val', str(config.episode_num_val),
        '--episode_num_test', str(config.episode_num_test),
        '--training_samples', str(config.training_samples),
    ]
    
    # Add hyperparameters
    for key, value in hyperparams.items():
        if key == 'weight_decay':
            cmd.extend(['--weight_decay', str(value)])
        else:
            cmd.extend([f'--{key}', str(value)])
    
    # Add HPO mode flag (skip test, return val at epoch 100)
    cmd.append('--hpo_mode')
    
    config_str = '_'.join(f"{k}={v}" for k, v in hyperparams.items())
    print(f"\n{'='*60}")
    print(f"Running: {config_str}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Parse output for val accuracy (last line should be JSON)
        lines = result.stdout.strip().split('\n')
        for line in reversed(lines):
            if line.startswith('HPO_RESULT:'):
                data = json.loads(line[11:])
                return data
        
        print("Warning: Could not parse HPO result")
        return {'val_acc': 0.0, 'val_std': 0.0}
        
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"Stderr: {e.stderr}")
        return {'val_acc': 0.0, 'val_std': 0.0}


def grid_search(config: HPOConfig, grid_name: str) -> Dict:
    """Run grid search for a specific hyperparameter.
    
    Returns:
        Dict mapping config -> results
    """
    if grid_name not in GRIDS:
        raise ValueError(f"Unknown grid: {grid_name}. Available: {list(GRIDS.keys())}")
    
    grid_values = GRIDS[grid_name]
    results = {}
    
    print(f"\n{'#'*60}")
    print(f"GRID SEARCH: {grid_name}")
    print(f"Values: {grid_values}")
    print(f"Training samples: {config.training_samples}")
    print(f"Epochs: {config.num_epochs}")
    print('#'*60)
    
    for value in grid_values:
        # Start with defaults
        hyperparams = DEFAULTS.copy()
        hyperparams[grid_name] = value
        
        result = run_single_config(config, hyperparams)
        results[value] = result
        
        print(f"  {grid_name}={value}: val_acc={result.get('val_acc', 0):.4f} ± {result.get('val_std', 0):.4f}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"GRID SEARCH RESULTS: {grid_name}")
    print('='*60)
    
    best_val = 0
    best_value = None
    
    for value, result in results.items():
        val_acc = result.get('val_acc', 0)
        val_std = result.get('val_std', 0)
        print(f"  {grid_name}={value}: {val_acc:.4f} ± {val_std:.4f}")
        
        if val_acc > best_val:
            best_val = val_acc
            best_value = value
    
    print(f"\nBest: {grid_name}={best_value} -> {best_val:.4f}")
    print('='*60)
    
    return results


# =============================================================================
# Phase B: Final Training & Testing
# =============================================================================

def final_train_test(config: HPOConfig, hyperparams: Dict):
    """Final training and testing with chosen hyperparameters."""
    
    print(f"\n{'#'*60}")
    print("PHASE B: FINAL TRAINING & TESTING")
    print(f"Hyperparameters: {hyperparams}")
    print('#'*60)
    
    # Build command (training mode, NO val-based early stopping)
    cmd = [
        sys.executable, 'main.py',
        '--mode', 'train',
        '--shot_num', str(config.shot_num),
        '--way_num', str(config.way_num),
        '--query_num', str(config.query_num),
        '--image_size', str(config.image_size),
        '--dataset_path', config.dataset_path,
        '--dataset_name', config.dataset_name,
        '--project', config.project + '-final',
        '--num_epochs', str(config.num_epochs),
        '--episode_num_train', str(config.episode_num_train),
        '--episode_num_val', str(config.episode_num_val),
        '--episode_num_test', str(config.episode_num_test),
        '--training_samples', str(config.training_samples),
    ]
    
    # Add hyperparameters
    for key, value in hyperparams.items():
        cmd.extend([f'--{key}', str(value)])
    
    # Run training
    print("\n[Phase B.1] Training...")
    subprocess.run(cmd, check=True)
    
    print("\n[Phase B.2] Testing at epoch 100...")
    print("Test results are in WandB and results/ folder.")


# =============================================================================
# CLI
# =============================================================================

def get_args():
    parser = argparse.ArgumentParser(description='SMNet Hyperparameter Search')
    
    parser.add_argument('--phase', type=str, required=True, choices=['A', 'B'],
                        help='A=Grid search, B=Final train+test')
    
    # Phase A options
    parser.add_argument('--grid', type=str, default='temperature',
                        choices=list(GRIDS.keys()),
                        help='Which hyperparameter to grid search')
    
    # Phase B / override options
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--lambda_init', type=float, default=None)
    
    # Dataset options
    parser.add_argument('--dataset_path', type=str, default='./scalogram_minh')
    parser.add_argument('--dataset_name', type=str, default='minh')
    parser.add_argument('--training_samples', type=int, default=600)
    parser.add_argument('--shot_num', type=int, default=1)
    
    return parser.parse_args()


def main():
    args = get_args()
    
    # Create config
    config = HPOConfig(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        training_samples=args.training_samples,
        shot_num=args.shot_num,
    )
    
    if args.phase == 'A':
        # Phase A: Grid search
        results = grid_search(config, args.grid)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f'hpo_results_{args.grid}_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump({str(k): v for k, v in results.items()}, f, indent=2)
        print(f"\nResults saved to: {results_file}")
        
    elif args.phase == 'B':
        # Phase B: Final train + test
        hyperparams = DEFAULTS.copy()
        
        # Override with user-specified values
        if args.temperature is not None:
            hyperparams['temperature'] = args.temperature
        if args.lr is not None:
            hyperparams['lr'] = args.lr
        if args.weight_decay is not None:
            hyperparams['weight_decay'] = args.weight_decay
        if args.lambda_init is not None:
            hyperparams['lambda_init'] = args.lambda_init
        
        final_train_test(config, hyperparams)


if __name__ == '__main__':
    main()
