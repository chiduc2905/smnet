"""Unified Ablation Runner - Run ALL ablation experiments.

This script runs ablation studies for USCMambaNet:
    1. DualPath ablation: local_only, global_only, both (feature extraction)
    2. Unified Attention ablation: with vs without unified attention
    3. Cross Attention ablation: with vs without prototype cross-attention

Each ablation produces results files for comparison.
NOTE: ArcFace/CosFace is NOT used in ablation experiments.

Usage:
    python run_ablation.py --ablation dualpath
    python run_ablation.py --ablation unified_attention
    python run_ablation.py --ablation cross_attention
    python run_ablation.py --ablation all
"""
import os
import sys
import argparse
import subprocess
from dataclasses import dataclass, field
from typing import List, Dict
from datetime import datetime


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AblationConfig:
    """Unified ablation configuration."""
    
    # Dataset
    dataset_path: str = '/mnt/disk2/nhatnc/res/scalogram_fewshot/proposed_model/smnet/scalogram_official'
    dataset_name: str = 'minh'
    
    # Few-shot settings
    way_num: int = 3
    shots: List[int] = field(default_factory=lambda: [1, 5])
    query_num: int = 5  # Same for train/val/test
    
    # Training
    training_samples_list: List[int] = field(default_factory=lambda: [30, 60, 150])
    num_epochs: int = 100
    lr: float = 1e-3
    eta_min: float = 1e-5
    weight_decay: float = 1e-4
    
    # Episodes
    episode_num_train: int = 200
    episode_num_val: int = 300
    episode_num_test: int = 300
    
    # WandB
    project: str = 'uscmamba-ablation'
    
    # Image
    image_size: int = 128
    
    # Seed for reproducibility
    seed: int = 42


# =============================================================================
# Ablation Definitions
# =============================================================================

ABLATION_DUALPATH = {
    'name': 'dualpath',
    'description': 'DualPath: local_only vs global_only vs both',
    'modes': ['local_only', 'global_only', 'both'],
}

ABLATION_UNIFIED_ATTENTION = {
    'name': 'unified_attention',
    'description': 'Unified Attention: Without vs With',
    'modes': ['without', 'with'],
}

ABLATION_CROSS_ATTENTION = {
    'name': 'cross_attention',
    'description': 'Cross Attention: Without vs With',
    'modes': ['without', 'with'],
}

ALL_ABLATIONS = [ABLATION_DUALPATH, ABLATION_UNIFIED_ATTENTION, ABLATION_CROSS_ATTENTION]


# =============================================================================
# Runner
# =============================================================================

def run_single_experiment(
    config: AblationConfig,
    ablation_type: str,
    mode: str,
    shot: int,
    training_samples: int
) -> bool:
    """Run a single ablation experiment.
    
    Args:
        config: AblationConfig
        ablation_type: 'dualpath', 'unified_attention', or 'cross_attention'
        mode: Mode within the ablation type
        shot: Shot number (1, 5)
        training_samples: Number of training samples
        
    Returns:
        True if successful
    """
    # Build command - use main_ablation.py
    cmd = [
        sys.executable, 'main_ablation.py',
        '--ablation_type', ablation_type,
        '--ablation_mode', mode,
        '--shot_num', str(shot),
        '--way_num', str(config.way_num),
        '--query_num', str(config.query_num),
        '--image_size', str(config.image_size),
        '--dataset_path', config.dataset_path,
        '--dataset_name', config.dataset_name,
        '--project', config.project,
        '--num_epochs', str(config.num_epochs),
        '--lr', str(config.lr),
        '--eta_min', str(config.eta_min),
        '--weight_decay', str(config.weight_decay),
        '--episode_num_train', str(config.episode_num_train),
        '--episode_num_val', str(config.episode_num_val),
        '--episode_num_test', str(config.episode_num_test),
        '--training_samples', str(training_samples),
        '--seed', str(config.seed),  # Fixed seed for reproducibility
    ]
    
    experiment_name = f"{ablation_type}_{mode}_{shot}shot_{training_samples}samples"
    print(f"\n  → Running: {experiment_name}")
    
    try:
        subprocess.run(cmd, check=True, capture_output=False)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed: {e}")
        return False
    except FileNotFoundError:
        print(f"  ✗ main_ablation.py not found")
        return False


def run_ablation_study(ablation: Dict, config: AblationConfig) -> Dict:
    """Run all experiments for a single ablation type.
    
    Args:
        ablation: Ablation definition dict
        config: AblationConfig
        
    Returns:
        Results dict
    """
    ablation_name = ablation['name']
    modes = ablation['modes']
    
    print("\n" + "=" * 70)
    print(f"ABLATION STUDY: {ablation['description']}")
    print("=" * 70)
    print(f"Modes: {modes}")
    print(f"Shots: {config.shots}")
    print(f"Training samples: {config.training_samples_list}")
    
    total = len(modes) * len(config.shots) * len(config.training_samples_list)
    print(f"Total experiments: {total}")
    print("=" * 70)
    
    results = {'success': 0, 'failed': [], 'total': total}
    current = 0
    
    for mode in modes:
        for shot in config.shots:
            for samples in config.training_samples_list:
                current += 1
                print(f"\n[{current}/{total}] {ablation_name}: {mode}, {shot}-shot, {samples} samples")
                
                success = run_single_experiment(
                    config=config,
                    ablation_type=ablation_name,
                    mode=mode,
                    shot=shot,
                    training_samples=samples
                )
                
                if success:
                    results['success'] += 1
                else:
                    results['failed'].append(f"{mode}_{shot}shot_{samples}samples")
    
    return results


def run_all_ablations(config: AblationConfig) -> Dict:
    """Run all ablation studies.
    
    Args:
        config: AblationConfig
        
    Returns:
        Combined results dict
    """
    all_results = {}
    
    for ablation in ALL_ABLATIONS:
        results = run_ablation_study(ablation, config)
        all_results[ablation['name']] = results
    
    return all_results


def print_summary(all_results: Dict):
    """Print summary of all ablation studies."""
    print("\n" + "=" * 70)
    print("ABLATION STUDY SUMMARY")
    print("=" * 70)
    
    total_success = 0
    total_failed = 0
    
    for name, results in all_results.items():
        status = "✓" if results['success'] == results['total'] else "⚠"
        print(f"\n{status} {name}:")
        print(f"    Success: {results['success']}/{results['total']}")
        
        if results['failed']:
            print(f"    Failed: {len(results['failed'])}")
            for exp in results['failed'][:5]:
                print(f"      - {exp}")
            if len(results['failed']) > 5:
                print(f"      ... and {len(results['failed']) - 5} more")
        
        total_success += results['success']
        total_failed += len(results['failed'])
    
    print("\n" + "-" * 70)
    print(f"TOTAL: {total_success} success, {total_failed} failed")
    print("=" * 70)


# =============================================================================
# CLI
# =============================================================================

def get_args():
    parser = argparse.ArgumentParser(description='Run USCMambaNet ablation studies')
    
    parser.add_argument('--ablation', type=str, default='all',
                        choices=['dualpath', 'unified_attention', 'cross_attention', 'all'],
                        help='Which ablation to run')
    
    parser.add_argument('--dataset_path', type=str, 
                        default='/mnt/disk2/nhatnc/res/scalogram_fewshot/proposed_model/smnet/scalogram_official')
    parser.add_argument('--dataset_name', type=str, default='minh')
    parser.add_argument('--project', type=str, default='uscmamba-ablation')
    
    parser.add_argument('--query_num', type=int, default=5,
                        help='Query samples per class (default: 5)')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--image_size', type=int, default=128)
    
    parser.add_argument('--training_samples', type=str, default='30,60,150',
                        help='Comma-separated list of training sample sizes')
    
    parser.add_argument('--dry_run', action='store_true',
                        help='Print experiments without running')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()


def main():
    args = get_args()
    
    # Parse training samples
    training_samples = [int(x.strip()) for x in args.training_samples.split(',')]
    
    # Create config
    config = AblationConfig(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        project=args.project,
        query_num=args.query_num,
        num_epochs=args.num_epochs,
        image_size=args.image_size,
        training_samples_list=training_samples,
        seed=args.seed,  # Pass seed
    )
    
    print("\n" + "=" * 70)
    print("USCMAMBA ABLATION STUDY")
    print("=" * 70)
    print(f"Ablations: {args.ablation}")
    print(f"Shots: {config.shots}")
    print(f"Training samples: {config.training_samples_list}")
    print(f"Image size: {config.image_size}")
    print(f"NOTE: ArcFace/CosFace is DISABLED for ablation experiments")
    print("=" * 70)
    
    if args.dry_run:
        print("\n[DRY RUN] Would run the following experiments:")
        
        if args.ablation == 'all':
            ablations = ALL_ABLATIONS
        else:
            ablations = [a for a in ALL_ABLATIONS if a['name'] == args.ablation]
        
        total = 0
        for ablation in ablations:
            n = len(ablation['modes']) * len(config.shots) * len(training_samples)
            print(f"\n  {ablation['name']}: {n} experiments")
            for mode in ablation['modes']:
                print(f"    - {mode}")
            total += n
        
        print(f"\n  TOTAL: {total} experiments")
        return
    
    # Run ablations
    if args.ablation == 'all':
        all_results = run_all_ablations(config)
    else:
        ablation = [a for a in ALL_ABLATIONS if a['name'] == args.ablation][0]
        all_results = {ablation['name']: run_ablation_study(ablation, config)}
    
    # Print summary
    print_summary(all_results)


if __name__ == '__main__':
    main()
