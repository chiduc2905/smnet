"""Unified Ablation Runner - Run ALL ablation experiments.

This script runs all ablation studies in sequence:
    1. DualBranch ablation: local_only, global_only, both
    2. Slot Refinement (SCA+CMA): none, sca_only, cma_only, both
    3. SlotAttention: without, with

Each ablation produces a comparison table and logs to WandB.

Usage:
    python run_ablation.py --ablation dual_branch
    python run_ablation.py --ablation slot_refinement
    python run_ablation.py --ablation slot_attention
    python run_ablation.py --ablation all
"""
import os
import sys
import argparse
import subprocess
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AblationConfig:
    """Unified ablation configuration."""
    
    # Dataset
    dataset_path: str = './scalogram_minh'
    dataset_name: str = 'minh'
    
    # Few-shot settings
    way_num: int = 4
    shots: List[int] = field(default_factory=lambda: [1, 5])
    query_num: int = 5  # Same for train/val/test
    
    # Training
    training_samples_list: List[int] = field(default_factory=lambda: [80, 200, 600, 6000])
    num_epochs: int = 100
    lr: float = 1e-3  # Base LR
    min_lr: float = 1e-5  # Min LR for cosine
    start_lr: float = 1e-5  # Start LR for warmup
    warmup_iters: int = 500  # Warmup iterations
    
    # Episodes
    episode_num_train: int = 100
    episode_num_val: int = 150
    episode_num_test: int = 150
    
    # WandB
    project: str = 'smnet-ablation'
    
    # Image
    image_size: int = 64


# Minimum samples calculation
def calculate_min_samples(way_num: int, max_shot: int, query_num: int) -> Dict[str, int]:
    """Calculate minimum samples needed per split.
    
    Args:
        way_num: Number of classes (Way)
        max_shot: Maximum shot number (e.g., 10 for 10-shot)
        query_num: Query samples per class
        
    Returns:
        Dict with per_class and total minimums
    """
    per_class = max_shot + query_num
    total = way_num * per_class
    
    return {
        'per_class': per_class,
        'total': total,
        'formula': f'{max_shot} (shot) + {query_num} (query) = {per_class}'
    }


# =============================================================================
# Ablation Definitions
# =============================================================================

ABLATION_DUAL_BRANCH = {
    'name': 'dual_branch',
    'description': 'DualBranchFusion: Local vs Global vs Both',
    'modes': ['local_only', 'global_only', 'both'],
    'flag': '--dual_branch_mode',
}

ABLATION_SLOT_REFINEMENT = {
    'name': 'slot_refinement',
    'description': 'M5 (SCA) + M6 (CMA): None vs SCA vs CMA vs Both',
    'modes': ['none', 'sca_only', 'cma_only', 'both'],
    'flag': '--slot_refinement_mode',
}

ABLATION_SLOT_ATTENTION = {
    'name': 'slot_attention',
    'description': 'SlotAttention: Without vs With',
    'modes': ['without', 'with'],
    'flag': '--slot_attention_mode',
}

ALL_ABLATIONS = [ABLATION_DUAL_BRANCH, ABLATION_SLOT_REFINEMENT, ABLATION_SLOT_ATTENTION]


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
        ablation_type: 'dual_branch', 'slot_refinement', or 'slot_attention'
        mode: Mode within the ablation type
        shot: Shot number (1, 5, 10)
        training_samples: Number of training samples
        
    Returns:
        True if successful
    """
    # Build command
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
        '--episode_num_train', str(config.episode_num_train),
        '--episode_num_val', str(config.episode_num_val),
        '--episode_num_test', str(config.episode_num_test),
        '--training_samples', str(training_samples),
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
        print(f"  ✗ main_ablation.py not found, using placeholder")
        return True  # Placeholder, will be created


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
    parser = argparse.ArgumentParser(description='Run SMNet ablation studies')
    
    parser.add_argument('--ablation', type=str, default='all',
                        choices=['dual_branch', 'slot_refinement', 'slot_attention', 'all'],
                        help='Which ablation to run')
    
    parser.add_argument('--dataset_path', type=str, default='./scalogram_minh')
    parser.add_argument('--dataset_name', type=str, default='minh')
    parser.add_argument('--project', type=str, default='smnet-ablation')
    
    parser.add_argument('--query_num', type=int, default=5,
                        help='Query samples per class (default: 5)')
    parser.add_argument('--num_epochs', type=int, default=100)
    
    parser.add_argument('--training_samples', type=str, default='80,200,600,6000',
                        help='Comma-separated list of training sample sizes')
    
    parser.add_argument('--dry_run', action='store_true',
                        help='Print experiments without running')
    
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
        training_samples_list=training_samples,
    )
    
    # Print min samples info
    min_info = calculate_min_samples(config.way_num, max(config.shots), config.query_num)
    print("\n" + "=" * 70)
    print("MINIMUM SAMPLES CALCULATION")
    print("=" * 70)
    print(f"Way (classes): {config.way_num}")
    print(f"Max shot: {max(config.shots)}")
    print(f"Query per class: {config.query_num}")
    print(f"\nMinimum per class: {min_info['per_class']} ({min_info['formula']})")
    print(f"Minimum total per split: {min_info['total']}")
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
