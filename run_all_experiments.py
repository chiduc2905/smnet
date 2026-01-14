"""Run all USCMambaNet experiments for 1-shot, 5-shot with various training sample sizes."""
import subprocess
import sys
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description='Run all USCMambaNet experiments')
    parser.add_argument('--project', type=str, default='uscmamba', help='WandB project name')
    parser.add_argument('--dataset_path', type=str, 
                        default='/mnt/disk2/nhatnc/res/scalogram_fewshot/proposed_model/smnet/scalogram_official',
                        help='Path to dataset')
    parser.add_argument('--dataset_name', type=str, default='minh', help='Dataset name for logging')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--mode_id', type=int, default=None, choices=list(range(1, 9)),
                        help='Run specific experiment (1-8). If not set, runs all experiments.')
    return parser.parse_args()


# Experiment modes mapping: mode_id -> (samples, shot)
# Mode 1: 30 samples, 1-shot    Mode 2: 30 samples, 5-shot
# Mode 3: 60 samples, 1-shot    Mode 4: 60 samples, 5-shot
# Mode 5: 150 samples, 1-shot   Mode 6: 150 samples, 5-shot
# Mode 7: All samples, 1-shot   Mode 8: All samples, 5-shot
EXPERIMENT_MODES = {
    1: (30, 1),
    2: (30, 5),
    3: (60, 1),
    4: (60, 5),
    5: (150, 1),
    6: (150, 5),
    7: (None, 1),  # None = All samples
    8: (None, 5),
}


# Configuration
SHOTS = [1, 5]

# Training samples: [min_5shot, small, medium, all]
# Dataset: corona=367, NotPD=210 (min), surface=392 → 3 classes
# Min for 5-shot: 5 (shot) + 5 (query) = 10/class → 30 total
# Max balanced: 210/class → 630 total, or None for all available
SAMPLES_LIST = [30, 60, 150, None]

# Query samples (same for train/val/test)
QUERY_NUM = 5  # Synced with pd_fewshot for identical episodes

# Model variants
MODELS = ['uscmamba']


def run_experiment(model, shot, samples, dataset_path, dataset_name, project, seed):
    """Run a single SMNet experiment."""
    print(f"\n{'='*60}")
    print(f"Model={model}, Shot={shot}, Samples={samples if samples else 'All'}")
    print('='*60)
    
    cmd = [
        sys.executable, 'main.py',
        '--model', model,
        '--shot_num', str(shot),
        '--way_num', '3',
        '--query_num', str(QUERY_NUM),
        '--image_size', '128',
        '--mode', 'train',
        '--project', project,
        '--dataset_path', dataset_path,
        '--dataset_name', dataset_name,
        '--num_epochs', '100',
        '--lr', '1e-3',
        '--eta_min', '1e-5',
        '--weight_decay', '1e-4',
        '--margin_type', 'none',  # Disabled ArcFace/CosFace for testing
        '--use_unified_attention', 'true',  # Full model with Unified Attention
        '--episode_num_train', '100',
        '--episode_num_val', '150',
        '--episode_num_test', '150',
        '--seed', str(seed),  # Fixed seed for reproducibility
    ]
    
    if samples is not None:
        cmd.extend(['--training_samples', str(samples)])
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return False


def main():
    args = get_args()
    
    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Determine experiments to run
    if args.mode_id is not None:
        # Run single experiment based on mode_id
        samples, shot = EXPERIMENT_MODES[args.mode_id]
        experiments = [('uscmamba', samples, shot)]
        print("=" * 60)
        print(f"USCMambaNet - Single Experiment (Mode {args.mode_id})")
        print("=" * 60)
        print(f"  Samples: {samples if samples else 'All'}")
        print(f"  Shot: {shot}")
        print(f"  Dataset: {args.dataset_path}")
        print("=" * 60)
    else:
        # Run all experiments
        experiments = [
            (model, samples, shot)
            for model in MODELS
            for samples in SAMPLES_LIST
            for shot in SHOTS
        ]
        print("=" * 60)
        print("USCMambaNet - Full Experiment Suite (All 8 modes)")
        print("=" * 60)
        print("Mode mapping:")
        for mid, (s, sh) in EXPERIMENT_MODES.items():
            print(f"  Mode {mid}: {s if s else 'All'} samples, {sh}-shot")
        print(f"Dataset: {args.dataset_path} ({args.dataset_name})")
        print(f"Total experiments: {len(experiments)}")
        print("=" * 60)
    
    success_count = 0
    failed_experiments = []
    total = len(experiments)
    
    for i, (model, samples, shot) in enumerate(experiments, 1):
        print(f"\n[{i}/{total}]", end=" ")
        
        success = run_experiment(
            model=model,
            shot=shot,
            samples=samples,
            dataset_path=args.dataset_path,
            dataset_name=args.dataset_name,
            project=args.project,
            seed=args.seed
        )
        
        if success:
            success_count += 1
        else:
            failed_experiments.append(f"uscmamba_{shot}shot_{samples if samples else 'all'}samples")
    
    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Total: {total}")
    print(f"Success: {success_count}")
    print(f"Failed: {len(failed_experiments)}")
    
    if failed_experiments:
        print("\nFailed experiments:")
        for exp in failed_experiments:
            print(f"  - {exp}")
    
    print("\n" + "=" * 60)
    print("Generating comparison charts...")
    print("="*60)
    
    # Generate comparison after all experiments
    generate_comparison_charts(args.dataset_name)
    
    print("\nAll experiments completed!")


def generate_comparison_charts(dataset_name):
    """Generate comparison bar charts from results."""
    import re
    try:
        from function.function import plot_model_comparison_bar
    except ImportError:
        print("Warning: Could not import plot function, skipping charts")
        return
    
    results_dir = 'results/'
    
    # Model display names
    model_display_names = {
        'smnet': 'SMNet',
        'smnet_light': 'SMNet-Light'
    }
    
    for samples in SAMPLES_LIST:
        samples_str = f"{samples}samples"
        
        model_results = {}
        
        for model in MODELS:
            display_name = model_display_names.get(model, model)
            model_results[display_name] = {}
            
            for shot in SHOTS:
                result_file = os.path.join(
                    results_dir,
                    f"results_{dataset_name}_{model}_{samples_str}_{shot}shot.txt"
                )
                
                if os.path.exists(result_file):
                    with open(result_file, 'r') as f:
                        content = f.read()
                        # Parse accuracy
                        match = re.search(r'Accuracy\s*:\s*([\d.]+)\s*±', content)
                        if match:
                            acc = float(match.group(1))
                            model_results[display_name][f'{shot}shot'] = acc
        
        # Remove incomplete results
        complete_results = {}
        for model, shots_dict in model_results.items():
            if all(f'{shot}shot' in shots_dict for shot in SHOTS):
                complete_results[model] = shots_dict
        
        if complete_results:
            save_path = os.path.join(results_dir, f"smnet_comparison_{dataset_name}_{samples_str}.png")
            try:
                plot_model_comparison_bar(complete_results, samples, save_path)
                print(f"  Chart saved: {save_path}")
            except Exception as e:
                print(f"  Error generating chart: {e}")
        else:
            print(f"  No complete results for {samples_str}")


if __name__ == '__main__':
    main()
