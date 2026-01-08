"""Run all USCMambaNet experiments for 1-shot, 5-shot with various training sample sizes."""
import subprocess
import sys
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description='Run all USCMambaNet experiments')
    parser.add_argument('--project', type=str, default='uscmamba', help='WandB project name')
    parser.add_argument('--dataset_path', type=str, 
                        default='/mnt/disk2/nhatnc/res/scalogram_fewshot/proposed_model/smnet/scalogram_v2_split',
                        help='Path to dataset')
    parser.add_argument('--dataset_name', type=str, default='minh', help='Dataset name for logging')
    return parser.parse_args()


# Configuration
SHOTS = [1, 5]

# Training samples: [min, small, medium, large]
SAMPLES_LIST = [80, 200, 600, 6000]

# Query samples (same for train/val/test)
QUERY_NUM = 1  # Changed from 5 to 1

# Model variants
MODELS = ['uscmamba']


def run_experiment(model, shot, samples, dataset_path, dataset_name, project):
    """Run a single SMNet experiment."""
    print(f"\n{'='*60}")
    print(f"Model={model}, Shot={shot}, Samples={samples if samples else 'All'}")
    print('='*60)
    
    cmd = [
        sys.executable, 'main.py',
        '--model', model,
        '--shot_num', str(shot),
        '--way_num', '4',
        '--query_num', str(QUERY_NUM),
        '--image_size', '64',
        '--mode', 'train',
        '--project', project,
        '--dataset_path', dataset_path,
        '--dataset_name', dataset_name,
        '--num_epochs', '100',
        '--lr', '1e-3',
        '--eta_min', '1e-5',
        '--weight_decay', '1e-4',
        '--margin', '0',  # No margin needed (ClassConditionalCosine handles discrimination)
        '--outlier_fraction', '0.2',  # Remove 20% outliers in 5-shot
        '--episode_num_train', '100',
        '--episode_num_val', '150',
        '--episode_num_test', '150',
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
    
    # Count total experiments
    total_experiments = len(MODELS) * len(SHOTS) * len(SAMPLES_LIST)
    current = 0
    
    print("="*60)
    print("USCMambaNet - Full Experiment Suite")
    print("="*60)
    print(f"Models: {MODELS}")
    print(f"Shots: {SHOTS}")
    print(f"Training samples: {SAMPLES_LIST}")
    print(f"Dataset: {args.dataset_path} ({args.dataset_name})")
    print(f"Total experiments: {total_experiments}")
    print("="*60)
    
    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    success_count = 0
    failed_experiments = []
    
    # Run all experiments
    for model in MODELS:
        for shot in SHOTS:
            for samples in SAMPLES_LIST:
                current += 1
                print(f"\n[{current}/{total_experiments}]", end=" ")
                
                success = run_experiment(
                    model=model,
                    shot=shot,
                    samples=samples,
                    dataset_path=args.dataset_path,
                    dataset_name=args.dataset_name,
                    project=args.project
                )
                
                if success:
                    success_count += 1
                else:
                    failed_experiments.append(f"{model}_{shot}shot_{samples}samples")
    
    # Summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Total: {total_experiments}")
    print(f"Success: {success_count}")
    print(f"Failed: {len(failed_experiments)}")
    
    if failed_experiments:
        print("\nFailed experiments:")
        for exp in failed_experiments:
            print(f"  - {exp}")
    
    print("\n" + "="*60)
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
