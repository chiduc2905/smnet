"""Run USCMambaNet experiments for 1-shot and 5-shot with various training sample sizes."""
import subprocess
import sys
import argparse
import os

from net.model_factory import get_model_choices, get_model_metadata

def get_args():
    parser = argparse.ArgumentParser(description='Run all USCMambaNet experiments')
    parser.add_argument('--project', type=str, default='uscmamba', help='WandB project name')
    parser.add_argument('--dataset_path', type=str, 
                        default='/mnt/disk2/nhatnc/res/scalogram_fewshot/proposed_model/smnet/scalogram_27_1',
                        help='Path to dataset')
    parser.add_argument('--dataset_name', type=str, default='knee_aug_split', help='Dataset name for logging')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--shot_num', type=int, default=None, choices=[1, 5],
                        help='Optional fixed shot number. If omitted, run both 1-shot and 5-shot.')
    parser.add_argument('--mode_id', type=int, default=None, choices=list(range(1, 5)),
                        help='Run specific experiment (1-4). If not set, runs all experiments.')
    parser.add_argument('--models', type=str, default='uscmamba',
                        help='Comma-separated model registry names to run (default: uscmamba)')
    parser.add_argument('--gpu_id', type=int, default=0, help='CUDA device id to pass through')
    return parser.parse_args()


# Experiment modes mapping: mode_id -> samples
# Mode 1: 60 samples
# Mode 2: 160 samples
# Mode 3: 240 samples
# Mode 4: All samples
EXPERIMENT_MODES = {
    1: 60,
    2: 160,
    3: 240,
    4: None,  # None = All samples
}


# Configuration
# Training samples: [small, medium, large, all]
SAMPLES_LIST = [60, 160, 240, None]
SHOTS_DEFAULT = [1, 5]

# Query samples
TRAIN_QUERY_NUM = 1
EVAL_QUERY_NUM = 1

def run_experiment(model, shot, samples, dataset_path, dataset_name, project, seed, gpu_id):
    """Run a single SMNet experiment."""
    print(f"\n{'='*60}")
    print(f"Model={model}, Shot={shot}, Samples={samples if samples else 'All'}")
    print('='*60)
    
    cmd = [
        sys.executable, 'main.py',
        '--model', model,
        '--shot_num', str(shot),
        '--way_num', '4',
        '--query_num_train', str(TRAIN_QUERY_NUM),
        '--query_num_val', str(EVAL_QUERY_NUM),
        '--query_num_test', str(EVAL_QUERY_NUM),
        '--image_size', '64',
        '--mode', 'train',
        '--project', project,
        '--dataset_path', dataset_path,
        '--dataset_name', dataset_name,
        '--num_epochs', '100',
        '--lr', '5e-4',
        '--step_size', '10',
        '--gamma', '0.5',
        '--weight_decay', '5e-4',
        '--temperature', '16.0',
        '--beta_maha', '0.25',
        '--uaps_eps', '1e-4',
        '--cross_attn_alpha', '0.3',
        '--delta_lambda', '0.35',
        '--grad_clip', '2.0',
        '--use_pair_expert', 'false',
        '--use_unified_attention', 'false',
        '--use_ms_global', 'true',
        '--ms_downsample', '2',
        '--atrous_rate', '2',
        '--use_late_attention', 'true',
        '--late_attn_window', '4',
        '--late_attn_dropout', '0.0',
        '--use_axis_proto', 'false',
        '--axis_proto_pool', 'mean',
        '--axis_proto_mix_init', '1.0,0.5,0.5',
        '--use_cross_attention', 'false',
        '--episode_num_train', '200',
        '--episode_num_val', '300',
        '--episode_num_test', '300',
        '--seed', str(seed),
        '--gpu_id', str(gpu_id),
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
    shots = [args.shot_num] if args.shot_num is not None else SHOTS_DEFAULT
    requested_models = [m.strip() for m in args.models.split(',') if m.strip()]
    valid_models = set(get_model_choices())
    invalid_models = [m for m in requested_models if m not in valid_models]
    if invalid_models:
        raise ValueError(f"Unsupported models: {invalid_models}. Valid choices: {sorted(valid_models)}")
    
    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Determine experiments to run
    if args.mode_id is not None:
        # Run experiment(s) for a single sample mode
        samples = EXPERIMENT_MODES[args.mode_id]
        experiments = [(model, samples, shot) for model in requested_models for shot in shots]
        print("=" * 60)
        print(f"USCMambaNet - Single Experiment (Mode {args.mode_id})")
        print("=" * 60)
        print(f"  Samples: {samples if samples else 'All'}")
        print(f"  Model(s): {', '.join(requested_models)}")
        print(f"  Shot(s): {', '.join(map(str, shots))}")
        print(f"  Dataset: {args.dataset_path}")
        print("=" * 60)
    else:
        # Run all experiments
        experiments = [
            (model, samples, shot)
            for model in requested_models
            for samples in SAMPLES_LIST
            for shot in shots
        ]
        print("=" * 60)
        print("USCMambaNet - Full Experiment Suite")
        print("=" * 60)
        print("Mode mapping:")
        for mid, s in EXPERIMENT_MODES.items():
            print(f"  Mode {mid}: {s if s else 'All'} samples")
        print(f"Models: {', '.join(requested_models)}")
        print(f"Shots: {', '.join(f'{s}-shot' for s in shots)}")
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
            seed=args.seed,
            gpu_id=args.gpu_id,
        )
        
        if success:
            success_count += 1
        else:
            failed_experiments.append(f"{model}_{shot}shot_{samples if samples else 'all'}samples")
    
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
    generate_comparison_charts(args.dataset_name, shots, requested_models)
    
    print("\nAll experiments completed!")


def generate_comparison_charts(dataset_name, shots, models):
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
        model_name: get_model_metadata(model_name)["display_name"]
        for model_name in models
    }
    
    for samples in SAMPLES_LIST:
        samples_str = f"{samples}samples" if samples is not None else "allsamples"
        
        model_results = {}
        
        for model in models:
            display_name = model_display_names.get(model, model)
            model_results[display_name] = {}
            
            for shot in shots:
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
            if all(f'{shot}shot' in shots_dict for shot in shots):
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
