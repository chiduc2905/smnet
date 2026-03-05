"""Run 32-sample experiments (1-shot, 5-shot) and export misclassified-file reports."""
import argparse
import csv
import os
import subprocess
import sys


def get_args():
    parser = argparse.ArgumentParser(
        description="Run USCMambaNet for 32 samples (1-shot, 5-shot) with misclassification reports"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="uscmamba",
        help="WandB project name",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/mnt/disk2/nhatnc/res/scalogram_fewshot/proposed_model/smnet/scalogram_27_1",
        help="Path to dataset",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="knee_aug_split",
        help="Dataset name for naming outputs",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--misclf_topk",
        type=int,
        default=30,
        help="Top-K misclassified files to keep in CSV",
    )
    parser.add_argument(
        "--only_shot",
        type=int,
        choices=[1, 5],
        default=None,
        help="Run a single shot only",
    )
    return parser.parse_args()


def run_experiment(shot, args):
    print(f"\n{'=' * 60}")
    print(f"Running: 32 samples | {shot}-shot")
    print(f"{'=' * 60}")

    cmd = [
        sys.executable,
        "main.py",
        "--model",
        "uscmamba",
        "--shot_num",
        str(shot),
        "--way_num",
        "4",
        "--query_num_train",
        "1",
        "--query_num_val",
        "1",
        "--query_num_test",
        "1",
        "--image_size",
        "64",
        "--mode",
        "train",
        "--project",
        args.project,
        "--dataset_path",
        args.dataset_path,
        "--dataset_name",
        args.dataset_name,
        "--num_epochs",
        "100",
        "--lr",
        "1e-3",
        "--eta_min",
        "1e-5",
        "--weight_decay",
        "5e-4",
        "--temperature",
        "16.0",
        "--beta_maha",
        "0.25",
        "--uaps_eps",
        "1e-4",
        "--cross_attn_alpha",
        "0.3",
        "--delta_lambda",
        "0.35",
        "--grad_clip",
        "2.0",
        "--margin_type",
        "none",
        "--lambda_margin",
        "0.0",
        "--lambda_center",
        "0.0",
        "--lambda_pair",
        "0.0",
        "--use_pair_expert",
        "false",
        "--lambda_pair_expert",
        "0.0",
        "--hard_mining_ratio",
        "0.0",
        "--use_unified_attention",
        "true",
        "--use_ms_global",
        "true",
        "--ms_downsample",
        "2",
        "--atrous_rate",
        "2",
        "--use_late_attention",
        "true",
        "--late_attn_window",
        "4",
        "--late_attn_dropout",
        "0.0",
        "--use_axis_proto",
        "true",
        "--axis_proto_pool",
        "mean",
        "--axis_proto_mix_init",
        "1.0,0.5,0.5",
        "--episode_num_train",
        "200",
        "--episode_num_val",
        "300",
        "--episode_num_test",
        "300",
        "--training_samples",
        "32",
        "--seed",
        str(args.seed),
        "--save_misclf_report",
        "--misclf_topk",
        str(args.misclf_topk),
    ]

    subprocess.run(cmd, check=True)


def print_top_misclassified(dataset_name, shot):
    samples_str = "32samples"
    path = os.path.join(
        "results",
        f"misclassified_files_{dataset_name}_uscmamba_{samples_str}_{shot}shot.csv",
    )
    if not os.path.exists(path):
        print(f"Misclassification CSV not found: {path}")
        return

    print(f"\nTop misclassified files ({shot}-shot):")
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 10:
                break
            print(
                f"{i+1:>2}. mis={row['mis_count']}/{row['seen_count']} "
                f"({float(row['mis_rate']):.2%}) | {row['top_confusion_true']} -> {row['top_confusion_pred']} | "
                f"{row['file_path']}"
            )


def main():
    args = get_args()
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    shots = [args.only_shot] if args.only_shot is not None else [1, 5]
    failed = []

    for shot in shots:
        try:
            run_experiment(shot, args)
            print_top_misclassified(args.dataset_name, shot)
        except subprocess.CalledProcessError:
            failed.append(shot)

    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"{'=' * 60}")
    if failed:
        print(f"Failed shots: {failed}")
    else:
        print("All requested shots completed successfully.")


if __name__ == "__main__":
    main()
