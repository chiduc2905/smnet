"""Few-shot benchmark training and evaluation entrypoint for SMNet models."""

import argparse
import csv
import os
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from sklearn.metrics import precision_recall_fscore_support
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from thop import clever_format, profile

    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False

from dataloader.dataloader import FewshotDataset
from function.debug_utils import print_grad_norm, print_logit_stats
from function.function import plot_confusion_matrix, plot_training_curves, plot_tsne, seed_func
from net.model_factory import build_model_from_args, get_model_choices, get_model_metadata


def _bool_flag(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def resolve_runtime_device(gpu_id):
    if not torch.cuda.is_available() or gpu_id < 0:
        return "cpu"
    device_count = torch.cuda.device_count()
    if gpu_id >= device_count:
        raise ValueError(f"gpu_id={gpu_id} out of range. Available CUDA devices: 0..{device_count - 1}")
    return f"cuda:{gpu_id}"


def get_args():
    parser = argparse.ArgumentParser(description="SMNet few-shot benchmark")

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/mnt/disk2/nhatnc/res/scalogram_fewshot/proposed_model/smnet/scalogram_27_1",
    )
    parser.add_argument("--dataset_name", type=str, default="knee_aug_split")
    parser.add_argument("--path_weights", type=str, default="checkpoints/")
    parser.add_argument("--path_results", type=str, default="results/")
    parser.add_argument("--weights", type=str, default=None, help="Checkpoint path for test mode")
    parser.add_argument("--project", type=str, default="uscmamba")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--gpu_id", type=int, default=0, help="CUDA device id, negative for CPU")
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--model", type=str, default="uscmamba", choices=get_model_choices())
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--token_dim", type=int, default=None)
    parser.add_argument("--conv64f_pool_last", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--temperature", type=float, default=16.0)

    parser.add_argument("--d_state", type=int, default=8)
    parser.add_argument("--global_expand", type=int, default=2)
    parser.add_argument("--beta_maha", type=float, default=0.25)
    parser.add_argument("--uaps_eps", type=float, default=1e-4)
    parser.add_argument("--cross_attn_alpha", type=float, default=0.3)
    parser.add_argument("--proto_pool_size", type=int, default=12)
    parser.add_argument("--num_prototypes", type=int, default=2)
    parser.add_argument("--detach_prototypes", action="store_true")
    parser.add_argument("--similarity_proj_dim", type=int, default=None)
    parser.add_argument("--delta_lambda", type=float, default=0.35)
    parser.add_argument("--dualpath_mode", type=str, default="both", choices=["local_only", "global_only", "both"])
    parser.add_argument("--use_unified_attention", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--use_cross_attention", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--use_pair_expert", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--use_ms_global", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--ms_downsample", type=int, default=2)
    parser.add_argument("--atrous_rate", type=int, default=2)
    parser.add_argument("--use_late_attention", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--late_attn_window", type=int, default=4)
    parser.add_argument("--late_attn_dropout", type=float, default=0.0)
    parser.add_argument("--use_axis_proto", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--axis_proto_pool", type=str, default="mean", choices=["mean", "max"])
    parser.add_argument("--axis_proto_mix_init", type=str, default="1.0,0.5,0.5")
    parser.add_argument("--no_projection", action="store_true")

    parser.add_argument("--ssm_state_dim", type=int, default=16)
    parser.add_argument("--ssm_depth", type=int, default=1)
    parser.add_argument("--use_sw", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--sw_weight", type=float, default=0.25)
    parser.add_argument("--sw_num_projections", type=int, default=64)
    parser.add_argument("--sw_p", type=float, default=2.0)
    parser.add_argument("--sw_normalize", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--token_merge_mode", type=str, default="concat", choices=["concat", "mean"])
    parser.add_argument("--token_metric_mode", type=str, default="token_only", choices=["token_only", "token_plus_global"])
    parser.add_argument("--global_metric", type=str, default="cosine", choices=["cosine", "sqeuclidean"])
    parser.add_argument("--global_metric_weight", type=float, default=1.0)
    parser.add_argument("--use_role_embedding", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--use_boundary_gate", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--max_episode_positions", type=int, default=32)
    parser.add_argument("--max_way_num", type=int, default=32)
    parser.add_argument("--num_support_permutations", type=int, default=3)
    parser.add_argument("--permutation_consistency_weight", type=float, default=0.1)
    parser.add_argument("--hierarchical_token_depth", type=int, default=1)
    parser.add_argument("--hierarchical_shot_depth", type=int, default=1)

    parser.add_argument("--num_support_atoms", type=int, default=4)
    parser.add_argument("--num_prior_atoms", type=int, default=4)
    parser.add_argument("--prior_bank_size", type=int, default=16)
    parser.add_argument("--prior_bank_atoms_per_entry", type=int, default=4)
    parser.add_argument("--prior_bank_topk", type=int, default=4)
    parser.add_argument("--trajectory_transport_weight", type=float, default=8.0)
    parser.add_argument("--confidence_logit_weight", type=float, default=0.5)

    parser.add_argument("--tem_evidence_dim", type=int, default=None)
    parser.add_argument(
        "--tem_serialization_orders",
        type=str,
        default="row_major,row_major_reverse,column_major,column_major_reverse",
    )
    parser.add_argument("--tem_use_delta", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--tem_use_support_context", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--tem_readout_mode", type=str, default="final", choices=["final", "mean"])

    parser.add_argument("--way_num", type=int, default=4)
    parser.add_argument("--shot_num", type=int, default=1)
    parser.add_argument("--query_num_train", type=int, default=1)
    parser.add_argument("--query_num_val", type=int, default=1)
    parser.add_argument("--query_num_test", type=int, default=1)
    parser.add_argument("--selected_classes", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--training_samples", type=int, default=None)
    parser.add_argument("--episode_num_train", type=int, default=200)
    parser.add_argument("--episode_num_val", type=int, default=300)
    parser.add_argument("--episode_num_test", type=int, default=300)

    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--grad_clip", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--save_misclf_report", action="store_true")
    parser.add_argument("--misclf_topk", type=int, default=30)

    return parser.parse_args()


def get_model(args):
    model = build_model_from_args(args)
    meta = get_model_metadata(args.model)
    print("\nModel Config:")
    print(f"  model: {meta['display_name']}")
    print(f"  architecture: {meta['architecture']}")
    return model


def evaluate(net, loader, args):
    device = torch.device(args.device)
    net.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for query, q_labels, support, _ in loader:
            batch_size = query.shape[0]
            channels, height, width = query.shape[2], query.shape[3], query.shape[4]
            support = support.view(batch_size, args.way_num, args.shot_num, channels, height, width).to(device)
            query = query.to(device)
            targets = q_labels.view(-1).to(device)

            scores = net(query, support)
            loss = F.cross_entropy(scores, targets)
            preds = scores.argmax(dim=1)

            total_loss += loss.item()
            total += targets.size(0)
            correct += (preds == targets).sum().item()

    avg_loss = total_loss / max(1, len(loader))
    acc = correct / total if total > 0 else 0.0
    return acc, avg_loss


def train_loop(net, train_X, train_y, val_X, val_y, args):
    device = torch.device(args.device)
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    best_acc = 0.0
    val_seed = args.seed + 1

    print(
        "Training protocol: "
        f"epochs={args.num_epochs}, train_episodes={args.episode_num_train}, "
        f"val_episodes={args.episode_num_val}, test_episodes={args.episode_num_test}, "
        f"way={args.way_num}, shot={args.shot_num}, "
        f"query(train/val/test)=({args.query_num_train}/{args.query_num_val}/{args.query_num_test}), "
        f"seed(train)=args.seed+epoch, seed(val)={val_seed}, "
        f"optimizer=AdamW, scheduler=StepLR(step_size={args.step_size}, gamma={args.gamma}), lr={args.lr:.2e}"
    )

    for epoch in range(1, args.num_epochs + 1):
        train_seed = args.seed + epoch
        train_ds = FewshotDataset(
            train_X,
            train_y,
            args.episode_num_train,
            args.way_num,
            args.shot_num,
            args.query_num_train,
            seed=train_seed,
        )
        train_gen = torch.Generator().manual_seed(train_seed)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=train_gen)

        net.train()
        total_loss = 0.0
        train_total = 0
        train_correct = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}")
        for step, (query, q_labels, support, _) in enumerate(pbar):
            optimizer.zero_grad()

            batch_size = query.shape[0]
            channels, height, width = query.shape[2], query.shape[3], query.shape[4]
            support = support.view(batch_size, args.way_num, args.shot_num, channels, height, width).to(device)
            query = query.to(device)
            targets = q_labels.view(-1).to(device)

            scores = net(query, support)
            loss = F.cross_entropy(scores, targets)
            preds = scores.argmax(dim=1)

            train_total += targets.size(0)
            train_correct += (preds == targets).sum().item()

            loss.backward()
            if args.debug and step == 0:
                print_grad_norm(net, epoch, step, print_every=1)
                print_logit_stats(scores, step, print_every=1)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        scheduler.step()

        val_ds = FewshotDataset(
            val_X,
            val_y,
            args.episode_num_val,
            args.way_num,
            args.shot_num,
            args.query_num_val,
            seed=val_seed,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            worker_init_fn=lambda worker_id: seed_func(val_seed + worker_id),
        )

        avg_loss = total_loss / max(1, len(train_loader))
        train_acc = train_correct / train_total if train_total > 0 else 0.0
        val_acc, val_loss = evaluate(net, val_loader, args)

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_loss"].append(avg_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch {epoch}: Loss={avg_loss:.4f}, Train={train_acc:.4f}, "
            f"Val={val_acc:.4f} (gap={train_acc - val_acc:+.4f})"
        )
        wandb.log(
            {
                "epoch": epoch,
                "loss/train": avg_loss,
                "loss/val": val_loss,
                "accuracy/train": train_acc,
                "accuracy/val": val_acc,
                "train_val_gap": train_acc - val_acc,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        if val_acc > best_acc:
            best_acc = val_acc
            wandb.run.summary["best_val_acc"] = best_acc
            best_path = get_best_model_path(args)
            torch.save(net.state_dict(), best_path)
            print(f"  Saved best model to {best_path}")

    curves_path = os.path.join(
        args.path_results,
        f"training_{args.dataset_name}_{args.model}_{sample_suffix(args)}_{args.shot_num}shot",
    )
    try:
        plot_training_curves(history, curves_path)
        if os.path.exists(f"{curves_path}_curves.png"):
            wandb.log({"training_curves": wandb.Image(f"{curves_path}_curves.png")})
    except RuntimeError as exc:
        print(f"Skipping training curves: {exc}")

    print(f"Best Validation Accuracy: {best_acc:.4f}")
    return best_acc


def calculate_p_value(acc, baseline, n):
    from scipy.stats import norm

    if n <= 0:
        return 1.0
    z = (acc - baseline) / np.sqrt(baseline * (1 - baseline) / n)
    return 2 * norm.sf(abs(z))


def extract_model_features(net, x):
    if hasattr(net, "extract_features"):
        feat = net.extract_features(x)
    elif hasattr(net, "encode"):
        feat = net.encode(x)
    elif hasattr(net, "encoder"):
        feat = net.encoder(x)
    else:
        raise AttributeError("Model does not expose extract_features/encode/encoder")

    if feat.dim() == 4:
        feat = F.adaptive_avg_pool2d(feat, 1).view(feat.size(0), -1)
    elif feat.dim() > 2:
        feat = feat.view(feat.size(0), -1)
    return F.normalize(feat, p=2, dim=-1)


def test_final(net, loader, args, test_X=None, test_y=None, test_file_paths=None):
    import time

    device = torch.device(args.device)
    num_episodes = len(loader)
    meta = get_model_metadata(args.model)
    print(f"\n{'=' * 60}")
    print(f"Final Test: {meta['display_name']} | {args.dataset_name} | {args.shot_num}-shot")
    print(f"{num_episodes} episodes x {args.way_num} classes x {args.query_num_test} query")
    print("=" * 60)

    net.eval()
    all_preds = []
    all_targets = []
    episode_accuracies = []
    episode_times = []
    query_seen = Counter()
    query_mis = Counter()
    query_pair = defaultdict(Counter)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            if len(batch) == 6:
                query, q_labels, support, _, q_indices, _ = batch
                q_indices_np = q_indices.view(-1).cpu().numpy()
            else:
                query, q_labels, support, _ = batch
                q_indices_np = None

            start_time = time.perf_counter()

            batch_size = query.shape[0]
            channels, height, width = query.shape[2], query.shape[3], query.shape[4]
            support = support.view(batch_size, args.way_num, args.shot_num, channels, height, width).to(device)
            query = query.to(device)
            targets = q_labels.view(-1).to(device)

            scores = net(query, support)
            preds = scores.argmax(dim=1)

            episode_times.append((time.perf_counter() - start_time) * 1000)
            episode_accuracies.append((preds == targets).float().mean().item())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            if q_indices_np is not None:
                preds_np = preds.cpu().numpy()
                targets_np = targets.cpu().numpy()
                for idx_i, true_i, pred_i in zip(q_indices_np, targets_np, preds_np):
                    idx_i = int(idx_i)
                    true_i = int(true_i)
                    pred_i = int(pred_i)
                    query_seen[idx_i] += 1
                    if pred_i != true_i:
                        query_mis[idx_i] += 1
                        query_pair[idx_i][(true_i, pred_i)] += 1

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    episode_accuracies = np.array(episode_accuracies)
    episode_times = np.array(episode_times)

    acc_mean = episode_accuracies.mean()
    acc_std = episode_accuracies.std()
    acc_ci95 = 1.96 * acc_std / np.sqrt(len(episode_accuracies))
    acc_worst = episode_accuracies.min()
    acc_best = episode_accuracies.max()
    time_mean = episode_times.mean()
    time_std = episode_times.std()

    prec, rec, f1, _ = precision_recall_fscore_support(
        all_targets,
        all_preds,
        labels=list(range(args.way_num)),
        average="macro",
        zero_division=0,
    )
    p_val = calculate_p_value(acc_mean, 1.0 / args.way_num, len(all_targets))

    print(f"\n{'=' * 60}")
    print("ACCURACY METRICS")
    print("=" * 60)
    print(f"  Mean Accuracy : {acc_mean * 100:.2f} +/- {acc_ci95 * 100:.2f}% (95% CI)")
    print(f"  Std Deviation : {acc_std * 100:.2f}%")
    print(f"  Worst-case    : {acc_worst * 100:.2f}%")
    print(f"  Best-case     : {acc_best * 100:.2f}%")
    print(f"  Precision     : {prec:.4f}")
    print(f"  Recall        : {rec:.4f}")
    print(f"  F1-Score      : {f1:.4f}")
    print(f"  p-value       : {p_val:.2e}")
    print(f"\nInference Time  : {time_mean:.2f} +/- {time_std:.2f} ms/episode")

    wandb.log(
        {
            "test_accuracy_mean": acc_mean,
            "test_accuracy_std": acc_std,
            "test_accuracy_ci95": acc_ci95,
            "test_accuracy_worst": acc_worst,
            "test_accuracy_best": acc_best,
            "test_precision": prec,
            "test_recall": rec,
            "test_f1": f1,
            "test_p_value": p_val,
            "inference_time_mean_ms": time_mean,
            "inference_time_std_ms": time_std,
        }
    )
    wandb.run.summary["test_accuracy_mean"] = acc_mean
    wandb.run.summary["test_accuracy_ci95"] = acc_ci95

    samples_str = f"_{sample_suffix(args)}"
    cm_base = os.path.join(
        args.path_results,
        f"confusion_matrix_{args.dataset_name}_{args.model}_{samples_str.strip('_')}_{args.shot_num}shot",
    )
    try:
        plot_confusion_matrix(all_targets, all_preds, args.way_num, cm_base, class_names=args.class_names)
        if os.path.exists(f"{cm_base}_2col.png"):
            wandb.log({"confusion_matrix": wandb.Image(f"{cm_base}_2col.png")})
    except RuntimeError as exc:
        print(f"Skipping confusion matrix plot: {exc}")

    if args.save_misclf_report and query_seen:
        save_misclassification_report(args, all_targets, all_preds, query_seen, query_mis, query_pair, test_file_paths)

    if test_X is not None and test_y is not None:
        with torch.no_grad():
            test_X_device = test_X.to(device)
            test_y_np = test_y.cpu().numpy()
            all_features = []
            batch_size = 32
            for start in range(0, len(test_X), batch_size):
                batch_X = test_X_device[start : start + batch_size]
                all_features.append(extract_model_features(net, batch_X).cpu().numpy())
            features = np.vstack(all_features)

        tsne_path = os.path.join(
            args.path_results,
            f"tsne_{args.dataset_name}_{args.model}_{samples_str.strip('_')}_{args.shot_num}shot",
        )
        try:
            plot_tsne(features, test_y_np, args.way_num, tsne_path, class_names=args.class_names)
            if os.path.exists(f"{tsne_path}_tsne.png"):
                wandb.log({"tsne_plot": wandb.Image(f"{tsne_path}_tsne.png")})
        except RuntimeError as exc:
            print(f"Skipping t-SNE plot: {exc}")

    txt_path = os.path.join(
        args.path_results,
        f"results_{args.dataset_name}_{args.model}_{samples_str.strip('_')}_{args.shot_num}shot.txt",
    )
    with open(txt_path, "w") as handle:
        handle.write(f"Model: {meta['display_name']} ({args.model})\n")
        handle.write(f"Dataset: {args.dataset_name}\n")
        handle.write(f"Shot: {args.shot_num}\n")
        handle.write(f"Training Samples: {args.training_samples if args.training_samples else 'All'}\n")
        handle.write("-" * 40 + "\n")
        handle.write(f"Accuracy : {acc_mean:.4f} +/- {acc_std:.4f}\n")
        handle.write(f"Worst-case : {acc_worst:.4f}\n")
        handle.write(f"Best-case : {acc_best:.4f}\n")
        handle.write(f"Precision : {prec:.4f}\n")
        handle.write(f"Recall : {rec:.4f}\n")
        handle.write(f"F1-Score : {f1:.4f}\n")
        handle.write(f"Inference Time: {time_mean:.2f} +/- {time_std:.2f} ms/episode\n")


def save_misclassification_report(args, all_targets, all_preds, query_seen, query_mis, query_pair, test_file_paths):
    samples_str = sample_suffix(args)
    pair_totals = Counter()
    true_totals = Counter()
    for true_i, pred_i in zip(all_targets.tolist(), all_preds.tolist()):
        true_totals[int(true_i)] += 1
        if int(true_i) != int(pred_i):
            pair_totals[(int(true_i), int(pred_i))] += 1

    pair_path = os.path.join(
        args.path_results,
        f"misclass_pairs_{args.dataset_name}_{args.model}_{samples_str}_{args.shot_num}shot.txt",
    )
    with open(pair_path, "w") as handle:
        handle.write("Most common confusion pairs (True -> Pred)\n")
        handle.write("-" * 70 + "\n")
        for (true_i, pred_i), count in pair_totals.most_common():
            denom = max(1, true_totals[true_i])
            rate = count / denom
            true_name = args.class_names[true_i] if true_i < len(args.class_names) else f"Class{true_i}"
            pred_name = args.class_names[pred_i] if pred_i < len(args.class_names) else f"Class{pred_i}"
            handle.write(f"{true_name} -> {pred_name}: {count} ({rate:.2%} of true {true_name})\n")

    rows = []
    for idx_i, seen in query_seen.items():
        mis = query_mis.get(idx_i, 0)
        if mis <= 0:
            continue
        top_pair, top_pair_cnt = query_pair[idx_i].most_common(1)[0]
        true_i, pred_i = top_pair
        file_path = f"index_{idx_i}"
        if test_file_paths is not None and 0 <= idx_i < len(test_file_paths):
            file_path = test_file_paths[idx_i]
        rows.append(
            {
                "file_index": idx_i,
                "file_path": file_path,
                "mis_count": mis,
                "seen_count": seen,
                "mis_rate": mis / max(1, seen),
                "top_confusion_true": args.class_names[true_i],
                "top_confusion_pred": args.class_names[pred_i],
                "top_confusion_count": top_pair_cnt,
            }
        )

    rows.sort(key=lambda item: (-item["mis_count"], -item["mis_rate"], item["file_index"]))
    if args.misclf_topk > 0:
        rows = rows[: args.misclf_topk]

    mis_path = os.path.join(
        args.path_results,
        f"misclassified_files_{args.dataset_name}_{args.model}_{samples_str}_{args.shot_num}shot.csv",
    )
    with open(mis_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "file_index",
                "file_path",
                "mis_count",
                "seen_count",
                "mis_rate",
                "top_confusion_true",
                "top_confusion_pred",
                "top_confusion_count",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def calculate_flops(model, input_size, device):
    if not THOP_AVAILABLE:
        return None, None
    try:
        dummy_input = torch.randn(1, *input_size).to(device)
        if hasattr(model, "encoder"):
            return profile(model.encoder, inputs=(dummy_input,), verbose=False)
    except Exception as exc:
        print(f"Warning: Could not calculate FLOPs: {exc}")
    return None, None


def log_model_parameters(model, args):
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    wandb.log(
        {
            "model/total_parameters": total_params,
            "model/trainable_parameters": trainable_params,
        }
    )
    wandb.run.summary["model_total_params"] = total_params
    wandb.run.summary["model_trainable_params"] = trainable_params

    macs, profile_params = calculate_flops(model, (3, args.image_size, args.image_size), args.device)
    if macs is not None:
        flops = macs * 2
        macs_readable, params_readable = clever_format([macs, profile_params], "%.2f")
        flops_readable = clever_format([flops], "%.2f")[0]
        wandb.log(
            {
                "model/macs": macs,
                "model/flops": flops,
                "model/macs_readable": macs_readable,
                "model/flops_readable": flops_readable,
                "model/profile_params_readable": params_readable,
            }
        )


def sample_suffix(args):
    return f"{args.training_samples}samples" if args.training_samples else "allsamples"


def get_best_model_path(args):
    suffix = "all" if args.training_samples is None else f"{args.training_samples}samples"
    return os.path.join(args.path_weights, f"{args.dataset_name}_{args.model}_{suffix}_{args.shot_num}shot_final.pth")


def filter_classes(images, labels, selected_classes, file_paths=None):
    mask = torch.zeros(len(labels), dtype=torch.bool)
    for class_id in selected_classes:
        mask |= labels == class_id
    filtered_images = images[mask]
    filtered_labels = labels[mask]
    label_map = {old: new for new, old in enumerate(selected_classes)}
    remapped = torch.tensor([label_map[label.item()] for label in filtered_labels], dtype=torch.long)
    filtered_paths = None
    if file_paths is not None and len(file_paths) == len(labels):
        filtered_paths = [path for path, keep in zip(file_paths, mask.tolist()) if keep]
    return filtered_images, remapped, filtered_paths


def subsample_train_split(train_X, train_y, args):
    if args.training_samples is None:
        return train_X, train_y
    if args.training_samples % args.way_num != 0:
        raise ValueError(
            f"training_samples ({args.training_samples}) must be divisible by way_num ({args.way_num}) "
            "for balanced class sampling."
        )
    per_class = args.training_samples // args.way_num
    sample_images = []
    sample_labels = []
    for class_id in range(args.way_num):
        indices = (train_y == class_id).nonzero(as_tuple=True)[0]
        if len(indices) < per_class:
            raise ValueError(f"Class {class_id}: need {per_class}, have {len(indices)}")
        generator = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(len(indices), generator=generator)[:per_class]
        sample_images.append(train_X[indices[perm]])
        sample_labels.append(train_y[indices[perm]])
    return torch.cat(sample_images), torch.cat(sample_labels)


def main():
    args = get_args()
    args.device = resolve_runtime_device(args.gpu_id)
    model_meta = get_model_metadata(args.model)

    print(f"\n{'=' * 60}")
    print(model_meta["display_name"])
    print("=" * 60)
    print(f"Config: {args.model} | {args.shot_num}-shot | {args.num_epochs} epochs | Device: {args.device}")
    print(f"Architecture: {model_meta['architecture']}")
    print(f"Dataset: {args.dataset_path}")

    run_name = f"{args.model}_{args.dataset_name}_{'all' if args.training_samples is None else f'{args.training_samples}samples'}_{args.shot_num}shot"
    config = vars(args).copy()
    config["architecture"] = model_meta["architecture"]
    wandb.init(project=args.project, config=config, name=run_name, group=f"{args.model}_{args.dataset_name}", job_type=args.mode)

    seed_func(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs(args.path_weights, exist_ok=True)
    os.makedirs(args.path_results, exist_ok=True)

    from dataset import load_dataset

    dataset = load_dataset(args.dataset_path, image_size=args.image_size)

    def to_tensor(images, labels):
        return torch.from_numpy(images.astype(np.float32)), torch.from_numpy(labels).long()

    train_X, train_y = to_tensor(dataset.X_train, dataset.y_train)
    val_X, val_y = to_tensor(dataset.X_val, dataset.y_val)
    test_X, test_y = to_tensor(dataset.X_test, dataset.y_test)
    train_file_paths = [path for path, _ in getattr(dataset, "train_files", [])] if hasattr(dataset, "train_files") else None
    val_file_paths = [path for path, _ in getattr(dataset, "val_files", [])] if hasattr(dataset, "val_files") else None
    test_file_paths = [path for path, _ in getattr(dataset, "test_files", [])] if hasattr(dataset, "test_files") else None

    pretty_map = {"surface": "Surface", "internal": "Internal", "corona": "Corona", "notpd": "NotPD", "nopd": "NotPD"}
    dataset_classes = list(getattr(dataset, "classes", []))
    all_class_names = [pretty_map.get(name.lower(), name) for name in dataset_classes] if dataset_classes else []
    if not all_class_names:
        all_class_names = [f"Class{i}" for i in range(int(len(torch.unique(train_y))))]

    if args.selected_classes:
        selected = [int(class_id.strip()) for class_id in args.selected_classes.split(",")]
        if any(class_id < 0 or class_id >= len(all_class_names) for class_id in selected):
            raise ValueError(f"selected_classes={selected} out of range for classes={all_class_names}")
        args.class_names = [all_class_names[class_id] for class_id in selected]
        args.way_num = len(selected)
        train_X, train_y, train_file_paths = filter_classes(train_X, train_y, selected, train_file_paths)
        val_X, val_y, val_file_paths = filter_classes(val_X, val_y, selected, val_file_paths)
        test_X, test_y, test_file_paths = filter_classes(test_X, test_y, selected, test_file_paths)
    else:
        args.class_names = all_class_names
        if args.way_num != len(args.class_names):
            print(
                f"way_num={args.way_num} does not match dataset classes={len(args.class_names)}. "
                f"Using way_num={len(args.class_names)}."
            )
            args.way_num = len(args.class_names)

    train_X, train_y = subsample_train_split(train_X, train_y, args)
    wandb.config.update(
        {
            "way_num": args.way_num,
            "query_num_train": args.query_num_train,
            "query_num_val": args.query_num_val,
            "query_num_test": args.query_num_test,
        },
        allow_val_change=True,
    )

    test_ds = FewshotDataset(
        test_X,
        test_y,
        args.episode_num_test,
        args.way_num,
        args.shot_num,
        args.query_num_test,
        seed=args.seed,
        return_indices=args.save_misclf_report,
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    net = get_model(args)
    log_model_parameters(net, args)

    if args.mode == "train":
        train_loop(net, train_X, train_y, val_X, val_y, args)
        best_path = get_best_model_path(args)
        print(f"Testing with best checkpoint: {best_path}")
        net.load_state_dict(torch.load(best_path, map_location=args.device))
        test_final(net, test_loader, args, test_X=test_X, test_y=test_y, test_file_paths=test_file_paths)
    else:
        if not args.weights:
            raise ValueError("Please specify --weights in test mode")
        net.load_state_dict(torch.load(args.weights, map_location=args.device))
        test_final(net, test_loader, args, test_X=test_X, test_y=test_y, test_file_paths=test_file_paths)

    wandb.finish()


if __name__ == "__main__":
    main()
