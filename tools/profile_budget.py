"""Profile parameter and train-step budget for USCMamba variants.

Usage:
    python tools/profile_budget.py
    python tools/profile_budget.py --device cuda --steps 50 --warmup 10
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from net.usc_mamba_net import USCMambaNet


def parse_args():
    parser = argparse.ArgumentParser(description="Profile model budget (params + train-step latency)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--steps", type=int, default=50, help="Measured optimization steps")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup optimization steps")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--way_num", type=int, default=4)
    parser.add_argument("--shot_num", type=int, default=1)
    parser.add_argument("--query_num", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--d_state", type=int, default=8)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_variant(variant: str, device: torch.device, args) -> USCMambaNet:
    common = dict(
        in_channels=3,
        hidden_dim=args.hidden_dim,
        d_state=args.d_state,
        global_expand=2,
        way_num=args.way_num,
        use_pair_expert=False,
        use_projection=True,
        dualpath_mode="both",
        use_unified_attention=True,
        use_cross_attention=True,
        temperature=16.0,
        beta_maha=0.25,
        uaps_eps=1e-4,
        cross_attn_alpha=0.3,
        proto_pool_size=12,
        num_prototypes=2,
        detach_prototypes=False,
        device=str(device),
    )

    if variant == "b0":
        return USCMambaNet(
            **common,
            use_ms_global=False,
            use_late_attention=False,
            use_axis_proto=False,
        )
    if variant == "b3":
        return USCMambaNet(
            **common,
            use_ms_global=True,
            ms_downsample=2,
            atrous_rate=2,
            use_late_attention=True,
            late_attn_window=4,
            late_attn_dropout=0.0,
            use_axis_proto=True,
            axis_proto_pool="mean",
            axis_proto_mix_init=(1.0, 0.5, 0.5),
        )
    raise ValueError(f"Unknown variant: {variant}")


def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_step_latency_ms(model: USCMambaNet, device: torch.device, args) -> float:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)

    bsz = 1
    nq = args.way_num * args.query_num
    c = 3
    h = args.image_size
    w = args.image_size
    support_shape = (bsz, args.way_num, args.shot_num, c, h, w)
    query_shape = (bsz, nq, c, h, w)

    targets = torch.arange(args.way_num, device=device).repeat(args.query_num)

    def one_step():
        query = torch.randn(query_shape, device=device)
        support = torch.randn(support_shape, device=device)
        optimizer.zero_grad(set_to_none=True)
        scores = model(query, support)
        loss = F.cross_entropy(scores, targets)
        loss.backward()
        optimizer.step()

    for _ in range(max(0, args.warmup)):
        one_step()

    times = []
    for _ in range(max(1, args.steps)):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        one_step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append((time.perf_counter() - t0) * 1000.0)

    return statistics.median(times)


def main():
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Profiling on device: {device}")

    results = {}
    for variant in ("b0", "b3"):
        try:
            model = build_variant(variant, device, args).to(device)
        except (ImportError, ModuleNotFoundError) as exc:
            print(f"Failed to build {variant.upper()}: missing dependency -> {exc}")
            print("Tip: run this script in your training conda environment with mamba-ssm/timm installed.")
            return
        params = count_trainable_params(model)
        try:
            latency = train_step_latency_ms(model, device, args)
        except (ImportError, ModuleNotFoundError) as exc:
            print(f"Failed while profiling {variant.upper()}: missing dependency -> {exc}")
            print("Tip: run this script in your training conda environment with mamba-ssm/timm installed.")
            return
        results[variant] = {"params": params, "latency_ms": latency}
        print(f"{variant.upper()}: params={params:,} | median_step={latency:.3f} ms")

    p_ratio = results["b3"]["params"] / max(1, results["b0"]["params"])
    t_ratio = results["b3"]["latency_ms"] / max(1e-9, results["b0"]["latency_ms"])

    print("\nRatios (B3 / B0)")
    print(f"  Params ratio : {p_ratio:.4f}x")
    print(f"  Time ratio   : {t_ratio:.4f}x")

    params_ok = p_ratio <= 1.10
    time_ok = t_ratio <= 1.20
    print("\nBudget check")
    print(f"  Params <= 1.10x: {'PASS' if params_ok else 'FAIL'}")
    print(f"  Time <= 1.20x  : {'PASS' if time_ok else 'FAIL'}")

    if not params_ok or not time_ok:
        print("\nSuggested fallback knobs:")
        if not params_ok:
            print("  - Disable atrous branch (set --atrous_rate 1 and bypass atrous fusion)")
        if not time_ok:
            print("  - Reduce --proto_pool_size from 12 to 8")
            print("  - Increase --late_attn_window from 4 to 8")


if __name__ == "__main__":
    main()
