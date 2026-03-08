"""Lightweight validation for the new few-shot benchmark architectures."""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from net.metrics.sliced_wasserstein import SlicedWassersteinDistance, merge_support_tokens_by_class
from net.model_factory import build_model_from_args
from net.ssm.class_memory_scan import ClassMemoryReadSSM, ClassMemoryWriteSSM, ShotDescriptorEncoder
from net.ssm.episodic_selective_scan import EpisodicSelectiveScanBlock


NEW_MODELS = [
    "conv64f_token_sw_metric_net",
    "class_memory_scan_mamba_net",
    "episodic_selective_scan_mamba_net",
    "permutation_robust_class_memory_mamba_net",
    "hierarchical_episodic_ssm_net",
]


def default_args(model_name: str) -> Namespace:
    return Namespace(
        model=model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        hidden_dim=64,
        token_dim=None,
        conv64f_pool_last="true",
        temperature=16.0,
        d_state=8,
        ssm_state_dim=16,
        ssm_depth=1,
        global_expand=2,
        beta_maha=0.25,
        uaps_eps=1e-4,
        cross_attn_alpha=0.3,
        proto_pool_size=12,
        num_prototypes=2,
        detach_prototypes=False,
        axis_proto_pool="mean",
        axis_proto_mix_init="1.0,0.5,0.5",
        similarity_proj_dim=None,
        delta_lambda=0.35,
        way_num=4,
        use_pair_expert="false",
        use_ms_global="true",
        ms_downsample=2,
        atrous_rate=2,
        no_projection=False,
        dualpath_mode="both",
        use_unified_attention="false",
        use_cross_attention="false",
        use_late_attention="true",
        late_attn_window=4,
        late_attn_dropout=0.0,
        use_axis_proto="false",
        use_sw="true",
        sw_weight=0.25,
        sw_num_projections=32,
        sw_p=2.0,
        sw_normalize="true",
        token_merge_mode="concat",
        token_metric_mode="token_plus_global",
        global_metric="cosine",
        global_metric_weight=1.0,
        use_role_embedding="true",
        use_boundary_gate="true",
        max_episode_positions=32,
        max_way_num=32,
        num_support_permutations=3,
        permutation_consistency_weight=0.1,
        hierarchical_token_depth=1,
        hierarchical_shot_depth=1,
    )


def check_sliced_wasserstein(device: torch.device) -> None:
    sw = SlicedWassersteinDistance(num_projections=32, reduction="none", normalize_inputs=True).to(device)
    x = torch.randn(3, 16, 64, device=device)
    y = x.clone()
    z = torch.randn(3, 16, 64, device=device)
    same = sw(x, y, reduction="none")
    diff = sw(x, z, reduction="none")
    assert same.shape == (3,), f"Unexpected SW output shape: {same.shape}"
    assert torch.allclose(same, torch.zeros_like(same), atol=1e-5), f"Identical SW should be ~0, got {same}"
    assert torch.all(diff >= same), "Different distributions should not be closer than identical ones"


def check_support_merge() -> None:
    support = torch.randn(4, 5, 16, 64)
    merged_concat = merge_support_tokens_by_class(support, merge_mode="concat")
    merged_mean = merge_support_tokens_by_class(support, merge_mode="mean")
    assert merged_concat.shape == (4, 80, 64), f"Concat merge shape mismatch: {merged_concat.shape}"
    assert merged_mean.shape == (4, 16, 64), f"Mean merge shape mismatch: {merged_mean.shape}"


def check_class_memory_flow(device: torch.device) -> None:
    token_dim = 64
    encoder = ShotDescriptorEncoder(token_dim, token_dim).to(device)
    writer = ClassMemoryWriteSSM(token_dim, state_dim=16, depth=2).to(device)
    reader = ClassMemoryReadSSM(token_dim, state_dim=16, depth=2).to(device)
    shot_tokens = torch.randn(5, 16, token_dim, device=device)
    shot_globals = torch.randn(5, token_dim, device=device)
    query_tokens = torch.randn(4, 16, token_dim, device=device)
    query_global = torch.randn(4, token_dim, device=device)

    shot_desc = encoder(shot_tokens, shot_globals)
    memory, refined_support = writer(shot_desc)
    readout, refined_query = reader(query_tokens, memory, query_global=query_global)

    assert shot_desc.shape == (5, token_dim)
    assert memory.shape == (token_dim,)
    assert refined_support.shape == (5, token_dim)
    assert readout.shape == (4, token_dim)
    assert refined_query.shape == (4, 16, token_dim)
    assert torch.isfinite(readout).all()


def check_episodic_selective_scan_block(device: torch.device) -> None:
    block = EpisodicSelectiveScanBlock(dim=64, state_dim=16, use_boundary_gate=True).to(device)
    inputs = torch.randn(3, 5, 64, device=device)
    metadata = torch.randn(3, 5, 64, device=device)
    boundary = torch.zeros(3, 5, dtype=torch.long, device=device)
    boundary[:, 0] = 1
    outputs, state = block(inputs, metadata, boundary)
    assert outputs.shape == inputs.shape, f"Episodic scan output mismatch: {outputs.shape}"
    assert state.shape == (3, 16), f"Episodic scan state mismatch: {state.shape}"
    assert torch.isfinite(outputs).all()


def check_model_shape(model_name: str, shot_num: int, device: torch.device) -> None:
    args = default_args(model_name)
    args.device = str(device)
    model = build_model_from_args(args).to(device)
    model.eval()

    bsz = 1
    way_num = 4
    query_num = 1
    nq = way_num * query_num
    query = torch.randn(bsz, nq, 3, 64, 64, device=device)
    support = torch.randn(bsz, way_num, shot_num, 3, 64, 64, device=device)
    with torch.no_grad():
        logits = model(query, support)
    assert logits.shape == (bsz * nq, way_num), f"{model_name} logits mismatch for shot={shot_num}: {logits.shape}"
    assert torch.isfinite(logits).all(), f"{model_name} produced NaN/Inf for shot={shot_num}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate new few-shot benchmark models")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Running checks on {device} ...")
    check_sliced_wasserstein(device)
    print("  [OK] Sliced Wasserstein distance")

    check_support_merge()
    print("  [OK] Support token merge")

    check_class_memory_flow(device)
    print("  [OK] Class-memory write/read flow")

    check_episodic_selective_scan_block(device)
    print("  [OK] Episodic selective scan block")

    for model_name in NEW_MODELS:
        for shot_num in (1, 5):
            check_model_shape(model_name, shot_num, device)
            print(f"  [OK] {model_name} forward shape ({shot_num}-shot)")

    print("All benchmark checks passed.")


if __name__ == "__main__":
    main()
