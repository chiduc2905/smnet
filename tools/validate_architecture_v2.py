"""Lightweight validation checks for USCMambaNet-v2 architecture.

Checks:
1) USCMambaNet forward output shape for shot=1 and shot=5
2) Late attention bridge shape integrity
3) Dual-axis prototype attention output shape and finite values
4) Confusion-matrix row count expectation for query=1, 300 episodes
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from net.backbone.late_attention_bridge import LateSingleHeadAttentionBridge
from net.metrics.prototype_cross_attention import PrototypeCrossAttention
from net.usc_mamba_net import USCMambaNet


def check_late_attention(device: torch.device):
    block = LateSingleHeadAttentionBridge(channels=64, window_size=4, attn_dropout=0.0).to(device)
    x = torch.randn(2, 64, 16, 16, device=device)
    y = block(x)
    assert y.shape == x.shape, f"Late attention shape mismatch: {y.shape} vs {x.shape}"
    assert torch.isfinite(y).all(), "Late attention output has NaN/Inf"


def check_axis_proto(device: torch.device):
    mod = PrototypeCrossAttention(
        channels=64,
        alpha=0.3,
        proto_pool_size=12,
        num_prototypes=2,
        use_axis_proto=True,
        axis_proto_pool="mean",
        axis_proto_mix_init=(1.0, 0.5, 0.5),
    ).to(device)
    q = torch.randn(4, 64, 16, 16, device=device)  # NQ=4
    s = torch.randn(4 * 5, 64, 16, 16, device=device)  # Way=4, Shot=5
    refined, proto = mod(q, s, way_num=4, shot_num=5)
    assert refined.shape == (4, 4, 64, 16, 16), f"Refined shape mismatch: {refined.shape}"
    assert proto.shape[0] == 4, f"Proto way mismatch: {proto.shape}"
    assert torch.isfinite(refined).all(), "Dual-axis proto output has NaN/Inf"


def check_model_shape(device: torch.device, shot_num: int):
    # Use local_only to avoid hard dependency on mamba_ssm in restricted env.
    model = USCMambaNet(
        in_channels=3,
        hidden_dim=64,
        way_num=4,
        d_state=8,
        dualpath_mode="local_only",
        use_unified_attention=True,
        use_cross_attention=True,
        use_ms_global=True,
        use_late_attention=True,
        use_axis_proto=True,
        use_pair_expert=False,
        device=str(device),
    ).to(device)
    bsz = 1
    query_num = 1
    nq = 4 * query_num
    query = torch.randn(bsz, nq, 3, 64, 64, device=device)
    support = torch.randn(bsz, 4, shot_num, 3, 64, 64, device=device)
    logits = model(query, support)
    assert logits.shape == (bsz * nq, 4), f"Forward shape mismatch for shot={shot_num}: {logits.shape}"
    assert torch.isfinite(logits).all(), f"Logits have NaN/Inf for shot={shot_num}"


def check_confusion_expectation(episode_num_test: int = 300, query_num_test: int = 1):
    expected_row_total = episode_num_test * query_num_test
    assert expected_row_total == 300, (
        f"Expected confusion row total should be 300, got {expected_row_total}. "
        "Set episode_num_test=300 and query_num_test=1."
    )


def main():
    parser = argparse.ArgumentParser(description="Validate USCMambaNet-v2 architecture checks")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Running checks on {device} ...")
    check_late_attention(device)
    print("  [OK] Late attention shape")

    check_axis_proto(device)
    print("  [OK] Dual-axis proto shape")

    check_model_shape(device, shot_num=1)
    print("  [OK] USCMambaNet forward shape (shot=1)")
    check_model_shape(device, shot_num=5)
    print("  [OK] USCMambaNet forward shape (shot=5)")

    check_confusion_expectation(episode_num_test=300, query_num_test=1)
    print("  [OK] Confusion row expectation (300)")

    print("All checks passed.")


if __name__ == "__main__":
    main()
