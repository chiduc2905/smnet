"""Model registry for the few-shot benchmark architectures."""

from __future__ import annotations

from typing import Callable, Dict, List

import torch

from net.class_memory_scan_mamba_net import ClassMemoryScanMambaNet
from net.conv64f_token_sw_metric_net import Conv64FTokenSWMetricNet
from net.episodic_selective_scan_mamba_net import EpisodicSelectiveScanMambaNet
from net.hierarchical_episodic_ssm_net import HierarchicalEpisodicSSMNet
from net.permutation_robust_class_memory_mamba_net import PermutationRobustClassMemoryMambaNet
from net.transport_evidence_mamba_net import TransportEvidenceMambaNet
from net.transport_prior_replay_mamba_net import TransportPriorReplayMambaNet
from net.usc_mamba_net import USCMambaNet


def _bool_flag(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def _device_string(args) -> str:
    if getattr(args, "device", None):
        return str(args.device)
    return "cuda" if torch.cuda.is_available() else "cpu"


def _axis_proto_mix_init(value: object) -> tuple[float, float, float]:
    if isinstance(value, (tuple, list)):
        if len(value) != 3:
            raise ValueError("axis_proto_mix_init must have exactly 3 values")
        return tuple(float(v) for v in value)
    parts = [p.strip() for p in str(value).split(",")]
    if len(parts) != 3:
        raise ValueError("--axis_proto_mix_init must have exactly 3 comma-separated values")
    return tuple(float(v) for v in parts)


def build_uscmamba_from_args(args) -> USCMambaNet:
    """Build the existing baseline without changing its behavior."""
    device = torch.device(_device_string(args))
    use_unified = _bool_flag(getattr(args, "use_unified_attention", "false"))
    use_cross = _bool_flag(getattr(args, "use_cross_attention", "false"))
    use_pair_expert = _bool_flag(getattr(args, "use_pair_expert", "false"))
    use_ms_global = _bool_flag(getattr(args, "use_ms_global", "true"), default=True)
    use_late_attention = _bool_flag(getattr(args, "use_late_attention", "true"), default=True)
    use_axis_proto = _bool_flag(getattr(args, "use_axis_proto", "false"))

    if use_cross:
        print("Info: --use_cross_attention=true requested, but cross-attention is disabled in current model.")
    use_cross = False
    if not use_cross:
        use_axis_proto = False

    axis_proto_mix_init = _axis_proto_mix_init(getattr(args, "axis_proto_mix_init", "1.0,0.5,0.5"))

    model = USCMambaNet(
        in_channels=3,
        hidden_dim=args.hidden_dim,
        d_state=args.d_state,
        global_expand=args.global_expand,
        temperature=args.temperature,
        beta_maha=args.beta_maha,
        uaps_eps=args.uaps_eps,
        cross_attn_alpha=args.cross_attn_alpha,
        proto_pool_size=args.proto_pool_size,
        num_prototypes=args.num_prototypes,
        detach_prototypes=args.detach_prototypes,
        use_axis_proto=use_axis_proto,
        axis_proto_pool=args.axis_proto_pool,
        axis_proto_mix_init=axis_proto_mix_init,
        use_late_attention=use_late_attention,
        late_attn_window=args.late_attn_window,
        late_attn_dropout=args.late_attn_dropout,
        similarity_proj_dim=args.similarity_proj_dim,
        delta_lambda=args.delta_lambda,
        way_num=args.way_num,
        use_pair_expert=use_pair_expert,
        use_ms_global=use_ms_global,
        ms_downsample=args.ms_downsample,
        atrous_rate=args.atrous_rate,
        use_projection=not args.no_projection,
        dualpath_mode=args.dualpath_mode,
        use_unified_attention=use_unified,
        use_cross_attention=use_cross,
        device=str(device),
    )
    return model.to(device)


def build_conv64f_token_sw_metric_net_from_args(args) -> Conv64FTokenSWMetricNet:
    return Conv64FTokenSWMetricNet(
        in_channels=3,
        hidden_dim=args.hidden_dim,
        token_dim=getattr(args, "token_dim", None),
        temperature=args.temperature,
        conv64f_pool_last=_bool_flag(getattr(args, "conv64f_pool_last", "true"), default=True),
        sw_num_projections=args.sw_num_projections,
        sw_p=args.sw_p,
        sw_normalize=_bool_flag(getattr(args, "sw_normalize", "true"), default=True),
        token_merge_mode=args.token_merge_mode,
        token_metric_mode=args.token_metric_mode,
        global_metric=args.global_metric,
        global_metric_weight=args.global_metric_weight,
    ).to(_device_string(args))


def build_class_memory_scan_mamba_net_from_args(args) -> ClassMemoryScanMambaNet:
    return ClassMemoryScanMambaNet(
        in_channels=3,
        hidden_dim=args.hidden_dim,
        token_dim=getattr(args, "token_dim", None),
        ssm_state_dim=args.ssm_state_dim,
        ssm_depth=args.ssm_depth,
        temperature=args.temperature,
        conv64f_pool_last=_bool_flag(getattr(args, "conv64f_pool_last", "true"), default=True),
        use_sw=_bool_flag(getattr(args, "use_sw", "true"), default=True),
        sw_weight=args.sw_weight,
        sw_num_projections=args.sw_num_projections,
        sw_p=args.sw_p,
        sw_normalize=_bool_flag(getattr(args, "sw_normalize", "true"), default=True),
        token_merge_mode=args.token_merge_mode,
    ).to(_device_string(args))


def build_episodic_selective_scan_mamba_net_from_args(args) -> EpisodicSelectiveScanMambaNet:
    return EpisodicSelectiveScanMambaNet(
        in_channels=3,
        hidden_dim=args.hidden_dim,
        token_dim=getattr(args, "token_dim", None),
        ssm_state_dim=args.ssm_state_dim,
        temperature=args.temperature,
        conv64f_pool_last=_bool_flag(getattr(args, "conv64f_pool_last", "true"), default=True),
        use_sw=_bool_flag(getattr(args, "use_sw", "true"), default=True),
        sw_weight=args.sw_weight,
        sw_num_projections=args.sw_num_projections,
        sw_p=args.sw_p,
        sw_normalize=_bool_flag(getattr(args, "sw_normalize", "true"), default=True),
        token_merge_mode=args.token_merge_mode,
        use_role_embedding=_bool_flag(getattr(args, "use_role_embedding", "true"), default=True),
        use_boundary_gate=_bool_flag(getattr(args, "use_boundary_gate", "true"), default=True),
        max_episode_positions=args.max_episode_positions,
        max_way_num=args.max_way_num,
    ).to(_device_string(args))


def build_permutation_robust_class_memory_mamba_net_from_args(
    args,
) -> PermutationRobustClassMemoryMambaNet:
    return PermutationRobustClassMemoryMambaNet(
        in_channels=3,
        hidden_dim=args.hidden_dim,
        token_dim=getattr(args, "token_dim", None),
        ssm_state_dim=args.ssm_state_dim,
        ssm_depth=args.ssm_depth,
        temperature=args.temperature,
        conv64f_pool_last=_bool_flag(getattr(args, "conv64f_pool_last", "true"), default=True),
        use_sw=_bool_flag(getattr(args, "use_sw", "true"), default=True),
        sw_weight=args.sw_weight,
        sw_num_projections=args.sw_num_projections,
        sw_p=args.sw_p,
        sw_normalize=_bool_flag(getattr(args, "sw_normalize", "true"), default=True),
        token_merge_mode=args.token_merge_mode,
        num_support_permutations=args.num_support_permutations,
        permutation_consistency_weight=args.permutation_consistency_weight,
    ).to(_device_string(args))


def build_hierarchical_episodic_ssm_net_from_args(args) -> HierarchicalEpisodicSSMNet:
    return HierarchicalEpisodicSSMNet(
        in_channels=3,
        hidden_dim=args.hidden_dim,
        token_dim=getattr(args, "token_dim", None),
        ssm_state_dim=args.ssm_state_dim,
        temperature=args.temperature,
        conv64f_pool_last=_bool_flag(getattr(args, "conv64f_pool_last", "true"), default=True),
        use_sw=_bool_flag(getattr(args, "use_sw", "true"), default=True),
        sw_weight=args.sw_weight,
        sw_num_projections=args.sw_num_projections,
        sw_p=args.sw_p,
        sw_normalize=_bool_flag(getattr(args, "sw_normalize", "true"), default=True),
        token_merge_mode=args.token_merge_mode,
        hierarchical_token_depth=args.hierarchical_token_depth,
        hierarchical_shot_depth=args.hierarchical_shot_depth,
    ).to(_device_string(args))


def build_transport_prior_replay_mamba_net_from_args(args) -> TransportPriorReplayMambaNet:
    return TransportPriorReplayMambaNet(
        in_channels=3,
        hidden_dim=args.hidden_dim,
        token_dim=getattr(args, "token_dim", None),
        ssm_state_dim=args.ssm_state_dim,
        temperature=args.temperature,
        conv64f_pool_last=_bool_flag(getattr(args, "conv64f_pool_last", "true"), default=True),
        num_support_atoms=int(getattr(args, "num_support_atoms", 4)),
        num_prior_atoms=int(getattr(args, "num_prior_atoms", 4)),
        prior_bank_size=int(getattr(args, "prior_bank_size", 16)),
        prior_bank_atoms_per_entry=int(getattr(args, "prior_bank_atoms_per_entry", 4)),
        prior_bank_topk=int(getattr(args, "prior_bank_topk", 4)),
        sw_num_projections=args.sw_num_projections,
        sw_p=args.sw_p,
        trajectory_transport_weight=float(getattr(args, "trajectory_transport_weight", 8.0)),
        confidence_logit_weight=float(getattr(args, "confidence_logit_weight", 0.5)),
    ).to(_device_string(args))


def build_transport_evidence_mamba_net_from_args(args) -> TransportEvidenceMambaNet:
    return TransportEvidenceMambaNet(
        in_channels=3,
        hidden_dim=args.hidden_dim,
        token_dim=getattr(args, "token_dim", None),
        evidence_dim=getattr(args, "tem_evidence_dim", None),
        ssm_state_dim=args.ssm_state_dim,
        ssm_depth=args.ssm_depth,
        temperature=args.temperature,
        conv64f_pool_last=_bool_flag(getattr(args, "conv64f_pool_last", "true"), default=True),
        sw_num_projections=args.sw_num_projections,
        sw_p=args.sw_p,
        sw_normalize=_bool_flag(getattr(args, "sw_normalize", "true"), default=True),
        use_transport_metrics=_bool_flag(getattr(args, "use_sw", "true"), default=True),
        token_merge_mode=args.token_merge_mode,
        serialization_orders=getattr(
            args,
            "tem_serialization_orders",
            "row_major,row_major_reverse,column_major,column_major_reverse",
        ),
        use_delta=_bool_flag(getattr(args, "tem_use_delta", "true"), default=True),
        use_support_context=_bool_flag(getattr(args, "tem_use_support_context", "true"), default=True),
        readout_mode=getattr(args, "tem_readout_mode", "final"),
    ).to(_device_string(args))


MODEL_REGISTRY: Dict[str, Callable] = {
    "uscmamba": build_uscmamba_from_args,
    "conv64f_token_sw_metric_net": build_conv64f_token_sw_metric_net_from_args,
    "class_memory_scan_mamba_net": build_class_memory_scan_mamba_net_from_args,
    "episodic_selective_scan_mamba_net": build_episodic_selective_scan_mamba_net_from_args,
    "permutation_robust_class_memory_mamba_net": build_permutation_robust_class_memory_mamba_net_from_args,
    "hierarchical_episodic_ssm_net": build_hierarchical_episodic_ssm_net_from_args,
    "transport_prior_replay_mamba_net": build_transport_prior_replay_mamba_net_from_args,
    "transport_evidence_mamba_net": build_transport_evidence_mamba_net_from_args,
}


MODEL_METADATA = {
    "uscmamba": {
        "display_name": "USCMambaNet",
        "architecture": "PatchEmbed → Conv stem → dual-branch fusion → late attention → cosine metric head",
    },
    "conv64f_token_sw_metric_net": {
        "display_name": "Conv64FTokenSWMetricNet",
        "architecture": "Conv64F → spatial tokens → sliced Wasserstein metric (+ optional global pooled distance)",
    },
    "class_memory_scan_mamba_net": {
        "display_name": "ClassMemoryScanMambaNet",
        "architecture": "Conv64F → support write SSM → class memory → query read SSM + auxiliary SW",
    },
    "episodic_selective_scan_mamba_net": {
        "display_name": "EpisodicSelectiveScanMambaNet",
        "architecture": "Conv64F → role-aware support scan → class state → role-aware query readout + SW alignment",
    },
    "permutation_robust_class_memory_mamba_net": {
        "display_name": "PermutationRobustClassMemoryMambaNet",
        "architecture": "Conv64F → multi-permutation class memory scan → permutation-aware readout + SW",
    },
    "hierarchical_episodic_ssm_net": {
        "display_name": "HierarchicalEpisodicSSMNet",
        "architecture": "Conv64F → token-level SSM → shot-level memory SSM → hierarchical query matcher + SW",
    },
    "transport_prior_replay_mamba_net": {
        "display_name": "TPR-MambaNet",
        "architecture": "Conv64F → multi-atom support prior calibration → replay-controlled query reader → trajectory transport head",
    },
    "transport_evidence_mamba_net": {
        "display_name": "TEM-Mamba",
        "architecture": "Conv64F → token serialization → prefix transport evidence → selective evidence reader → shared scalar scorer",
    },
}


def build_model_from_args(args):
    if args.model not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model: {args.model}")
    return MODEL_REGISTRY[args.model](args)


def get_model_metadata(model_name: str) -> Dict[str, str]:
    return MODEL_METADATA.get(model_name, {"display_name": model_name, "architecture": model_name})


def get_model_choices() -> List[str]:
    return list(MODEL_REGISTRY.keys())
