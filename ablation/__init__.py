"""Ablation Study Package.

Contains ablation configurations and utilities for SMNet components:

1. ablation_dual_branch.py:    DualBranchFusion (local/global/both)
2. ablation_slot_refinement.py: SCA + CMA modules (none/sca/cma/both)
3. ablation_slot_attention.py:  SlotAttention (with/without)
"""

from .ablation_dual_branch import (
    DualBranchMode,
    DualBranchAblationConfig,
    DualBranchFusionAblation,
    get_ablation_configs as get_dual_branch_configs
)

from .ablation_slot_refinement import (
    SlotRefinementMode,
    SlotRefinementAblationConfig,
    SlotRefinementAblation,
    get_ablation_configs as get_slot_refinement_configs
)

from .ablation_slot_attention import (
    SlotAttentionMode,
    SlotAttentionAblationConfig,
    SlotAttentionAblation,
    NoSlotAttention,
    get_ablation_configs as get_slot_attention_configs
)


__all__ = [
    # DualBranch
    'DualBranchMode',
    'DualBranchAblationConfig',
    'DualBranchFusionAblation',
    'get_dual_branch_configs',
    
    # Slot Refinement
    'SlotRefinementMode',
    'SlotRefinementAblationConfig', 
    'SlotRefinementAblation',
    'get_slot_refinement_configs',
    
    # Slot Attention
    'SlotAttentionMode',
    'SlotAttentionAblationConfig',
    'SlotAttentionAblation',
    'NoSlotAttention',
    'get_slot_attention_configs',
]
