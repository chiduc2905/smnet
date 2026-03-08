"""Permutation-robust class-memory encoding utilities."""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from net.ssm.class_memory_scan import ClassMemoryWriteSSM


class SupportPermutationSampler(nn.Module):
    """Generate a small set of support-shot permutations per class."""

    def __init__(self, num_permutations: int = 3) -> None:
        super().__init__()
        self.num_permutations = max(1, int(num_permutations))

    def forward(self, shot_num: int, device: torch.device, training: bool) -> List[torch.Tensor]:
        identity = torch.arange(shot_num, device=device)
        if shot_num <= 1 or self.num_permutations == 1:
            return [identity]

        permutations = [identity]
        max_extra = self.num_permutations - 1
        if training:
            for _ in range(max_extra):
                permutations.append(torch.randperm(shot_num, device=device))
            return permutations

        candidates = [
            torch.roll(identity, shifts=shift, dims=0)
            for shift in range(1, shot_num)
        ]
        candidates.append(torch.flip(identity, dims=[0]))
        for perm in candidates[:max_extra]:
            permutations.append(perm)
        return permutations


class PermutationConsistentMemoryEncoder(nn.Module):
    """Encode class memory across multiple support-shot permutations."""

    def __init__(
        self,
        memory_writer: ClassMemoryWriteSSM,
        permutation_sampler: SupportPermutationSampler,
    ) -> None:
        super().__init__()
        self.memory_writer = memory_writer
        self.permutation_sampler = permutation_sampler

    def forward(
        self,
        shot_descriptors: torch.Tensor,
        training: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if shot_descriptors.dim() != 2:
            raise ValueError(
                "shot_descriptors must have shape (Shot, Dim), "
                f"got {tuple(shot_descriptors.shape)}"
            )
        permutations = self.permutation_sampler(
            shot_num=shot_descriptors.shape[0],
            device=shot_descriptors.device,
            training=training,
        )
        memories = []
        trajectories = []
        for perm in permutations:
            memory, refined = self.memory_writer(shot_descriptors.index_select(0, perm))
            memories.append(memory)
            trajectories.append(refined)

        memory_stack = torch.stack(memories, dim=0)
        memory_mean = memory_stack.mean(dim=0)
        dispersion = memory_stack.var(dim=0, unbiased=False).mean().sqrt()
        trajectory_stack = torch.stack(trajectories, dim=0).mean(dim=0)
        return memory_mean, dispersion, trajectory_stack
