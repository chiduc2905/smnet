"""Episodic sampler for N-way K-shot few-shot learning."""

import torch
from torch.utils.data import Dataset


class FewshotDataset(Dataset):
    """N-way K-shot episode generator.

    Each episode contains:
    - Support set: way_num classes × shot_num samples
    - Query set: way_num classes × query_num samples

    Labels are episode-relative: 0, 1, ..., way_num-1
    """

    def __init__(
        self,
        data,
        labels,
        episode_num,
        way_num,
        shot_num,
        query_num,
        seed=None,
        hard_pool=None,
        hard_ratio=0.0,
        augment=False,
        augment_cfg=None,
        return_indices=False,
    ):
        """Initialize episodic sampler."""
        self.data = data
        self.labels = labels
        self.episode_num = episode_num
        self.way_num = way_num
        self.shot_num = shot_num
        self.query_num = query_num
        self.seed = seed if seed is not None else 0
        self.hard_ratio = float(hard_ratio)
        self.augment = bool(augment)
        self.return_indices = bool(return_indices)

        self.hard_pool = {}
        if hard_pool is not None:
            for c in range(self.way_num):
                idx = hard_pool.get(c, [])
                if isinstance(idx, torch.Tensor):
                    idx = idx.long().cpu()
                else:
                    idx = torch.tensor(list(idx), dtype=torch.long)
                self.hard_pool[c] = idx

        default_aug = {
            "time_shift_max": 4,
            "time_shift_prob": 0.5,
            "amp_scale_min": 0.9,
            "amp_scale_max": 1.1,
            "amp_scale_prob": 0.5,
            "time_mask_width": 4,
            "time_mask_prob": 0.25,
            "freq_mask_width": 4,
            "freq_mask_prob": 0.25,
        }
        self.augment_cfg = default_aug
        if augment_cfg:
            self.augment_cfg.update(augment_cfg)

        # Pre-compute indices for each class
        self.class_indices = {}
        for c in range(way_num):
            self.class_indices[c] = (labels == c).nonzero(as_tuple=True)[0]

        # Validate data availability
        self._validate()

    def _validate(self):
        """Check if enough samples exist for requested shot/query."""
        required = self.shot_num + self.query_num
        for c in range(self.way_num):
            available = len(self.class_indices[c])
            if available < required:
                print(f"Warning: Class {c} has {available} samples, need {required}")

    def __len__(self):
        return self.episode_num

    def _sample_hard_index(self, class_id, available, gen):
        """Sample one hard index for class from available indices."""
        if class_id not in self.hard_pool:
            return None
        hard_idx = self.hard_pool[class_id]
        if hard_idx.numel() == 0:
            return None

        hard_set = set(hard_idx.tolist())
        candidates = [i.item() for i in available if i.item() in hard_set]
        if not candidates:
            return None

        pos = torch.randint(0, len(candidates), (1,), generator=gen).item()
        return int(candidates[pos])

    def _augment_one(self, img, gen):
        """Lightweight waveform-preserving augmentation."""
        cfg = self.augment_cfg
        out = img.clone()
        _, h, w = out.shape

        if torch.rand(1, generator=gen).item() < cfg["time_shift_prob"]:
            shift = int(
                torch.randint(
                    -cfg["time_shift_max"],
                    cfg["time_shift_max"] + 1,
                    (1,),
                    generator=gen,
                ).item()
            )
            if shift != 0:
                out = torch.roll(out, shifts=shift, dims=2)

        if torch.rand(1, generator=gen).item() < cfg["amp_scale_prob"]:
            scale = cfg["amp_scale_min"] + (
                cfg["amp_scale_max"] - cfg["amp_scale_min"]
            ) * torch.rand(1, generator=gen).item()
            out = out * scale

        if cfg["time_mask_width"] > 0 and torch.rand(1, generator=gen).item() < cfg["time_mask_prob"]:
            width = min(int(cfg["time_mask_width"]), w)
            start = int(torch.randint(0, max(1, w - width + 1), (1,), generator=gen).item())
            out[:, :, start : start + width] = 0.0

        if cfg["freq_mask_width"] > 0 and torch.rand(1, generator=gen).item() < cfg["freq_mask_prob"]:
            width = min(int(cfg["freq_mask_width"]), h)
            start = int(torch.randint(0, max(1, h - width + 1), (1,), generator=gen).item())
            out[:, start : start + width, :] = 0.0

        return out

    def _augment_batch(self, x, gen):
        if not self.augment:
            return x
        return torch.stack([self._augment_one(xi, gen) for xi in x], dim=0)

    def __getitem__(self, index):
        """Generate one episode."""
        gen = torch.Generator()
        gen.manual_seed(self.seed * 10000 + index)

        support_images, support_targets = [], []
        query_images, query_targets = [], []
        support_indices, query_indices = [], []

        for class_id in range(self.way_num):
            indices = self.class_indices[class_id]
            perm = torch.randperm(len(indices), generator=gen)
            shuffled = indices[perm]

            hard_idx = None
            use_hard = (
                self.query_num > 0
                and self.hard_ratio > 0
                and torch.rand(1, generator=gen).item() < self.hard_ratio
            )
            if use_hard:
                hard_idx = self._sample_hard_index(class_id, shuffled, gen)

            if hard_idx is not None:
                hard_idx_t = torch.tensor([hard_idx], dtype=shuffled.dtype)
                remain = shuffled[shuffled != hard_idx]
                need_support = self.shot_num
                need_query_extra = max(self.query_num - 1, 0)
                s_idx = remain[:need_support]
                q_idx_extra = remain[need_support : need_support + need_query_extra]
                q_idx = torch.cat([hard_idx_t, q_idx_extra], dim=0)
            else:
                s_idx = shuffled[: self.shot_num]
                q_idx = shuffled[self.shot_num : self.shot_num + self.query_num]

            support_images.append(self.data[s_idx])
            query_images.append(self.data[q_idx])
            support_indices.append(s_idx)
            query_indices.append(q_idx)

            support_targets.append(torch.full((len(s_idx),), class_id, dtype=torch.long))
            query_targets.append(torch.full((len(q_idx),), class_id, dtype=torch.long))

        query_images = torch.cat(query_images)
        query_targets = torch.cat(query_targets)
        support_images = torch.cat(support_images)
        support_targets = torch.cat(support_targets)
        query_indices = torch.cat(query_indices)
        support_indices = torch.cat(support_indices)

        if self.augment:
            query_images = self._augment_batch(query_images, gen)
            support_images = self._augment_batch(support_images, gen)

        if self.return_indices:
            return (
                query_images,
                query_targets,
                support_images,
                support_targets,
                query_indices,
                support_indices,
            )

        return (query_images, query_targets, support_images, support_targets)
