from __future__ import annotations

from collections import Counter
from typing import Iterator

import numpy as np
from torch.utils.data import Sampler


def sqrt_inverse_frequency_weights(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) == 0 or np.any(counts <= 0):
        raise ValueError("training labels must contain non-empty class counts")
    raw = 1.0 / np.sqrt(counts.astype(np.float64)); raw /= raw.mean()
    result = np.ones(int(classes.max()) + 1, dtype=np.float32); result[classes] = raw
    if not np.isfinite(result).all(): raise ValueError("non-finite class weight")
    return result


class PositiveSafeBatchSampler(Sampler[list[int]]):
    """Deterministic P×K batches with a bounded rare-class reuse factor."""
    def __init__(self, labels: np.ndarray, classes_per_batch: int, samples_per_class: int,
                 batches_per_epoch: int, seed: int, max_oversampling: float = 3.0):
        labels = np.asarray(labels, dtype=np.int64)
        self.indices = {int(c): np.flatnonzero(labels == c) for c in np.unique(labels)}
        self.p, self.k, self.batches = int(classes_per_batch), int(samples_per_class), int(batches_per_epoch)
        self.seed, self.epoch = int(seed), 0
        if self.p < 1 or self.p > len(self.indices) or self.k < 2 or self.batches < 1:
            raise ValueError("invalid P×K sampler configuration")
        planned = self._planned_counts()
        self.oversampling_factors = {c: planned[c] / len(self.indices[c]) for c in planned}
        if max(self.oversampling_factors.values()) > float(max_oversampling) + 1e-12:
            raise ValueError(f"oversampling factor {max(self.oversampling_factors.values()):.3f} exceeds cap {max_oversampling}")

    def _planned_counts(self) -> dict[int, int]:
        classes = sorted(self.indices); base, remainder = divmod(self.batches * self.p, len(classes))
        return {c: int((base + (i < remainder)) * self.k) for i, c in enumerate(classes)}

    @property
    def planned_sample_counts(self) -> dict[int, int]: return self._planned_counts()
    def set_epoch(self, epoch: int) -> None: self.epoch = int(epoch)
    def __len__(self) -> int: return self.batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch); classes = np.asarray(sorted(self.indices))
        if self.p == len(classes): schedule = np.tile(classes, (self.batches, 1))
        else: schedule = np.stack([rng.choice(classes, self.p, replace=False) for _ in range(self.batches)])
        for row in schedule:
            batch = []
            for c in row:
                pool = self.indices[int(c)]; batch.extend(rng.choice(pool, self.k, replace=len(pool) < self.k).tolist())
            rng.shuffle(batch); yield batch


def positive_anchor_audit(labels: np.ndarray, batch_size: int, seed: int) -> dict:
    labels = np.asarray(labels, dtype=np.int64); order = np.random.default_rng(seed).permutation(len(labels))
    totals = Counter(labels.tolist()); valid = Counter(); batch_mass = Counter(); batches = 0
    for start in range(0, len(order), batch_size):
        current = labels[order[start:start + batch_size]]; counts = Counter(current.tolist()); batches += 1
        for c, count in counts.items():
            batch_mass[c] += count
            if count >= 2: valid[c] += count
    per_class = {c: {"train_windows": total, "mean_samples_per_batch": batch_mass[c] / batches,
                     "positive_anchor_rate": valid[c] / total, "zero_positive_anchor_rate": 1 - valid[c] / total}
                 for c, total in sorted(totals.items())}
    overall = sum(valid.values()) / len(labels)
    return {"batches": batches, "overall_valid_anchor_rate": overall,
            "overall_zero_positive_anchor_rate": 1 - overall, "per_class": per_class}
