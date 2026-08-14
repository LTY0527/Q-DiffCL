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


class CrossWellPositiveSafeBatchSampler(Sampler[list[int]]):
    """Deterministic P×K batches that spread each class across its training WELLs."""
    def __init__(self, labels: np.ndarray, well_ids: np.ndarray, classes_per_batch: int,
                 samples_per_class: int, batches_per_epoch: int, seed: int,
                 max_oversampling: float = 3.0):
        labels = np.asarray(labels, dtype=np.int64); well_ids = np.asarray(well_ids, dtype=object)
        if labels.ndim != 1 or well_ids.ndim != 1 or len(labels) != len(well_ids) or len(labels) == 0:
            raise ValueError("labels and WELL ids must be non-empty aligned vectors")
        if any(value is None or str(value) == "" for value in well_ids):
            raise ValueError("Cross-WELL sampling requires a WELL id for every training window")
        self.indices = {
            int(c): {
                str(well): np.flatnonzero((labels == c) & (well_ids == well))
                for well in sorted(set(well_ids[labels == c].tolist()))
            }
            for c in np.unique(labels)
        }
        self.p, self.k, self.batches = int(classes_per_batch), int(samples_per_class), int(batches_per_epoch)
        self.seed, self.epoch = int(seed), 0
        if self.p < 1 or self.p > len(self.indices) or self.k < 2 or self.batches < 1:
            raise ValueError("invalid P×K sampler configuration")
        planned = self._planned_counts()
        class_counts = {c: sum(len(pool) for pool in wells.values()) for c, wells in self.indices.items()}
        self.oversampling_factors = {c: planned[c] / class_counts[c] for c in planned}
        if max(self.oversampling_factors.values()) > float(max_oversampling) + 1e-12:
            raise ValueError(f"oversampling factor {max(self.oversampling_factors.values()):.3f} exceeds cap {max_oversampling}")

    def _planned_counts(self) -> dict[int, int]:
        classes = sorted(self.indices); base, remainder = divmod(self.batches * self.p, len(classes))
        return {c: int((base + (i < remainder)) * self.k) for i, c in enumerate(classes)}

    @property
    def planned_sample_counts(self) -> dict[int, int]: return self._planned_counts()
    @property
    def class_wells(self) -> dict[int, tuple[str, ...]]:
        return {c: tuple(sorted(wells)) for c, wells in self.indices.items()}
    def set_epoch(self, epoch: int) -> None: self.epoch = int(epoch)
    def __len__(self) -> int: return self.batches

    @staticmethod
    def _allocate(pool_sizes: np.ndarray, count: int) -> np.ndarray:
        """Cover WELLs first, then use sqrt support while avoiding within-batch reuse."""
        allocation = np.zeros(len(pool_sizes), dtype=np.int64)
        selected = np.arange(len(pool_sizes)) if count >= len(pool_sizes) else np.argsort(-pool_sizes)[:count]
        allocation[selected] = 1; remaining = count - len(selected)
        while remaining > 0 and np.any(allocation < pool_sizes):
            active = np.flatnonzero(allocation < pool_sizes)
            weights = np.sqrt(pool_sizes[active].astype(np.float64)); ideal = remaining * weights / weights.sum()
            addition = np.minimum(np.floor(ideal).astype(np.int64), pool_sizes[active] - allocation[active])
            if not addition.any(): addition[int(np.argmax(ideal))] = 1
            allocation[active] += addition; remaining -= int(addition.sum())
        if remaining > 0:
            # Only possible when the entire class has fewer unique windows than K.
            weights = np.sqrt(pool_sizes.astype(np.float64)); ideal = remaining * weights / weights.sum()
            allocation += np.floor(ideal).astype(np.int64); remaining = count - int(allocation.sum())
            for index in np.argsort(-(ideal - np.floor(ideal)))[:remaining]: allocation[index] += 1
        return allocation

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch); classes = np.asarray(sorted(self.indices))
        if self.p == len(classes): schedule = np.tile(classes, (self.batches, 1))
        else: schedule = np.stack([rng.choice(classes, self.p, replace=False) for _ in range(self.batches)])
        queues = {c: {well: rng.permutation(pool) for well, pool in wells.items()} for c, wells in self.indices.items()}
        positions = {c: {well: 0 for well in wells} for c, wells in self.indices.items()}
        for row in schedule:
            batch = []
            for c in row:
                c = int(c); chosen = []; chosen_set = set(); deferred = {well: [] for well in queues[c]}
                while len(chosen) < self.k:
                    active = [well for well in sorted(queues[c]) if positions[c][well] < len(queues[c][well])]
                    if not active:
                        for well, pool in self.indices[c].items():
                            candidates = np.asarray([index for index in pool if int(index) not in chosen_set], dtype=np.int64)
                            deferred[well].extend(index for index in pool.tolist() if int(index) in chosen_set)
                            queues[c][well] = rng.permutation(candidates); positions[c][well] = 0
                        active = [well for well in sorted(queues[c]) if len(queues[c][well])]
                        if not active:
                            for well, pool in self.indices[c].items():
                                queues[c][well] = rng.permutation(pool); positions[c][well] = 0
                            active = [well for well in sorted(queues[c]) if len(queues[c][well])]
                    active = list(np.asarray(active, dtype=object)[rng.permutation(len(active))])
                    sizes = np.asarray([len(queues[c][well]) - positions[c][well] for well in active], dtype=np.int64)
                    allocation = self._allocate(sizes, min(self.k - len(chosen), int(sizes.sum())))
                    for well, count in zip(active, allocation):
                        start = positions[c][well]; stop = start + int(count)
                        values = queues[c][well][start:stop].tolist(); positions[c][well] = stop
                        chosen.extend(values); chosen_set.update(map(int, values))
                for well, held in deferred.items():
                    if held:
                        remaining = queues[c][well][positions[c][well]:]
                        queues[c][well] = np.concatenate((remaining, rng.permutation(np.asarray(held, dtype=np.int64))))
                        positions[c][well] = 0
                batch.extend(chosen)
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
