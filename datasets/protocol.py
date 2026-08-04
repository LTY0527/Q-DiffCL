from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np


SplitName = Literal["train", "validation", "test"]
TransitionPolicy = Literal[
    "exclude_transition", "label_by_last_step", "label_by_fault_ratio", "transition_class"
]


@dataclass(frozen=True)
class Run:
    run_uid: str
    values: np.ndarray  # [time, channels]
    samples: np.ndarray
    fault_id: int = 0
    first_faulty_sample: float | None = None

    @property
    def run_id(self) -> str:
        """Backward-compatible alias; manifests use run_uid."""
        return self.run_uid


@dataclass(frozen=True)
class SplitManifest:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    def validate(self) -> None:
        groups = [set(self.train), set(self.validation), set(self.test)]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("Run leakage detected: split Run IDs must be disjoint")


def split_runs(run_ids: Sequence[str], ratios: tuple[float, float, float], seed: int) -> SplitManifest:
    if not np.isclose(sum(ratios), 1.0) or any(x <= 0 for x in ratios):
        raise ValueError("split ratios must be positive and sum to one")
    ids = np.asarray(sorted(set(run_ids)), dtype=object)
    if len(ids) < 3:
        raise ValueError("at least three independent runs are required")
    ids = ids[np.random.default_rng(seed).permutation(len(ids))]
    n_train = max(1, int(len(ids) * ratios[0]))
    n_val = max(1, int(len(ids) * ratios[1]))
    n_train = min(n_train, len(ids) - 2)
    n_val = min(n_val, len(ids) - n_train - 1)
    manifest = SplitManifest(tuple(ids[:n_train]), tuple(ids[n_train:n_train+n_val]), tuple(ids[n_train+n_val:]))
    manifest.validate()
    return manifest


def make_run_uid(source_split: str, fault_id: int, simulation_run: int) -> str:
    state = "normal" if fault_id == 0 else f"fault_{fault_id:02d}"
    return f"{source_split}:{state}:{simulation_run:04d}"


def split_training_runs_stratified(
    runs: Sequence[Run], validation_ratio: float, seed: int, test_run_uids: Sequence[str] = (),
) -> SplitManifest:
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be in (0, 1)")
    rng = np.random.default_rng(seed)
    train: list[str] = []
    validation: list[str] = []
    for fault_id in sorted({run.fault_id for run in runs}):
        group = sorted((run.run_uid for run in runs if run.fault_id == fault_id))
        if len(group) < 2:
            raise ValueError(f"fault {fault_id} needs at least two training runs")
        order = rng.permutation(len(group))
        validation_count = max(1, min(len(group) - 1, round(len(group) * validation_ratio)))
        validation.extend(group[index] for index in order[:validation_count])
        train.extend(group[index] for index in order[validation_count:])
    manifest = SplitManifest(tuple(sorted(train)), tuple(sorted(validation)), tuple(sorted(test_run_uids)))
    manifest.validate()
    return manifest


def label_run(run: Run, task: str = "binary_fault_detection") -> np.ndarray:
    if run.fault_id == 0:
        return np.zeros(len(run.samples), dtype=np.int64)
    if run.first_faulty_sample is None:
        raise ValueError(f"first_faulty_sample is required for faulty run {run.run_uid}")
    labels = np.zeros(len(run.samples), dtype=np.int64)
    fault_label = 1 if task == "binary_fault_detection" else run.fault_id
    labels[run.samples >= run.first_faulty_sample] = fault_label
    return labels


class Standardizer:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, train_values: np.ndarray) -> "Standardizer":
        if not np.isfinite(train_values).all():
            raise ValueError("clean training data contain NaN or Inf")
        self.mean_ = train_values.mean(axis=0)
        std = train_values.std(axis=0)
        self.scale_ = np.where(std > 1e-12, std, 1.0)
        return self

    def fit_many(self, arrays: Sequence[np.ndarray]) -> "Standardizer":
        count = 0
        total = total_sq = None
        for values in arrays:
            if not np.isfinite(values).all(): raise ValueError("clean training data contain NaN or Inf")
            current = np.asarray(values, dtype=np.float64)
            total = current.sum(0) if total is None else total + current.sum(0)
            total_sq = (current ** 2).sum(0) if total_sq is None else total_sq + (current ** 2).sum(0)
            count += len(current)
        if count == 0: raise ValueError("cannot fit scaler on empty training data")
        self.mean_ = total / count
        variance = np.maximum(total_sq / count - self.mean_ ** 2, 0.0)
        std = np.sqrt(variance); self.scale_ = np.where(std > 1e-12, std, 1.0)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("fit Standardizer on the training split first")
        return (values - self.mean_) / self.scale_


def window_runs(
    runs: Sequence[Run], length: int, stride: int, policy: TransitionPolicy,
    fault_ratio_threshold: float | None = None, task: str = "binary_fault_detection",
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, object]]:
    if length <= 0 or stride <= 0:
        raise ValueError("window length and stride must be positive")
    if policy == "label_by_fault_ratio" and fault_ratio_threshold is None:
        raise ValueError("fault_ratio_threshold must be configured")
    windows: list[np.ndarray] = []
    targets: list[int] = []
    ids: list[str] = []
    transition_total = excluded = 0
    transition_label = 2 if task == "binary_fault_detection" else max((run.fault_id for run in runs), default=0) + 1
    before: dict[int, int] = {}
    after: dict[int, int] = {}
    records: list[dict[str, object]] = []
    for run in runs:
        point_labels = label_run(run, task)
        for start in range(0, len(run.values) - length + 1, stride):
            stop = start + length
            window_labels = point_labels[start:stop]
            unique = np.unique(window_labels)
            transition = len(unique) > 1
            raw_label = int(window_labels[-1])
            before[raw_label] = before.get(raw_label, 0) + 1
            if transition:
                transition_total += 1
            fault_label = 1 if task == "binary_fault_detection" else run.fault_id
            if policy == "exclude_transition" and transition:
                excluded += 1
                records.append({
                    "run_uid": run.run_uid, "start_sample": int(run.samples[start]),
                    "end_sample": int(run.samples[stop - 1]), "faultNumber": run.fault_id,
                    "is_transition": True, "final_label": None, "excluded": True,
                })
                continue
            if policy == "transition_class" and transition:
                label = transition_label
            elif policy == "label_by_last_step":
                label = int(window_labels[-1])
            elif policy == "label_by_fault_ratio":
                ratio = float(np.mean(window_labels != 0))
                label = fault_label if ratio >= float(fault_ratio_threshold) else 0
            else:
                label = int(window_labels[-1])
            x = run.values[start:stop].T.astype(np.float32, copy=True)  # [C,L]
            windows.append(x)
            targets.append(label)
            window_id = f"{run.run_uid}:samples_{int(run.samples[start])}_{int(run.samples[stop - 1])}"
            ids.append(window_id)
            after[label] = after.get(label, 0) + 1
            records.append({
                "run_uid": run.run_uid, "start_sample": int(run.samples[start]),
                "end_sample": int(run.samples[stop - 1]), "faultNumber": run.fault_id,
                "is_transition": transition, "final_label": label, "excluded": False,
            })
    if not windows:
        raise ValueError("windowing produced no samples")
    stats = {
        "total_windows": len(windows) + excluded,
        "transition_windows": transition_total,
        "excluded_transition_windows": excluded,
        "transition_ratio": transition_total / (len(windows) + excluded),
        "class_distribution_before_exclusion": before,
        "class_distribution_after_exclusion": after,
        "window_metadata": records,
    }
    return np.stack(windows), np.asarray(targets), ids, stats
