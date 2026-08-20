from __future__ import annotations

from typing import Any

import numpy as np


STAGE_ORDER = ("normal", "early", "mature")


def normalized_log_spectrum(values: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit and apply one train-only log-spectrum normalization."""
    x = np.asarray(values, dtype=np.float32)
    if x.ndim != 3 or not len(x) or not np.isfinite(x).all():
        raise ValueError("train windows must be finite non-empty [N,C,T]")
    log_spectrum = np.log1p(np.abs(np.fft.rfft(x.astype(np.float64), axis=-1)))
    mean = log_spectrum.mean(0); scale = log_spectrum.std(0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    normalized = ((log_spectrum - mean) / scale).astype(np.float32)
    return normalized, {"fit_split": "train", "formula": "featurewise z-score(log1p(abs(rfft(x))))",
                        "shape": list(mean.shape), "finite": bool(np.isfinite(normalized).all())}


def _shift_one(features: np.ndarray, groups: np.ndarray, eps: float) -> dict[str, float]:
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("cross-group shift requires at least two groups")
    flat = features.reshape(len(features), -1).astype(np.float64)
    global_mean = flat.mean(0)
    between_values = []; within_values = []
    for group in unique:
        current = flat[groups == group]; group_mean = current.mean(0)
        between_values.append(np.mean((group_mean - global_mean) ** 2))
        within_values.append(np.mean((current - group_mean) ** 2))
    between = float(np.mean(between_values)); within = float(np.mean(within_values))
    return {"score": between / (within + eps), "between": between, "within": within,
            "groups": int(len(unique)), "samples": int(len(features))}


def cross_group_shift_demand(features: np.ndarray, groups: np.ndarray, stages: np.ndarray,
                             eps: float = 1e-12) -> dict[str, Any]:
    """Stage-stratified Proxy A with sample-count weighted aggregation."""
    z = np.asarray(features); group = np.asarray(groups, dtype=object); stage = np.asarray(stages, dtype=object)
    if not (z.ndim == 3 and len(z) == len(group) == len(stage)):
        raise ValueError("features/groups/stages must align")
    details = {}
    for name in STAGE_ORDER:
        selector = stage == name
        if selector.any() and len(np.unique(group[selector])) >= 2:
            details[name] = _shift_one(z[selector], group[selector], eps)
    if not details:
        raise ValueError("no stage has cross-group support")
    total = sum(item["samples"] for item in details.values())
    score = sum(item["score"] * item["samples"] for item in details.values()) / total
    return {"score": float(score), "aggregation": "stage sample-count weighted",
            "stage": details, "samples": int(total)}


def _separability(features: np.ndarray, labels: np.ndarray, eps: float) -> dict[str, float]:
    unique = np.unique(labels)
    if len(unique) != 2:
        raise ValueError("separability proxy requires exactly two classes")
    flat = features.reshape(len(features), -1).astype(np.float64)
    global_mean = flat.mean(0); between_values = []; within_values = []
    for label in unique:
        current = flat[labels == label]; class_mean = current.mean(0)
        between_values.append(np.mean((class_mean - global_mean) ** 2))
        within_values.append(np.mean((current - class_mean) ** 2))
    between = float(np.mean(between_values)); within = float(np.mean(within_values))
    separability = between / (within + eps)
    return {"score": 1.0 / (1.0 + separability), "separability": separability,
            "between": between, "within": within, "samples": int(len(features))}


def separability_difficulty_demand(features: np.ndarray, stages: np.ndarray,
                                   eps: float = 1e-12) -> dict[str, Any]:
    """Proxy B for Normal-vs-Fault and the preregistered Normal-vs-Early view."""
    z = np.asarray(features); stage = np.asarray(stages, dtype=object)
    if z.ndim != 3 or len(z) != len(stage):
        raise ValueError("features and stages must align")
    overall = _separability(z, (stage != "normal").astype(np.int8), eps)
    early_selector = np.isin(stage, ("normal", "early"))
    early = _separability(z[early_selector], (stage[early_selector] == "early").astype(np.int8), eps)
    return {"score": overall["score"], "binary_normal_fault": overall,
            "normal_early": early}


def group_bootstrap(
    features: np.ndarray,
    groups: np.ndarray,
    stages: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Resample complete groups and relabel duplicate draws as independent slots."""
    z = np.asarray(features); group = np.asarray(groups, dtype=object); stage = np.asarray(stages, dtype=object)
    unique = np.unique(group); rng = np.random.default_rng(seed)
    shift = np.empty(repeats, dtype=np.float64); difficulty = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        draws = rng.choice(unique, len(unique), replace=True); indices = []; relabeled = []
        for slot, drawn in enumerate(draws):
            current = np.flatnonzero(group == drawn); indices.extend(current.tolist()); relabeled.extend([slot] * len(current))
        indices = np.asarray(indices, dtype=np.int64); relabeled = np.asarray(relabeled, dtype=np.int64)
        shift[repeat] = cross_group_shift_demand(z[indices], relabeled, stage[indices])["score"]
        difficulty[repeat] = separability_difficulty_demand(z[indices], stage[indices])["score"]
    return {"shift_demand": shift, "difficulty_demand": difficulty}


def leave_one_group_out(features: np.ndarray, groups: np.ndarray, stages: np.ndarray) -> dict[str, np.ndarray]:
    z = np.asarray(features); group = np.asarray(groups, dtype=object); stage = np.asarray(stages, dtype=object)
    shift = []; difficulty = []; ids = []
    for omitted in np.unique(group):
        keep = group != omitted
        try:
            shift_value = cross_group_shift_demand(z[keep], group[keep], stage[keep])["score"]
            difficulty_value = separability_difficulty_demand(z[keep], stage[keep])["score"]
        except ValueError:
            continue
        ids.append(str(omitted)); shift.append(shift_value); difficulty.append(difficulty_value)
    return {"group_ids": np.asarray(ids, dtype=object), "shift_demand": np.asarray(shift),
            "difficulty_demand": np.asarray(difficulty)}
