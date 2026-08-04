from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SPLITS = ("train", "validation", "test")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_strings(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode("utf-8")).hexdigest()


def split_window_id(window_id: str) -> tuple[str, int, int]:
    run_uid, samples = window_id.rsplit(":samples_", 1)
    start, end = samples.split("_", 1)
    return run_uid, int(start), int(end)


def mask_id(observation: np.ndarray) -> str:
    packed = np.packbits(np.asarray(observation, dtype=np.uint8), axis=None)
    return "sha256:" + hashlib.sha256(packed.tobytes()).hexdigest()


def per_sample_masked_mae(clean: np.ndarray, restored: np.ndarray,
                          observation: np.ndarray) -> np.ndarray:
    clean = np.asarray(clean); restored = np.asarray(restored); observation = np.asarray(observation, dtype=bool)
    if clean.shape != restored.shape or clean.shape != observation.shape:
        raise ValueError("clean, restored and observation must have matching shapes")
    error = np.abs(clean - restored)
    missing = ~observation
    count = missing.reshape(len(clean), -1).sum(1)
    if np.any(count == 0):
        raise ValueError("every fixed view must contain at least one missing value")
    return (error * missing).reshape(len(clean), -1).sum(1) / count


def fit_quality_scale(train_errors: np.ndarray, estimator: str = "median") -> float:
    values = np.asarray(train_errors, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("train errors must be a non-empty finite non-negative vector")
    if estimator == "median":
        scale = float(np.median(values))
    elif estimator == "mean":
        scale = float(np.mean(values))
    else:
        raise ValueError("quality scale estimator must be median or mean")
    return max(scale, 1e-12)


def quality_scores(errors: np.ndarray, scale: float, q_min: float) -> np.ndarray:
    values = np.asarray(errors, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("quality errors must be finite and non-negative")
    if not np.isfinite(scale) or scale <= 0: raise ValueError("quality scale must be finite and positive")
    if not 0 <= q_min <= 1: raise ValueError("q_min must be in [0, 1]")
    return np.clip(np.exp(-values / scale), q_min, 1.0).astype(np.float32)


def distribution(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all(): raise ValueError("values must be finite")
    return {
        "min": float(x.min()), "max": float(x.max()), "mean": float(x.mean()),
        "std": float(x.std()), "p10": float(np.quantile(x, .10)),
        "p25": float(np.quantile(x, .25)), "p50": float(np.quantile(x, .50)),
        "p75": float(np.quantile(x, .75)), "p90": float(np.quantile(x, .90)),
    }


def validate_view_splits(
    views: Mapping[str, Mapping[str, Any]], expected_runs: Mapping[str, Sequence[str]]
) -> None:
    run_sets: dict[str, set[str]] = {}
    for split in SPLITS:
        if split not in views or split not in expected_runs: raise ValueError(f"missing split: {split}")
        value = views[split]
        required = ("clean", "degraded", "restored", "observation", "labels", "window_id", "run_uid", "mask_id")
        if any(key not in value for key in required): raise ValueError(f"incomplete fixed view: {split}")
        count = len(value["labels"])
        if any(len(value[key]) != count for key in required): raise ValueError(f"view length mismatch: {split}")
        if len(set(map(str, value["window_id"]))) != count: raise ValueError(f"duplicate window_id: {split}")
        run_sets[split] = set(map(str, value["run_uid"]))
        if run_sets[split] != set(map(str, expected_runs[split])):
            raise ValueError(f"run manifest mismatch: {split}")
    if any(run_sets[a] & run_sets[b] for index, a in enumerate(SPLITS) for b in SPLITS[index + 1:]):
        raise ValueError("run leakage detected across fixed view splits")
