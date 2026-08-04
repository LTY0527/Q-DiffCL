from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class RobustGainCalibration:
    median: float
    iqr: float
    q_min: float
    q_max: float
    eps: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def relative_gain(simple_error: np.ndarray, diffusion_error: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    simple = np.asarray(simple_error, dtype=np.float64); diffusion = np.asarray(diffusion_error, dtype=np.float64)
    if simple.shape != diffusion.shape or simple.ndim != 1: raise ValueError("errors must be matching vectors")
    if not np.isfinite(simple).all() or not np.isfinite(diffusion).all() or np.any(simple < 0) or np.any(diffusion < 0):
        raise ValueError("errors must be finite and non-negative")
    return (simple - diffusion) / (simple + float(eps))


def fit_robust_gain_calibration(train_gain: np.ndarray, q_min: float, q_max: float,
                                eps: float = 1e-12) -> RobustGainCalibration:
    gain = np.asarray(train_gain, dtype=np.float64)
    if gain.ndim != 1 or not len(gain) or not np.isfinite(gain).all(): raise ValueError("train gain must be finite")
    if not 0 <= q_min < q_max <= 1: raise ValueError("quality bounds must satisfy 0 <= q_min < q_max <= 1")
    median = float(np.median(gain)); iqr = float(np.quantile(gain, .75) - np.quantile(gain, .25))
    return RobustGainCalibration(median, max(iqr, eps), float(q_min), float(q_max), float(eps))


def relative_quality(gain: np.ndarray, calibration: RobustGainCalibration) -> np.ndarray:
    values = np.asarray(gain, dtype=np.float64)
    if not np.isfinite(values).all(): raise ValueError("relative gain must be finite")
    z = np.clip((values - calibration.median) / calibration.iqr, -60, 60)
    sigmoid = 1 / (1 + np.exp(-z))
    return (calibration.q_min + (calibration.q_max - calibration.q_min) * sigmoid).astype(np.float32)


def semantic_score(clean_probabilities: np.ndarray, restored_probabilities: np.ndarray,
                   floor: float = 0.1) -> np.ndarray:
    clean = np.asarray(clean_probabilities, dtype=np.float64)
    restored = np.asarray(restored_probabilities, dtype=np.float64)
    if clean.shape != restored.shape or clean.ndim != 2: raise ValueError("probabilities must have matching [N,C] shapes")
    if not np.isfinite(clean).all() or not np.isfinite(restored).all(): raise ValueError("probabilities must be finite")
    if not 0 <= floor <= 1: raise ValueError("semantic floor must be in [0,1]")
    total_variation = np.abs(clean - restored).sum(1) / 2
    return np.clip(1 - total_variation, floor, 1).astype(np.float32)


def relative_semantic_quality(relative: np.ndarray, semantic: np.ndarray,
                              q_min: float) -> np.ndarray:
    relative = np.asarray(relative, dtype=np.float64); semantic = np.asarray(semantic, dtype=np.float64)
    if relative.shape != semantic.shape or relative.ndim != 1: raise ValueError("quality inputs must be matching vectors")
    if not np.isfinite(relative).all() or not np.isfinite(semantic).all(): raise ValueError("quality inputs must be finite")
    return np.clip(q_min + (relative - q_min) * semantic, q_min, 1).astype(np.float32)
