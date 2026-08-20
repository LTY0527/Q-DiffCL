from __future__ import annotations

from typing import Any

import numpy as np


def safe_capacity(
    values: np.ndarray,
    criticality: np.ndarray,
    gamma: float = 1.0,
    eps: float = 1e-12,
) -> dict[str, np.ndarray]:
    """Map finite ``[N,C,T]`` windows to parameter-free safe budgets."""
    x = np.asarray(values, dtype=np.float32)
    mask = np.asarray(criticality, dtype=np.float32)
    if x.ndim != 3 or not len(x) or not np.isfinite(x).all():
        raise ValueError("values must be finite non-empty [N,C,T]")
    expected = (x.shape[1], x.shape[2] // 2 + 1)
    if mask.shape != expected or not np.isfinite(mask).all():
        raise ValueError(f"criticality must be finite with shape {expected}")
    if np.any((mask < 0) | (mask > 1)):
        raise ValueError("criticality must lie in [0,1]")
    if float(gamma) not in (0.5, 1.0, 2.0):
        raise ValueError("gamma must be one of {0.5,1.0,2.0}")
    power = np.abs(np.fft.rfft(x.astype(np.float64), axis=-1)) ** 2
    total = power.sum(axis=(1, 2))
    weighted = (power * mask[None]).sum(axis=(1, 2))
    ratio = np.divide(weighted, total + float(eps), out=np.zeros_like(weighted), where=total > 0)
    ratio = np.clip(ratio, 0.0, 1.0)
    capacity = np.clip(1.0 - ratio, 0.0, 1.0)
    rho = np.clip(capacity ** float(gamma), 0.0, 1.0)
    if not np.isfinite(rho).all():
        raise FloatingPointError("safe-capacity budget is non-finite")
    return {
        "critical_energy_ratio": ratio.astype(np.float32),
        "safe_capacity": capacity.astype(np.float32),
        "rho": rho.astype(np.float32),
    }


def distribution(values: np.ndarray) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("distribution values must be finite non-empty [N]")
    q = np.quantile(x, [.05, .25, .50, .75, .95])
    return {
        "count": int(len(x)), "mean": float(x.mean()), "std": float(x.std()),
        "min": float(x.min()), "p05": float(q[0]), "p25": float(q[1]),
        "median": float(q[2]), "p75": float(q[3]), "p95": float(q[4]),
        "max": float(x.max()),
    }
