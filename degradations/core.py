from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from utils import deterministic_seed


@dataclass(frozen=True)
class DegradationResult:
    data: np.ndarray
    observation_mask: np.ndarray
    corruption_mask: np.ndarray
    degradation_type: str
    severity: Any
    random_seed: int
    sample_id: str
    space: str
    order: tuple[str, ...] = ()


def _result(x: np.ndarray, obs: np.ndarray, corruption: np.ndarray, kind: str,
            severity: Any, seed: int, sample_id: str, space: str,
            order: tuple[str, ...] = ()) -> DegradationResult:
    if space not in {"raw_space", "normalized_space"}:
        raise ValueError("space must be raw_space or normalized_space")
    return DegradationResult(x.astype(np.float32), obs.astype(bool), corruption.astype(bool),
                             kind, severity, seed, sample_id, space, order)


def apply_degradation(data: np.ndarray, kind: str, severity: Any, master_seed: int,
                      sample_id: str | int, space: str = "normalized_space",
                      channel_std: np.ndarray | None = None) -> DegradationResult:
    if data.ndim != 2:
        raise ValueError("a sample must have shape [channels, length]")
    sample_id = str(sample_id)
    seed = deterministic_seed(master_seed, sample_id, kind)
    rng = np.random.default_rng(seed)
    x = np.asarray(data, dtype=np.float32).copy()
    observation = np.ones_like(x, dtype=bool)
    corruption = np.zeros_like(x, dtype=bool)
    if kind == "none":
        return _result(x, observation, corruption, kind, severity, seed, sample_id, space)
    if kind == "mcar_missing":
        rate = float(severity)
        if not 0 <= rate <= 1:
            raise ValueError("missing rate must be in [0,1]")
        observation = rng.random(x.shape) >= rate
        corruption = ~observation
        x[~observation] = 0.0
    elif kind == "block_missing":
        rate = float(severity)
        block = min(x.shape[1], max(0, round(rate * x.shape[1])))
        for channel in range(x.shape[0]):
            start = int(rng.integers(0, x.shape[1] - block + 1)) if block else 0
            observation[channel, start:start + block] = False
        corruption = ~observation
        x[~observation] = 0.0
    elif kind == "channel_dropout":
        count = int(severity)
        if not 0 <= count <= x.shape[0]:
            raise ValueError("invalid channel dropout count")
        channels = rng.choice(x.shape[0], size=count, replace=False)
        observation[channels, :] = False
        corruption = ~observation
        x[~observation] = 0.0
    elif kind == "gaussian_noise":
        snr_db = float(severity)
        power = float(np.mean(x.astype(np.float64) ** 2))
        noise_power = power / (10 ** (snr_db / 10))
        noise = rng.normal(0.0, np.sqrt(noise_power), size=x.shape).astype(np.float32)
        x += noise
        corruption[:] = noise != 0
    elif kind == "spike_noise":
        settings = {"low": (0.01, 2.0), "medium": (0.03, 4.0), "high": (0.05, 8.0)}
        if severity not in settings:
            raise ValueError("spike severity must be low, medium, or high")
        rate, scale = settings[severity]
        corruption = rng.random(x.shape) < rate
        x[corruption] += rng.normal(0, scale, size=int(corruption.sum())).astype(np.float32)
    elif kind == "linear_drift":
        scale = np.ones(x.shape[0]) if channel_std is None else np.asarray(channel_std)
        if scale.shape != (x.shape[0],):
            raise ValueError("channel_std must have one value per channel")
        drift = scale[:, None] * float(severity) * np.linspace(0, 1, x.shape[1])[None, :]
        x += drift.astype(np.float32)
        corruption[:] = drift != 0
    elif kind == "mixed":
        if not isinstance(severity, dict) or not severity:
            raise ValueError("mixed severity must be a mapping")
        order = tuple(severity.keys())
        current = x
        for index, (subkind, subseverity) in enumerate(severity.items()):
            sub = apply_degradation(current, subkind, subseverity, master_seed + index,
                                    sample_id, space, channel_std)
            current = sub.data
            observation &= sub.observation_mask
            corruption |= sub.corruption_mask
        return _result(current, observation, corruption, kind, severity, seed, sample_id, space, order)
    else:
        raise ValueError(f"unknown degradation: {kind}")
    return _result(x, observation, corruption, kind, severity, seed, sample_id, space)
