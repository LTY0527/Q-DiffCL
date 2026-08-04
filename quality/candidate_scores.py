from __future__ import annotations

import numpy as np


def _validate_candidates(candidates: np.ndarray, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(candidates, dtype=np.float64); mask = np.asarray(observation, dtype=bool)
    if values.ndim != 4 or mask.shape != (values.shape[0], values.shape[2], values.shape[3]):
        raise ValueError("candidates must be [N,K,C,L] and observation [N,C,L]")
    if not np.isfinite(values).all(): raise ValueError("candidates must be finite")
    if np.any((~mask).reshape(len(mask), -1).sum(1) == 0): raise ValueError("each sample needs missing positions")
    return values, mask


def center_scores(candidates: np.ndarray, observation: np.ndarray) -> np.ndarray:
    """H1: no-reference distance to the within-window candidate median."""
    values, mask = _validate_candidates(candidates, observation)
    center = np.median(values, axis=1)
    missing = (~mask)[:, None, :, :]
    distance = (np.abs(values - center[:, None]) * missing).sum(axis=(2, 3)) / missing.sum(axis=(2, 3))
    return (-distance).astype(np.float32)


def semantic_scores(candidate_probabilities: np.ndarray) -> np.ndarray:
    """H2: no-reference negative KL to the within-window teacher consensus."""
    probabilities = np.asarray(candidate_probabilities, dtype=np.float64)
    if probabilities.ndim != 3 or not np.isfinite(probabilities).all():
        raise ValueError("candidate probabilities must be finite [N,K,C]")
    probabilities = np.clip(probabilities, 1e-12, 1)
    probabilities /= probabilities.sum(axis=2, keepdims=True)
    consensus = probabilities.mean(axis=1, keepdims=True)
    divergence = (probabilities * (np.log(probabilities) - np.log(consensus))).sum(axis=2)
    return (-divergence).astype(np.float32)


def within_sample_zscore(scores: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all(): raise ValueError("scores must be finite [N,K]")
    centered = values - values.mean(axis=1, keepdims=True)
    scale = values.std(axis=1, keepdims=True)
    return np.divide(centered, scale, out=np.zeros_like(centered), where=scale > eps).astype(np.float32)


def combined_scores(center: np.ndarray, semantic: np.ndarray, lambda_sem: float = 1.0) -> np.ndarray:
    if np.asarray(center).shape != np.asarray(semantic).shape: raise ValueError("H1/H2 scores must match")
    if not np.isfinite(lambda_sem): raise ValueError("lambda_sem must be finite")
    return within_sample_zscore(center) + float(lambda_sem) * within_sample_zscore(semantic)


def soft_candidate_weights(scores: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all(): raise ValueError("scores must be finite [N,K]")
    if not np.isfinite(temperature) or temperature <= 0: raise ValueError("temperature must be positive")
    stable = (values - values.max(axis=1, keepdims=True)) / temperature
    weights = np.exp(stable); weights /= weights.sum(axis=1, keepdims=True)
    return weights.astype(np.float32)
