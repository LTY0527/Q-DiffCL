from __future__ import annotations

import numpy as np


def teacher_consistency(clean_predictions: np.ndarray, degraded_predictions: np.ndarray,
                        teacher_available: bool) -> float | str:
    if not teacher_available: return "UNAVAILABLE"
    return float(np.mean(np.asarray(clean_predictions) == np.asarray(degraded_predictions)))


def representation_diagnostics(clean: np.ndarray, degraded: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    clean = np.asarray(clean, dtype=float)
    degraded = np.asarray(degraded, dtype=float)
    labels = np.asarray(labels)
    if clean.shape != degraded.shape or clean.ndim != 2:
        raise ValueError("embeddings must be matching [samples, dimensions] arrays")
    norm = np.linalg.norm(clean, axis=1) * np.linalg.norm(degraded, axis=1)
    cosine = np.divide((clean * degraded).sum(1), norm, out=np.zeros_like(norm), where=norm > 0)
    classes = np.unique(labels)
    centers = {int(c): degraded[labels == c].mean(0) for c in classes}
    within_values = [np.linalg.norm(degraded[labels == c] - centers[int(c)], axis=1).mean() for c in classes]
    between = [np.linalg.norm(centers[int(a)] - centers[int(b)]) for i, a in enumerate(classes) for b in classes[i+1:]]
    within = float(np.mean(within_values)) if within_values else 0.0
    between_mean = float(np.mean(between)) if between else 0.0
    centered = degraded - degraded.mean(0)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular ** 2
    covariance_singular = energy / max(len(degraded) - 1, 1)
    probabilities = energy / energy.sum() if energy.sum() > 0 else np.zeros_like(energy)
    nonzero = probabilities > 0
    effective_rank = float(np.exp(-(probabilities[nonzero] * np.log(probabilities[nonzero])).sum())) if nonzero.any() else 0.0
    shifts = [np.linalg.norm(clean[labels == c].mean(0) - centers[int(c)]) for c in classes]
    return {
        "clean_degraded_cosine": float(cosine.mean()),
        "within_class_distance": within,
        "between_class_distance": between_mean,
        "fisher_ratio": None if within <= 1e-12 else between_mean / within,
        "class_center_shift": float(np.mean(shifts)),
        "dimension_std": degraded.std(0).tolist(),
        "covariance_singular_values": covariance_singular.tolist(),
        "effective_rank": effective_rank,
        "top_singular_value_ratio": float(energy[0] / energy.sum()) if energy.sum() else 0.0,
    }
