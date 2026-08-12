from __future__ import annotations

import numpy as np


FIXED_STAGE_BETAS = {"normal": 1.0, "early": 0.6, "middle": 0.8, "stable": 1.0}


def apply_stage_perturbation_budget(base: np.ndarray, candidate: np.ndarray, stages: np.ndarray,
                                    betas: dict[str, float]) -> np.ndarray:
    if betas != FIXED_STAGE_BETAS:
        raise ValueError("stage perturbation betas are frozen")
    base = np.asarray(base, np.float32); candidate = np.asarray(candidate, np.float32); stages = np.asarray(stages).astype(str)
    if base.shape != candidate.shape or len(stages) != len(base): raise ValueError("stage budget shape mismatch")
    if not np.isin(stages, tuple(betas)).all(): raise ValueError("unknown budget stage")
    beta = np.asarray([betas[stage] for stage in stages], np.float32)[:, None, None]
    result = base + beta * (candidate - base)
    if not np.isfinite(result).all(): raise FloatingPointError("non-finite stage budget view")
    return result.astype(np.float32)

