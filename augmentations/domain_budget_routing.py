from __future__ import annotations

from typing import Any

import numpy as np

from baselines.external_augmentations import traditional_view


def domain_budget_route(
    clean: np.ndarray,
    diffused: np.ndarray,
    sample_ids: np.ndarray,
    rho: float,
    sigma_base: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Route a domain between frozen SCALING and frozen Q-DiffCL views."""
    clean_array = np.asarray(clean, dtype=np.float32)
    diffused_array = np.asarray(diffused, dtype=np.float32)
    ids = np.asarray(sample_ids)
    coefficient = float(rho); base = float(sigma_base)
    if not 0.0 <= coefficient <= 1.0:
        raise ValueError("DCBR rho must lie in [0,1]")
    if base < 0 or not np.isfinite(base):
        raise ValueError("DCBR sigma_base must be finite and nonnegative")
    if clean_array.shape != diffused_array.shape or len(ids) != len(clean_array):
        raise ValueError("DCBR clean/diffused/sample ids must align")
    if not np.isfinite(clean_array).all() or not np.isfinite(diffused_array).all():
        raise ValueError("DCBR inputs must be finite")
    scaling_std = (1.0 - coefficient) * base
    if coefficient == 1.0:
        routed = diffused_array.copy()
    else:
        source = clean_array if coefficient == 0.0 else diffused_array
        routed = traditional_view(source, ids, "SCALING", int(seed), 0.0, scaling_std)
    if not np.isfinite(routed).all():
        raise FloatingPointError("DCBR produced non-finite augmentation")
    return routed, {
        "rho": coefficient, "sigma_base": base, "effective_scaling_std": scaling_std,
        "rho_zero_exact_scaling_protocol": coefficient == 0.0,
        "rho_one_exact_final_view": coefficient == 1.0,
        "finite": True, "shape_preserved": routed.shape == clean_array.shape,
        "inference_parameters": 0,
    }
