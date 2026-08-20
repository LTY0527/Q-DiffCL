from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from baselines.external_augmentations import traditional_view


def _route_uniform(router_seed: int, sample_id: object) -> float:
    payload = f"SVR|{int(router_seed)}|{sample_id}".encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return integer / float(1 << 64)


def stochastic_view_route(
    clean: np.ndarray,
    final_diffused: np.ndarray,
    sample_ids: np.ndarray,
    p: float,
    router_seed: int,
    scaling_seed: int,
    scaling_std: float = 0.05,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Choose exactly one frozen augmentation branch for every sample.

    Routing randomness is keyed by stable sample IDs, so it is independent of
    traversal/batch order.  The same uniforms are used across candidate values
    of ``p``, which makes candidate masks nested and directly auditable.
    """
    clean_array = np.asarray(clean, dtype=np.float32)
    diffused_array = np.asarray(final_diffused, dtype=np.float32)
    ids = np.asarray(sample_ids)
    probability = float(p)
    sigma = float(scaling_std)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("SVR p must lie in [0,1]")
    if not np.isfinite(sigma) or sigma < 0.0:
        raise ValueError("SVR scaling_std must be finite and nonnegative")
    if clean_array.shape != diffused_array.shape or len(ids) != len(clean_array):
        raise ValueError("SVR clean/diffused/sample ids must align")
    if not np.isfinite(clean_array).all() or not np.isfinite(diffused_array).all():
        raise ValueError("SVR inputs must be finite")

    uniforms = np.asarray([_route_uniform(router_seed, item) for item in ids], dtype=np.float64)
    route_mask = uniforms < probability
    if probability == 0.0:
        routed = traditional_view(clean_array, ids, "SCALING", int(scaling_seed), 0.0, sigma)
    elif probability == 1.0:
        routed = diffused_array.copy()
    else:
        scaled = traditional_view(clean_array, ids, "SCALING", int(scaling_seed), 0.0, sigma)
        selector = route_mask.reshape((-1,) + (1,) * (clean_array.ndim - 1))
        routed = np.where(selector, diffused_array, scaled).astype(np.float32, copy=False)
    if not np.isfinite(routed).all():
        raise FloatingPointError("SVR produced non-finite augmentation")

    id_text = "\n".join(map(str, ids.tolist()))
    fairness_payload = f"{int(router_seed)}\n{id_text}".encode("utf-8")
    mask_payload = np.packbits(route_mask, bitorder="little").tobytes()
    qdiff_count = int(route_mask.sum())
    return routed, {
        "p": probability,
        "router_seed": int(router_seed),
        "scaling_seed": int(scaling_seed),
        "scaling_std": sigma,
        "qdiffcl_route_count": qdiff_count,
        "scaling_route_count": int(len(route_mask) - qdiff_count),
        "sample_count": int(len(route_mask)),
        "realized_route_ratio": float(route_mask.mean()) if len(route_mask) else 0.0,
        "fairness_sha256": hashlib.sha256(fairness_payload).hexdigest(),
        "route_mask_sha256": hashlib.sha256(mask_payload).hexdigest(),
        "exactly_one_branch_per_sample": True,
        "simultaneous_augmentation": False,
        "p_zero_exact_scaling": probability == 0.0,
        "p_one_exact_final": probability == 1.0,
        "finite": True,
        "shape_preserved": routed.shape == clean_array.shape,
        "inference_parameters": 0,
    }
