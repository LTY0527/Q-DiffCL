from __future__ import annotations

import hashlib
from typing import Any, Callable

import numpy as np


def _rankdata(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).ravel()
    n = a.size
    if n == 0:
        return np.array([], dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    obs = a[order]
    repeat = np.concatenate(([False], obs[1:] == obs[:-1]))
    if not repeat.any():
        return ranks
    ones = np.ones(n, dtype=np.float64)
    count = np.concatenate((ones[~repeat], [0]))
    count = np.diff(np.nonzero(count)[0])
    cumulative = np.cumsum(np.concatenate(([0], count)))
    for start, size in zip(cumulative[:-1], count):
        if size > 1:
            idx = order[start:start + size]
            ranks[idx] = ranks[idx].mean()
    return ranks


def spearman_rank_reliability(c_boot: np.ndarray, c_ref: np.ndarray) -> float:
    x = np.asarray(c_boot, dtype=np.float64).ravel()
    y = np.asarray(c_ref, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError(f"shape mismatch: c_boot {x.shape} vs c_ref {y.shape}")
    if x.size < 2:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    dx = rx - rx.mean()
    dy = ry - ry.mean()
    denom = float(np.sqrt(np.sum(dx * dx) * np.sum(dy * dy)))
    if denom < 1e-30:
        return float("nan")
    return float(np.sum(dx * dy) / denom)


def mask_reliability(m_boot: np.ndarray, m_ref: np.ndarray) -> float:
    b = np.asarray(m_boot, dtype=bool)
    r = np.asarray(m_ref, dtype=bool)
    if b.shape != r.shape:
        raise ValueError(f"shape mismatch: m_boot {b.shape} vs m_ref {r.shape}")
    inter = np.logical_and(b, r).sum()
    union = np.logical_or(b, r).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def combine_reliability(r_rank: float, r_mask: float,
                        rank_weight: float = 0.5, mask_weight: float = 0.5) -> float:
    r_rank_v = max(0.0, float(r_rank)) if np.isfinite(r_rank) else 0.0
    r_mask_v = max(0.0, min(1.0, float(r_mask))) if np.isfinite(r_mask) else 0.0
    return float(max(0.0, min(1.0, rank_weight * r_rank_v + mask_weight * r_mask_v)))


def _top_mask(values: np.ndarray, ratio: float) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    flat = v.ravel()
    n = flat.size
    k = max(1, int(round(n * float(ratio))))
    if k >= n:
        return np.ones_like(v, dtype=bool)
    threshold = np.partition(flat, n - k)[n - k]
    mask = v >= threshold
    while mask.sum() > k:
        tie_value = float(np.min(v[mask]))
        flat_idx = np.argsort(-v.ravel(), kind="mergesort")
        to_zero = flat_idx[k:][v.ravel()[flat_idx[k:]] == tie_value]
        m2 = mask.ravel()
        m2[to_zero] = False
        mask = m2.reshape(v.shape)
    return mask.astype(bool)


def _robust_normalize(x: np.ndarray) -> np.ndarray:
    v = np.asarray(x, dtype=np.float64)
    q25, q75 = np.quantile(v, [0.25, 0.75])
    scale = float(q75 - q25)
    if scale < 1e-12:
        return np.zeros_like(v, dtype=np.float64)
    med = float(np.median(v))
    return (v - med) / scale


def _fisher(group1: np.ndarray, group2: np.ndarray) -> np.ndarray:
    g1 = np.asarray(group1, dtype=np.float64)
    g2 = np.asarray(group2, dtype=np.float64)
    diff = g1.mean(axis=0) - g2.mean(axis=0)
    v1 = g1.var(axis=0, ddof=1) if g1.shape[0] > 1 else 0.0
    v2 = g2.var(axis=0, ddof=1) if g2.shape[0] > 1 else 0.0
    denom = np.sqrt(v1 / max(1, g1.shape[0]) + v2 / max(1, g2.shape[0]))
    safe = np.where(denom > 1e-12, denom, 1.0)
    return np.where(denom > 1e-12, diff * diff / safe, 0.0)


def _scores_from_runs(normal_runs: np.ndarray, fault_runs: np.ndarray, early_runs: np.ndarray,
                      weights: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    discriminative = _fisher(normal_runs, fault_runs)
    early = _fisher(normal_runs, early_runs)
    composite = (float(weights.get("weight_discriminative", 0.5)) * _robust_normalize(discriminative)
                 + float(weights.get("weight_early", 0.5)) * _robust_normalize(early))
    return discriminative, early, composite


def _stratified_grouped_bootstrap(group_ids: list[Any], normal_mask: np.ndarray, fault_mask: np.ndarray,
                                  early_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    group_arr = np.asarray(group_ids)
    unique_groups = np.unique(group_arr)
    n = unique_groups.size
    idx = np.arange(n)
    sampled = unique_groups[rng.integers(0, n, n)]
    selected = np.concatenate([np.where(group_arr == g)[0] for g in sampled])
    return selected


def summarize_bootstrap_reliability(
    features: np.ndarray,
    run_uids: np.ndarray,
    labels: np.ndarray,
    stages: np.ndarray,
    *,
    weights: dict[str, float] | None = None,
    critical_ratio: float = 0.30,
    bootstrap_repeats: int = 64,
    bootstrap_seed: int = 22042,
    sampling_unit: str = "run_uid",
    bootstrap_strategy: str = "grouped",
) -> dict[str, Any]:
    if weights is None:
        weights = {"weight_discriminative": 0.5, "weight_early": 0.5, "weight_run_stability": 0.0}
    features = np.asarray(features, dtype=np.float64)
    run_uids = np.asarray(run_uids)
    labels = np.asarray(labels)
    stages = np.asarray(stages)
    n = features.shape[0]
    if not (run_uids.shape[0] == labels.shape[0] == stages.shape[0] == n):
        raise ValueError("feature/run_uids/labels/stages length mismatch")
    if bootstrap_strategy == "window_iid":
        raise ValueError("window_iid bootstrap explicitly forbidden; use grouped sampling")

    normal_mask = labels == 0
    fault_mask = labels != 0
    early_mask = stages == "early"
    uid_normal = run_uids[normal_mask]
    uid_fault = run_uids[fault_mask]
    uid_early = run_uids[early_mask]
    normal_runs = np.stack([features[run_uids == u].mean(axis=0) for u in np.unique(uid_normal)])
    fault_runs = np.stack([features[run_uids == u].mean(axis=0) for u in np.unique(uid_fault)])
    if early_mask.sum() == 0 or np.unique(uid_early).size == 0:
        early_runs = fault_runs
    else:
        early_runs = np.stack([features[run_uids == u].mean(axis=0) for u in np.unique(uid_early)])

    _, _, c_ref = _scores_from_runs(normal_runs, fault_runs, early_runs, weights)
    m_ref = _top_mask(c_ref, critical_ratio)

    rng = np.random.default_rng(int(bootstrap_seed))
    rank_values: list[float] = []
    mask_values: list[float] = []
    combined_values: list[float] = []
    all_normal_groups = list(np.unique(uid_normal))
    all_fault_groups = list(np.unique(uid_fault))
    all_early_groups = list(np.unique(uid_early)) if early_mask.sum() > 0 else list(np.unique(uid_fault))

    for _ in range(int(bootstrap_repeats)):
        if sampling_unit == "run_uid" or sampling_unit == "instance_id":
            sampled_normal = normal_runs[rng.integers(0, len(normal_runs), len(normal_runs))]
            sampled_fault = fault_runs[rng.integers(0, len(fault_runs), len(fault_runs))]
            sampled_early = early_runs[rng.integers(0, len(early_runs), len(early_runs))]
        else:
            raise ValueError(f"unknown sampling_unit: {sampling_unit}")
        _, _, c_boot = _scores_from_runs(sampled_normal, sampled_fault, sampled_early, weights)
        m_boot = _top_mask(c_boot, critical_ratio)
        sr = spearman_rank_reliability(c_boot, c_ref)
        mr = mask_reliability(m_boot, m_ref)
        cr = combine_reliability(sr, mr)
        rank_values.append(sr)
        mask_values.append(mr)
        combined_values.append(cr)

    rank_arr = np.array(rank_values, dtype=np.float64)
    mask_arr = np.array(mask_values, dtype=np.float64)
    comb_arr = np.array(combined_values, dtype=np.float64)

    def _quants(arr: np.ndarray) -> dict[str, float]:
        if arr.size == 0:
            return {"q05": float("nan"), "q25": float("nan"), "q50": float("nan"), "q75": float("nan"), "q95": float("nan")}
        q05, q25, q50, q75, q95 = np.quantile(arr, [0.05, 0.25, 0.5, 0.75, 0.95])
        return {"q05": float(q05), "q25": float(q25), "q50": float(q50), "q75": float(q75), "q95": float(q95)}

    r_rank = float(np.nanmedian(rank_arr)) if np.isfinite(rank_arr).any() else float("nan")
    r_mask = float(np.nanmedian(mask_arr)) if np.isfinite(mask_arr).any() else float("nan")
    R = combine_reliability(r_rank, r_mask)

    ref_hash = hashlib.sha256(np.ascontiguousarray(c_ref, dtype=np.float64).tobytes()).hexdigest()
    mask_hash = hashlib.sha256(np.ascontiguousarray(m_ref, dtype=bool).view(np.uint8).tobytes()).hexdigest()

    return {
        "r_rank": r_rank,
        "r_mask": r_mask,
        "R": R,
        "r_rank_quantiles": _quants(rank_arr),
        "r_mask_quantiles": _quants(mask_arr),
        "R_quantiles": _quants(comb_arr),
        "bootstrap_repeats": int(bootstrap_repeats),
        "bootstrap_seed": int(bootstrap_seed),
        "critical_ratio": float(critical_ratio),
        "weights": dict(weights),
        "sampling_unit": sampling_unit,
        "n_normal_runs": int(normal_runs.shape[0]),
        "n_fault_runs": int(fault_runs.shape[0]),
        "n_early_runs": int(early_runs.shape[0]),
        "reference_map_sha256": ref_hash,
        "reference_mask_sha256": mask_hash,
        "c_ref_shape": list(c_ref.shape),
        "m_ref_true_count": int(m_ref.sum()),
    }


def choose_rho_utility(row: dict[str, float]) -> tuple[float, float, float, float]:
    return (
        float(row.get("macro_f1", float("-inf"))),
        float(row.get("auprc", float("-inf"))),
        -float(row.get("far", float("inf"))),
        -float(row.get("rho", float("inf"))),
    )


def compute_full_rho_regret(candidate_rows: list[dict[str, Any]], rho_ref: float = 1.0) -> dict[str, Any]:
    if not candidate_rows:
        return {"regret_rho1": float("nan"), "V_max": float("nan"), "V_ref": float("nan"), "rho_opt": float("nan")}
    utilities = [(choose_rho_utility(r), float(r.get("rho", float("nan")))) for r in candidate_rows]
    best_util, rho_opt = max(utilities, key=lambda t: t[0])
    ref_candidates = [t for t in utilities if abs(t[1] - rho_ref) < 1e-9]
    if not ref_candidates:
        return {"regret_rho1": float("nan"), "V_max": float("nan"), "V_ref": float("nan"), "rho_opt": float(rho_opt)}
    ref_util = ref_candidates[0][0]

    def util_to_scalar(u: tuple[float, float, float, float]) -> float:
        return 1e4 * u[0] + 1e2 * u[1] + 1e0 * u[2] + 1e-2 * u[3]

    V_max = util_to_scalar(best_util)
    V_ref = util_to_scalar(ref_util)
    return {
        "regret_rho1": float(max(0.0, V_max - V_ref)),
        "V_max": V_max,
        "V_ref": V_ref,
        "rho_opt": float(rho_opt),
    }


def _spearman(xs: np.ndarray, ys: np.ndarray) -> float:
    if xs.size != ys.size or xs.size < 2:
        return float("nan")
    mask = np.isfinite(xs) & np.isfinite(ys)
    if mask.sum() < 2:
        return float("nan")
    rx = _rankdata(xs[mask])
    ry = _rankdata(ys[mask])
    dx = rx - rx.mean()
    dy = ry - ry.mean()
    denom = float(np.sqrt(np.sum(dx * dx) * np.sum(dy * dy)))
    if denom < 1e-30:
        return float("nan")
    return float(np.sum(dx * dy) / denom)


def outer_aware_association(cells: list[dict[str, Any]], *, bootstrap_seed: int = 22042,
                            bootstrap_repeats: int = 1000) -> dict[str, Any]:
    datasets = [str(c.get("dataset", "")) for c in cells]
    outers = [int(c.get("outer_id", -1)) for c in cells]
    R = np.array([float(c.get("R", float("nan"))) for c in cells], dtype=np.float64)
    regret = np.array([float(c.get("regret_rho1", float("nan"))) for c in cells], dtype=np.float64)

    def slice_mask(ds: str | None) -> np.ndarray:
        if ds is None:
            return np.ones(len(cells), dtype=bool)
        return np.array([d == ds for d in datasets], dtype=bool)

    result: dict[str, Any] = {"n_cells": int(len(cells)), "test_rows_used": False}
    rng = np.random.default_rng(int(bootstrap_seed))
    for label, ds in [("3W", "3W"), ("TEP", "TEP"), ("pooled", None)]:
        mask = slice_mask(ds)
        if mask.sum() < 2:
            result[f"{label.lower()}_spearman"] = float("nan")
            result[f"{label.lower()}_n"] = int(mask.sum())
            continue
        local_outers = np.array(outers)[mask]
        local_R = R[mask]
        local_regret = regret[mask]
        exact = _spearman(local_R, local_regret)
        result[f"{label.lower()}_spearman"] = float(exact)
        result[f"{label.lower()}_n"] = int(mask.sum())
        if ds is None:
            unique_pairs = sorted(set(zip(datasets, outers)))
        else:
            unique_pairs = sorted(set(zip(np.array(datasets)[mask], local_outers)))
        boot: list[float] = []
        for _ in range(int(bootstrap_repeats)):
            idx = rng.integers(0, len(unique_pairs), len(unique_pairs))
            chosen = [unique_pairs[i] for i in idx]
            rows_mask = np.array([(d, o) in chosen for d, o in zip(np.array(datasets)[mask] if ds else datasets,
                                                                   local_outers if ds else outers)], dtype=bool)
            if rows_mask.sum() < 2:
                continue
            boot.append(_spearman(local_R[rows_mask] if ds else R[rows_mask],
                                  local_regret[rows_mask] if ds else regret[rows_mask]))
        if boot:
            bar = np.array(boot, dtype=np.float64)
            q05, q50, q95 = np.nanquantile(bar, [0.05, 0.5, 0.95])
            lt = float(np.nanmean((bar < 0).astype(np.float64)))
            result[f"{label.lower()}_bootstrap_q05"] = float(q05)
            result[f"{label.lower()}_bootstrap_q50"] = float(q50)
            result[f"{label.lower()}_bootstrap_q95"] = float(q95)
            if label.lower() == "pooled":
                result["bootstrap_q05"] = float(q05)
                result["bootstrap_q50"] = float(q50)
                result["bootstrap_q95"] = float(q95)
                result["P_assoc_lt_zero"] = float(lt)
    return result
