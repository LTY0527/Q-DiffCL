from __future__ import annotations

from typing import Any

import numpy as np

from .criticality import _run_means, _scores_from_runs, build_criticality, fault_type


RANK_THRESHOLD = 0.70
FROZEN_BOOTSTRAP_REPEATS = 64
FROZEN_WEIGHTS = {
    "weight_discriminative": 0.5,
    "weight_early": 0.3,
    "weight_run_stability": 0.2,
}


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    """Return average-tie percentile ranks independently for each replicate."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 2 or values.shape[0] < 1 or not np.isfinite(values).all():
        raise ValueError("rank input must be finite [R,...]")
    flattened = values.reshape(values.shape[0], -1)
    ranked = np.empty_like(flattened)
    for row_index, row in enumerate(flattened):
        order = np.argsort(row, kind="mergesort")
        sorted_values = row[order]
        start = 0
        while start < len(row):
            stop = start + 1
            while stop < len(row) and sorted_values[stop] == sorted_values[start]:
                stop += 1
            average_rank = 0.5 * (start + stop - 1)
            ranked[row_index, order[start:stop]] = average_rank / max(len(row) - 1, 1)
            start = stop
    return ranked.reshape(values.shape).astype(np.float32)


def summarize_rank_distribution(composites: np.ndarray, threshold: float = RANK_THRESHOLD) -> dict[str, Any]:
    ranks = percentile_ranks(composites)
    q25 = np.quantile(ranks, 0.25, axis=0)
    q75 = np.quantile(ranks, 0.75, axis=0)
    reliable_critical = q25 >= float(threshold)
    reliable_noncritical = q75 < float(threshold)
    ambiguous = ~(reliable_critical | reliable_noncritical)
    if np.any(reliable_critical & reliable_noncritical) or not np.all(
            reliable_critical | reliable_noncritical | ambiguous):
        raise RuntimeError("domain reliability masks do not form a partition")
    return {
        "ranks": ranks,
        "rank_median": np.median(ranks, axis=0).astype(np.float32),
        "rank_q25": q25.astype(np.float32),
        "rank_q75": q75.astype(np.float32),
        "rank_iqr": (q75 - q25).astype(np.float32),
        "reliable_critical": reliable_critical,
        "reliable_noncritical": reliable_noncritical,
        "ambiguous": ambiguous,
        "rank_threshold": float(threshold),
    }


def _validate_frozen(settings: dict[str, Any]) -> None:
    for key, expected in FROZEN_WEIGHTS.items():
        if float(settings[key]) != expected:
            raise ValueError("DRFD freezes R1 D/E/S weights at 0.5/0.3/0.2")
    if float(settings["critical_ratio"]) != 0.30:
        raise ValueError("DRFD freezes critical_ratio at 0.30")


def _complete_r1_composite(features: np.ndarray, bundle: dict[str, np.ndarray],
                           stages: np.ndarray, settings: dict[str, Any]) -> np.ndarray:
    run_uids = np.asarray(bundle["run_uid"])
    labels = np.asarray(bundle["labels"])
    normal_runs, _ = _run_means(features, run_uids, labels == 0)
    fault_runs, _ = _run_means(features, run_uids, labels != 0)
    early_runs, _ = _run_means(features, run_uids, np.asarray(stages) == "early")
    zeros = np.zeros(features.shape[1:], dtype=np.float64)
    return _scores_from_runs(normal_runs, fault_runs, early_runs, zeros, settings)[-1]


def build_three_w_leave_one_well_out(
    features: np.ndarray,
    bundle: dict[str, np.ndarray],
    stages: np.ndarray,
    well_ids: np.ndarray,
    settings: dict[str, Any],
    raw_log_amplitude: np.ndarray | None = None,
) -> dict[str, Any]:
    """Recompute the complete frozen R1 composite after removing each train WELL."""
    _validate_frozen(settings)
    features = np.asarray(features)
    stages = np.asarray(stages)
    well_ids = np.asarray(well_ids, dtype=object)
    labels = np.asarray(bundle["labels"])
    if not (len(features) == len(labels) == len(stages) == len(well_ids)):
        raise ValueError("3W DRFD train arrays must align")
    if not len(features) or not np.isfinite(features).all():
        raise ValueError("3W DRFD requires finite train features")
    wells = np.unique(well_ids)
    if len(wells) < 3 or len(wells) == len(well_ids):
        raise ValueError("3W reliability requires aggregate WELL units")
    r1 = build_criticality(features, bundle, stages, settings, raw_log_amplitude)
    composites = []
    profiles = []
    for well in wells:
        keep = well_ids != well
        subset = {key: np.asarray(value)[keep] for key, value in bundle.items()
                  if np.asarray(value).ndim and len(np.asarray(value)) == len(keep)}
        composite = _complete_r1_composite(features[keep], subset, stages[keep], settings)
        composites.append(composite)
        profiles.append({"omitted_well": str(well), "remaining_windows": int(keep.sum())})
    summary = summarize_rank_distribution(np.stack(composites))
    return {
        "fit_split": "train",
        "resampling": "leave-one-WELL-out",
        "unit_ids": list(map(str, wells)),
        "replicate_count": int(len(wells)),
        "profiles": profiles,
        "composites": np.stack(composites),
        "r1": r1,
        **summary,
    }


def build_tep_stratified_run_bootstrap(
    features: np.ndarray,
    bundle: dict[str, np.ndarray],
    stages: np.ndarray,
    settings: dict[str, Any],
    raw_log_amplitude: np.ndarray | None = None,
) -> dict[str, Any]:
    """Bootstrap train runs within faultNumber strata and recompute frozen R1 D/E/S."""
    _validate_frozen(settings)
    if int(settings["bootstrap_repeats"]) != FROZEN_BOOTSTRAP_REPEATS:
        raise ValueError("TEP DRFD bootstrap_repeats must freeze at 64")
    features = np.asarray(features)
    stages = np.asarray(stages)
    run_uids = np.asarray(bundle["run_uid"], dtype=object)
    labels = np.asarray(bundle["labels"])
    if not (len(features) == len(labels) == len(stages) == len(run_uids)):
        raise ValueError("TEP DRFD train arrays must align")
    unique_runs = np.unique(run_uids)
    if len(unique_runs) == len(run_uids):
        raise ValueError("TEP reliability requires run-level aggregate units")
    strata = np.asarray([fault_type(str(run)) for run in unique_runs], dtype=np.int64)
    normal = {str(run): features[(run_uids == run) & (labels == 0)].mean(0)
              for run in unique_runs if np.any((run_uids == run) & (labels == 0))}
    fault = {str(run): features[(run_uids == run) & (labels != 0)].mean(0)
             for run in unique_runs if np.any((run_uids == run) & (labels != 0))}
    early = {str(run): features[(run_uids == run) & (stages == "early")].mean(0)
             for run in unique_runs if np.any((run_uids == run) & (stages == "early"))}
    rng = np.random.default_rng(int(settings["bootstrap_seed"]))
    zeros = np.zeros(features.shape[1:], dtype=np.float64)
    composites = []
    for _ in range(FROZEN_BOOTSTRAP_REPEATS):
        sampled = []
        for kind in np.unique(strata):
            group = unique_runs[strata == kind]
            sampled.extend(group[rng.integers(0, len(group), len(group))])
        normal_runs = np.stack([normal[str(run)] for run in sampled if str(run) in normal])
        fault_runs = np.stack([fault[str(run)] for run in sampled if str(run) in fault])
        early_runs = np.stack([early[str(run)] for run in sampled if str(run) in early])
        composites.append(_scores_from_runs(normal_runs, fault_runs, early_runs, zeros, settings)[-1])
    r1 = build_criticality(features, bundle, stages, settings, raw_log_amplitude)
    summary = summarize_rank_distribution(np.stack(composites))
    return {
        "fit_split": "train",
        "resampling": "faultNumber-stratified-run-bootstrap",
        "unit_ids": list(map(str, unique_runs)),
        "stratified_unit_counts": {str(kind): int(np.sum(strata == kind)) for kind in np.unique(strata)},
        "replicate_count": FROZEN_BOOTSTRAP_REPEATS,
        "composites": np.stack(composites),
        "r1": r1,
        **summary,
    }
