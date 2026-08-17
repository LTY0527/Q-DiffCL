from __future__ import annotations

from typing import Any

import numpy as np

from .criticality import (_fisher, _robust_normalize, _run_means, _top_mask,
                          build_criticality, mask_jaccard)
from .hierarchical import _soft_mask


HORIZONS = 8
LEAD_DECAY = .35
EWIC_WEIGHTS = {"weight_discriminative": .5, "weight_early": .3,
                "weight_run_stability": .2}


def onset_horizons(bundle: dict[str, np.ndarray], onset_by_source: dict[str, int],
                   stride: int, horizon_count: int = HORIZONS) -> np.ndarray:
    """Map complete post-onset windows to 1-based onset-relative horizons."""
    if int(horizon_count) != HORIZONS:
        raise ValueError("EWIC must freeze H=8")
    starts = np.asarray(bundle["start_sample"], dtype=np.int64)
    labels = np.asarray(bundle["labels"], dtype=np.int64)
    sources = np.asarray([str(uid).split(":", 1)[0] for uid in bundle["run_uid"]])
    result = np.zeros(len(labels), dtype=np.int16)
    for source in np.unique(sources):
        onset = int(onset_by_source[str(source)])
        selected = (sources == source) & (labels != 0) & (starts >= onset)
        values = ((starts[selected] - onset) // int(stride)) + 1
        result[selected] = np.where(values <= HORIZONS, values, 0)
    return result


def _horizon_run_means(features: np.ndarray, run_uids: np.ndarray,
                       horizons: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    return _run_means(features, run_uids, horizons == horizon)


def _lead_score(normal_runs: np.ndarray, horizon_groups: list[np.ndarray],
                weights: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    maps = [_fisher(normal_runs, group) for group in horizon_groups]
    normalized = [_robust_normalize(value) for value in maps]
    return np.sum(np.stack(normalized) * weights[:, None, None], axis=0), maps


def build_early_warning_criticality(features: np.ndarray, bundle: dict[str, np.ndarray],
                                    stages: np.ndarray, horizons: np.ndarray,
                                    settings: dict[str, Any], raw_log_amplitude: np.ndarray | None = None,
                                    early_features: np.ndarray | None = None,
                                    early_bundle: dict[str, np.ndarray] | None = None) -> dict[str, Any]:
    """Replace R1 E with lead-time-weighted, run/WELL-reliable train-only early Fisher."""
    for key, value in EWIC_WEIGHTS.items():
        if float(settings[key]) != value:
            raise ValueError("EWIC must freeze R1 D/E/S weights at 0.5/0.3/0.2")
    if int(settings.get("horizon_count", HORIZONS)) != HORIZONS:
        raise ValueError("EWIC must freeze H=8")
    if float(settings.get("lead_decay", LEAD_DECAY)) != LEAD_DECAY:
        raise ValueError("EWIC must freeze lead decay at 0.35")
    ratio = float(settings["critical_ratio"])
    if ratio != .30:
        raise ValueError("EWIC critical ratio must freeze at 0.30")
    repeats = int(settings["bootstrap_repeats"])
    if repeats < 1:
        raise ValueError("EWIC bootstrap_repeats must be positive")
    horizon_features = np.asarray(features if early_features is None else early_features)
    horizon_bundle = bundle if early_bundle is None else early_bundle
    horizons = np.asarray(horizons, dtype=np.int16)
    if len(horizons) != len(horizon_features) or np.any((horizons < 0) | (horizons > HORIZONS)):
        raise ValueError("EWIC horizons must align with early features and lie in [0,8]")

    r1 = build_criticality(features, bundle, np.asarray(stages), settings, raw_log_amplitude)
    run_uids = np.asarray(bundle["run_uid"]); labels = np.asarray(bundle["labels"])
    normal_runs, _ = _run_means(features, run_uids, labels == 0)
    horizon_uids = np.asarray(horizon_bundle["run_uid"])
    groups = []; group_ids = []; coverage = {}
    for horizon in range(1, HORIZONS + 1):
        means, ids = _horizon_run_means(horizon_features, horizon_uids, horizons, horizon)
        groups.append(means); group_ids.append(ids)
        coverage[str(horizon)] = {"windows": int((horizons == horizon).sum()), "runs_or_wells": int(len(ids))}
    raw_weights = np.exp(-LEAD_DECAY * np.arange(HORIZONS)); weights = raw_weights / raw_weights.sum()
    lead, horizon_maps = _lead_score(normal_runs, groups, weights)

    all_fault_ids = np.unique(np.concatenate(group_ids)); rng = np.random.default_rng(int(settings["bootstrap_seed"]))
    selections = []; overlaps = []; lead_hard = _top_mask(lead, ratio)
    group_lookup = [{uid: value for uid, value in zip(ids, means)} for ids, means in zip(group_ids, groups)]
    for _ in range(repeats):
        sampled_ids = all_fault_ids[rng.integers(0, len(all_fault_ids), len(all_fault_ids))]
        sampled_groups = []
        for lookup in group_lookup:
            values = [lookup[uid] for uid in sampled_ids if uid in lookup]
            if not values:
                raise ValueError("EWIC bootstrap produced an empty horizon")
            sampled_groups.append(np.stack(values))
        boot_lead, _ = _lead_score(normal_runs, sampled_groups, weights)
        selected = _top_mask(boot_lead, ratio); selections.append(selected)
        overlaps.append(mask_jaccard(selected, lead_hard))
    reliability = np.mean(np.stack(selections), axis=0).astype(np.float32)
    invariant = lead * reliability
    invariant_normalized = _robust_normalize(invariant)
    composite = (.5 * _robust_normalize(r1["discriminative"])
                 + .3 * invariant_normalized + .2 * _robust_normalize(r1["stability"]))
    hard_mask = _top_mask(composite, ratio); soft_mask = _soft_mask(composite, hard_mask)
    return {
        "fit_split": "train", "r1": r1, "horizon_count": HORIZONS, "lead_decay": LEAD_DECAY,
        "lead_weights": weights, "horizon_fisher": horizon_maps,
        "horizon_normalized": [_robust_normalize(value) for value in horizon_maps],
        "horizon_coverage": coverage, "early_lead": lead, "early_reliability": reliability,
        "early_invariant": invariant, "early_invariant_normalized": invariant_normalized,
        "composite": composite, "hard_mask": hard_mask, "soft_mask": soft_mask,
        "bootstrap_overlap": np.asarray(overlaps), "bootstrap_unit_count": int(len(all_fault_ids)),
        "component_weights": {"discriminative": .5, "early_invariant": .3, "stability": .2},
    }
