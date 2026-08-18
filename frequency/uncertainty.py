from __future__ import annotations

from typing import Any

import numpy as np

from .criticality import (_run_means, _scores_from_runs, _top_mask,
                          build_criticality, mask_jaccard)


FROZEN_REPEATS = 64
FROZEN_WEIGHTS = {"weight_discriminative": .5, "weight_early": .3,
                  "weight_run_stability": .2}


def assignment_confidence(selection_probability: np.ndarray) -> np.ndarray:
    probability = np.asarray(selection_probability, dtype=np.float64)
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("selection probability must be finite in [0,1]")
    return (2 * np.abs(probability - .5)).astype(np.float32)


def _unit_means(features: np.ndarray, selector: np.ndarray, unit_ids: np.ndarray) -> dict[str, np.ndarray]:
    return {str(unit): features[selector & (unit_ids == unit)].mean(0)
            for unit in np.unique(unit_ids[selector])}


def _sample_stratified_units(unit_ids: np.ndarray, strata: np.ndarray,
                             rng: np.random.Generator) -> np.ndarray:
    sampled = []
    for stratum in np.unique(strata):
        group = np.unique(unit_ids[strata == stratum])
        sampled.extend(group[rng.integers(0, len(group), len(group))])
    return np.asarray(sampled, dtype=object)


def build_uncertainty_gated_criticality(features: np.ndarray, bundle: dict[str, np.ndarray],
                                        stages: np.ndarray, unit_ids: np.ndarray,
                                        unit_strata: np.ndarray, settings: dict[str, Any],
                                        raw_log_amplitude: np.ndarray | None = None) -> dict[str, Any]:
    """Bootstrap the complete frozen R1 D/E/S composite using train units only."""
    for key, value in FROZEN_WEIGHTS.items():
        if float(settings[key]) != value:
            raise ValueError("UG-R1 must freeze D/E/S weights at 0.5/0.3/0.2")
    if float(settings["critical_ratio"]) != .30:
        raise ValueError("UG-R1 critical ratio must freeze at 0.30")
    if int(settings["bootstrap_repeats"]) != FROZEN_REPEATS:
        raise ValueError("UG-R1 bootstrap repeats must freeze at 64")
    features = np.asarray(features); labels = np.asarray(bundle["labels"])
    stages = np.asarray(stages); unit_ids = np.asarray(unit_ids, dtype=object); unit_strata = np.asarray(unit_strata)
    if not (len(features) == len(labels) == len(stages) == len(unit_ids) == len(unit_strata)):
        raise ValueError("UG-R1 train arrays must align")
    if not len(features) or not np.isfinite(features).all():
        raise ValueError("UG-R1 requires finite train features")
    if len(np.unique(unit_ids)) == len(unit_ids):
        raise ValueError("UG-R1 forbids window-level bootstrap units")
    r1 = build_criticality(features, bundle, stages, settings, raw_log_amplitude)
    normal = _unit_means(features, labels == 0, unit_ids)
    fault = _unit_means(features, labels != 0, unit_ids)
    early = _unit_means(features, stages == "early", unit_ids)
    unique_units = np.unique(unit_ids)
    unit_to_stratum = {}
    for unit in unique_units:
        values = np.unique(unit_strata[unit_ids == unit])
        if len(values) != 1: raise ValueError(f"bootstrap unit {unit} spans multiple strata")
        unit_to_stratum[str(unit)] = int(values[0])
    unit_level_strata = np.asarray([unit_to_stratum[str(unit)] for unit in unique_units], dtype=np.int64)
    rng = np.random.default_rng(int(settings["bootstrap_seed"])); selected = []; overlaps = []
    reference = np.asarray(r1["masks"]["composite"], bool)
    zero_multiclass = np.zeros(features.shape[1:], dtype=np.float64)
    for _ in range(FROZEN_REPEATS):
        sampled = _sample_stratified_units(unique_units, unit_level_strata, rng)
        normal_runs = np.stack([normal[str(unit)] for unit in sampled if str(unit) in normal])
        fault_runs = np.stack([fault[str(unit)] for unit in sampled if str(unit) in fault])
        early_runs = np.stack([early[str(unit)] for unit in sampled if str(unit) in early])
        _, _, _, composite = _scores_from_runs(normal_runs, fault_runs, early_runs, zero_multiclass, settings)
        mask = _top_mask(composite, .30); selected.append(mask); overlaps.append(mask_jaccard(mask, reference))
    probability = np.mean(np.stack(selected), axis=0).astype(np.float32)
    confidence = assignment_confidence(probability)
    return {"fit_split": "train", "r1": r1, "selection_probability": probability,
            "assignment_confidence": confidence, "bootstrap_repeats": FROZEN_REPEATS,
            "bootstrap_overlap": np.asarray(overlaps), "bootstrap_unit_count": int(len(unique_units)),
            "bootstrap_unit_ids": list(map(str, unique_units)),
            "stratified_unit_counts": {str(kind): int(np.sum(unit_level_strata == kind))
                                       for kind in np.unique(unit_level_strata)},
            "bootstrap_scope": "train units only; complete D/E/S robust-normalized composite"}
