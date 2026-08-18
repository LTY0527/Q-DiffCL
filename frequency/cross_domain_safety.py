from __future__ import annotations

from typing import Any

import numpy as np

from .criticality import _fisher, _robust_normalize, _run_means, fault_type
from .domain_reliability import (_complete_r1_composite, _validate_frozen,
                                 percentile_ranks)


def _semantic_rank(features: np.ndarray, bundle: dict[str, np.ndarray],
                   stages: np.ndarray, selector: np.ndarray) -> np.ndarray:
    run_uids = np.asarray(bundle["run_uid"]); labels = np.asarray(bundle["labels"])
    normal, _ = _run_means(features, run_uids, selector & (labels == 0))
    fault, _ = _run_means(features, run_uids, selector & (labels != 0))
    early, _ = _run_means(features, run_uids, selector & (np.asarray(stages) == "early"))
    relevance = .7 * _robust_normalize(_fisher(normal, fault)) + .3 * _robust_normalize(_fisher(normal, early))
    return percentile_ranks(relevance[None])[0]


def _summarize(source_ranks: list[np.ndarray], heldout_ranks: list[np.ndarray],
               units: list[str], invalid: list[dict[str, str]], minimum_support: int = 3) -> dict[str, Any]:
    if source_ranks:
        source = np.stack(source_ranks); heldout = np.stack(heldout_ranks)
        unsafe = (source < .70) & (heldout >= .70)
        unsafe_rate = unsafe.mean(0); support = np.full(unsafe_rate.shape, len(source), dtype=np.int64)
    else:
        raise ValueError("CDVS has no valid pseudo-unseen domain")
    safe_prob = 1 - unsafe_rate
    safe_prob[support < int(minimum_support)] = 0
    return {"fit_split": "train", "source_ranks": source.astype(np.float32),
            "heldout_ranks": heldout.astype(np.float32), "unsafe": unsafe,
            "unsafe_rate": unsafe_rate.astype(np.float32), "safe_prob": safe_prob.astype(np.float32),
            "valid_support": support, "valid_unit_ids": units, "invalid_units": invalid,
            "minimum_support": int(minimum_support), "test_or_validation_used": False}


def build_three_w_cross_domain_safety(features: np.ndarray, bundle: dict[str, np.ndarray],
                                      stages: np.ndarray, well_ids: np.ndarray,
                                      settings: dict[str, Any]) -> dict[str, Any]:
    _validate_frozen(settings)
    features = np.asarray(features); stages = np.asarray(stages); well_ids = np.asarray(well_ids, dtype=object)
    if not (len(features) == len(stages) == len(well_ids) == len(bundle["labels"])):
        raise ValueError("3W CDVS train arrays must align")
    source_ranks = []; heldout_ranks = []; valid = []; invalid = []
    for well in np.unique(well_ids):
        heldout = well_ids == well; source = ~heldout
        subset = {key: np.asarray(value)[source] for key, value in bundle.items()
                  if np.asarray(value).ndim and len(np.asarray(value)) == len(source)}
        try:
            source_composite = _complete_r1_composite(features[source], subset, stages[source], settings)
            heldout_rank = _semantic_rank(features, bundle, stages, heldout)
        except ValueError as error:
            invalid.append({"unit_id": str(well), "reason": str(error)}); continue
        source_ranks.append(percentile_ranks(source_composite[None])[0]); heldout_ranks.append(heldout_rank)
        valid.append(str(well))
    result = _summarize(source_ranks, heldout_ranks, valid, invalid)
    result.update({"resampling": "pseudo-unseen-WELL", "all_unit_ids": list(map(str, np.unique(well_ids)))})
    return result


def stratified_run_folds(run_uids: np.ndarray, fold_count: int, seed: int) -> list[np.ndarray]:
    runs = np.unique(np.asarray(run_uids, dtype=object)); strata = np.asarray([fault_type(str(run)) for run in runs])
    rng = np.random.default_rng(int(seed)); folds: list[list[object]] = [[] for _ in range(int(fold_count))]
    for kind in np.unique(strata):
        group = runs[strata == kind].copy(); rng.shuffle(group)
        for index, run in enumerate(group): folds[index % fold_count].append(run)
    arrays = [np.asarray(fold, dtype=object) for fold in folds]
    flattened = np.concatenate(arrays)
    if len(flattened) != len(runs) or len(np.unique(flattened)) != len(runs):
        raise RuntimeError("TEP pseudo-domain folds overlap or omit runs")
    return arrays


def build_tep_cross_domain_safety(features: np.ndarray, bundle: dict[str, np.ndarray],
                                  stages: np.ndarray, settings: dict[str, Any],
                                  fold_count: int = 8, protocol_seed: int = 7) -> dict[str, Any]:
    _validate_frozen(settings)
    features = np.asarray(features); stages = np.asarray(stages); run_uids = np.asarray(bundle["run_uid"], dtype=object)
    if int(fold_count) != 8: raise ValueError("TEP CDVS freezes eight pseudo-domain folds")
    folds = stratified_run_folds(run_uids, fold_count, protocol_seed)
    source_ranks = []; heldout_ranks = []; valid = []; invalid = []
    for index, fold in enumerate(folds):
        heldout = np.isin(run_uids, fold); source = ~heldout
        subset = {key: np.asarray(value)[source] for key, value in bundle.items()
                  if np.asarray(value).ndim and len(np.asarray(value)) == len(source)}
        try:
            composite = _complete_r1_composite(features[source], subset, stages[source], settings)
            heldout_rank = _semantic_rank(features, bundle, stages, heldout)
        except ValueError as error:
            invalid.append({"unit_id": f"fold_{index}", "reason": str(error)}); continue
        source_ranks.append(percentile_ranks(composite[None])[0]); heldout_ranks.append(heldout_rank)
        valid.append(f"fold_{index}")
    result = _summarize(source_ranks, heldout_ranks, valid, invalid)
    result.update({"resampling": "eight-stratified-pseudo-unseen-run-folds",
                   "folds": [list(map(str, fold)) for fold in folds], "fold_count": 8,
                   "protocol_seed": int(protocol_seed)})
    return result
