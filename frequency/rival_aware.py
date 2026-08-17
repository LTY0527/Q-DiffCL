from __future__ import annotations

from typing import Any

import numpy as np

from .criticality import (_robust_normalize, _run_means, _top_mask,
                          build_criticality, fault_type)
from .hierarchical import SHARED_WEIGHTS, _soft_mask


HARD_RIVAL_QUANTILE = .25


def _pairwise_fisher(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.square(first.mean(0) - second.mean(0)) / (first.var(0) + second.var(0) + 1e-8)


def _diagnostic_for_class(run_means: np.ndarray, run_types: np.ndarray,
                          target: int) -> tuple[np.ndarray, dict[int, dict[str, np.ndarray]]]:
    rivals = [int(kind) for kind in np.unique(run_types) if int(kind) != target]
    if target not in run_types or not rivals:
        raise ValueError(f"RRDC diagnostic class {target} lacks fault-vs-fault support")
    target_group = run_means[run_types == target]
    pairwise = {}
    normalized = []
    for rival in rivals:
        raw = _pairwise_fisher(target_group, run_means[run_types == rival])
        score = _robust_normalize(raw)
        pairwise[rival] = {"raw_fisher": raw, "normalized_fisher": score}
        normalized.append(score)
    aggregate = np.quantile(np.stack(normalized), HARD_RIVAL_QUANTILE, axis=0)
    return _robust_normalize(aggregate), pairwise


def _stratified_bootstrap(run_means: np.ndarray, run_types: np.ndarray,
                          rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    means = []; types = []
    for kind in np.unique(run_types):
        group = run_means[run_types == kind]
        sampled = group[rng.integers(0, len(group), len(group))]
        means.extend(sampled); types.extend([kind] * len(sampled))
    return np.stack(means), np.asarray(types, dtype=np.int64)


def build_rival_aware_criticality(features: np.ndarray, bundle: dict[str, np.ndarray],
                                  stages: np.ndarray, settings: dict[str, Any],
                                  raw_log_amplitude: np.ndarray | None = None) -> dict[str, Any]:
    """Fit frozen R1 shared plus reliable fault-vs-fault diagnostic maps on train only."""
    for key, value in SHARED_WEIGHTS.items():
        if float(settings[key]) != value:
            raise ValueError("RRDC shared criticality must freeze R1 D/E/S=0.5/0.3/0.2")
    if float(settings.get("hard_rival_quantile", HARD_RIVAL_QUANTILE)) != HARD_RIVAL_QUANTILE:
        raise ValueError("RRDC hard-rival aggregation must freeze Q25")
    ratio = float(settings["critical_ratio"])
    if ratio != .30:
        raise ValueError("RRDC critical ratio must freeze at 0.30")
    repeats = int(settings["bootstrap_repeats"])
    if repeats < 1:
        raise ValueError("RRDC bootstrap_repeats must be positive")

    shared_settings = dict(settings); shared_settings["weight_multiclass"] = 0.0
    shared_settings.pop("multiclass_mode", None)
    shared = build_criticality(features, bundle, stages, shared_settings, raw_log_amplitude)
    labels = np.asarray(bundle["labels"]); run_uids = np.asarray(bundle["run_uid"])
    fault_means, fault_ids = _run_means(np.asarray(features), run_uids, labels != 0)
    run_types = np.asarray([fault_type(str(uid)) for uid in fault_ids], dtype=np.int64)
    classes = list(map(int, settings.get("diagnostic_classes", sorted(np.unique(run_types)))))
    if set(classes) != set(map(int, np.unique(run_types))) or 0 in classes:
        raise ValueError("RRDC diagnostic classes must exactly cover train fault classes and exclude Normal")

    diagnostic = {}; final = {}; reliability_rng = np.random.default_rng(int(settings["bootstrap_seed"]))
    bootstraps = [_stratified_bootstrap(fault_means, run_types, reliability_rng) for _ in range(repeats)]
    for kind in classes:
        score, pairwise = _diagnostic_for_class(fault_means, run_types, kind)
        selections = []
        for sampled_means, sampled_types in bootstraps:
            boot_score, _ = _diagnostic_for_class(sampled_means, sampled_types, kind)
            selections.append(_top_mask(boot_score, ratio))
        reliability = np.mean(np.stack(selections), axis=0).astype(np.float32)
        reliable_score = score * reliability
        diag_hard = _top_mask(reliable_score, ratio); diag_soft = _soft_mask(reliable_score, diag_hard)
        combined = shared["composite"] + reliable_score
        final_hard = _top_mask(combined, ratio); final_soft = _soft_mask(combined, final_hard)
        summaries = {rival: {
            "raw_fisher_mean": float(item["raw_fisher"].mean()),
            "raw_fisher_median": float(np.median(item["raw_fisher"])),
            "normalized_fisher_mean": float(item["normalized_fisher"].mean()),
        } for rival, item in pairwise.items()}
        hardest = min(summaries, key=lambda rival: summaries[rival]["raw_fisher_median"])
        diagnostic[kind] = {
            "score": score, "reliability": reliability, "reliable_score": reliable_score,
            "hard_mask": diag_hard, "soft_mask": diag_soft, "pairwise": pairwise,
            "pairwise_summary": summaries, "hardest_rival": int(hardest),
            "hardest_rival_score": summaries[hardest]["raw_fisher_median"],
        }
        final[kind] = {"score": combined, "hard_mask": final_hard, "soft_mask": final_soft}
    soft_masks = {0: shared["soft_mask"], **{kind: item["soft_mask"] for kind, item in final.items()}}
    return {
        "fit_split": "train", "shared": shared, "diagnostic": diagnostic, "final": final,
        "soft_masks": soft_masks, "fault_run_counts": {kind: int((run_types == kind).sum()) for kind in classes},
        "diagnostic_classes": classes, "hard_rival_quantile": HARD_RIVAL_QUANTILE,
        "bootstrap_repeats": repeats, "combination": "C_shared + C_diag * R",
    }
