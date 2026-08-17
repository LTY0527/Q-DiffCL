from __future__ import annotations

from typing import Any

import numpy as np

from .criticality import (_robust_normalize, _run_means, _top_mask,
                          build_criticality, fault_type)


SHARED_WEIGHTS = {"weight_discriminative": .5, "weight_early": .3,
                  "weight_run_stability": .2}


def _soft_mask(score: np.ndarray, hard_mask: np.ndarray) -> np.ndarray:
    threshold = float(np.min(score[hard_mask]))
    scale = max(float(np.quantile(score, .75) - np.quantile(score, .25)), 1e-8)
    return (1 / (1 + np.exp(np.clip(-(score - threshold) / scale, -30, 30)))).astype(np.float32)


def _class_balanced_one_vs_rest(run_means: np.ndarray, run_types: np.ndarray,
                                target: int) -> np.ndarray:
    kinds = list(map(int, np.unique(run_types)))
    if target not in kinds or len(kinds) < 2:
        raise ValueError(f"diagnostic class {target} lacks one-vs-rest support")
    target_group = run_means[run_types == target]
    rest_kinds = [kind for kind in kinds if kind != target]
    target_centroid = target_group.mean(0); target_variance = target_group.var(0)
    rest_centroid = np.mean([run_means[run_types == kind].mean(0) for kind in rest_kinds], axis=0)
    rest_variance = np.mean([run_means[run_types == kind].var(0) for kind in rest_kinds], axis=0)
    raw = np.square(target_centroid - rest_centroid) / (target_variance + rest_variance + 1e-8)
    return _robust_normalize(raw)


def build_hierarchical_criticality(features: np.ndarray, bundle: dict[str, np.ndarray],
                                   stages: np.ndarray, settings: dict[str, Any],
                                   raw_log_amplitude: np.ndarray | None = None) -> dict[str, Any]:
    """Fit frozen R1 shared semantics and equal-class diagnostic semantics on train only."""
    shared_settings = dict(settings)
    for key, value in SHARED_WEIGHTS.items():
        if float(shared_settings[key]) != value:
            raise ValueError("HFSC shared criticality must freeze R1 D/E/S=0.5/0.3/0.2")
    if float(settings.get("hierarchical_shared_weight", .5)) != .5 or float(settings.get("hierarchical_diagnostic_weight", .5)) != .5:
        raise ValueError("HFSC hierarchy must freeze shared/diagnostic=0.5/0.5")
    shared_settings["weight_multiclass"] = 0.0; shared_settings.pop("multiclass_mode", None)
    shared = build_criticality(features, bundle, stages, shared_settings, raw_log_amplitude)
    labels = np.asarray(bundle["labels"]); run_uids = np.asarray(bundle["run_uid"])
    fault_means, fault_ids = _run_means(np.asarray(features), run_uids, labels != 0)
    run_types = np.asarray([fault_type(str(uid)) for uid in fault_ids], dtype=np.int64)
    classes = list(map(int, settings.get("diagnostic_classes", sorted(np.unique(run_types)))))
    if set(classes) != set(map(int, np.unique(run_types))):
        raise ValueError("HFSC diagnostic classes must exactly cover train fault classes")
    ratio = float(settings["critical_ratio"]); diagnostic = {}; hierarchical = {}
    for kind in classes:
        score = _class_balanced_one_vs_rest(fault_means, run_types, kind)
        diag_hard = _top_mask(score, ratio); diag_soft = _soft_mask(score, diag_hard)
        combined = .5 * shared["composite"] + .5 * score
        hier_hard = _top_mask(combined, ratio); hier_soft = _soft_mask(combined, hier_hard)
        diagnostic[kind] = {"score": score, "hard_mask": diag_hard, "soft_mask": diag_soft}
        hierarchical[kind] = {"score": combined, "hard_mask": hier_hard, "soft_mask": hier_soft}
    soft_masks = {0: shared["soft_mask"], **{kind: item["soft_mask"] for kind, item in hierarchical.items()}}
    return {"fit_split": "train", "shared": shared, "diagnostic": diagnostic,
            "hierarchical": hierarchical, "soft_masks": soft_masks,
            "fault_run_counts": {kind: int((run_types == kind).sum()) for kind in classes},
            "diagnostic_classes": classes, "shared_weight": .5, "diagnostic_weight": .5}
