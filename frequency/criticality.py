from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


STAGES = ("prefault", "early", "middle", "stable")


def fault_type(run_uid: str) -> int:
    match = re.search(r":fault_(\d+):", str(run_uid))
    return int(match.group(1)) if match else 0


def source_split(run_uid: str) -> str:
    value = str(run_uid).split(":", 1)[0]
    if value not in {"training", "testing"}:
        raise ValueError(f"unknown TEP source split: {value}")
    return value


def classify_stage(run_uid: str, start_sample: int, end_sample: int, config: dict[str, Any]) -> str:
    """Classify a window using its true onset and full-window post-fault progress.

    Raw delta is end-onset. Because transition windows are excluded, subtracting
    L-1 makes progress zero for a fully faulty window starting exactly at onset.
    This preserves the requested first-N-post-fault-window interpretation even
    when training and testing onsets have different absolute sample numbers.
    """
    if fault_type(run_uid) == 0:
        return "prefault"
    onset = int(config["protocol"]["fault_onset"][source_split(run_uid)])
    start, end = int(start_sample), int(end_sample)
    if end < onset:
        return "prefault"
    if start < onset <= end:
        return "transition"
    length = int(config["protocol"]["window_length"])
    stride = int(config["protocol"]["stride"])
    progress = (end - onset) - (length - 1)
    if progress < 0:
        raise ValueError("fully post-fault window has negative stage progress")
    if progress < int(config["stage"]["early_horizon_windows"]) * stride:
        return "early"
    if progress < int(config["stage"]["middle_horizon_windows"]) * stride:
        return "middle"
    return "stable"


def fault_stages(bundle: dict[str, np.ndarray], config: dict[str, Any]) -> np.ndarray:
    stages = np.asarray([
        classify_stage(uid, start, end, config)
        for uid, start, end in zip(bundle["run_uid"], bundle["start_sample"], bundle["end_sample"])
    ])
    if np.any(stages == "transition"):
        raise RuntimeError("fixed views contain transition windows despite exclude_transition protocol")
    labels = np.asarray(bundle["labels"])
    if np.any((labels != 0) & (stages == "prefault")):
        raise RuntimeError("fault-labelled fixed view classified as prefault")
    return stages


def log_amplitude_phase(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tensor = torch.as_tensor(np.asarray(values), dtype=torch.float32)
    spectrum = torch.fft.rfft(tensor, dim=-1)
    log_amplitude = torch.log1p(torch.abs(spectrum))
    phase = torch.angle(spectrum)
    if not torch.isfinite(log_amplitude).all() or not torch.isfinite(phase).all():
        raise FloatingPointError("non-finite frequency representation")
    return log_amplitude.cpu().numpy(), phase.cpu().numpy()


@dataclass(frozen=True)
class FrequencyScaler:
    mean: np.ndarray
    scale: np.ndarray
    fit_split: str = "train"

    def transform(self, values: np.ndarray) -> np.ndarray:
        result = (np.asarray(values) - self.mean) / self.scale
        if not np.isfinite(result).all():
            raise FloatingPointError("non-finite standardized spectrum")
        return result.astype(np.float32)


def fit_frequency_scaler(train_log_amplitude: np.ndarray, split: str = "train") -> FrequencyScaler:
    if split != "train":
        raise ValueError("frequency scaler may only be fitted on train")
    values = np.asarray(train_log_amplitude, dtype=np.float64)
    if values.ndim != 3 or not len(values) or not np.isfinite(values).all():
        raise ValueError("train log amplitude must be finite [N,C,F]")
    scale = values.std(0)
    return FrequencyScaler(values.mean(0), np.where(scale > 1e-8, scale, 1.0), split)


def _run_means(features: np.ndarray, run_uids: np.ndarray, selector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selected_runs = np.unique(run_uids[selector])
    if not len(selected_runs):
        raise ValueError("frequency criticality group has no runs")
    return np.stack([features[selector & (run_uids == uid)].mean(0) for uid in selected_runs]), selected_runs


def _fisher(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.square(first.mean(0) - second.mean(0)) / (first.var(0) + second.var(0) + 1e-8)


def _stability(normal_runs: np.ndarray, fault_runs: np.ndarray) -> np.ndarray:
    reference = np.median(normal_runs, axis=0)
    difference = fault_runs - reference
    direction = np.abs(np.mean(np.sign(difference), axis=0))
    magnitude = np.abs(difference)
    robust_cv = ((np.quantile(magnitude, .75, axis=0) - np.quantile(magnitude, .25, axis=0))
                 / (np.median(magnitude, axis=0) + 1e-8))
    return direction / (1 + robust_cv)


def _robust_normalize(values: np.ndarray) -> np.ndarray:
    median = np.median(values)
    iqr = np.quantile(values, .75) - np.quantile(values, .25)
    return np.clip((values - median) / max(float(iqr), 1e-8), -8, 8)


def _top_mask(values: np.ndarray, ratio: float) -> np.ndarray:
    count = max(1, min(values.size - 1, int(round(values.size * ratio))))
    indices = np.argpartition(values.reshape(-1), -count)[-count:]
    result = np.zeros(values.size, dtype=bool); result[indices] = True
    return result.reshape(values.shape)


def mask_jaccard(first: np.ndarray, second: np.ndarray) -> float:
    first, second = np.asarray(first, dtype=bool), np.asarray(second, dtype=bool)
    union = np.logical_or(first, second).sum()
    return float(np.logical_and(first, second).sum() / union) if union else 1.0


def _scores_from_runs(normal_runs: np.ndarray, fault_runs: np.ndarray, early_runs: np.ndarray,
                      multiclass: np.ndarray, weights: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    discriminative = _fisher(normal_runs, fault_runs)
    early = _fisher(normal_runs, early_runs)
    stability = _stability(normal_runs, fault_runs)
    composite = (float(weights["weight_discriminative"]) * _robust_normalize(discriminative)
                 + float(weights["weight_early"]) * _robust_normalize(early)
                 + float(weights["weight_run_stability"]) * _robust_normalize(stability)
                 + float(weights.get("weight_multiclass", 0.0)) * _robust_normalize(multiclass))
    return discriminative, early, stability, composite


def _multiclass_fisher(type_run_means: np.ndarray, type_labels: np.ndarray) -> np.ndarray:
    if len(type_run_means) != len(type_labels) or not len(type_run_means):
        raise ValueError("multiclass criticality requires aligned fault run means and type labels")
    grand = type_run_means.mean(0); between = np.zeros_like(grand); within = np.zeros_like(grand)
    for kind in np.unique(type_labels):
        group = type_run_means[type_labels == kind]
        between += len(group) * np.square(group.mean(0) - grand)
        within += np.square(group - group.mean(0)).sum(0)
    return between / (within + 1e-8)


def _balanced_multiclass_fisher(type_run_means: np.ndarray, type_labels: np.ndarray) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Return an equal-class one-vs-rest score and its per-class contributions.

    Both the rest centroid and its variance are averages over classes, so neither
    run count nor window count can make a frequent class dominate M.
    """
    if len(type_run_means) != len(type_labels) or not len(type_run_means):
        raise ValueError("balanced multiclass criticality requires aligned run means and type labels")
    kinds = np.unique(type_labels)
    if len(kinds) < 2:
        raise ValueError("balanced multiclass criticality requires at least two classes")
    centroids = {int(kind): type_run_means[type_labels == kind].mean(0) for kind in kinds}
    variances = {int(kind): type_run_means[type_labels == kind].var(0) for kind in kinds}
    contributions = {}
    for kind in map(int, kinds):
        rest = [int(other) for other in kinds if int(other) != kind]
        rest_centroid = np.mean([centroids[other] for other in rest], axis=0)
        rest_variance = np.mean([variances[other] for other in rest], axis=0)
        raw = np.square(centroids[kind] - rest_centroid) / (variances[kind] + rest_variance + 1e-8)
        contributions[kind] = _robust_normalize(raw)
    balanced = np.mean(np.stack(list(contributions.values())), axis=0)
    return balanced, contributions


def _stratified_bootstrap_type_means(type_run_means: np.ndarray, type_labels: np.ndarray,
                                     rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    sampled_means = []; sampled_labels = []
    for kind in np.unique(type_labels):
        group = type_run_means[type_labels == kind]
        sampled = group[rng.integers(0, len(group), len(group))]
        sampled_means.extend(sampled); sampled_labels.extend([kind] * len(sampled))
    return np.stack(sampled_means), np.asarray(sampled_labels)


def build_criticality(features: np.ndarray, bundle: dict[str, np.ndarray], stages: np.ndarray,
                      settings: dict[str, Any], raw_log_amplitude: np.ndarray | None = None) -> dict[str, Any]:
    """Fit all criticality statistics from one explicitly train-only bundle."""
    run_uids = np.asarray(bundle["run_uid"]); labels = np.asarray(bundle["labels"])
    kinds = np.asarray([fault_type(value) for value in run_uids])
    normal_runs, normal_ids = _run_means(features, run_uids, labels == 0)
    fault_runs, fault_ids = _run_means(features, run_uids, labels != 0)
    early_runs, early_ids = _run_means(features, run_uids, stages == "early")
    # M is multiclass over every train class/type, including normal type 0.
    # D remains the dedicated normal-vs-fault component.
    type_ids = np.unique(np.concatenate((normal_ids, fault_ids)))
    type_run_means = np.stack([features[run_uids == uid].mean(0) for uid in type_ids])
    type_labels = np.asarray([fault_type(str(uid)) for uid in type_ids])
    multiclass_mode = str(settings.get("multiclass_mode", "fisher"))
    multiclass_contributions: dict[int, np.ndarray] = {}
    multiclass_reliability = np.ones(features.shape[1:], dtype=np.float64)
    if multiclass_mode == "balanced_reliable":
        balanced_multiclass, multiclass_contributions = _balanced_multiclass_fisher(type_run_means, type_labels)
        reliability_rng = np.random.default_rng(int(settings["bootstrap_seed"]))
        selected = []
        for _ in range(int(settings["bootstrap_repeats"])):
            sampled_means, sampled_labels = _stratified_bootstrap_type_means(
                type_run_means, type_labels, reliability_rng)
            boot_balanced, _ = _balanced_multiclass_fisher(sampled_means, sampled_labels)
            selected.append(_top_mask(boot_balanced, float(settings["critical_ratio"])))
        multiclass_reliability = np.mean(np.stack(selected), axis=0)
        multiclass_fisher = balanced_multiclass * multiclass_reliability
    elif multiclass_mode == "fisher":
        multiclass_fisher = _multiclass_fisher(type_run_means, type_labels)
    else:
        raise ValueError(f"unknown multiclass criticality mode: {multiclass_mode}")
    discriminative, early, stability, composite = _scores_from_runs(
        normal_runs, fault_runs, early_runs, multiclass_fisher, settings)
    ratio = float(settings["critical_ratio"])
    energy_source = features if raw_log_amplitude is None else np.asarray(raw_log_amplitude)
    energy = np.mean(np.square(np.expm1(np.maximum(energy_source, 0))), axis=0)
    masks = {"energy": _top_mask(energy, ratio), "fisher": _top_mask(discriminative, ratio),
             "multiclass": _top_mask(multiclass_fisher, ratio), "composite": _top_mask(composite, ratio)}
    threshold = float(np.min(composite[masks["composite"]]))
    scale = max(float(np.quantile(composite, .75) - np.quantile(composite, .25)), 1e-8)
    soft_mask = 1 / (1 + np.exp(np.clip(-(composite - threshold) / scale, -30, 30)))

    rng = np.random.default_rng(int(settings["bootstrap_seed"])); overlaps = []
    for _ in range(int(settings["bootstrap_repeats"])):
        sampled_normal = normal_runs[rng.integers(0, len(normal_runs), len(normal_runs))]
        sampled_fault = fault_runs[rng.integers(0, len(fault_runs), len(fault_runs))]
        sampled_early = early_runs[rng.integers(0, len(early_runs), len(early_runs))]
        boot_multiclass = multiclass_fisher
        if float(settings.get("weight_multiclass", 0.0)) != 0:
            sampled_type_means, sampled_type_labels = _stratified_bootstrap_type_means(type_run_means, type_labels, rng)
            if multiclass_mode == "balanced_reliable":
                boot_balanced, _ = _balanced_multiclass_fisher(sampled_type_means, sampled_type_labels)
                boot_multiclass = boot_balanced * multiclass_reliability
            else:
                boot_multiclass = _multiclass_fisher(sampled_type_means, sampled_type_labels)
        _, _, _, boot_composite = _scores_from_runs(
            sampled_normal, sampled_fault, sampled_early, boot_multiclass, settings)
        overlaps.append(mask_jaccard(_top_mask(boot_composite, ratio), masks["composite"]))
    return {
        "fit_split": "train", "discriminative": discriminative, "early": early,
        "stability": stability, "composite": composite, "soft_mask": soft_mask.astype(np.float32),
        "energy": energy, "multiclass_fisher": multiclass_fisher, "masks": masks,
        "multiclass_mode": multiclass_mode,
        "multiclass_reliability": multiclass_reliability,
        "multiclass_class_contributions": multiclass_contributions,
        "component_weights": {"discriminative": float(settings["weight_discriminative"]),
                              "early": float(settings["weight_early"]),
                              "stability": float(settings["weight_run_stability"]),
                              "multiclass": float(settings.get("weight_multiclass", 0.0))},
        "multiclass_type_run_counts": {int(kind): int((type_labels == kind).sum()) for kind in np.unique(type_labels)},
        "bootstrap_overlap": np.asarray(overlaps),
        "run_counts": {"normal": len(normal_ids), "fault": len(fault_ids), "early_fault": len(early_ids)},
        "train_direction": fault_runs.mean(0) - normal_runs.mean(0),
    }
