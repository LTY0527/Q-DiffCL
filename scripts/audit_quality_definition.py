from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr

from diffusion.fixed_views import SPLITS, distribution, per_sample_masked_mae
from losses import (fit_robust_gain_calibration,
                    quality_weighted_supervised_contrastive_loss,
                    relative_gain, relative_quality,
                    relative_semantic_quality, semantic_score)
from losses.supcon import normalized_positive_weights
from scripts.run_diffusion_quality_retest import (epoch_orders, fit_train_only_quality,
                                                  load_fixed_views)
from scripts.run_rapid_idea_validation import _simple_interpolate
from trainers import build_model
from utils import environment_metadata, seed_everything, write_json


def fault_type(run_uid: str) -> int:
    match = re.search(r":fault_(\d+):", str(run_uid))
    return int(match.group(1)) if match else 0


def standardized_mean_difference(values: np.ndarray, labels: np.ndarray) -> float:
    first = np.asarray(values)[np.asarray(labels) == 0]; second = np.asarray(values)[np.asarray(labels) != 0]
    pooled = math.sqrt((float(first.var()) + float(second.var())) / 2)
    return 0.0 if pooled <= 1e-12 else float((second.mean() - first.mean()) / pooled)


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=float); right = np.asarray(right, dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3 or np.unique(left[valid]).size < 2 or np.unique(right[valid]).size < 2: return None
    value = float(spearmanr(left[valid], right[valid]).statistic)
    return value if np.isfinite(value) else None


def grouped(values: np.ndarray, groups: np.ndarray) -> dict[str, dict[str, float]]:
    result = {}
    for group in sorted(np.unique(groups), key=str):
        selected = np.asarray(values)[groups == group]
        result[str(group)] = {"count": int(len(selected)), "mean": float(selected.mean()), "std": float(selected.std()),
                              "p25": float(np.quantile(selected, .25)), "p50": float(np.quantile(selected, .5)),
                              "p75": float(np.quantile(selected, .75))}
    return result


@torch.no_grad()
def teacher_features(model: torch.nn.Module, clean: np.ndarray, restored: np.ndarray,
                     batch_size: int, device: str) -> dict[str, np.ndarray]:
    clean_probabilities = []; restored_probabilities = []
    for start in range(0, len(clean), batch_size):
        clean_output = model(torch.from_numpy(clean[start:start + batch_size]).float().to(device))["logits"]
        restored_output = model(torch.from_numpy(restored[start:start + batch_size]).float().to(device))["logits"]
        clean_probabilities.append(torch.softmax(clean_output, 1).cpu().numpy())
        restored_probabilities.append(torch.softmax(restored_output, 1).cpu().numpy())
    clean_p = np.concatenate(clean_probabilities); restored_p = np.concatenate(restored_probabilities)
    entropy = -(clean_p * np.log(np.maximum(clean_p, 1e-12))).sum(1) / math.log(clean_p.shape[1])
    prediction_distance = np.abs(clean_p - restored_p).sum(1) / 2
    return {"teacher_entropy": entropy, "semantic_score": 1 - prediction_distance,
            "teacher_consistency": (clean_p.argmax(1) == restored_p.argmax(1)).astype(np.float32),
            "clean_probability": clean_p, "restored_probability": restored_p}


def split_factors(bundle: dict[str, np.ndarray], split: str, teacher: torch.nn.Module,
                  batch_size: int, device: str) -> dict[str, np.ndarray]:
    simple = _simple_interpolate(bundle["degraded"], bundle["observation"])
    features = teacher_features(teacher, bundle["clean"], bundle["restored"], batch_size, device)
    starts = bundle["start_sample"].astype(float); types = np.asarray([fault_type(value) for value in bundle["run_uid"]])
    boundary = 161 if split == "test" else 21
    onset_position = np.where(types > 0, starts - boundary, np.nan)
    return {
        "e_diff": per_sample_masked_mae(bundle["clean"], bundle["restored"], bundle["observation"]),
        "e_simple": per_sample_masked_mae(bundle["clean"], simple, bundle["observation"]),
        "e_zero_fill": per_sample_masked_mae(bundle["clean"], bundle["degraded"], bundle["observation"]),
        "raw_variance": bundle["clean"].var(axis=(1, 2)),
        "first_difference_amplitude": np.abs(np.diff(bundle["clean"], axis=-1)).mean(axis=(1, 2)),
        "fault_type": types, "onset_relative_start": onset_position, **features,
    }


def _gradient_norm(model: torch.nn.Module) -> float:
    return float(math.sqrt(sum(float(parameter.grad.detach().square().sum()) for parameter in model.parameters() if parameter.grad is not None)))


def loss_scale_audit(bundle: dict[str, np.ndarray], q0: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    device = str(config["device"]); seed = int(config["random_seed"]); seed_everything(seed)
    model = build_model(config["model"], bundle["clean"].shape[1], 2).to(device)
    order = epoch_orders(len(bundle["labels"]), 1, seed + 10_000)[0]
    hard_losses = []; weighted_losses = []; hard_gradients = []; weighted_gradients = []
    raw_weight_means = []; normalized_weight_means = []; batch_q = []
    for start in range(0, len(order), int(config["batch_size"])):
        indices = order[start:start + int(config["batch_size"])]
        clean = torch.from_numpy(bundle["clean"][indices]).float().to(device)
        restored = torch.from_numpy(bundle["restored"][indices]).float().to(device)
        labels = torch.from_numpy(bundle["labels"][indices]).long().to(device)
        q = torch.from_numpy(q0[indices]).float().to(device)
        pair_labels = torch.cat([labels, labels]); weights = torch.cat([torch.ones_like(q), q])
        positive = pair_labels[:, None].eq(pair_labels[None, :]) & ~torch.eye(len(pair_labels), dtype=torch.bool, device=device)
        normalized, valid = normalized_positive_weights(positive, weights)
        raw = positive.float() * weights[None, :]
        raw_weight_means.append(float((raw.sum(1)[valid] / positive.sum(1)[valid]).mean()))
        normalized_weight_means.append(float((normalized.sum(1)[valid] / positive.sum(1)[valid]).mean()))
        batch_q.append(float(q.mean()))
        for weighted, losses, gradients in ((False, hard_losses, hard_gradients), (True, weighted_losses, weighted_gradients)):
            model.zero_grad(set_to_none=True)
            features = torch.cat([model(clean)["projection"], model(restored)["projection"]])
            loss = quality_weighted_supervised_contrastive_loss(
                features, pair_labels, weights if weighted else None, float(config["temperature"]),
            )
            loss.backward(); losses.append(float(loss.detach())); gradients.append(_gradient_norm(model))
    return {
        "hard_mean_batch_loss": float(np.mean(hard_losses)),
        "q0_mean_batch_loss": float(np.mean(weighted_losses)),
        "loss_ratio_q0_to_hard": float(np.mean(weighted_losses) / np.mean(hard_losses)),
        "q_mean": float(np.mean(batch_q)),
        "raw_effective_weight_mean": float(np.mean(raw_weight_means)),
        "normalized_effective_weight_mean": float(np.mean(normalized_weight_means)),
        "hard_mean_gradient_norm": float(np.mean(hard_gradients)),
        "q0_mean_gradient_norm": float(np.mean(weighted_gradients)),
        "gradient_norm_ratio_q0_to_hard": float(np.mean(weighted_gradients) / np.mean(hard_gradients)),
        "normalization": "per-anchor positive weights normalized to mean=1",
        "overall_loss_scale_preserved": bool(0.8 <= np.mean(weighted_losses) / np.mean(hard_losses) <= 1.2),
    }


def q0_audit(factors: dict[str, np.ndarray], bundle: dict[str, np.ndarray], q0: np.ndarray) -> dict[str, Any]:
    labels = bundle["labels"]; binary = (labels != 0).astype(int)
    correlations = {name: safe_spearman(factors["e_diff"], value) for name, value in factors.items()
                    if name not in {"fault_type"} and np.asarray(value).ndim == 1}
    run_means = np.asarray([factors["e_diff"][bundle["run_uid"] == uid].mean() for uid in np.unique(bundle["run_uid"])])
    return {
        "error_distribution": distribution(factors["e_diff"]),
        "error_by_normal_fault": grouped(factors["e_diff"], binary),
        "q_by_normal_fault": grouped(q0, binary),
        "error_smd_fault_minus_normal": standardized_mean_difference(factors["e_diff"], binary),
        "q_smd_fault_minus_normal": standardized_mean_difference(q0, binary),
        "fault_type_error": grouped(factors["e_diff"], factors["fault_type"]),
        "fault_type_q": grouped(q0, factors["fault_type"]),
        "run_mean_error_distribution": distribution(run_means),
        "run_mean_error_range": float(run_means.max() - run_means.min()),
        "error_factor_spearman": correlations,
        "q_label_spearman": safe_spearman(q0, binary),
        "q_teacher_consistency_spearman": safe_spearman(q0, factors["teacher_consistency"]),
        "q_semantic_score_spearman": safe_spearman(q0, factors["semantic_score"]),
    }


def build_quality_candidates(
    factors: dict[str, dict[str, np.ndarray]], q0: dict[str, np.ndarray], config: dict[str, Any],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    settings = config["quality"]; q1_settings = settings["q1"]
    gains = {split: relative_gain(factors[split]["e_simple"], factors[split]["e_diff"], float(q1_settings["eps"])) for split in SPLITS}
    calibration = fit_robust_gain_calibration(
        gains["train"], float(settings["q_min"]), float(settings["q_max"]), float(q1_settings["eps"]),
    )
    q1 = {split: relative_quality(gains[split], calibration) for split in SPLITS}
    floor = float(settings["q2"]["semantic_floor"])
    semantics = {split: semantic_score(factors[split]["clean_probability"], factors[split]["restored_probability"], floor) for split in SPLITS}
    q2 = {split: relative_semantic_quality(q1[split], semantics[split], float(settings["q_min"])) for split in SPLITS}
    for split in SPLITS:
        factors[split]["relative_gain"] = gains[split]
        factors[split]["semantic_score"] = semantics[split]
    return {"q0_abs": q0, "q1_relative": q1, "q2_relative_semantic": q2}, {
        "q1_train_only_calibration": calibration.to_dict(),
        "q2_semantic_formula": settings["q2"]["formula"], "q2_semantic_floor": floor,
        "fit_split": "train", "labels_used_as_quality_input": False,
    }


def _split_half(indices: np.ndarray, quality: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = indices[np.argsort(quality[indices], kind="stable")]
    middle = len(ordered) // 2
    return ordered[:middle], ordered[middle:]


def within_class_quality_groups(quality: np.ndarray, factors: dict[str, np.ndarray],
                                bundle: dict[str, np.ndarray]) -> dict[str, Any]:
    result = {}
    for class_name, selector in (("normal", bundle["labels"] == 0), ("fault", bundle["labels"] != 0)):
        indices = np.flatnonzero(selector); low, high = _split_half(indices, quality)
        groups = {}
        for name, chosen in (("low", low), ("high", high)):
            types, counts = np.unique(factors["fault_type"][chosen], return_counts=True)
            groups[name] = {
                "count": int(len(chosen)), "mean_q": float(quality[chosen].mean()),
                "mean_diffusion_error": float(factors["e_diff"][chosen].mean()),
                "mean_relative_gain": float(factors["relative_gain"][chosen].mean()),
                "mean_semantic_score": float(factors["semantic_score"][chosen].mean()),
                "teacher_consistency": float(factors["teacher_consistency"][chosen].mean()),
                "fault_type_counts": {str(int(key)): int(value) for key, value in zip(types, counts)},
            }
        result[class_name] = groups
    return result


def candidate_audit(name: str, quality: np.ndarray, factors: dict[str, np.ndarray],
                    bundle: dict[str, np.ndarray], q0_gap: float, config: dict[str, Any]) -> dict[str, Any]:
    binary = (bundle["labels"] != 0).astype(int); normal = quality[binary == 0]; fault = quality[binary == 1]
    gap = float(abs(normal.mean() - fault.mean()))
    fault_type_correlations = {}
    for kind in np.unique(factors["fault_type"]):
        if kind == 0: continue
        selector = factors["fault_type"] == kind
        fault_type_correlations[str(int(kind))] = safe_spearman(quality[selector], factors["e_diff"][selector])
    tolerance = float(config["audit_gate"]["semantic_correlation_tolerance"])
    minimum_reduction = float(config["audit_gate"]["minimum_bias_gap_reduction"])
    maximum_floor = float(config["audit_gate"]["maximum_q_min_fraction"])
    q0_semantic = config.get("_q0_semantic_reference")
    semantic_correlation = safe_spearman(quality, factors["semantic_score"])
    checks = {
        "mean_gap_reduced_vs_q0": bool(name != "q0_abs" and gap <= q0_gap * (1 - minimum_reduction)),
        "semantic_relation_not_worse": bool(q0_semantic is None or semantic_correlation is not None and semantic_correlation >= q0_semantic - tolerance),
        "fault_not_systematically_below_normal_by_over_0_05": bool(float(fault.mean()) >= float(normal.mean()) - .05),
        "not_concentrated_at_q_min": bool(float(np.mean(quality <= float(config["quality"]["q_min"]) + 1e-6)) <= maximum_floor),
    }
    return {
        "distribution": distribution(quality), "normal_mean": float(normal.mean()), "fault_mean": float(fault.mean()),
        "normal_fault_mean_gap": gap, "gap_reduction_vs_q0": None if name == "q0_abs" else float(1 - gap / max(q0_gap, 1e-12)),
        "q_label_spearman": safe_spearman(quality, binary),
        "q_semantic_score_spearman": semantic_correlation,
        "q_teacher_consistency_spearman": safe_spearman(quality, factors["teacher_consistency"]),
        "q_relative_gain_spearman": safe_spearman(quality, factors["relative_gain"]),
        "q_smd_fault_minus_normal": standardized_mean_difference(quality, binary),
        "q_min_fraction": float(np.mean(quality <= float(config["quality"]["q_min"]) + 1e-6)),
        "within_fault_type_q_error_spearman": fault_type_correlations,
        "within_class_groups": within_class_quality_groups(quality, factors, bundle),
        "gate_checks": checks, "passed_audit": bool(name != "q0_abs" and all(checks.values())),
    }


def audit(config: dict[str, Any]) -> dict[str, Any]:
    views, fixed_manifest = load_fixed_views(config)
    legacy_config = {"quality": {"formula": "exp(-masked_mae/scale)",
                                  "scale_estimator": config["quality"]["q0"]["scale_estimator"],
                                  "q_min": config["quality"]["q_min"]}}
    q0, q0_summary = fit_train_only_quality(views, legacy_config)
    device = str(config["device"]); teacher = build_model(config["model"], views["train"]["clean"].shape[1], 2).to(device)
    teacher.load_state_dict(torch.load(config["teacher_checkpoint"], map_location=device, weights_only=True)); teacher.eval()
    factors = {split: split_factors(views[split], split, teacher, int(config["batch_size"]), device) for split in SPLITS}
    candidates, calibration = build_quality_candidates(factors, q0, config)
    q0_gap = float(abs(q0["train"][views["train"]["labels"] == 0].mean() - q0["train"][views["train"]["labels"] != 0].mean()))
    config["_q0_semantic_reference"] = safe_spearman(q0["train"], factors["train"]["semantic_score"])
    candidate_audits = {
        name: {split: candidate_audit(name, values[split], factors[split], views[split], q0_gap, config) for split in SPLITS}
        for name, values in candidates.items()
    }
    passing = [name for name in ("q1_relative", "q2_relative_semantic") if candidate_audits[name]["train"]["passed_audit"]]
    status = "QUALITY_DEFINITION_AUDIT_PASSED" if passing else "QUALITY_DEFINITION_AUDIT_NO_GO"
    result = {
        "markers": config["markers"], "status": status, **environment_metadata(),
        "fixed_view_manifest": fixed_manifest, "q0": q0_summary,
        "loss_scale_audit": loss_scale_audit(views["train"], q0["train"], config),
        "split_audits": {split: q0_audit(factors[split], views[split], q0[split]) for split in SPLITS},
        "quality_calibration": calibration, "candidate_audits": candidate_audits,
        "passing_candidates": passing, "training_retest_allowed": bool(passing),
    }
    output = Path(config["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    write_json(output / "quality_definition_audit.json", result)
    rows = ["candidate,split,mean,std,normal_mean,fault_mean,mean_gap,q_label_spearman,q_semantic_spearman,passed_audit"]
    for name, split_values in candidate_audits.items():
        for split, value in split_values.items():
            rows.append(",".join(map(str, [name, split, value["distribution"]["mean"], value["distribution"]["std"],
                                              value["normal_mean"], value["fault_mean"], value["normal_fault_mean_gap"],
                                              value["q_label_spearman"], value["q_semantic_score_spearman"], value["passed_audit"]])))
    (output / "quality_definition_audit.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/quality_definition_audit.yaml")
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = audit(config)
    print(json.dumps({"status": result["status"], "passing_candidates": result["passing_candidates"],
                      "loss_scale": result["loss_scale_audit"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
