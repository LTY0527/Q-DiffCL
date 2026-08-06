from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from diffusion import FrequencyForwardDiffusion
from frequency import fault_stages, fit_cross_channel_spectral_structure
from scripts.audit_semantic_diffusion_augmentation import bases, traditional_augmentation
from scripts.diagnose_frequency_selective_far import (
    _fit_replay, correlation_drift, score_profile,
)
from scripts.run_diffusion_quality_retest import (
    _metrics, _probabilities, epoch_orders, load_fixed_views,
)
from scripts.run_stage_frequency_diffusion_mvp import (
    METHODS, _build_frequency_components, _configure, _fit_method, _runtime,
    augmentation_mechanism_metrics, detection_delays, early_fault_recall,
)
from trainers import build_model
from utils import environment_metadata, seed_everything, write_json


VARIANT_ORDER = ("R0", "R1", "R2", "R3")


def select_repair_variant(records: dict[str, dict[str, Any]], c1_metrics: dict[str, float],
                          selection: dict[str, float]) -> tuple[str, dict[str, Any]]:
    decisions = {}
    eligible = []
    for name in VARIANT_ORDER:
        metrics = records[name]["validation"]["metrics"]
        checks = {
            "far_within_c1_limit": metrics["far"] <= c1_metrics["far"] + float(selection["maximum_far_above_c1"]),
            "auprc_within_c1_limit": metrics["auprc"] >= c1_metrics["auprc"] - float(selection["maximum_auprc_drop"]),
            "recall_within_c1_limit": metrics["fault_recall"] >= c1_metrics["fault_recall"] - float(selection["maximum_recall_drop"]),
        }
        decisions[name] = {"checks": checks, "eligible": all(checks.values())}
        if decisions[name]["eligible"]:
            eligible.append(name)
    if not eligible:
        raise RuntimeError("no repair variant satisfies validation constraints")
    best_macro = max(records[name]["validation"]["metrics"]["macro_f1"] for name in eligible)
    contenders = [name for name in eligible if best_macro - records[name]["validation"]["metrics"]["macro_f1"]
                  <= float(selection["macro_f1_tolerance"])]
    minimum_far = min(records[name]["validation"]["metrics"]["far"] for name in contenders)
    contenders = [name for name in contenders if records[name]["validation"]["metrics"]["far"] <= minimum_far + 1e-12]
    maximum_early = max(records[name]["validation"]["early_fault"]["recall"] for name in contenders)
    contenders = [name for name in contenders if records[name]["validation"]["early_fault"]["recall"] >= maximum_early - 1e-12]
    selected = min(contenders, key=lambda name: (records[name]["validation_structure"]["normal"]["corr_drift"],
                                                  VARIANT_ORDER.index(name)))
    return selected, {"selection_split": "validation", "test_used": False, "decisions": decisions,
                      "macro_f1_tolerance": float(selection["macro_f1_tolerance"]),
                      "eligible": eligible, "selected": selected,
                      "reason": "Macro-F1 tolerance, then FAR, Early Recall, correlation drift"}


def evaluate_seed7_gate(c1: dict[str, Any], c2: dict[str, Any], c2s: dict[str, Any],
                        c2_audit: dict[str, Any], c2s_audit: dict[str, Any],
                        c2_drift: float, c2s_drift: float, gate: dict[str, float]) -> tuple[dict[str, bool], bool]:
    first, second = c1["metrics"], c2s["metrics"]
    delay_improved = (c2s["detection_delay"]["mean_delay_samples"] is not None
                      and c1["detection_delay"]["mean_delay_samples"] is not None
                      and c2s["detection_delay"]["mean_delay_samples"] < c1["detection_delay"]["mean_delay_samples"])
    checks = {
        "macro_f1_above_c1": second["macro_f1"] > first["macro_f1"],
        "far_within_c1_limit": second["far"] <= first["far"] + float(gate["maximum_far_above_c1"]),
        "auprc_within_c1_limit": second["auprc"] >= first["auprc"] - float(gate["maximum_auprc_drop"]),
        "recall_within_c1_limit": second["fault_recall"] >= first["fault_recall"] - float(gate["maximum_recall_drop"]),
        "early_or_delay_improved": c2s["early_fault"]["recall"] > c1["early_fault"]["recall"] or delay_improved,
        "correlation_drift_below_c2": c2s_drift < c2_drift,
        "critical_retention_not_below_c2": c2s_audit["critical_fisher_retention"] >= c2_audit["critical_fisher_retention"] - 1e-6,
        "augmentation_not_collapsed": c2s_audit["time_normalized_l1"] >= .8 * c2_audit["time_normalized_l1"],
        "finite": bool(c2s_audit["finite"]),
    }
    return checks, all(checks.values())


def _model_from_checkpoint(checkpoint: str, runtime: dict[str, Any], channels: int, device: str):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_model(runtime["model"], channels, 2).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def _test_record(model, threshold: float, views, base, stages, runtime, device: str) -> dict[str, Any]:
    probability, _ = _probabilities(model, base["test"], int(runtime["batch_size"]), device)
    scores = probability[:, 1]; prediction = scores >= threshold
    return {"metrics": _metrics(views["test"]["labels"], scores, threshold),
            "score_profile": score_profile(views["test"]["labels"], scores, threshold,
                                             float(runtime["diagnosis"]["threshold_band_width"])),
            "early_fault": early_fault_recall(prediction, stages["test"]),
            "detection_delay": detection_delays(views["test"], prediction, runtime)}


def _make_correlated(iid, base, config):
    structure = fit_cross_channel_spectral_structure(
        base["train"], float(config["correlated_noise"]["shrinkage_to_diagonal"]),
        float(config["correlated_noise"]["eigenvalue_floor"]),
        bool(config["correlated_noise"]["marginal_variance_matching"]), "train")
    return FrequencyForwardDiffusion(
        iid.statistics, iid.alpha_bars, iid.soft_mask.cpu().numpy(), iid.t_uniform, iid.t_critical,
        iid.preserve_phase, iid.preserve_dc, iid.device, structure,
        float(config["channel_budget"]["maximum_ratio_to_c1"])), structure


def run(config: dict[str, Any]) -> dict[str, Any]:
    result_path = Path(config["output_dir"]) / "repair" / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    diagnosis = json.loads(Path(config["diagnosis_result"]).read_text(encoding="utf-8"))
    if not diagnosis["repair_allowed"]:
        return {"status": "FREQUENCY_SELECTIVE_FAR_CAUSE_UNRESOLVED", "repair_skipped": True}
    base_config = yaml.safe_load(Path(config["base_config"]).read_text(encoding="utf-8")); _configure(base_config)
    views, _ = load_fixed_views(base_config); base = bases(views)
    stages = {split: fault_stages(views[split], base_config) for split in views}
    critical, iid = _build_frequency_components(base_config, views, base, stages, str(config["device"]))
    correlated, structure = _make_correlated(iid, base, config)
    seed = int(config["diagnosis"]["seed"]); runtime = _runtime(base_config, seed); runtime["diagnosis"] = config["diagnosis"]
    pretrain_orders = epoch_orders(len(base["train"]), int(runtime["epochs"]), seed + 10_000)
    probe_orders = epoch_orders(len(base["train"]), int(runtime["probe_epochs"]), seed + 20_000)
    seed_everything(seed); template = build_model(runtime["model"], base["train"].shape[1], 2)
    initial_state = copy.deepcopy(template.state_dict())
    variants = {}; augmented_validation = {}; diagnostics = {}
    for name in VARIANT_ORDER:
        specification = config["repair_variants"][name]
        augmenter = iid if specification["noise_structure"] == "iid" else correlated
        augmented = {}; diagnostics[name] = {}
        for split, offset in (("train", 0), ("validation", 100)):
            sampling_seed = seed + int(base_config["spectral_diffusion"]["sampling_seed_offset"]) + offset
            augmented[split], diagnostics[name][split] = augmenter.augment(
                base[split], "selective", sampling_seed, int(specification["t_noncritical"]),
                int(runtime["batch_size"]), str(specification["noise_structure"]),
                specification["noise_structure"] == "correlated")
        augmented_validation[name] = augmented["validation"]
        if name == "R0":
            variants[name] = copy.deepcopy(diagnosis["replays"]["8"])
        elif name == "R1":
            variants[name] = copy.deepcopy(diagnosis["replays"]["5"])
        else:
            checkpoint = Path(config["output_dir"]) / "repair" / f"{name}_seed7_model.pt"
            variants[name] = _fit_replay(name, augmented, views, base, stages, initial_state, pretrain_orders,
                                          probe_orders, runtime, str(config["device"]), checkpoint, False)
        variants[name]["specification"] = specification
        variants[name]["validation_structure"] = correlation_drift(
            base["validation"], augmented["validation"], views["validation"]["labels"],
            float(config["diagnosis"]["high_correlation_quantile"]))
        variants[name]["validation_audit"] = augmentation_mechanism_metrics(
            base["validation"], augmented["validation"], views["validation"]["labels"], stages["validation"],
            critical["masks"]["composite"], diagnostics[name]["validation"])
    selected, selection_record = select_repair_variant(
        variants, diagnosis["replays"]["C1"]["validation"]["metrics"], config["selection"])
    selected_spec = config["repair_variants"][selected]
    selected_augmenter = iid if selected_spec["noise_structure"] == "iid" else correlated
    sampling_seed = seed + int(base_config["spectral_diffusion"]["sampling_seed_offset"]) + 200
    selected_test_augmented, selected_test_diag = selected_augmenter.augment(
        base["test"], "selective", sampling_seed, int(selected_spec["t_noncritical"]), int(runtime["batch_size"]),
        str(selected_spec["noise_structure"]), selected_spec["noise_structure"] == "correlated")
    model = _model_from_checkpoint(variants[selected]["checkpoint"], runtime, base["train"].shape[1], str(config["device"]))
    c2s = _test_record(model, float(variants[selected]["validation_threshold"]), views, base, stages, runtime, str(config["device"]))
    c2s_audit = augmentation_mechanism_metrics(
        base["test"], selected_test_augmented, views["test"]["labels"], stages["test"],
        critical["masks"]["composite"], selected_test_diag)
    c2s_drift = correlation_drift(base["validation"], augmented_validation[selected], views["validation"]["labels"],
                                  float(config["diagnosis"]["high_correlation_quantile"]))["normal"]["corr_drift"]
    c2_drift = diagnosis["validation_diagnosis"]["candidates"]["8"]["time_structure"]["normal"]["corr_drift"]
    c1 = diagnosis["replays"]["C1"]["test"]; c2 = diagnosis["replays"]["8"]["test"]
    old = json.loads(Path(config["old_mvp_result"]).read_text(encoding="utf-8"))
    c2_old_audit = old["augmentation_audits"][METHODS[2]]["test"]
    checks, passed = evaluate_seed7_gate(c1, c2, c2s, c2_old_audit, c2s_audit,
                                         c2_drift, c2s_drift, config["seed7_gate"])
    result = {"markers": config["markers"],
              "status": "FREQUENCY_SELECTIVE_STRUCTURE_FIX_SEED7_GO" if passed else "FREQUENCY_SELECTIVE_STRUCTURE_FIX_SEED7_NO_GO",
              "cause_category": diagnosis["cause_category"], "selection": selection_record,
              "selected_variant": selected, "selected_specification": selected_spec,
              "validation_variants": variants, "seed7": {"C0": old["results"][METHODS[0]], "C1": c1, "C2": c2, "C2-S": c2s},
              "seed7_gate_checks": checks, "seed7_passed": passed, "three_seed_run": False,
              "c2s_test_audit": c2s_audit, "c2_validation_normal_corr_drift": c2_drift,
              "c2_test_audit": c2_old_audit,
              "c2s_validation_normal_corr_drift": c2s_drift,
              "structure_fit": {"fit_split": structure.fit_split, "shape": list(structure.covariance.shape),
                                "shrinkage": structure.shrinkage_to_diagonal,
                                "eigenvalue_floor": structure.eigenvalue_floor},
              "test_used_for_selection": False, **environment_metadata()}
    result_path.parent.mkdir(parents=True, exist_ok=True); write_json(result_path, result)
    from scripts.summarize_frequency_selective_far_fix import summarize
    summarize(result, config["fix_report"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/frequency_selective_far_fix.yaml")
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = run(config); print(json.dumps({"status": result["status"], "selected": result.get("selected_variant"),
                                            "three_seed_run": result.get("three_seed_run")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
