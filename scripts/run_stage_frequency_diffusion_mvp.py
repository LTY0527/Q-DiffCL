from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from diffusion import DiffusionSchedule, FrequencyForwardDiffusion, fit_spectral_statistics
from diffusion.fixed_views import sha256_strings
from frequency import build_criticality, fault_stages, fit_frequency_scaler, log_amplitude_phase
from frequency.criticality import fault_type
from metrics import representation_diagnostics
from scripts.audit_semantic_diffusion_augmentation import bases, traditional_augmentation
from scripts.run_diffusion_quality_retest import (
    _fit_probe, _fit_supcon, _metrics, _probabilities, _state_hash,
    best_probe_record, epoch_orders, load_fixed_views,
)
from trainers import build_model
from utils import environment_metadata, seed_everything, write_json


METHODS = ("C0 传统增强", "C1 统一频谱扩散", "C2 频率选择性扩散")


def _configure(config: dict[str, Any]) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(config["cublas_workspace_config"])
    enabled = bool(config["deterministic_algorithms"])
    torch.use_deterministic_algorithms(enabled)
    torch.backends.cudnn.deterministic = enabled; torch.backends.cudnn.benchmark = False


def _runtime(config: dict[str, Any], seed: int) -> dict[str, Any]:
    value = dict(config["training"]); value["random_seed"] = int(seed)
    value["protocol"] = config["protocol"]; value["detection"] = config["detection"]
    return value


def _fingerprint(config: dict[str, Any], mask: np.ndarray, selected_t: int) -> str:
    frozen = {"markers": config["markers"], "fixed_views": config["fixed_views"],
              "protocol": config["protocol"], "stage": config["stage"], "fft": config["fft"],
              "criticality": config["criticality"], "spectral_diffusion": config["spectral_diffusion"],
              "training": config["training"], "detection": config["detection"],
              "selected_t_noncritical": int(selected_t),
              "soft_mask_sha256": hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()}
    return hashlib.sha256(json.dumps(frozen, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def _fisher(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.square(first.mean(0) - second.mean(0)) / (first.var(0) + second.var(0) + 1e-8)


def augmentation_mechanism_metrics(base: np.ndarray, augmented: np.ndarray, labels: np.ndarray,
                                   stages: np.ndarray, critical_mask: np.ndarray,
                                   diagnostics: dict[str, Any]) -> dict[str, Any]:
    base_log, base_phase = log_amplitude_phase(base); changed_log, changed_phase = log_amplitude_phase(augmented)
    difference = np.abs(changed_log - base_log); critical, noncritical = critical_mask, ~critical_mask
    normal, fault, early = labels == 0, labels != 0, stages == "early"
    base_fisher = _fisher(base_log[normal], base_log[fault]); changed_fisher = _fisher(changed_log[normal], changed_log[fault])
    base_early = _fisher(base_log[normal], base_log[early]); changed_early = _fisher(changed_log[normal], changed_log[early])
    angular = np.abs(np.arctan2(np.sin(changed_phase - base_phase), np.cos(changed_phase - base_phase)))
    valid_phase = np.expm1(changed_log) > 1e-6
    scale = np.maximum(base_log.std(axis=(1, 2)), 1e-6)
    frequency_l1 = np.abs(changed_log - base_log).mean(axis=(1, 2)) / scale
    spike_limit = max(10.0, float(np.abs(base).mean() + 10 * base.std()))
    result = {
        **diagnostics,
        "actual_total_frequency_l1": float(frequency_l1.mean()),
        "critical_frequency_l1": float(difference[:, critical].mean()),
        "noncritical_frequency_l1": float(difference[:, noncritical].mean()),
        "critical_noncritical_perturbation_ratio": float(difference[:, critical].mean()
                                                         / max(float(difference[:, noncritical].mean()), 1e-12)),
        "critical_fisher_retention": float(changed_fisher[critical].mean()
                                             / max(float(base_fisher[critical].mean()), 1e-12)),
        "early_frequency_retention": float(changed_early[critical].mean()
                                             / max(float(base_early[critical].mean()), 1e-12)),
        "critical_band_energy_retention": float(np.expm1(changed_log)[:, critical].mean()
                                                  / max(float(np.expm1(base_log)[:, critical].mean()), 1e-12)),
        "measured_phase_error": float(angular[valid_phase].mean()) if valid_phase.any() else 0.0,
        "channel_variance_ratio_mean": float(np.mean(augmented.std((0, 2)) / np.maximum(base.std((0, 2)), 1e-6))),
        "abnormal_spike_fraction": float(np.mean(np.abs(augmented) > spike_limit)),
        "finite": bool(np.isfinite(augmented).all()),
    }
    return result


def select_t_noncritical(records: list[dict[str, Any]]) -> tuple[int, list[dict[str, Any]]]:
    if not records or any(record.get("selection_split") != "validation" for record in records):
        raise ValueError("t_noncritical selection must use validation records only")
    finite = [record for record in records if record["metrics"]["finite"]]
    if not finite: raise RuntimeError("no finite t_noncritical candidate")
    for record in finite:
        metrics = record["metrics"]
        record["selection_score"] = (metrics["critical_fisher_retention"]
                                     + metrics["early_frequency_retention"]
                                     - abs(metrics["expected_total_noise_budget"]
                                           - record["uniform_budget"]) * 1000)
    selected = max(finite, key=lambda row: (row["selection_score"], -int(row["t_noncritical"])))
    return int(selected["t_noncritical"]), records


def early_fault_recall(prediction: np.ndarray, stages: np.ndarray) -> dict[str, Any]:
    selector = stages == "early"
    return {"count": int(selector.sum()), "recall": float(np.mean(prediction[selector] == 1)) if selector.any() else None}


def detection_delays(bundle: dict[str, np.ndarray], prediction: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    sustained = int(config["detection"]["sustained_alarm_windows"]); per_run = {}
    fault_runs = sorted({str(uid) for uid in bundle["run_uid"] if fault_type(str(uid)) != 0})
    detected_delays = []
    for uid in fault_runs:
        selector = np.asarray(bundle["run_uid"]) == uid
        ends = np.asarray(bundle["end_sample"])[selector]; predictions = np.asarray(prediction)[selector]
        onset = int(config["protocol"]["fault_onset"][uid.split(":", 1)[0]])
        order = np.argsort(ends); ends, predictions = ends[order], predictions[order]
        post = ends >= onset; ends, predictions = ends[post], predictions[post]
        delay = None
        for index in range(0, len(predictions) - sustained + 1):
            if np.all(predictions[index:index + sustained] == 1):
                delay = int(ends[index + sustained - 1] - onset); break
        per_run[uid] = {"detected": delay is not None, "delay_samples": delay}
        if delay is not None: detected_delays.append(delay)
    return {"sustained_alarm_windows": sustained, "run_count": len(fault_runs),
            "detected_runs": len(detected_delays), "missed_runs": len(fault_runs) - len(detected_delays),
            "detection_rate": float(len(detected_delays) / len(fault_runs)) if fault_runs else None,
            "mean_delay_samples": float(np.mean(detected_delays)) if detected_delays else None,
            "median_delay_samples": float(np.median(detected_delays)) if detected_delays else None,
            "per_run": per_run}


def _fit_method(name: str, augmented: dict[str, np.ndarray], base: dict[str, np.ndarray],
                views: dict[str, dict[str, np.ndarray]], stages: dict[str, np.ndarray],
                initial_state: dict[str, torch.Tensor], pretrain_orders: list[np.ndarray],
                probe_orders: list[np.ndarray], runtime: dict[str, Any], device: str) -> dict[str, Any]:
    seed_everything(int(runtime["random_seed"])); started = time.perf_counter()
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    model = build_model(runtime["model"], base["train"].shape[1], 2).to(device); model.load_state_dict(initial_state)
    train = {"clean": base["train"], "restored": augmented["train"], "labels": views["train"]["labels"]}
    validation = {"clean": base["validation"], "restored": augmented["validation"],
                  "labels": views["validation"]["labels"]}
    ones_train = np.ones(len(train["labels"]), np.float32); ones_validation = np.ones(len(validation["labels"]), np.float32)
    pretrain = _fit_supcon(model, train, validation, ones_train, ones_validation, pretrain_orders, runtime, device)
    probe_train = {"clean": base["train"], "labels": views["train"]["labels"]}
    probe_validation = {"restored": base["validation"], "labels": views["validation"]["labels"]}
    seed_everything(int(runtime["random_seed"]) + 1)
    probe = _fit_probe(model, probe_train, probe_validation, probe_orders, runtime, device)
    best_probe = best_probe_record(probe); threshold = float(best_probe["validation_threshold"])
    probability, embedding = _probabilities(model, base["test"], int(runtime["batch_size"]), device)
    _, augmented_embedding = _probabilities(model, augmented["test"], int(runtime["batch_size"]), device)
    prediction = (probability[:, 1] >= threshold).astype(np.int64)
    diagnostics = representation_diagnostics(embedding, augmented_embedding, views["test"]["labels"])
    return {"method": name, "metrics": _metrics(views["test"]["labels"], probability[:, 1], threshold),
            "early_fault": early_fault_recall(prediction, stages["test"]),
            "detection_delay": detection_delays(views["test"], prediction, runtime),
            "representation": {key: diagnostics[key] for key in ("fisher_ratio", "class_center_shift", "effective_rank")},
            "validation_threshold": threshold, "best_pretrain_epoch": int(min(pretrain, key=lambda row: row["validation_supcon_loss"])["epoch"]),
            "best_probe_epoch": int(best_probe["epoch"]), "pretrain_history": pretrain, "probe_history": probe,
            "initialization_sha256": _state_hash(initial_state), "sample_total_weight": 1.0,
            "training_seconds": time.perf_counter() - started,
            "peak_gpu_mib": torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0}


def _build_frequency_components(config: dict[str, Any], views: dict[str, dict[str, np.ndarray]],
                                base: dict[str, np.ndarray], stages: dict[str, np.ndarray], device: str):
    train_log = log_amplitude_phase(base["train"])[0]; scaler = fit_frequency_scaler(train_log, "train")
    critical = build_criticality(scaler.transform(train_log), views["train"], stages["train"],
                                 config["criticality"], train_log)
    statistics = fit_spectral_statistics(base["train"], float(config["spectral_diffusion"]["clip_quantile"]), "train")
    schedule = DiffusionSchedule.cosine(int(config["spectral_diffusion"]["diffusion_steps"]), device)
    augmenter = FrequencyForwardDiffusion(
        statistics, schedule.alpha_bars, critical["soft_mask"], int(config["spectral_diffusion"]["t_uniform"]),
        int(config["spectral_diffusion"]["t_critical"]), bool(config["spectral_diffusion"]["preserve_phase"]),
        bool(config["spectral_diffusion"]["preserve_dc"]), device)
    return critical, augmenter


def _select_candidate(config: dict[str, Any], base: dict[str, np.ndarray], views: dict[str, dict[str, np.ndarray]],
                      stages: dict[str, np.ndarray], critical: dict[str, Any], augmenter: FrequencyForwardDiffusion,
                      seed: int) -> tuple[int, list[dict[str, Any]]]:
    sampling_seed = seed + int(config["spectral_diffusion"]["sampling_seed_offset"]) + 200
    _, uniform_diag = augmenter.augment(base["validation"], "uniform", sampling_seed,
                                        batch_size=int(config["training"]["batch_size"]))
    records = []
    for timestep in config["spectral_diffusion"]["t_noncritical_candidates"]:
        augmented, diagnostic = augmenter.augment(base["validation"], "selective", sampling_seed, int(timestep),
                                                  int(config["training"]["batch_size"]))
        metrics = augmentation_mechanism_metrics(base["validation"], augmented, views["validation"]["labels"],
                                                 stages["validation"], critical["masks"]["composite"], diagnostic)
        records.append({"selection_split": "validation", "t_noncritical": int(timestep),
                        "uniform_budget": uniform_diag["expected_total_noise_budget"], "metrics": metrics})
    return select_t_noncritical(records)


def _augment_all(config: dict[str, Any], base: dict[str, np.ndarray], views: dict[str, dict[str, np.ndarray]],
                 stages: dict[str, np.ndarray], critical: dict[str, Any], augmenter: FrequencyForwardDiffusion,
                 seed: int, selected_t: int) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    augmented = {name: {} for name in METHODS}; audits = {name: {} for name in METHODS}
    split_offset = {"train": 0, "validation": 100, "test": 200}
    for split in ("train", "validation", "test"):
        augmented[METHODS[0]][split] = traditional_augmentation(
            base[split], views[split]["window_id"], config["training"]["traditional_augmentation"], seed)
        sampling_seed = seed + int(config["spectral_diffusion"]["sampling_seed_offset"]) + split_offset[split]
        augmented[METHODS[1]][split], c1_diag = augmenter.augment(
            base[split], "uniform", sampling_seed, batch_size=int(config["training"]["batch_size"]))
        augmented[METHODS[2]][split], c2_diag = augmenter.augment(
            base[split], "selective", sampling_seed, selected_t, int(config["training"]["batch_size"]))
        for method, diag in ((METHODS[1], c1_diag), (METHODS[2], c2_diag)):
            audits[method][split] = augmentation_mechanism_metrics(
                base[split], augmented[method][split], views[split]["labels"], stages[split],
                critical["masks"]["composite"], diag)
        audits[METHODS[0]][split] = {"finite": bool(np.isfinite(augmented[METHODS[0]][split]).all()),
                                     "time_normalized_l1": float(np.mean(np.abs(augmented[METHODS[0]][split] - base[split])
                                                                         / np.maximum(base[split].std(), 1e-6)))}
    return augmented, audits


def single_seed_gate(results: dict[str, Any], audits: dict[str, Any], config: dict[str, Any]) -> tuple[dict[str, bool], bool]:
    c1, c2 = results[METHODS[1]], results[METHODS[2]]; first, second = c1["metrics"], c2["metrics"]
    c1a, c2a = audits[METHODS[1]]["test"], audits[METHODS[2]]["test"]
    delay_better = (c2["detection_delay"]["mean_delay_samples"] is not None
                    and c1["detection_delay"]["mean_delay_samples"] is not None
                    and c2["detection_delay"]["mean_delay_samples"] < c1["detection_delay"]["mean_delay_samples"])
    checks = {
        "macro_f1_improved": second["macro_f1"] > first["macro_f1"],
        "far_reduced": second["far"] < first["far"],
        "auprc_maintained": second["auprc"] >= first["auprc"] - float(config["mvp_gate"]["maximum_auprc_drop"]),
        "recall_maintained": second["fault_recall"] >= first["fault_recall"] - float(config["mvp_gate"]["maximum_recall_drop"]),
        "early_or_delay_improved": (c2["early_fault"]["recall"] > c1["early_fault"]["recall"] or delay_better),
        "critical_band_better_preserved": (c2a["critical_frequency_l1"] <= .95 * c1a["critical_frequency_l1"]
                                             and c2a["critical_fisher_retention"] >= c1a["critical_fisher_retention"] - .01),
        "total_noise_budget_fair": abs(c2a["expected_total_noise_budget"] - c1a["expected_total_noise_budget"]) <= 1e-6,
        "no_numerical_abnormality": bool(c1a["finite"] and c2a["finite"]),
    }
    strong = (second["macro_f1"] >= first["macro_f1"] + float(config["mvp_gate"]["strong_macro_f1_gain"])
              or second["far"] <= first["far"] - float(config["mvp_gate"]["strong_far_reduction"])
              or c2["early_fault"]["recall"] >= c1["early_fault"]["recall"] + float(config["mvp_gate"]["strong_early_recall_gain"]))
    checks["strong_engineering_signal"] = strong
    passed = sum(checks[name] for name in list(checks)[:8]) >= 5 and strong and checks["total_noise_budget_fair"] and checks["no_numerical_abnormality"]
    return checks, bool(passed)


def run_seed(config: dict[str, Any], seed: int, selected_t: int, candidate_records: list[dict[str, Any]],
             views, base, stages, critical, augmenter) -> dict[str, Any]:
    output = Path(config["output_dir"]) / f"seed_{seed}"; result_path = output / "result.json"
    if result_path.exists(): return json.loads(result_path.read_text(encoding="utf-8"))
    runtime = _runtime(config, seed); augmented, audits = _augment_all(config, base, views, stages, critical, augmenter, seed, selected_t)
    pretrain_orders = epoch_orders(len(base["train"]), int(runtime["epochs"]), seed + 10_000)
    probe_orders = epoch_orders(len(base["train"]), int(runtime["probe_epochs"]), seed + 20_000)
    seed_everything(seed); template = build_model(runtime["model"], base["train"].shape[1], 2)
    initial_state = copy.deepcopy(template.state_dict())
    results = {method: _fit_method(method, augmented[method], base, views, stages, initial_state,
                                   pretrain_orders, probe_orders, runtime, str(config["device"])) for method in METHODS}
    checks, passed = single_seed_gate(results, audits, config)
    initialization_hashes = {value["initialization_sha256"] for value in results.values()}
    fairness = {"same_split_mask_base": True, "same_encoder_projection_initialization": len(initialization_hashes) == 1,
                "same_pretrain_batch_order": True, "same_probe_batch_order": True,
                "same_optimizer_lr_epochs_temperature_probe_threshold_protocol": True,
                "sample_total_weight_one": all(value["sample_total_weight"] == 1 for value in results.values()),
                "only_variable": "positive augmentation view source",
                "initialization_sha256": next(iter(initialization_hashes)),
                "pretrain_order_sha256": sha256_strings([','.join(map(str, order)) for order in pretrain_orders]),
                "probe_order_sha256": sha256_strings([','.join(map(str, order)) for order in probe_orders])}
    if not all(value for key, value in fairness.items() if key not in {"only_variable", "initialization_sha256", "pretrain_order_sha256", "probe_order_sha256"}):
        raise RuntimeError("C0/C1/C2 fairness invariant failed")
    result = {"markers": config["markers"], "status": "FREQUENCY_SELECTIVE_DIFFUSION_SINGLE_SEED_GO" if passed else "FREQUENCY_SELECTIVE_DIFFUSION_MVP_NO_GO",
              "seed": int(seed), "selected_t_noncritical": int(selected_t), "candidate_selection": candidate_records,
              "selection_split": "validation", "test_used_for_selection": False,
              "configuration_fingerprint": _fingerprint(config, critical["soft_mask"], selected_t),
              "results": results, "augmentation_audits": audits, "gate_checks": checks,
              "fairness": fairness, "three_seed_allowed": passed, **environment_metadata()}
    write_json(result_path, result); return result


def run(config: dict[str, Any]) -> dict[str, Any]:
    _configure(config); audit = json.loads(Path(config["frequency_audit_result"]).read_text(encoding="utf-8"))
    if audit["status"] != "FREQUENCY_CRITICALITY_AUDIT_GO":
        return {"status": "FREQUENCY_CRITICALITY_AUDIT_NO_GO", "training_skipped": True}
    views, _ = load_fixed_views(config); base = bases(views)
    stages = {split: fault_stages(views[split], config) for split in views}
    critical, augmenter = _build_frequency_components(config, views, base, stages, str(config["device"]))
    seed7 = int(config["training"]["selection_seed"])
    selected_t, candidate_records = _select_candidate(config, base, views, stages, critical, augmenter, seed7)
    first = run_seed(config, seed7, selected_t, candidate_records, views, base, stages, critical, augmenter)
    results = {str(seed7): first}
    if first["three_seed_allowed"]:
        for seed in map(int, config["training"]["seeds"]):
            if seed != seed7:
                results[str(seed)] = run_seed(config, seed, selected_t, candidate_records, views, base, stages, critical, augmenter)
    from scripts.summarize_stage_frequency_diffusion_mvp import summarize
    return summarize(config, results, selected_t)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/stage_frequency_diffusion_mvp.yaml")
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = run(config); print(json.dumps({"status": result["status"], "selected_t_noncritical": result.get("selected_t_noncritical"),
                                           "seeds": list(result.get("seed_results", {}))}, ensure_ascii=False))


if __name__ == "__main__": main()
