from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from metrics import representation_diagnostics
from scripts.audit_semantic_diffusion_augmentation import bases, traditional_augmentation
from scripts.diagnose_frequency_selective_far import correlation_drift, score_profile
from scripts.run_diffusion_quality_retest import (
    _fit_ce_rep, _fit_probe, _fit_supcon, _metrics, _probabilities, _state_hash,
    best_probe_record, epoch_orders, load_fixed_views,
)
from scripts.run_stage_frequency_diffusion_mvp import (
    _build_frequency_components, _configure, _runtime, augmentation_mechanism_metrics,
    detection_delays, early_fault_recall,
)
from frequency import fault_stages
from trainers import build_model
from utils import environment_metadata, seed_everything, write_json


METHODS = ("C0", "C1", "R1")


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def sha256_strings(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def validate_frozen_config(config: dict[str, Any], base: dict[str, Any], far_fix: dict[str, Any]) -> None:
    frozen = config["frozen"]
    if list(map(int, config["seeds"])) != [7, 42, 2026] or list(config["methods"]) != list(METHODS):
        raise ValueError("three-seed protocol must freeze seeds [7,42,2026] and C0/C1/R1")
    for key in ("protocol", "stage", "criticality"):
        for field, value in frozen[key].items():
            if base[key][field] != value:
                raise ValueError(f"frozen {key}.{field} differs from base protocol")
    spectral = frozen["spectral_diffusion"]
    expected = {"diffusion_steps": 50, "t_uniform": 3, "t_critical": 1, "t_noncritical": 5,
                "preserve_phase": True, "preserve_dc": True,
                "noise_budget_matching": "alpha_bar_mean", "noise_structure": "iid"}
    if spectral != expected:
        raise ValueError("R1 spectral configuration is not frozen")
    if far_fix["repair_variants"]["R1"] != {"noise_structure": "iid", "t_noncritical": 5}:
        raise ValueError("R1 no longer matches the selected FAR-fix variant")
    if frozen["detection"] != base["detection"]:
        raise ValueError("detection protocol differs from base")


def _stage_metrics(stages: np.ndarray, prediction: np.ndarray) -> dict[str, dict[str, Any]]:
    result = {}
    for stage in ("early", "middle", "stable"):
        selected = stages == stage
        result[stage] = {"count": int(selected.sum()),
                         "recall": float(prediction[selected].mean()) if selected.any() else None}
    return result


def _fit_method(name: str, augmented: dict[str, np.ndarray], audit: dict[str, Any], views, base, stages,
                initial_state, pretrain_orders, probe_orders, runtime, device: str,
                checkpoint: Path, metadata: dict[str, Any],
                evaluation_splits: tuple[str, ...] = ("validation", "test"),
                representation_objective: str = "hard_supcon") -> dict[str, Any]:
    if not evaluation_splits or any(split not in ("validation", "test") for split in evaluation_splits):
        raise ValueError("evaluation_splits must contain validation and/or test")
    metrics_path = checkpoint.parent / "metrics.json"
    if checkpoint.exists() and metrics_path.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        record = json.loads(metrics_path.read_text(encoding="utf-8"))
        if payload.get("metadata") != metadata or record.get("metadata") != metadata:
            raise RuntimeError(f"resume metadata mismatch: {checkpoint}")
        return record
    if checkpoint.exists() or metrics_path.exists():
        raise RuntimeError(f"incomplete method output cannot be safely resumed: {checkpoint.parent}")
    seed_everything(int(runtime["random_seed"])); started = time.perf_counter()
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    model = build_model(runtime["model"], base["train"].shape[1], 2).to(device)
    model.load_state_dict(initial_state)
    train = {"clean": base["train"], "restored": augmented["train"], "labels": views["train"]["labels"]}
    validation = {"clean": base["validation"], "restored": augmented["validation"],
                  "labels": views["validation"]["labels"]}
    if representation_objective == "hard_supcon":
        pretrain = _fit_supcon(model, train, validation, np.ones(len(train["labels"]), np.float32),
                               np.ones(len(validation["labels"]), np.float32), pretrain_orders, runtime, device)
    elif representation_objective == "ce_rep":
        pretrain = _fit_ce_rep(model, train, validation, pretrain_orders, runtime, device)
    else:
        raise ValueError(f"unknown representation objective: {representation_objective}")
    seed_everything(int(runtime["random_seed"]) + 1)
    probe = _fit_probe(model, {"clean": base["train"], "labels": views["train"]["labels"]},
                       {"restored": base["validation"], "labels": views["validation"]["labels"]},
                       probe_orders, runtime, device)
    best = best_probe_record(probe); threshold = float(best["validation_threshold"])
    splits = {}
    for split in evaluation_splits:
        probability, embedding = _probabilities(model, base[split], int(runtime["batch_size"]), device)
        _, augmented_embedding = _probabilities(model, augmented[split], int(runtime["batch_size"]), device)
        scores = probability[:, 1]; prediction = scores >= threshold
        splits[split] = {
            "metrics": _metrics(views[split]["labels"], scores, threshold),
            "score_profile": score_profile(views[split]["labels"], scores, threshold,
                                             float(runtime["diagnosis"]["threshold_band_width"])),
            "stages": _stage_metrics(stages[split], prediction),
            "early_fault": early_fault_recall(prediction, stages[split]),
            "detection_delay": detection_delays(views[split], prediction, runtime),
            "representation": representation_diagnostics(embedding, augmented_embedding, views[split]["labels"]),
        }
    validation_loss = "validation_supcon_loss" if representation_objective == "hard_supcon" else "validation_ce_loss"
    record = {"method": name, "seed": int(runtime["random_seed"]), "validation_threshold": threshold,
              "representation_objective": representation_objective,
              "best_pretrain_epoch": int(min(pretrain, key=lambda row: row[validation_loss])["epoch"]),
              "best_probe_epoch": int(best["epoch"]), "pretrain_history": pretrain, "probe_history": probe,
              "initialization_sha256": _state_hash(initial_state), **splits,
              "evaluation_splits": list(evaluation_splits), "augmentation_audit": audit, "metadata": metadata,
              "training_seconds": time.perf_counter() - started,
              "peak_gpu_mib": torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0}
    checkpoint.parent.mkdir(parents=True, exist_ok=False)
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, checkpoint)
    write_json(metrics_path, record)
    return record


def _augment(config, base_config, views, base, stages, critical, augmenter, seed: int):
    augmented = {name: {} for name in METHODS}; audits = {name: {} for name in METHODS}
    for split, offset in (("train", 0), ("validation", 100), ("test", 200)):
        augmented["C0"][split] = traditional_augmentation(
            base[split], views[split]["window_id"], base_config["training"]["traditional_augmentation"], seed)
        sampling_seed = seed + int(base_config["spectral_diffusion"]["sampling_seed_offset"]) + offset
        augmented["C1"][split], c1_diag = augmenter.augment(
            base[split], "uniform", sampling_seed, batch_size=int(base_config["training"]["batch_size"]),
            noise_structure="iid")
        augmented["R1"][split], r1_diag = augmenter.augment(
            base[split], "selective", sampling_seed, 5, int(base_config["training"]["batch_size"]),
            noise_structure="iid")
        for name, diag in (("C1", c1_diag), ("R1", r1_diag)):
            mechanism = augmentation_mechanism_metrics(
                base[split], augmented[name][split], views[split]["labels"], stages[split],
                critical["masks"]["composite"], diag)
            mechanism["correlation_drift"] = correlation_drift(
                base[split], augmented[name][split], views[split]["labels"],
                float(config["diagnosis"]["high_correlation_quantile"]))
            audits[name][split] = mechanism
        audits["C0"][split] = {"finite": bool(np.isfinite(augmented["C0"][split]).all()),
                                "time_normalized_l1": float(np.mean(np.abs(augmented["C0"][split] - base[split])
                                                                        / max(float(base[split].std()), 1e-6))),
                                "correlation_drift": correlation_drift(
                                    base[split], augmented["C0"][split], views[split]["labels"],
                                    float(config["diagnosis"]["high_correlation_quantile"]))}
    if abs(audits["C1"]["train"]["expected_total_noise_budget"]
           - audits["R1"]["train"]["expected_total_noise_budget"]) > 1e-6:
        raise RuntimeError("C1/R1 total spectral noise budgets differ")
    return augmented, audits


def run(config: dict[str, Any]) -> dict[str, Any]:
    output = Path(config["output_dir"]); final_path = output / "result.json"
    if final_path.exists(): return json.loads(final_path.read_text(encoding="utf-8"))
    base_config = yaml.safe_load(Path(config["base_config"]).read_text(encoding="utf-8"))
    far_fix = yaml.safe_load(Path(config["far_fix_config"]).read_text(encoding="utf-8"))
    validate_frozen_config(config, base_config, far_fix); _configure(base_config)
    views, _ = load_fixed_views(base_config); base = bases(views)
    stages = {split: fault_stages(views[split], base_config) for split in views}
    critical, augmenter = _build_frequency_components(base_config, views, base, stages, str(config["device"]))
    mask_hash = array_sha256(critical["masks"]["composite"])
    manifest_hash = file_sha256(config["fixed_views_manifest"])
    code_files = ["scripts/run_frequency_selective_r1_3seed.py", "scripts/run_stage_frequency_diffusion_mvp.py",
                  "scripts/run_diffusion_quality_retest.py", "diffusion/frequency_selective.py",
                  "trainers/__init__.py", "trainers/baseline.py", "metrics/__init__.py",
                  "metrics/classification.py", "metrics/representation.py"]
    code_hash = sha256_strings([f"{path}:{file_sha256(path)}" for path in code_files])
    frozen_hash = hashlib.sha256(json.dumps(config["frozen"], sort_keys=True).encode()).hexdigest()
    seed_results = {}
    for seed in map(int, config["seeds"]):
        runtime = _runtime(base_config, seed); runtime["diagnosis"] = config["diagnosis"]
        pretrain_orders = epoch_orders(len(base["train"]), int(runtime["epochs"]), seed + 10_000)
        probe_orders = epoch_orders(len(base["train"]), int(runtime["probe_epochs"]), seed + 20_000)
        seed_everything(seed); template = build_model(runtime["model"], base["train"].shape[1], 2)
        initial_state = copy.deepcopy(template.state_dict())
        hashes = {"manifest_sha256": manifest_hash, "mask_sha256": mask_hash, "frozen_config_sha256": frozen_hash,
                  "training_code_sha256": code_hash, "initialization_sha256": _state_hash(initial_state),
                  "pretrain_order_sha256": sha256_strings([','.join(map(str, order)) for order in pretrain_orders]),
                  "probe_order_sha256": sha256_strings([','.join(map(str, order)) for order in probe_orders])}
        augmented, audits = _augment(config, base_config, views, base, stages, critical, augmenter, seed)
        methods = {}
        for name in METHODS:
            metadata = {**hashes, "method": name, "seed": seed,
                        "augmentation": "traditional" if name == "C0" else "uniform_iid_t3" if name == "C1" else "selective_iid_t5"}
            methods[name] = _fit_method(
                name, augmented[name], audits[name], views, base, stages, initial_state, pretrain_orders,
                probe_orders, runtime, str(config["device"]), output / f"seed_{seed}" / name / "model.pt", metadata)
        if len({methods[name]["initialization_sha256"] for name in METHODS}) != 1:
            raise RuntimeError("same-seed initialization fairness failed")
        seed_record = {"seed": seed, "methods": methods, "fairness": hashes,
                       "same_initialization": True, "same_pretrain_order": True, "same_probe_order": True,
                       "same_fixed_views": True, "same_mask": True, "c1_r1_same_noise_seed": True,
                       "c1_r1_equal_total_budget": True}
        write_json(output / f"seed_{seed}" / "result.json", seed_record); seed_results[str(seed)] = seed_record
    from scripts.summarize_frequency_selective_r1_3seed import summarize
    result = summarize(config, seed_results, {"manifest_sha256": manifest_hash, "mask_sha256": mask_hash,
                                              "frozen_config_sha256": frozen_hash, "training_code_sha256": code_hash})
    result.update(environment_metadata()); write_json(final_path, result)
    summarize(config, seed_results, result["fingerprints"], result=result, report_path=config["report"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/frequency_selective_r1_3seed.yaml")
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = run(config); print(json.dumps({"status": result["status"], "seeds": sorted(result["seed_results"]),
                                            "r1_vs_c1": result["summary"]["R1-C1"]}, ensure_ascii=False))


if __name__ == "__main__": main()
