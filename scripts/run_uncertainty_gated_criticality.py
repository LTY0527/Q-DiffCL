from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from diffusion import DiffusionSchedule, FrequencyForwardDiffusion, fit_spectral_statistics
from frequency import (build_uncertainty_gated_criticality, fault_stages,
                       fit_frequency_scaler, log_amplitude_phase)
from frequency.criticality import fault_type
from scripts.audit_semantic_diffusion_augmentation import bases
from scripts.run_3w_diffusion_1seed import UG_R1_METHOD, run as run_three_w
from scripts.run_diffusion_quality_retest import _state_hash, epoch_orders, load_fixed_views
from scripts.run_frequency_selective_r1_3seed import (_fit_method as fit_tep_method,
                                                       file_sha256, sha256_strings)
from scripts.run_stage_frequency_diffusion_mvp import (_configure, _runtime,
                                                        augmentation_mechanism_metrics)
from trainers import build_model
from utils import seed_everything, write_json


def _settings(stage: dict) -> dict:
    return copy.deepcopy(stage["criticality"])


def _ready(record: dict) -> dict:
    r1 = record["r1"]
    return {"fit_split": "train", "r1_composite": r1["composite"].tolist(),
            "r1_hard_mask": r1["masks"]["composite"].astype(int).tolist(),
            "r1_soft_mask": r1["soft_mask"].tolist(),
            "selection_probability": record["selection_probability"].tolist(),
            "assignment_confidence": record["assignment_confidence"].tolist(),
            "bootstrap_repeats": record["bootstrap_repeats"],
            "bootstrap_overlap": record["bootstrap_overlap"].tolist(),
            "bootstrap_unit_count": record["bootstrap_unit_count"],
            "bootstrap_unit_ids": record["bootstrap_unit_ids"],
            "stratified_unit_counts": record["stratified_unit_counts"], "bootstrap_scope": record["bootstrap_scope"]}


def run_three_w_stage(config: dict, data_root: Path, selected_seeds: list[int] | None = None) -> dict:
    stage = config["three_w"]
    if list(map(int, stage["seeds"])) != [42, 43, 44]: raise ValueError("UG-R1 3W seeds must be 42/43/44")
    base = yaml.safe_load(Path(stage["base_config"]).read_text(encoding="utf-8")); output = Path(stage["output_dir"])
    output.mkdir(parents=True, exist_ok=True); manifest_path = output / "result_manifest.json"
    completed = json.loads(manifest_path.read_text(encoding="utf-8")).get("seed_results", {}) if manifest_path.exists() else {}
    for seed in map(int, selected_seeds if selected_seeds is not None else stage["seeds"]):
        current = copy.deepcopy(base); current["seed"] = seed; current["protocol_seed"] = int(stage["protocol_seed"])
        current.pop("criticality_source", None); current["uncertainty_gated_criticality"] = True
        current["methods"] = [UG_R1_METHOD]; current["training"]["supcon_batching"] = "original"
        current["criticality"] = _settings(stage); current["output_dir"] = str(output / f"seed_{seed}")
        result = run_three_w(current, data_root); path = Path(current["output_dir"]) / "result.json"
        completed[str(seed)] = {"result_path": str(path), "status": "complete", "methods": list(result["methods"])}
        write_json(manifest_path, {"seed_results": completed})
    payload = {"stage": "3W", "method": "UG-R1", "seeds": sorted(map(int, completed)),
               "new_training_count": len(completed), "reused_baseline_count": 6, "seed_results": completed}
    write_json(manifest_path, payload); return payload


def _build_tep(config: dict):
    stage = config["tep"]; base_config = yaml.safe_load(Path(stage["base_config"]).read_text(encoding="utf-8"))
    _configure(base_config); views, _ = load_fixed_views(base_config); clean = bases(views)
    stages = {split: fault_stages(views[split], base_config) for split in views}
    train_log = log_amplitude_phase(clean["train"])[0]; scaler = fit_frequency_scaler(train_log, "train")
    unit_ids = np.asarray(views["train"]["run_uid"], dtype=object)
    unit_strata = np.asarray([fault_type(str(uid)) for uid in unit_ids], dtype=np.int64)
    criticality = build_uncertainty_gated_criticality(scaler.transform(train_log), views["train"], stages["train"],
        unit_ids, unit_strata, _settings(stage), train_log)
    statistics = fit_spectral_statistics(clean["train"], float(stage["spectral_diffusion"]["clip_quantile"]), "train")
    schedule = DiffusionSchedule.cosine(int(stage["spectral_diffusion"]["diffusion_steps"]), str(config["device"]))
    augmenter = FrequencyForwardDiffusion(statistics, schedule.alpha_bars, criticality["r1"]["soft_mask"],
        int(stage["spectral_diffusion"]["t_uniform"]), int(stage["spectral_diffusion"]["t_critical"]),
        bool(stage["spectral_diffusion"]["preserve_phase"]), bool(stage["spectral_diffusion"]["preserve_dc"]), str(config["device"]))
    return base_config, views, clean, stages, criticality, augmenter


def run_tep_stage(config: dict, selected_seeds: list[int] | None = None) -> dict:
    stage = config["tep"]
    if list(map(int, stage["seeds"])) != [7, 42, 2026]: raise ValueError("UG-R1 TEP seeds must be 7/42/2026")
    base_config, views, clean, stages, criticality, augmenter = _build_tep(config)
    baseline = json.loads(Path(stage["existing_result"]).read_text(encoding="utf-8"))
    output = Path(stage["output_dir"]); output.mkdir(parents=True, exist_ok=True); final_path = output / "result.json"
    completed = json.loads(final_path.read_text(encoding="utf-8")).get("seed_results", {}) if final_path.exists() else {}
    for seed in map(int, selected_seeds if selected_seeds is not None else stage["seeds"]):
        metrics_path = output / f"seed_{seed}" / "UG_R1" / "metrics.json"
        if str(seed) in completed and metrics_path.exists(): continue
        runtime = _runtime(base_config, seed); runtime["diagnosis"] = {"threshold_band_width": .05, "high_correlation_quantile": .90}
        augmented = {}; audits = {}
        for split, offset in (("train", 0), ("validation", 100), ("test", 200)):
            sampling_seed = seed + int(stage["spectral_diffusion"]["sampling_seed_offset"]) + offset
            augmented[split], diag = augmenter.augment(clean[split], "uncertainty_gated", sampling_seed,
                int(stage["spectral_diffusion"]["t_noncritical"]), int(base_config["training"]["batch_size"]),
                noise_structure="iid", assignment_confidence=criticality["assignment_confidence"])
            audits[split] = augmentation_mechanism_metrics(clean[split], augmented[split], views[split]["labels"],
                stages[split], criticality["r1"]["masks"]["composite"], diag)
            old_budget = baseline["seed_results"][str(seed)]["methods"]["R1"]["augmentation_audit"][split]["expected_total_noise_budget"]
            if abs(audits[split]["expected_total_noise_budget"] - old_budget) > 1e-6:
                raise RuntimeError(f"TEP UG-R1/R1 {split} noise budgets differ")
        pretrain_orders = epoch_orders(len(clean["train"]), int(runtime["epochs"]), seed + 10_000)
        probe_orders = epoch_orders(len(clean["train"]), int(runtime["probe_epochs"]), seed + 20_000)
        seed_everything(seed); template = build_model(runtime["model"], clean["train"].shape[1], 2)
        initial_state = copy.deepcopy(template.state_dict()); old = baseline["seed_results"][str(seed)]["fairness"]
        fairness = {"manifest_sha256": file_sha256(base_config["fixed_views"]["manifest"]),
                    "initialization_sha256": _state_hash(initial_state),
                    "pretrain_order_sha256": sha256_strings([','.join(map(str, order)) for order in pretrain_orders]),
                    "probe_order_sha256": sha256_strings([','.join(map(str, order)) for order in probe_orders])}
        for key, value in fairness.items():
            if value != old[key]: raise RuntimeError(f"TEP UG-R1 fairness differs for seed {seed}: {key}")
        metadata = {**old, "method": "UG_R1", "seed": seed, "augmentation": "uncertainty_gated_r1_iid_t5"}
        print("TEP", seed, "start UG-R1", flush=True)
        record = fit_tep_method("UG_R1", augmented, audits, views, clean, stages, initial_state,
            pretrain_orders, probe_orders, runtime, str(config["device"]), metrics_path.parent / "model.pt", metadata)
        completed[str(seed)] = {"seed": seed, "method": record, "fairness": fairness}
        write_json(final_path, {"seed_results": completed}); print("TEP", seed, "done UG-R1", flush=True)
    payload = {"stage": "TEP_BINARY_DETECTION", "method": "UG-R1", "seeds": sorted(map(int, completed)),
               "new_training_count": len(completed), "reused_baseline_count": 6,
               "criticality": _ready(criticality), "seed_results": completed,
               "test_used_for_uncertainty_or_fit": False}
    write_json(final_path, payload); return payload


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/uncertainty_gated_criticality.yaml")
    parser.add_argument("--stage", choices=("3w", "tep"), required=True); parser.add_argument("--data-root", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+"); args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.stage == "3w":
        if args.data_root is None: raise ValueError("3W stage requires --data-root")
        result = run_three_w_stage(config, args.data_root, args.seeds)
    else: result = run_tep_stage(config, args.seeds)
    print(json.dumps({key: value for key, value in result.items() if key not in {"seed_results", "criticality"}}, ensure_ascii=False))


if __name__ == "__main__": main()
