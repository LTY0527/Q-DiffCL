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
from frequency import (build_rival_aware_criticality, fault_stages,
                       fit_frequency_scaler, log_amplitude_phase)
from scripts.audit_semantic_diffusion_augmentation import bases
from scripts.run_3w_diffusion_1seed import RRDC_METHOD, run as run_three_w
from scripts.run_diffusion_quality_retest import (_fit_supcon, epoch_orders,
                                                  load_fixed_views)
from scripts.run_frequency_selective_r1_3seed import file_sha256, sha256_strings
from scripts.run_hierarchical_fault_semantic_criticality import (
    _evaluate_multiclass, _fit_multiclass_probe, multiclass_labels)
from scripts.run_stage_frequency_diffusion_mvp import _configure
from trainers import build_model
from utils import seed_everything, write_json


def _settings(stage: dict) -> dict:
    value = copy.deepcopy(stage["criticality"])
    value["diagnostic_classes"] = list(map(int, stage["diagnostic_classes"]))
    return value


def run_three_w_stage(config: dict, data_root: Path, selected_seeds: list[int] | None = None) -> dict:
    stage = config["three_w"]
    if list(map(int, stage["seeds"])) != [42, 43, 44]:
        raise ValueError("RRDC 3W seeds must be 42/43/44")
    base = yaml.safe_load(Path(stage["base_config"]).read_text(encoding="utf-8"))
    output = Path(stage["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "result_manifest.json"
    completed = json.loads(manifest_path.read_text(encoding="utf-8")).get("seed_results", {}) if manifest_path.exists() else {}
    for seed in map(int, selected_seeds if selected_seeds is not None else stage["seeds"]):
        current = copy.deepcopy(base); current["seed"] = seed; current["protocol_seed"] = int(stage["protocol_seed"])
        current.pop("criticality_source", None); current["rival_aware_criticality"] = True
        current["methods"] = [RRDC_METHOD]; current["training"]["supcon_batching"] = "original"
        current["criticality"] = _settings(stage); current["output_dir"] = str(output / f"seed_{seed}")
        result = run_three_w(current, data_root); path = Path(current["output_dir"]) / "result.json"
        completed[str(seed)] = {"result_path": str(path), "status": "complete", "methods": list(result["methods"])}
        write_json(manifest_path, {"seed_results": completed})
    payload = {"stage": "3W", "method": "RRDC", "seeds": sorted(map(int, completed)),
               "new_training_count": len(completed), "reused_baseline_count": 12,
               "criticality_fit_scope": "train-only fault-vs-fault run/WELL-level", "seed_results": completed}
    write_json(manifest_path, payload); return payload


def _criticality_ready(record: dict) -> dict:
    result = {"fit_split": record["fit_split"], "fault_run_counts": record["fault_run_counts"],
              "diagnostic_classes": record["diagnostic_classes"], "hard_rival_quantile": record["hard_rival_quantile"],
              "bootstrap_repeats": record["bootstrap_repeats"], "combination": record["combination"],
              "shared_hard_mask": record["shared"]["masks"]["composite"].astype(int).tolist(), "diagnostic": {}, "final": {}}
    for kind, item in record["diagnostic"].items():
        result["diagnostic"][str(kind)] = {
            "score": item["score"].tolist(), "reliability": item["reliability"].tolist(),
            "reliable_score": item["reliable_score"].tolist(), "hard_mask": item["hard_mask"].astype(int).tolist(),
            "soft_mask": item["soft_mask"].tolist(), "hardest_rival": item["hardest_rival"],
            "hardest_rival_score": item["hardest_rival_score"],
            "pairwise_summary": {str(k): v for k, v in item["pairwise_summary"].items()},
        }
    for kind, item in record["final"].items():
        result["final"][str(kind)] = {"score": item["score"].tolist(),
                                     "hard_mask": item["hard_mask"].astype(int).tolist(),
                                     "soft_mask": item["soft_mask"].tolist()}
    return result


def run_tep_stage(config: dict, selected_seeds: list[int] | None = None) -> dict:
    stage = config["tep"]
    if list(map(int, stage["seeds"])) != [7, 42, 2026]:
        raise ValueError("RRDC TEP seeds must be 7/42/2026")
    base_config = yaml.safe_load(Path(stage["base_config"]).read_text(encoding="utf-8"))
    base_config["fixed_views"] = {"manifest": stage["fixed_views_manifest"]}; _configure(base_config)
    views, _ = load_fixed_views(base_config); clean = bases(views)
    stages = {split: fault_stages(views[split], base_config) for split in views}
    labels = {split: multiclass_labels(views[split]) for split in views}
    train_log = log_amplitude_phase(clean["train"])[0]
    scaler = fit_frequency_scaler(train_log, "train")
    criticality = build_rival_aware_criticality(
        scaler.transform(train_log), views["train"], stages["train"], _settings(stage), train_log)
    statistics = fit_spectral_statistics(clean["train"], float(stage["spectral_diffusion"]["clip_quantile"]), "train")
    schedule = DiffusionSchedule.cosine(int(stage["spectral_diffusion"]["diffusion_steps"]), str(config["device"]))
    augmenter = FrequencyForwardDiffusion(statistics, schedule.alpha_bars, criticality["shared"]["soft_mask"],
        int(stage["spectral_diffusion"]["t_uniform"]), int(stage["spectral_diffusion"]["t_critical"]),
        bool(stage["spectral_diffusion"]["preserve_phase"]), bool(stage["spectral_diffusion"]["preserve_dc"]), str(config["device"]))
    output = Path(stage["output_dir"]); output.mkdir(parents=True, exist_ok=True); final_path = output / "result.json"
    completed = json.loads(final_path.read_text(encoding="utf-8")).get("seed_results", {}) if final_path.exists() else {}
    runtime_base = copy.deepcopy(stage["training"]); runtime_base["protocol"] = base_config["protocol"]
    runtime_base["detection"] = stage["detection"]
    baseline = json.loads(Path(stage["existing_hfsc_result"]).read_text(encoding="utf-8"))
    for seed in map(int, selected_seeds if selected_seeds is not None else stage["seeds"]):
        result_path = output / f"seed_{seed}" / RRDC_METHOD / "metrics.json"
        if str(seed) in completed and result_path.exists(): continue
        runtime = copy.deepcopy(runtime_base); runtime["random_seed"] = seed
        augmented = {}; audits = {}
        for split, offset in (("train", 0), ("validation", 100), ("test", 200)):
            sampling_seed = seed + int(stage["spectral_diffusion"]["sampling_seed_offset"]) + offset
            augmented[split], audits[split] = augmenter.augment_hierarchical(
                clean[split], labels[split], criticality["soft_masks"], sampling_seed,
                int(stage["spectral_diffusion"]["t_noncritical"]), int(runtime["batch_size"]))
            baseline_budget = baseline["seed_results"][str(seed)]["methods"]["R1"]["augmentation_audit"][split]["expected_total_noise_budget"]
            if abs(audits[split]["expected_total_noise_budget"] - baseline_budget) > 1e-6:
                raise RuntimeError(f"TEP RRDC/R1 {split} total noise budgets differ for seed {seed}")
        pretrain_orders = epoch_orders(len(clean["train"]), int(runtime["epochs"]), seed + 10_000)
        probe_orders = epoch_orders(len(clean["train"]), int(runtime["probe_epochs"]), seed + 20_000)
        seed_everything(seed); template = build_model(runtime["model"], clean["train"].shape[1], 21)
        initial_state = copy.deepcopy(template.state_dict())
        fairness = {"fixed_views_sha256": file_sha256(stage["fixed_views_manifest"]),
                    "initialization_sha256": hashlib.sha256(b''.join(v.cpu().numpy().tobytes() for _, v in sorted(initial_state.items()))).hexdigest(),
                    "pretrain_order_sha256": sha256_strings([','.join(map(str, order)) for order in pretrain_orders]),
                    "probe_order_sha256": sha256_strings([','.join(map(str, order)) for order in probe_orders])}
        old_fairness = baseline["seed_results"][str(seed)]["fairness"]
        if fairness != old_fairness: raise RuntimeError(f"TEP RRDC fairness differs from HFSC baselines for seed {seed}")
        print("TEP", seed, "start RRDC", flush=True); seed_everything(seed)
        model = build_model(runtime["model"], clean["train"].shape[1], 21).to(str(config["device"])); model.load_state_dict(initial_state)
        pretrain = _fit_supcon(model,
            {"clean": clean["train"], "restored": augmented["train"], "labels": labels["train"]},
            {"clean": clean["validation"], "restored": augmented["validation"], "labels": labels["validation"]},
            np.ones(len(labels["train"]), np.float32), np.ones(len(labels["validation"]), np.float32),
            pretrain_orders, runtime, str(config["device"]))
        probe = _fit_multiclass_probe(model, clean["train"], labels["train"], clean["validation"],
                                      labels["validation"], probe_orders, runtime, str(config["device"]))
        metrics = _evaluate_multiclass(model, clean, labels, views, stages, runtime, str(config["device"]))
        result_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "fairness": fairness}, result_path.parent / "model.pt")
        record = {"metrics": metrics, "augmentation_audit": audits, "pretrain_history": pretrain,
                  "probe_history": probe, "fairness": fairness}
        write_json(result_path, record); completed[str(seed)] = {"seed": seed, "method": record, "fairness": fairness}
        write_json(final_path, {"seed_results": completed}); print("TEP", seed, "done RRDC", metrics["diagnosis"]["macro_f1"], flush=True)
    payload = {"stage": "TEP_21_CLASS_DIAGNOSIS", "seeds": sorted(map(int, completed)),
               "new_training_count": len(completed), "method": "RRDC", "reused_baseline_count": 12,
               "criticality_fit_scope": "train-only fault-vs-fault run-level", "criticality": _criticality_ready(criticality),
               "seed_results": completed, "test_used_for_selection": False, "paper_final_claim_allowed": False}
    write_json(final_path, payload); return payload


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/rival_aware_reliable_diagnostic_criticality.yaml")
    parser.add_argument("--stage", choices=("3w", "tep"), required=True); parser.add_argument("--data-root", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+"); args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.stage == "3w":
        if args.data_root is None: raise ValueError("3W stage requires --data-root")
        result = run_three_w_stage(config, args.data_root, args.seeds)
    else: result = run_tep_stage(config, args.seeds)
    print(json.dumps({k: v for k, v in result.items() if k not in {"seed_results", "criticality"}}, ensure_ascii=False))


if __name__ == "__main__": main()
