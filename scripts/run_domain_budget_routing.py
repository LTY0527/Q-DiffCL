from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from augmentations import domain_budget_route
from diffusion import DiffusionSchedule, FrequencyForwardDiffusion, fit_spectral_statistics
from diffusion.fixed_views import sha256_file, sha256_strings
from frequency import fault_stages
from scripts.run_3w_diffusion_1seed import METHODS as THREE_W_METHODS, run as run_three_w_model
from scripts.run_budget_shrinkage_diagnostic import (_mask, _same_fairness,
                                                     _variance, rho_name,
                                                     validation_metrics)
from scripts.run_diffusion_quality_retest import _state_hash, epoch_orders
from scripts.run_frequency_selective_r1_3seed import _fit_method, file_sha256
from scripts.run_stage_frequency_diffusion_mvp import _configure, _runtime, augmentation_mechanism_metrics
from trainers import build_model
from utils import seed_everything, select_device, write_json


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def variant_name(rho: float) -> str:
    return f"DCBR_{int(round(float(rho) * 100)):03d}"


def _validate(config: dict[str, Any]) -> None:
    final = yaml.safe_load(Path(config["final_config"]).read_text(encoding="utf-8"))
    if not final.get("frozen") or final["weights"] != {
        "weight_discriminative": .5, "weight_early": .5, "weight_run_stability": 0.0}:
        raise RuntimeError("FINAL_QDIFFCL changed")
    if list(map(float, config["rhos"])) != [0, .25, .5, .75, 1]:
        raise RuntimeError("DCBR candidate grid changed")
    if float(config["sigma_base"]) != .05:
        raise RuntimeError("frozen SCALING sigma changed")
    for dataset, key in (("3W", "three_w"), ("TEP", "tep")):
        mask = _read(config[key]["final_mask"])["criticality"]
        if mask["mask_sha256"] != final["mask_sha256"][dataset] or mask["fit_split"] != "train":
            raise RuntimeError(f"{dataset} FINAL mask changed")


def _manifest(path: Path) -> dict[str, Any]:
    return _read(path).get("results", {}) if path.exists() else {}


def _store(path: Path, results: dict[str, Any], key: str, record: dict[str, Any]) -> None:
    results[key] = record
    write_json(path, {"results": results, "evaluation_split": "validation", "test_read": False})


def _reference(config: dict[str, Any], stage_key: str, seed: int) -> dict[str, Any]:
    records = _read(config[stage_key]["reference_manifest"])["results"]
    if f"rho_100|{seed}" not in records:
        frozen = _read(config[stage_key]["final_test_manifest"])["results"][f"FINAL_QDIFFCL|{seed}"]
        return {"fairness": frozen["fairness"], "evaluation_split": "validation",
                "test_metrics_read": False, "method": {}, "source": config[stage_key]["final_test_manifest"]}
    item = copy.deepcopy(records[f"rho_100|{seed}"])
    if item["evaluation_split"] != "validation" or item["test_metrics_read"] is not False:
        raise RuntimeError("DCBR FINAL reference is not validation-only")
    return item


def run_three_w(config: dict[str, Any], data_root: Path, rhos: list[float], seeds: list[int], device: str) -> dict[str, Any]:
    budget_config = yaml.safe_load(Path(config["budget_config"]).read_text(encoding="utf-8"))
    stage = config["three_w"]; mask = _mask(budget_config, "three_w")
    output = Path(stage["output_dir"]); path = output / "manifest.json"; results = _manifest(path)
    for rho in rhos:
        variance, budget = _variance(budget_config, mask, device, rho)
        variance_path = output / "variances" / f"{rho_name(rho)}.npy"; variance_path.parent.mkdir(parents=True, exist_ok=True)
        if variance_path.exists() and not np.array_equal(np.load(variance_path, allow_pickle=False), variance):
            raise RuntimeError("existing DCBR 3W variance changed")
        if not variance_path.exists(): np.save(variance_path, variance, allow_pickle=False)
        for seed in seeds:
            key = f"{variant_name(rho)}|{seed}"
            if key in results: continue
            reference = _reference(config, "three_w", seed)
            if rho == 1.0:
                record = {**reference, "variant": variant_name(rho), "rho": rho,
                          "scaling_std": 0.0, "budget_audit": budget,
                          "routing_boundary": "exact_reused_FINAL"}
                _store(path, results, key, record); continue
            base = yaml.safe_load(Path(stage["base_config"]).read_text(encoding="utf-8"))
            current = copy.deepcopy(base); current.update({"seed": seed, "protocol_seed": int(stage["protocol_seed"]),
                "criticality_source": str(stage["final_mask"]), "selective_variance_override": str(variance_path),
                "methods": [THREE_W_METHODS[2]], "evaluation_split": "validation",
                "output_dir": str(output / variant_name(rho) / f"seed_{seed}"),
                "domain_budget_routing": {"rho": rho, "sigma_base": float(config["sigma_base"]),
                    "scaling_seed": seed, "validation_scaling_seed": seed + 100}})
            current["training"]["supcon_batching"] = "original"
            result = run_three_w_model(current, data_root)
            if result["evaluation_split"] != "validation" or not _same_fairness(result["fairness"], reference["fairness"], "3W"):
                raise RuntimeError("3W DCBR validation/fairness audit failed")
            diagnostics = result["augmentation_diagnostics"][THREE_W_METHODS[2]]
            record = {"variant": variant_name(rho), "rho": rho, "seed": seed,
                      "evaluation_split": "validation", "method": result["methods"][THREE_W_METHODS[2]],
                      "fairness": result["fairness"], "mask_sha256": mask["mask_sha256"],
                      "budget_audit": budget, "scaling_std": (1-rho)*float(config["sigma_base"]),
                      "routing_audit": diagnostics, "test_metrics_read": False, "training": "new_training"}
            _store(path, results, key, record)
    return results


def _load_tep_context(config: dict[str, Any]) -> tuple[Any, ...]:
    stage = config["tep"]; base = yaml.safe_load(Path(stage["base_config"]).read_text(encoding="utf-8")); _configure(base)
    manifest = _read(base["fixed_views"]["manifest"]); views = {}
    for split in ("train", "validation"):
        record = manifest["splits"][split]; path = Path(record["path"])
        if sha256_file(path) != record["sha256"]: raise RuntimeError(f"TEP {split} view hash changed")
        with np.load(path, allow_pickle=False) as archive: views[split] = {key: archive[key] for key in archive.files}
        if sha256_strings(list(map(str, views[split]["window_id"]))) != record["window_ids_sha256"]:
            raise RuntimeError(f"TEP {split} window order changed")
    clean = {split: views[split]["clean"].astype(np.float32) for split in views}
    stages = {split: fault_stages(views[split], base) for split in views}
    return base, views, clean, stages


def run_tep(config: dict[str, Any], context: tuple[Any, ...], rhos: list[float], seeds: list[int], device: str) -> dict[str, Any]:
    stage = config["tep"]; base, views, clean, stages = context
    budget_config = yaml.safe_load(Path(config["budget_config"]).read_text(encoding="utf-8")); mask = _mask(budget_config, "tep")
    output = Path(stage["output_dir"]); path = output / "manifest.json"; results = _manifest(path)
    statistics = fit_spectral_statistics(clean["train"], float(config["spectral_diffusion"]["clip_quantile"]), "train")
    schedule = DiffusionSchedule.cosine(int(config["spectral_diffusion"]["diffusion_steps"]), device)
    augmenter = FrequencyForwardDiffusion(statistics, schedule.alpha_bars, np.asarray(mask["soft_mask"], np.float32),
                                          3, 1, True, True, device)
    for rho in rhos:
        variance, budget = _variance(budget_config, mask, device, rho)
        for seed in seeds:
            key = f"{variant_name(rho)}|{seed}"
            if key in results: continue
            reference = _reference(config, "tep", seed)
            if rho == 1.0:
                record = {**reference, "variant": variant_name(rho), "rho": rho,
                          "scaling_std": 0.0, "budget_audit": budget,
                          "routing_boundary": "exact_reused_FINAL"}
                _store(path, results, key, record); continue
            augmented = {}; audits = {}
            for split, offset in (("train", 0), ("validation", 100)):
                sampling_seed = seed + int(config["spectral_diffusion"]["sampling_seed_offset"]) + offset
                if rho == 0:
                    diffused = clean[split].copy(); diffusion_audit = {"expected_total_noise_budget": 0.0,
                                                                      "phase_preserved": True, "dc_preserved": True,
                                                                      "finite": True, "exact_clean_bypass": True}
                else:
                    diffused, diffusion_audit = augmenter.augment(
                        clean[split], "budget_scaled_selective", sampling_seed, 5,
                        int(base["training"]["batch_size"]), variance_override=variance)
                augmented[split], route = domain_budget_route(
                    clean[split], diffused, views[split]["window_id"], rho, float(config["sigma_base"]), seed + offset)
                audits[split] = {"diffusion": diffusion_audit, "routing": route,
                    "mechanism": augmentation_mechanism_metrics(clean[split], augmented[split], views[split]["labels"],
                                                                 stages[split], np.asarray(mask["hard_mask"], bool), diffusion_audit)}
            runtime = _runtime(base, seed); runtime["diagnosis"] = {"threshold_band_width": .05, "high_correlation_quantile": .90}
            pretrain_orders = epoch_orders(len(clean["train"]), int(runtime["epochs"]), seed + 10000)
            probe_orders = epoch_orders(len(clean["train"]), int(runtime["probe_epochs"]), seed + 20000)
            seed_everything(seed); template = build_model(runtime["model"], clean["train"].shape[1], 2); initial = copy.deepcopy(template.state_dict())
            fairness = {"manifest_sha256": file_sha256(base["fixed_views"]["manifest"]),
                        "initialization_sha256": _state_hash(initial),
                        "pretrain_order_sha256": hashlib.sha256("\n".join(','.join(map(str,o)) for o in pretrain_orders).encode()).hexdigest(),
                        "probe_order_sha256": hashlib.sha256("\n".join(','.join(map(str,o)) for o in probe_orders).encode()).hexdigest()}
            if not _same_fairness(fairness, reference["fairness"], "TEP"):
                raise RuntimeError("TEP DCBR fairness audit failed")
            checkpoint = output / variant_name(rho) / f"seed_{seed}" / "model.pt"
            method = _fit_method(variant_name(rho), augmented, audits, views, clean, stages, initial,
                                 pretrain_orders, probe_orders, runtime, device, checkpoint,
                                 {**fairness, "rho": rho, "evaluation_splits": ["validation"], "test_metrics_read": False},
                                 evaluation_splits=("validation",))
            record = {"variant": variant_name(rho), "rho": rho, "seed": seed,
                      "evaluation_split": "validation", "method": method, "fairness": fairness,
                      "mask_sha256": mask["mask_sha256"], "budget_audit": budget,
                      "scaling_std": (1-rho)*float(config["sigma_base"]), "routing_audit": audits,
                      "test_metrics_read": False, "training": "new_training"}
            _store(path, results, key, record)
    return results


def select_top2(config: dict[str, Any], requested: tuple[str, ...] = ("3W", "TEP")) -> dict[str, Any]:
    selection_path = Path(config["output"]["selection"])
    previous = _read(selection_path) if selection_path.exists() else {}
    selected = dict(previous.get("top2", {})); metrics = dict(previous.get("stage_b_metrics", {}))
    for dataset, key in (("3W", "three_w"), ("TEP", "tep")):
        if dataset not in requested:
            continue
        records = _manifest(Path(config[key]["output_dir"]) / "manifest.json"); seed = int(config[key]["stage_b_seed"])
        current = {float(rho): validation_metrics(dataset, records[f"{variant_name(rho)}|{seed}"])
                   for rho in map(float, config["rhos"])}
        selected[dataset] = sorted(current, key=lambda rho: (current[rho]["macro_f1"], current[rho]["auprc"],
                                                             -current[rho]["far"], -rho), reverse=True)[:2]
        metrics[dataset] = current
    payload = {"split": "validation", "test_read": False, "top2": selected, "stage_b_metrics": metrics}
    write_json(selection_path, payload); return payload


def run(config: dict[str, Any], data_root: Path, stage_name: str, dataset: str) -> dict[str, Any]:
    _validate(config)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", str(yaml.safe_load(Path(config["tep"]["base_config"]).read_text(encoding="utf-8")).get("cublas_workspace_config", ":4096:8")))
    device = select_device(str(config["device"])); datasets = ("3w", "tep") if dataset == "both" else (dataset,)
    if stage_name in ("B", "all"):
        rhos = list(map(float, config["rhos"]))
        if "3w" in datasets: run_three_w(config, data_root, rhos, [int(config["three_w"]["stage_b_seed"])], device)
        if "tep" in datasets: run_tep(config, _load_tep_context(config), rhos, [int(config["tep"]["stage_b_seed"])], device)
    requested = tuple("3W" if name == "3w" else "TEP" for name in datasets)
    selection = select_top2(config, requested)
    if stage_name in ("C", "all"):
        for current, key in (("3w", "three_w"), ("tep", "tep")):
            if current not in datasets: continue
            rhos = sorted(set([0.0, 1.0, *map(float, selection["top2"]["3W" if current == "3w" else "TEP"])]))
            seeds = list(map(int, config[key]["stage_c_seeds"]))
            if current == "3w": run_three_w(config, data_root, rhos, seeds, device)
            else: run_tep(config, _load_tep_context(config), rhos, seeds, device)
    write_json(Path(config["output"]["manifest"]), {"stage": stage_name, "selection": selection,
               "validation_only": True, "test_read": False})
    return {"stage": stage_name, "top2": selection["top2"], "test_read": False}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/domain_calibrated_budget_routing.yaml")
    parser.add_argument("--data-root", type=Path, required=True); parser.add_argument("--stage", choices=("B","C","all"), default="all")
    parser.add_argument("--dataset", choices=("3w","tep","both"), default="both")
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    print(json.dumps(run(config, args.data_root, args.stage, args.dataset), ensure_ascii=False))


if __name__ == "__main__": main()
