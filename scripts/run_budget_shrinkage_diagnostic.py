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

from diffusion import (DiffusionSchedule, FrequencyForwardDiffusion,
                       fit_spectral_statistics, scale_spectral_budget,
                       spectral_noise_variance)
from frequency import fault_stages
from scripts.run_3w_diffusion_1seed import METHODS as THREE_W_METHODS, run as run_three_w_model
from scripts.run_diffusion_quality_retest import _state_hash, epoch_orders, load_fixed_views
from scripts.run_frequency_selective_r1_3seed import _fit_method, file_sha256, sha256_strings
from scripts.run_stage_frequency_diffusion_mvp import _configure, _runtime, augmentation_mechanism_metrics
from trainers import build_model
from utils import seed_everything, select_device, write_json


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rho_name(rho: float) -> str:
    return f"rho_{int(round(float(rho) * 100)):03d}"


def _validate(config: dict[str, Any]) -> None:
    final = yaml.safe_load(Path(config["final_config"]).read_text(encoding="utf-8"))
    expected_weights = {"weight_discriminative": .5, "weight_early": .5, "weight_run_stability": 0.0}
    if not final.get("frozen") or final.get("weights") != expected_weights:
        raise ValueError("FINAL_QDIFFCL weights changed")
    if list(map(float, config["rhos"])) != [0.0, .25, .5, .75, 1.0]:
        raise ValueError("budget diagnostic must use the frozen five-point rho grid")
    spectral = config["spectral_diffusion"]
    for key, value in final["spectral_diffusion"].items():
        if key in spectral and spectral[key] != value:
            raise ValueError(f"frozen spectral setting changed: {key}")
    if final["criticality"]["critical_ratio"] != .30:
        raise ValueError("FINAL critical ratio changed")
    for dataset, stage_key in (("3W", "three_w"), ("TEP", "tep")):
        payload = _read(config[stage_key]["final_mask"])["criticality"]
        if payload["mask_sha256"] != final["mask_sha256"][dataset]:
            raise RuntimeError(f"{dataset} FINAL mask hash changed")
        if payload.get("test_or_validation_used"):
            raise RuntimeError(f"{dataset} FINAL mask is not train-only")


def _mask(config: dict[str, Any], stage_key: str) -> dict[str, Any]:
    return _read(config[stage_key]["final_mask"])["criticality"]


def _variance(config: dict[str, Any], mask: dict[str, Any], device: str, rho: float) -> tuple[np.ndarray, dict[str, Any]]:
    schedule = DiffusionSchedule.cosine(int(config["spectral_diffusion"]["diffusion_steps"]), device)
    soft = torch.as_tensor(mask["soft_mask"], dtype=torch.float32, device=device)
    final = spectral_noise_variance(
        schedule.alpha_bars, soft.shape[0], soft.shape[1], "selective",
        int(config["spectral_diffusion"]["t_uniform"]), bool(config["spectral_diffusion"]["preserve_dc"]),
        soft, int(config["spectral_diffusion"]["t_critical"]),
        int(config["spectral_diffusion"]["t_noncritical"]),
    )
    scaled = scale_spectral_budget(final, float(rho), bool(config["spectral_diffusion"]["preserve_dc"]))
    hard = torch.as_tensor(mask["hard_mask"], dtype=torch.bool, device=device)
    positive = final > 0
    ratio_error = 0.0 if not positive.any() else float(torch.max(torch.abs(scaled[positive] / final[positive] - float(rho))))
    expected_budget = float((final * float(rho)).mean())
    audit = {"rho": float(rho), "final_total_budget": float(final.mean()),
             "effective_total_budget": float(scaled.mean()),
             "expected_effective_total_budget": expected_budget,
             "total_budget_absolute_error": abs(float(scaled.mean()) - expected_budget),
             "critical_budget": float(scaled[hard].mean()),
             "noncritical_budget": float(scaled[~hard].mean()),
             "relative_allocation_max_error": ratio_error, "mask_sha256": mask["mask_sha256"],
             "preserve_dc": bool(torch.all(scaled[:, 0] == 0)), "finite": bool(torch.isfinite(scaled).all())}
    tolerance = float(config["audit"]["budget_absolute_tolerance"])
    if audit["total_budget_absolute_error"] > tolerance or ratio_error > float(config["audit"]["allocation_ratio_tolerance"]):
        raise RuntimeError("budget shrinkage audit failed")
    return scaled.cpu().numpy(), audit


def _reference(config: dict[str, Any], stage_key: str, seed: int) -> dict[str, Any]:
    manifest = _read(config[stage_key]["rho1_reference_manifest"])["results"]
    record = copy.deepcopy(manifest[f"DE_50_50|{seed}"])
    if record.get("evaluation_split") != "validation" or record.get("test_metrics_read") is not False:
        raise RuntimeError("rho=1 reference is not validation-only")
    if stage_key == "tep" and "test" in record["method"]:
        raise RuntimeError("TEP rho=1 reference contains test metrics")
    return record


def _same_fairness(left: dict[str, Any], right: dict[str, Any], dataset: str) -> bool:
    keys = (("initialization_sha256", "window_refs_sha256", "supcon_batch_order_sha256")
            if dataset == "3W" else
            ("manifest_sha256", "initialization_sha256", "pretrain_order_sha256", "probe_order_sha256"))
    return all(left.get(key) == right.get(key) for key in keys)


def _manifest(path: Path) -> dict[str, Any]:
    return _read(path).get("results", {}) if path.exists() else {}


def _store(path: Path, results: dict[str, Any], key: str, record: dict[str, Any]) -> None:
    results[key] = record; write_json(path, {"results": results, "test_metrics_read": False})


def run_three_w(config: dict[str, Any], data_root: Path, rhos: list[float], seeds: list[int], device: str) -> dict[str, Any]:
    stage = config["three_w"]; mask = _mask(config, "three_w")
    output = Path(stage["output_dir"]); manifest_path = output / "manifest.json"; results = _manifest(manifest_path)
    for rho in rhos:
        variance, budget = _variance(config, mask, device, rho)
        variance_path = output / "variances" / f"{rho_name(rho)}.npy"
        variance_path.parent.mkdir(parents=True, exist_ok=True)
        if variance_path.exists():
            if not np.array_equal(np.load(variance_path, allow_pickle=False), variance):
                raise RuntimeError("existing 3W variance artifact differs")
        else:
            np.save(variance_path, variance, allow_pickle=False)
        for seed in seeds:
            key = f"{rho_name(rho)}|{seed}"
            if key in results: continue
            reference = _reference(config, "three_w", seed)
            if np.isclose(rho, 1.0):
                record = {**reference, "rho": 1.0, "budget_audit": budget,
                          "source": str(stage["rho1_reference_manifest"]), "training": "reused_validation_reference"}
                _store(manifest_path, results, key, record); continue
            base = yaml.safe_load(Path(stage["base_config"]).read_text(encoding="utf-8"))
            current = copy.deepcopy(base)
            current.update({"seed": int(seed), "protocol_seed": int(stage["protocol_seed"]),
                            "criticality_source": str(stage["final_mask"]),
                            "selective_variance_override": str(variance_path),
                            "methods": [THREE_W_METHODS[2]], "evaluation_split": "validation",
                            "output_dir": str(output / rho_name(rho) / f"seed_{seed}")})
            current["training"]["supcon_batching"] = "original"
            result = run_three_w_model(current, data_root)
            if result.get("evaluation_split") != "validation":
                raise RuntimeError("3W budget diagnostic evaluated test")
            if not _same_fairness(result["fairness"], reference["fairness"], "3W"):
                raise RuntimeError("3W rho fairness differs from FINAL")
            diagnostics = result["augmentation_diagnostics"][THREE_W_METHODS[2]]
            for split in ("train", "validation"):
                if abs(float(diagnostics[split]["expected_total_noise_budget"]) - budget["effective_total_budget"]) > 1e-8:
                    raise RuntimeError("3W realized budget differs from requested rho")
            record = {"variant": rho_name(rho), "rho": float(rho), "seed": int(seed),
                      "evaluation_split": "validation", "method": result["methods"][THREE_W_METHODS[2]],
                      "fairness": result["fairness"], "mask_sha256": mask["mask_sha256"],
                      "budget_audit": budget, "augmentation_diagnostics": diagnostics,
                      "result_path": str(Path(current["output_dir"]) / "result.json"),
                      "test_metrics_read": False, "training": "new_training"}
            _store(manifest_path, results, key, record)
    return results


def _tep_context(config: dict[str, Any]) -> tuple[Any, ...]:
    stage = config["tep"]; base = yaml.safe_load(Path(stage["base_config"]).read_text(encoding="utf-8")); _configure(base)
    views, _ = load_fixed_views(base); clean = {split: views[split]["clean"].astype(np.float32) for split in views}
    stages = {split: fault_stages(views[split], base) for split in views}
    return base, views, clean, stages


def run_tep(config: dict[str, Any], context: tuple[Any, ...], rhos: list[float], seeds: list[int], device: str) -> dict[str, Any]:
    stage = config["tep"]; base, views, clean, stages = context; mask = _mask(config, "tep")
    output = Path(stage["output_dir"]); manifest_path = output / "manifest.json"; results = _manifest(manifest_path)
    statistics = fit_spectral_statistics(clean["train"], float(config["spectral_diffusion"]["clip_quantile"]), "train")
    schedule = DiffusionSchedule.cosine(int(config["spectral_diffusion"]["diffusion_steps"]), device)
    augmenter = FrequencyForwardDiffusion(statistics, schedule.alpha_bars, np.asarray(mask["soft_mask"], np.float32),
                                          3, 1, True, True, device)
    for rho in rhos:
        variance, budget = _variance(config, mask, device, rho)
        for seed in seeds:
            key = f"{rho_name(rho)}|{seed}"
            if key in results: continue
            reference = _reference(config, "tep", seed)
            if np.isclose(rho, 1.0):
                record = {**reference, "rho": 1.0, "budget_audit": budget,
                          "source": str(stage["rho1_reference_manifest"]), "training": "reused_validation_reference"}
                _store(manifest_path, results, key, record); continue
            augmented, audits = {}, {}
            for split, offset in (("train", 0), ("validation", 100)):
                sampling_seed = int(seed) + int(config["spectral_diffusion"]["sampling_seed_offset"]) + offset
                augmented[split], diagnostics = augmenter.augment(
                    clean[split], "budget_scaled_selective", sampling_seed, 5,
                    int(base["training"]["batch_size"]), noise_structure="iid", variance_override=variance)
                if abs(float(diagnostics["expected_total_noise_budget"]) - budget["effective_total_budget"]) > 1e-8:
                    raise RuntimeError("TEP realized budget differs from requested rho")
                audits[split] = augmentation_mechanism_metrics(
                    clean[split], augmented[split], views[split]["labels"], stages[split],
                    np.asarray(mask["hard_mask"], bool), diagnostics)
            runtime = _runtime(base, int(seed)); runtime["diagnosis"] = {"threshold_band_width": .05, "high_correlation_quantile": .90}
            pretrain_orders = epoch_orders(len(clean["train"]), int(runtime["epochs"]), int(seed) + 10000)
            probe_orders = epoch_orders(len(clean["train"]), int(runtime["probe_epochs"]), int(seed) + 20000)
            seed_everything(int(seed)); template = build_model(runtime["model"], clean["train"].shape[1], 2)
            initial = copy.deepcopy(template.state_dict())
            fairness = {"manifest_sha256": file_sha256(base["fixed_views"]["manifest"]),
                        "initialization_sha256": _state_hash(initial),
                        "pretrain_order_sha256": sha256_strings([','.join(map(str, order)) for order in pretrain_orders]),
                        "probe_order_sha256": sha256_strings([','.join(map(str, order)) for order in probe_orders])}
            if not _same_fairness(fairness, reference["fairness"], "TEP"):
                raise RuntimeError("TEP rho fairness differs from FINAL")
            metadata = {**fairness, "method": rho_name(rho), "rho": float(rho), "seed": int(seed),
                        "mask_sha256": mask["mask_sha256"], "evaluation_splits": ["validation"],
                        "test_metrics_read": False}
            checkpoint = output / rho_name(rho) / f"seed_{seed}" / "model.pt"
            method = _fit_method(rho_name(rho), augmented, audits, views, clean, stages, initial,
                                 pretrain_orders, probe_orders, runtime, device, checkpoint, metadata,
                                 evaluation_splits=("validation",))
            if "test" in method or method.get("evaluation_splits") != ["validation"]:
                raise RuntimeError("TEP budget diagnostic evaluated test")
            record = {"variant": rho_name(rho), "rho": float(rho), "seed": int(seed),
                      "evaluation_split": "validation", "method": method, "fairness": fairness,
                      "mask_sha256": mask["mask_sha256"], "budget_audit": budget,
                      "test_metrics_read": False, "training": "new_training"}
            _store(manifest_path, results, key, record)
    return results


def validation_metrics(dataset: str, record: dict[str, Any]) -> dict[str, float]:
    if record.get("evaluation_split") != "validation" or record.get("test_metrics_read") is not False:
        raise RuntimeError("selection attempted to read a non-validation record")
    if dataset == "3W":
        metric = record["method"]["metrics"]
        return {"macro_f1": float(metric["macro_f1"]), "auprc": float(metric["auprc_multiclass_macro"]),
                "far": float(metric["far"]), "early_recall": float(metric["early_recall"]),
                "detection_delay": float(metric["mean_detection_delay_seconds"])}
    if "test" in record["method"]:
        raise RuntimeError("TEP selection record contains test")
    current = record["method"]["validation"]
    return {"macro_f1": float(current["metrics"]["macro_f1"]), "auprc": float(current["metrics"]["auprc"]),
            "far": float(current["metrics"]["far"]), "early_recall": float(current["early_fault"]["recall"]),
            "detection_delay": float(current["detection_delay"]["mean_delay_samples"])}


def select_stage2(config: dict[str, Any]) -> dict[str, Any]:
    records = {"3W": _manifest(Path(config["three_w"]["output_dir"]) / "manifest.json"),
               "TEP": _manifest(Path(config["tep"]["output_dir"]) / "manifest.json")}
    seed = {"3W": int(config["three_w"]["stage1_seed"]), "TEP": int(config["tep"]["stage1_seed"])}
    metrics = {dataset: {float(rho): validation_metrics(dataset, records[dataset][f"{rho_name(rho)}|{seed[dataset]}"])
                         for rho in map(float, config["rhos"])} for dataset in ("3W", "TEP")}
    low = max(map(float, config["stage2_selection"]["low_pool"]),
              key=lambda rho: (metrics["TEP"][rho]["macro_f1"], -metrics["TEP"][rho]["far"], -rho))
    reference = {dataset: max(metrics[dataset][rho]["macro_f1"] for rho in metrics[dataset]) for dataset in metrics}
    intermediate = max(map(float, config["stage2_selection"]["intermediate_pool"]),
                       key=lambda rho: (np.mean([metrics[d][rho]["macro_f1"] / max(reference[d], 1e-12) for d in metrics]),
                                        -np.mean([metrics[d][rho]["far"] for d in metrics]), -rho))
    selected = sorted(set([low, intermediate, float(config["stage2_selection"]["fixed_reference"])]))
    if len(selected) > int(config["stage2_selection"]["maximum_candidates"]):
        raise RuntimeError("Stage 2 selected too many rho values")
    payload = {"selection_split": "validation", "test_metrics_read": False, "selected_rhos": selected,
               "low_selected": low, "intermediate_selected": intermediate,
               "rules": config["stage2_selection"], "stage1_metrics": metrics}
    write_json(Path(config["output"]["selection"]), payload); return payload


def run(config: dict[str, Any], data_root: Path, stage_name: str, dataset: str) -> dict[str, Any]:
    _validate(config)
    base = yaml.safe_load(Path(config["tep"]["base_config"]).read_text(encoding="utf-8"))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", str(base.get("cublas_workspace_config", ":4096:8")))
    device = select_device(str(config["device"])); datasets = ("3w", "tep") if dataset == "both" else (dataset,)
    if stage_name in ("1", "all"):
        rhos = list(map(float, config["rhos"]))
        if "3w" in datasets: run_three_w(config, data_root, rhos, [int(config["three_w"]["stage1_seed"])], device)
        tep_context = _tep_context(config) if "tep" in datasets else None
        if "tep" in datasets: run_tep(config, tep_context, rhos, [int(config["tep"]["stage1_seed"])], device)
    selection = select_stage2(config)
    if stage_name in ("2", "all"):
        rhos = list(map(float, selection["selected_rhos"]))
        if "3w" in datasets: run_three_w(config, data_root, rhos, list(map(int, config["three_w"]["stage2_seeds"])), device)
        tep_context = _tep_context(config) if "tep" in datasets else None
        if "tep" in datasets: run_tep(config, tep_context, rhos, list(map(int, config["tep"]["stage2_seeds"])), device)
    counts = {name: len(_manifest(Path(config[key]["output_dir"]) / "manifest.json"))
              for name, key in (("3W", "three_w"), ("TEP", "tep"))}
    write_json(Path(config["output"]["manifest"]), {"stage": stage_name, "records": counts,
               "selected_rhos": selection["selected_rhos"], "test_metrics_read": False})
    return {"stage": stage_name, "records": counts, "selected_rhos": selection["selected_rhos"]}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/budget_shrinkage_diagnostic.yaml")
    parser.add_argument("--data-root", type=Path, required=True); parser.add_argument("--stage", choices=("1", "2", "all"), default="all")
    parser.add_argument("--dataset", choices=("3w", "tep", "both"), default="both")
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    print(json.dumps(run(config, args.data_root, args.stage, args.dataset), ensure_ascii=False))


if __name__ == "__main__":
    main()
