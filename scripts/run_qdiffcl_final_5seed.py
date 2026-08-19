from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from diffusion import DiffusionSchedule, FrequencyForwardDiffusion, fit_spectral_statistics
from metrics import representation_diagnostics
from scripts.diagnose_frequency_selective_far import score_profile
from scripts.run_3w_diffusion_1seed import METHODS as THREE_W_METHODS, run as run_three_w_model
from scripts.run_diffusion_quality_retest import _metrics, _probabilities, _state_hash, epoch_orders
from scripts.run_frequency_selective_r1_3seed import _fit_method, _stage_metrics, file_sha256, sha256_strings
from scripts.run_r1_des_ablation import build_masks
from scripts.run_stage_frequency_diffusion_mvp import _runtime, augmentation_mechanism_metrics, detection_delays, early_fault_recall
from trainers import build_model
from utils import seed_everything, select_device, write_json


METHODS = ("UNIFORM", "CURRENT_R1", "FINAL_QDIFFCL")


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_config(config: dict[str, Any]) -> None:
    final = yaml.safe_load(Path(config["final_config"]).read_text(encoding="utf-8"))
    expected_weights = {"weight_discriminative": .5, "weight_early": .5, "weight_run_stability": 0.0}
    if not final.get("frozen") or final.get("weights") != expected_weights:
        raise ValueError("FINAL_QDIFFCL must remain frozen at 0.5D+0.5E")
    if list(map(int, config["three_w"]["seeds"])) != [42, 43, 44, 45, 46]:
        raise ValueError("3W final reliability seeds changed")
    if list(map(int, config["tep"]["seeds"])) != [7, 42, 43, 44, 2026]:
        raise ValueError("TEP final reliability seeds changed")
    if config["variants"]["FINAL_DE"] != expected_weights:
        raise ValueError("FINAL_DE weights differ from final config")
    frozen = config["spectral_diffusion"]
    expected = {"t_uniform": 3, "t_critical": 1, "t_noncritical": 5,
                "preserve_phase": True, "preserve_dc": True, "noise_structure": "iid"}
    if any(frozen[key] != value for key, value in expected.items()) or config["criticality_base"]["critical_ratio"] != .30:
        raise ValueError("frozen diffusion protocol changed")


def _fairness_subset(record: dict[str, Any], dataset: str) -> dict[str, Any]:
    if dataset == "3W":
        keys = ("initialization_sha256", "window_refs_sha256", "supcon_batch_order_sha256")
    else:
        keys = ("manifest_sha256", "initialization_sha256", "pretrain_order_sha256", "probe_order_sha256")
    return {key: record.get(key) for key in keys}


def _assert_fairness(left: dict[str, Any], right: dict[str, Any], dataset: str, context: str) -> None:
    a, b = _fairness_subset(left, dataset), _fairness_subset(right, dataset)
    # Historical 3W results predate explicit batch-order recording in some manifests.
    for key in tuple(a):
        if a[key] is None or b[key] is None:
            a.pop(key); b.pop(key)
    if a != b:
        raise RuntimeError(f"fairness mismatch {dataset} {context}: {a} != {b}")


def _store(records: dict[str, Any], path: Path, dataset: str, method: str, seed: int,
           method_record: dict[str, Any], fairness: dict[str, Any], source: str, training: str) -> None:
    records[f"{method}|{seed}"] = {
        "dataset": dataset, "method": method, "seed": int(seed), "metrics": method_record,
        "fairness": fairness, "source": source, "training": training,
        "test_used_for_weight_selection": False,
    }
    write_json(path, {"results": records})


def run_three_w(config: dict[str, Any], data_root: Path, masks: dict[str, Any]) -> dict[str, Any]:
    stage = config["three_w"]; output = Path(stage["output_dir"]); manifest_path = output / "manifest.json"
    records = _read(manifest_path).get("results", {}) if manifest_path.exists() else {}
    existing3 = _read(stage["existing_3seed_manifest"]); existing5 = _read(stage["existing_5seed_manifest"])
    base = yaml.safe_load(Path(stage["base_config"]).read_text(encoding="utf-8"))
    for seed in map(int, stage["seeds"]):
        if seed in (42, 43, 44):
            result_path = existing3["seed_results"][str(seed)]["result_path"]
        else:
            result_path = existing5["seed_results"][str(seed)]["uniform_r1_result_path"]
        historical = _read(result_path)
        for method, source_method in (("UNIFORM", THREE_W_METHODS[1]), ("CURRENT_R1", THREE_W_METHODS[2])):
            _store(records, manifest_path, "3W", method, seed, historical["methods"][source_method],
                   historical["fairness"], str(result_path), "reused_existing")

        key = f"FINAL_QDIFFCL|{seed}"
        if key in records:
            continue
        current = copy.deepcopy(base)
        current.update({"seed": seed, "protocol_seed": int(stage["protocol_seed"]),
                        "criticality_source": str(Path(config["output"]["mask_dir"]) / "3w_FINAL_DE.json"),
                        "methods": [THREE_W_METHODS[2]],
                        "output_dir": str(output / "FINAL_QDIFFCL" / f"seed_{seed}"),
                        "evaluation_split": "test"})
        current["training"]["supcon_batching"] = "original"
        training = "new_training"
        if seed in (42, 43, 44):
            source_dir = Path(stage["final_validation_dir"]) / f"seed_{seed}"
            checkpoint = source_dir / f"{THREE_W_METHODS[2]}_model.pt"
            source_result = _read(source_dir / "result.json")
            current["pretrained_checkpoint_by_method"] = {THREE_W_METHODS[2]: str(checkpoint)}
            training = "reused_validation_checkpoint_test_evaluation_only"
        result = run_three_w_model(current, data_root)
        final_method = result["methods"][THREE_W_METHODS[2]]
        _assert_fairness(result["fairness"], historical["fairness"], "3W", f"seed={seed} FINAL/base")
        if seed in (42, 43, 44):
            _assert_fairness(result["fairness"], source_result["fairness"], "3W", f"seed={seed} FINAL/source")
        _store(records, manifest_path, "3W", "FINAL_QDIFFCL", seed, final_method,
               result["fairness"], str(Path(current["output_dir"]) / "result.json"), training)
    return records


def _tep_fairness(base_config: dict[str, Any], clean: dict[str, np.ndarray], runtime: dict[str, Any], seed: int,
                  initial_state: dict[str, torch.Tensor], pretrain_orders: list[np.ndarray],
                  probe_orders: list[np.ndarray]) -> dict[str, Any]:
    return {"manifest_sha256": file_sha256(base_config["fixed_views"]["manifest"]),
            "initialization_sha256": _state_hash(initial_state),
            "pretrain_order_sha256": sha256_strings([','.join(map(str, order)) for order in pretrain_orders]),
            "probe_order_sha256": sha256_strings([','.join(map(str, order)) for order in probe_orders])}


def _evaluate_tep_checkpoint(source_checkpoint: Path, source_metrics: Path, base_config: dict[str, Any],
                             views: dict[str, Any], clean: dict[str, np.ndarray], stages: dict[str, np.ndarray],
                             seed: int, device: str) -> tuple[dict[str, Any], dict[str, Any]]:
    source = _read(source_metrics)
    if source.get("evaluation_splits") != ["validation"] or "test" in source:
        raise RuntimeError("FINAL source checkpoint is not validation-only")
    runtime = _runtime(base_config, seed); runtime["diagnosis"] = {"threshold_band_width": .05, "high_correlation_quantile": .90}
    model = build_model(runtime["model"], clean["train"].shape[1], 2).to(device)
    payload = torch.load(source_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    probability, embedding = _probabilities(model, clean["test"], int(runtime["batch_size"]), device)
    scores = probability[:, 1]; threshold = float(source["validation_threshold"]); prediction = scores >= threshold
    test = {"metrics": _metrics(views["test"]["labels"], scores, threshold),
            "score_profile": score_profile(views["test"]["labels"], scores, threshold,
                                           float(runtime["diagnosis"]["threshold_band_width"])),
            "stages": _stage_metrics(stages["test"], prediction),
            "early_fault": early_fault_recall(prediction, stages["test"]),
            "detection_delay": detection_delays(views["test"], prediction, runtime),
            "representation": representation_diagnostics(embedding, embedding, views["test"]["labels"])}
    return {"method": "FINAL_QDIFFCL", "seed": seed, "validation_threshold": threshold,
            "test": test, "source_validation_metrics": str(source_metrics),
            "reused_checkpoint_source": str(source_checkpoint)}, payload["metadata"]


def run_tep(config: dict[str, Any], context: tuple[Any, ...], masks: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    stage = config["tep"]; base_config, views, clean, stages, _ = context
    output = Path(stage["output_dir"]); manifest_path = output / "manifest.json"
    records = _read(manifest_path).get("results", {}) if manifest_path.exists() else {}
    existing = _read(stage["existing_3seed_result"]); device = select_device(str(config["device"]))
    statistics = fit_spectral_statistics(clean["train"], float(stage["spectral_diffusion"]["clip_quantile"]), "train")
    schedule = DiffusionSchedule.cosine(int(config["spectral_diffusion"]["diffusion_steps"]), device)
    augmenters = {name: FrequencyForwardDiffusion(statistics, schedule.alpha_bars,
                  np.asarray(masks[name]["soft_mask"], np.float32), 3, 1, True, True, device)
                  for name in ("CURRENT_R1", "FINAL_DE")}
    for seed in map(int, stage["seeds"]):
        if seed in (7, 42, 2026):
            baseline = existing["seed_results"][str(seed)]
            for method, source_method in (("UNIFORM", "C1"), ("CURRENT_R1", "R1")):
                _store(records, manifest_path, "TEP", method, seed, baseline["methods"][source_method],
                       baseline["fairness"], str(stage["existing_3seed_result"]), "reused_existing")
            key = f"FINAL_QDIFFCL|{seed}"
            if key not in records:
                source_dir = Path(stage["final_validation_dir"]) / f"seed_{seed}"
                record, fairness = _evaluate_tep_checkpoint(source_dir / "model.pt", source_dir / "metrics.json",
                                                             base_config, views, clean, stages, seed, device)
                _assert_fairness(fairness, baseline["fairness"], "TEP", f"seed={seed} FINAL/base")
                _store(records, manifest_path, "TEP", "FINAL_QDIFFCL", seed, record, fairness,
                       str(source_dir / "model.pt"), "reused_validation_checkpoint_test_evaluation_only")
            continue

        runtime = _runtime(base_config, seed); runtime["diagnosis"] = {"threshold_band_width": .05, "high_correlation_quantile": .90}
        pretrain_orders = epoch_orders(len(clean["train"]), int(runtime["epochs"]), seed + 10000)
        probe_orders = epoch_orders(len(clean["train"]), int(runtime["probe_epochs"]), seed + 20000)
        seed_everything(seed); template = build_model(runtime["model"], clean["train"].shape[1], 2)
        initial_state = copy.deepcopy(template.state_dict())
        fairness = _tep_fairness(base_config, clean, runtime, seed, initial_state, pretrain_orders, probe_orders)
        for method, mode, mask_name in (("UNIFORM", "uniform", "CURRENT_R1"),
                                        ("CURRENT_R1", "selective", "CURRENT_R1"),
                                        ("FINAL_QDIFFCL", "selective", "FINAL_DE")):
            key = f"{method}|{seed}"
            if key in records:
                continue
            augmented, audits = {}, {}
            for split, offset in (("train", 0), ("validation", 100), ("test", 200)):
                sampling_seed = seed + int(stage["spectral_diffusion"]["sampling_seed_offset"]) + offset
                if mode == "uniform":
                    augmented[split], diagnostics = augmenters[mask_name].augment(
                        clean[split], mode, sampling_seed, batch_size=int(base_config["training"]["batch_size"]),
                        noise_structure="iid")
                else:
                    augmented[split], diagnostics = augmenters[mask_name].augment(
                        clean[split], mode, sampling_seed, 5, int(base_config["training"]["batch_size"]),
                        noise_structure="iid")
                audits[split] = augmentation_mechanism_metrics(
                    clean[split], augmented[split], views[split]["labels"], stages[split],
                    np.asarray(masks[mask_name]["hard_mask"], bool), diagnostics)
            metadata = {**fairness, "method": method, "seed": seed, "mask_sha256": masks[mask_name]["mask_sha256"],
                        "final_weights_frozen": True, "test_used_for_weight_selection": False}
            checkpoint = output / method / f"seed_{seed}" / "model.pt"
            record = _fit_method(method, augmented, audits, views, clean, stages, initial_state,
                                 pretrain_orders, probe_orders, runtime, device, checkpoint, metadata)
            _store(records, manifest_path, "TEP", method, seed, record, fairness,
                   str(checkpoint.parent / "metrics.json"), "new_training")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/qdiffcl_final_5seed.yaml")
    parser.add_argument("--data-root", type=Path, required=True); parser.add_argument("--dataset", choices=("3w", "tep", "both"), default="both")
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")); validate_config(config)
    three_masks, tep_context, audit = build_masks(config, args.data_root)
    final = yaml.safe_load(Path(config["final_config"]).read_text(encoding="utf-8"))
    for dataset in ("3W", "TEP"):
        if audit[dataset]["FINAL_DE"]["mask_sha256"] != final["mask_sha256"][dataset]:
            raise RuntimeError(f"FINAL mask hash changed on {dataset}")
    datasets = {}
    if args.dataset in ("3w", "both"): datasets["3W"] = run_three_w(config, args.data_root, three_masks)
    if args.dataset in ("tep", "both"): datasets["TEP"] = run_tep(config, tep_context, tep_context[-1], audit)
    record_counts = {}
    for dataset, key in (("3W", "three_w"), ("TEP", "tep")):
        dataset_manifest = Path(config[key]["output_dir"]) / "manifest.json"
        if dataset_manifest.exists():
            record_counts[dataset] = len(_read(dataset_manifest).get("results", {}))
    write_json(Path(config["output"]["manifest"]), {"stage": "FINAL_QDIFFCL_5SEED", "weights": [0.5, 0.5, 0.0],
               "seeds": {"3W": config["three_w"]["seeds"], "TEP": config["tep"]["seeds"]},
               "records": record_counts,
               "test_used_for_weight_selection": False, "weights_reopened": False})
    print(json.dumps({"datasets": {key: len(value) for key, value in datasets.items()}, "weights": [0.5, .5, 0]}, ensure_ascii=False))


if __name__ == "__main__": main()
