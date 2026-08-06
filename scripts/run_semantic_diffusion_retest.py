from __future__ import annotations

import argparse
import copy
import csv
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from diffusion import DiffusionSchedule
from diffusion.semantic_augmentation import SemanticPartialDiffusion1D
from metrics import representation_diagnostics, select_binary_threshold
from scripts.audit_semantic_diffusion_augmentation import (
    augmentation_metrics,
    bases,
    generate_repeats,
    teacher_arrays,
    traditional_augmentation,
)
from scripts.run_diffusion_quality_retest import (
    _fit_probe,
    _fit_supcon,
    _metrics,
    _probabilities,
    _state_hash,
    epoch_orders,
    load_fixed_views,
    best_probe_record,
)
from diffusion.fixed_views import sha256_strings
from trainers import build_model
from utils import environment_metadata, seed_everything, write_json


METHODS = ("B0 传统增强", "B1 普通部分扩散增强", "B2 语义约束部分扩散增强")


def audit_allows_retest(audit: dict[str, Any]) -> bool:
    return bool(audit.get("downstream_retest_allowed", False))


def b2_has_positive_signal(b1: dict[str, Any], b2: dict[str, Any]) -> bool:
    first, second = b1["metrics"], b2["metrics"]
    maintained = (second["fault_recall"] >= first["fault_recall"] - .01
                  and second["auprc"] >= first["auprc"] - .005)
    improvement = (second["macro_f1"] > first["macro_f1"]
                   or second["far"] <= first["far"] - .02)
    return bool(maintained and improvement)


def semantic_validity_mask(
    base: np.ndarray, augmented: np.ndarray, base_probability: np.ndarray,
    base_feature: np.ndarray, probability: np.ndarray, feature: np.ndarray,
    thresholds: dict[str, float],
) -> np.ndarray:
    probability_kl = (base_probability * (np.log(np.maximum(base_probability, 1e-12))
                                           - np.log(np.maximum(probability, 1e-12)))).sum(1)
    cosine = (base_feature * feature).sum(1) / np.maximum(
        np.linalg.norm(base_feature, axis=1) * np.linalg.norm(feature, axis=1), 1e-12)
    scale = np.maximum(base.std(axis=(1, 2)), 1e-6)
    diversity = np.abs(augmented - base).mean(axis=(1, 2)) / scale
    return (np.isfinite(augmented).all(axis=(1, 2))
            & (probability_kl <= float(thresholds["maximum_probability_kl"]))
            & (cosine >= float(thresholds["minimum_feature_cosine"]))
            & (diversity >= float(thresholds["minimum_normalized_l1"]))
            & (diversity <= float(thresholds["maximum_normalized_l1"])))


def gated_views(
    base: np.ndarray, first: np.ndarray, second: np.ndarray, fallback: np.ndarray,
    teacher: torch.nn.Module, batch_size: int, device: str, thresholds: dict[str, float],
) -> tuple[np.ndarray, dict[str, float]]:
    base_probability, base_feature = teacher_arrays(teacher, base, batch_size, device)
    first_probability, first_feature = teacher_arrays(teacher, first, batch_size, device)
    second_probability, second_feature = teacher_arrays(teacher, second, batch_size, device)
    first_valid = semantic_validity_mask(base, first, base_probability, base_feature,
                                         first_probability, first_feature, thresholds)
    second_valid = semantic_validity_mask(base, second, base_probability, base_feature,
                                          second_probability, second_feature, thresholds)
    result = fallback.copy()
    result[second_valid] = second[second_valid]
    result[first_valid] = first[first_valid]
    return result, {
        "first_acceptance_rate": float(first_valid.mean()),
        "resample_acceptance_rate": float((~first_valid & second_valid).mean()),
        "traditional_fallback_rate": float((~first_valid & ~second_valid).mean()),
    }


def _fault_type(run_uid: str) -> int:
    match = re.search(r":fault_(\d+):", str(run_uid))
    return int(match.group(1)) if match else 0


def subgroup_metrics(bundle: dict[str, np.ndarray], scores: np.ndarray,
                     threshold: float) -> dict[str, Any]:
    prediction = scores >= threshold
    labels = bundle["labels"]
    types = np.asarray([_fault_type(value) for value in bundle["run_uid"]])
    normal, fault = labels == 0, labels != 0
    result: dict[str, Any] = {
        "normal": {"count": int(normal.sum()), "far": float(prediction[normal].mean())},
        "fault": {"count": int(fault.sum()), "fault_recall": float(prediction[fault].mean())},
        "fault_types": {},
    }
    for kind in sorted(set(types) - {0}):
        selector = (types == kind) & fault
        result["fault_types"][str(kind)] = {
            "count": int(selector.sum()), "fault_recall": float(prediction[selector].mean()),
            "mean_fault_score": float(scores[selector].mean()),
        }
    if "start_sample" in bundle:
        early = fault & (bundle["start_sample"] <= 289)
        result["early_fault"] = {"count": int(early.sum()),
                                 "fault_recall": float(prediction[early].mean()) if early.any() else None}
    else:
        result["early_fault"] = {"count": 0, "fault_recall": None}
    return result


def _generator(config: dict[str, Any], checkpoint: str, channels: int, device: str) -> SemanticPartialDiffusion1D:
    settings = config["generator"]
    model = SemanticPartialDiffusion1D(channels, int(settings["semantic_dimension"]),
                                       int(settings["hidden_channels"]), int(settings["hidden_channels"]),
                                       int(settings["residual_blocks"])).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    return model.eval()


def _fit_method(
    name: str, augmented: dict[str, np.ndarray], base: dict[str, np.ndarray],
    views: dict[str, dict[str, np.ndarray]], initial_state: dict[str, torch.Tensor],
    pretrain_orders: list[np.ndarray], probe_orders: list[np.ndarray],
    config: dict[str, Any], device: str, checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    seed = int(config["random_seed"])
    seed_everything(seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = build_model(config["model"], base["train"].shape[1], 2).to(device)
    model.load_state_dict(initial_state)
    train, validation = dict(views["train"]), dict(views["validation"])
    train["clean"], train["restored"] = base["train"], augmented["train"]
    validation["clean"], validation["restored"] = base["validation"], augmented["validation"]
    weights_train = np.ones(len(train["labels"]), np.float32)
    weights_validation = np.ones(len(validation["labels"]), np.float32)
    pretrain = _fit_supcon(model, train, validation, weights_train, weights_validation,
                           pretrain_orders, config, device)
    probe_train, probe_validation = dict(train), dict(validation)
    probe_train["clean"] = base["train"]
    probe_validation["restored"] = base["validation"]
    seed_everything(seed + 1)
    probe = _fit_probe(model, probe_train, probe_validation, probe_orders, config, device)
    validation_probability, _ = _probabilities(model, base["validation"], int(config["batch_size"]), device)
    threshold = select_binary_threshold(views["validation"]["labels"], validation_probability[:, 1])
    probability, test_embedding = _probabilities(model, base["test"], int(config["batch_size"]), device)
    _, augmented_embedding = _probabilities(model, augmented["test"], int(config["batch_size"]), device)
    score = probability[:, 1]
    diagnostics = representation_diagnostics(test_embedding, augmented_embedding, views["test"]["labels"])
    if checkpoint_path is not None:
        if checkpoint_path.exists():
            raise FileExistsError(f"refusing to overwrite classifier checkpoint: {checkpoint_path}")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
    return {
        "metrics": _metrics(views["test"]["labels"], score, threshold),
        "representation": {key: diagnostics[key] for key in ("fisher_ratio", "class_center_shift", "effective_rank")},
        "subgroups": subgroup_metrics(views["test"], score, threshold),
        "validation_threshold": threshold,
        "initialization_sha256": _state_hash(initial_state),
        "sample_total_weight": 1.0,
        "pretrain_history": pretrain,
        "probe_history": probe,
        "best_pretrain_epoch": int(min(pretrain, key=lambda row: row["validation_supcon_loss"])["epoch"]),
        "best_probe_epoch": int(best_probe_record(probe)["epoch"]),
        "best_probe_validation_threshold": float(best_probe_record(probe)["validation_threshold"]),
        "training_seconds": time.perf_counter() - started,
        "peak_gpu_mib": torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0,
        "method": name,
    }


def run(config: dict[str, Any]) -> dict[str, Any]:
    audit = json.loads(Path(config["audit_result"]).read_text(encoding="utf-8"))
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if not audit_allows_retest(audit):
        result = {
            "markers": config["markers"], "status": "SEMANTIC_DIFFUSION_AUGMENTATION_NO_GO",
            "training_skipped": True, "reason": "第一级语义—多样性可行域审计未通过",
            **environment_metadata(),
        }
        write_json(output / "result.json", result)
        return result

    started = time.perf_counter()
    device, seed = str(config["device"]), int(config["random_seed"])
    views, fixed_manifest = load_fixed_views(config)
    base = bases(views)
    teacher = build_model(config["teacher_model"], base["train"].shape[1], 2).to(device)
    teacher.load_state_dict(torch.load(config["teacher_checkpoint"], map_location=device, weights_only=True))
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    semantic = {split: teacher_arrays(teacher, values, int(config["batch_size"]), device)[1]
                for split, values in base.items()}
    checkpoints = audit["generator_checkpoints"]
    generators = {"G0": _generator(config, checkpoints["G0"], base["train"].shape[1], device),
                  "G1": _generator(config, checkpoints["G1"], base["train"].shape[1], device)}
    schedule = DiffusionSchedule.cosine(int(config["generator"]["diffusion_steps"]), device)
    t_aug = int(audit["selected_t_aug"])
    clip_min, clip_max = base["train"].min((0, 2)), base["train"].max((0, 2))
    augmented: dict[str, dict[str, np.ndarray]] = {name: {} for name in METHODS}
    for split in ("train", "validation", "test"):
        augmented[METHODS[0]][split] = traditional_augmentation(
            base[split], views[split]["window_id"], config["traditional_augmentation"], seed)
        for method, generator_name in ((METHODS[1], "G0"), (METHODS[2], "G1")):
            augmented[method][split] = generate_repeats(
                generators[generator_name], base[split], views[split]["observation"], semantic[split],
                schedule, t_aug, 1, int(config["batch_size"]), device, seed, clip_min, clip_max,
            )[0]

    pretrain_orders = epoch_orders(len(base["train"]), int(config["epochs"]), seed + 10_000)
    probe_orders = epoch_orders(len(base["train"]), int(config["probe_epochs"]), seed + 20_000)
    pretrain_order_hash = sha256_strings([",".join(map(str, order.tolist())) for order in pretrain_orders])
    probe_order_hash = sha256_strings([",".join(map(str, order.tolist())) for order in probe_orders])
    seed_everything(seed)
    template = build_model(config["model"], base["train"].shape[1], 2)
    initial_state = copy.deepcopy(template.state_dict())
    results = {name: _fit_method(name, augmented[name], base, views, initial_state,
                                 pretrain_orders, probe_orders, config, device) for name in METHODS}

    b3_status: dict[str, Any] = {"status": "SKIPPED_REQUIRES_B2_POSITIVE_SIGNAL"}
    if b2_has_positive_signal(results[METHODS[1]], results[METHODS[2]]):
        b3_views: dict[str, np.ndarray] = {}
        gate_diagnostics: dict[str, Any] = {}
        for split in ("train", "validation", "test"):
            second = generate_repeats(
                generators["G1"], base[split], views[split]["observation"], semantic[split], schedule,
                t_aug, 1, int(config["batch_size"]), device, seed + 500_000, clip_min, clip_max,
            )[0]
            b3_views[split], gate_diagnostics[split] = gated_views(
                base[split], augmented[METHODS[2]][split], second, augmented[METHODS[0]][split],
                teacher, int(config["batch_size"]), device, config["semantic_gate"],
            )
        augmented["B3 语义有效性门控"] = b3_views
        results["B3 语义有效性门控"] = _fit_method(
            "B3 语义有效性门控", augmented["B3 语义有效性门控"], base, views, initial_state,
            pretrain_orders, probe_orders, config, device,
        )
        results["B3 语义有效性门控"]["gate_diagnostics"] = gate_diagnostics
        b3_status = {"status": "RUN", "gate_diagnostics": gate_diagnostics}

    test_probability, test_feature = teacher_arrays(teacher, base["test"], int(config["batch_size"]), device)
    for name, value in results.items():
        value["augmentation_audit"] = augmentation_metrics(
            base["test"], augmented[name]["test"][None], views["test"]["labels"], views["test"]["run_uid"],
            test_probability, test_feature, teacher, int(config["batch_size"]), device,
        )

    b0, b1, b2 = (results[name] for name in METHODS)
    checks = {
        "b2_semantic_consistency_above_b1": audit["test_selected_t_only"]["G1"]["teacher_consistency"] > audit["test_selected_t_only"]["G0"]["teacher_consistency"],
        "b2_fault_flip_below_b1": audit["test_selected_t_only"]["G1"]["fault_to_normal_flip"] < audit["test_selected_t_only"]["G0"]["fault_to_normal_flip"],
        "b2_macro_f1_above_b1": b2["metrics"]["macro_f1"] > b1["metrics"]["macro_f1"],
        "b2_far_below_b1": b2["metrics"]["far"] < b1["metrics"]["far"],
        "b2_recall_auprc_maintained": b2["metrics"]["fault_recall"] >= b1["metrics"]["fault_recall"] - .01 and b2["metrics"]["auprc"] >= b1["metrics"]["auprc"] - .005,
        "b2_at_least_b0_overall": b2["metrics"]["macro_f1"] >= b0["metrics"]["macro_f1"] and b2["metrics"]["auprc"] >= b0["metrics"]["auprc"] - .005,
        "fair_initialization_batch_order_and_weight": len({value["initialization_sha256"] for value in results.values()}) == 1 and all(value["sample_total_weight"] == 1.0 for value in results.values()),
        "nonzero_diversity": audit["test_selected_t_only"]["G1"]["normalized_l1"] >= float(config["semantic_gate"]["minimum_normalized_l1"]),
    }
    status = "SEMANTIC_DIFFUSION_AUGMENTATION_GO" if sum(checks.values()) >= 6 and checks["fair_initialization_batch_order_and_weight"] else "SEMANTIC_DIFFUSION_AUGMENTATION_NO_GO"
    result = {
        "markers": config["markers"], "status": status, "training_skipped": False,
        **environment_metadata(), "fixed_view_manifest": fixed_manifest,
        "selected_t_aug": t_aug, "selection_split": "validation", "results": results,
        "b3": b3_status, "gate_checks": checks,
        "fairness": {"same_split_seed_initialization_batch_order_optimizer_lr_epochs_temperature_projection_probe_threshold_test": True,
                     "initialization_sha256": _state_hash(initial_state),
                     "pretrain_batch_order_sha256": pretrain_order_hash, "probe_batch_order_sha256": probe_order_hash,
                     "sample_total_weight": 1.0, "only_variable": "positive augmentation view source"},
        "total_seconds": time.perf_counter() - started,
    }
    write_json(output / "result.json", result)
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = ["method", "macro_f1", "auprc", "fault_recall", "far", "auroc", "training_seconds", "peak_gpu_mib"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for name, value in results.items():
            writer.writerow({"method": name, **{key: value["metrics"][key] for key in ("macro_f1", "auprc", "fault_recall", "far", "auroc")},
                             "training_seconds": value["training_seconds"], "peak_gpu_mib": value["peak_gpu_mib"]})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/semantic_diffusion_augmentation_retest.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = run(config)
    print(json.dumps({"status": result["status"], "training_skipped": result["training_skipped"],
                      "b3": result.get("b3")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
