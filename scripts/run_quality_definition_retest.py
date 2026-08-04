from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from scripts.audit_quality_definition import (build_quality_candidates,
                                              fault_type, split_factors)
from scripts.run_diffusion_quality_retest import (
    _fit_probe, _fit_supcon, _metrics, _probabilities, _state_hash,
    epoch_orders, fit_train_only_quality, load_fixed_views,
)
from metrics import representation_diagnostics, select_binary_threshold
from trainers import build_model
from utils import environment_metadata, seed_everything, write_json


def within_class_performance(labels: np.ndarray, scores: np.ndarray, quality: np.ndarray,
                             threshold: float) -> dict[str, Any]:
    prediction = scores >= threshold; result = {}
    for class_name, selector in (("normal", labels == 0), ("fault", labels != 0)):
        indices = np.flatnonzero(selector); ordered = indices[np.argsort(quality[indices], kind="stable")]
        low, high = ordered[:len(ordered) // 2], ordered[len(ordered) // 2:]
        groups = {}
        for name, chosen in (("low", low), ("high", high)):
            groups[name] = {
                "count": int(len(chosen)), "mean_q": float(quality[chosen].mean()),
                "accuracy": float(np.mean(prediction[chosen] == labels[chosen])),
                "mean_fault_score": float(scores[chosen].mean()),
            }
        result[class_name] = groups
    return result


def fault_type_performance(bundle: dict[str, np.ndarray], scores: np.ndarray,
                           threshold: float) -> dict[str, Any]:
    kinds = np.asarray([fault_type(value) for value in bundle["run_uid"]]); prediction = scores >= threshold
    result = {}
    for kind in sorted(set(kinds) - {0}):
        selector = (kinds == kind) & (bundle["labels"] != 0)
        result[str(int(kind))] = {"count": int(selector.sum()), "fault_recall": float(prediction[selector].mean()),
                                  "mean_fault_score": float(scores[selector].mean())}
    return result


def run(config: dict[str, Any]) -> dict[str, Any]:
    audit_path = Path(config["output_dir"]) / "quality_definition_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    output = Path(config["retest_output_dir"]); output.mkdir(parents=True, exist_ok=True)
    if not audit["training_retest_allowed"]:
        result = {"markers": config["markers"], "status": "QUALITY_DEFINITION_AUDIT_NO_GO",
                  "training_skipped": True, "reason": "Q1/Q2 均未通过 train-only 质量偏置审计",
                  "passing_candidates": [], **environment_metadata()}
        write_json(output / "result.json", result)
        return result

    started = time.perf_counter(); views, _ = load_fixed_views(config); seed = int(config["random_seed"]); device = str(config["device"])
    legacy = {"quality": {"formula": "exp(-masked_mae/scale)", "scale_estimator": config["quality"]["q0"]["scale_estimator"],
                          "q_min": config["quality"]["q_min"]}}
    q0, _ = fit_train_only_quality(views, legacy)
    teacher = build_model(config["model"], views["train"]["clean"].shape[1], 2).to(device)
    teacher.load_state_dict(torch.load(config["teacher_checkpoint"], map_location=device, weights_only=True)); teacher.eval()
    factors = {split: split_factors(views[split], split, teacher, int(config["batch_size"]), device) for split in ("train", "validation", "test")}
    qualities, _ = build_quality_candidates(factors, q0, config)

    names = [("Hard SupCon", "hard") , ("Q0 Abs-MAE", "q0_abs")]
    if "q1_relative" in audit["passing_candidates"]: names.append(("Q1 Relative Gain", "q1_relative"))
    if "q2_relative_semantic" in audit["passing_candidates"]: names.append(("Q2 Relative + Semantic", "q2_relative_semantic"))
    pretrain_orders = epoch_orders(len(views["train"]["labels"]), int(config["epochs"]), seed + 10_000)
    probe_orders = epoch_orders(len(views["train"]["labels"]), int(config["probe_epochs"]), seed + 20_000)
    seed_everything(seed); template = build_model(config["model"], views["train"]["clean"].shape[1], 2)
    initial_state = copy.deepcopy(template.state_dict()); results = {}
    for display, key in names:
        seed_everything(seed); method_started = time.perf_counter()
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        model = build_model(config["model"], views["train"]["clean"].shape[1], 2).to(device); model.load_state_dict(initial_state)
        train_q = np.ones(len(views["train"]["labels"]), np.float32) if key == "hard" else qualities[key]["train"]
        val_q = np.ones(len(views["validation"]["labels"]), np.float32) if key == "hard" else qualities[key]["validation"]
        pretrain = _fit_supcon(model, views["train"], views["validation"], train_q, val_q, pretrain_orders, config, device)
        seed_everything(seed + 1); probe = _fit_probe(model, views["train"], views["validation"], probe_orders, config, device)
        val_probability, _ = _probabilities(model, views["validation"]["restored"], int(config["batch_size"]), device)
        threshold = select_binary_threshold(views["validation"]["labels"], val_probability[:, 1])
        probability, restored_embedding = _probabilities(model, views["test"]["restored"], int(config["batch_size"]), device)
        _, clean_embedding = _probabilities(model, views["test"]["clean"], int(config["batch_size"]), device)
        score = probability[:, 1]; metrics = _metrics(views["test"]["labels"], score, threshold)
        diagnostics = representation_diagnostics(clean_embedding, restored_embedding, views["test"]["labels"])
        test_q = np.ones(len(score), np.float32) if key == "hard" else qualities[key]["test"]
        results[display] = {
            "quality_key": key, "metrics": metrics, "validation_threshold": threshold,
            "initialization_sha256": _state_hash(initial_state), "pretrain_history": pretrain, "probe_history": probe,
            "mean_gradient_norm": float(np.mean([item["mean_gradient_norm"] for item in pretrain])),
            "representation": {name: diagnostics[name] for name in ("fisher_ratio", "class_center_shift", "effective_rank")},
            "within_class_quality_groups": within_class_performance(views["test"]["labels"], score, test_q, threshold),
            "fault_type_performance": fault_type_performance(views["test"], score, threshold),
            "training_seconds": time.perf_counter() - method_started,
            "peak_gpu_mib": torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0,
        }
    result = {"markers": config["markers"], "status": "QUALITY_DEFINITION_RETEST_COMPLETED", **environment_metadata(),
              "passing_candidates": audit["passing_candidates"], "results": results,
              "fairness": {"same_fixed_views": True, "same_initialization": True, "same_batch_order": True,
                           "same_optimizer_epochs_temperature_probe": True}, "total_seconds": time.perf_counter() - started}
    write_json(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/quality_definition_audit.yaml")
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = run(config); print(json.dumps({"status": result["status"], "training_skipped": result.get("training_skipped", False)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
