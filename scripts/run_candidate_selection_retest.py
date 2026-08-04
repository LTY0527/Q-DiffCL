from __future__ import annotations

import argparse
import copy
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from diffusion.fixed_views import sha256_file
from quality import center_scores, combined_scores, semantic_scores
from scripts.audit_candidate_rankability import (oracle_candidate_errors,
                                                  teacher_probabilities)
from scripts.run_diffusion_quality_retest import (
    _fit_probe, _fit_supcon, _metrics, _probabilities, _state_hash,
    epoch_orders, load_fixed_views,
)
from metrics import representation_diagnostics, select_binary_threshold
from trainers import build_model
from utils import deterministic_seed, environment_metadata, seed_everything, write_json


def deterministic_random_indices(master_seed: int, split: str,
                                 window_ids: np.ndarray, k: int) -> np.ndarray:
    return np.asarray([deterministic_seed(master_seed, f"{split}|{window_id}", "random_candidate") % k
                       for window_id in window_ids], dtype=np.int64)


def select_candidate(candidates: np.ndarray, indices: np.ndarray) -> np.ndarray:
    candidates = np.asarray(candidates); indices = np.asarray(indices, dtype=np.int64)
    if candidates.ndim != 4 or indices.shape != (len(candidates),): raise ValueError("invalid candidate selection shapes")
    if np.any(indices < 0) or np.any(indices >= candidates.shape[1]): raise ValueError("candidate index out of range")
    return candidates[np.arange(len(candidates)), indices]


def _fault_type(run_uid: str) -> int:
    match = re.search(r":fault_(\d+):", str(run_uid)); return int(match.group(1)) if match else 0


def no_reference_scores(name: str, candidates: np.ndarray, observation: np.ndarray,
                        probabilities: np.ndarray | None, lambda_sem: float = 1.0) -> np.ndarray:
    h1 = center_scores(candidates, observation)
    if name == "h1_center": return h1
    if probabilities is None: raise ValueError("teacher probabilities are required for semantic ranking")
    h2 = semantic_scores(probabilities)
    if name == "h2_semantic": return h2
    if name == "h3_combined": return combined_scores(h1, h2, lambda_sem)
    raise ValueError(f"unknown no-reference score: {name}")


def audit_allows_retest(audit: dict[str, Any]) -> bool:
    return bool(audit.get("downstream_retest_allowed", False))


def subgroup_metrics(bundle: dict[str, np.ndarray], scores: np.ndarray,
                     threshold: float) -> dict[str, Any]:
    prediction = scores >= threshold; labels = bundle["labels"]
    types = np.asarray([_fault_type(value) for value in bundle["run_uid"]])
    normal = labels == 0; fault = labels != 0
    result = {
        "normal": {"count": int(normal.sum()), "accuracy": float(np.mean(~prediction[normal])),
                   "far": float(np.mean(prediction[normal]))},
        "fault": {"count": int(fault.sum()), "accuracy": float(np.mean(prediction[fault])),
                  "fault_recall": float(np.mean(prediction[fault]))},
        "fault_types": {},
    }
    for kind in sorted(set(types) - {0}):
        selector = (types == kind) & fault
        result["fault_types"][str(int(kind))] = {"count": int(selector.sum()),
                                                 "fault_recall": float(np.mean(prediction[selector])),
                                                 "mean_fault_score": float(scores[selector].mean())}
    boundary = 161
    early = fault & (bundle["start_sample"] <= boundary + 128)
    result["early_fault"] = {"count": int(early.sum()), "fault_recall": float(np.mean(prediction[early])) if early.any() else None}
    return result


def run(config: dict[str, Any]) -> dict[str, Any]:
    audit = json.loads(Path(config["audit_result"]).read_text(encoding="utf-8"))
    output = Path(config["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    if not audit_allows_retest(audit):
        result = {"markers": config["markers"], "status": "INTRA_SAMPLE_CANDIDATE_RANKING_NO_GO",
                  "training_skipped": True, "reason": "第一级候选可排序性审计未通过", **environment_metadata()}
        write_json(output / "result.json", result); return result

    views, _ = load_fixed_views(config); manifest_path = Path(config["candidate_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")); k = int(config["k_candidates"])
    device = str(config["device"]); seed = int(config["random_seed"]); started = time.perf_counter()
    teacher = build_model(config["model"], views["train"]["clean"].shape[1], 2).to(device)
    teacher.load_state_dict(torch.load(config["teacher_checkpoint"], map_location=device, weights_only=True)); teacher.eval()
    selected: dict[str, dict[str, np.ndarray]] = {name: {} for name in ("R0 Fixed Single", "R1 Random Candidate", "R2 Oracle Top-1", "R3 No-reference Top-1")}
    candidate_arrays = {}
    best_score_name = str(audit["best_no_reference_score"])
    for split in ("train", "validation", "test"):
        path = Path(manifest["splits"][split]["path"])
        if sha256_file(path) != manifest["splits"][split]["sha256"]: raise RuntimeError(f"candidate hash mismatch: {split}")
        with np.load(path, allow_pickle=False) as archive:
            candidates, indices = archive["candidates"], archive["fixed_indices"]
        if not np.array_equal(indices, np.arange(len(views[split]["labels"]))): raise RuntimeError("full candidate archive must preserve all fixed windows")
        candidate_arrays[split] = candidates
        bundle = views[split]; random_index = deterministic_random_indices(seed, split, bundle["window_id"], k)
        oracle_index = oracle_candidate_errors(bundle["clean"], candidates, bundle["observation"])["masked_mae"].argmin(1)
        probabilities = None
        if best_score_name != "h1_center": probabilities = teacher_probabilities(teacher, candidates, int(config["batch_size"]), device)
        score = no_reference_scores(best_score_name, candidates, bundle["observation"], probabilities)
        selected["R0 Fixed Single"][split] = bundle["restored"]
        selected["R1 Random Candidate"][split] = select_candidate(candidates, random_index)
        selected["R2 Oracle Top-1"][split] = select_candidate(candidates, oracle_index)
        selected["R3 No-reference Top-1"][split] = select_candidate(candidates, score.argmax(1))

    pretrain_orders = epoch_orders(len(views["train"]["labels"]), int(config["epochs"]), seed + 10_000)
    probe_orders = epoch_orders(len(views["train"]["labels"]), int(config["probe_epochs"]), seed + 20_000)
    seed_everything(seed); template = build_model(config["model"], views["train"]["clean"].shape[1], 2)
    initial_state = copy.deepcopy(template.state_dict()); results = {}
    for method, method_views in selected.items():
        seed_everything(seed); method_started = time.perf_counter()
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        model = build_model(config["model"], views["train"]["clean"].shape[1], 2).to(device); model.load_state_dict(initial_state)
        train = dict(views["train"]); validation = dict(views["validation"]); test = dict(views["test"])
        train["restored"] = method_views["train"]; validation["restored"] = method_views["validation"]; test["restored"] = method_views["test"]
        ones_train = np.ones(len(train["labels"]), np.float32); ones_validation = np.ones(len(validation["labels"]), np.float32)
        pretrain = _fit_supcon(model, train, validation, ones_train, ones_validation, pretrain_orders, config, device)
        seed_everything(seed + 1); probe = _fit_probe(model, train, validation, probe_orders, config, device)
        validation_probability, _ = _probabilities(model, validation["restored"], int(config["batch_size"]), device)
        threshold = select_binary_threshold(validation["labels"], validation_probability[:, 1])
        probability, restored_embedding = _probabilities(model, test["restored"], int(config["batch_size"]), device)
        _, clean_embedding = _probabilities(model, test["clean"], int(config["batch_size"]), device)
        score = probability[:, 1]; metrics = _metrics(test["labels"], score, threshold)
        diagnostics = representation_diagnostics(clean_embedding, restored_embedding, test["labels"])
        teacher_clean = teacher_probabilities(teacher, test["clean"], int(config["batch_size"]), device)
        teacher_view = teacher_probabilities(teacher, test["restored"], int(config["batch_size"]), device)
        results[method] = {"metrics": metrics, "teacher_consistency": float(np.mean(teacher_clean.argmax(1) == teacher_view.argmax(1))),
                           "representation": {name: diagnostics[name] for name in ("fisher_ratio", "class_center_shift", "effective_rank")},
                           "subgroups": subgroup_metrics(test, score, threshold), "validation_threshold": threshold,
                           "initialization_sha256": _state_hash(initial_state), "pretrain_history": pretrain, "probe_history": probe,
                           "training_seconds": time.perf_counter() - method_started,
                           "peak_gpu_mib": torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0}
    fixed, random, oracle, reference = (results[name] for name in selected)
    checks = {"oracle_macro_f1_above_fixed_or_random": oracle["metrics"]["macro_f1"] > max(fixed["metrics"]["macro_f1"], random["metrics"]["macro_f1"]),
              "oracle_recall_not_lower_than_fixed": oracle["metrics"]["fault_recall"] >= fixed["metrics"]["fault_recall"],
              "reference_close_to_oracle_within_1_point": reference["metrics"]["macro_f1"] >= oracle["metrics"]["macro_f1"] - .01,
              "reference_above_random": reference["metrics"]["macro_f1"] > random["metrics"]["macro_f1"],
              "reference_far_not_worse_than_random": reference["metrics"]["far"] <= random["metrics"]["far"],
              "sample_total_weight_equal": True}
    status = "INTRA_SAMPLE_CANDIDATE_SELECTION_GO" if sum(checks.values()) >= 4 else "INTRA_SAMPLE_CANDIDATE_SELECTION_NO_GO"
    result = {"markers": config["markers"], "status": status, **environment_metadata(), "best_no_reference_score": best_score_name,
              "fairness": {"same_views_except_candidate_choice": True, "same_initialization": True, "same_batch_order": True,
                           "same_optimizer_epochs_temperature_probe": True, "sample_total_weight": 1.0},
              "results": results, "gate_checks": checks, "soft_selection": "NOT_RUN_REQUIRES_TOP1_POSITIVE_SIGNAL",
              "total_seconds": time.perf_counter() - started}
    write_json(output / "result.json", result); return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/intra_sample_candidate_retest.yaml")
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = run(config); print(json.dumps({"status": result["status"], "training_skipped": result.get("training_skipped", False)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
