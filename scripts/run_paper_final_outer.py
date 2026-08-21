from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, recall_score

import scripts.run_3w_clean_baseline as base3w
from augmentations import domain_budget_route
from baselines.external_augmentations import FreRAAdapter, traditional_view
from datasets.protocol import Run, Standardizer, window_runs
from datasets.tep import REQUIRED_FILES, frame_to_runs, read_rdata_frame
from datasets.three_w import discover_instances
from diffusion import (DiffusionSchedule, FrequencyForwardDiffusion, fit_spectral_statistics,
                       scale_spectral_budget, spectral_noise_variance)
from frequency import build_criticality, fault_stages, fit_frequency_scaler, log_amplitude_phase
from scripts.run_3w_clean_collapse_diagnosis import train_probe
from scripts.run_3w_diffusion_1seed import state_hash, supcon_orders
from scripts.run_3w_final_primary_grouped import FINAL_PRIMARY_CLASSES, build_model as build_3w_model
from scripts.run_diffusion_quality_retest import (_fit_probe, _fit_supcon, _metrics, _probabilities,
                                                  _state_hash, best_probe_record, epoch_orders)
from scripts.run_external_baselines import _fit_frera
from scripts.run_stage_frequency_diffusion_mvp import _configure, _runtime
from trainers import build_model
from trainers.balanced import sqrt_inverse_frequency_weights
from utils import environment_metadata, seed_everything, select_device, write_json


METHODS = ("NO_AUG", "JITTER", "SCALING", "JITTER_SCALING", "UNIFORM_DIFFUSION",
           "FRERA", "FINAL_QDIFFCL", "DCBR")
TRAINED_METHODS = METHODS[:-1]
SPLITS = ("train", "validation", "test")


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_strings(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode("utf-8")).hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True, encoding="utf-8").strip()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rho_name(value: float) -> str:
    return f"rho_{int(round(float(value) * 100)):03d}"


def run_id(dataset: str, outer_seed: int, model_seed: int, method: str) -> str:
    return f"{dataset.lower()}-outer{outer_seed}-seed{model_seed}-{method.lower()}"


def validate_frozen(config: dict[str, Any], require_outer_branch: bool = True) -> dict[str, Any]:
    protocol = yaml.safe_load(Path(config["protocol_config"]).read_text(encoding="utf-8"))
    freeze = _read(config["freeze_manifest"]); dry = _read(config["dry_run_manifest"])
    leakage = _read(config["leakage_audit"]); final = yaml.safe_load(Path(config["final_config"]).read_text(encoding="utf-8"))
    if freeze["status"] != "PAPER_FINAL_FREEZE_READY" or freeze["outer_training_run"] or freeze["outer_test_metrics_read"]:
        raise RuntimeError("pre-outer freeze boundary is not intact")
    if leakage["status"] not in ("PAPER_FINAL_PROTOCOL_DRY_RUN_GO", "PAPER_FINAL_PROTOCOL_AMENDMENT_GO") or leakage["outer_training_run"] or leakage["outer_test_metrics_read"]:
        raise RuntimeError("paper-final leakage audit is not GO")
    if dry.get("outer_metrics") is not None:
        raise RuntimeError("dry-run manifest unexpectedly contains outer metrics")
    if subprocess.call(["git", "merge-base", "--is-ancestor", freeze["head"], "HEAD"]) != 0:
        raise RuntimeError("the frozen source HEAD is not an ancestor of the current checkout")
    expected_weights = {"weight_discriminative": .5, "weight_early": .5, "weight_run_stability": 0.0}
    algorithm = config["algorithm"]
    checks = {
        "methods": tuple(config["methods"]) == METHODS,
        "weights": algorithm["criticality_weights"] == expected_weights,
        "ratio": float(algorithm["critical_ratio"]) == .3,
        "timesteps": (int(algorithm["t_critical"]), int(algorithm["t_noncritical"])) == (1, 5),
        "rho_grid": list(map(float, algorithm["rho_candidates"])) == [0, .25, .5, .75, 1],
        "final_weights": final.get("frozen") and final.get("weights") == expected_weights,
        "protocol_seeds_3w": list(map(int, config["three_w"]["outer_seeds"])) == [31001, 31002, 31003],
        "protocol_seeds_tep": list(map(int, config["tep"]["outer_seeds"])) == [32001, 32002, 32003],
        "model_seeds_3w": list(map(int, config["three_w"]["model_seeds"])) == [42, 43, 44, 45, 46],
        "model_seeds_tep": list(map(int, config["tep"]["model_seeds"])) == [7, 42, 43, 44, 2026],
        "protocol_frozen": bool(protocol.get("frozen")),
    }
    if not all(checks.values()):
        raise RuntimeError(f"frozen outer configuration changed: {checks}")
    current_config_hashes = {name: _sha256_file(name) for name in freeze["hashes"]["configs"]}
    if current_config_hashes != freeze["hashes"]["configs"]:
        raise RuntimeError("an A-F frozen config hash changed")
    current_split_hashes = {name: _sha256_file(name) for name in freeze["hashes"]["splits"]}
    if current_split_hashes != freeze["hashes"]["splits"]:
        raise RuntimeError("a frozen split/hash artifact changed")
    if not torch.cuda.is_available():
        raise RuntimeError("formal qdiffcl environment does not expose CUDA; stop before outer evaluation")
    if require_outer_branch:
        if _git("branch", "--show-current") != str(config["git_freeze"]["outer_branch"]):
            raise RuntimeError("outer evaluation must run from the frozen outer branch")
        tags = _git("tag", "--points-at", "HEAD").splitlines()
        if str(config["git_freeze"]["pre_outer_tag"]) not in tags:
            raise RuntimeError("HEAD is not the tagged pre-outer freeze commit")
    return {"freeze": freeze, "protocol": protocol, "dry": dry, "leakage": leakage, "checks": checks}


def split_record(config: dict[str, Any], dataset: str, outer_seed: int) -> dict[str, Any]:
    dry = _read(config["dry_run_manifest"])
    key = "three_w" if dataset == "3W" else "tep"
    matches = [row for row in dry[key] if int(row["outer_split_seed"]) == int(outer_seed)]
    if len(matches) != 1:
        raise RuntimeError(f"missing unique frozen split for {dataset} outer seed {outer_seed}")
    row = matches[0]; groups = [set(row["groups"][name]) for name in SPLITS]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("group leakage in frozen outer split")
    expected = config["three_w" if dataset == "3W" else "tep"]["expected_counts"]
    if {name: len(row["groups"][name]) for name in SPLITS} != {name: int(expected[name]) for name in SPLITS}:
        raise RuntimeError("outer split group count changed")
    return row


def build_manifest(config: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    path = Path(config["output"]["manifest"])
    if path.exists():
        manifest = _read(path)
        if manifest["phase_g_config_sha256"] != _sha256_file(Path(__file__).parents[1] / "configs/paper_final_outer.yaml"):
            if manifest.get("first_outer_metric_at") is not None:
                raise RuntimeError("cannot replace a manifest after an outer metric exists")
            archive = path.with_name("paper_final_outer_pre_amendment_manifest.json")
            if archive.exists():
                archive = path.with_name(f"paper_final_outer_pre_{str(manifest['phase_g_config_sha256'])[:12]}_failure_manifest.json")
            if not archive.exists(): write_json(archive, manifest)
            path.unlink()
        else:
            return manifest
    cells = []
    for dataset, key in (("3W", "three_w"), ("TEP", "tep")):
        for outer_seed in map(int, config[key]["outer_seeds"]):
            split = split_record(config, dataset, outer_seed)
            for model_seed in map(int, config[key]["model_seeds"]):
                for method in METHODS:
                    cells.append({"run_id": run_id(dataset, outer_seed, model_seed, method), "dataset": dataset,
                                  "outer_seed": outer_seed, "model_seed": model_seed, "method": method,
                                  "split_hash": _canonical_hash(split["groups"]), "status": "pending"})
    payload = {
        "status": "PAPER_FINAL_OUTER_PREPARED", "created_at": _now(), "started_at": None,
        "pre_outer_commit_sha": _git("rev-parse", "HEAD"), "base_head": audit["freeze"]["head"],
        "branch": _git("branch", "--show-current"), "freeze_tag": config["git_freeze"]["pre_outer_tag"], "freeze_manifest_sha256": _sha256_file(config["freeze_manifest"]),
        "phase_g_config_sha256": _sha256_file(Path(__file__).parents[1] / "configs/paper_final_outer.yaml"),
        "runner_sha256": _sha256_file(__file__), "data_hashes": {k: v["collection_sha256"] for k, v in audit["freeze"]["hashes"]["data"].items()},
        "frozen_config_hashes": audit["freeze"]["hashes"]["configs"],
        "frozen_split_hashes": audit["freeze"]["hashes"]["splits"],
        "environment": environment_metadata(), "python_executable": os.path.abspath(os.sys.executable),
        "expected_cells": 225, "logical_rows": len(cells), "detected_alias_cells": 0,
        "calibration_sub_jobs": {"DCBR_non_rho1_candidates": 120, "per_dataset": 60},
        "revised_3w_split_sha256": _sha256_file(config["dry_run_manifest"]),
        "tep_split_hash": _canonical_hash(audit["dry"]["tep"]),
        "first_outer_access_at": None,
        "estimated_wall_clock_hours": 16.0, "estimated_disk_gib": 4.0,
        "output_roots": {"3W": str(Path(config["output"]["root"]) / "3w"),
                         "TEP": str(Path(config["output"]["root"]) / "tep")},
        "methods": list(METHODS), "model_seeds": {"3W": config["three_w"]["model_seeds"], "TEP": config["tep"]["model_seeds"]},
        "outer_seeds": {"3W": config["three_w"]["outer_seeds"], "TEP": config["tep"]["outer_seeds"]},
        "cells": cells, "failures": [], "resumes": [], "first_outer_metric_at": None,
    }
    if len(cells) != 240:
        raise RuntimeError("logical 8-method result matrix must contain 240 rows")
    # The frozen 225 count treats 3W DCBR rho=1 as 15 FINAL aliases.
    write_json(path, payload); return payload


def _cell_dir(config: dict[str, Any], dataset: str, outer_seed: int, model_seed: int, method: str) -> Path:
    return (Path(config["output"]["root"]) / dataset.lower() / f"outer_{outer_seed}" /
            f"model_seed_{model_seed}" / method)


def _context_dir(config: dict[str, Any], dataset: str, outer_seed: int) -> Path:
    return Path(config["output"]["root"]) / dataset.lower() / f"outer_{outer_seed}" / "_context"


def _criticality(clean: np.ndarray, bundle: dict[str, np.ndarray], stages: np.ndarray,
                 algorithm: dict[str, Any]) -> dict[str, Any]:
    log = log_amplitude_phase(clean)[0]
    features = fit_frequency_scaler(log, "train").transform(log)
    settings = {"critical_ratio": float(algorithm["critical_ratio"]),
                "bootstrap_repeats": int(algorithm["bootstrap_repeats"]),
                "bootstrap_seed": int(algorithm["bootstrap_seed"]), **algorithm["criticality_weights"]}
    return build_criticality(features, bundle, stages, settings, log)


def _stage_from_three_w_refs(refs: list[Any]) -> np.ndarray:
    return np.asarray([ref.stage for ref in refs], dtype=str)


def prepare_three_w(config: dict[str, Any], outer_seed: int, device: str) -> dict[str, Any]:
    stage = config["three_w"]; grouped = yaml.safe_load(Path(stage["base_config"]).read_text(encoding="utf-8"))
    base = yaml.safe_load(Path(grouped["base_config"]).read_text(encoding="utf-8")); row = split_record(config, "3W", outer_seed)
    split = {name: set(row["groups"][name]) for name in SPLITS}; data_root = Path(stage["data_root"])
    base3w.PRIMARY_CLASSES = FINAL_PRIMARY_CLASSES; base3w.CLASS_TO_TARGET = {value: i for i, value in enumerate(FINAL_PRIMARY_CLASSES)}
    instances = [item for item in discover_instances(data_root) if item.source == "WELL" and item.event_class in FINAL_PRIMARY_CLASSES]
    by_instance = {item.instance_id: item for item in instances}; by_split = {name: [item for item in instances if item.well_id in wells] for name, wells in split.items()}
    pre_cfg = copy.deepcopy(base); pre_cfg["protocol"]["feature_min_train_coverage"] = 0.0; pre_cfg["protocol"]["split_seed"] = outer_seed
    features = tuple(grouped["frozen_process_features"]); preprocessor = base3w.fit_preprocessor(by_split["train"], features, pre_cfg)
    if tuple(preprocessor["retained_features"]) != features:
        raise RuntimeError("3W frozen process features changed")
    refs_by_split: dict[str, list[Any]] = {}; refs_by_instance: dict[str, list[Any]] = {}
    for name, items in by_split.items():
        refs = []
        for item in items:
            current = base3w.instance_refs(item, int(base["protocol"]["window_length"]), int(base["protocol"]["stride"]), int(base["protocol"]["transient_offset"]))
            refs.extend(current); refs_by_instance[item.instance_id] = current
        refs_by_split[name] = refs
    train_refs = base3w.stratified_refs(refs_by_split["train"], int(grouped["train_windows_per_class"]), outer_seed)
    val_refs = base3w.stratified_refs(refs_by_split["validation"], int(grouped["validation_windows_per_class"]), outer_seed + 1)
    length = int(base["protocol"]["window_length"])
    train_x, train_y = base3w.materialize(train_refs, by_instance, preprocessor, length, False)
    val_x, val_y = base3w.materialize(val_refs, by_instance, preprocessor, length, False)
    def uid(ref: Any) -> str:
        item = by_instance[ref.instance_id]; original = FINAL_PRIMARY_CLASSES[ref.target] if ref.target else 0
        return f"training:fault_{original:02d}:{item.well_id}"
    train_bundle = {"run_uid": np.asarray([uid(ref) for ref in train_refs]), "labels": train_y}
    critical = _criticality(train_x, train_bundle, _stage_from_three_w_refs(train_refs), config["algorithm"])
    training = {"epochs": int(grouped["pretrain_epochs"]), "probe_epochs": int(grouped["probe_epochs"]),
                "early_stopping_patience": int(grouped["pretrain_epochs"]), "batch_size": int(grouped["batch_size"]),
                "learning_rate": float(grouped["learning_rate"]), "temperature": float(grouped["temperature"]),
                "supcon_batching": "original"}
    context = {"dataset": "3W", "outer_seed": outer_seed, "base": base, "grouped": grouped, "split": split,
               "by_split": by_split, "by_instance": by_instance, "refs_by_instance": refs_by_instance,
               "train_refs": train_refs, "validation_refs": val_refs,
               "preprocessor": preprocessor, "train": train_x, "validation": val_x,
               "labels": {"train": train_y, "validation": val_y},
               "ids": {"train": np.asarray([f"{r.instance_id}:{r.start}:{r.target}" for r in train_refs]),
                       "validation": np.asarray([f"{r.instance_id}:{r.start}:{r.target}" for r in val_refs])},
               "training": training, "critical": critical}
    context["context_hash"] = _canonical_hash({"dataset": "3W", "outer_seed": outer_seed, "groups": row["groups"],
                                                 "preprocessor": preprocessor, "critical_soft_mask": critical["soft_mask"].tolist()})
    out = _context_dir(config, "3W", outer_seed); out.mkdir(parents=True, exist_ok=True)
    write_json(out / "preprocessor.json", preprocessor)
    write_json(out / "audit.json", {"context_hash": context["context_hash"], "groups": row["groups"],
                                      "group_hash": row["group_hash"], "fit_scope": "outer-train only",
                                      "criticality": {"weights": critical["component_weights"], "ratio": .3,
                                                      "soft_mask_sha256": hashlib.sha256(np.ascontiguousarray(critical["soft_mask"]).tobytes()).hexdigest()}})
    return context


def _load_selected_tep_runs(config: dict[str, Any]) -> list[Run]:
    stage = config["tep"]; data_cfg = yaml.safe_load(Path(stage["data_config"]).read_text(encoding="utf-8")); root = Path(stage["data_root"])
    data_cfg["dataset"]["root"] = str(root); runs = []
    for filename in REQUIRED_FILES:
        source = "testing" if "Testing" in filename else "training"
        limit = stage["selected_run_limits"][source]
        runs.extend(frame_to_runs(read_rdata_frame(root / filename), data_cfg, source, limit))
    if len(runs) != 400 or len({run.run_uid for run in runs}) != 400:
        raise RuntimeError(f"TEP registered Run universe changed: {len(runs)}")
    return runs


def _window_bundle(runs: list[Run], scaler: Standardizer, base: dict[str, Any]) -> dict[str, np.ndarray]:
    normalized = [Run(run.run_uid, scaler.transform(run.values), run.samples, run.fault_id, run.first_faulty_sample) for run in runs]
    x, y, ids, stats = window_runs(normalized, int(base["protocol"]["window_length"]), int(base["protocol"]["stride"]), "exclude_transition", None, "binary_fault_detection")
    records = [row for row in stats["window_metadata"] if not row["excluded"]]
    if len(records) != len(x) or [f"{row['run_uid']}:samples_{row['start_sample']}_{row['end_sample']}" for row in records] != ids:
        raise RuntimeError("TEP window metadata/order mismatch")
    return {"clean": x.astype(np.float32), "labels": y.astype(np.int64), "window_id": np.asarray(ids),
            "run_uid": np.asarray([row["run_uid"] for row in records]),
            "start_sample": np.asarray([row["start_sample"] for row in records], dtype=np.int64),
            "end_sample": np.asarray([row["end_sample"] for row in records], dtype=np.int64),
            "faultNumber": np.asarray([row["faultNumber"] for row in records], dtype=np.int64)}


def prepare_tep(config: dict[str, Any], outer_seed: int, device: str) -> dict[str, Any]:
    stage = config["tep"]; base = yaml.safe_load(Path(stage["base_config"]).read_text(encoding="utf-8")); _configure(base)
    row = split_record(config, "TEP", outer_seed); by_uid = {run.run_uid: run for run in _load_selected_tep_runs(config)}
    if set(by_uid) != set().union(*(set(row["groups"][name]) for name in SPLITS)):
        raise RuntimeError("TEP raw Run universe differs from frozen dry-run groups")
    groups = {name: [by_uid[uid] for uid in row["groups"][name]] for name in SPLITS}
    scaler = Standardizer().fit_many([run.values for run in groups["train"]])
    bundles = {name: _window_bundle(groups[name], scaler, base) for name in SPLITS}
    stages = {name: fault_stages(bundles[name], base) for name in SPLITS}
    critical = _criticality(bundles["train"]["clean"], bundles["train"], stages["train"], config["algorithm"])
    runtime = _runtime(base, 0); runtime["diagnosis"] = {"threshold_band_width": .05, "high_correlation_quantile": .90}
    context = {"dataset": "TEP", "outer_seed": outer_seed, "base": base, "split": {name: set(row["groups"][name]) for name in SPLITS},
               "bundles": bundles, "stages": stages, "critical": critical, "runtime": runtime,
               "train": bundles["train"]["clean"], "validation": bundles["validation"]["clean"],
               "labels": {name: bundles[name]["labels"] for name in SPLITS},
               "ids": {name: bundles[name]["window_id"] for name in SPLITS}}
    scaler_payload = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(), "fit_groups": sorted(row["groups"]["train"])}
    context["context_hash"] = _canonical_hash({"dataset": "TEP", "outer_seed": outer_seed, "groups": row["groups"],
                                                 "scaler": scaler_payload, "critical_soft_mask": critical["soft_mask"].tolist()})
    out = _context_dir(config, "TEP", outer_seed); out.mkdir(parents=True, exist_ok=True)
    write_json(out / "scaler.json", scaler_payload)
    write_json(out / "audit.json", {"context_hash": context["context_hash"], "groups": row["groups"],
                                      "group_hash": row["group_hash"], "fit_scope": "outer-train only",
                                      "window_counts": {name: len(bundles[name]["labels"]) for name in SPLITS},
                                      "criticality": {"weights": critical["component_weights"], "ratio": .3,
                                                      "soft_mask_sha256": hashlib.sha256(np.ascontiguousarray(critical["soft_mask"]).tobytes()).hexdigest()}})
    return context


def _augmenter(context: dict[str, Any], config: dict[str, Any], device: str) -> tuple[FrequencyForwardDiffusion, np.ndarray]:
    algorithm = config["algorithm"]; statistics = fit_spectral_statistics(context["train"], float(algorithm["clip_quantile"]), "train")
    schedule = DiffusionSchedule.cosine(int(algorithm["diffusion_steps"]), device)
    soft = np.asarray(context["critical"]["soft_mask"], dtype=np.float32)
    augmenter = FrequencyForwardDiffusion(statistics, schedule.alpha_bars, soft, int(algorithm["t_uniform"]),
                                          int(algorithm["t_critical"]), bool(algorithm["preserve_phase"]),
                                          bool(algorithm["preserve_dc"]), device)
    tensor = torch.as_tensor(soft, dtype=torch.float32, device=device)
    variance = spectral_noise_variance(schedule.alpha_bars, tensor.shape[0], tensor.shape[1], "selective",
                                       int(algorithm["t_uniform"]), bool(algorithm["preserve_dc"]), tensor,
                                       int(algorithm["t_critical"]), int(algorithm["t_noncritical"]))
    return augmenter, variance.cpu().numpy()


def augmentation_views(context: dict[str, Any], config: dict[str, Any], method: str, model_seed: int,
                       device: str, rho: float | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if method not in TRAINED_METHODS and method != "DCBR":
        raise ValueError(method)
    result: dict[str, np.ndarray] = {}; audit: dict[str, Any] = {}
    augmenter = final_variance = None
    if method in ("UNIFORM_DIFFUSION", "FINAL_QDIFFCL", "DCBR"):
        augmenter, final_variance = _augmenter(context, config, device)
    for split, offset in (("train", 0), ("validation", 100)):
        clean = context[split]; ids = context["ids"][split]
        if method == "NO_AUG":
            changed = clean.copy(); details = {"mode": "clean_pair", "exact_clean": True}
        elif method in ("JITTER", "SCALING", "JITTER_SCALING"):
            changed = traditional_view(clean, ids, method, model_seed + offset,
                                       float(config["traditional"]["jitter_std"]), float(config["traditional"]["scaling_std"]))
            details = {"mode": method, "finite": bool(np.isfinite(changed).all())}
        elif method == "UNIFORM_DIFFUSION":
            changed, details = augmenter.augment(clean, "uniform", model_seed + int(config["algorithm"]["sampling_seed_offset"]) + offset,
                                                 batch_size=int(context["training"]["batch_size"] if context["dataset"] == "3W" else context["runtime"]["batch_size"]),
                                                 noise_structure="iid")
        elif method == "FINAL_QDIFFCL":
            changed, details = augmenter.augment(clean, "selective", model_seed + int(config["algorithm"]["sampling_seed_offset"]) + offset,
                                                 int(config["algorithm"]["t_noncritical"]),
                                                 int(context["training"]["batch_size"] if context["dataset"] == "3W" else context["runtime"]["batch_size"]),
                                                 noise_structure="iid")
        elif method == "FRERA":
            changed = clean.copy(); details = {"mode": "FRERA_internal_learned_view", "dispatcher_placeholder": "clean"}
        else:
            if rho is None or float(rho) not in list(map(float, config["algorithm"]["rho_candidates"])):
                raise ValueError("DCBR requires a frozen rho candidate")
            if float(rho) == 0:
                diffused = clean.copy(); diffusion = {"exact_clean_bypass": True, "expected_total_noise_budget": 0.0}
            else:
                scaled = scale_spectral_budget(torch.as_tensor(final_variance, device=device), float(rho), bool(config["algorithm"]["preserve_dc"])).cpu().numpy()
                diffused, diffusion = augmenter.augment(clean, "budget_scaled_selective",
                    model_seed + int(config["algorithm"]["sampling_seed_offset"]) + offset,
                    int(config["algorithm"]["t_noncritical"]),
                    int(context["training"]["batch_size"] if context["dataset"] == "3W" else context["runtime"]["batch_size"]),
                    variance_override=scaled, noise_structure="iid")
            changed, routing = domain_budget_route(clean, diffused, ids, float(rho), float(config["algorithm"]["sigma_base"]), model_seed + offset)
            details = {"mode": "DCBR", "rho": float(rho), "diffusion": diffusion, "routing": routing}
        if changed.shape != clean.shape or not np.isfinite(changed).all():
            raise RuntimeError(f"invalid augmentation for {method} {split}")
        result[split] = changed.astype(np.float32, copy=False); audit[split] = details
    return result, audit


def _training_spec(context: dict[str, Any], model_seed: int) -> tuple[dict[str, Any], list[np.ndarray], list[np.ndarray], dict[str, torch.Tensor], dict[str, Any]]:
    if context["dataset"] == "3W":
        runtime = dict(context["training"]); well_ids = np.asarray([context["by_instance"][ref.instance_id].well_id for ref in context["train_refs"]], dtype=object)
        pretrain_orders, sampler = supcon_orders(context["labels"]["train"], runtime, model_seed, well_ids, include_batch_audit=False)
        probe_orders = []
        seed_everything(model_seed); template = build_3w_model(context["base"]["training"]["model"], context["train"].shape[1], "cpu")
        initial = copy.deepcopy(template.state_dict())
        fairness = {"initialization_sha256": state_hash(initial), "pretrain_order_sha256": sampler["batch_order_sha256"],
                    "window_refs_sha256": _sha256_strings([f"{r.instance_id}:{r.start}:{r.target}" for r in context["train_refs"] + context["validation_refs"]])}
    else:
        runtime = _runtime(context["base"], model_seed); runtime["diagnosis"] = {"threshold_band_width": .05, "high_correlation_quantile": .90}
        pretrain_orders = epoch_orders(len(context["train"]), int(runtime["epochs"]), model_seed + 10000)
        probe_orders = epoch_orders(len(context["train"]), int(runtime["probe_epochs"]), model_seed + 20000)
        seed_everything(model_seed); template = build_model(runtime["model"], context["train"].shape[1], 2)
        initial = copy.deepcopy(template.state_dict())
        fairness = {"initialization_sha256": _state_hash(initial),
                    "pretrain_order_sha256": _sha256_strings([",".join(map(str, order)) for order in pretrain_orders]),
                    "probe_order_sha256": _sha256_strings([",".join(map(str, order)) for order in probe_orders])}
    return runtime, pretrain_orders, probe_orders, initial, fairness


def _validation_metrics_three_w(model: torch.nn.Module, context: dict[str, Any], device: str) -> dict[str, float]:
    probability = base3w.probabilities(model, context["validation"], context["labels"]["validation"],
                                       int(context["training"]["batch_size"]), device)
    y = context["labels"]["validation"]; prediction = probability.argmax(1); fault = y != 0
    return {"macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
            "auprc": float(average_precision_score(fault.astype(int), 1 - probability[:, 0])),
            "far": float(np.mean(prediction[y == 0] != 0)), "threshold": None}


def _checkpoint_payload(path: Path, metadata: dict[str, Any]) -> dict[str, Any] | None:
    if not path.exists(): return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("metadata") != metadata:
        raise RuntimeError(f"resume metadata mismatch: {path}")
    return payload


def _compatible_metadata(existing: dict[str, Any], expected: dict[str, Any], config: dict[str, Any]) -> bool:
    left, right = copy.deepcopy(existing), copy.deepcopy(expected)
    left_hash = str(left.pop("phase_g_config_sha256", "")); right_hash = str(right.pop("phase_g_config_sha256", ""))
    allowed = {right_hash, *map(str, config["git_freeze"].get("provenance_compatible_config_hashes", []))}
    return left == right and left_hash in allowed


def train_method(context: dict[str, Any], config: dict[str, Any], method: str, model_seed: int,
                 device: str, output: Path, rho: float | None = None) -> dict[str, Any]:
    runtime, pre_orders, probe_orders, initial, fairness = _training_spec(context, model_seed)
    metadata = {"dataset": context["dataset"], "outer_seed": context["outer_seed"], "model_seed": model_seed,
                "method": method, "rho": rho, "context_hash": context["context_hash"], "fairness": fairness,
                "phase_g_config_sha256": _sha256_file(Path(__file__).parents[1] / "configs/paper_final_outer.yaml")}
    checkpoint = output / "model.pt"; validation_path = output / "validation.json"
    if checkpoint.exists() != validation_path.exists():
        raise RuntimeError(f"incomplete cell cannot be safely resumed: {output}")
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False); record = _read(validation_path)
        if not _compatible_metadata(payload.get("metadata", {}), metadata, config): raise RuntimeError(f"resume metadata mismatch: {checkpoint}")
        if not _compatible_metadata(record["metadata"], metadata, config): raise RuntimeError("validation resume metadata mismatch")
        record["resumed"] = True; return record
    views, augmentation_audit = augmentation_views(context, config, method, model_seed, device, rho)
    started = time.perf_counter(); seed_everything(model_seed)
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    if context["dataset"] == "3W":
        model = build_3w_model(context["base"]["training"]["model"], context["train"].shape[1], device); model.load_state_dict(initial)
    else:
        model = build_model(runtime["model"], context["train"].shape[1], 2).to(device); model.load_state_dict(initial)
    if method == "FRERA":
        frera = config["frera"]
        pretrain, augmenter = _fit_frera(model, context["train"], context["labels"]["train"], context["validation"],
                                         context["labels"]["validation"], pre_orders, runtime, device, frera)
        augmentation_audit["official_commit"] = FreRAAdapter.OFFICIAL_COMMIT
        del augmenter
    else:
        pretrain = _fit_supcon(model,
            {"clean": context["train"], "restored": views["train"], "labels": context["labels"]["train"]},
            {"clean": context["validation"], "restored": views["validation"], "labels": context["labels"]["validation"]},
            np.ones(len(context["train"]), np.float32), np.ones(len(context["validation"]), np.float32),
            pre_orders, runtime, device)
    seed_everything(model_seed + 1)
    if context["dataset"] == "3W":
        probe = train_probe(model, context["train"], context["labels"]["train"], context["validation"], context["labels"]["validation"],
                            sqrt_inverse_frequency_weights(context["labels"]["train"]), int(runtime["probe_epochs"]),
                            float(runtime["learning_rate"]), int(runtime["batch_size"]), model_seed, device)
        validation = _validation_metrics_three_w(model, context, device); threshold = None
    else:
        probe = _fit_probe(model, {"clean": context["train"], "labels": context["labels"]["train"]},
                           {"restored": context["validation"], "labels": context["labels"]["validation"]},
                           probe_orders, runtime, device)
        best = best_probe_record(probe); threshold = float(best["validation_threshold"])
        probability, _ = _probabilities(model, context["validation"], int(runtime["batch_size"]), device)
        validation = _metrics(context["labels"]["validation"], probability[:, 1], threshold)
        validation = {"macro_f1": validation["macro_f1"], "auprc": validation["auprc"], "far": validation["far"], "threshold": threshold}
    record = {"metadata": metadata, "validation": validation, "validation_only": True, "outer_test_read": False,
              "pretrain_history": pretrain, "probe_history": probe, "augmentation_audit": augmentation_audit,
              "training_seconds": time.perf_counter() - started,
              "peak_gpu_mib": torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0,
              "resumed": False}
    output.mkdir(parents=True, exist_ok=False)
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata, "threshold": threshold}, checkpoint)
    write_json(validation_path, record)
    return record


def select_dcbr_rho(config: dict[str, Any], context: dict[str, Any], records: dict[int, dict[float, dict[str, Any]]]) -> dict[str, Any]:
    rhos = list(map(float, config["algorithm"]["rho_candidates"])); rows = []
    for rho in rhos:
        metrics = [records[seed][rho]["validation"] for seed in sorted(records)]
        rows.append({"rho": rho, "model_seeds": sorted(records),
                     "macro_f1": float(np.mean([row["macro_f1"] for row in metrics])),
                     "auprc": float(np.mean([row["auprc"] for row in metrics])),
                     "far": float(np.mean([row["far"] for row in metrics]))})
    selected = max(rows, key=lambda row: (row["macro_f1"], row["auprc"], -row["far"], -row["rho"]))
    return {"dataset": context["dataset"], "outer_seed": context["outer_seed"], "selection_split": "inner-validation",
            "selection_unit": "domain-level mean across frozen model seeds", "candidate_rows": rows,
            "selected_rho": selected["rho"], "outer_test_read": False, "selected_at": _now()}


def _load_model(context: dict[str, Any], checkpoint: Path, device: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if context["dataset"] == "3W": model = build_3w_model(context["base"]["training"]["model"], context["train"].shape[1], device)
    else: model = build_model(context["runtime"]["model"], context["train"].shape[1], 2).to(device)
    model.load_state_dict(payload["model_state_dict"]); model.eval(); return model, payload


def _multiclass_metrics(y: np.ndarray, prediction: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    fault = y != 0
    return {"macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
            "auprc": float(average_precision_score(fault.astype(int), 1 - probability[:, 0])),
            "far": float(np.mean(prediction[y == 0] != 0)),
            "fault_recall": float(np.mean(prediction[fault] != 0)),
            "confusion_matrix": confusion_matrix(y, prediction, labels=np.arange(len(FINAL_PRIMARY_CLASSES))).tolist()}


def evaluate_three_w(model: torch.nn.Module, context: dict[str, Any], device: str) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]]]:
    ys = []; predictions = []; probabilities = []; groups = []; faults = []; instances = []; stages = []; starts = []; ends = []
    length = int(context["base"]["protocol"]["window_length"]); batch = int(context["training"]["batch_size"])
    delay_by_instance: dict[str, float | None] = {}
    for item in context["by_split"]["test"]:
        refs = context["refs_by_instance"].get(item.instance_id, [])
        if not refs: continue
        x, y = base3w.materialize(refs, {item.instance_id: item}, context["preprocessor"], length, False)
        prob = base3w.probabilities(model, x, y, batch, device); pred = prob.argmax(1)
        ys.append(y); predictions.append(pred); probabilities.append(prob); groups.extend([item.well_id] * len(y)); faults.extend([item.event_class] * len(y)); instances.extend([item.instance_id] * len(y)); stages.extend([ref.stage for ref in refs]); starts.extend([ref.start for ref in refs]); ends.extend([ref.start + length - 1 for ref in refs])
        onset = refs[0].onset_seconds; detections = [ref.end_seconds - float(onset) for ref, value in zip(refs, pred) if onset is not None and ref.target != 0 and ref.end_seconds >= onset and value != 0]
        delay_by_instance[item.instance_id] = float(detections[0]) if detections else None
    raw = {"label": np.concatenate(ys), "prediction": np.concatenate(predictions), "probability": np.concatenate(probabilities),
           "group_id": np.asarray(groups), "fault_id": np.asarray(faults), "instance_id": np.asarray(instances),
           "stage": np.asarray(stages), "start": np.asarray(starts), "end": np.asarray(ends)}
    metrics = _multiclass_metrics(raw["label"], raw["prediction"], raw["probability"])
    early = raw["stage"] == "early"; metrics["early_recall"] = float(np.mean(raw["prediction"][early] != 0)) if early.any() else None
    delays = [value for value in delay_by_instance.values() if value is not None]; metrics["detection_delay"] = float(np.mean(delays)) if delays else None
    groupwise = []
    for group in sorted(set(groups)):
        sel = raw["group_id"] == group; current = _multiclass_metrics(raw["label"][sel], raw["prediction"][sel], raw["probability"][sel])
        current.update({"group_id": group, "windows": int(sel.sum()), "early_recall": float(np.mean(raw["prediction"][sel & early] != 0)) if np.any(sel & early) else None})
        group_delays = [value for key, value in delay_by_instance.items() if key in set(raw["instance_id"][sel]) and value is not None]
        current["detection_delay"] = float(np.mean(group_delays)) if group_delays else None; groupwise.append(current)
    return metrics, raw, groupwise


def evaluate_tep(model: torch.nn.Module, context: dict[str, Any], threshold: float, device: str) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]]]:
    bundle = context["bundles"]["test"]; probability, _ = _probabilities(model, bundle["clean"], int(context["runtime"]["batch_size"]), device)
    scores = probability[:, 1]; prediction = (scores >= threshold).astype(np.int64); labels = bundle["labels"]
    raw = {"label": labels, "prediction": prediction, "score": scores, "probability": probability,
           "group_id": bundle["run_uid"], "fault_id": bundle["faultNumber"], "stage": context["stages"]["test"],
           "start": bundle["start_sample"], "end": bundle["end_sample"]}
    metrics = _metrics(labels, scores, threshold); early = raw["stage"] == "early"
    metrics["early_recall"] = float(prediction[early].mean()) if early.any() else None
    groupwise = []; delays = []
    for group in sorted(set(map(str, raw["group_id"]))):
        sel = raw["group_id"] == group; current = _metrics(labels[sel], scores[sel], threshold)
        fault = int(raw["fault_id"][sel][0]); delay = None
        if fault:
            post = sel & (raw["stage"] != "prefault"); detected = np.flatnonzero(post & (prediction == 1))
            if len(detected):
                onset = int(context["base"]["protocol"]["fault_onset"][str(group).split(":", 1)[0]])
                delay = float(raw["end"][detected[0]] - onset); delays.append(delay)
        current.update({"group_id": group, "fault_id": fault, "windows": int(sel.sum()),
                        "early_recall": float(prediction[sel & early].mean()) if np.any(sel & early) else None,
                        "detection_delay": delay}); groupwise.append(current)
    metrics["detection_delay"] = float(np.mean(delays)) if delays else None
    return metrics, raw, groupwise


def evaluate_cell(config: dict[str, Any], context: dict[str, Any], method: str, model_seed: int,
                  device: str, source: Path, selected_rho: float | None = None) -> dict[str, Any]:
    output = _cell_dir(config, context["dataset"], context["outer_seed"], model_seed, method)
    result_path = output / "result.json"; predictions_path = output / "predictions.npz"
    if result_path.exists() and predictions_path.exists(): return _read(result_path)
    if result_path.exists() or predictions_path.exists(): raise RuntimeError(f"incomplete outer result: {output}")
    model, payload = _load_model(context, source / "model.pt", device); threshold = payload.get("threshold")
    if context["dataset"] == "3W": metrics, raw, groupwise = evaluate_three_w(model, context, device)
    else:
        if threshold is None: raise RuntimeError("TEP outer evaluation requires inner-validation threshold")
        metrics, raw, groupwise = evaluate_tep(model, context, float(threshold), device)
    output.mkdir(parents=True, exist_ok=True); np.savez_compressed(predictions_path, **raw)
    checkpoint_hash = _sha256_file(source / "model.pt")
    record = {"run_id": run_id(context["dataset"], context["outer_seed"], model_seed, method), "dataset": context["dataset"],
              "outer_seed": context["outer_seed"], "model_seed": model_seed, "method": method, "selected_rho": selected_rho,
              "threshold": threshold, "metrics": metrics, "groupwise": groupwise, "prediction_path": str(predictions_path),
              "prediction_sha256": _sha256_file(predictions_path), "checkpoint_path": str(source / "model.pt"),
              "checkpoint_sha256": checkpoint_hash, "context_hash": context["context_hash"], "commit_sha": _git("rev-parse", "HEAD"),
              "environment": environment_metadata(), "outer_test_evaluated_once": True, "completed_at": _now()}
    write_json(result_path, record); return record


def verify_data_hashes(freeze: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for dataset, record in freeze["hashes"]["data"].items():
        root = Path(record["root"]); digest = hashlib.sha256(); count = total = 0
        for item in record["files"]:
            path = root / item["path"]
            if not path.is_file() or path.stat().st_size != int(item["bytes"]):
                raise RuntimeError(f"frozen {dataset} data file missing or resized: {path}")
            value = _sha256_file(path)
            if value != item["sha256"]:
                raise RuntimeError(f"frozen {dataset} data file hash changed: {path}")
            digest.update(f"{item['path']}\0{item['bytes']}\0{value}\n".encode()); count += 1; total += int(item["bytes"])
        if digest.hexdigest() != record["collection_sha256"] or count != record["file_count"] or total != record["total_bytes"]:
            raise RuntimeError(f"{dataset} data collection hash changed")
        result[dataset] = {"collection_sha256": digest.hexdigest(), "file_count": count, "total_bytes": total}
    return result


def _update_manifest_cell(manifest: dict[str, Any], record: dict[str, Any], alias: bool = False) -> None:
    matches = [cell for cell in manifest["cells"] if cell["run_id"] == record["run_id"]]
    if len(matches) != 1: raise RuntimeError(f"manifest cell missing: {record['run_id']}")
    cell = matches[0]; cell.update({"status": "alias" if alias else "complete", "result_path": str(_cell_dir_from_record(record) / "result.json"),
                                   "completed_at": record["completed_at"], "selected_rho": record.get("selected_rho")})


def _cell_dir_from_record(record: dict[str, Any]) -> Path:
    return Path(record["prediction_path"]).parent


def _store_manifest(config: dict[str, Any], manifest: dict[str, Any]) -> None:
    manifest["completed_logical_rows"] = sum(cell["status"] in ("complete", "alias") for cell in manifest["cells"])
    manifest["alias_rows"] = sum(cell["status"] == "alias" for cell in manifest["cells"])
    manifest["failed_rows"] = sum(cell["status"] == "failed" for cell in manifest["cells"])
    write_json(Path(config["output"]["manifest"]), manifest)


def run_outer_split(config: dict[str, Any], manifest: dict[str, Any], dataset: str, outer_seed: int, device: str) -> dict[str, Any]:
    context = prepare_three_w(config, outer_seed, device) if dataset == "3W" else prepare_tep(config, outer_seed, device)
    seeds = list(map(int, config["three_w" if dataset == "3W" else "tep"]["model_seeds"])); candidates: dict[int, dict[float, dict[str, Any]]] = {}
    for model_seed in seeds:
        for method in TRAINED_METHODS:
            path = _cell_dir(config, dataset, outer_seed, model_seed, method) / "_training"
            train_method(context, config, method, model_seed, device, path)
        candidates[model_seed] = {}
        final_path = _cell_dir(config, dataset, outer_seed, model_seed, "FINAL_QDIFFCL") / "_training"
        candidates[model_seed][1.0] = _read(final_path / "validation.json")
        for rho in map(float, config["algorithm"]["rho_candidates"]):
            if rho == 1.0: continue
            candidate_path = (_cell_dir(config, dataset, outer_seed, model_seed, "DCBR") / "_candidates" / rho_name(rho))
            candidates[model_seed][rho] = train_method(context, config, "DCBR", model_seed, device, candidate_path, rho)
    selection = select_dcbr_rho(config, context, candidates); selection_path = _context_dir(config, dataset, outer_seed) / "dcbr_selection.json"
    if selection_path.exists() and _read(selection_path) != selection:
        # selected_at is intentionally ignored for deterministic resume comparison.
        old = _read(selection_path); old.pop("selected_at", None); current = dict(selection); current.pop("selected_at", None)
        if old != current: raise RuntimeError("existing DCBR domain selection changed")
        selection = _read(selection_path)
    else: write_json(selection_path, selection)
    rho = float(selection["selected_rho"])
    if manifest["first_outer_metric_at"] is None:
        timestamp = _now(); manifest["first_outer_access_at"] = timestamp; manifest["first_outer_metric_at"] = timestamp
        manifest["status"] = "PAPER_FINAL_OUTER_RUNNING"; manifest["started_at"] = manifest["started_at"] or timestamp; _store_manifest(config, manifest)
    for model_seed in seeds:
        for method in METHODS:
            if method == "DCBR":
                alias = rho == 1.0
                source = (_cell_dir(config, dataset, outer_seed, model_seed, "FINAL_QDIFFCL") / "_training" if alias else
                          _cell_dir(config, dataset, outer_seed, model_seed, "DCBR") / "_candidates" / rho_name(rho))
                selected_rho = rho
            else:
                alias = False; source = _cell_dir(config, dataset, outer_seed, model_seed, method) / "_training"; selected_rho = None
            record = evaluate_cell(config, context, method, model_seed, device, source, selected_rho)
            _update_manifest_cell(manifest, record, alias); _store_manifest(config, manifest)
    return selection


def _result_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    root = Path(config["output"]["root"]); records = []
    for path in sorted(root.glob("*/outer_*/model_seed_*/*/result.json")):
        if "_training" in path.parts or "_candidates" in path.parts: continue
        records.append(_read(path))
    return records


def write_intermediate_tables(config: dict[str, Any]) -> None:
    records = _result_records(config); raw_columns = ["run_id", "dataset", "outer_seed", "model_seed", "method", "selected_rho", "threshold",
        "macro_f1", "auprc", "far", "fault_recall", "early_recall", "detection_delay", "prediction_path", "prediction_sha256", "checkpoint_sha256"]
    Path(config["output"]["raw_csv"]).parent.mkdir(parents=True, exist_ok=True)
    with Path(config["output"]["raw_csv"]).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=raw_columns); writer.writeheader()
        for record in records:
            metrics = record["metrics"]
            writer.writerow({"run_id": record["run_id"], "dataset": record["dataset"], "outer_seed": record["outer_seed"],
                "model_seed": record["model_seed"], "method": record["method"], "selected_rho": record.get("selected_rho"),
                "threshold": record.get("threshold"), "macro_f1": metrics.get("macro_f1"), "auprc": metrics.get("auprc"),
                "far": metrics.get("far"), "fault_recall": metrics.get("fault_recall"), "early_recall": metrics.get("early_recall"),
                "detection_delay": metrics.get("detection_delay"), "prediction_path": record["prediction_path"],
                "prediction_sha256": record["prediction_sha256"], "checkpoint_sha256": record["checkpoint_sha256"]})
    group_columns = ["run_id", "dataset", "outer_seed", "model_seed", "method", "group_id", "fault_id", "windows",
                     "macro_f1", "auprc", "far", "fault_recall", "early_recall", "detection_delay"]
    with Path(config["output"]["groupwise_csv"]).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=group_columns); writer.writeheader()
        for record in records:
            for row in record["groupwise"]:
                writer.writerow({"run_id": record["run_id"], "dataset": record["dataset"], "outer_seed": record["outer_seed"],
                    "model_seed": record["model_seed"], "method": record["method"],
                    **{key: row.get(key) for key in group_columns[5:]}})


def _macro_f1_from_confusions(confusions: np.ndarray) -> np.ndarray:
    confusions = np.asarray(confusions, dtype=np.float64)
    tp = np.diagonal(confusions, axis1=-2, axis2=-1); predicted = confusions.sum(axis=-2); actual = confusions.sum(axis=-1)
    denominator = predicted + actual; f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    return f1.mean(axis=-1)


def _group_confusions(record: dict[str, Any], classes: int) -> tuple[list[str], np.ndarray]:
    with np.load(record["prediction_path"], allow_pickle=False) as archive:
        groups = archive["group_id"].astype(str); labels = archive["label"].astype(int); prediction = archive["prediction"].astype(int)
    unique = sorted(set(groups)); values = []
    for group in unique:
        values.append(confusion_matrix(labels[groups == group], prediction[groups == group], labels=np.arange(classes)))
    return unique, np.stack(values)


def paired_group_bootstrap(config: dict[str, Any], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    repeats = int(config["statistics"]["bootstrap_repeats"]); seed = int(config["statistics"]["bootstrap_seed"])
    by_key = {(row["dataset"], int(row["outer_seed"]), int(row["model_seed"]), row["method"]): row for row in records}
    output = []
    for dataset in ("3W", "TEP"):
        classes = len(FINAL_PRIMARY_CLASSES) if dataset == "3W" else 2
        methods = [method for method in METHODS if method != "FINAL_QDIFFCL"]
        for method_index, method in enumerate(methods):
            pair_draws = []; observed = []; pair_count = 0
            keys = sorted(key for key in by_key if key[0] == dataset and key[3] == method)
            for _, outer_seed, model_seed, _ in keys:
                current = by_key[(dataset, outer_seed, model_seed, method)]; reference = by_key[(dataset, outer_seed, model_seed, "FINAL_QDIFFCL")]
                groups_a, cm_a = _group_confusions(current, classes); groups_b, cm_b = _group_confusions(reference, classes)
                if groups_a != groups_b: raise RuntimeError(f"paired outer groups differ for {dataset} {outer_seed} {model_seed} {method}")
                rng = np.random.default_rng(seed + method_index * 100000 + outer_seed + model_seed)
                counts = rng.multinomial(len(groups_a), np.full(len(groups_a), 1 / len(groups_a)), size=repeats)
                sampled_a = np.einsum("rg,gij->rij", counts, cm_a); sampled_b = np.einsum("rg,gij->rij", counts, cm_b)
                pair_draws.append(_macro_f1_from_confusions(sampled_a) - _macro_f1_from_confusions(sampled_b))
                observed.append(float(current["metrics"]["macro_f1"]) - float(reference["metrics"]["macro_f1"])); pair_count += 1
            draws = np.mean(np.stack(pair_draws), axis=0); low, high = np.quantile(draws, [.025, .975])
            output.append({"dataset": dataset, "method": method, "reference": "FINAL_QDIFFCL", "metric": "macro_f1",
                           "paired_cells": pair_count, "effect": float(np.mean(observed)), "ci_low": float(low), "ci_high": float(high),
                           "positive_count": int(np.sum(np.asarray(observed) > 0)), "nonworse_count": int(np.sum(np.asarray(observed) >= 0)),
                           "worst_delta": float(np.min(observed)), "bootstrap_unit": "WELL" if dataset == "3W" else "Run",
                           "bootstrap_repeats": repeats})
    return output


def summarize_records(config: dict[str, Any], records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(records) != 240: raise RuntimeError(f"outer result matrix incomplete: {len(records)}/240 logical rows")
    metric_names = ("macro_f1", "auprc", "far", "fault_recall", "early_recall", "detection_delay")
    summary = []
    for dataset in ("3W", "TEP"):
        for method in METHODS:
            selected = [row for row in records if row["dataset"] == dataset and row["method"] == method]
            split_rows = []
            for outer_seed in sorted({int(row["outer_seed"]) for row in selected}):
                cells = [row for row in selected if int(row["outer_seed"]) == outer_seed]
                current = {"dataset": dataset, "method": method, "level": "split", "outer_seed": outer_seed, "cells": len(cells)}
                for metric in metric_names:
                    values = [float(row["metrics"][metric]) for row in cells if row["metrics"].get(metric) is not None]
                    current[f"{metric}_mean"] = float(np.mean(values)) if values else None
                    current[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else None
                split_rows.append(current); summary.append(current)
            aggregate = {"dataset": dataset, "method": method, "level": "overall", "outer_seed": "ALL", "cells": len(selected)}
            for metric in metric_names:
                values = [row[f"{metric}_mean"] for row in split_rows if row.get(f"{metric}_mean") is not None]
                aggregate[f"{metric}_mean"] = float(np.mean(values)) if values else None
                aggregate[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else None
            aggregate["worst_seed_macro_f1"] = float(min(row["metrics"]["macro_f1"] for row in selected)); summary.append(aggregate)
    bootstrap = paired_group_bootstrap(config, records); return summary, bootstrap


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader(); writer.writerows(rows)


def _overall(summary: list[dict[str, Any]], dataset: str, method: str) -> dict[str, Any]:
    return next(row for row in summary if row["dataset"] == dataset and row["method"] == method and row["level"] == "overall")


def write_final_documents(config: dict[str, Any], records: list[dict[str, Any]], summary: list[dict[str, Any]], bootstrap: list[dict[str, Any]]) -> None:
    lines = ["# Q-DiffCL Paper-final Outer Evaluation", "", "状态：完整冻结 outer matrix 已执行；下表先在每个 outer split 内汇总 5 个 model seeds，再汇总 3 个 splits。", "",
             "| Dataset | Method | Macro-F1 | AUPRC | FAR | Early Recall | Delay | Worst cell |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for dataset in ("3W", "TEP"):
        for method in METHODS:
            row = _overall(summary, dataset, method)
            def value(name: str) -> str:
                current = row.get(name); return "N/A" if current is None else f"{current:.4f}"
            lines.append(f"| {dataset} | {method} | {value('macro_f1_mean')} ± {value('macro_f1_std')} | {value('auprc_mean')} | {value('far_mean')} | {value('early_recall_mean')} | {value('detection_delay_mean')} | {value('worst_seed_macro_f1')} |")
    lines += ["", "## Paired group-aware effects vs FINAL_QDIFFCL", "", "正效应表示行方法优于 FINAL；CI 以 WELL/Run 为重采样单位，未把 correlated windows 当独立样本。", "",
              "| Dataset | Method - FINAL | ΔMacro-F1 | 95% CI | Positive | Non-worse | Worst Δ |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in bootstrap:
        lines.append(f"| {row['dataset']} | {row['method']} | {row['effect']:+.4f} | [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | {row['positive_count']}/{row['paired_cells']} | {row['nonworse_count']}/{row['paired_cells']} | {row['worst_delta']:+.4f} |")
    Path(config["output"]["summary_doc"]).write_text("\n".join(lines) + "\n", encoding="utf-8")

    worst_cells = sorted(records, key=lambda row: float(row["metrics"]["macro_f1"]))[:12]
    group_rows = [(record, group) for record in records for group in record["groupwise"] if group.get("macro_f1") is not None]
    worst_groups = sorted(group_rows, key=lambda pair: float(pair[1]["macro_f1"]))[:20]
    failure = ["# Paper-final Outer Failure Analysis", "", "低结果先经 split、fit scope、hash、checkpoint 与 resume 审计；没有协议错误的低值作为真实 generalization evidence 保留。", "", "## Lowest cells", "",
               "| Dataset | Outer | Seed | Method | Macro-F1 |", "|---|---:|---:|---|---:|"]
    for row in worst_cells: failure.append(f"| {row['dataset']} | {row['outer_seed']} | {row['model_seed']} | {row['method']} | {row['metrics']['macro_f1']:.4f} |")
    failure += ["", "## Lowest groups", "", "| Dataset | Method | Group | Macro-F1 | Fault |", "|---|---|---|---:|---:|"]
    for record, group in worst_groups: failure.append(f"| {record['dataset']} | {record['method']} | {group['group_id']} | {group['macro_f1']:.4f} | {group.get('fault_id', 'mixed')} |")
    failure += ["", "这些结果可能反映 domain shift、hard WELL/fault、seed instability 或过增强；failure analysis 不产生新方法版本。"]
    Path(config["output"]["failure_doc"]).write_text("\n".join(failure) + "\n", encoding="utf-8")

    effects = {(row["dataset"], row["method"]): row for row in bootstrap}
    safe = []; dependent = []
    for dataset in ("3W", "TEP"):
        no_aug = effects[(dataset, "NO_AUG")]
        relation = "FINAL 显著优于 NO_AUG" if no_aug["ci_high"] < 0 else "FINAL 与 NO_AUG 的差异不确定" if no_aug["ci_low"] <= 0 <= no_aug["ci_high"] else "NO_AUG 显著优于 FINAL"
        (safe if no_aug["ci_high"] < 0 else dependent).append(f"- {dataset}: {relation}（paired Δ NO_AUG-FINAL={no_aug['effect']:+.4f}, 95% CI [{no_aug['ci_low']:+.4f}, {no_aug['ci_high']:+.4f}]）。")
    claims = ["# Q-DiffCL Paper-final Claims", "", "## SAFE TO CLAIM", "", *(safe or ["- Outer 结果不支持跨两个数据集都成立的无条件性能优越性表述；可安全陈述已完成冻结 nested/grouped evaluation。"]),
              "", "## DATASET-DEPENDENT CLAIM", "", *dependent,
              "- selective/soft matched-budget mechanism 的优势仍是 3W 支持、TEP 不一致；DCBR 的作用按数据集分别表述。",
              "", "## DEVELOPMENT EVIDENCE ONLY", "", "- 2×2 contrastive interaction、critical-ratio sensitivity、TEP onset trajectory 与机制 ablation 没有在 outer matrix 重跑。",
              "", "## LIMITATION", "", "- soft allocation 跨数据集不一致；critical_ratio=0.30 不是 universal optimum。", "- limited-data 与更广 missingness robustness 未完成；FRERA augmentation-only timing 缺失。", "- AutoDA 仅 method-native supplementary；DiCL 存在公平复现缺口。",
              "", "## DO NOT CLAIM", "", "- 不宣称 universal Soft superiority、universal cross-WELL superiority、0.30 universal optimum，或未评估的 robustness。"]
    Path(config["output"]["claims_doc"]).write_text("\n".join(claims) + "\n", encoding="utf-8")


def update_evidence_documents(config: dict[str, Any], summary: list[dict[str, Any]], bootstrap: list[dict[str, Any]]) -> None:
    matrix_path = Path("docs/paper_evidence_matrix.md"); matrix = matrix_path.read_text(encoding="utf-8")
    old = "| Generalization | nested grouped paper-final protocol and pre-outer freeze | `PENDING OUTER EVALUATION` |"
    new = "| Generalization | completed frozen nested/grouped outer matrix; split-first aggregation and 2,000× WELL/Run bootstrap | `OUTER EVALUATION COMPLETE; DATASET-SPECIFIC EFFECTS IN paper_final_outer_summary.md` |"
    if old not in matrix and new not in matrix: raise RuntimeError("Evidence Matrix generalization row changed unexpectedly")
    matrix_path.write_text(matrix.replace(old, new), encoding="utf-8")
    chain_path = Path("docs/paper_evidence_chain_summary.md"); chain = chain_path.read_text(encoding="utf-8")
    marker = "## Paper-final outer evaluation"
    if marker not in chain:
        rows = []
        for dataset in ("3W", "TEP"):
            final = _overall(summary, dataset, "FINAL_QDIFFCL"); dcbr = _overall(summary, dataset, "DCBR")
            effect = next(row for row in bootstrap if row["dataset"] == dataset and row["method"] == "DCBR")
            rows.append(f"- {dataset}: FINAL Macro-F1 `{final['macro_f1_mean']:.4f}`; DCBR `{dcbr['macro_f1_mean']:.4f}`; paired DCBR-FINAL `{effect['effect']:+.4f}` (95% CI `{effect['ci_low']:+.4f}` to `{effect['ci_high']:+.4f}`).")
        chain += "\n" + marker + "\n\nThe frozen nested/grouped outer matrix is complete. Statistics retain WELL/Run grouping and aggregate model seeds inside each split before the three-split summary.\n\n" + "\n".join(rows) + "\n\nClaim categories and limitations are frozen in `docs/paper_final_claims.md`.\n"
        chain_path.write_text(chain, encoding="utf-8")


def final_protocol_audit(config: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    checks = {"logical_rows_240": len(records) == 240, "unique_run_ids": len({row["run_id"] for row in records}) == 240,
              "outer_once_markers": all(row.get("outer_test_evaluated_once") is True for row in records),
              "prediction_hashes": all(_sha256_file(row["prediction_path"]) == row["prediction_sha256"] for row in records),
              "checkpoint_hashes": all(_sha256_file(row["checkpoint_path"]) == row["checkpoint_sha256"] for row in records)}
    group_scope = True
    for row in records:
        frozen = split_record(config, row["dataset"], int(row["outer_seed"]))
        with np.load(row["prediction_path"], allow_pickle=False) as archive: observed = set(map(str, archive["group_id"]))
        if observed != set(map(str, frozen["groups"]["test"])): group_scope = False; break
    checks["exact_outer_test_groups"] = group_scope
    payload = {"status": "PAPER_FINAL_PROTOCOL_AUDIT_GO" if all(checks.values()) else "PAPER_FINAL_PROTOCOL_AUDIT_HOLD",
               "checks": checks, "fit_scope": {"scaler_imputation_criticality_frequency": "outer-train",
                                                "rho_threshold_checkpoint": "inner-validation", "outer-test": "evaluation-only"},
               "audited_at": _now(), "outer_metrics_used_for_selection": False}
    write_json(Path(config["output"]["protocol_audit"]), payload)
    if payload["status"] != "PAPER_FINAL_PROTOCOL_AUDIT_GO": raise RuntimeError(f"final protocol audit failed: {checks}")
    return payload


def finalize(config: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    write_intermediate_tables(config); records = _result_records(config); summary, bootstrap = summarize_records(config, records)
    _write_csv(config["output"]["summary_csv"], summary); _write_csv(config["output"]["bootstrap_csv"], bootstrap)
    write_final_documents(config, records, summary, bootstrap); update_evidence_documents(config, summary, bootstrap)
    protocol_audit = final_protocol_audit(config, records)
    aliases = sum(cell["status"] == "alias" for cell in manifest["cells"]); unique = len(records) - aliases
    manifest.update({"status": "PAPER_FINAL_OUTER_RESULTS_COMPLETE_PENDING_FINAL_AUDIT", "finished_at": _now(),
                     "completed_logical_rows": len(records), "completed_unique_cells": unique, "detected_alias_cells": aliases,
                     "failed_rows": sum(cell["status"] == "failed" for cell in manifest["cells"]),
                     "protocol_audit": protocol_audit,
                     "result_files": {key: config["output"][key] for key in ("raw_csv", "groupwise_csv", "summary_csv", "bootstrap_csv")}})
    _store_manifest(config, manifest); return manifest


def preflight(config: dict[str, Any], verify_data: bool = True) -> dict[str, Any]:
    audit = validate_frozen(config); data = verify_data_hashes(audit["freeze"]) if verify_data else None
    manifest = build_manifest(config, audit)
    if data is not None:
        manifest["preflight_data_revalidation"] = data; manifest["preflight_completed_at"] = _now(); _store_manifest(config, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/paper_final_outer.yaml")
    parser.add_argument("--prepare-only", action="store_true"); parser.add_argument("--dataset", choices=("3w", "tep", "both"), default="both")
    parser.add_argument("--outer-seed", type=int); parser.add_argument("--skip-data-rehash", action="store_true"); parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    manifest = preflight(config, not args.skip_data_rehash)
    print(json.dumps({"status": manifest["status"], "expected_cells": manifest["expected_cells"], "logical_rows": manifest["logical_rows"],
                      "estimated_wall_clock_hours": manifest["estimated_wall_clock_hours"], "estimated_disk_gib": manifest["estimated_disk_gib"],
                      "output_roots": manifest["output_roots"]}, ensure_ascii=False), flush=True)
    if args.prepare_only: return
    if args.summarize_only:
        result = finalize(config, manifest); print(json.dumps({"status": result["status"], "completed_unique_cells": result["completed_unique_cells"], "aliases": result["detected_alias_cells"]}, ensure_ascii=False)); return
    device = select_device(str(config["device"])); selected = []
    datasets = ("3W", "TEP") if args.dataset == "both" else (args.dataset.upper(),)
    for dataset in datasets:
        key = "three_w" if dataset == "3W" else "tep"
        seeds = [int(args.outer_seed)] if args.outer_seed is not None else list(map(int, config[key]["outer_seeds"]))
        if any(seed not in config[key]["outer_seeds"] for seed in seeds): raise ValueError(f"unregistered outer seed for {dataset}")
        for seed in seeds:
            try:
                selected.append(run_outer_split(config, manifest, dataset, seed, device)); write_intermediate_tables(config)
            except Exception as error:
                manifest["status"] = "PAPER_FINAL_BLOCKED_BY_PROTOCOL_OR_IMPLEMENTATION"
                manifest["failures"].append({"dataset": dataset, "outer_seed": seed, "type": type(error).__name__, "message": str(error), "at": _now()})
                _store_manifest(config, manifest); raise
    write_intermediate_tables(config)
    if len(_result_records(config)) == 240:
        finalize(config, manifest)
    print(json.dumps({"completed_logical_rows": manifest.get("completed_logical_rows", 0), "selections": selected}, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()
