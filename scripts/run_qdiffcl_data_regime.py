from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml

from datasets import Run, Standardizer
from datasets.three_w import discover_instances
from frequency import fault_stages
from scripts import run_3w_clean_baseline as base3w
from scripts import run_paper_final_outer as phase_g
from scripts.audit_qdiffcl_data_regime import atomic_json, canonical_hash, sha256_file
from utils import environment_metadata


FORMAL_METHODS = (
    "NO_AUG", "UNIFORM_DIFFUSION", "JITTER_SCALING",
    "FINAL_QDIFFCL_FIXED", "CALIBRATED_RHO",
)
PHASE_METHOD = {
    "NO_AUG": "NO_AUG", "UNIFORM_DIFFUSION": "UNIFORM_DIFFUSION",
    "JITTER_SCALING": "JITTER_SCALING", "FINAL_QDIFFCL_FIXED": "FINAL_QDIFFCL",
    "CALIBRATED_RHO": "DCBR",
}
SPLITS = ("train", "validation", "test")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True, encoding="utf-8").strip()


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def choose_rho(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=lambda row: (row["macro_f1"], row["auprc"], -row["far"], -row["rho"]))


def reuse_compatible(existing: dict[str, Any], expected: dict[str, Any]) -> bool:
    required = (
        "dataset", "outer_id", "method_semantics", "weights", "model_seed", "train_hash",
        "validation_hash", "test_hash", "training_budget_hash", "preprocessing_hash",
        "evaluation_hash", "protocol_hash", "checkpoint_sha256", "prediction_sha256",
    )
    return all(existing.get(key) == expected.get(key) and existing.get(key) is not None for key in required)


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["_config_path"] = str(source)
    return config


def fraction_key(fraction: float) -> str:
    return str(float(fraction))


def fraction_token(fraction: float) -> str:
    return f"f{int(round(100 * fraction)):03d}"


def manifest_path(dataset: str, outer_id: int) -> Path:
    prefix = "3w" if dataset == "3W" else "tep"
    return Path("configs/data_regime_manifests") / f"{prefix}_outer_{outer_id}.json"


def load_fraction_manifest(dataset: str, outer_id: int, fraction: float) -> tuple[dict[str, Any], dict[str, Any]]:
    path = manifest_path(dataset, outer_id)
    manifest = read_json(path)
    claimed = manifest.pop("sha256")
    if canonical_hash(manifest) != claimed:
        raise RuntimeError(f"fraction manifest hash mismatch: {path}")
    manifest["sha256"] = claimed
    record = copy.deepcopy(manifest["fractions"][fraction_key(fraction)])
    record_claimed = record.pop("sha256")
    if canonical_hash(record) != record_claimed:
        raise RuntimeError(f"fraction record hash mismatch: {path} fraction={fraction}")
    record["sha256"] = record_claimed
    return manifest, record


def legal_dataset_fractions(config: dict[str, Any]) -> list[tuple[str, float]]:
    audit = Path("analysis/results/qdiffcl_data_regime_e_identifiability.csv")
    if not audit.exists():
        raise RuntimeError("E-identifiability audit is missing")
    import csv

    rows = list(csv.DictReader(audit.open(encoding="utf-8", newline="")))
    failed = {(row["dataset"], float(row["fraction"])) for row in rows if row["E_defined"].lower() != "true"}
    result = []
    for dataset in ("3W", "TEP"):
        for fraction in map(float, config["fractions"]):
            if (dataset, fraction) not in failed:
                result.append((dataset, fraction))
    return result


def validate_protocol(config: dict[str, Any], require_lock: bool) -> dict[str, Any]:
    expected = {
        "weights": config["algorithm"]["criticality_weights"] == {
            "weight_discriminative": .5, "weight_early": .5, "weight_run_stability": 0.,
        },
        "ratio": float(config["algorithm"]["critical_ratio"]) == .3,
        "timesteps": (
            int(config["algorithm"]["t_uniform"]), int(config["algorithm"]["t_critical"]),
            int(config["algorithm"]["t_noncritical"]),
        ) == (3, 1, 5),
        "rho_grid": list(map(float, config["rho_grid"])) == [0, .25, .5, .75, 1],
        "methods": tuple(config["methods"]) == FORMAL_METHODS,
        "outer_3w": list(map(int, config["three_w"]["outer_seeds"])) == [31001, 31002, 31003],
        "outer_tep": list(map(int, config["tep"]["outer_seeds"])) == [32001, 32002, 32003],
        "seeds_3w": config["three_w"]["model_seeds"] == [42, 43, 44, 45, 46],
        "seeds_tep": config["tep"]["model_seeds"] == [7, 42, 43, 44, 2026],
        "branch": git("branch", "--show-current") == config["git_freeze"]["branch"],
        "cuda": torch.cuda.is_available(),
    }
    if not all(expected.values()):
        raise RuntimeError(f"DATA_REGIME_SANITY_HOLD: {expected}")
    lock = None
    if require_lock:
        lock_path = Path(config["git_freeze"]["protocol_lock_manifest"])
        if not lock_path.exists():
            raise RuntimeError("DATA_REGIME_PROTOCOL_LOCK_HOLD: lock manifest missing")
        lock = read_json(lock_path)
        observed_protocol_hash = protocol_hash(config)
        if lock["protocol_hash"] != observed_protocol_hash:
            amendment_path = Path("analysis/results/qdiffcl_data_regime_runtime_amendment.json")
            amendment = read_json(amendment_path) if amendment_path.exists() else {}
            post_files = amendment.get("post_fix_files", {})
            amendment_checks = {
                "classification": amendment.get("classification") == "NUMERICALLY_EQUIVALENT_RUNTIME_AMENDMENT",
                "parent_lock": amendment.get("parent_protocol_lock") == lock["protocol_lock_commit"],
                "post_protocol_hash": amendment.get("post_protocol_hash") == observed_protocol_hash,
                "config": amendment.get("scientific_config_sha256_after") == sha256_file(config["_config_path"]),
                "tep_loader": post_files.get("datasets/tep.py") == sha256_file("datasets/tep.py"),
                "runner": post_files.get("scripts/run_qdiffcl_data_regime.py") == sha256_file(__file__),
                "equivalence": amendment.get("equivalence_evidence", {}).get("status") == "TEP_MEMORY_SAFE_LOADER_EQUIVALENCE_GO",
                "test_blind": amendment.get("test_metrics_used_to_choose_repair") is False,
            }
            if not all(amendment_checks.values()):
                raise RuntimeError(f"DATA_REGIME_PROTOCOL_LOCK_HOLD: runtime amendment invalid: {amendment_checks}")
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", lock["protocol_lock_commit"], "HEAD"], check=False,
        ).returncode == 0
        if not ancestor:
            raise RuntimeError("DATA_REGIME_PROTOCOL_LOCK_HOLD: lock commit is not an ancestor")
    return {"checks": expected, "lock": lock}


def protocol_hash(config: dict[str, Any]) -> str:
    paths = [Path(config["_config_path"]), Path("scripts/run_qdiffcl_data_regime.py"),
             Path("scripts/audit_qdiffcl_data_regime.py"), Path("scripts/summarize_qdiffcl_data_regime.py"),
             Path("tests/test_qdiffcl_data_regime.py")]
    paths.extend(sorted(Path("configs/data_regime_manifests").glob("*.json")))
    return canonical_hash({str(path): sha256_file(path) for path in paths})


def record_protocol_lock(config: dict[str, Any]) -> dict[str, Any]:
    message = git("log", "-1", "--pretty=%s")
    if message != config["git_freeze"]["protocol_lock_message"]:
        raise RuntimeError(f"HEAD is not the protocol-lock commit: {message}")
    record = {
        "status": "DATA_REGIME_PROTOCOL_LOCKED", "protocol_lock_commit": git("rev-parse", "HEAD"),
        "protocol_hash": protocol_hash(config),
        "fraction_manifest_hashes": {
            str(path): sha256_file(path) for path in sorted(Path("configs/data_regime_manifests").glob("*.json"))
        },
        "outer_test_metrics_read": False, "created_at": now(),
    }
    atomic_json(Path(config["git_freeze"]["protocol_lock_manifest"]), record)
    return record


def _context_root(config: dict[str, Any], namespace: str, dataset: str, fraction: float, outer_id: int) -> Path:
    return (Path(config["output"]["root"]) / namespace / dataset.lower() /
            fraction_token(fraction) / f"outer_{outer_id}")


def _mask_hash(context: dict[str, Any]) -> str:
    mask = np.ascontiguousarray(context["critical"]["soft_mask"])
    return hashlib.sha256(mask.tobytes()).hexdigest()


def fit_fraction_preprocessor(
    train_instances: list[Any], features: tuple[str, ...], base_config: dict[str, Any],
) -> dict[str, Any]:
    """Fit only on the fraction while preserving the frozen feature dimension.

    A channel with no finite observation in the selected source units has no
    fraction-local statistic to estimate. It is retained as an all-zero neutral
    normalized channel; no statistic from unused train units is borrowed.
    """
    fit_config = copy.deepcopy(base_config)
    fit_config["protocol"]["feature_min_train_coverage"] = 1e-12
    fitted = base3w.fit_preprocessor(train_instances, features, fit_config)
    empty = [name for name in features if name not in fitted["statistics"]]
    for name in empty:
        fitted["statistics"][name] = {
            "train_coverage": 0.0, "lower": 0.0, "upper": 0.0, "median": 0.0,
            "mean": 0.0, "scale": 1.0, "sample_count": 0,
            "fraction_local_no_observation_policy": "neutral_zero_channel",
        }
    fitted["retained_features"] = list(features)
    fitted["excluded_features"] = {}
    fitted["empty_train_features"] = empty
    fitted["coverage_threshold"] = 0.0
    fitted["fit_scope"] = "selected fraction outer-train observations only"
    return fitted


def prepare_three_w(
    config: dict[str, Any], outer_id: int, fraction: float, namespace: str,
) -> dict[str, Any]:
    manifest, fraction_record = load_fraction_manifest("3W", outer_id, fraction)
    stage = config["three_w"]
    grouped = yaml.safe_load(Path(stage["base_config"]).read_text(encoding="utf-8"))
    base = yaml.safe_load(Path(grouped["base_config"]).read_text(encoding="utf-8"))
    base3w.PRIMARY_CLASSES = phase_g.FINAL_PRIMARY_CLASSES
    base3w.CLASS_TO_TARGET = {value: index for index, value in enumerate(phase_g.FINAL_PRIMARY_CLASSES)}
    split = {name: set(manifest["outer_protocol"]["groups"][name]) for name in SPLITS}
    instances = [
        item for item in discover_instances(Path(stage["data_root"]))
        if item.source == "WELL" and item.event_class in phase_g.FINAL_PRIMARY_CLASSES
    ]
    by_instance = {item.instance_id: item for item in instances}
    selected = set(fraction_record["source_ids"])
    if not selected <= set(by_instance):
        raise RuntimeError("3W fraction references unknown source instances")
    by_split = {
        "train": [by_instance[source_id] for source_id in sorted(selected)],
        "validation": [item for item in instances if item.well_id in split["validation"]],
        "test": [item for item in instances if item.well_id in split["test"]],
    }
    if any(item.well_id not in split["train"] for item in by_split["train"]):
        raise RuntimeError("3W fraction escaped frozen outer-train wells")
    pre_cfg = copy.deepcopy(base)
    pre_cfg["protocol"]["split_seed"] = outer_id
    features = tuple(grouped["frozen_process_features"])
    preprocessor = fit_fraction_preprocessor(by_split["train"], features, pre_cfg)
    length = int(base["protocol"]["window_length"]); stride = int(base["protocol"]["stride"])
    offset = int(base["protocol"]["transient_offset"])
    refs_by_split: dict[str, list[Any]] = {}; refs_by_instance: dict[str, list[Any]] = {}
    for name, items in by_split.items():
        refs = []
        for item in items:
            current = base3w.instance_refs(item, length, stride, offset)
            refs.extend(current); refs_by_instance[item.instance_id] = current
        refs_by_split[name] = refs
    train_refs = base3w.stratified_refs(refs_by_split["train"], int(grouped["train_windows_per_class"]), outer_id)
    val_refs = base3w.stratified_refs(refs_by_split["validation"], int(grouped["validation_windows_per_class"]), outer_id + 1)
    train_x, train_y = base3w.materialize(train_refs, by_instance, preprocessor, length, False)
    val_x, val_y = base3w.materialize(val_refs, by_instance, preprocessor, length, False)

    def uid(ref: Any) -> str:
        item = by_instance[ref.instance_id]
        original = phase_g.FINAL_PRIMARY_CLASSES[ref.target] if ref.target else 0
        return f"training:fault_{original:02d}:{item.instance_id}"

    bundle = {"run_uid": np.asarray([uid(ref) for ref in train_refs]), "labels": train_y}
    critical = phase_g._criticality(train_x, bundle, phase_g._stage_from_three_w_refs(train_refs), config["algorithm"])
    training = {
        "epochs": int(grouped["pretrain_epochs"]), "probe_epochs": int(grouped["probe_epochs"]),
        "early_stopping_patience": int(config["three_w"]["training_budget"]["early_stopping_patience"]),
        "batch_size": int(grouped["batch_size"]), "learning_rate": float(grouped["learning_rate"]),
        "temperature": float(grouped["temperature"]), "device": "cuda", "supcon_batching": "original",
    }
    context = {
        "dataset": "3W", "outer_seed": outer_id, "fraction": fraction, "base": base, "grouped": grouped,
        "split": split, "by_split": by_split, "by_instance": by_instance, "refs_by_instance": refs_by_instance,
        "train_refs": train_refs, "validation_refs": val_refs, "preprocessor": preprocessor,
        "train": train_x, "validation": val_x, "labels": {"train": train_y, "validation": val_y},
        "ids": {
            "train": np.asarray([f"{r.instance_id}:{r.start}:{r.target}" for r in train_refs]),
            "validation": np.asarray([f"{r.instance_id}:{r.start}:{r.target}" for r in val_refs]),
        },
        "training": training, "critical": critical, "fraction_manifest": manifest,
        "fraction_record": fraction_record,
    }
    context["context_hash"] = canonical_hash({
        "dataset": "3W", "outer_id": outer_id, "fraction": fraction,
        "fraction_hash": fraction_record["sha256"], "groups": manifest["outer_protocol"]["groups"],
        "preprocessor": preprocessor, "critical_mask": critical["soft_mask"].tolist(),
    })
    write_context_audit(config, context, namespace)
    return context


def prepare_tep(config: dict[str, Any], outer_id: int, fraction: float, namespace: str) -> dict[str, Any]:
    manifest, fraction_record = load_fraction_manifest("TEP", outer_id, fraction)
    stage = config["tep"]
    base = yaml.safe_load(Path(stage["base_config"]).read_text(encoding="utf-8"))
    phase_g._configure(base)
    by_uid = {run.run_uid: run for run in phase_g._load_selected_tep_runs(config)}
    selected = set(fraction_record["source_ids"])
    frozen = manifest["outer_protocol"]["groups"]
    if not selected <= set(frozen["train"]):
        raise RuntimeError("TEP fraction escaped frozen outer-train Runs")
    groups = {
        "train": [by_uid[uid] for uid in sorted(selected)],
        "validation": [by_uid[uid] for uid in frozen["validation"]],
        "test": [by_uid[uid] for uid in frozen["test"]],
    }
    scaler = Standardizer().fit_many([run.values for run in groups["train"]])
    bundles = {name: phase_g._window_bundle(groups[name], scaler, base) for name in SPLITS}
    stages = {name: fault_stages(bundles[name], base) for name in SPLITS}
    critical = phase_g._criticality(bundles["train"]["clean"], bundles["train"], stages["train"], config["algorithm"])
    runtime = phase_g._runtime(base, 0)
    runtime["diagnosis"] = {"threshold_band_width": .05, "high_correlation_quantile": .90}
    context = {
        "dataset": "TEP", "outer_seed": outer_id, "fraction": fraction, "base": base,
        "split": {name: set(frozen[name]) for name in SPLITS}, "bundles": bundles, "stages": stages,
        "critical": critical, "runtime": runtime, "train": bundles["train"]["clean"],
        "validation": bundles["validation"]["clean"], "labels": {name: bundles[name]["labels"] for name in SPLITS},
        "ids": {name: bundles[name]["window_id"] for name in SPLITS},
        "fraction_manifest": manifest, "fraction_record": fraction_record,
    }
    scaler_payload = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(), "fit_groups": sorted(selected)}
    context["context_hash"] = canonical_hash({
        "dataset": "TEP", "outer_id": outer_id, "fraction": fraction,
        "fraction_hash": fraction_record["sha256"], "groups": frozen,
        "scaler": scaler_payload, "critical_mask": critical["soft_mask"].tolist(),
    })
    context["scaler_payload"] = scaler_payload
    write_context_audit(config, context, namespace)
    return context


def write_context_audit(config: dict[str, Any], context: dict[str, Any], namespace: str) -> None:
    root = _context_root(config, namespace, context["dataset"], context["fraction"], context["outer_seed"])
    criticality_path = root / "_context" / "criticality.npz"
    _atomic_npz(criticality_path, {
        key: np.asarray(context["critical"][key])
        for key in ("discriminative", "early", "composite", "soft_mask")
    })
    composite = np.asarray(context["critical"]["composite"])
    top_count = max(1, int(round(composite.size * float(config["algorithm"]["critical_ratio"]))))
    top_flat = np.argsort(composite.reshape(-1), kind="stable")[-top_count:]
    payload = {
        "dataset": context["dataset"], "outer_id": context["outer_seed"], "fraction": context["fraction"],
        "context_hash": context["context_hash"], "fraction_manifest_hash": context["fraction_record"]["sha256"],
        "selected_train_source_ids": context["fraction_record"]["source_ids"],
        "validation_groups": sorted(context["split"]["validation"]), "test_groups": sorted(context["split"]["test"]),
        "normalization_fit_scope": "selected fraction outer-train source units only",
        "criticality_fit_scope": "selected fraction outer-train windows only",
        "criticality_mask_sha256": _mask_hash(context),
        "criticality_path": str(criticality_path), "criticality_sha256": sha256_file(criticality_path),
        "top_frequency_flat_indices": sorted(map(int, top_flat)),
        "mask_ratio": float(config["algorithm"]["critical_ratio"]),
        "criticality_weights": context["critical"]["component_weights"],
        "train_windows": len(context["train"]), "validation_windows": len(context["validation"]),
        "outer_test_read": False,
    }
    if context["dataset"] == "TEP":
        payload["scaler"] = context["scaler_payload"]
    else:
        payload["preprocessor_sha256"] = canonical_hash(context["preprocessor"])
        payload["empty_train_features"] = context["preprocessor"]["empty_train_features"]
    atomic_json(root / "_context" / "audit.json", payload)


def prepare_context(config: dict[str, Any], dataset: str, outer_id: int, fraction: float, namespace: str) -> dict[str, Any]:
    if dataset == "3W":
        return prepare_three_w(config, outer_id, fraction, namespace)
    return prepare_tep(config, outer_id, fraction, namespace)


def _cell_root(
    config: dict[str, Any], namespace: str, dataset: str, fraction: float, outer_id: int,
    model_seed: int, method: str,
) -> Path:
    return (_context_root(config, namespace, dataset, fraction, outer_id) /
            f"model_seed_{model_seed}" / method)


def _training_accounting(context: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    runtime = context["training"] if context["dataset"] == "3W" else context["runtime"]
    pre_epochs = len(record.get("pretrain_history", [])); probe_epochs = len(record.get("probe_history", []))
    samples = len(context["train"]); batch = int(runtime["batch_size"])
    return {
        "configured_epoch_cap": int(runtime["epochs"]), "configured_probe_epoch_cap": int(runtime["probe_epochs"]),
        "actual_epochs_run": pre_epochs, "actual_probe_epochs_run": probe_epochs,
        "optimizer_steps": math.ceil(samples / batch) * pre_epochs,
        "probe_optimizer_steps": math.ceil(samples / batch) * probe_epochs,
        "examples_seen": samples * pre_epochs, "probe_examples_seen": samples * probe_epochs,
        "batch_size": batch, "early_stop_epoch": pre_epochs,
        "early_stop_reason": "patience_or_epoch_cap" if pre_epochs < int(runtime["epochs"]) else "epoch_cap",
    }


def train_validation(
    config: dict[str, Any], context: dict[str, Any], namespace: str, method: str,
    model_seed: int, device: str, rho: float | None = None,
) -> tuple[dict[str, Any], Path]:
    phase_method = PHASE_METHOD[method]
    root = _cell_root(config, namespace, context["dataset"], context["fraction"], context["outer_seed"], model_seed, method)
    if method == "CALIBRATED_RHO":
        if rho is None:
            raise ValueError("CALIBRATED_RHO requires rho")
        root = root / "_candidates" / phase_g.rho_name(rho)
    else:
        root = root / "_training"
    already_complete = (root / "model.pt").exists() and (root / "validation.json").exists()
    record = phase_g.train_method(context, config, phase_method, model_seed, device, root, rho)
    if already_complete and "data_regime" in record:
        return record, root
    record["data_regime"] = {
        "fraction": context["fraction"], "fraction_manifest_hash": context["fraction_record"]["sha256"],
        "criticality_mask_sha256": _mask_hash(context), "config_sha256": sha256_file(config["_config_path"]),
        "training_accounting": _training_accounting(context, record), "outer_test_read": False,
    }
    atomic_json(root / "validation.json", record)
    return record, root


def select_rho(
    config: dict[str, Any], context: dict[str, Any], namespace: str, device: str,
    rhos: Iterable[float] | None = None, seeds: Iterable[int] | None = None,
) -> dict[str, Any]:
    dataset = context["dataset"]
    values = list(map(float, config["rho_grid"] if rhos is None else rhos))
    selection_seeds = list(map(int, config["three_w" if dataset == "3W" else "tep"]["rho_selection_seeds"] if seeds is None else seeds))
    records: dict[int, dict[float, dict[str, Any]]] = {}
    candidate_number = 0
    for seed in selection_seeds:
        records[seed] = {}
        for rho in values:
            candidate_number += 1
            heartbeat(
                config, dataset=dataset, fraction=context["fraction"], outer=context["outer_seed"],
                method="CALIBRATED_RHO", seed=seed, rho=rho, stage="rho_validation_training",
                completed=candidate_number - 1, expected=len(selection_seeds) * len(values),
                last_successful_cell=None,
            )
            if rho == 1.0:
                record, _ = train_validation(config, context, namespace, "FINAL_QDIFFCL_FIXED", seed, device)
            else:
                record, _ = train_validation(config, context, namespace, "CALIBRATED_RHO", seed, device, rho)
            if record.get("outer_test_read") or not record.get("validation_only"):
                raise RuntimeError("rho candidate crossed the validation-only boundary")
            records[seed][rho] = record
    rows = []
    for rho in values:
        metrics = [records[seed][rho]["validation"] for seed in selection_seeds]
        rows.append({
            "rho": rho, "model_seeds": selection_seeds,
            "macro_f1": float(np.mean([row["macro_f1"] for row in metrics])),
            "auprc": float(np.mean([row["auprc"] for row in metrics])),
            "far": float(np.mean([row["far"] for row in metrics])),
        })
    chosen = choose_rho(rows)
    selection = {
        "dataset": dataset, "fraction": context["fraction"], "outer_id": context["outer_seed"],
        "selection_split": "outer-validation only", "selection_seeds": selection_seeds,
        "rho_grid": values, "candidate_rows": rows, "selected_rho": chosen["rho"],
        "tie_breaker": config["rho_selection_order"], "outer_test_read": False,
        "historical_dcbr_global_rho": config["historical_dcbr_global_rho"][dataset],
        "lineage_name": "DATA_REGIME_RHO_STAR", "selected_at": now(),
    }
    path = _context_root(config, namespace, dataset, context["fraction"], context["outer_seed"]) / "rho_selection.json"
    if path.exists():
        old = read_json(path)
        comparable = {key: selection[key] for key in selection if key not in {"selected_at"}}
        old_comparable = {key: old[key] for key in old if key not in {"selected_at"}}
        if old_comparable != comparable:
            raise RuntimeError("frozen rho selection changed on resume")
        return old
    atomic_json(path, selection)
    return selection


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def evaluate_once(
    config: dict[str, Any], context: dict[str, Any], namespace: str, method: str,
    model_seed: int, device: str, source: Path, selected_rho: float | None,
) -> dict[str, Any]:
    root = _cell_root(config, namespace, context["dataset"], context["fraction"], context["outer_seed"], model_seed, method)
    result_path = root / "result.json"; prediction_path = root / "predictions.npz"; started_path = root / "outer_test_started.json"
    if result_path.exists() and prediction_path.exists():
        record = read_json(result_path)
        if record["prediction_sha256"] != sha256_file(prediction_path) or record["checkpoint_sha256"] != sha256_file(source / "model.pt"):
            raise RuntimeError("completed result hash mismatch")
        return record
    if result_path.exists() or prediction_path.exists() or started_path.exists():
        raise RuntimeError("outer-test once guard: incomplete evaluation requires audit, not rerun")
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(started_path, {"started_at": now(), "outer_test_evaluated_once": True})
    model, payload = phase_g._load_model(context, source / "model.pt", device)
    threshold = payload.get("threshold")
    if context["dataset"] == "3W":
        metrics, raw, groupwise = phase_g.evaluate_three_w(model, context, device)
    else:
        if threshold is None:
            raise RuntimeError("TEP requires a validation-selected threshold")
        metrics, raw, groupwise = phase_g.evaluate_tep(model, context, float(threshold), device)
    _atomic_npz(prediction_path, raw)
    record = {
        "cell_id": cell_id(context["dataset"], context["fraction"], context["outer_seed"], method, model_seed),
        "dataset": context["dataset"], "fraction": context["fraction"], "outer_id": context["outer_seed"],
        "method": method, "model_seed": model_seed, "selected_rho": selected_rho, "threshold": threshold,
        "validation_metrics_path": str(source / "validation.json"), "test_metrics": metrics, "groupwise": groupwise,
        "checkpoint_path": str(source / "model.pt"), "checkpoint_sha256": sha256_file(source / "model.pt"),
        "prediction_path": str(prediction_path), "prediction_sha256": sha256_file(prediction_path),
        "config_sha256": sha256_file(config["_config_path"]),
        "fraction_manifest_hash": context["fraction_record"]["sha256"],
        "criticality_mask_sha256": _mask_hash(context), "context_hash": context["context_hash"],
        "source_commit": git("rev-parse", "HEAD"), "environment": environment_metadata(),
        "outer_test_evaluated_once": True, "completed_at": now(),
    }
    record["result_payload_sha256"] = canonical_hash(record)
    atomic_json(result_path, record)
    return record


def cell_id(dataset: str, fraction: float, outer_id: int, method: str, seed: int) -> str:
    return f"{dataset.lower()}-{fraction_token(fraction)}-outer{outer_id}-seed{seed}-{method.lower()}"


def accounting(config: dict[str, Any]) -> dict[str, int]:
    pairs = legal_dataset_fractions(config)
    formal = sum(len(config["three_w" if dataset == "3W" else "tep"]["outer_seeds"]) *
                 len(FORMAL_METHODS) * len(config["three_w" if dataset == "3W" else "tep"]["model_seeds"])
                 for dataset, _ in pairs)
    rho = sum(len(config["three_w" if dataset == "3W" else "tep"]["outer_seeds"]) *
              len(config["rho_grid"]) * len(config["three_w" if dataset == "3W" else "tep"]["rho_selection_seeds"])
              for dataset, _ in pairs)
    return {
        "formal_cells_expected": formal, "formal_cells_reused": 0, "formal_cells_new": formal,
        "rho_candidate_cells_expected": rho, "rho_candidate_cells_reused": 0, "rho_candidate_cells_new": rho,
        "completed": 0, "remaining": formal + rho, "duplicate_count": 0,
    }


def build_run_manifest(config: dict[str, Any]) -> dict[str, Any]:
    path = Path(config["output"]["manifest"])
    expected = accounting(config)
    if path.exists():
        manifest = read_json(path)
        current_hash = protocol_hash(config)
        if manifest["protocol_hash"] != current_hash:
            lock = read_json(config["git_freeze"]["protocol_lock_manifest"])
            if manifest["protocol_hash"] != lock["protocol_hash"]:
                raise RuntimeError("run manifest is not anchored to the original protocol lock")
        return manifest
    cells = []
    for dataset, fraction in legal_dataset_fractions(config):
        stage = config["three_w" if dataset == "3W" else "tep"]
        for outer_id in stage["outer_seeds"]:
            for method in FORMAL_METHODS:
                for seed in stage["model_seeds"]:
                    cells.append({
                        "cell_id": cell_id(dataset, fraction, outer_id, method, seed), "dataset": dataset,
                        "fraction": fraction, "outer_id": outer_id, "method": method, "model_seed": seed,
                        "status": "pending",
                    })
    manifest = {
        "status": "QDIFFCL_DATA_REGIME_V1_RESUMABLE", "evidence_class": config["evidence_class"],
        "protocol_hash": protocol_hash(config), "protocol_lock": read_json(config["git_freeze"]["protocol_lock_manifest"]),
        "accounting": expected, "cells": cells, "failures": [], "created_at": now(),
    }
    atomic_json(path, manifest)
    return manifest


def update_run_manifest(config: dict[str, Any], manifest: dict[str, Any], record: dict[str, Any]) -> None:
    matches = [row for row in manifest["cells"] if row["cell_id"] == record["cell_id"]]
    if len(matches) != 1:
        raise RuntimeError("formal cell missing or duplicated in manifest")
    matches[0].update({
        "status": "complete", "result_path": str(_cell_root(
            config, config["output"]["namespace"], record["dataset"], record["fraction"], record["outer_id"],
            record["model_seed"], record["method"],
        ) / "result.json"), "result_sha256": record["result_payload_sha256"], "completed_at": record["completed_at"],
    })
    complete = sum(row["status"] == "complete" for row in manifest["cells"])
    manifest["accounting"]["completed"] = complete
    manifest["accounting"]["remaining"] = len(manifest["cells"]) - complete
    atomic_json(Path(config["output"]["manifest"]), manifest)


def heartbeat(config: dict[str, Any], **values: Any) -> None:
    payload = {
        "timestamp": now(), "pid": os.getpid(), "failed": 0,
        "last_artifact_write_time": now(), **values,
    }
    atomic_json(Path(config["output"]["runtime_status"]), payload)


def run_smoke_core(config: dict[str, Any], device: str) -> None:
    dataset, fraction, outer_id, seed = "3W", .10, 31001, 42
    context = prepare_context(config, dataset, outer_id, fraction, config["output"]["smoke_namespace"])
    for method in ("NO_AUG", "FINAL_QDIFFCL_FIXED"):
        record, _ = train_validation(config, context, config["output"]["smoke_namespace"], method, seed, device)
        if not all(np.isfinite(float(record["validation"][key])) for key in ("macro_f1", "auprc", "far")):
            raise RuntimeError("non-finite smoke metric")
    atomic_json(_context_root(config, config["output"]["smoke_namespace"], dataset, fraction, outer_id) /
                "core_smoke.json", {"status": "DATA_REGIME_CORE_SMOKE_GO", "outer_test_read": False})


def run_smoke_rho(config: dict[str, Any], device: str) -> None:
    namespace = config["output"]["smoke_namespace"]
    context = prepare_context(config, "3W", 31001, .10, namespace)
    selection = select_rho(config, context, namespace, device, rhos=[0., 1.], seeds=[42])
    if selection["outer_test_read"] or selection["rho_grid"] != [0., 1.]:
        raise RuntimeError("rho smoke guard failed")
    atomic_json(_context_root(config, namespace, "3W", .10, 31001) / "rho_smoke.json",
                {"status": "DATA_REGIME_RHO_SMOKE_GO", "outer_test_read": False})


def run_formal_context(
    config: dict[str, Any], manifest: dict[str, Any], dataset: str, fraction: float,
    outer_id: int, device: str, context: dict[str, Any] | None = None,
) -> None:
    namespace = config["output"]["namespace"]
    if context is None:
        context = prepare_context(config, dataset, outer_id, fraction, namespace)
    selection = select_rho(config, context, namespace, device)
    rho = float(selection["selected_rho"])
    stage = config["three_w" if dataset == "3W" else "tep"]
    for seed in map(int, stage["model_seeds"]):
        sources: dict[str, Path] = {}
        for method in FORMAL_METHODS[:-1]:
            _, sources[method] = train_validation(config, context, namespace, method, seed, device)
        if rho == 1.0:
            sources["CALIBRATED_RHO"] = sources["FINAL_QDIFFCL_FIXED"]
        else:
            _, sources["CALIBRATED_RHO"] = train_validation(
                config, context, namespace, "CALIBRATED_RHO", seed, device, rho,
            )
        for method in FORMAL_METHODS:
            heartbeat(config, dataset=dataset, fraction=fraction, outer=outer_id, method=method, seed=seed,
                      stage="locked_outer_test", completed=manifest["accounting"]["completed"],
                      expected=manifest["accounting"]["formal_cells_expected"], last_successful_cell=None)
            record = evaluate_once(config, context, namespace, method, seed, device, sources[method], rho if method == "CALIBRATED_RHO" else None)
            update_run_manifest(config, manifest, record)


def selected_scope(config: dict[str, Any], args: argparse.Namespace) -> list[tuple[str, float, int]]:
    result = []
    for dataset, fraction in legal_dataset_fractions(config):
        if args.dataset and dataset != args.dataset:
            continue
        if args.fraction is not None and not math.isclose(fraction, args.fraction):
            continue
        stage = config["three_w" if dataset == "3W" else "tep"]
        for outer_id in map(int, stage["outer_seeds"]):
            if args.outer_id is None or outer_id == args.outer_id:
                result.append((dataset, fraction, outer_id))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen Q-DiffCL Data-Regime runner")
    parser.add_argument("--config", default="configs/qdiffcl_data_regime_v1.yaml")
    parser.add_argument("--stage", choices=("accounting", "smoke-core", "smoke-rho", "lock-record", "rho-selection", "formal", "all"), default="accounting")
    parser.add_argument("--dataset", choices=("3W", "TEP"))
    parser.add_argument("--fraction", type=float, choices=(1.0, .25, .10))
    parser.add_argument("--outer-id", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = load_config(args.config)
    require_lock = args.stage in {"rho-selection", "formal", "all"}
    validate_protocol(config, require_lock=require_lock)
    print(json.dumps(accounting(config), indent=2))
    if args.stage == "accounting":
        return
    if args.stage == "smoke-core":
        run_smoke_core(config, args.device); return
    if args.stage == "smoke-rho":
        run_smoke_rho(config, args.device); return
    if args.stage == "lock-record":
        print(json.dumps(record_protocol_lock(config), indent=2, ensure_ascii=False)); return
    scopes = selected_scope(config, args)
    run_manifest = build_run_manifest(config) if args.stage in {"formal", "all"} else None
    for dataset, fraction, outer_id in scopes:
        heartbeat(config, dataset=dataset, fraction=fraction, outer=outer_id, method=None, seed=None,
                  stage="prepare_fraction_local_context", completed=0, expected=len(scopes),
                  last_successful_cell=None)
        context = prepare_context(config, dataset, outer_id, fraction, config["output"]["namespace"])
        if args.stage in {"rho-selection", "all"}:
            select_rho(config, context, config["output"]["namespace"], args.device)
        if args.stage in {"formal", "all"}:
            assert run_manifest is not None
            run_formal_context(config, run_manifest, dataset, fraction, outer_id, args.device, context)


if __name__ == "__main__":
    main()
