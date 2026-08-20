from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

import scripts.run_3w_clean_baseline as base3w
from baselines.external_augmentations import FreRAAdapter, traditional_view
from datasets.three_w import discover_instances
from frequency import fault_stages
from losses import quality_weighted_supervised_contrastive_loss
from metrics import representation_diagnostics
from scripts.diagnose_frequency_selective_far import score_profile
from scripts.run_3w_clean_collapse_diagnosis import train_probe
from scripts.run_3w_diffusion_1seed import METHODS as THREE_W_METHODS, state_hash, supcon_orders
from scripts.run_3w_final_primary_grouped import FINAL_PRIMARY_CLASSES, build_model as build_3w_model
from scripts.run_diffusion_quality_retest import (
    _fit_probe, _fit_supcon, _metrics, _probabilities, _state_hash, best_probe_record,
    epoch_orders, load_fixed_views,
)
from scripts.run_frequency_selective_r1_3seed import _stage_metrics, sha256_strings
from scripts.run_stage_frequency_diffusion_mvp import (
    _configure, _runtime, detection_delays, early_fault_recall,
)
from trainers import build_model
from trainers.balanced import sqrt_inverse_frequency_weights
from utils import seed_everything, select_device, write_json


TRADITIONAL = ("NO_AUG", "JITTER", "SCALING", "JITTER_SCALING")
REUSED = ("UNIFORM_DIFFUSION", "FINAL_QDIFFCL")
FRERA = "FRERA"


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    final = yaml.safe_load(Path(config["final_config"]).read_text(encoding="utf-8"))
    expected = {"weight_discriminative": .5, "weight_early": .5, "weight_run_stability": 0.0}
    if not final.get("frozen") or final.get("weights") != expected:
        raise ValueError("FINAL_QDIFFCL is not frozen at 0.5D+0.5E")
    reliability = yaml.safe_load(Path(config["final_reliability_config"]).read_text(encoding="utf-8"))
    spectral = reliability["spectral_diffusion"]
    frozen = {"t_uniform": 3, "t_critical": 1, "t_noncritical": 5,
              "preserve_phase": True, "preserve_dc": True, "noise_structure": "iid"}
    if any(spectral[key] != value for key, value in frozen.items()):
        raise ValueError("frozen spectral diffusion settings changed")
    if list(config["methods"]["tier1"]) != [*TRADITIONAL, *REUSED]:
        raise ValueError("Tier 1 method set or order changed")
    if config["methods"]["tier2"] != [FRERA]:
        raise ValueError("Tier 2 method set changed")
    source = config["frera"]
    if source["source_commit"] != FreRAAdapter.OFFICIAL_COMMIT:
        raise ValueError("FreRA source commit differs from adapter audit")


def _manifest(config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = Path(config["output"]["manifest"])
    payload = _read(path) if path.exists() else {
        "stage": "EXTERNAL_BASELINES", "results": {}, "failures": {},
        "final_reopened": False, "test_used_for_hyperparameter_tuning": False,
    }
    return path, payload


def _store(path: Path, manifest: dict[str, Any], dataset: str, method: str, seed: int,
           record: dict[str, Any], fairness: dict[str, Any], source: str, training: str) -> None:
    manifest["results"][f"{dataset}|{method}|{seed}"] = {
        "dataset": dataset, "method": method, "seed": int(seed), "record": record,
        "fairness": fairness, "source": source, "training": training,
        "test_used_for_hyperparameter_tuning": False,
    }
    write_json(path, manifest)


def _effect_audit(clean: np.ndarray, changed: np.ndarray) -> dict[str, Any]:
    delta = np.asarray(changed, np.float64) - np.asarray(clean, np.float64)
    scale = max(float(np.asarray(clean, np.float64).std()), 1e-6)
    return {"finite": bool(np.isfinite(changed).all()), "shape_preserved": changed.shape == clean.shape,
            "changed_fraction": float(np.mean(delta != 0)),
            "normalized_l1": float(np.mean(np.abs(delta)) / scale),
            "augmentation_effective": bool(np.any(delta != 0))}


def _batch_indices(order: np.ndarray, size: int):
    for start in range(0, len(order), size):
        yield order[start:start + size]


def _frera_epoch(model: torch.nn.Module, augmenter: FreRAAdapter, clean: np.ndarray,
                 labels: np.ndarray, order: np.ndarray, runtime: dict[str, Any], device: str,
                 model_optimizer: torch.optim.Optimizer | None,
                 f_optimizer: torch.optim.Optimizer | None,
                 settings: dict[str, Any]) -> tuple[float, float]:
    training = model_optimizer is not None
    model.train(training); augmenter.train(training); losses = []; regularizers = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for indices in _batch_indices(order, int(runtime["batch_size"])):
            x = torch.from_numpy(clean[indices]).float().to(device)
            y = torch.from_numpy(labels[indices]).long().to(device)
            if training:
                model_optimizer.zero_grad(); f_optimizer.zero_grad()
            changed = augmenter(x, float(settings["f_temperature"]))
            features = torch.cat([model(x)["projection"], model(changed)["projection"]], 0)
            pair_y = torch.cat([y, y], 0); weights = torch.ones(len(pair_y), device=device)
            contrastive = quality_weighted_supervised_contrastive_loss(
                features, pair_y, weights, float(runtime["temperature"])
            )
            regularizer = augmenter.l1_regularizer() * float(settings["l1_weight"]) / clean.shape[-1]
            loss = contrastive + regularizer
            if training:
                loss.backward(); model_optimizer.step(); f_optimizer.step()
            losses.append(float(loss.detach())); regularizers.append(float(regularizer.detach()))
    return float(np.mean(losses)), float(np.mean(regularizers))


def _fit_frera(model: torch.nn.Module, train_x: np.ndarray, train_y: np.ndarray,
               val_x: np.ndarray, val_y: np.ndarray, orders: list[np.ndarray],
               runtime: dict[str, Any], device: str, settings: dict[str, Any]) -> tuple[list[dict[str, Any]], FreRAAdapter]:
    augmenter = FreRAAdapter(train_x.shape[-1]).to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=float(runtime["learning_rate"]))
    f_optimizer = torch.optim.AdamW(augmenter.parameters(), lr=float(settings["f_learning_rate"]))
    best_loss = float("inf"); best_model = best_aug = None; stale = 0; history = []
    validation_order = np.arange(len(val_y))
    for epoch, order in enumerate(orders):
        loss, regularizer = _frera_epoch(model, augmenter, train_x, train_y, order, runtime, device,
                                         model_optimizer, f_optimizer, settings)
        val_loss, val_regularizer = _frera_epoch(model, augmenter, val_x, val_y, validation_order,
                                                 runtime, device, None, None, settings)
        history.append({"epoch": epoch, "loss": loss, "validation_supcon_loss": val_loss,
                        "l1_regularizer": regularizer, "validation_l1_regularizer": val_regularizer})
        if val_loss < best_loss - 1e-6:
            best_loss, stale = val_loss, 0
            best_model = copy.deepcopy(model.state_dict()); best_aug = copy.deepcopy(augmenter.state_dict())
        else:
            stale += 1
            if stale >= int(runtime["early_stopping_patience"]):
                break
    if best_model is None or best_aug is None:
        raise RuntimeError("FreRA validation selection produced no checkpoint")
    model.load_state_dict(best_model); augmenter.load_state_dict(best_aug)
    return history, augmenter


def _prepare_three_w(stage: dict[str, Any], data_root: Path, seed: int, device: str) -> dict[str, Any]:
    config = yaml.safe_load(Path(stage["base_config"]).read_text(encoding="utf-8"))
    grouped = Path(config["grouped_output"]); split_index = int(config["canonical_split_index"])
    split_manifest = _read(grouped / "grouped_split_manifest.json")["splits"][split_index]
    split = {name: set(wells) for name, wells in split_manifest["wells"].items()}
    grouped_config = yaml.safe_load(Path(config["base_config"]).read_text(encoding="utf-8"))
    base = yaml.safe_load(Path(grouped_config["base_config"]).read_text(encoding="utf-8"))
    base3w.PRIMARY_CLASSES = FINAL_PRIMARY_CLASSES
    base3w.CLASS_TO_TARGET = {value: index for index, value in enumerate(FINAL_PRIMARY_CLASSES)}
    instances = [item for item in discover_instances(data_root)
                 if item.source == "WELL" and item.event_class in FINAL_PRIMARY_CLASSES]
    by_instance = {item.instance_id: item for item in instances}
    by_split = {name: [item for item in instances if item.well_id in wells] for name, wells in split.items()}
    preprocessor = _read(grouped / f"split_{split_index:02d}" / "preprocessor.json")
    refs_by_split: dict[str, list[Any]] = {}; refs_by_instance: dict[str, list[Any]] = {}
    for name, items in by_split.items():
        refs = []
        for item in items:
            current = base3w.instance_refs(item, int(base["protocol"]["window_length"]),
                                           int(base["protocol"]["stride"]),
                                           int(base["protocol"]["transient_offset"]))
            refs.extend(current); refs_by_instance[item.instance_id] = current
        refs_by_split[name] = refs
    protocol_seed = int(config.get("protocol_seed", 42))
    train_refs = base3w.stratified_refs(refs_by_split["train"], int(grouped_config["train_windows_per_class"]), protocol_seed)
    val_refs = base3w.stratified_refs(refs_by_split["validation"], int(grouped_config["validation_windows_per_class"]), protocol_seed + 1)
    length = int(base["protocol"]["window_length"])
    train_x, train_y = base3w.materialize(train_refs, by_instance, preprocessor, length, False)
    val_x, val_y = base3w.materialize(val_refs, by_instance, preprocessor, length, False)
    training = dict(config["training"]); training["supcon_batching"] = "original"
    well_ids = np.asarray([by_instance[ref.instance_id].well_id for ref in train_refs], dtype=object)
    orders, sampler = supcon_orders(train_y, training, seed, well_ids)
    seed_everything(seed); template = build_3w_model(base["training"]["model"], train_x.shape[1], device)
    initial = copy.deepcopy(template.state_dict())
    ref_hash = hashlib.sha256("\n".join(
        f"{ref.instance_id}:{ref.start}:{ref.target}" for ref in train_refs + val_refs).encode()).hexdigest()
    ids = {"train": np.asarray([f"{r.instance_id}:{r.start}:{r.target}" for r in train_refs]),
           "validation": np.asarray([f"{r.instance_id}:{r.start}:{r.target}" for r in val_refs])}
    return {"config": config, "base": base, "train_x": train_x, "train_y": train_y,
            "val_x": val_x, "val_y": val_y, "ids": ids, "orders": orders, "sampler": sampler,
            "initial": initial, "by_split": by_split, "refs_by_instance": refs_by_instance,
            "preprocessor": preprocessor, "ref_hash": ref_hash}


def _three_w_fairness(context: dict[str, Any]) -> dict[str, Any]:
    return {"initialization_sha256": state_hash(context["initial"]),
            "window_refs_sha256": context["ref_hash"],
            "supcon_batch_order_sha256": context["sampler"]["batch_order_sha256"],
            "same_grouped_split": True, "same_train_only_preprocessor": True,
            "same_clean_windows_and_labels": True, "same_balanced_probe": True}


def _evaluate_three_w(model: torch.nn.Module, context: dict[str, Any], device: str) -> tuple[dict[str, Any], list[Any]]:
    evaluation = copy.deepcopy(context["base"])
    evaluation["protocol"]["append_missing_mask"] = False
    evaluation["training"]["batch_size"] = int(context["config"]["training"]["batch_size"])
    return base3w.evaluate_stream(model, context["by_split"]["test"], context["refs_by_instance"],
                                  context["preprocessor"], evaluation, device)


def _train_three_w_method(method: str, seed: int, context: dict[str, Any], stage: dict[str, Any],
                          config: dict[str, Any], device: str) -> dict[str, Any]:
    output = Path(stage["output_dir"]) / f"seed_{seed}" / method
    metrics_path = output / "metrics.json"; checkpoint = output / "model.pt"
    if metrics_path.exists() and checkpoint.exists():
        return _read(metrics_path)
    if metrics_path.exists() or checkpoint.exists():
        raise RuntimeError(f"incomplete output: {output}")
    started = time.perf_counter(); seed_everything(seed)
    model = build_3w_model(context["base"]["training"]["model"], context["train_x"].shape[1], device)
    model.load_state_dict(context["initial"])
    if method in TRADITIONAL:
        settings = config["traditional"]
        train_view = traditional_view(context["train_x"], context["ids"]["train"], method, seed,
                                      settings["jitter_std"], settings["scaling_std"])
        val_view = traditional_view(context["val_x"], context["ids"]["validation"], method, seed + 100,
                                    settings["jitter_std"], settings["scaling_std"])
        pretrain = _fit_supcon(model,
            {"clean": context["train_x"], "restored": train_view, "labels": context["train_y"]},
            {"clean": context["val_x"], "restored": val_view, "labels": context["val_y"]},
            np.ones(len(context["train_y"]), np.float32), np.ones(len(context["val_y"]), np.float32),
            context["orders"], context["config"]["training"], device)
        audit = {"train": _effect_audit(context["train_x"], train_view),
                 "validation": _effect_audit(context["val_x"], val_view)}
    elif method == FRERA:
        pretrain, augmenter = _fit_frera(model, context["train_x"], context["train_y"],
                                         context["val_x"], context["val_y"], context["orders"],
                                         context["config"]["training"], device, config["frera"])
        augmenter.eval()
        with torch.no_grad():
            train_view = augmenter(torch.from_numpy(context["train_x"][:1024]).to(device)).cpu().numpy()
            val_view = augmenter(torch.from_numpy(context["val_x"][:1024]).to(device)).cpu().numpy()
        audit = {"train": _effect_audit(context["train_x"][:1024], train_view),
                 "validation": _effect_audit(context["val_x"][:1024], val_view),
                 "official_commit": FreRAAdapter.OFFICIAL_COMMIT,
                 "setting": "shared_backbone_adaptation"}
    else:
        raise ValueError(method)
    probe = train_probe(model, context["train_x"], context["train_y"], context["val_x"], context["val_y"],
                        sqrt_inverse_frequency_weights(context["train_y"]),
                        int(context["config"]["training"]["probe_epochs"]),
                        float(context["config"]["training"]["learning_rate"]),
                        int(context["config"]["training"]["batch_size"]), seed, device)
    metrics, per_instance = _evaluate_three_w(model, context, device)
    record = {"metrics": metrics, "per_instance": per_instance, "pretrain_history": pretrain,
              "probe_history": probe, "augmentation_audit": audit,
              "initialization_sha256": state_hash(context["initial"]),
              "training_seconds": time.perf_counter() - started,
              "peak_gpu_mib": torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0}
    output.mkdir(parents=True, exist_ok=False); torch.save(model.state_dict(), checkpoint); write_json(metrics_path, record)
    return record


def _prepare_tep(stage: dict[str, Any], seed: int, device: str) -> dict[str, Any]:
    base = yaml.safe_load(Path(stage["base_config"]).read_text(encoding="utf-8")); _configure(base)
    views, _ = load_fixed_views(base); clean = {split: views[split]["clean"].astype(np.float32) for split in views}
    stages = {split: fault_stages(views[split], base) for split in views}; runtime = _runtime(base, seed)
    runtime["diagnosis"] = {"threshold_band_width": .05, "high_correlation_quantile": .90}
    pretrain_orders = epoch_orders(len(clean["train"]), int(runtime["epochs"]), seed + 10000)
    probe_orders = epoch_orders(len(clean["train"]), int(runtime["probe_epochs"]), seed + 20000)
    seed_everything(seed); template = build_model(runtime["model"], clean["train"].shape[1], 2)
    initial = copy.deepcopy(template.state_dict())
    fairness = {"manifest_sha256": _sha256(base["fixed_views"]["manifest"]),
                "initialization_sha256": _state_hash(initial),
                "pretrain_order_sha256": sha256_strings([','.join(map(str, o)) for o in pretrain_orders]),
                "probe_order_sha256": sha256_strings([','.join(map(str, o)) for o in probe_orders])}
    return {"base": base, "views": views, "clean": clean, "stages": stages, "runtime": runtime,
            "pretrain_orders": pretrain_orders, "probe_orders": probe_orders,
            "initial": initial, "fairness": fairness}


def _evaluate_tep(model: torch.nn.Module, context: dict[str, Any], threshold: float, device: str) -> dict[str, Any]:
    views, clean, stages, runtime = context["views"], context["clean"], context["stages"], context["runtime"]
    probability, embedding = _probabilities(model, clean["test"], int(runtime["batch_size"]), device)
    scores = probability[:, 1]; prediction = scores >= threshold
    return {"metrics": _metrics(views["test"]["labels"], scores, threshold),
            "score_profile": score_profile(views["test"]["labels"], scores, threshold, .05),
            "stages": _stage_metrics(stages["test"], prediction),
            "early_fault": early_fault_recall(prediction, stages["test"]),
            "detection_delay": detection_delays(views["test"], prediction, runtime),
            "representation": representation_diagnostics(embedding, embedding, views["test"]["labels"])}


def _train_tep_method(method: str, seed: int, context: dict[str, Any], stage: dict[str, Any],
                      config: dict[str, Any], device: str) -> dict[str, Any]:
    output = Path(stage["output_dir"]) / f"seed_{seed}" / method
    metrics_path = output / "metrics.json"; checkpoint = output / "model.pt"
    if metrics_path.exists() and checkpoint.exists():
        return _read(metrics_path)
    if metrics_path.exists() or checkpoint.exists():
        raise RuntimeError(f"incomplete output: {output}")
    started = time.perf_counter(); seed_everything(seed)
    model = build_model(context["runtime"]["model"], context["clean"]["train"].shape[1], 2).to(device)
    model.load_state_dict(context["initial"])
    if method in TRADITIONAL:
        views = {}; audits = {}; settings = config["traditional"]
        for split, offset in (("train", 0), ("validation", 100), ("test", 200)):
            ids = context["views"][split]["window_id"]
            views[split] = traditional_view(context["clean"][split], ids, method, seed + offset,
                                            settings["jitter_std"], settings["scaling_std"])
            audits[split] = _effect_audit(context["clean"][split], views[split])
        pretrain = _fit_supcon(model,
            {"clean": context["clean"]["train"], "restored": views["train"], "labels": context["views"]["train"]["labels"]},
            {"clean": context["clean"]["validation"], "restored": views["validation"], "labels": context["views"]["validation"]["labels"]},
            np.ones(len(views["train"]), np.float32), np.ones(len(views["validation"]), np.float32),
            context["pretrain_orders"], context["runtime"], device)
    elif method == FRERA:
        pretrain, augmenter = _fit_frera(model, context["clean"]["train"], context["views"]["train"]["labels"],
                                         context["clean"]["validation"], context["views"]["validation"]["labels"],
                                         context["pretrain_orders"], context["runtime"], device, config["frera"])
        audits = {"official_commit": FreRAAdapter.OFFICIAL_COMMIT, "setting": "shared_backbone_adaptation"}
        augmenter.eval()
        with torch.no_grad():
            for split in ("train", "validation", "test"):
                sample = context["clean"][split][:1024]
                changed = augmenter(torch.from_numpy(sample).to(device)).cpu().numpy()
                audits[split] = _effect_audit(sample, changed)
    else:
        raise ValueError(method)
    seed_everything(seed + 1)
    probe = _fit_probe(model,
        {"clean": context["clean"]["train"], "labels": context["views"]["train"]["labels"]},
        {"restored": context["clean"]["validation"], "labels": context["views"]["validation"]["labels"]},
        context["probe_orders"], context["runtime"], device)
    best = best_probe_record(probe); threshold = float(best["validation_threshold"])
    record = {"method": method, "seed": seed, "validation_threshold": threshold,
              "best_pretrain_epoch": int(min(pretrain, key=lambda row: row["validation_supcon_loss"])["epoch"]),
              "best_probe_epoch": int(best["epoch"]), "pretrain_history": pretrain, "probe_history": probe,
              "test": _evaluate_tep(model, context, threshold, device), "augmentation_audit": audits,
              "initialization_sha256": _state_hash(context["initial"]),
              "training_seconds": time.perf_counter() - started,
              "peak_gpu_mib": torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0}
    output.mkdir(parents=True, exist_ok=False); torch.save(model.state_dict(), checkpoint); write_json(metrics_path, record)
    return record


def _reuse_final(stage: dict[str, Any], dataset: str, seed: int, method: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    source_method = "UNIFORM" if method == "UNIFORM_DIFFUSION" else "FINAL_QDIFFCL"
    path = Path(stage["final_manifest"]); item = _read(path)["results"][f"{source_method}|{seed}"]
    return item["metrics"], item["fairness"], str(path)


def _reuse_three_w_no_aug(stage: dict[str, Any], seed: int) -> tuple[dict[str, Any], dict[str, Any], str] | None:
    if seed not in (42, 43, 44):
        return None
    manifest = _read(stage["existing_3seed_manifest"]); path = manifest["seed_results"][str(seed)]["result_path"]
    result = _read(path)
    return result["methods"][THREE_W_METHODS[0]], result["fairness"], str(path)


def _best_traditional(manifest: dict[str, Any], dataset: str, seeds: list[int]) -> str:
    means = {}
    for method in ("JITTER", "SCALING", "JITTER_SCALING"):
        values = []
        for seed in seeds:
            item = manifest["results"].get(f"{dataset}|{method}|{seed}")
            if item:
                record = item["record"]; metrics = record["metrics"] if dataset == "3W" else record["test"]["metrics"]
                values.append(float(metrics["macro_f1"]))
        if len(values) != len(seeds):
            raise RuntimeError(f"Stage B incomplete for {dataset} {method}")
        means[method] = float(np.mean(values))
    return max(means, key=means.get)


def run(config: dict[str, Any], data_root: Path, dataset: str, stage_name: str) -> dict[str, Any]:
    validate_config(config); manifest_path, manifest = _manifest(config)
    # Must be set before the first CUDA context is created when a single process
    # runs 3W and then enables TEP deterministic algorithms.
    tep_base = yaml.safe_load(Path(config["tep"]["base_config"]).read_text(encoding="utf-8"))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", str(tep_base.get("cublas_workspace_config", ":4096:8")))
    device = select_device(str(config["device"])); selected = []
    datasets = ("3W", "TEP") if dataset == "both" else (dataset.upper(),)
    for current in datasets:
        stage = config["three_w" if current == "3W" else "tep"]
        if stage_name == "A": seeds = [int(stage["stage_a_seed"])]
        elif stage_name == "B": seeds = list(map(int, stage["stage_b_seeds"]))
        else: seeds = list(map(int, stage["stage_c_seeds"]))
        methods = [*TRADITIONAL, *REUSED, FRERA]
        if stage_name == "C":
            methods = [*REUSED, _best_traditional(manifest, current, list(map(int, stage["stage_b_seeds"]))), FRERA]
        for seed in seeds:
            context = None
            for method in methods:
                key = f"{current}|{method}|{seed}"
                if key in manifest["results"]:
                    continue
                try:
                    if method in REUSED:
                        record, fairness, source = _reuse_final(stage, current, seed, method)
                        _store(manifest_path, manifest, current, method, seed, record, fairness, source, "reused_existing")
                    elif current == "3W" and method == "NO_AUG" and (reused := _reuse_three_w_no_aug(stage, seed)):
                        record, fairness, source = reused
                        _store(manifest_path, manifest, current, method, seed, record, fairness, source, "reused_existing")
                    else:
                        if context is None:
                            context = (_prepare_three_w(stage, data_root, seed, device) if current == "3W"
                                       else _prepare_tep(stage, seed, device))
                        if current == "3W":
                            record = _train_three_w_method(method, seed, context, stage, config, device)
                            fairness = _three_w_fairness(context)
                        else:
                            record = _train_tep_method(method, seed, context, stage, config, device)
                            fairness = context["fairness"]
                        _store(manifest_path, manifest, current, method, seed, record, fairness,
                               str(Path(stage["output_dir"]) / f"seed_{seed}" / method / "metrics.json"), "new_training")
                    manifest["failures"].pop(key, None); write_json(manifest_path, manifest)
                    selected.append(key)
                except Exception as error:
                    manifest["failures"][key] = {"type": type(error).__name__, "message": str(error)}
                    write_json(manifest_path, manifest)
                    raise
    manifest["last_completed_stage"] = stage_name; write_json(manifest_path, manifest)
    return {"completed": selected, "manifest": str(manifest_path), "stage": stage_name}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/external_baselines.yaml")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("3w", "tep", "both"), default="both")
    parser.add_argument("--stage", choices=("A", "B", "C"), default="A")
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    print(json.dumps(run(config, args.data_root, args.dataset, args.stage), ensure_ascii=False))


if __name__ == "__main__":
    main()
