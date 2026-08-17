from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

import scripts.run_3w_clean_baseline as base3w
from datasets.three_w import discover_instances
from diffusion import DiffusionSchedule, FrequencyForwardDiffusion, fit_spectral_statistics
from frequency import (build_early_warning_criticality, fault_stages,
                       fit_frequency_scaler, log_amplitude_phase, mask_jaccard,
                       onset_horizons)
from frequency.criticality import fault_type
from metrics.fixed_far import fixed_far_metrics
from scripts.audit_semantic_diffusion_augmentation import bases
from scripts.run_3w_diffusion_1seed import (EWIC_METHOD, METHODS as THREE_W_METHODS,
                                           run as run_three_w)
from scripts.run_3w_final_primary_grouped import FINAL_PRIMARY_CLASSES, build_model as build_three_w_model
from scripts.run_diffusion_quality_retest import _probabilities, epoch_orders, load_fixed_views
from scripts.run_frequency_selective_r1_3seed import (_fit_method as fit_tep_method,
                                                       array_sha256, file_sha256,
                                                       sha256_strings)
from scripts.run_stage_frequency_diffusion_mvp import (_configure, _runtime,
                                                        augmentation_mechanism_metrics,
                                                        detection_delays)
from trainers import build_model
from utils import seed_everything, write_json


TEP_METHODS = ("C1", "R1", "EWIC")


def _settings(stage: dict) -> dict:
    return copy.deepcopy(stage["criticality"])


def _criticality_ready(record: dict) -> dict:
    return {"fit_split": "train", "horizon_count": record["horizon_count"], "lead_decay": record["lead_decay"],
            "lead_weights": record["lead_weights"].tolist(), "horizon_coverage": record["horizon_coverage"],
            "r1_early": record["r1"]["early"].tolist(),
            "r1_composite": record["r1"]["composite"].tolist(),
            "r1_hard_mask": record["r1"]["masks"]["composite"].astype(int).tolist(),
            "r1_soft_mask": record["r1"]["soft_mask"].tolist(),
            "horizon_fisher": [value.tolist() for value in record["horizon_fisher"]],
            "horizon_normalized": [value.tolist() for value in record["horizon_normalized"]],
            "early_lead": record["early_lead"].tolist(), "early_reliability": record["early_reliability"].tolist(),
            "early_invariant": record["early_invariant"].tolist(), "composite": record["composite"].tolist(),
            "hard_mask": record["hard_mask"].astype(int).tolist(), "soft_mask": record["soft_mask"].tolist(),
            "bootstrap_overlap": record["bootstrap_overlap"].tolist(),
            "bootstrap_unit_count": record["bootstrap_unit_count"]}


def _horizon_profile(prediction: np.ndarray, horizons: np.ndarray, run_uids: np.ndarray) -> dict[str, Any]:
    rows = {}; first = {}
    for horizon in range(1, 9):
        selected = horizons == horizon; runs = np.unique(run_uids[selected])
        rate = float(np.mean([np.any(prediction[selected & (run_uids == uid)] == 1) for uid in runs])) if len(runs) else None
        rows[str(horizon)] = {"windows": int(selected.sum()),
                              "recall": float(prediction[selected].mean()) if selected.any() else None,
                              "run_detection_rate": rate, "run_count": int(len(runs))}
    for uid in np.unique(run_uids[horizons > 0]):
        selected = (run_uids == uid) & (horizons > 0) & (prediction == 1)
        first[str(uid)] = int(horizons[selected].min()) if selected.any() else None
    values = [value for value in first.values() if value is not None]
    return {"horizons": rows, "first_alarm_horizon_per_run": first,
            "mean_first_alarm_horizon": float(np.mean(values)) if values else None,
            "detected_within_h8_rate": float(len(values) / len(first)) if first else None}


def _tep_profiles(bundle: dict[str, np.ndarray], stages: np.ndarray, horizons: np.ndarray,
                  prediction: np.ndarray, runtime: dict) -> dict[str, Any]:
    run_uids = np.asarray(bundle["run_uid"]); per_fault = {}
    for kind in range(1, 21):
        selected = np.asarray([fault_type(str(uid)) == kind for uid in run_uids])
        early = selected & (stages == "early")
        subset = {key: np.asarray(value)[selected] for key, value in bundle.items()
                  if np.asarray(value).ndim > 0 and len(np.asarray(value)) == len(selected)}
        delay = detection_delays(subset, prediction[selected], runtime)
        per_fault[str(kind)] = {"early_windows": int(early.sum()),
                                "early_recall": float(prediction[early].mean()) if early.any() else None,
                                "detection_delay": delay["mean_delay_samples"],
                                "detection_rate": delay["detection_rate"]}
    return {"horizon_profile": _horizon_profile(prediction, horizons, run_uids),
            "per_fault": per_fault, "detection_delay": detection_delays(bundle, prediction, runtime),
            "early_recall": float(prediction[stages == "early"].mean())}


def _fixed_far_ready(value: dict[str, Any], profile_builder) -> dict[str, Any]:
    result = {}
    for name, item in value.items():
        prediction = item.pop("prediction")
        result[name] = {**item, **profile_builder(prediction)}
    return result


def run_three_w_stage(config: dict, data_root: Path, selected_seeds: list[int] | None = None) -> dict:
    stage = config["three_w"]
    if list(map(int, stage["seeds"])) != [42, 43, 44]: raise ValueError("EWIC 3W seeds must be 42/43/44")
    base = yaml.safe_load(Path(stage["base_config"]).read_text(encoding="utf-8")); output = Path(stage["output_dir"])
    output.mkdir(parents=True, exist_ok=True); manifest_path = output / "result_manifest.json"
    completed = json.loads(manifest_path.read_text(encoding="utf-8")).get("seed_results", {}) if manifest_path.exists() else {}
    for seed in map(int, selected_seeds if selected_seeds is not None else stage["seeds"]):
        current = copy.deepcopy(base); current["seed"] = seed; current["protocol_seed"] = int(stage["protocol_seed"])
        current.pop("criticality_source", None); current["early_warning_criticality"] = True
        current["methods"] = [EWIC_METHOD]; current["training"]["supcon_batching"] = "original"
        current["criticality"] = _settings(stage); current["output_dir"] = str(output / f"seed_{seed}")
        result = run_three_w(current, data_root); path = Path(current["output_dir"]) / "result.json"
        completed[str(seed)] = {"result_path": str(path), "status": "complete", "methods": list(result["methods"])}
        write_json(manifest_path, {"seed_results": completed})
    payload = {"stage": "3W_EARLY_DETECTION", "method": "EWIC", "seeds": sorted(map(int, completed)),
               "new_training_count": len(completed), "reused_baseline_count": 6, "seed_results": completed}
    write_json(manifest_path, payload); return payload


def _three_w_context(stage: dict, data_root: Path):
    config = yaml.safe_load(Path(stage["base_config"]).read_text(encoding="utf-8"))
    base_config = yaml.safe_load(Path(config["base_config"]).read_text(encoding="utf-8"))
    base = yaml.safe_load(Path(base_config["base_config"]).read_text(encoding="utf-8"))
    grouped = Path(config["grouped_output"]); split_index = int(config["canonical_split_index"])
    manifest = json.loads((grouped / "grouped_split_manifest.json").read_text(encoding="utf-8"))
    split = {name: set(wells) for name, wells in manifest["splits"][split_index]["wells"].items()}
    base3w.PRIMARY_CLASSES = FINAL_PRIMARY_CLASSES
    base3w.CLASS_TO_TARGET = {value: index for index, value in enumerate(FINAL_PRIMARY_CLASSES)}
    instances = [item for item in discover_instances(data_root)
                 if item.source == "WELL" and item.event_class in FINAL_PRIMARY_CLASSES]
    by_instance = {item.instance_id: item for item in instances}
    by_split = {name: [item for item in instances if item.well_id in wells] for name, wells in split.items()}
    preprocessor = json.loads((grouped / f"split_{split_index:02d}" / "preprocessor.json").read_text(encoding="utf-8"))
    refs_by_instance = {}
    for item in instances:
        refs_by_instance[item.instance_id] = base3w.instance_refs(item, int(base["protocol"]["window_length"]),
            int(base["protocol"]["stride"]), int(base["protocol"]["transient_offset"]))
    return config, base, by_instance, by_split, preprocessor, refs_by_instance


def _three_w_inference(model, instances, refs_by_instance, by_instance, preprocessor, base, device):
    probabilities = []; refs = []
    for instance in instances:
        current = refs_by_instance[instance.instance_id]
        if not current: continue
        x, y = base3w.materialize(current, by_instance, preprocessor, int(base["protocol"]["window_length"]), False)
        probabilities.append(base3w.probabilities(model, x, y, int(base["training"]["batch_size"]), device)); refs.extend(current)
    return np.concatenate(probabilities), refs


def _three_w_profiles(refs, prediction: np.ndarray, by_instance: dict, stride: int, length: int) -> dict[str, Any]:
    horizons = []; run_uids = []; original = []; wells = []
    for ref in refs:
        item = by_instance[ref.instance_id]; run_uids.append(ref.instance_id); wells.append(item.well_id)
        original.append(FINAL_PRIMARY_CLASSES[ref.target])
        if ref.target == 0 or ref.onset_seconds is None: horizons.append(0)
        else:
            progress = ref.end_seconds - float(ref.onset_seconds) - (length - 1)
            value = int(np.floor(progress / stride)) + 1
            horizons.append(value if 1 <= value <= 8 else 0)
    horizons = np.asarray(horizons); run_uids = np.asarray(run_uids); original = np.asarray(original); wells = np.asarray(wells)
    early = np.asarray([ref.stage == "early" for ref in refs]); per_instance = {}; delays = []
    for uid in np.unique(run_uids):
        selected = run_uids == uid; current_refs = [ref for ref, keep in zip(refs, selected) if keep]
        current_prediction = prediction[selected]; onset = current_refs[0].onset_seconds
        detections = [ref.end_seconds - float(onset) for ref, pred in zip(current_refs, current_prediction)
                      if onset is not None and ref.target != 0 and ref.end_seconds >= onset and pred == 1]
        delay = float(detections[0]) if detections else None
        if onset is not None and any(ref.target != 0 for ref in current_refs) and delay is not None: delays.append(delay)
        per_instance[str(uid)] = {"well_id": str(wells[selected][0]), "class": int(original[selected].max()),
                                  "early_recall": float(prediction[selected & early].mean()) if np.any(selected & early) else None,
                                  "delay_seconds": delay}
    per_class = {}
    for kind in (2, 8, 9):
        selected = original == kind; instance_rows = [row for row in per_instance.values() if row["class"] == kind]
        class_delays = [row["delay_seconds"] for row in instance_rows if row["delay_seconds"] is not None]
        per_class[str(kind)] = {"early_recall": float(prediction[selected & early].mean()) if np.any(selected & early) else None,
                                "mean_delay_seconds": float(np.mean(class_delays)) if class_delays else None,
                                "detected_instance_rate": float(len(class_delays) / len(instance_rows)) if instance_rows else None}
    per_well = {}
    for well in np.unique(wells):
        selected = wells == well; instance_rows = [row for row in per_instance.values() if row["well_id"] == well]
        well_delays = [row["delay_seconds"] for row in instance_rows if row["delay_seconds"] is not None]
        per_well[str(well)] = {"early_recall": float(prediction[selected & early].mean()) if np.any(selected & early) else None,
                               "mean_delay_seconds": float(np.mean(well_delays)) if well_delays else None,
                               "instances": len(instance_rows)}
    return {"early_recall": float(prediction[early].mean()) if early.any() else None,
            "mean_detection_delay_seconds": float(np.mean(delays)) if delays else None,
            "horizon_profile": _horizon_profile(prediction, horizons, run_uids),
            "per_class": per_class, "per_well": per_well, "per_instance": per_instance}


def evaluate_three_w(config: dict, data_root: Path) -> dict:
    stage = config["three_w"]; base_config, base, by_instance, by_split, preprocessor, refs_by_instance = _three_w_context(stage, data_root)
    old_manifest = json.loads(Path(stage["existing_manifest"]).read_text(encoding="utf-8"))
    new_manifest = json.loads((Path(stage["output_dir"]) / "result_manifest.json").read_text(encoding="utf-8"))
    evaluations = {}
    for seed in map(int, stage["seeds"]):
        old_result_path = Path(old_manifest["seed_results"][str(seed)]["result_path"]); old_dir = old_result_path.parent
        new_result = json.loads(Path(new_manifest["seed_results"][str(seed)]["result_path"]).read_text(encoding="utf-8"))
        paths = {"UNIFORM": old_dir / f"{THREE_W_METHODS[1]}_model.pt",
                 "R1": old_dir / f"{THREE_W_METHODS[2]}_model.pt",
                 "EWIC": Path(stage["output_dir"]) / f"seed_{seed}" / f"{EWIC_METHOD}_model.pt"}
        methods = {}
        for method, path in paths.items():
            device = str(config["device"]); model = build_three_w_model(base["training"]["model"], len(preprocessor["retained_features"]), device)
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            validation_probability, validation_refs = _three_w_inference(model, by_split["validation"], refs_by_instance,
                                                                          by_instance, preprocessor, base, device)
            test_probability, test_refs = _three_w_inference(model, by_split["test"], refs_by_instance,
                                                              by_instance, preprocessor, base, device)
            validation_y = np.asarray([ref.target != 0 for ref in validation_refs], np.int64)
            test_y = np.asarray([ref.target != 0 for ref in test_refs], np.int64)
            validation_score = 1 - validation_probability[:, 0]; test_score = 1 - test_probability[:, 0]
            prediction = (test_probability.argmax(1) != 0).astype(np.int64)
            source_result = (new_result["methods"][EWIC_METHOD] if method == "EWIC" else
                             json.loads(old_result_path.read_text(encoding="utf-8"))["methods"][THREE_W_METHODS[1 if method == "UNIFORM" else 2]])
            profile_builder = lambda p: _three_w_profiles(test_refs, p, by_instance,
                int(base["protocol"]["stride"]), int(base["protocol"]["window_length"]))
            fixed = fixed_far_metrics(validation_y, validation_score, test_y, test_score)
            methods[method] = {"standard": {"metrics": source_result["metrics"], **profile_builder(prediction)},
                               "fixed_far": _fixed_far_ready(fixed, profile_builder)}
        evaluations[str(seed)] = methods
    first = json.loads(Path(new_manifest["seed_results"][str(stage["seeds"][0])]["result_path"]).read_text(encoding="utf-8"))
    payload = {"seeds": stage["seeds"], "methods": ["UNIFORM", "R1", "EWIC"], "evaluations": evaluations,
               "criticality": first["criticality"]}
    write_json(Path(stage["output_dir"]) / "evaluation.json", payload); return payload


def _build_tep(config: dict):
    stage = config["tep"]; base_config = yaml.safe_load(Path(stage["base_config"]).read_text(encoding="utf-8"))
    _configure(base_config); views, _ = load_fixed_views(base_config); clean = bases(views)
    stages = {split: fault_stages(views[split], base_config) for split in views}
    horizons = {split: onset_horizons(views[split], base_config["protocol"]["fault_onset"],
                                      int(base_config["protocol"]["stride"])) for split in views}
    train_log = log_amplitude_phase(clean["train"])[0]; scaler = fit_frequency_scaler(train_log, "train")
    criticality = build_early_warning_criticality(scaler.transform(train_log), views["train"], stages["train"],
        horizons["train"], _settings(stage), train_log)
    statistics = fit_spectral_statistics(clean["train"], float(stage["spectral_diffusion"]["clip_quantile"]), "train")
    schedule = DiffusionSchedule.cosine(int(stage["spectral_diffusion"]["diffusion_steps"]), str(config["device"]))
    augmenter = FrequencyForwardDiffusion(statistics, schedule.alpha_bars, criticality["soft_mask"],
        int(stage["spectral_diffusion"]["t_uniform"]), int(stage["spectral_diffusion"]["t_critical"]),
        bool(stage["spectral_diffusion"]["preserve_phase"]), bool(stage["spectral_diffusion"]["preserve_dc"]), str(config["device"]))
    return base_config, views, clean, stages, horizons, criticality, augmenter


def run_tep_stage(config: dict, selected_seeds: list[int] | None = None) -> dict:
    stage = config["tep"]
    if list(map(int, stage["seeds"])) != [7, 42, 2026]: raise ValueError("EWIC TEP seeds must be 7/42/2026")
    base_config, views, clean, stages, horizons, criticality, augmenter = _build_tep(config)
    baseline = json.loads(Path(stage["existing_result"]).read_text(encoding="utf-8"))
    output = Path(stage["output_dir"]); output.mkdir(parents=True, exist_ok=True); final_path = output / "result.json"
    completed = json.loads(final_path.read_text(encoding="utf-8")).get("seed_results", {}) if final_path.exists() else {}
    manifest_hash = file_sha256(base_config["fixed_views"]["manifest"])
    for seed in map(int, selected_seeds if selected_seeds is not None else stage["seeds"]):
        metrics_path = output / f"seed_{seed}" / "EWIC" / "metrics.json"
        if str(seed) in completed and metrics_path.exists(): continue
        runtime = _runtime(base_config, seed); runtime["diagnosis"] = {"threshold_band_width": .05, "high_correlation_quantile": .90}
        augmented = {}; audits = {}
        for split, offset in (("train", 0), ("validation", 100), ("test", 200)):
            sampling_seed = seed + int(stage["spectral_diffusion"]["sampling_seed_offset"]) + offset
            augmented[split], diag = augmenter.augment(clean[split], "selective", sampling_seed,
                int(stage["spectral_diffusion"]["t_noncritical"]), int(base_config["training"]["batch_size"]), noise_structure="iid")
            audits[split] = augmentation_mechanism_metrics(clean[split], augmented[split], views[split]["labels"],
                stages[split], criticality["hard_mask"], diag)
            old_budget = baseline["seed_results"][str(seed)]["methods"]["R1"]["augmentation_audit"][split]["expected_total_noise_budget"]
            if abs(audits[split]["expected_total_noise_budget"] - old_budget) > 1e-6:
                raise RuntimeError(f"TEP EWIC/R1 {split} noise budgets differ")
        pretrain_orders = epoch_orders(len(clean["train"]), int(runtime["epochs"]), seed + 10_000)
        probe_orders = epoch_orders(len(clean["train"]), int(runtime["probe_epochs"]), seed + 20_000)
        seed_everything(seed); template = build_model(runtime["model"], clean["train"].shape[1], 2)
        initial_state = copy.deepcopy(template.state_dict())
        fairness = {"manifest_sha256": manifest_hash, "initialization_sha256": hashlib.sha256(
                    b''.join(name.encode() + value.cpu().numpy().tobytes() for name, value in sorted(initial_state.items()))).hexdigest(),
                    "pretrain_order_sha256": sha256_strings([','.join(map(str, order)) for order in pretrain_orders]),
                    "probe_order_sha256": sha256_strings([','.join(map(str, order)) for order in probe_orders])}
        old = baseline["seed_results"][str(seed)]["fairness"]
        for key in fairness:
            if fairness[key] != old[key]: raise RuntimeError(f"TEP EWIC fairness differs for seed {seed}: {key}")
        metadata = {**old, "method": "EWIC", "seed": seed, "augmentation": "ewic_selective_iid_t5"}
        print("TEP", seed, "start EWIC", flush=True)
        record = fit_tep_method("EWIC", augmented, audits, views, clean, stages, initial_state,
            pretrain_orders, probe_orders, runtime, str(config["device"]), metrics_path.parent / "model.pt", metadata)
        completed[str(seed)] = {"seed": seed, "method": record, "fairness": fairness}
        write_json(final_path, {"seed_results": completed}); print("TEP", seed, "done EWIC", flush=True)
    payload = {"stage": "TEP_BINARY_EARLY_DETECTION", "method": "EWIC", "seeds": sorted(map(int, completed)),
               "new_training_count": len(completed), "reused_baseline_count": 6,
               "criticality": _criticality_ready(criticality), "seed_results": completed,
               "test_used_for_threshold_or_fit": False}
    write_json(final_path, payload); return payload


def _load_tep_model(path: Path, runtime: dict, channels: int, device: str):
    model = build_model(runtime["model"], channels, 2).to(device)
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"]); return model


def evaluate_tep(config: dict) -> dict:
    stage = config["tep"]; base_config, views, clean, stages, horizons, criticality, _ = _build_tep(config)
    baseline = json.loads(Path(stage["existing_result"]).read_text(encoding="utf-8")); result = json.loads((Path(stage["output_dir"]) / "result.json").read_text(encoding="utf-8"))
    evaluations = {}
    for seed in map(int, stage["seeds"]):
        runtime = _runtime(base_config, seed); paths = {
            "C1": Path("outputs/frequency_selective_r1_3seed") / f"seed_{seed}" / "C1" / "model.pt",
            "R1": Path("outputs/frequency_selective_r1_3seed") / f"seed_{seed}" / "R1" / "model.pt",
            "EWIC": Path(stage["output_dir"]) / f"seed_{seed}" / "EWIC" / "model.pt"}
        methods = {}
        for method, path in paths.items():
            model = _load_tep_model(path, runtime, clean["train"].shape[1], str(config["device"]))
            validation_scores = _probabilities(model, clean["validation"], int(runtime["batch_size"]), str(config["device"]))[0][:, 1]
            test_scores = _probabilities(model, clean["test"], int(runtime["batch_size"]), str(config["device"]))[0][:, 1]
            source = (result["seed_results"][str(seed)]["method"] if method == "EWIC"
                      else baseline["seed_results"][str(seed)]["methods"][method])
            threshold = float(source["validation_threshold"]); prediction = (test_scores >= threshold).astype(np.int64)
            profile = _tep_profiles(views["test"], stages["test"], horizons["test"], prediction, runtime)
            fixed = fixed_far_metrics(views["validation"]["labels"], validation_scores, views["test"]["labels"], test_scores)
            methods[method] = {"standard": {"metrics": source["test"]["metrics"], **profile},
                               "fixed_far": _fixed_far_ready(fixed, lambda p: _tep_profiles(
                                   views["test"], stages["test"], horizons["test"], p, runtime))}
        evaluations[str(seed)] = methods
    payload = {"seeds": stage["seeds"], "methods": list(TEP_METHODS), "evaluations": evaluations,
               "criticality": _criticality_ready(criticality)}
    write_json(Path(stage["output_dir"]) / "evaluation.json", payload); return payload


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/early_warning_invariant_criticality.yaml")
    parser.add_argument("--stage", choices=("3w", "tep"), required=True); parser.add_argument("--data-root", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+"); parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.stage == "3w":
        if args.data_root is None: raise ValueError("3W stage requires --data-root")
        result = evaluate_three_w(config, args.data_root) if args.evaluate else run_three_w_stage(config, args.data_root, args.seeds)
    else:
        result = evaluate_tep(config) if args.evaluate else run_tep_stage(config, args.seeds)
    print(json.dumps({key: value for key, value in result.items() if key not in {"seed_results", "evaluations", "criticality"}}, ensure_ascii=False))


if __name__ == "__main__": main()
