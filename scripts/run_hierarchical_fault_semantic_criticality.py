from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score, recall_score

from diffusion import DiffusionSchedule, FrequencyForwardDiffusion, fit_spectral_statistics
from frequency import (build_criticality, build_hierarchical_criticality, fault_stages,
                       fit_frequency_scaler, log_amplitude_phase)
from frequency.criticality import fault_type
from metrics import classification_metrics, select_binary_threshold
from scripts.audit_semantic_diffusion_augmentation import bases
from scripts.run_3w_diffusion_1seed import HFSC_METHOD, run as run_three_w
from scripts.run_diffusion_quality_retest import _batches, _fit_supcon, _probabilities, epoch_orders, load_fixed_views
from scripts.run_frequency_selective_r1_3seed import array_sha256, file_sha256, sha256_strings
from scripts.run_stage_frequency_diffusion_mvp import _configure, detection_delays, early_fault_recall
from trainers import build_model
from trainers.balanced import sqrt_inverse_frequency_weights
from utils import seed_everything, write_json


TEP_METHODS = ("UNIFORM", "R1", "R2", "HFSC")


def _settings(stage: dict) -> dict:
    value = copy.deepcopy(stage["shared_criticality"])
    value["diagnostic_classes"] = list(map(int, stage["diagnostic_classes"]))
    return value


def run_three_w_stage(config: dict, data_root: Path, selected_seeds: list[int] | None = None) -> dict:
    stage = config["three_w"]
    if list(map(int, stage["seeds"])) != [42, 43, 44]: raise ValueError("HFSC 3W seeds must be 42/43/44")
    base = yaml.safe_load(Path(stage["base_config"]).read_text(encoding="utf-8"))
    output = Path(stage["output_dir"]); output.mkdir(parents=True, exist_ok=True); manifest_path = output / "result_manifest.json"
    completed = json.loads(manifest_path.read_text(encoding="utf-8")).get("seed_results", {}) if manifest_path.exists() else {}
    for seed in map(int, selected_seeds if selected_seeds is not None else stage["seeds"]):
        current = copy.deepcopy(base); current["seed"] = seed; current["protocol_seed"] = int(stage["protocol_seed"])
        current.pop("criticality_source", None); current["hierarchical_criticality"] = True
        current["methods"] = [HFSC_METHOD]; current["training"]["supcon_batching"] = "original"
        current["criticality"] = _settings(stage); current["output_dir"] = str(output / f"seed_{seed}")
        result = run_three_w(current, data_root); path = Path(current["output_dir"]) / "result.json"
        completed[str(seed)] = {"result_path": str(path), "status": "complete", "methods": list(result["methods"])}
        write_json(manifest_path, {"stage": "3W", "seed_results": completed})
    payload = {"stage": "3W", "method": "HFSC", "seeds": sorted(map(int, completed)),
               "new_training_count": len(completed), "reused_baseline_count": 9,
               "criticality_fit_scope": "train-only shared and fault-only run-level diagnostic",
               "seed_results": completed}
    write_json(manifest_path, payload); return payload


def multiclass_labels(bundle: dict[str, np.ndarray]) -> np.ndarray:
    binary = np.asarray(bundle["labels"], dtype=np.int64)
    result = np.asarray([0 if label == 0 else fault_type(str(uid))
                         for label, uid in zip(binary, bundle["run_uid"])], dtype=np.int64)
    if np.any((binary == 0) != (result == 0)): raise RuntimeError("TEP normal/fault multiclass mapping mismatch")
    return result


def _fit_multiclass_probe(model: torch.nn.Module, train_x: np.ndarray, train_y: np.ndarray,
                          validation_x: np.ndarray, validation_y: np.ndarray,
                          orders: list[np.ndarray], runtime: dict, device: str) -> list[dict]:
    for parameter in model.parameters(): parameter.requires_grad = False
    seed_everything(int(runtime["random_seed"]) + 30_000); model.classification_head.reset_parameters()
    for parameter in model.classification_head.parameters(): parameter.requires_grad = True
    weight = torch.from_numpy(sqrt_inverse_frequency_weights(train_y)).to(device)
    optimizer = torch.optim.Adam(model.classification_head.parameters(), lr=float(runtime["learning_rate"]))
    history = []; best_state = None; best_score = -1.0
    for epoch, order in enumerate(orders):
        model.train(); losses = []
        for indices in _batches(order, int(runtime["batch_size"])):
            x = torch.from_numpy(train_x[indices]).float().to(device); y = torch.from_numpy(train_y[indices]).long().to(device)
            optimizer.zero_grad(); loss = F.cross_entropy(model(x)["logits"], y, weight=weight)
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        probability, _ = _probabilities(model, validation_x, int(runtime["batch_size"]), device)
        score = float(f1_score(validation_y, probability.argmax(1), average="macro", zero_division=0))
        history.append({"epoch": epoch, "train_weighted_ce": float(np.mean(losses)), "validation_macro_f1": score})
        if score > best_score: best_score, best_state = score, copy.deepcopy(model.state_dict())
    if best_state is None: raise RuntimeError("multiclass probe produced no checkpoint")
    model.load_state_dict(best_state); return history


def _per_class(y: np.ndarray, prediction: np.ndarray) -> list[dict]:
    return [{"class": kind, "recall": float(recall_score(y == kind, prediction == kind, zero_division=0)),
             "f1": float(f1_score(y == kind, prediction == kind, zero_division=0)),
             "support": int((y == kind).sum())} for kind in range(21)]


def _evaluate_multiclass(model: torch.nn.Module, clean: dict[str, np.ndarray], labels: dict[str, np.ndarray],
                         views: dict, stages: dict, runtime: dict, device: str) -> dict:
    validation_probability, _ = _probabilities(model, clean["validation"], int(runtime["batch_size"]), device)
    threshold = select_binary_threshold((labels["validation"] != 0).astype(int), 1 - validation_probability[:, 0])
    probability, _ = _probabilities(model, clean["test"], int(runtime["batch_size"]), device)
    prediction = probability.argmax(1); binary_y = (labels["test"] != 0).astype(np.int64)
    binary_score = 1 - probability[:, 0]; binary_prediction = (binary_score >= threshold).astype(np.int64)
    diagnosis = classification_metrics(labels["test"], prediction, probability)
    detection = classification_metrics(binary_y, binary_prediction, np.column_stack([1 - binary_score, binary_score]))
    detection["fault_recall"] = float(recall_score(binary_y, binary_prediction, pos_label=1, zero_division=0))
    return {"diagnosis": diagnosis, "per_class": _per_class(labels["test"], prediction),
            "detection": detection, "early_recall": early_fault_recall(binary_prediction, stages["test"])["recall"],
            "detection_delay": detection_delays(views["test"], binary_prediction, runtime)["mean_delay_samples"],
            "validation_binary_threshold": threshold}


def _build_tep_components(stage: dict, views: dict, clean: dict, stages: dict, device: str):
    train_log = log_amplitude_phase(clean["train"])[0]; scaler = fit_frequency_scaler(train_log, "train")
    standardized = scaler.transform(train_log); shared = build_criticality(
        standardized, views["train"], stages["train"], stage["shared_criticality"], train_log)
    r2 = build_criticality(standardized, views["train"], stages["train"], stage["r2_criticality"], train_log)
    hfsc = build_hierarchical_criticality(standardized, views["train"], stages["train"], _settings(stage), train_log)
    statistics = fit_spectral_statistics(clean["train"], float(stage["spectral_diffusion"]["clip_quantile"]), "train")
    schedule = DiffusionSchedule.cosine(int(stage["spectral_diffusion"]["diffusion_steps"]), device)
    def augmenter(mask):
        return FrequencyForwardDiffusion(statistics, schedule.alpha_bars, mask,
            int(stage["spectral_diffusion"]["t_uniform"]), int(stage["spectral_diffusion"]["t_critical"]),
            bool(stage["spectral_diffusion"]["preserve_phase"]), bool(stage["spectral_diffusion"]["preserve_dc"]), device)
    return shared, r2, hfsc, augmenter(shared["soft_mask"]), augmenter(r2["soft_mask"])


def run_tep_stage(config: dict, selected_seeds: list[int] | None = None) -> dict:
    stage = config["tep"]; base_config = yaml.safe_load(Path(stage["base_config"]).read_text(encoding="utf-8"))
    base_config["fixed_views"] = {"manifest": stage["fixed_views_manifest"]}; _configure(base_config)
    views, _ = load_fixed_views(base_config); clean = bases(views)
    stages = {split: fault_stages(views[split], base_config) for split in views}
    labels = {split: multiclass_labels(views[split]) for split in views}
    if any(set(np.unique(value)) != set(range(21)) for value in labels.values()):
        raise RuntimeError("TEP multiclass protocol must cover labels 0-20 in every split")
    shared, r2, hfsc, shared_aug, r2_aug = _build_tep_components(stage, views, clean, stages, str(config["device"]))
    output = Path(stage["output_dir"]); output.mkdir(parents=True, exist_ok=True); final_path = output / "result.json"
    completed = json.loads(final_path.read_text(encoding="utf-8")).get("seed_results", {}) if final_path.exists() else {}
    runtime_base = copy.deepcopy(stage["training"]); runtime_base["protocol"] = base_config["protocol"]
    runtime_base["detection"] = stage["detection"]
    for seed in map(int, selected_seeds if selected_seeds is not None else stage["seeds"]):
        runtime = copy.deepcopy(runtime_base); runtime["random_seed"] = seed
        existing_paths = {method: output / f"seed_{seed}" / method / "metrics.json" for method in TEP_METHODS}
        if str(seed) in completed and all(path.exists() for path in existing_paths.values()):
            continue
        augmented = {method: {} for method in TEP_METHODS}; audits = {method: {} for method in TEP_METHODS}
        for split, offset in (("train", 0), ("validation", 100), ("test", 200)):
            sampling_seed = seed + int(stage["spectral_diffusion"]["sampling_seed_offset"]) + offset
            batch = int(runtime["batch_size"]); noncritical = int(stage["spectral_diffusion"]["t_noncritical"])
            augmented["UNIFORM"][split], audits["UNIFORM"][split] = shared_aug.augment(clean[split], "uniform", sampling_seed, batch_size=batch)
            augmented["R1"][split], audits["R1"][split] = shared_aug.augment(clean[split], "selective", sampling_seed, noncritical, batch)
            augmented["R2"][split], audits["R2"][split] = r2_aug.augment(clean[split], "selective", sampling_seed, noncritical, batch)
            augmented["HFSC"][split], audits["HFSC"][split] = shared_aug.augment_hierarchical(
                clean[split], labels[split], hfsc["soft_masks"], sampling_seed, noncritical, batch)
            budgets = [audits[method][split]["expected_total_noise_budget"] for method in TEP_METHODS]
            if max(budgets) - min(budgets) > 1e-6: raise RuntimeError(f"TEP {split} noise budgets differ")
        pretrain_orders = epoch_orders(len(clean["train"]), int(runtime["epochs"]), seed + 10_000)
        probe_orders = epoch_orders(len(clean["train"]), int(runtime["probe_epochs"]), seed + 20_000)
        seed_everything(seed); template = build_model(runtime["model"], clean["train"].shape[1], 21)
        initial_state = copy.deepcopy(template.state_dict()); methods = {}
        fairness = {"fixed_views_sha256": file_sha256(stage["fixed_views_manifest"]),
                    "initialization_sha256": hashlib.sha256(b''.join(value.cpu().numpy().tobytes() for _, value in sorted(initial_state.items()))).hexdigest(),
                    "pretrain_order_sha256": sha256_strings([','.join(map(str, order)) for order in pretrain_orders]),
                    "probe_order_sha256": sha256_strings([','.join(map(str, order)) for order in probe_orders])}
        for method in TEP_METHODS:
            method_dir = output / f"seed_{seed}" / method; result_path = method_dir / "metrics.json"
            if result_path.exists(): methods[method] = json.loads(result_path.read_text(encoding="utf-8")); continue
            print("TEP", seed, "start", method, flush=True); seed_everything(seed)
            model = build_model(runtime["model"], clean["train"].shape[1], 21).to(str(config["device"])); model.load_state_dict(initial_state)
            pretrain = _fit_supcon(model,
                {"clean": clean["train"], "restored": augmented[method]["train"], "labels": labels["train"]},
                {"clean": clean["validation"], "restored": augmented[method]["validation"], "labels": labels["validation"]},
                np.ones(len(labels["train"]), np.float32), np.ones(len(labels["validation"]), np.float32),
                pretrain_orders, runtime, str(config["device"]))
            probe = _fit_multiclass_probe(model, clean["train"], labels["train"], clean["validation"],
                                          labels["validation"], probe_orders, runtime, str(config["device"]))
            metrics = _evaluate_multiclass(model, clean, labels, views, stages, runtime, str(config["device"]))
            method_dir.mkdir(parents=True, exist_ok=True); torch.save({"model": model.state_dict(), "fairness": fairness}, method_dir / "model.pt")
            methods[method] = {"metrics": metrics, "augmentation_audit": audits[method],
                               "pretrain_history": pretrain, "probe_history": probe, "fairness": fairness}
            write_json(result_path, methods[method]); print("TEP", seed, "done", method, metrics["diagnosis"]["macro_f1"], flush=True)
        completed[str(seed)] = {"seed": seed, "methods": methods, "fairness": fairness}
        write_json(final_path, {"seed_results": completed})
    payload = {"stage": "TEP_21_CLASS_DIAGNOSIS", "seeds": sorted(map(int, completed)),
               "new_training_count": len(completed) * len(TEP_METHODS), "methods": list(TEP_METHODS),
               "criticality_fit_scope": "train-only", "multiclass_label_mapping": "binary normal->0; faulty window->faultNumber(run_uid)",
               "criticality": {"shared_mask_sha256": array_sha256(shared["masks"]["composite"]),
                               "r2_mask_sha256": array_sha256(r2["masks"]["composite"]),
                               "hfsc_fault_run_counts": hfsc["fault_run_counts"],
                               "diagnostic": {str(k): {"score": v["score"].tolist(),
                                                       "hard_mask": v["hard_mask"].astype(int).tolist(),
                                                       "soft_mask": v["soft_mask"].tolist()}
                                              for k, v in hfsc["diagnostic"].items()},
                               "hierarchical": {str(k): {"score": v["score"].tolist(),
                                                         "hard_mask": v["hard_mask"].astype(int).tolist(),
                                                         "soft_mask": v["soft_mask"].tolist()}
                                                for k, v in hfsc["hierarchical"].items()}},
               "seed_results": completed, "test_used_for_selection": False, "paper_final_claim_allowed": False}
    write_json(final_path, payload); return payload


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/hierarchical_fault_semantic_criticality.yaml")
    parser.add_argument("--stage", choices=("3w", "tep"), required=True); parser.add_argument("--data-root", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+"); args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.stage == "3w":
        if args.data_root is None: raise ValueError("3W stage requires --data-root")
        result = run_three_w_stage(config, args.data_root, args.seeds)
    else: result = run_tep_stage(config, args.seeds)
    print(json.dumps({key: result[key] for key in result if key not in {"seed_results", "criticality"}}, ensure_ascii=False))


if __name__ == "__main__": main()
