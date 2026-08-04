from __future__ import annotations

import copy
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import recall_score
from torch.utils.data import DataLoader, TensorDataset

from degradations import apply_degradation
from losses import quality_weighted_supervised_contrastive_loss
from metrics import classification_metrics, representation_diagnostics
from scripts.common import prepare_real
from trainers import build_model
from utils import environment_metadata, seed_everything, write_json


MARKERS = ["RAPID_IDEA_VALIDATION", "SINGLE_SEED", "SUBSET_DATA", "NOT_FOR_PAPER_CLAIMS"]


def _loader(*arrays: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensors = [torch.from_numpy(array) for array in arrays]
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _evaluate(model: torch.nn.Module, x: np.ndarray, y: np.ndarray, batch_size: int, device: str) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    model.eval(); probabilities, predictions, embeddings = [], [], []
    with torch.no_grad():
        for (xb,) in _loader(x.astype(np.float32), batch_size=batch_size, shuffle=False):
            output = model(xb.to(device)); probability = torch.softmax(output["logits"], dim=1)
            probabilities.append(probability.cpu().numpy()); predictions.append(probability.argmax(1).cpu().numpy())
            embeddings.append(output["embedding"].cpu().numpy())
    probability = np.concatenate(probabilities); prediction = np.concatenate(predictions); embedding = np.concatenate(embeddings)
    metrics = classification_metrics(y, prediction, probability)
    metrics["fault_recall"] = float(recall_score(y, prediction, pos_label=1, zero_division=0))
    return metrics, prediction, probability, embedding


def _fit_ce(model: torch.nn.Module, train_x: np.ndarray, train_y: np.ndarray,
            val_x: np.ndarray, val_y: np.ndarray, config: dict[str, Any], device: str,
            trainable_only: bool = False) -> list[dict[str, float]]:
    parameters = [p for p in model.parameters() if p.requires_grad] if trainable_only else model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=float(config["learning_rate"]))
    best_state, best_score, stale = None, -1.0, 0; history = []
    for epoch in range(int(config["epochs"])):
        model.train(); losses = []
        for xb, yb in _loader(train_x.astype(np.float32), train_y.astype(np.int64), batch_size=int(config["batch_size"]), shuffle=True):
            xb, yb = xb.to(device), yb.to(device); optimizer.zero_grad()
            loss = F.cross_entropy(model(xb)["logits"], yb); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        validation, _, _, _ = _evaluate(model, val_x, val_y, int(config["batch_size"]), device)
        score = float(validation["macro_f1"]); history.append({"epoch": epoch, "loss": float(np.mean(losses)), "validation_macro_f1": score})
        if score > best_score + 1e-6:
            best_score, best_state, stale = score, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= int(config["early_stopping_patience"]): break
    if best_state is not None: model.load_state_dict(best_state)
    return history


def _contrastive_epoch(model: torch.nn.Module, clean: np.ndarray, view: np.ndarray,
                       labels: np.ndarray, quality: np.ndarray, config: dict[str, Any],
                       device: str, optimizer: torch.optim.Optimizer | None) -> float:
    training = optimizer is not None; model.train(training); losses = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for clean_b, view_b, labels_b, quality_b in _loader(
            clean.astype(np.float32), view.astype(np.float32), labels.astype(np.int64), quality.astype(np.float32),
            batch_size=int(config["batch_size"]), shuffle=training,
        ):
            clean_b, view_b, labels_b, quality_b = clean_b.to(device), view_b.to(device), labels_b.to(device), quality_b.to(device)
            if optimizer is not None: optimizer.zero_grad()
            clean_z = model(clean_b)["projection"]; view_z = model(view_b)["projection"]
            features = torch.cat([clean_z, view_z], dim=0); pair_labels = torch.cat([labels_b, labels_b], dim=0)
            weights = torch.cat([torch.ones_like(quality_b), quality_b], dim=0)
            loss = quality_weighted_supervised_contrastive_loss(features, pair_labels, weights, float(config["temperature"]))
            if optimizer is not None: loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
    return float(np.mean(losses))


def _fit_supcon(model: torch.nn.Module, train_clean: np.ndarray, train_view: np.ndarray,
                train_y: np.ndarray, train_q: np.ndarray, val_clean: np.ndarray,
                val_view: np.ndarray, val_y: np.ndarray, val_q: np.ndarray,
                config: dict[str, Any], device: str) -> list[dict[str, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"])); history = []
    best_state, best_loss, stale = None, float("inf"), 0
    for epoch in range(int(config["epochs"])):
        train_loss = _contrastive_epoch(model, train_clean, train_view, train_y, train_q, config, device, optimizer)
        val_loss = _contrastive_epoch(model, val_clean, val_view, val_y, val_q, config, device, None)
        history.append({"epoch": epoch, "loss": train_loss, "validation_supcon_loss": val_loss})
        if val_loss < best_loss - 1e-6:
            best_loss, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= int(config["early_stopping_patience"]): break
    if best_state is not None: model.load_state_dict(best_state)
    return history


def _simple_interpolate(data: np.ndarray, observation: np.ndarray) -> np.ndarray:
    result = data.copy(); time_index = np.arange(data.shape[-1])
    for sample in range(len(result)):
        for channel in range(result.shape[1]):
            observed = observation[sample, channel]
            if observed.any(): result[sample, channel, ~observed] = np.interp(time_index[~observed], time_index[observed], result[sample, channel, observed])
            else: result[sample, channel] = 0.0
    return result


def _masked_mae(clean: np.ndarray, restored: np.ndarray, observation: np.ndarray) -> tuple[float, np.ndarray]:
    missing = ~observation; errors = np.abs(clean - restored)
    per_sample = np.array([errors[index][missing[index]].mean() if missing[index].any() else 0.0 for index in range(len(clean))])
    return float(errors[missing].mean()), per_sample


def _quality(per_sample_mae: np.ndarray) -> np.ndarray:
    scale = float(np.mean(per_sample_mae)) + 1e-12
    return np.clip(np.exp(-per_sample_mae / scale), 0.0, 1.0).astype(np.float32)


def _view_bundle(clean: np.ndarray, ids: list[str], config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    degraded_items = [apply_degradation(x, "mcar_missing", 0.3, int(config["random_seed"]), uid, "normalized_space") for x, uid in zip(clean, ids)]
    degraded = np.stack([item.data for item in degraded_items]); observation = np.stack([item.observation_mask for item in degraded_items])
    simple = _simple_interpolate(degraded, observation)
    degraded_mae, degraded_per_sample = _masked_mae(clean, degraded, observation)
    simple_mae, simple_per_sample = _masked_mae(clean, simple, observation)
    return degraded, simple, observation, _quality(degraded_per_sample), degraded_mae, simple_mae


def _kept_ids(stats: dict[str, Any]) -> list[str]:
    return [f"{item['run_uid']}:samples_{item['start_sample']}_{item['end_sample']}" for item in stats["window_metadata"] if not item["excluded"]]


def main() -> None:
    config = yaml.safe_load(Path("configs/rapid_idea_validation.yaml").read_text(encoding="utf-8"))
    output = Path(config["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    seed_everything(int(config["random_seed"])); device = "cuda"; started = time.perf_counter()
    clean_data, manifest, window_stats = prepare_real(config, degrade=False)
    bundles: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation", "test"):
        clean, labels = clean_data[split]; ids = _kept_ids(window_stats[split])
        degraded, simple, observation, degraded_q, degraded_mae, simple_mae = _view_bundle(clean, ids, config)
        _, simple_per_sample = _masked_mae(clean, simple, observation)
        bundles[split] = {"clean": clean, "degraded": degraded, "simple": simple, "observation": observation,
                          "labels": labels, "ids": ids, "degraded_q": degraded_q, "simple_q": _quality(simple_per_sample),
                          "degraded_mae": degraded_mae, "simple_mae": simple_mae}
    write_json(output / "split_manifest.json", manifest.__dict__); write_json(output / "config.json", config)
    results: dict[str, Any] = {}; models: dict[str, torch.nn.Module] = {}
    groups = [
        ("G1-0 Clean CE", "ce", "clean", False),
        ("G1-1 Degraded CE", "ce", "degraded", False),
        ("G1-2 Hard SupCon", "supcon", "degraded", False),
        ("G1-3 Oracle Quality SupCon", "supcon", "degraded", True),
        ("G1-4 Simple Recovery + Quality", "supcon", "simple", True),
    ]
    for name, method, view_name, weighted in groups:
        seed_everything(int(config["random_seed"])); torch.cuda.reset_peak_memory_stats(); group_started = time.perf_counter()
        model = build_model(config["model"], bundles["train"]["clean"].shape[1], 2).to(device)
        if method == "ce":
            history = _fit_ce(model, bundles["train"][view_name], bundles["train"]["labels"], bundles["validation"][view_name], bundles["validation"]["labels"], config, device)
        else:
            quality_key = f"{view_name}_q"
            train_q = bundles["train"][quality_key] if weighted else np.ones(len(bundles["train"]["labels"]), np.float32)
            val_q = bundles["validation"][quality_key] if weighted else np.ones(len(bundles["validation"]["labels"]), np.float32)
            pretrain = _fit_supcon(model, bundles["train"]["clean"], bundles["train"][view_name], bundles["train"]["labels"], train_q,
                                   bundles["validation"]["clean"], bundles["validation"][view_name], bundles["validation"]["labels"], val_q, config, device)
            for parameter in model.encoder.parameters(): parameter.requires_grad = False
            history = {"pretrain": pretrain, "linear_probe": _fit_ce(model, bundles["train"]["clean"], bundles["train"]["labels"], bundles["validation"][view_name], bundles["validation"]["labels"], config, device, True)}
        metrics, prediction, probability, embedding = _evaluate(model, bundles["test"][view_name], bundles["test"]["labels"], int(config["batch_size"]), device)
        _, clean_prediction, _, clean_embedding = _evaluate(model, bundles["test"]["clean"], bundles["test"]["labels"], int(config["batch_size"]), device)
        diagnostics = representation_diagnostics(clean_embedding, embedding, bundles["test"]["labels"])
        elapsed = time.perf_counter() - group_started
        results[name] = {"view": view_name, "weighted": weighted, "metrics": metrics,
                         "semantic_consistency": float(np.mean(clean_prediction == prediction)),
                         "class_center_shift": diagnostics["class_center_shift"], "effective_rank": diagnostics["effective_rank"],
                         "masked_mae": bundles["test"]["simple_mae"] if view_name == "simple" else None,
                         "training_seconds": elapsed, "peak_gpu_mib": torch.cuda.max_memory_allocated() / 1024 ** 2,
                         "history": history}
        models[name] = model; torch.save(model.state_dict(), output / f"{name.split()[0].replace('-', '_')}.pt")
    teacher = models["G1-0 Clean CE"]
    _, teacher_clean_prediction, _, _ = _evaluate(teacher, bundles["test"]["clean"], bundles["test"]["labels"], int(config["batch_size"]), device)
    for name, result in results.items():
        _, teacher_view_prediction, _, _ = _evaluate(teacher, bundles["test"][result["view"]], bundles["test"]["labels"], int(config["batch_size"]), device)
        result["teacher_prediction_consistency"] = float(np.mean(teacher_clean_prediction == teacher_view_prediction))
    hard = results["G1-2 Hard SupCon"]["metrics"]; oracle = results["G1-3 Oracle Quality SupCon"]["metrics"]; simple = results["G1-4 Simple Recovery + Quality"]["metrics"]
    oracle_gain = max(float(oracle["macro_f1"]) - float(hard["macro_f1"]), float(oracle["auprc"]) - float(hard["auprc"]))
    simple_gain = max(float(simple["macro_f1"]) - float(hard["macro_f1"]), float(simple["auprc"]) - float(hard["auprc"]))
    gate_one = "GO" if oracle_gain >= 0.015 or simple_gain >= 0.015 else "NO-GO"
    summary = {"markers": MARKERS, **environment_metadata(), "total_seconds": time.perf_counter() - started,
               "split_counts": {key: len(value) for key, value in manifest.__dict__.items()},
               "window_counts": {key: len(bundles[key]["labels"]) for key in bundles},
               "class_counts": {key: np.bincount(bundles[key]["labels"], minlength=2).tolist() for key in bundles},
               "degraded_masked_mae": bundles["test"]["degraded_mae"], "simple_masked_mae": bundles["test"]["simple_mae"],
               "results": results, "gate_one": gate_one, "oracle_gain_signal": oracle_gain, "simple_gain_signal": simple_gain,
               "gate_two": "NOT_RUN", "gate_three": "NOT_RUN"}
    write_json(output / "gate1_results.json", summary)
    with (output / "gate1_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "macro_f1", "auprc", "fault_recall", "far", "masked_mae", "teacher_consistency", "effective_rank", "training_seconds", "peak_gpu_mib"]); writer.writeheader()
        for name, result in results.items():
            writer.writerow({"method": name, "macro_f1": result["metrics"]["macro_f1"], "auprc": result["metrics"]["auprc"], "fault_recall": result["metrics"]["fault_recall"], "far": result["metrics"]["far"], "masked_mae": result["masked_mae"], "teacher_consistency": result["teacher_prediction_consistency"], "effective_rank": result["effective_rank"], "training_seconds": result["training_seconds"], "peak_gpu_mib": result["peak_gpu_mib"]})
    print(json.dumps({"gate_one": gate_one, "oracle_gain_signal": oracle_gain, "simple_gain_signal": simple_gain, "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
