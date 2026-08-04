from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import recall_score

from diffusion.fixed_views import (SPLITS, distribution, fit_quality_scale,
                                   per_sample_masked_mae, quality_scores,
                                   sha256_file, sha256_strings,
                                   validate_view_splits)
from losses import quality_weighted_supervised_contrastive_loss
from metrics import classification_metrics, representation_diagnostics, select_binary_threshold
from trainers import build_model
from utils import environment_metadata, seed_everything, write_json


def load_fixed_views(config: dict[str, Any]) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    manifest_path = Path(config["fixed_views"]["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    views: dict[str, dict[str, np.ndarray]] = {}
    for split in SPLITS:
        record = manifest["splits"][split]
        path = Path(record["path"])
        if sha256_file(path) != record["sha256"]: raise RuntimeError(f"fixed view hash mismatch: {split}")
        with np.load(path, allow_pickle=False) as archive:
            views[split] = {key: archive[key] for key in archive.files}
        if sha256_strings(list(map(str, views[split]["window_id"]))) != record["window_ids_sha256"]:
            raise RuntimeError(f"fixed window order mismatch: {split}")
        if sha256_strings(list(map(str, views[split]["mask_id"]))) != record["mask_ids_sha256"]:
            raise RuntimeError(f"fixed mask order mismatch: {split}")
    expected = {split: manifest["splits"][split]["run_uids"] for split in SPLITS}
    validate_view_splits(views, expected)
    return views, manifest


def fit_train_only_quality(views: dict[str, dict[str, np.ndarray]], config: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    errors = {split: per_sample_masked_mae(views[split]["clean"], views[split]["restored"], views[split]["observation"]) for split in SPLITS}
    settings = config["quality"]
    scale = fit_quality_scale(errors["train"], str(settings["scale_estimator"]))
    scores = {split: quality_scores(errors[split], scale, float(settings["q_min"])) for split in SPLITS}
    summary = {
        "formula": settings["formula"], "scale": scale,
        "scale_estimator": settings["scale_estimator"], "scale_fit_split": "train",
        "q_min": float(settings["q_min"]),
        "error_distribution": {split: distribution(errors[split]) for split in SPLITS},
        "quality_distribution": {split: distribution(scores[split]) for split in SPLITS},
    }
    return scores, summary


def epoch_orders(count: int, epochs: int, seed: int) -> list[np.ndarray]:
    return [np.random.default_rng(seed + epoch).permutation(count) for epoch in range(epochs)]


def _batches(order: np.ndarray, batch_size: int):
    for start in range(0, len(order), batch_size): yield order[start:start + batch_size]


def _contrastive_loss(model: torch.nn.Module, bundle: dict[str, np.ndarray], quality: np.ndarray,
                      order: np.ndarray, config: dict[str, Any], device: str,
                      optimizer: torch.optim.Optimizer | None) -> tuple[float, dict[str, float]]:
    training = optimizer is not None; model.train(training); losses = []; valid_anchors = []; gradient_norms = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for indices in _batches(order, int(config["batch_size"])):
            clean = torch.from_numpy(bundle["clean"][indices]).float().to(device)
            restored = torch.from_numpy(bundle["restored"][indices]).float().to(device)
            labels = torch.from_numpy(bundle["labels"][indices]).long().to(device)
            q = torch.from_numpy(quality[indices]).float().to(device)
            if optimizer is not None: optimizer.zero_grad()
            features = torch.cat([model(clean)["projection"], model(restored)["projection"]], 0)
            pair_labels = torch.cat([labels, labels], 0)
            candidate_weights = torch.cat([torch.ones_like(q), q], 0)
            loss = quality_weighted_supervised_contrastive_loss(features, pair_labels, candidate_weights, float(config["temperature"]))
            if optimizer is not None:
                loss.backward()
                gradient_norms.append(float(sum(parameter.grad.detach().square().sum() for parameter in model.parameters() if parameter.grad is not None).sqrt()))
                optimizer.step()
            losses.append(float(loss.detach()))
            counts = torch.bincount(pair_labels)
            valid_anchors.append(float(sum(int(count) for count in counts if count > 1)))
    return float(np.mean(losses)), {
        "mean_q": float(np.mean(quality[order])), "min_q": float(np.min(quality[order])),
        "max_q": float(np.max(quality[order])), "mean_valid_anchors": float(np.mean(valid_anchors)),
        "mean_gradient_norm": float(np.mean(gradient_norms)) if gradient_norms else 0.0,
    }


def _fit_supcon(model: torch.nn.Module, train: dict[str, np.ndarray], validation: dict[str, np.ndarray],
                train_q: np.ndarray, val_q: np.ndarray, orders: list[np.ndarray],
                config: dict[str, Any], device: str) -> list[dict[str, Any]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    history = []; best_state = None; best_loss = float("inf"); stale = 0
    validation_order = np.arange(len(validation["labels"]))
    for epoch, order in enumerate(orders):
        loss, diagnostics = _contrastive_loss(model, train, train_q, order, config, device, optimizer)
        val_loss, _ = _contrastive_loss(model, validation, val_q, validation_order, config, device, None)
        history.append({"epoch": epoch, "loss": loss, "validation_supcon_loss": val_loss, **diagnostics})
        if val_loss < best_loss - 1e-6:
            best_loss, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= int(config["early_stopping_patience"]): break
    if best_state is not None: model.load_state_dict(best_state)
    return history


def _probabilities(model: torch.nn.Module, x: np.ndarray, batch_size: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval(); probabilities = []; embeddings = []
    with torch.no_grad():
        for indices in _batches(np.arange(len(x)), batch_size):
            output = model(torch.from_numpy(x[indices]).float().to(device))
            probabilities.append(torch.softmax(output["logits"], 1).cpu().numpy())
            embeddings.append(output["embedding"].cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(embeddings)


def _fit_probe(model: torch.nn.Module, train: dict[str, np.ndarray], validation: dict[str, np.ndarray],
               orders: list[np.ndarray], config: dict[str, Any], device: str) -> list[dict[str, float]]:
    for parameter in model.parameters(): parameter.requires_grad = False
    for parameter in model.classification_head.parameters(): parameter.requires_grad = True
    optimizer = torch.optim.Adam(model.classification_head.parameters(), lr=float(config["learning_rate"]))
    history = []; best_state = None; best_score = -1.0; stale = 0
    for epoch, order in enumerate(orders):
        model.train(); losses = []
        for indices in _batches(order, int(config["batch_size"])):
            x = torch.from_numpy(train["clean"][indices]).float().to(device)
            y = torch.from_numpy(train["labels"][indices]).long().to(device)
            optimizer.zero_grad(); loss = F.cross_entropy(model(x)["logits"], y)
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        probability, _ = _probabilities(model, validation["restored"], int(config["batch_size"]), device)
        prediction = probability[:, 1] >= .5
        score = float(classification_metrics(validation["labels"], prediction, probability)["macro_f1"])
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "validation_macro_f1": score})
        if score > best_score + 1e-6:
            best_score, best_state, stale = score, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= int(config["probe_early_stopping_patience"]): break
    if best_state is not None: model.load_state_dict(best_state)
    return history


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    prediction = (scores >= threshold).astype(np.int64)
    result = classification_metrics(y, prediction, np.column_stack([1 - scores, scores]))
    result["fault_recall"] = float(recall_score(y, prediction, pos_label=1, zero_division=0))
    return result


def _quality_groups(y: np.ndarray, scores: np.ndarray, quality: np.ndarray, threshold: float) -> dict[str, Any]:
    boundary = float(np.median(quality)); groups = {}
    for name, selector in (("low", quality < boundary), ("high", quality >= boundary)):
        groups[name] = {"count": int(selector.sum()), "mean_quality": float(quality[selector].mean()),
                        "metrics": _metrics(y[selector], scores[selector], threshold)}
    groups["median_boundary"] = boundary
    return groups


def run(config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter(); seed = int(config["random_seed"]); device = str(config["device"])
    views, view_manifest = load_fixed_views(config)
    quality, quality_summary = fit_train_only_quality(views, config)
    pretrain_orders = epoch_orders(len(views["train"]["labels"]), int(config["epochs"]), seed + 10_000)
    probe_orders = epoch_orders(len(views["train"]["labels"]), int(config["probe_epochs"]), seed + 20_000)
    order_hash = sha256_strings([",".join(map(str, value.tolist())) for value in pretrain_orders])

    seed_everything(seed)
    template = build_model(config["model"], views["train"]["clean"].shape[1], 2)
    initial_state = copy.deepcopy(template.state_dict()); initial_hash = _state_hash(initial_state)
    results: dict[str, Any] = {}
    for method, weighted in (("Diffusion + Hard SupCon", False), ("Diffusion + Oracle Quality SupCon", True)):
        seed_everything(seed); torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        method_started = time.perf_counter()
        model = build_model(config["model"], views["train"]["clean"].shape[1], 2).to(device)
        model.load_state_dict(initial_state)
        train_q = quality["train"] if weighted else np.ones(len(views["train"]["labels"]), np.float32)
        val_q = quality["validation"] if weighted else np.ones(len(views["validation"]["labels"]), np.float32)
        pretrain = _fit_supcon(model, views["train"], views["validation"], train_q, val_q, pretrain_orders, config, device)
        seed_everything(seed + 1)
        probe = _fit_probe(model, views["train"], views["validation"], probe_orders, config, device)
        val_probability, _ = _probabilities(model, views["validation"]["restored"], int(config["batch_size"]), device)
        threshold = select_binary_threshold(views["validation"]["labels"], val_probability[:, 1])
        test_probability, restored_embedding = _probabilities(model, views["test"]["restored"], int(config["batch_size"]), device)
        _, clean_embedding = _probabilities(model, views["test"]["clean"], int(config["batch_size"]), device)
        metrics = _metrics(views["test"]["labels"], test_probability[:, 1], threshold)
        diagnostics = representation_diagnostics(clean_embedding, restored_embedding, views["test"]["labels"])
        results[method] = {
            "weighted": weighted, "initialization_sha256": initial_hash, "batch_order_sha256": order_hash,
            "validation_threshold": threshold, "metrics": metrics,
            "representation": {key: diagnostics[key] for key in ("fisher_ratio", "class_center_shift", "effective_rank")},
            "quality_groups": _quality_groups(views["test"]["labels"], test_probability[:, 1], quality["test"], threshold),
            "pretrain_history": pretrain, "probe_history": probe,
            "training_seconds": time.perf_counter() - method_started,
            "peak_gpu_mib": torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0,
        }

    teacher = build_model(config["model"], views["train"]["clean"].shape[1], 2).to(device)
    teacher.load_state_dict(torch.load(config["teacher_checkpoint"], map_location=device, weights_only=True))
    clean_teacher, _ = _probabilities(teacher, views["test"]["clean"], int(config["batch_size"]), device)
    restored_teacher, _ = _probabilities(teacher, views["test"]["restored"], int(config["batch_size"]), device)
    teacher_consistency = float(np.mean(clean_teacher.argmax(1) == restored_teacher.argmax(1)))
    for value in results.values(): value["teacher_consistency"] = teacher_consistency

    hard = results["Diffusion + Hard SupCon"]; oracle = results["Diffusion + Oracle Quality SupCon"]
    hm, om = hard["metrics"], oracle["metrics"]
    checks = {
        "macro_f1_improved": om["macro_f1"] > hm["macro_f1"],
        "far_reduced": om["far"] < hm["far"],
        "fault_recall_drop_within_1_point": om["fault_recall"] >= hm["fault_recall"] - .01,
        "auprc_drop_within_half_point": om["auprc"] >= hm["auprc"] - .005,
        "oracle_high_quality_macro_f1_above_low": oracle["quality_groups"]["high"]["metrics"]["macro_f1"] > oracle["quality_groups"]["low"]["metrics"]["macro_f1"],
        "fair_initialization_and_batch_order": hard["initialization_sha256"] == oracle["initialization_sha256"] and hard["batch_order_sha256"] == oracle["batch_order_sha256"],
    }
    status = "QUALITY_WEIGHTING_IDEA_GO" if sum(checks.values()) >= 4 and checks["fair_initialization_and_batch_order"] else "QUALITY_WEIGHTING_IDEA_NO_GO"
    result = {
        "markers": config["markers"], "status": status, **environment_metadata(),
        "fixed_view_manifest_sha256": sha256_file(config["fixed_views"]["manifest"]),
        "fixed_view_manifest": view_manifest, "quality": quality_summary,
        "fairness": {"initialization_sha256": initial_hash, "batch_order_sha256": order_hash,
                     "same_optimizer": True, "same_epochs": True, "same_probe": True,
                     "same_views": True, "same_temperature": True},
        "results": results, "gate_checks": checks, "total_seconds": time.perf_counter() - started,
    }
    output = Path(config["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    write_json(output / "result.json", result)
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = ["method", "macro_f1", "auprc", "fault_recall", "far", "auroc", "teacher_consistency", "training_seconds", "peak_gpu_mib"]
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader()
        for method, value in results.items():
            writer.writerow({"method": method, **{key: value["metrics"][key] for key in ("macro_f1", "auprc", "fault_recall", "far", "auroc")},
                             "teacher_consistency": value["teacher_consistency"], "training_seconds": value["training_seconds"], "peak_gpu_mib": value["peak_gpu_mib"]})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/diffusion_quality_retest.yaml")
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = run(config)
    print(json.dumps({"status": result["status"], "checks": result["gate_checks"], "output": config["output_dir"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
