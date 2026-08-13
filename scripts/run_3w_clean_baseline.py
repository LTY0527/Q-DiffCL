from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, recall_score, roc_auc_score)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, TensorDataset

from datasets.three_w import ThreeWInstance, discover_instances, process_features
from losses import supervised_contrastive_loss
from models import TCNClassifier
from utils import seed_everything, select_device


PRIMARY_CLASSES = (0, 1, 2, 4, 5, 7, 8, 9)
CLASS_TO_TARGET = {value: index for index, value in enumerate(PRIMARY_CLASSES)}


@dataclass(frozen=True)
class WindowRef:
    instance_id: str
    start: int
    target: int
    stage: str
    end_seconds: float
    onset_seconds: float | None


def load_split(path: Path) -> dict[str, set[str]]:
    result = {name: set() for name in ("train", "validation", "test")}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            result[row["split"]].add(row["well_id"])
    groups = list(result.values())
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("WELL leakage in frozen split manifest")
    return result


def read_frame(instance: ThreeWInstance, columns: Iterable[str] | None = None):
    import pyarrow.parquet as pq
    table = pq.read_table(instance.path, columns=None if columns is None else list(columns))
    return table.to_pandas()


def stage_and_target(raw: np.ndarray, event_class: int, offset: int) -> tuple[np.ndarray, np.ndarray]:
    stages = np.full(len(raw), "unlabeled", dtype=object)
    targets = np.full(len(raw), -1, dtype=np.int16)
    finite = np.isfinite(raw)
    normal = finite & (raw == 0)
    transient = finite & (raw == event_class + offset)
    event = finite & (raw == event_class)
    stages[normal] = "normal"; targets[normal] = CLASS_TO_TARGET[0]
    if event_class != 0:
        stages[transient] = "early"; targets[transient] = CLASS_TO_TARGET[event_class]
        stages[event] = "established"; targets[event] = CLASS_TO_TARGET[event_class]
    return stages, targets


def fit_preprocessor(
    train_instances: list[ThreeWInstance], all_features: tuple[str, ...], config: dict[str, Any],
) -> dict[str, Any]:
    rng = np.random.default_rng(int(config["protocol"]["split_seed"]))
    samples: dict[str, list[np.ndarray]] = {name: [] for name in all_features}
    finite_counts = Counter(); total_rows = 0
    per_instance = int(config["protocol"]["statistics_sample_per_instance"])
    for instance in train_instances:
        frame = read_frame(instance, all_features)
        total_rows += len(frame)
        if len(frame) > per_instance:
            indices = np.sort(rng.choice(len(frame), per_instance, replace=False))
            sampled = frame.iloc[indices]
        else:
            sampled = frame
        for name in all_features:
            full = frame[name].to_numpy(dtype=np.float64, na_value=np.nan)
            finite_counts[name] += int(np.isfinite(full).sum())
            values = sampled[name].to_numpy(dtype=np.float64, na_value=np.nan)
            values = values[np.isfinite(values)]
            if len(values): samples[name].append(values)
    threshold = float(config["protocol"]["feature_min_train_coverage"])
    retained = [name for name in all_features if finite_counts[name] / total_rows >= threshold]
    low_q, high_q = map(float, config["protocol"]["quantile_clip"])
    statistics = {}
    for name in retained:
        values = np.concatenate(samples[name])
        lower, upper = np.quantile(values, [low_q, high_q])
        clipped = np.clip(values, lower, upper)
        median = float(np.median(clipped)); mean = float(clipped.mean()); scale = float(clipped.std())
        statistics[name] = {
            "train_coverage": finite_counts[name] / total_rows, "lower": float(lower), "upper": float(upper),
            "median": median, "mean": mean, "scale": scale if scale > 1e-12 else 1.0,
            "sample_count": len(values),
        }
    return {
        "fit_scope": "training WELL observations only", "total_train_rows": total_rows,
        "coverage_threshold": threshold, "quantile_clip": [low_q, high_q],
        "all_process_features": list(all_features), "retained_features": retained,
        "excluded_features": {name: "train coverage below threshold" for name in all_features if name not in retained},
        "statistics": statistics,
    }


def transform_frame(frame, preprocessor: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    values, masks = [], []
    for name in preprocessor["retained_features"]:
        stat = preprocessor["statistics"][name]
        raw = frame[name].to_numpy(dtype=np.float64, na_value=np.nan)
        mask = np.isfinite(raw)
        clipped = np.clip(raw, stat["lower"], stat["upper"])
        filled = np.where(mask, clipped, stat["median"])
        normalized = (filled - stat["mean"]) / stat["scale"]
        values.append(normalized.astype(np.float32)); masks.append(mask)
    x = np.stack(values, axis=1)
    observation = np.stack(masks, axis=1)
    if not np.isfinite(x).all():
        raise RuntimeError("non-finite value after train-fitted preprocessing")
    return x, observation


def instance_refs(instance: ThreeWInstance, length: int, stride: int, offset: int) -> list[WindowRef]:
    frame = read_frame(instance, ("class", "timestamp"))
    raw = frame["class"].to_numpy(dtype=np.float64, na_value=np.nan)
    stages, targets = stage_and_target(raw, instance.event_class, offset)
    timestamps = frame.index.to_numpy(dtype="datetime64[ns]") if "timestamp" not in frame.columns else frame["timestamp"].to_numpy(dtype="datetime64[ns]")
    seconds = (timestamps - timestamps[0]).astype("timedelta64[ns]").astype(np.int64) / 1e9
    onset_index = np.flatnonzero(np.isin(stages, ("early", "established")))
    onset = float(seconds[onset_index[0]]) if len(onset_index) else None
    refs = []
    for start in range(0, len(frame) - length + 1, stride):
        end = start + length - 1
        if targets[end] < 0:
            continue
        refs.append(WindowRef(instance.instance_id, start, int(targets[end]), str(stages[end]), float(seconds[end]), onset))
    return refs


def stratified_refs(refs: list[WindowRef], per_class: int, seed: int) -> list[WindowRef]:
    rng = np.random.default_rng(seed); grouped: dict[int, list[WindowRef]] = defaultdict(list)
    for ref in refs: grouped[ref.target].append(ref)
    chosen = []
    for target in range(len(PRIMARY_CLASSES)):
        items = grouped[target]
        if not items: raise RuntimeError(f"split has no windows for target {target}")
        indices = rng.choice(len(items), min(per_class, len(items)), replace=False)
        chosen.extend(items[index] for index in indices)
    return sorted(chosen, key=lambda item: (item.instance_id, item.start))


def materialize(refs: list[WindowRef], by_instance: dict[str, ThreeWInstance], preprocessor: dict[str, Any], length: int, append_mask: bool) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[str, list[tuple[int, WindowRef]]] = defaultdict(list)
    for index, ref in enumerate(refs): grouped[ref.instance_id].append((index, ref))
    channels = len(preprocessor["retained_features"]) * (2 if append_mask else 1)
    x = np.empty((len(refs), channels, length), dtype=np.float32); y = np.empty(len(refs), dtype=np.int64)
    for instance_id, items in grouped.items():
        frame = read_frame(by_instance[instance_id], preprocessor["retained_features"])
        values, mask = transform_frame(frame, preprocessor)
        for index, ref in items:
            window = values[ref.start:ref.start + length].T
            if append_mask: window = np.concatenate((window, mask[ref.start:ref.start + length].T.astype(np.float32)))
            x[index] = window; y[index] = ref.target
    return x, y


def loader(x: np.ndarray, y: np.ndarray, batch: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=batch,
                      shuffle=shuffle, generator=generator, num_workers=0)


def probabilities(model, x: np.ndarray, y: np.ndarray, batch: int, device: str) -> np.ndarray:
    model.eval(); result = []
    with torch.no_grad():
        for xb, _ in loader(x, y, batch, False, 0): result.append(torch.softmax(model(xb.to(device))["logits"], 1).cpu().numpy())
    return np.concatenate(result)


def train_model(train_x, train_y, val_x, val_y, config, output: Path):
    runtime = config["training"]; device = select_device(runtime["device"]); batch = int(runtime["batch_size"])
    model = TCNClassifier(train_x.shape[1], int(runtime["model"]["hidden_channels"]),
                          int(runtime["model"]["projection_dim"]), len(PRIMARY_CLASSES),
                          int(runtime["model"]["levels"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(runtime["learning_rate"])); pretrain = []
    best_state, best_loss = None, math.inf
    for epoch in range(int(runtime["pretrain_epochs"])):
        model.train(); losses = []
        for xb, yb in loader(train_x, train_y, batch, True, int(runtime["seed"]) + epoch):
            xb, yb = xb.to(device), yb.to(device); optimizer.zero_grad()
            loss = supervised_contrastive_loss(model(xb)["projection"], yb, float(runtime["temperature"]))
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        model.eval(); validation = []
        with torch.no_grad():
            for xb, yb in loader(val_x, val_y, batch, False, 0):
                validation.append(float(supervised_contrastive_loss(model(xb.to(device))["projection"], yb.to(device), float(runtime["temperature"]))))
        record = {"epoch": epoch, "train_supcon_loss": float(np.mean(losses)), "validation_supcon_loss": float(np.mean(validation))}; pretrain.append(record)
        if record["validation_supcon_loss"] < best_loss:
            best_loss = record["validation_supcon_loss"]; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print("pretrain", record, flush=True)
    model.load_state_dict(best_state)
    for parameter in model.encoder.parameters(): parameter.requires_grad = False
    for parameter in model.projection_head.parameters(): parameter.requires_grad = False
    optimizer = torch.optim.Adam(model.classification_head.parameters(), lr=float(runtime["learning_rate"])); probe = []
    best_state, best_f1 = None, -1.0
    for epoch in range(int(runtime["probe_epochs"])):
        model.train(); losses = []
        for xb, yb in loader(train_x, train_y, batch, True, int(runtime["seed"]) + 10_000 + epoch):
            xb, yb = xb.to(device), yb.to(device); optimizer.zero_grad(); loss = F.cross_entropy(model(xb)["logits"], yb)
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        val_probability = probabilities(model, val_x, val_y, batch, device); val_prediction = val_probability.argmax(1)
        score = float(f1_score(val_y, val_prediction, average="macro", zero_division=0))
        record = {"epoch": epoch, "train_ce": float(np.mean(losses)), "validation_macro_f1": score}; probe.append(record)
        if score > best_f1: best_f1 = score; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print("probe", record, flush=True)
    model.load_state_dict(best_state); output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / "model.pt")
    return model, device, pretrain, probe


def evaluate_stream(model, instances, refs_by_instance, preprocessor, config, device):
    length = int(config["protocol"]["window_length"]); append_mask = bool(config["protocol"]["append_missing_mask"])
    batch = int(config["training"]["batch_size"]); ys, predictions, probability_rows = [], [], []
    early_true = early_correct = 0; delays = []; missed = 0
    per_instance = {}
    for instance in instances:
        refs = refs_by_instance.get(instance.instance_id, [])
        if not refs: continue
        x, y = materialize(refs, {instance.instance_id: instance}, preprocessor, length, append_mask)
        probability = probabilities(model, x, y, batch, device); prediction = probability.argmax(1)
        ys.append(y); predictions.append(prediction); probability_rows.append(probability)
        early = np.asarray([ref.stage == "early" for ref in refs]); early_true += int(early.sum()); early_correct += int((prediction[early] != 0).sum())
        onset = refs[0].onset_seconds
        fault_windows = [(ref, int(pred)) for ref, pred in zip(refs, prediction) if ref.target != 0 and onset is not None and ref.end_seconds >= onset]
        detections = [ref.end_seconds - float(onset) for ref, pred in fault_windows if pred != 0]
        if onset is not None and fault_windows:
            if detections: delays.append(float(detections[0]))
            else: missed += 1
        per_instance[instance.instance_id] = {"windows": len(refs), "onset_seconds": onset, "delay_seconds": detections[0] if detections else None}
    y = np.concatenate(ys); prediction = np.concatenate(predictions); probability = np.concatenate(probability_rows)
    precision, recall, f1, support = precision_recall_fscore_support(y, prediction, labels=np.arange(len(PRIMARY_CLASSES)), zero_division=0)
    binary_true = y != 0; fault_score = 1 - probability[:, 0]
    metrics = {
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, prediction, average="macro", zero_division=0)),
        "fault_recall": float(np.mean(prediction[binary_true] != 0)),
        "auprc_fault_vs_normal": float(average_precision_score(binary_true.astype(int), fault_score)),
        "auroc_fault_vs_normal": float(roc_auc_score(binary_true.astype(int), fault_score)),
        "auprc_multiclass_macro": float(average_precision_score(label_binarize(y, classes=np.arange(len(PRIMARY_CLASSES))), probability, average="macro")),
        "far": float(np.mean(prediction[y == 0] != 0)), "accuracy": float(np.mean(y == prediction)),
        "early_recall": float(early_correct / early_true) if early_true else None,
        "mean_detection_delay_seconds": float(np.mean(delays)) if delays else None,
        "median_detection_delay_seconds": float(np.median(delays)) if delays else None,
        "detected_instance_rate": len(delays) / (len(delays) + missed) if delays or missed else None,
        "confusion_matrix": confusion_matrix(y, prediction, labels=np.arange(len(PRIMARY_CLASSES))).tolist(),
        "per_class": [
            {"target": index, "original_class": original, "precision": float(precision[index]),
             "recall": float(recall[index]), "f1": float(f1[index]), "support": int(support[index])}
            for index, original in enumerate(PRIMARY_CLASSES)
        ], "predicted_class_histogram": {str(original): int(np.sum(prediction == index)) for index, original in enumerate(PRIMARY_CLASSES)},
        "true_class_histogram": {str(original): int(np.sum(y == index)) for index, original in enumerate(PRIMARY_CLASSES)},
        "windows": len(y), "early_windows": early_true, "evaluated_fault_instances": len(delays) + missed,
    }
    return metrics, per_instance


def split_counts(instances, split_wells):
    result = {}
    for split, wells in split_wells.items():
        selected = [item for item in instances if item.well_id in wells and item.event_class in PRIMARY_CLASSES]
        result[split] = {"wells": len(wells), "instances": len(selected), "per_class_instances": dict(Counter(item.event_class for item in selected)),
                         "per_class_wells": {c: len({item.well_id for item in selected if item.event_class == c}) for c in PRIMARY_CLASSES}}
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/3w_clean_baseline.yaml"); parser.add_argument("--data-root", type=Path, required=True); parser.add_argument("--evaluate-existing", action="store_true")
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")); config["dataset"]["data_root"] = str(args.data_root)
    seed_everything(int(config["training"]["seed"])); output = Path(config["output_dir"]); output.mkdir(parents=True, exist_ok=True); started = time.perf_counter()
    split_wells = load_split(Path(config["protocol"]["split_manifest"])); instances = [item for item in discover_instances(args.data_root) if item.source == "WELL" and item.event_class in PRIMARY_CLASSES]
    by_instance = {item.instance_id: item for item in instances}; by_split = {name: [item for item in instances if item.well_id in wells] for name, wells in split_wells.items()}
    all_features = process_features(read_frame(instances[0]).columns)
    if args.evaluate_existing:
        preprocessor = json.loads((output / "preprocessor.json").read_text(encoding="utf-8"))
    else:
        preprocessor = fit_preprocessor(by_split["train"], all_features, config)
        (output / "preprocessor.json").write_text(json.dumps(preprocessor, indent=2, ensure_ascii=False), encoding="utf-8")
    refs_by_split, refs_by_instance = {}, {}
    for split, items in by_split.items():
        refs = []
        for index, instance in enumerate(items, 1):
            current = instance_refs(instance, int(config["protocol"]["window_length"]), int(config["protocol"]["stride"]), int(config["protocol"]["transient_offset"]))
            refs.extend(current); refs_by_instance[instance.instance_id] = current
            if index % 100 == 0: print("refs", split, index, len(items), flush=True)
        refs_by_split[split] = refs
    length = int(config["protocol"]["window_length"]); append_mask = bool(config["protocol"]["append_missing_mask"])
    train_refs = stratified_refs(refs_by_split["train"], int(config["protocol"]["train_windows_per_class"]), int(config["training"]["seed"]))
    val_refs = stratified_refs(refs_by_split["validation"], int(config["protocol"]["validation_windows_per_class"]), int(config["training"]["seed"]) + 1)
    if args.evaluate_existing:
        runtime = config["training"]; device = select_device(runtime["device"])
        channels = len(preprocessor["retained_features"]) * (2 if append_mask else 1)
        model = TCNClassifier(channels, int(runtime["model"]["hidden_channels"]), int(runtime["model"]["projection_dim"]),
                              len(PRIMARY_CLASSES), int(runtime["model"]["levels"])).to(device)
        model.load_state_dict(torch.load(output / "model.pt", map_location=device, weights_only=True))
        previous = json.loads((output / "result.json").read_text(encoding="utf-8")); pretrain = previous["pretrain_history"]; probe = previous["probe_history"]
        train_y = np.asarray([ref.target for ref in train_refs]); val_y = np.asarray([ref.target for ref in val_refs])
    else:
        train_x, train_y = materialize(train_refs, by_instance, preprocessor, length, append_mask); val_x, val_y = materialize(val_refs, by_instance, preprocessor, length, append_mask)
        model, device, pretrain, probe = train_model(train_x, train_y, val_x, val_y, config, output)
    metrics, per_instance = evaluate_stream(model, by_split["test"], refs_by_instance, preprocessor, config, device)
    ref_counts = {split: dict(Counter(ref.target for ref in refs)) for split, refs in refs_by_split.items()}
    result = {"status": "3W_CLEAN_BASELINE_1SEED_GO", "seed": int(config["training"]["seed"]), "primary_classes": list(PRIMARY_CLASSES),
              "excluded_classes": [3, 6], "split_counts": split_counts(instances, split_wells), "all_window_counts_by_target": ref_counts,
              "sampled_train_windows": dict(Counter(train_y.tolist())), "sampled_validation_windows": dict(Counter(val_y.tolist())),
              "metrics": metrics, "pretrain_history": pretrain, "probe_history": probe, "runtime_seconds": time.perf_counter() - started,
              "protocol_checks": {"well_disjoint": True, "split_before_window": True, "train_only_preprocessor_fit": True,
                                  "native_mask_appended": append_mask, "extra_mcar": False, "diffusion": False}}
    # GO requires finite/convergent training and every primary class to be predicted at least once.
    finite_training = all(np.isfinite(row["train_supcon_loss"]) for row in pretrain) and all(np.isfinite(row["train_ce"]) for row in probe)
    zero_recall = [row["original_class"] for row in metrics["per_class"] if row["recall"] == 0]
    if not finite_training or zero_recall:
        result["status"] = "3W_CLEAN_BASELINE_1SEED_HOLD"
        result["hold_reason"] = {"finite_training": finite_training, "zero_recall_primary_classes": zero_recall}
    (output / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "per_instance_delay.json").write_text(json.dumps(per_instance, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output / "per_class_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics["per_class"][0])); writer.writeheader(); writer.writerows(metrics["per_class"])
    np.savetxt(output / "confusion_matrix.csv", np.asarray(metrics["confusion_matrix"], dtype=int), delimiter=",", fmt="%d")
    print(json.dumps({"status": result["status"], **{k: metrics[k] for k in ("macro_f1", "fault_recall", "auprc_fault_vs_normal", "far", "early_recall", "mean_detection_delay_seconds")}}, ensure_ascii=False))


if __name__ == "__main__": main()
