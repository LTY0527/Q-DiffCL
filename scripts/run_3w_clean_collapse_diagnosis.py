from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from datasets.three_w import discover_instances, process_features
from losses import supervised_contrastive_loss
from models import TCNClassifier
from scripts.run_3w_clean_baseline import (PRIMARY_CLASSES, evaluate_stream, instance_refs,
                                            load_split, materialize, probabilities,
                                            read_frame, stratified_refs)
from trainers.balanced import (PositiveSafeBatchSampler, positive_anchor_audit,
                               sqrt_inverse_frequency_weights)
from utils import seed_everything, select_device


def model_from_config(base, channels, device):
    spec = base["training"]["model"]
    return TCNClassifier(channels, int(spec["hidden_channels"]), int(spec["projection_dim"]),
                         len(PRIMARY_CLASSES), int(spec["levels"])).to(device)


def regular_loader(x, y, batch, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=batch,
                      shuffle=shuffle, generator=generator, num_workers=0)


def train_probe(model, train_x, train_y, val_x, val_y, weights, epochs, lr, batch, seed, device):
    for p in model.parameters(): p.requires_grad = False
    torch.manual_seed(seed + 30_000)
    model.classification_head.reset_parameters()
    for p in model.classification_head.parameters(): p.requires_grad = True
    weight = torch.from_numpy(weights).to(device); optimizer = torch.optim.Adam(model.classification_head.parameters(), lr=lr)
    best, best_score, history = None, -1.0, []
    for epoch in range(epochs):
        model.train(); losses = []
        for xb, yb in regular_loader(train_x, train_y, batch, True, seed + epoch):
            xb, yb = xb.to(device), yb.to(device); optimizer.zero_grad()
            loss = F.cross_entropy(model(xb)["logits"], yb, weight=weight); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        probability = probabilities(model, val_x, val_y, batch, device); score = float(f1_score(val_y, probability.argmax(1), average="macro", zero_division=0))
        row = {"epoch": epoch, "train_weighted_ce": float(np.mean(losses)), "validation_macro_f1": score}; history.append(row)
        if score > best_score: best_score = score; best = copy.deepcopy(model.state_dict())
    model.load_state_dict(best); return history


def train_balanced_supcon(model, x, y, val_x, val_y, config, device):
    sampler_cfg = config["sampler"]
    sampler = PositiveSafeBatchSampler(y, int(sampler_cfg["classes_per_batch"]), int(sampler_cfg["samples_per_class"]),
                                       int(sampler_cfg["batches_per_epoch"]), int(config["seed"]), float(sampler_cfg["max_oversampling"]))
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y)); optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    best, best_loss, history = None, math.inf, []
    for epoch in range(int(config["balanced_supcon_epochs"])):
        sampler.set_epoch(epoch); model.train(); losses = []
        for xb, yb in DataLoader(dataset, batch_sampler=sampler, num_workers=0):
            xb, yb = xb.to(device), yb.to(device); optimizer.zero_grad()
            loss = supervised_contrastive_loss(model(xb)["projection"], yb, float(config["temperature"])); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        model.eval(); validation = []
        with torch.no_grad():
            for xb, yb in regular_loader(val_x, val_y, 256, False, 0):
                validation.append(float(supervised_contrastive_loss(model(xb.to(device))["projection"], yb.to(device), float(config["temperature"]))))
        row = {"epoch": epoch, "train_supcon_loss": float(np.mean(losses)), "validation_supcon_loss": float(np.mean(validation)),
               "sample_counts": sampler.planned_sample_counts, "oversampling_factor": sampler.oversampling_factors}; history.append(row)
        if row["validation_supcon_loss"] < best_loss: best_loss = row["validation_supcon_loss"]; best = copy.deepcopy(model.state_dict())
        print("D2", row["epoch"], row["train_supcon_loss"], row["validation_supcon_loss"], flush=True)
    model.load_state_dict(best); return history, sampler


def train_balanced_ce(model, x, y, val_x, val_y, config, device):
    sampler_cfg = config["sampler"]
    sampler = PositiveSafeBatchSampler(y, int(sampler_cfg["classes_per_batch"]), int(sampler_cfg["samples_per_class"]),
                                       int(sampler_cfg["batches_per_epoch"]), int(config["seed"]), float(sampler_cfg["max_oversampling"]))
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y)); optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    best, best_score, history = None, -1.0, []
    for epoch in range(int(config["balanced_ce_epochs"])):
        sampler.set_epoch(epoch); model.train(); losses = []
        for xb, yb in DataLoader(dataset, batch_sampler=sampler, num_workers=0):
            xb, yb = xb.to(device), yb.to(device); optimizer.zero_grad(); loss = F.cross_entropy(model(xb)["logits"], yb)
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        probability = probabilities(model, val_x, val_y, 256, device); score = float(f1_score(val_y, probability.argmax(1), average="macro", zero_division=0))
        row = {"epoch": epoch, "train_ce": float(np.mean(losses)), "validation_macro_f1": score}; history.append(row)
        if score > best_score: best_score = score; best = copy.deepcopy(model.state_dict())
        print("D3", row, flush=True)
    model.load_state_dict(best); return history


def embedding_audit(model, bundles, device):
    result = {}; model.eval()
    with torch.no_grad():
        for split, (x, y) in bundles.items():
            embeddings = []
            for xb, _ in regular_loader(x, y, 256, False, 0): embeddings.append(model(xb.to(device))["embedding"].cpu().numpy())
            z = np.concatenate(embeddings); centroids = {int(c): z[y == c].mean(0) for c in np.unique(y)}
            intra = {int(c): float(np.linalg.norm(z[y == c] - centroids[c], axis=1).mean()) for c in centroids}
            pairs = [np.linalg.norm(centroids[a] - centroids[b]) for a in centroids for b in centroids if a < b]
            result[split] = {"mean": float(z.mean()), "std": float(z.std()), "mean_norm": float(np.linalg.norm(z, axis=1).mean()),
                             "per_class_intra_distance": intra, "mean_inter_centroid_distance": float(np.mean(pairs))}
    return result


def condition_row(name, metrics):
    return {"condition": name, "macro_f1": metrics["macro_f1"], "macro_recall": metrics["recall_macro"],
            "fault_recall": metrics["fault_recall"], "binary_auroc": metrics.get("auroc_fault_vs_normal"),
            "multiclass_auprc": metrics["auprc_multiclass_macro"], "far": metrics["far"],
            "early_recall": metrics["early_recall"], "mean_detection_delay": metrics["mean_detection_delay_seconds"],
            "detection_instance_rate": metrics["detected_instance_rate"]}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/3w_clean_collapse_diagnosis.yaml"); parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args(); diagnosis = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")); base = yaml.safe_load(Path(diagnosis["base_config"]).read_text(encoding="utf-8"))
    seed = int(diagnosis["seed"]); seed_everything(seed); device = select_device(base["training"]["device"]); output = Path(diagnosis["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    base_output = Path(diagnosis["base_output"]); preprocessor = json.loads((base_output / "preprocessor.json").read_text(encoding="utf-8")); d0 = json.loads((base_output / "result.json").read_text(encoding="utf-8"))
    split_wells = load_split(Path(base["protocol"]["split_manifest"])); instances = [x for x in discover_instances(args.data_root) if x.source == "WELL" and x.event_class in PRIMARY_CLASSES]
    by_instance = {x.instance_id: x for x in instances}; by_split = {s: [x for x in instances if x.well_id in wells] for s, wells in split_wells.items()}; refs_by_split = {}; refs_by_instance = {}
    for split, items in by_split.items():
        refs = []
        for instance in items:
            current = instance_refs(instance, int(base["protocol"]["window_length"]), int(base["protocol"]["stride"]), int(base["protocol"]["transient_offset"])); refs.extend(current); refs_by_instance[instance.instance_id] = current
        refs_by_split[split] = refs
    train_refs = stratified_refs(refs_by_split["train"], int(base["protocol"]["train_windows_per_class"]), seed)
    val_refs = stratified_refs(refs_by_split["validation"], int(base["protocol"]["validation_windows_per_class"]), seed + 1)
    append = bool(base["protocol"]["append_missing_mask"]); length = int(base["protocol"]["window_length"])
    train_x, train_y = materialize(train_refs, by_instance, preprocessor, length, append); val_x, val_y = materialize(val_refs, by_instance, preprocessor, length, append)
    test_embedding_refs = stratified_refs(refs_by_split["test"], 2000, seed + 2)
    test_embedding_x, test_embedding_y = materialize(test_embedding_refs, by_instance, preprocessor, length, append)
    channels = train_x.shape[1]; weights = sqrt_inverse_frequency_weights(train_y)
    anchor = positive_anchor_audit(train_y, int(base["training"]["batch_size"]), seed)
    audit = {"window_counts": {s: dict(Counter(ref.target for ref in refs)) for s, refs in refs_by_split.items()},
             "largest_smallest_ratio": {s: max(Counter(r.target for r in refs).values()) / min(Counter(r.target for r in refs).values()) for s, refs in refs_by_split.items()},
             "ordinary_batch_positive_pairs": anchor, "probe_weights": weights.tolist(), "probe_weight_source": "sampled training windows only"}
    d0_model = model_from_config(base, channels, device); d0_model.load_state_dict(torch.load(base_output / "model.pt", map_location=device, weights_only=True))
    audit["d0_embedding"] = embedding_audit(d0_model, {"train": (train_x, train_y), "validation": (val_x, val_y),
                                                           "test_diagnostic_sample": (test_embedding_x, test_embedding_y)}, device)
    d0_metrics, _ = evaluate_stream(d0_model, by_split["test"], refs_by_instance, preprocessor, base, device)
    # D1: same D0 encoder, reset and balance only the probe.
    d1_model = copy.deepcopy(d0_model); d1_history = train_probe(d1_model, train_x, train_y, val_x, val_y, weights, int(diagnosis["probe_epochs"]), float(diagnosis["learning_rate"]), 256, seed, device)
    d1_metrics, _ = evaluate_stream(d1_model, by_split["test"], refs_by_instance, preprocessor, base, device)
    torch.save(d1_model.state_dict(), output / "D1_model.pt")
    # D2: same architecture, positive-safe SupCon batch, then the identical balanced probe.
    seed_everything(seed); d2_model = model_from_config(base, channels, device); d2_pretrain, sampler = train_balanced_supcon(d2_model, train_x, train_y, val_x, val_y, diagnosis, device)
    d2_probe = train_probe(d2_model, train_x, train_y, val_x, val_y, weights, int(diagnosis["probe_epochs"]), float(diagnosis["learning_rate"]), 256, seed, device)
    d2_metrics, _ = evaluate_stream(d2_model, by_split["test"], refs_by_instance, preprocessor, base, device)
    torch.save(d2_model.state_dict(), output / "D2_model.pt")
    # D3: balanced-batch CE sanity; no SupCon.
    seed_everything(seed); d3_model = model_from_config(base, channels, device); d3_history = train_balanced_ce(d3_model, train_x, train_y, val_x, val_y, diagnosis, device)
    d3_metrics, _ = evaluate_stream(d3_model, by_split["test"], refs_by_instance, preprocessor, base, device)
    torch.save(d3_model.state_dict(), output / "D3_model.pt")
    conditions = {"D0": d0_metrics, "D1": d1_metrics, "D2": d2_metrics, "D3": d3_metrics}
    zero = {name: [row["original_class"] for row in metrics["per_class"] if row["recall"] == 0] for name, metrics in conditions.items()}
    d2_improved = d2_metrics["macro_f1"] > d0_metrics["macro_f1"] + 0.02 and len(zero["D2"]) <= 1 and d2_metrics["far"] <= min(0.8, d0_metrics["far"] + 0.15)
    if d2_improved: status = "3W_BALANCED_CLEAN_BASELINE_1SEED_GO"
    elif not zero["D1"] and d1_metrics["macro_f1"] > d0_metrics["macro_f1"]: status = "PROBE_IMBALANCE_DOMINANT"
    elif len(zero["D3"]) >= 3: status = "3W_DATA_PROTOCOL_REAUDIT_REQUIRED"
    else: status = "3W_SUPCON_REPRESENTATION_HOLD"
    result = {"status": status, "base_commit": "831d12c", "seed": seed, "local_source_of_truth_note": "D0 is local Seed 7/20-epoch checkpoint; prompt Seed 42/35-epoch numbers do not exist locally",
              "audit": audit, "sampler": {"p": sampler.p, "k": sampler.k, "batches": len(sampler), "planned_sample_counts": sampler.planned_sample_counts, "oversampling_factors": sampler.oversampling_factors},
              "conditions": conditions, "zero_recall_classes": zero, "histories": {"D1_probe": d1_history, "D2_pretrain": d2_pretrain, "D2_probe": d2_probe, "D3": d3_history}}
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = [condition_row(name, metrics) for name, metrics in conditions.items()]
    with (output / "3W_CLEAN_COLLAPSE_DIAGNOSIS.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    for name, metrics in conditions.items():
        with (output / f"{name}_per_class.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics["per_class"][0])); writer.writeheader(); writer.writerows(metrics["per_class"])
        np.savetxt(output / f"{name}_confusion_matrix.csv", np.asarray(metrics["confusion_matrix"], dtype=int), delimiter=",", fmt="%d")
    print(json.dumps({"status": status, "comparison": rows, "zero_recall": zero}, ensure_ascii=False))


if __name__ == "__main__": main()
