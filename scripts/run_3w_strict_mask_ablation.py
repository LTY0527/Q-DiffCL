from __future__ import annotations

import argparse
import copy
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score

import scripts.run_3w_clean_baseline as base3w
from datasets.three_w import discover_instances
from losses import supervised_contrastive_loss
from models import TCNClassifier
from scripts.run_3w_clean_collapse_diagnosis import regular_loader, train_probe
from trainers.balanced import sqrt_inverse_frequency_weights
from utils import seed_everything, select_device


STRICT_CLASSES = (0, 2, 4, 7, 8, 9)


def load_strict_manifest(path: Path) -> dict[str, set[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = {name: set(payload[name]) for name in ("train", "validation", "test")}
    groups = list(result.values())
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("WELL leakage in strict manifest")
    return result


def build_model(spec, channels, device):
    return TCNClassifier(channels, int(spec["hidden_channels"]), int(spec["projection_dim"]),
                         len(STRICT_CLASSES), int(spec["levels"])).to(device)


def pretrain(model, train_x, train_y, val_x, val_y, config, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"])); best, best_loss, history = None, float("inf"), []
    for epoch in range(int(config["pretrain_epochs"])):
        model.train(); losses = []
        for xb, yb in regular_loader(train_x, train_y, int(config["batch_size"]), True, int(config["seed"]) + epoch):
            xb, yb = xb.to(device), yb.to(device); optimizer.zero_grad()
            loss = supervised_contrastive_loss(model(xb)["projection"], yb, float(config["temperature"])); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        model.eval(); val_losses = []
        with torch.no_grad():
            for xb, yb in regular_loader(val_x, val_y, int(config["batch_size"]), False, 0):
                val_losses.append(float(supervised_contrastive_loss(model(xb.to(device))["projection"], yb.to(device), float(config["temperature"]))))
        row = {"epoch": epoch, "train_supcon_loss": float(np.mean(losses)), "validation_supcon_loss": float(np.mean(val_losses))}; history.append(row)
        if row["validation_supcon_loss"] < best_loss: best_loss = row["validation_supcon_loss"]; best = copy.deepcopy(model.state_dict())
        print("pretrain", row, flush=True)
    model.load_state_dict(best); return history


def condition_row(name, metrics):
    return {"condition": name, "channels": 22 if name == "PROCESS_ONLY" else 44, "macro_f1": metrics["macro_f1"],
            "macro_recall": metrics["recall_macro"], "fault_recall": metrics["fault_recall"],
            "binary_auprc": metrics["auprc_fault_vs_normal"], "multiclass_auprc": metrics["auprc_multiclass_macro"],
            "far": metrics["far"], "early_recall": metrics["early_recall"],
            "mean_detection_delay_seconds": metrics["mean_detection_delay_seconds"], "detection_instance_rate": metrics["detected_instance_rate"]}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/3w_strict_mask_ablation.yaml"); parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")); base = yaml.safe_load(Path(config["base_config"]).read_text(encoding="utf-8"))
    seed = int(config["seed"]); seed_everything(seed); output = Path(config["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    # Reconfigure only the dataset label vocabulary; shared functions retain frozen label/onset semantics.
    base3w.PRIMARY_CLASSES = STRICT_CLASSES; base3w.CLASS_TO_TARGET = {value: index for index, value in enumerate(STRICT_CLASSES)}
    split_wells = load_strict_manifest(Path(config["split_manifest"])); instances = [item for item in discover_instances(args.data_root) if item.source == "WELL" and item.event_class in STRICT_CLASSES]
    by_instance = {item.instance_id: item for item in instances}; by_split = {name: [item for item in instances if item.well_id in wells] for name, wells in split_wells.items()}
    previous = json.loads(Path("outputs/3w_clean_baseline_seed7/preprocessor.json").read_text(encoding="utf-8")); frozen_features = tuple(previous["retained_features"])
    preprocessor_config = copy.deepcopy(base); preprocessor_config["protocol"]["feature_min_train_coverage"] = 0.0
    preprocessor = base3w.fit_preprocessor(by_split["train"], frozen_features, preprocessor_config)
    if tuple(preprocessor["retained_features"]) != frozen_features: raise RuntimeError("strict protocol changed frozen process features")
    (output / "preprocessor.json").write_text(json.dumps(preprocessor, ensure_ascii=False, indent=2), encoding="utf-8")
    refs_by_split, refs_by_instance = {}, {}
    for split, items in by_split.items():
        refs = []
        for instance in items:
            current = base3w.instance_refs(instance, int(base["protocol"]["window_length"]), int(base["protocol"]["stride"]), int(base["protocol"]["transient_offset"])); refs.extend(current); refs_by_instance[instance.instance_id] = current
        refs_by_split[split] = refs
    train_refs = base3w.stratified_refs(refs_by_split["train"], int(config["train_windows_per_class"]), seed)
    val_refs = base3w.stratified_refs(refs_by_split["validation"], int(config["validation_windows_per_class"]), seed + 1)
    length = int(base["protocol"]["window_length"])
    # Materialize once with masks, derive process-only from the exact same windows.
    combined_train_x, train_y = base3w.materialize(train_refs, by_instance, preprocessor, length, True)
    combined_val_x, val_y = base3w.materialize(val_refs, by_instance, preprocessor, length, True)
    process_channels = len(frozen_features); process_train_x = combined_train_x[:, :process_channels]; process_val_x = combined_val_x[:, :process_channels]
    weights = sqrt_inverse_frequency_weights(train_y); device = select_device(base["training"]["device"]); conditions = {}; histories = {}
    for name, train_x, val_x, append_mask in (("PROCESS_ONLY", process_train_x, process_val_x, False), ("PROCESS_PLUS_MASK", combined_train_x, combined_val_x, True)):
        seed_everything(seed); model = build_model(base["training"]["model"], train_x.shape[1], device)
        pretrain_history = pretrain(model, train_x, train_y, val_x, val_y, config, device)
        probe_history = train_probe(model, train_x, train_y, val_x, val_y, weights, int(config["probe_epochs"]), float(config["learning_rate"]), int(config["batch_size"]), seed, device)
        arm_base = copy.deepcopy(base); arm_base["protocol"]["append_missing_mask"] = append_mask; arm_base["training"]["batch_size"] = int(config["batch_size"])
        metrics, per_instance = base3w.evaluate_stream(model, by_split["test"], refs_by_instance, preprocessor, arm_base, device)
        conditions[name] = metrics; histories[name] = {"pretrain": pretrain_history, "probe": probe_history}
        torch.save(model.state_dict(), output / f"{name}_model.pt")
        with (output / f"{name}_per_class.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics["per_class"][0])); writer.writeheader(); writer.writerows(metrics["per_class"])
        np.savetxt(output / f"{name}_confusion_matrix.csv", np.asarray(metrics["confusion_matrix"], dtype=int), delimiter=",", fmt="%d")
    rows = [condition_row(name, metrics) for name, metrics in conditions.items()]
    with (output / "3w_mask_ablation_results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    zero = {name: [row["original_class"] for row in metrics["per_class"] if row["recall"] == 0] for name, metrics in conditions.items()}
    winner = max(conditions, key=lambda name: (conditions[name]["macro_f1"], -conditions[name]["far"]))
    winner_metrics = conditions[winner]
    stable = len(zero[winner]) <= 1 and winner_metrics["macro_f1"] >= 0.35 and winner_metrics["far"] <= 0.60
    status = "3W_STRICT_CLEAN_BASELINE_1SEED_GO" if stable else "3W_REAL_ONLY_PRIMARY_HOLD"
    support = {split: {str(original): len({item.well_id for item in by_split[split] if any(ref.target == target for ref in refs_by_instance[item.instance_id])}) for target, original in enumerate(STRICT_CLASSES)} for split in ("train", "validation", "test")}
    result = {"status": status, "base_commit": "294caa5", "seed": seed, "old_primary_classes": config["old_primary_classes"], "strict_primary_classes": list(STRICT_CLASSES),
              "secondary_classes": config["secondary_classes"], "split_well_support_by_target": support, "frozen_features": list(frozen_features),
              "shared_sampled_train_windows": dict(Counter(train_y.tolist())), "shared_sampled_validation_windows": dict(Counter(val_y.tolist())),
              "probe_weights_train_only": weights.tolist(), "conditions": conditions, "zero_recall_classes": zero, "selected_input": winner,
              "histories": histories, "checks": {"well_disjoint": True, "same_window_refs": True, "train_only_preprocessor": True,
                                                   "process_only_channels": process_channels, "process_plus_mask_channels": combined_train_x.shape[1], "no_diffusion": True}}
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": status, "selected": winner, "results": rows, "zero_recall": zero}, ensure_ascii=False))


if __name__ == "__main__": main()
