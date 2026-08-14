from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml

import scripts.run_3w_clean_baseline as base3w
from datasets.three_w import discover_instances
from scripts.run_3w_clean_collapse_diagnosis import train_probe
from scripts.run_3w_strict_mask_ablation import STRICT_CLASSES, build_model, pretrain
from trainers.balanced import sqrt_inverse_frequency_weights
from utils import seed_everything, select_device


def constrained_fold_split(all_wells: set[str], held_out: str, well_targets: dict[str, set[int]],
                           validation_count: int, seed: int) -> dict[str, set[str]]:
    remaining = np.asarray(sorted(all_wells - {held_out}), dtype=object); required = set(range(len(STRICT_CLASSES)))
    rng = np.random.default_rng(seed)
    for _ in range(200_000):
        shuffled = remaining[rng.permutation(len(remaining))]; validation = set(shuffled[:validation_count]); train = set(shuffled[validation_count:])
        val_coverage = set().union(*(well_targets[well] for well in validation)); train_coverage = set().union(*(well_targets[well] for well in train))
        if required <= val_coverage and required <= train_coverage:
            return {"train": train, "validation": validation, "test": {held_out}}
    raise RuntimeError(f"cannot construct coverage-safe LOOW split for {held_out}")


def json_write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/3w_class47_loow.yaml"); parser.add_argument("--data-root", type=Path, required=True); parser.add_argument("--max-folds", type=int)
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")); strict = yaml.safe_load(Path(config["base_config"]).read_text(encoding="utf-8")); base = yaml.safe_load(Path(strict["base_config"]).read_text(encoding="utf-8"))
    seed = int(config["seed"]); output = Path(config["output_dir"]); output.mkdir(parents=True, exist_ok=True); device = select_device(base["training"]["device"])
    base3w.PRIMARY_CLASSES = STRICT_CLASSES; base3w.CLASS_TO_TARGET = {value: index for index, value in enumerate(STRICT_CLASSES)}
    instances = [item for item in discover_instances(args.data_root) if item.source == "WELL" and item.event_class in STRICT_CLASSES]
    all_wells = {item.well_id for item in instances}; by_instance = {item.instance_id: item for item in instances}; refs_by_instance = {}
    well_targets = {well: set() for well in all_wells}
    for item in instances:
        refs = base3w.instance_refs(item, int(base["protocol"]["window_length"]), int(base["protocol"]["stride"]), int(base["protocol"]["transient_offset"])); refs_by_instance[item.instance_id] = refs; well_targets[item.well_id].update(ref.target for ref in refs)
    target_wells = {original: sorted(well for well in all_wells if base3w.CLASS_TO_TARGET[original] in well_targets[well]) for original in config["target_classes"]}
    unique_folds = sorted(set().union(*map(set, target_wells.values())))
    completed_now = 0
    for fold_index, held_out in enumerate(unique_folds):
        fold_path = output / "folds" / f"{held_out}.json"
        if bool(config.get("resume")) and fold_path.exists():
            print("skip", held_out, flush=True); continue
        if args.max_folds is not None and completed_now >= args.max_folds: break
        print("start", fold_index + 1, len(unique_folds), held_out, flush=True)
        split = constrained_fold_split(all_wells, held_out, well_targets, int(config["validation_wells"]), seed + fold_index)
        by_split = {name: [item for item in instances if item.well_id in wells] for name, wells in split.items()}
        previous = json.loads(Path("outputs/3w_clean_baseline_seed7/preprocessor.json").read_text(encoding="utf-8")); frozen_features = tuple(previous["retained_features"])
        preprocessor_config = copy.deepcopy(base); preprocessor_config["protocol"]["feature_min_train_coverage"] = 0.0
        preprocessor = base3w.fit_preprocessor(by_split["train"], frozen_features, preprocessor_config)
        refs_by_split = {name: [ref for item in items for ref in refs_by_instance[item.instance_id]] for name, items in by_split.items()}
        train_refs = base3w.stratified_refs(refs_by_split["train"], int(strict["train_windows_per_class"]), seed)
        val_refs = base3w.stratified_refs(refs_by_split["validation"], int(strict["validation_windows_per_class"]), seed + 1)
        length = int(base["protocol"]["window_length"])
        train_x, train_y = base3w.materialize(train_refs, by_instance, preprocessor, length, False); val_x, val_y = base3w.materialize(val_refs, by_instance, preprocessor, length, False)
        weights = sqrt_inverse_frequency_weights(train_y); seed_everything(seed); model = build_model(base["training"]["model"], train_x.shape[1], device)
        pretrain_history = pretrain(model, train_x, train_y, val_x, val_y, strict, device)
        probe_history = train_probe(model, train_x, train_y, val_x, val_y, weights, int(strict["probe_epochs"]), float(strict["learning_rate"]), int(strict["batch_size"]), seed, device)
        arm_base = copy.deepcopy(base); arm_base["protocol"]["append_missing_mask"] = False; arm_base["training"]["batch_size"] = int(strict["batch_size"])
        metrics, per_instance = base3w.evaluate_stream(model, by_split["test"], refs_by_instance, preprocessor, arm_base, device)
        target_records = {}
        for original in config["target_classes"]:
            if held_out in target_wells[original]:
                target = base3w.CLASS_TO_TARGET[original]; row = metrics["per_class"][target]
                target_records[str(original)] = {"recall": row["recall"], "f1": row["f1"], "precision": row["precision"], "support": row["support"]}
        payload = {"held_out_well": held_out, "fold_index": fold_index, "target_records": target_records,
                   "split": {name: sorted(wells) for name, wells in split.items()}, "split_well_target_coverage": {name: sorted(set().union(*(well_targets[w] for w in wells))) for name, wells in split.items()},
                   "metrics": metrics, "per_instance": per_instance, "pretrain_history": pretrain_history, "probe_history": probe_history,
                   "checks": {"held_out_absent_from_train": held_out not in split["train"], "held_out_absent_from_validation": held_out not in split["validation"],
                              "preprocessor_fit_wells": sorted(split["train"]), "process_only_channels": train_x.shape[1], "seed": seed}}
        json_write(fold_path, payload)
        with (output / "folds" / f"{held_out}_per_class.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics["per_class"][0])); writer.writeheader(); writer.writerows(metrics["per_class"])
        np.savetxt(output / "folds" / f"{held_out}_confusion_matrix.csv", np.asarray(metrics["confusion_matrix"], dtype=int), delimiter=",", fmt="%d")
        print("done", held_out, target_records, flush=True); completed_now += 1
    json_write(output / "fold_manifest.json", {"seed": seed, "strict_classes": list(STRICT_CLASSES), "target_wells": target_wells, "unique_folds": unique_folds})


if __name__ == "__main__": main()
