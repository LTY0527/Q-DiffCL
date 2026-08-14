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

import scripts.run_3w_clean_baseline as base3w
from datasets.three_w import discover_instances
from models import TCNClassifier
from scripts.run_3w_clean_collapse_diagnosis import train_probe
from scripts.run_3w_strict_mask_ablation import pretrain
from trainers.balanced import sqrt_inverse_frequency_weights
from utils import seed_everything, select_device


FINAL_PRIMARY_CLASSES = (0, 2, 8, 9)


def json_write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def split_coverage(split: dict[str, set[str]], well_targets: dict[str, set[int]]) -> dict[str, dict[int, int]]:
    return {
        name: {target: sum(target in well_targets[well] for well in wells) for target in range(len(FINAL_PRIMARY_CLASSES))}
        for name, wells in split.items()
    }


def grouped_split(
    all_wells: set[str],
    well_targets: dict[str, set[int]],
    counts: dict[str, int],
    minimum_fault_wells: dict[str, int],
    seed: int,
    previous_tests: list[set[str]] | None = None,
    maximum_test_jaccard: float = 1.0,
) -> dict[str, set[str]]:
    if sum(counts.values()) != len(all_wells):
        raise ValueError("split WELL counts do not cover the available WELL groups")
    ordered = np.asarray(sorted(all_wells), dtype=object)
    rng = np.random.default_rng(seed)
    previous_tests = previous_tests or []
    for _ in range(200_000):
        shuffled = ordered[rng.permutation(len(ordered))]
        train_end = counts["train"]
        validation_end = train_end + counts["validation"]
        split = {
            "train": set(shuffled[:train_end]),
            "validation": set(shuffled[train_end:validation_end]),
            "test": set(shuffled[validation_end:]),
        }
        coverage = split_coverage(split, well_targets)
        if any(coverage[name][target] < minimum_fault_wells[name] for name in split for target in range(1, len(FINAL_PRIMARY_CLASSES))):
            continue
        if any(len(split["test"] & old) / len(split["test"] | old) > maximum_test_jaccard for old in previous_tests):
            continue
        return split
    raise RuntimeError(f"cannot construct coverage-safe grouped split for seed {seed}")


def build_model(spec, channels: int, device: str):
    return TCNClassifier(
        channels,
        int(spec["hidden_channels"]),
        int(spec["projection_dim"]),
        len(FINAL_PRIMARY_CLASSES),
        int(spec["levels"]),
    ).to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/3w_final_primary_grouped.yaml")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--max-splits", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base = yaml.safe_load(Path(config["base_config"]).read_text(encoding="utf-8"))
    seed = int(config["seed"])
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    device = select_device(base["training"]["device"])

    base3w.PRIMARY_CLASSES = FINAL_PRIMARY_CLASSES
    base3w.CLASS_TO_TARGET = {value: index for index, value in enumerate(FINAL_PRIMARY_CLASSES)}
    instances = [item for item in discover_instances(args.data_root) if item.source == "WELL" and item.event_class in FINAL_PRIMARY_CLASSES]
    all_wells = {item.well_id for item in instances}
    by_instance = {item.instance_id: item for item in instances}
    refs_by_instance = {}
    well_targets = {well: set() for well in all_wells}
    for item in instances:
        refs = base3w.instance_refs(
            item,
            int(base["protocol"]["window_length"]),
            int(base["protocol"]["stride"]),
            int(base["protocol"]["transient_offset"]),
        )
        refs_by_instance[item.instance_id] = refs
        well_targets[item.well_id].update(ref.target for ref in refs)

    splits = []
    previous_tests: list[set[str]] = []
    for split_seed in config["split_seeds"]:
        split = grouped_split(
            all_wells,
            well_targets,
            {name: int(value) for name, value in config["split_well_counts"].items()},
            {name: int(value) for name, value in config["minimum_fault_wells"].items()},
            int(split_seed),
            previous_tests,
            float(config["maximum_test_jaccard"]),
        )
        splits.append(split)
        previous_tests.append(split["test"])
    json_write(
        output / "grouped_split_manifest.json",
        {
            "seed": seed,
            "split_seeds": config["split_seeds"],
            "primary_classes": list(FINAL_PRIMARY_CLASSES),
            "splits": [
                {
                    "split_index": index,
                    "wells": {name: sorted(wells) for name, wells in split.items()},
                    "target_well_coverage": split_coverage(split, well_targets),
                }
                for index, split in enumerate(splits)
            ],
        },
    )

    completed_now = 0
    frozen_features = tuple(config["frozen_process_features"])
    for split_index, split in enumerate(splits):
        split_dir = output / f"split_{split_index:02d}"
        result_path = split_dir / "result.json"
        if bool(config.get("resume")) and result_path.exists():
            print("skip", split_index, flush=True)
            continue
        if args.max_splits is not None and completed_now >= args.max_splits:
            break
        print("start", split_index + 1, len(splits), flush=True)
        by_split = {name: [item for item in instances if item.well_id in wells] for name, wells in split.items()}
        preprocessor_config = copy.deepcopy(base)
        preprocessor_config["protocol"]["feature_min_train_coverage"] = 0.0
        preprocessor_config["protocol"]["split_seed"] = int(config["split_seeds"][split_index])
        preprocessor = base3w.fit_preprocessor(by_split["train"], frozen_features, preprocessor_config)
        if tuple(preprocessor["retained_features"]) != frozen_features:
            raise RuntimeError("final protocol changed frozen process features")
        json_write(split_dir / "preprocessor.json", preprocessor)

        refs_by_split = {
            name: [ref for item in items for ref in refs_by_instance[item.instance_id]] for name, items in by_split.items()
        }
        train_refs = base3w.stratified_refs(refs_by_split["train"], int(config["train_windows_per_class"]), seed)
        validation_refs = base3w.stratified_refs(
            refs_by_split["validation"], int(config["validation_windows_per_class"]), seed + 1
        )
        length = int(base["protocol"]["window_length"])
        train_x, train_y = base3w.materialize(train_refs, by_instance, preprocessor, length, False)
        validation_x, validation_y = base3w.materialize(validation_refs, by_instance, preprocessor, length, False)
        weights = sqrt_inverse_frequency_weights(train_y)
        seed_everything(seed)
        model = build_model(base["training"]["model"], train_x.shape[1], device)
        pretrain_history = pretrain(model, train_x, train_y, validation_x, validation_y, config, device)
        probe_history = train_probe(
            model,
            train_x,
            train_y,
            validation_x,
            validation_y,
            weights,
            int(config["probe_epochs"]),
            float(config["learning_rate"]),
            int(config["batch_size"]),
            seed,
            device,
        )
        evaluation_config = copy.deepcopy(base)
        evaluation_config["protocol"]["append_missing_mask"] = False
        evaluation_config["training"]["batch_size"] = int(config["batch_size"])
        metrics, per_instance = base3w.evaluate_stream(
            model, by_split["test"], refs_by_instance, preprocessor, evaluation_config, device
        )
        torch.save(model.state_dict(), split_dir / "clean_model.pt")
        with (split_dir / "per_class.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics["per_class"][0]))
            writer.writeheader()
            writer.writerows(metrics["per_class"])
        np.savetxt(
            split_dir / "confusion_matrix.csv",
            np.asarray(metrics["confusion_matrix"], dtype=int),
            delimiter=",",
            fmt="%d",
        )
        json_write(
            result_path,
            {
                "split_index": split_index,
                "split_seed": int(config["split_seeds"][split_index]),
                "primary_classes": list(FINAL_PRIMARY_CLASSES),
                "split": {name: sorted(wells) for name, wells in split.items()},
                "target_well_coverage": split_coverage(split, well_targets),
                "sampled_train_windows": dict(Counter(train_y.tolist())),
                "sampled_validation_windows": dict(Counter(validation_y.tolist())),
                "probe_weights_train_only": weights.tolist(),
                "metrics": metrics,
                "per_instance": per_instance,
                "pretrain_history": pretrain_history,
                "probe_history": probe_history,
                "checks": {
                    "well_disjoint": True,
                    "train_only_preprocessor": True,
                    "process_only_channels": train_x.shape[1],
                    "no_diffusion": True,
                },
            },
        )
        print("done", split_index, {row["original_class"]: row["recall"] for row in metrics["per_class"]}, flush=True)
        completed_now += 1


if __name__ == "__main__":
    main()
