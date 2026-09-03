from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score


MANIFEST_ROOT = Path("configs/data_regime_manifests")
RESULT_ROOT = Path("outputs/qdiffcl_data_regime_v1/DATA_REGIME_GENERALIZATION_V1")


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader(); writer.writerows(rows)


def _group_id(dataset: str, source_id: str) -> str:
    if dataset == "3W":
        return source_id.split("_", 1)[0]
    return source_id.rsplit(":", 1)[0]


def composition_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    composition = []; groups = []; nested = []
    for path in sorted(MANIFEST_ROOT.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8")); dataset = manifest["dataset"]
        fractions = manifest["fractions"]
        full = set(fractions["1.0"]["source_ids"])
        quarter = set(fractions["0.25"]["source_ids"])
        tenth = set(fractions["0.1"]["source_ids"])
        nested.append({
            "dataset": dataset, "outer_id": manifest["outer_id"],
            "ten_in_quarter": tenth < quarter, "quarter_in_full": quarter < full,
            "ten_count": len(tenth), "quarter_count": len(quarter), "full_count": len(full),
        })
        for fraction_text, record in fractions.items():
            for class_id, count in sorted(record["class_counts"].items(), key=lambda item: int(item[0])):
                composition.append({
                    "dataset": dataset, "outer_id": manifest["outer_id"],
                    "fraction": float(fraction_text), "class_id": int(class_id),
                    "source_unit_count": int(count), "realized_units": record["realized_units"],
                    "realized_fraction": record["realized_fraction"],
                    "independent_group_count": record["group_counts"],
                    "onset_bearing_units": record["stage_counts"]["onset_bearing_units"],
                })
            counts = Counter(_group_id(dataset, source) for source in record["source_ids"])
            for group_id, count in sorted(counts.items()):
                groups.append({
                    "dataset": dataset, "outer_id": manifest["outer_id"],
                    "fraction": float(fraction_text), "group_id": group_id,
                    "source_unit_count": count,
                })
    return composition, groups, nested


def no_aug_per_class_rows() -> list[dict[str, Any]]:
    rows = []
    for path in sorted(RESULT_ROOT.glob("3w/f*/outer_*/model_seed_*/NO_AUG/predictions.npz")):
        fraction = int(path.parents[3].name[1:]) / 100
        outer = int(path.parents[2].name.split("_")[1])
        seed = int(path.parents[1].name.split("_")[-1])
        with np.load(path, allow_pickle=False) as archive:
            labels = archive["label"].astype(np.int64)
            prediction = archive["prediction"].astype(np.int64)
        values = f1_score(labels, prediction, labels=[0, 1, 2, 3], average=None, zero_division=0)
        for target, original_class, value in zip((0, 1, 2, 3), (0, 2, 8, 9), values):
            rows.append({
                "dataset": "3W", "fraction": fraction, "outer_id": outer, "model_seed": seed,
                "target": target, "original_class": original_class, "no_aug_f1": float(value),
            })
    return rows


def main() -> None:
    composition, groups, nested = composition_rows(); performance = no_aug_per_class_rows()
    _write(Path("analysis/results/qdiffcl_data_regime_fraction_composition.csv"), composition)
    _write(Path("analysis/results/qdiffcl_data_regime_fraction_groups.csv"), groups)
    _write(Path("analysis/results/qdiffcl_data_regime_no_aug_per_class.csv"), performance)
    by_fraction_class: dict[tuple[float, int], list[float]] = defaultdict(list)
    for row in performance:
        by_fraction_class[(row["fraction"], row["original_class"])].append(row["no_aug_f1"])
    summary = [
        {"fraction": fraction, "original_class": class_id, "mean_no_aug_f1": float(np.mean(values)),
         "cells": len(values)}
        for (fraction, class_id), values in sorted(by_fraction_class.items())
    ]
    result = {
        "status": "FRACTION_COMPARABILITY_AUDITED",
        "nested_checks": nested, "all_nested": all(row["ten_in_quarter"] and row["quarter_in_full"] for row in nested),
        "composition_rows": len(composition), "group_rows": len(groups),
        "no_aug_per_class_rows": len(performance), "no_aug_per_class_summary": summary,
        "interpretation": "source-unit diversity regime; fixed window caps mean this is not proportional window-count subsampling",
    }
    Path("analysis/results/qdiffcl_data_regime_fraction_comparability.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
