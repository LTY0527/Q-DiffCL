from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = (
    "macro_f1",
    "recall_macro",
    "auprc_fault_vs_normal",
    "auprc_multiclass_macro",
    "far",
    "early_recall",
    "mean_detection_delay_seconds",
)


def build_summary(rows: list[dict], classes: list[int]) -> dict:
    zero_splits = {
        str(original): sum(float(row[f"class_{original}_recall"]) == 0 for row in rows) for original in classes
    }
    stable = all(count == 0 for count in zero_splits.values())
    return {
        "status": "3W_FINAL_PRIMARY_STABILITY_GO" if stable else "3W_FINAL_PRIMARY_STABILITY_HOLD",
        "primary_classes": classes,
        "split_count": len(rows),
        "gate": "every Primary class must have Recall > 0 in every grouped test split",
        "zero_recall_split_counts": zero_splits,
        "metric_summary": {
            name: {
                "mean": float(np.mean([float(row[name]) for row in rows])),
                "std": float(np.std([float(row[name]) for row in rows])),
            }
            for name in METRICS
        },
        "diffusion_allowed": stable,
    }


def summarize(output: Path) -> tuple[list[dict], dict]:
    manifest = json.loads((output / "grouped_split_manifest.json").read_text(encoding="utf-8"))
    results = []
    for item in manifest["splits"]:
        path = output / f"split_{int(item['split_index']):02d}" / "result.json"
        if not path.exists():
            raise RuntimeError(f"incomplete grouped split: {path}")
        results.append(json.loads(path.read_text(encoding="utf-8")))
    rows = []
    for result in results:
        metrics = result["metrics"]
        row = {"split_index": result["split_index"], "split_seed": result["split_seed"]}
        row.update({name: metrics[name] for name in METRICS})
        for item in metrics["per_class"]:
            original = item["original_class"]
            row[f"class_{original}_recall"] = item["recall"]
            row[f"class_{original}_f1"] = item["f1"]
        rows.append(row)
    return rows, build_summary(rows, manifest["primary_classes"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/3w_final_primary_grouped_seed42"))
    parser.add_argument("--csv", type=Path, default=Path("docs/3w_final_primary_grouped_results.csv"))
    parser.add_argument("--json", type=Path, default=Path("docs/3w_final_primary_stability.json"))
    args = parser.parse_args()
    rows, summary = summarize(args.output_dir)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
