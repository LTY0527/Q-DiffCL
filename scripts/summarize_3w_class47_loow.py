from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def summarize(output: Path):
    manifest = json.loads((output / "fold_manifest.json").read_text(encoding="utf-8")); folds = {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in sorted((output / "folds").glob("WELL-*.json"))}
    missing = set(manifest["unique_folds"]) - set(folds)
    if missing: raise RuntimeError(f"incomplete LOOW folds: {sorted(missing)}")
    rows, summary = [], {}
    for class_text, wells in manifest["target_wells"].items():
        recalls = []
        for well in wells:
            fold = folds[well]; target = fold["target_records"][class_text]; metrics = fold["metrics"]
            row = {"target_class": int(class_text), "held_out_well": well, "target_recall": target["recall"], "target_f1": target["f1"],
                   "target_support": target["support"], "macro_f1": metrics["macro_f1"], "far": metrics["far"],
                   "early_recall": metrics["early_recall"], "mean_detection_delay_seconds": metrics["mean_detection_delay_seconds"], "detection_instance_rate": metrics["detected_instance_rate"]}
            rows.append(row); recalls.append(float(target["recall"]))
        positive = sum(value > 0 for value in recalls); classification = "SYSTEMATIC_CROSS_WELL_FAILURE" if positive <= len(recalls) / 2 else "SPLIT_SPECIFIC_FAILURE"
        summary[class_text] = {"held_out_wells": len(recalls), "recall_mean": float(np.mean(recalls)), "recall_median": float(np.median(recalls)), "recall_std": float(np.std(recalls)),
                               "zero_recall_wells": int(sum(value == 0 for value in recalls)), "positive_recall_wells": positive,
                               "best_recall": float(max(recalls)), "worst_recall": float(min(recalls)), "classification": classification}
    return rows, summary


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, default=Path("outputs/3w_class47_loow_seed42")); parser.add_argument("--csv", type=Path, default=Path("docs/3w_class47_loow_results.csv")); parser.add_argument("--json", type=Path, default=Path("docs/3w_final_protocol.json")); args = parser.parse_args()
    rows, summary = summarize(args.output_dir); secondary = [1, 3, 5, 6]; primary = [0, 2, 8, 9]
    for class_text, item in summary.items():
        (secondary if item["classification"] == "SYSTEMATIC_CROSS_WELL_FAILURE" else primary).append(int(class_text))
    primary, secondary = sorted(set(primary)), sorted(set(secondary)); status = "3W_FINAL_PRIMARY_PROTOCOL_GO" if all(item["classification"] != "SYSTEMATIC_CROSS_WELL_FAILURE" for item in summary.values()) else "3W_REAL_ONLY_GENERALIZATION_HOLD"
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    payload = {"status": status, "base_commit": "b30d7c8", "seed": 42, "input": "PROCESS_ONLY", "class_summary": summary,
               "FINAL_3W_PRIMARY_CLASSES": primary, "FINAL_3W_SECONDARY_CLASSES": secondary,
               "diffusion_allowed": status == "3W_FINAL_PRIMARY_PROTOCOL_GO", "note": "Protocol frozen from full class 4/7 LOOW; no diffusion was run."}
    args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__": main()
