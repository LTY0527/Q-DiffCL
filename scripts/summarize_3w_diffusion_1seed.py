from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scripts.run_3w_diffusion_1seed import METHODS


def classify(uniform: dict, r1: dict, gate: dict) -> tuple[str, dict, dict]:
    delta = {
        "macro_f1": r1["macro_f1"] - uniform["macro_f1"],
        "macro_recall": r1["recall_macro"] - uniform["recall_macro"],
        "binary_auprc": r1["auprc_fault_vs_normal"] - uniform["auprc_fault_vs_normal"],
        "multiclass_auprc": r1["auprc_multiclass_macro"] - uniform["auprc_multiclass_macro"],
        "far": r1["far"] - uniform["far"],
        "early_recall": r1["early_recall"] - uniform["early_recall"],
        "mean_detection_delay_seconds": r1["mean_detection_delay_seconds"] - uniform["mean_detection_delay_seconds"],
    }
    for r1_class, uniform_class in zip(r1["per_class"], uniform["per_class"]):
        original = r1_class["original_class"]
        if original != uniform_class["original_class"]:
            raise RuntimeError("per-class metric order mismatch")
        delta[f"class_{original}_recall"] = r1_class["recall"] - uniform_class["recall"]
        delta[f"class_{original}_f1"] = r1_class["f1"] - uniform_class["f1"]
    delay_limit = max(
        uniform["mean_detection_delay_seconds"] * float(gate["maximum_delay_ratio"]),
        uniform["mean_detection_delay_seconds"] + float(gate["maximum_delay_absolute_increase_seconds"]),
    )
    checks = {
        "macro_f1_improved": delta["macro_f1"] > 0,
        "auprc_improved": delta["multiclass_auprc"] > 0,
        "far_not_worse": delta["far"] <= 0,
        "early_recall_not_materially_worse": delta["early_recall"] >= -float(gate["maximum_early_recall_drop"]),
        "delay_not_materially_worse": r1["mean_detection_delay_seconds"] <= delay_limit,
    }
    catastrophic_delay = r1["mean_detection_delay_seconds"] > max(
        uniform["mean_detection_delay_seconds"] * float(gate["catastrophic_delay_ratio"]),
        uniform["mean_detection_delay_seconds"] + float(gate["catastrophic_delay_increase_seconds"]),
    )
    catastrophic = (
        delta["macro_f1"] <= -float(gate["catastrophic_macro_f1_drop"])
        or delta["multiclass_auprc"] <= -float(gate["catastrophic_auprc_drop"])
        or delta["far"] >= float(gate["catastrophic_far_increase"])
        or delta["early_recall"] <= -float(gate["catastrophic_early_recall_drop"])
        or catastrophic_delay
    )
    if all(checks.values()):
        status = "3W_FREQUENCY_SELECTIVE_R1_1SEED_GO"
    elif catastrophic:
        status = "3W_R1_1SEED_HOLD"
    else:
        status = "3W_R1_1SEED_INCONCLUSIVE"
    return status, delta, checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/3w_diffusion_1seed.yaml")
    parser.add_argument("--result", type=Path, default=Path("outputs/3w_diffusion_1seed_seed42/result.json"))
    parser.add_argument("--csv", type=Path, default=Path("docs/3w_diffusion_1seed_comparison.csv"))
    parser.add_argument("--json", type=Path, default=Path("docs/3w_diffusion_1seed_summary.json"))
    args = parser.parse_args()
    import yaml

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = json.loads(args.result.read_text(encoding="utf-8"))
    rows = []
    for method in METHODS:
        metrics = result["methods"][method]["metrics"]
        row = {
            "method": method,
            "macro_f1": metrics["macro_f1"],
            "macro_recall": metrics["recall_macro"],
            "binary_auprc": metrics["auprc_fault_vs_normal"],
            "multiclass_auprc": metrics["auprc_multiclass_macro"],
            "far": metrics["far"],
            "early_recall": metrics["early_recall"],
            "mean_detection_delay_seconds": metrics["mean_detection_delay_seconds"],
        }
        for item in metrics["per_class"]:
            row[f"class_{item['original_class']}_recall"] = item["recall"]
            row[f"class_{item['original_class']}_f1"] = item["f1"]
        rows.append(row)
    status, delta, checks = classify(
        result["methods"][METHODS[1]]["metrics"], result["methods"][METHODS[2]]["metrics"], config["gate"]
    )
    payload = {
        "status": status,
        "seed": result["seed"],
        "canonical_split_index": result["canonical_split_index"],
        "r1_minus_uniform": delta,
        "gate_checks": checks,
        "three_seed_allowed": status == "3W_FREQUENCY_SELECTIVE_R1_1SEED_GO",
    }
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
