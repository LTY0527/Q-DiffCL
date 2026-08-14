from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from scripts.run_3w_diffusion_1seed import METHODS
from scripts.summarize_3w_diffusion_1seed import classify


CORE = (
    "macro_f1",
    "recall_macro",
    "auprc_fault_vs_normal",
    "auprc_multiclass_macro",
    "far",
    "early_recall",
    "mean_detection_delay_seconds",
)


def three_seed_decision(seed_records: list[dict], single_gate: dict, three_gate: dict) -> tuple[str, dict]:
    deltas = []
    per_seed_checks = []
    zero_counts: dict[int, int] = {}
    for record in seed_records:
        uniform = record["methods"][METHODS[1]]["metrics"]
        r1 = record["methods"][METHODS[2]]["metrics"]
        _, delta, checks = classify(uniform, r1, single_gate)
        deltas.append(delta)
        per_seed_checks.append(checks)
        for item in r1["per_class"]:
            original = int(item["original_class"])
            zero_counts[original] = zero_counts.get(original, 0) + int(float(item["recall"]) == 0)
    mean = {name: float(np.mean([row[name] for row in deltas])) for name in deltas[0]}
    std = {name: float(np.std([row[name] for row in deltas])) for name in deltas[0]}
    wins = {
        "macro_f1_nonworse": sum(row["macro_f1"] >= 0 for row in deltas),
        "macro_f1_improved": sum(row["macro_f1"] > 0 for row in deltas),
        "far_improved": sum(row["far"] < 0 for row in deltas),
        "binary_auprc_nonworse": sum(row["binary_auprc"] >= 0 for row in deltas),
        "multiclass_auprc_nonworse": sum(row["multiclass_auprc"] >= 0 for row in deltas),
    }
    tolerance = {
        "early_mean_within": mean["early_recall"] >= -float(single_gate["maximum_early_recall_drop"]),
        "early_majority_within": sum(check["early_recall_not_materially_worse"] for check in per_seed_checks) >= 2,
        "delay_majority_within": sum(check["delay_not_materially_worse"] for check in per_seed_checks) >= 2,
    }
    systematic_zero = [original for original, count in zero_counts.items() if count >= 2]
    go_checks = {
        "macro_mean_improved": mean["macro_f1"] > 0,
        "macro_majority_nonworse": wins["macro_f1_nonworse"] >= int(three_gate["minimum_macro_nonworse_seeds"]),
        "far_mean_improved": mean["far"] < 0,
        "far_majority_improved": wins["far_improved"] >= int(three_gate["minimum_far_improved_seeds"]),
        "binary_auprc_mean_nonworse": mean["binary_auprc"] >= 0,
        "multiclass_auprc_mean_nonworse": mean["multiclass_auprc"] >= 0,
        "auprc_majority_nonworse": min(wins["binary_auprc_nonworse"], wins["multiclass_auprc_nonworse"]) >= int(three_gate["minimum_auprc_nonworse_seeds"]),
        **tolerance,
        "no_systematic_zero_recall": not systematic_zero,
    }
    catastrophic = (
        mean["macro_f1"] <= -float(single_gate["catastrophic_macro_f1_drop"])
        or mean["multiclass_auprc"] <= -float(single_gate["catastrophic_auprc_drop"])
        or mean["far"] >= float(single_gate["catastrophic_far_increase"])
        or mean["early_recall"] <= -float(single_gate["catastrophic_early_recall_drop"])
        or bool(systematic_zero)
    )
    if all(go_checks.values()):
        status = "3W_FREQUENCY_SELECTIVE_R1_3SEED_GO"
    elif catastrophic:
        status = "3W_FREQUENCY_SELECTIVE_R1_3SEED_HOLD"
    else:
        status = "3W_FREQUENCY_SELECTIVE_R1_3SEED_INCONCLUSIVE"
    return status, {
        "paired_mean": mean,
        "paired_std": std,
        "wins": wins,
        "zero_recall_seed_counts": zero_counts,
        "systematic_zero_recall_classes": systematic_zero,
        "go_checks": go_checks,
        "per_seed_deltas": deltas,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/3w_diffusion_3seed.yaml")
    parser.add_argument("--manifest", type=Path, default=Path("outputs/3w_diffusion_3seed/result_manifest.json"))
    parser.add_argument("--results-csv", type=Path, default=Path("docs/3w_diffusion_3seed_results.csv"))
    parser.add_argument("--paired-csv", type=Path, default=Path("docs/3w_diffusion_3seed_paired_summary.csv"))
    parser.add_argument("--json", type=Path, default=Path("docs/3w_diffusion_3seed_summary.json"))
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    single = yaml.safe_load(Path(config["base_config"]).read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = []
    reference_fairness = None
    for seed in map(str, manifest["seeds"]):
        record = json.loads(Path(manifest["seed_results"][seed]["result_path"]).read_text(encoding="utf-8"))
        fairness = record["fairness"]
        current = (fairness["window_refs_sha256"], fairness["critical_soft_mask_sha256"])
        if reference_fairness is not None and current != reference_fairness:
            raise RuntimeError("3-seed runs changed window refs or critical frequency mask")
        reference_fairness = current
        records.append(record)
    rows = []
    for record in records:
        for method in METHODS:
            metrics = record["methods"][method]["metrics"]
            row = {"seed": record["seed"], "method": method, **{name: metrics[name] for name in CORE}}
            for item in metrics["per_class"]:
                row[f"class_{item['original_class']}_recall"] = item["recall"]
                row[f"class_{item['original_class']}_f1"] = item["f1"]
            rows.append(row)
    status, summary = three_seed_decision(records, single["gate"], config["gate"])
    method_summary = {}
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        numeric = [name for name in method_rows[0] if name not in {"seed", "method"}]
        method_summary[method] = {
            name: {
                "mean": float(np.mean([float(row[name]) for row in method_rows])),
                "std": float(np.std([float(row[name]) for row in method_rows])),
            }
            for name in numeric
        }
    paired_rows = []
    for record, delta in zip(records, summary["per_seed_deltas"]):
        paired_rows.append({"seed": record["seed"], "statistic": "seed_delta", **delta})
    paired_rows.append({"seed": "ALL", "statistic": "mean", **summary["paired_mean"]})
    paired_rows.append({"seed": "ALL", "statistic": "std", **summary["paired_std"]})
    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.results_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with args.paired_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader(); writer.writerows(paired_rows)
    payload = {"status": status, "seeds": manifest["seeds"], "method_summary": method_summary, **summary}
    args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
