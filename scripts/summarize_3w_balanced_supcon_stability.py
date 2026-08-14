from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from scripts.run_3w_diffusion_1seed import METHODS, supcon_orders
from scripts.summarize_3w_diffusion_1seed import classify


CORE = ("macro_f1", "recall_macro", "auprc_fault_vs_normal", "auprc_multiclass_macro", "far", "early_recall", "mean_detection_delay_seconds")


def stability_decision(summary: dict, gate: dict) -> tuple[str, dict]:
    original, balanced = summary["ORIGINAL"], summary["BALANCED"]
    recall_reduction = 1 - balanced["r1_class9_recall_std"] / max(original["r1_class9_recall_std"], 1e-12)
    f1_reduction = 1 - balanced["r1_class9_f1_std"] / max(original["r1_class9_f1_std"], 1e-12)
    delay_limit = max(original["r1_metric_mean"]["mean_detection_delay_seconds"] * float(gate["maximum_delay_ratio"]),
                      original["r1_metric_mean"]["mean_detection_delay_seconds"] + float(gate["maximum_delay_absolute_increase_seconds"]))
    checks = {
        "paired_macro_majority_improved": balanced["paired_macro_improved_seeds"] >= int(gate["minimum_macro_improved_seeds"]),
        "seed44_binary_anomaly_removed": balanced["seed44_binary_auprc_delta"] >= -float(gate["maximum_seed44_binary_auprc_drop"]),
        "class9_recall_std_reduced": recall_reduction >= float(gate["minimum_class9_std_reduction_ratio"]),
        "class9_f1_std_reduced": f1_reduction >= float(gate["minimum_class9_std_reduction_ratio"]),
        "macro_mean_preserved": balanced["r1_metric_mean"]["macro_f1"] >= original["r1_metric_mean"]["macro_f1"] - float(gate["maximum_macro_mean_drop"]),
        "binary_auprc_mean_preserved": balanced["r1_metric_mean"]["auprc_fault_vs_normal"] >= original["r1_metric_mean"]["auprc_fault_vs_normal"] - float(gate["maximum_binary_auprc_mean_drop"]),
        "far_mean_preserved": balanced["r1_metric_mean"]["far"] <= original["r1_metric_mean"]["far"] + float(gate["maximum_far_mean_increase"]),
        "early_mean_preserved": balanced["r1_metric_mean"]["early_recall"] >= original["r1_metric_mean"]["early_recall"] - float(gate["maximum_early_mean_drop"]),
        "delay_mean_preserved": balanced["r1_metric_mean"]["mean_detection_delay_seconds"] <= delay_limit,
        "finite_training": bool(balanced["finite_training"]),
        "all_positive_pairs": bool(balanced["all_positive_pairs"]),
    }
    catastrophic = not checks["finite_training"] or not checks["all_positive_pairs"] or (
        balanced["r1_metric_mean"]["macro_f1"] < original["r1_metric_mean"]["macro_f1"] - float(gate["maximum_macro_mean_drop"])
        or balanced["r1_metric_mean"]["far"] > original["r1_metric_mean"]["far"] + float(gate["maximum_far_mean_increase"])
    )
    if all(checks.values()): status = "BALANCED_SUPCON_STABILITY_GO"
    elif not catastrophic and (checks["seed44_binary_anomaly_removed"] or checks["class9_recall_std_reduced"] or checks["class9_f1_std_reduced"]):
        status = "BALANCED_SUPCON_PARTIAL_GO"
    else: status = "BALANCED_SUPCON_NO_GO"
    return status, {"checks": checks, "class9_recall_std_reduction": recall_reduction, "class9_f1_std_reduction": f1_reduction}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/3w_balanced_supcon_stability.yaml")
    parser.add_argument("--balanced-manifest", type=Path, default=Path("outputs/3w_balanced_supcon_stability/result_manifest.json"))
    parser.add_argument("--results-csv", type=Path, default=Path("docs/3w_balanced_supcon_stability_results.csv"))
    parser.add_argument("--paired-csv", type=Path, default=Path("docs/3w_balanced_supcon_stability_paired.csv"))
    parser.add_argument("--json", type=Path, default=Path("docs/3w_balanced_supcon_stability.json"))
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base = yaml.safe_load(Path(config["base_config"]).read_text(encoding="utf-8")); original_manifest = json.loads(Path(config["original_manifest"]).read_text(encoding="utf-8"))
    balanced_manifest = json.loads(args.balanced_manifest.read_text(encoding="utf-8")); records = {"ORIGINAL": {}, "BALANCED": {}}
    reference = None; rows = []
    for batching, manifest in (("ORIGINAL", original_manifest), ("BALANCED", balanced_manifest)):
        for seed in map(int, config["seeds"]):
            result = json.loads(Path(manifest["seed_results"][str(seed)]["result_path"]).read_text(encoding="utf-8"))
            current = (result["fairness"]["window_refs_sha256"], result["fairness"]["critical_soft_mask_sha256"])
            if reference is not None and current != reference: raise RuntimeError("balanced audit changed window refs or critical mask")
            reference = current; records[batching][seed] = result
            if batching == "ORIGINAL":
                labels = np.repeat(np.arange(4), 4000); _, audit = supcon_orders(labels, {**base["training"], "supcon_batching": "original"}, seed)
            else: audit = result["supcon_sampler"]
            for method in config["methods"]:
                metrics = result["methods"][method]["metrics"]
                row = {"batching": batching, "seed": seed, "method": method, **{name: metrics[name] for name in CORE},
                       "initialization_sha256": result["methods"][method]["initialization_sha256"],
                       "batch_order_sha256": audit["batch_order_sha256"], "window_refs_sha256": current[0], "critical_mask_sha256": current[1]}
                for item in metrics["per_class"]: row[f"class_{item['original_class']}_recall"] = item["recall"]; row[f"class_{item['original_class']}_f1"] = item["f1"]
                rows.append(row)
    paired_rows = []; summary = {}
    for batching in ("ORIGINAL", "BALANCED"):
        deltas = []; r1_metrics = []; class9_recall = []; class9_f1 = []; finite = True; positive = True
        for seed in map(int, config["seeds"]):
            result = records[batching][seed]; uniform = result["methods"][METHODS[1]]["metrics"]; r1 = result["methods"][METHODS[2]]["metrics"]
            _, delta, _ = classify(uniform, r1, base["gate"]); deltas.append(delta); paired_rows.append({"batching": batching, "seed": seed, **delta})
            r1_metrics.append(r1); row9 = next(row for row in r1["per_class"] if row["original_class"] == 9); class9_recall.append(row9["recall"]); class9_f1.append(row9["f1"])
            if batching == "BALANCED":
                finite &= all(np.isfinite(row["loss"]) and np.isfinite(row["validation_supcon_loss"]) for method in config["methods"] for row in result["methods"][method]["pretrain_history"])
                positive &= bool(result["supcon_sampler"]["all_classes_retain_positive_pairs"])
        summary[batching] = {"paired_mean": {name: float(np.mean([row[name] for row in deltas])) for name in deltas[0]},
                             "paired_std": {name: float(np.std([row[name] for row in deltas])) for name in deltas[0]},
                             "paired_macro_improved_seeds": sum(row["macro_f1"] > 0 for row in deltas),
                             "seed44_binary_auprc_delta": deltas[2]["binary_auprc"],
                             "r1_class9_recall_mean": float(np.mean(class9_recall)), "r1_class9_recall_std": float(np.std(class9_recall)),
                             "r1_class9_f1_mean": float(np.mean(class9_f1)), "r1_class9_f1_std": float(np.std(class9_f1)),
                             "r1_metric_mean": {name: float(np.mean([row[name] for row in r1_metrics])) for name in CORE},
                             "r1_metric_std": {name: float(np.std([row[name] for row in r1_metrics])) for name in CORE},
                             "finite_training": finite, "all_positive_pairs": positive}
    status, decision = stability_decision(summary, config["gate"])
    sampler = {str(seed): records["BALANCED"][seed]["supcon_sampler"] for seed in map(int, config["seeds"])}
    payload = {"status": status, "summary": summary, **decision, "sampler": sampler,
               "window_refs_sha256": reference[0], "critical_mask_sha256": reference[1]}
    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.results_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with args.paired_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0])); writer.writeheader(); writer.writerows(paired_rows)
    args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": status, "summary": summary, **decision}, ensure_ascii=False))


if __name__ == "__main__": main()
