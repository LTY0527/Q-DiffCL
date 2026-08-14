from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from scripts.run_3w_diffusion_1seed import METHODS
from scripts.summarize_3w_diffusion_1seed import classify


CORE = ("macro_f1", "recall_macro", "auprc_fault_vs_normal", "auprc_multiclass_macro", "far", "early_recall", "mean_detection_delay_seconds")


def stability_decision(summary: dict, gate: dict) -> tuple[str, dict]:
    original, balanced, cross = summary["ORIGINAL"], summary["BALANCED"], summary["CROSS_WELL"]
    recall_reduction = 1 - cross["r1_class9_recall_std"] / max(original["r1_class9_recall_std"], 1e-12)
    f1_reduction = 1 - cross["r1_class9_f1_std"] / max(original["r1_class9_f1_std"], 1e-12)
    baseline_ratio = max(original["paired_view_cross_well_positive_ratio"], balanced["paired_view_cross_well_positive_ratio"])
    delay_limit = max(original["r1_metric_mean"]["mean_detection_delay_seconds"] * float(gate["maximum_delay_ratio"]),
                      original["r1_metric_mean"]["mean_detection_delay_seconds"] + float(gate["maximum_delay_absolute_increase_seconds"]))
    checks = {
        "paired_macro_majority_improved": cross["paired_macro_improved_seeds"] >= int(gate["minimum_macro_improved_seeds"]),
        "no_large_seed_binary_auprc_drop": min(cross["paired_binary_auprc_by_seed"].values()) >= -float(gate["maximum_any_seed_binary_auprc_drop"]),
        "class9_recall_std_reduced": recall_reduction >= float(gate["minimum_class9_std_reduction_ratio"]),
        "class9_f1_std_reduced": f1_reduction >= float(gate["minimum_class9_std_reduction_ratio"]),
        "cross_well_ratio_increased": cross["paired_view_cross_well_positive_ratio"] >= baseline_ratio + float(gate["minimum_cross_well_ratio_increase"]),
        "macro_mean_preserved": cross["r1_metric_mean"]["macro_f1"] >= original["r1_metric_mean"]["macro_f1"] - float(gate["maximum_macro_mean_drop"]),
        "binary_auprc_mean_preserved": cross["r1_metric_mean"]["auprc_fault_vs_normal"] >= original["r1_metric_mean"]["auprc_fault_vs_normal"] - float(gate["maximum_binary_auprc_mean_drop"]),
        "far_mean_preserved": cross["r1_metric_mean"]["far"] <= original["r1_metric_mean"]["far"] + float(gate["maximum_far_mean_increase"]),
        "early_mean_preserved": cross["r1_metric_mean"]["early_recall"] >= original["r1_metric_mean"]["early_recall"] - float(gate["maximum_early_mean_drop"]),
        "delay_mean_preserved": cross["r1_metric_mean"]["mean_detection_delay_seconds"] <= delay_limit,
        "finite_training": bool(cross["finite_training"]), "all_positive_pairs": bool(cross["all_positive_pairs"]),
        "all_classes_have_cross_well_support": bool(cross["all_classes_have_cross_well_support"]),
    }
    catastrophic = (not checks["finite_training"] or not checks["all_positive_pairs"]
                    or not checks["all_classes_have_cross_well_support"] or not checks["macro_mean_preserved"]
                    or not checks["far_mean_preserved"])
    if all(checks.values()): status = "CROSS_WELL_SUPCON_GO"
    elif not catastrophic and checks["cross_well_ratio_increased"] and (checks["class9_recall_std_reduced"] or checks["class9_f1_std_reduced"]):
        status = "CROSS_WELL_SUPCON_PARTIAL_GO"
    else: status = "CROSS_WELL_SUPCON_NO_GO"
    return status, {"checks": checks, "baseline_paired_view_cross_well_positive_ratio": baseline_ratio,
                    "class9_recall_std_reduction": recall_reduction, "class9_f1_std_reduction": f1_reduction}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/3w_cross_well_supcon.yaml")
    parser.add_argument("--cross-manifest", type=Path, default=Path("outputs/3w_cross_well_supcon/result_manifest.json"))
    parser.add_argument("--results-csv", type=Path, default=Path("docs/3w_cross_well_supcon_results.csv"))
    parser.add_argument("--paired-csv", type=Path, default=Path("docs/3w_cross_well_supcon_paired.csv"))
    parser.add_argument("--json", type=Path, default=Path("docs/3w_cross_well_supcon.json"))
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base = yaml.safe_load(Path(config["base_config"]).read_text(encoding="utf-8"))
    manifests = {"ORIGINAL": json.loads(Path(config["original_manifest"]).read_text(encoding="utf-8")),
                 "BALANCED": json.loads(Path(config["balanced_manifest"]).read_text(encoding="utf-8")),
                 "CROSS_WELL": json.loads(args.cross_manifest.read_text(encoding="utf-8"))}
    comparison_by_seed = {
        seed: json.loads(Path(manifests["CROSS_WELL"]["seed_results"][str(seed)]["result_path"]).read_text(encoding="utf-8"))[
            "supcon_batching_comparison"
        ]
        for seed in map(int, config["seeds"])
    }
    records = {name: {} for name in manifests}; reference = None; rows = []
    for batching, manifest in manifests.items():
        for seed in map(int, config["seeds"]):
            result = json.loads(Path(manifest["seed_results"][str(seed)]["result_path"]).read_text(encoding="utf-8"))
            current = (result["fairness"]["window_refs_sha256"], result["fairness"]["critical_soft_mask_sha256"])
            if reference is not None and current != reference: raise RuntimeError("Cross-WELL audit changed window refs or critical mask")
            reference = current; records[batching][seed] = result
            for method in config["methods"]:
                metrics = result["methods"][method]["metrics"]
                batch_hash = comparison_by_seed[seed][batching]["batch_order_sha256"]
                row = {"batching": batching, "seed": seed, "method": method, **{name: metrics[name] for name in CORE},
                       "initialization_sha256": result["methods"][method]["initialization_sha256"], "batch_order_sha256": batch_hash,
                       "window_refs_sha256": current[0], "critical_mask_sha256": current[1]}
                for item in metrics["per_class"]: row[f"class_{item['original_class']}_recall"] = item["recall"]; row[f"class_{item['original_class']}_f1"] = item["f1"]
                rows.append(row)
    paired_rows = []; summary = {}
    for batching in manifests:
        deltas = []; r1_metrics = []; class9_recall = []; class9_f1 = []; finite = True; positive = True
        ratios = []; clean_ratios = []; support = True; class_wells = None; per_class_ratios = None
        for seed in map(int, config["seeds"]):
            result = records[batching][seed]; uniform = result["methods"][METHODS[1]]["metrics"]; r1 = result["methods"][METHODS[2]]["metrics"]
            _, delta, _ = classify(uniform, r1, base["gate"]); deltas.append(delta); paired_rows.append({"batching": batching, "seed": seed, **delta})
            r1_metrics.append(r1); row9 = next(row for row in r1["per_class"] if row["original_class"] == 9)
            class9_recall.append(row9["recall"]); class9_f1.append(row9["f1"])
            cross_audit = records["CROSS_WELL"][seed]["supcon_batching_comparison"][batching]
            ratios.append(cross_audit["paired_view_cross_well_positive_ratio"]); clean_ratios.append(cross_audit["clean_cross_well_positive_ratio"])
            support &= not cross_audit["classes_without_cross_well_support"]; class_wells = cross_audit["class_well_counts"]
            per_class_ratios = cross_audit["per_class_cross_well_positive_ratio"]
            if batching == "CROSS_WELL":
                history = [row for method in config["methods"] for row in result["methods"][method]["pretrain_history"]]
                finite &= all(np.isfinite(row["loss"]) and np.isfinite(row["validation_supcon_loss"]) for row in history)
                positive &= bool(result["supcon_sampler"]["all_classes_retain_positive_pairs"])
        summary[batching] = {"paired_mean": {name: float(np.mean([row[name] for row in deltas])) for name in deltas[0]},
                             "paired_std": {name: float(np.std([row[name] for row in deltas])) for name in deltas[0]},
                             "paired_macro_improved_seeds": sum(row["macro_f1"] > 0 for row in deltas),
                             "paired_binary_auprc_by_seed": {str(seed): deltas[i]["binary_auprc"] for i, seed in enumerate(config["seeds"])},
                             "r1_class9_recall_mean": float(np.mean(class9_recall)), "r1_class9_recall_std": float(np.std(class9_recall)),
                             "r1_class9_f1_mean": float(np.mean(class9_f1)), "r1_class9_f1_std": float(np.std(class9_f1)),
                             "r1_metric_mean": {name: float(np.mean([row[name] for row in r1_metrics])) for name in CORE},
                             "r1_metric_std": {name: float(np.std([row[name] for row in r1_metrics])) for name in CORE},
                             "clean_cross_well_positive_ratio": float(np.mean(clean_ratios)),
                             "paired_view_cross_well_positive_ratio": float(np.mean(ratios)),
                             "class_well_counts": class_wells, "per_class_cross_well_positive_ratio": per_class_ratios,
                             "all_classes_have_cross_well_support": support, "finite_training": finite, "all_positive_pairs": positive}
    status, decision = stability_decision(summary, config["gate"])
    sampler = {str(seed): records["CROSS_WELL"][seed]["supcon_sampler"] for seed in map(int, config["seeds"])}
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
