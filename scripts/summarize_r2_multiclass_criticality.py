from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import spearmanr

from frequency import mask_jaccard
from scripts.run_3w_diffusion_1seed import METHODS, R2_METHOD
from scripts.summarize_frequency_selective_r1_3seed import _flat, mean_sample_std


CORE = ("macro_f1", "recall_macro", "auprc_fault_vs_normal", "auprc_multiclass_macro", "far", "early_recall", "mean_detection_delay_seconds")


def metric_delta(first: dict, second: dict) -> dict[str, float]:
    result = {name: float(second[name] - first[name]) for name in CORE}
    first_classes = {int(row["original_class"]): row for row in first["per_class"]}
    for row in second["per_class"]:
        original = int(row["original_class"]); baseline = first_classes[original]
        result[f"class_{original}_recall"] = float(row["recall"] - baseline["recall"])
        result[f"class_{original}_f1"] = float(row["f1"] - baseline["f1"])
    return result


def three_w_decision(summary: dict, gate: dict) -> tuple[str, dict]:
    delta = summary["R2-UNIFORM"]; r1 = summary["methods"]["R1"]; r2 = summary["methods"]["R2"]
    recall_limit = r1["class_9_recall_std"] * (1 + float(gate["maximum_class9_std_increase_ratio"]))
    f1_limit = r1["class_9_f1_std"] * (1 + float(gate["maximum_class9_std_increase_ratio"]))
    checks = {
        "macro_f1_mean_gain": delta["mean"]["macro_f1"] >= float(gate["minimum_macro_f1_mean_gain"]),
        "macro_f1_wins": delta["wins"]["macro_f1"] >= int(gate["minimum_macro_f1_wins"]),
        "multiclass_auprc_mean_gain": delta["mean"]["auprc_multiclass_macro"] > float(gate["minimum_multiclass_auprc_mean_gain"]),
        "binary_auprc_mean_preserved": delta["mean"]["auprc_fault_vs_normal"] >= -float(gate["maximum_binary_auprc_mean_drop"]),
        "no_large_binary_auprc_seed_drop": min(delta["by_seed"][seed]["auprc_fault_vs_normal"] for seed in delta["by_seed"]) >= -float(gate["maximum_single_binary_auprc_drop"]),
        "far_mean_preserved": delta["mean"]["far"] <= float(gate["maximum_far_mean_increase"]),
        "class9_recall_std_preserved": r2["class_9_recall_std"] <= recall_limit,
        "class9_f1_std_preserved": r2["class_9_f1_std"] <= f1_limit,
    }
    main = all(value for key, value in checks.items() if not key.startswith("class9_"))
    failed_stability = sum(not checks[key] for key in ("class9_recall_std_preserved", "class9_f1_std_preserved"))
    status = "R2_3W_GO" if all(checks.values()) else "R2_3W_PARTIAL_GO" if main and failed_stability == 1 else "R2_3W_NO_GO"
    return status, {"checks": checks, "class9_recall_std_limit": recall_limit, "class9_f1_std_limit": f1_limit}


def _distribution(value: np.ndarray) -> dict[str, float]:
    return {"min": float(value.min()), "median": float(np.median(value)), "mean": float(value.mean()),
            "p95": float(np.quantile(value, .95)), "max": float(value.max())}


TEP_CORE = ("macro_f1", "auprc", "fault_recall", "far", "early_recall", "mean_delay")


def tep_decision(summary: dict, gate: dict) -> tuple[str, dict]:
    delta = summary["R2-C1"]; by_seed = delta["by_seed"]
    catastrophic = {seed: bool(row["macro_f1"] < -float(gate["catastrophic_macro_f1_drop"])
                                or row["auprc"] < -float(gate["catastrophic_auprc_drop"])
                                or row["far"] > float(gate["catastrophic_far_increase"])
                                or row["early_recall"] < -float(gate["catastrophic_early_recall_drop"]))
                    for seed, row in by_seed.items()}
    checks = {"macro_f1_mean_gain_for_go": delta["mean"]["macro_f1"] >= float(gate["minimum_macro_f1_mean_gain_for_go"]),
              "macro_f1_wins": delta["wins"]["macro_f1"] >= int(gate["minimum_macro_f1_wins_vs_c1"]),
              "auprc_mean_preserved": delta["mean"]["auprc"] >= -float(gate["maximum_mean_auprc_drop"]),
              "recall_mean_preserved": delta["mean"]["fault_recall"] >= -float(gate["maximum_mean_recall_drop"]),
              "far_mean_preserved": delta["mean"]["far"] <= float(gate["maximum_mean_far_increase"]),
              "early_mean_preserved": delta["mean"]["early_recall"] >= -float(gate["maximum_mean_early_recall_drop"]),
              "no_catastrophic_seed": not any(catastrophic.values())}
    preserved = all(value for key, value in checks.items() if key not in {"macro_f1_mean_gain_for_go", "macro_f1_wins"})
    status = "R2_CROSS_DATASET_GO" if all(checks.values()) else "R2_CROSS_DATASET_PARTIAL_GO" if preserved else "R2_CROSS_DATASET_NO_GO"
    return status, {"checks": checks, "catastrophic_by_seed": catastrophic}


def summarize_tep(config: dict, result_path: Path) -> dict:
    stage = config["tep"]; previous = json.loads(Path(stage["previous_result"]).read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8")); metrics = {}; rows = []
    for seed in map(str, stage["seeds"]):
        metrics[seed] = {"C1": _flat(previous["seed_results"][seed]["methods"]["C1"]),
                         "R1": _flat(previous["seed_results"][seed]["methods"]["R1"]),
                         "R2": _flat(result["seed_results"][seed]["methods"]["R2"])}
        for method in ("C1", "R1", "R2"): rows.append({"seed": seed, "method": method, **metrics[seed][method]})
    summary = {method: {key: mean_sample_std([metrics[seed][method][key] for seed in metrics])
                        for key in metrics[next(iter(metrics))][method] if key != "missed_fault_runs"}
               for method in ("C1", "R1", "R2")}
    paired_rows = []
    for name, baseline in (("R2-C1", "C1"), ("R2-R1", "R1")):
        by_seed = {seed: {key: float(metrics[seed]["R2"][key] - metrics[seed][baseline][key]) for key in TEP_CORE}
                   for seed in metrics}
        summary[name] = {"by_seed": by_seed,
                         "mean": {key: float(np.mean([row[key] for row in by_seed.values()])) for key in TEP_CORE},
                         "std": {key: float(np.std([row[key] for row in by_seed.values()], ddof=1)) for key in TEP_CORE},
                         "wins": {key: sum(row[key] > 0 for row in by_seed.values()) for key in TEP_CORE}}
        paired_rows.extend({"comparison": name, "seed": seed, **row} for seed, row in by_seed.items())
    status, gate = tep_decision(summary, stage["gate"])
    payload = {"status": status, "stage": "TEP", "markers": stage["markers"], "new_runs": 3, "reused_runs": 6,
               "criticality_fit_scope": result["criticality_fit_scope"], "summary": summary,
               "mask_audit": result["mask_audit"], **gate, "paper_final_claim_allowed": False}
    docs = config["docs"]; Path(docs["tep_json"]).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    results_csv = Path(docs["tep_json"]).with_name("r2_multiclass_criticality_tep_results.csv")
    paired_csv = Path(docs["tep_json"]).with_name("r2_multiclass_criticality_tep_paired.csv")
    with results_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with paired_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0])); writer.writeheader(); writer.writerows(paired_rows)
    return payload


def summarize_three_w(config: dict, manifest_path: Path) -> dict:
    stage = config["three_w"]; original_manifest = json.loads(Path(stage["original_manifest"]).read_text(encoding="utf-8"))
    r2_manifest = json.loads(manifest_path.read_text(encoding="utf-8")); records = {}; rows = []; reference = None
    for seed in map(int, stage["seeds"]):
        old = json.loads(Path(original_manifest["seed_results"][str(seed)]["result_path"]).read_text(encoding="utf-8"))
        r2 = json.loads(Path(r2_manifest["seed_results"][str(seed)]["result_path"]).read_text(encoding="utf-8"))
        if old["fairness"]["window_refs_sha256"] != r2["fairness"]["window_refs_sha256"]:
            raise RuntimeError("R2 changed frozen 3W window refs")
        if old["fairness"]["initialization_sha256"] != r2["fairness"]["initialization_sha256"]:
            raise RuntimeError("R2 changed same-seed initialization")
        current = r2["fairness"]["critical_soft_mask_sha256"]
        if reference is not None and current != reference: raise RuntimeError("R2 mask changed across seeds")
        reference = current; records[str(seed)] = {"old": old, "r2": r2}
        for label, metrics in (("UNIFORM", old["methods"][METHODS[1]]["metrics"]),
                               ("R1", old["methods"][METHODS[2]]["metrics"]),
                               ("R2", r2["methods"][R2_METHOD]["metrics"])):
            row = {"seed": seed, "method": label, **{name: metrics[name] for name in CORE}}
            for item in metrics["per_class"]: row[f"class_{item['original_class']}_recall"] = item["recall"]; row[f"class_{item['original_class']}_f1"] = item["f1"]
            rows.append(row)
    numeric = [name for name in rows[0] if name not in {"seed", "method"}]; methods = {}
    for method in ("UNIFORM", "R1", "R2"):
        selected = [row for row in rows if row["method"] == method]
        methods[method] = {name: {"mean": float(np.mean([row[name] for row in selected])),
                                  "std": float(np.std([row[name] for row in selected]))} for name in numeric}
        methods[method]["class_9_recall_std"] = methods[method]["class_9_recall"]["std"]
        methods[method]["class_9_f1_std"] = methods[method]["class_9_f1"]["std"]
    comparisons = {}; paired_rows = []
    for name, baseline in (("R2-UNIFORM", "UNIFORM"), ("R2-R1", "R1")):
        by_seed = {}
        for seed in map(str, stage["seeds"]):
            old = records[seed]["old"]; r2 = records[seed]["r2"]["methods"][R2_METHOD]["metrics"]
            base = old["methods"][METHODS[1] if baseline == "UNIFORM" else METHODS[2]]["metrics"]
            by_seed[seed] = metric_delta(base, r2); paired_rows.append({"comparison": name, "seed": seed, **by_seed[seed]})
        comparisons[name] = {"by_seed": by_seed,
                             "mean": {key: float(np.mean([row[key] for row in by_seed.values()])) for key in next(iter(by_seed.values()))},
                             "std": {key: float(np.std([row[key] for row in by_seed.values()])) for key in next(iter(by_seed.values()))},
                             "wins": {key: sum(row[key] > 0 for row in by_seed.values()) for key in next(iter(by_seed.values()))}}
    r1_critical = json.loads(Path(stage["r1_criticality_source"]).read_text(encoding="utf-8"))["criticality"]
    r2_critical = records[str(stage["seeds"][0])]["r2"]["criticality"]
    r1_mask = np.asarray(r1_critical["composite_mask"], dtype=bool); r2_mask = np.asarray(r2_critical["composite_mask"], dtype=bool)
    r1_composite = np.asarray(r1_critical["composite"]); r2_composite = np.asarray(r2_critical["composite"])
    multiclass = np.asarray(r2_critical["multiclass_fisher"])
    mask = {"r1_sha256": hashlib.sha256(np.ascontiguousarray(r1_mask).tobytes()).hexdigest(),
            "r2_sha256": hashlib.sha256(np.ascontiguousarray(r2_mask).tobytes()).hexdigest(),
            "jaccard": mask_jaccard(r1_mask, r2_mask), "changed_bins": int(np.logical_xor(r1_mask, r2_mask).sum()),
            "selected_bins": int(r2_mask.sum()), "composite_spearman": float(spearmanr(r1_composite.ravel(), r2_composite.ravel()).statistic),
            "multiclass_mask_overlap": mask_jaccard(np.asarray(r2_critical["multiclass_mask"], bool), r2_mask),
            "multiclass_changed_ranking": bool(not np.array_equal(r1_mask, r2_mask)),
            "component_weights": r2_critical["component_weights"],
            "components": {"D": _distribution(np.asarray(r2_critical["discriminative"])),
                           "E": _distribution(np.asarray(r2_critical["early"])),
                           "S": _distribution(np.asarray(r2_critical["stability"])), "M": _distribution(multiclass)}}
    summary = {"methods": methods, **comparisons}; status, gate = three_w_decision(summary, stage["gate"])
    payload = {"status": status, "stage": "3W", "seeds": stage["seeds"], "formal_new_runs": 3,
               "invalidated_preliminary_runs": int(stage.get("invalidated_preliminary_runs", 0)),
               "invalidated_reason": stage.get("invalidated_reason"),
               "actual_executed_new_runs": 3 + int(stage.get("invalidated_preliminary_runs", 0)),
               "reused_runs": 6, "criticality_fit_scope": "train only", "summary": summary, "mask_audit": mask, **gate,
               "stage_b_allowed": status in {"R2_3W_GO", "R2_3W_PARTIAL_GO"}}
    docs = config["docs"]; Path(docs["three_w_json"]).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with Path(docs["three_w_results_csv"]).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with Path(docs["three_w_paired_csv"]).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0])); writer.writeheader(); writer.writerows(paired_rows)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/r2_multiclass_criticality.yaml")
    parser.add_argument("--stage", choices=("3w", "tep"), default="3w")
    parser.add_argument("--manifest", type=Path, default=Path("outputs/r2_multiclass_criticality_3w_all_classes/result_manifest.json"))
    parser.add_argument("--tep-result", type=Path, default=Path("outputs/r2_multiclass_criticality_tep/result.json"))
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.stage == "3w": result = summarize_three_w(config, args.manifest)
    else: result = summarize_tep(config, args.tep_result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__": main()
