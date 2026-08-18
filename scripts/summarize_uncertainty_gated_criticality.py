from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from scripts.run_3w_diffusion_1seed import METHODS as THREE_W_METHODS, UG_R1_METHOD
from scripts.summarize_hierarchical_fault_semantic_criticality import _distribution


def _paired(rows: dict[str, dict[str, float]]) -> dict:
    result = {"by_seed": rows}
    for key in next(iter(rows.values())):
        values = [row[key] for row in rows.values()]
        result[key] = {**_distribution(values), "nonnegative_seeds": sum(value >= 0 for value in values),
                       "positive_seeds": sum(value > 0 for value in values)}
    return result


def _hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _quantiles(value: np.ndarray) -> dict[str, float]:
    q = np.quantile(value, [0, .1, .25, .5, .75, .9, 1])
    return dict(zip(("min", "p10", "p25", "median", "p75", "p90", "max"), map(float, q)))


def _mechanism(criticality: dict, diagnostic: dict, uniform_budget: float) -> dict:
    if "r1" in criticality:
        r1 = criticality["r1"]; composite = np.asarray(r1["composite"], float)
        hard = np.asarray(r1["composite_mask"], bool); soft = np.asarray(r1["soft_mask"], float)
    else:
        composite = np.asarray(criticality["r1_composite"], float)
        hard = np.asarray(criticality["r1_hard_mask"], bool); soft = np.asarray(criticality["r1_soft_mask"], float)
    probability = np.asarray(criticality["selection_probability"], float)
    confidence = np.asarray(criticality["assignment_confidence"], float)
    r1_timestep = 1 + (1 - soft) * 4; ug_timestep = 3 + confidence * (r1_timestep - 3)
    uncertain_indices = np.argsort(np.abs(probability.reshape(-1) - .5))[:20]
    uncertain = [{"channel": int(np.unravel_index(index, probability.shape)[0]),
                  "frequency_bin": int(np.unravel_index(index, probability.shape)[1]),
                  "selection_probability": float(probability.reshape(-1)[index]),
                  "assignment_confidence": float(confidence.reshape(-1)[index]),
                  "r1_timestep": float(r1_timestep.reshape(-1)[index]),
                  "ug_timestep": float(ug_timestep.reshape(-1)[index])} for index in uncertain_indices]
    changed = np.abs(ug_timestep - r1_timestep) > 1e-6
    r1_assignment = np.sign(r1_timestep - 3); ug_assignment = np.sign(ug_timestep - 3)
    effective_union = (r1_assignment != 0) | (ug_assignment != 0)
    agreement = float(np.mean(r1_assignment[effective_union] == ug_assignment[effective_union])) if effective_union.any() else 1.0
    return {"r1_composite_sha256": _hash(composite), "r1_hard_mask_sha256": _hash(hard),
            "r1_composite_map": composite.tolist(), "r1_hard_mask": hard.astype(int).tolist(),
            "r1_soft_mask": soft.tolist(), "selection_probability_map": probability.tolist(),
            "assignment_confidence_map": confidence.tolist(),
            "r1_timestep_map": r1_timestep.tolist(), "ug_timestep_map": ug_timestep.tolist(),
            "selection_probability": _quantiles(probability), "assignment_confidence": _quantiles(confidence),
            "confidence_mean": float(confidence.mean()), "confidence_below_025_fraction": float(np.mean(confidence < .25)),
            "confidence_above_075_fraction": float(np.mean(confidence >= .75)),
            "changed_bins": int(changed.sum()), "unchanged_bins": int((~changed).sum()),
            "timestep_mean_absolute_change": float(np.mean(np.abs(ug_timestep - r1_timestep))),
            "r1_timestep": _quantiles(r1_timestep), "ug_timestep": _quantiles(ug_timestep),
            "effective_assignment_agreement": agreement, "most_uncertain_bins": uncertain,
            "bootstrap_overlap": _distribution(list(map(float, criticality["bootstrap_overlap"]))),
            "bootstrap_unit_count": int(criticality["bootstrap_unit_count"]),
            "stratified_unit_counts": criticality["stratified_unit_counts"],
            "bootstrap_scope": criticality["bootstrap_scope"],
            "noise_budget": {"uniform": uniform_budget, "ug_total": diagnostic["expected_total_noise_budget"],
                             "absolute_error": abs(diagnostic["expected_total_noise_budget"] - uniform_budget),
                             "critical": diagnostic.get("critical_noise_budget"),
                             "noncritical": diagnostic.get("noncritical_noise_budget"),
                             "uncertain": diagnostic.get("uncertain_noise_budget"),
                             "stable_critical": diagnostic.get("stable_critical_noise_budget"),
                             "stable_noncritical": diagnostic.get("stable_noncritical_noise_budget")}}


def _three_metrics(metrics: dict) -> dict[str, float]:
    result = {key: float(metrics[key]) for key in ("macro_f1", "recall_macro", "auprc_fault_vs_normal",
              "auprc_multiclass_macro", "far", "early_recall", "mean_detection_delay_seconds")}
    for item in metrics["per_class"]:
        if int(item["original_class"]) in (2, 8, 9):
            result[f"class_{item['original_class']}_recall"] = float(item["recall"])
            result[f"class_{item['original_class']}_f1"] = float(item["f1"])
    return result


def summarize_three_w(config: dict) -> tuple[dict, list[dict]]:
    stage = config["three_w"]; old_manifest = json.loads(Path(stage["existing_manifest"]).read_text(encoding="utf-8"))
    new_manifest = json.loads((Path(stage["output_dir"]) / "result_manifest.json").read_text(encoding="utf-8"))
    rows = []; records = {}; paired_rows = []
    for seed in map(str, stage["seeds"]):
        old = json.loads(Path(old_manifest["seed_results"][seed]["result_path"]).read_text(encoding="utf-8"))
        new = json.loads(Path(new_manifest["seed_results"][seed]["result_path"]).read_text(encoding="utf-8"))
        for key in ("window_refs_sha256", "initialization_sha256", "critical_soft_mask_sha256"):
            if old["fairness"][key] != new["fairness"][key]: raise RuntimeError(f"3W UG-R1 fairness differs: {seed}/{key}")
        if not old["fairness"]["same_pretrain_batch_order"] or not new["fairness"]["same_pretrain_batch_order"]:
            raise RuntimeError(f"3W pretrain order fairness flag failed: {seed}")
        records[seed] = {"old": old, "new": new}
        for method, metrics in (("UNIFORM", old["methods"][THREE_W_METHODS[1]]["metrics"]),
                                ("R1", old["methods"][THREE_W_METHODS[2]]["metrics"]),
                                ("UG_R1", new["methods"][UG_R1_METHOD]["metrics"])):
            rows.append({"seed": seed, "method": method, **_three_metrics(metrics)})
    method_stats = {method: {key: _distribution([row[key] for row in rows if row["method"] == method])
                             for key in rows[0] if key not in {"seed", "method"}}
                    for method in ("UNIFORM", "R1", "UG_R1")}
    comparisons = {}
    for baseline in ("R1", "UNIFORM"):
        by_seed = {}
        for seed in map(str, stage["seeds"]):
            current = next(row for row in rows if row["seed"] == seed and row["method"] == "UG_R1")
            base = next(row for row in rows if row["seed"] == seed and row["method"] == baseline)
            by_seed[seed] = {key: current[key] - base[key] for key in current if key not in {"seed", "method"}}
            paired_rows.append({"dataset": "3W", "comparison": f"UG_R1-{baseline}", "seed": seed, **by_seed[seed]})
        comparisons[f"UG_R1-{baseline}"] = _paired(by_seed)
    delta = comparisons["UG_R1-R1"]; r1 = method_stats["R1"]; ug = method_stats["UG_R1"]
    catastrophic = {seed: bool(row["macro_f1"] < -.03 or row["far"] > .05)
                    for seed, row in delta["by_seed"].items()}
    stability = (ug["macro_f1"]["std"] <= .9 * r1["macro_f1"]["std"]
                 or ug["far"]["std"] <= .9 * r1["far"]["std"])
    checks = {"mean_macro_f1_preserved": delta["macro_f1"]["mean"] >= -.005,
              "macro_f1_at_least_2of3_nonnegative": delta["macro_f1"]["nonnegative_seeds"] >= 2,
              "mean_far_preserved": delta["far"]["mean"] <= .01, "stability_improved_10pct": stability,
              "class9_recall_std_not_worse_10pct": ug["class_9_recall"]["std"] <= 1.1 * r1["class_9_recall"]["std"],
              "class9_f1_std_not_worse_10pct": ug["class_9_f1"]["std"] <= 1.1 * r1["class_9_f1"]["std"],
              "no_catastrophic_seed": not any(catastrophic.values())}
    first = records[str(stage["seeds"][0])]["new"]
    diag = first["augmentation_diagnostics"][UG_R1_METHOD]["train"]
    uniform_budget = records[str(stage["seeds"][0])]["old"]["augmentation_diagnostics"][THREE_W_METHODS[1]]["train"]["expected_total_noise_budget"]
    payload = {"stage": "3W", "supported": all(checks.values()), "checks": checks, "catastrophic_by_seed": catastrophic,
               "methods": method_stats, "comparisons": comparisons,
               "mechanism_audit": _mechanism(first["criticality"], diag, uniform_budget),
               "new_runs": 3, "reused_runs": 6, "paper_final_claim_allowed": False}
    return payload, paired_rows


def _tep_metrics(record: dict) -> dict[str, float]:
    test = record["test"]; metrics = test["metrics"]; delay = test["detection_delay"]
    return {"macro_f1": float(metrics["macro_f1"]), "binary_auprc": float(metrics["auprc"]),
            "fault_recall": float(metrics["fault_recall"]), "far": float(metrics["far"]),
            "early_recall": float(test["early_fault"]["recall"]), "detection_delay": float(delay["mean_delay_samples"]),
            "detected_rate": float(delay["detection_rate"]), "missed_runs": int(delay["missed_runs"])}


def summarize_tep(config: dict) -> tuple[dict, list[dict]]:
    stage = config["tep"]; old = json.loads(Path(stage["existing_result"]).read_text(encoding="utf-8"))
    new = json.loads((Path(stage["output_dir"]) / "result.json").read_text(encoding="utf-8")); rows = []; paired_rows = []
    for seed in map(str, stage["seeds"]):
        for method, record in (("UNIFORM", old["seed_results"][seed]["methods"]["C1"]),
                               ("R1", old["seed_results"][seed]["methods"]["R1"]),
                               ("UG_R1", new["seed_results"][seed]["method"])):
            rows.append({"seed": seed, "method": method, **_tep_metrics(record)})
    method_stats = {method: {key: _distribution([row[key] for row in rows if row["method"] == method])
                             for key in rows[0] if key not in {"seed", "method"}}
                    for method in ("UNIFORM", "R1", "UG_R1")}
    comparisons = {}
    for baseline in ("R1", "UNIFORM"):
        by_seed = {}
        for seed in map(str, stage["seeds"]):
            current = next(row for row in rows if row["seed"] == seed and row["method"] == "UG_R1")
            base = next(row for row in rows if row["seed"] == seed and row["method"] == baseline)
            by_seed[seed] = {key: current[key] - base[key] for key in current if key not in {"seed", "method"}}
            paired_rows.append({"dataset": "TEP", "comparison": f"UG_R1-{baseline}", "seed": seed, **by_seed[seed]})
        comparisons[f"UG_R1-{baseline}"] = _paired(by_seed)
    delta = comparisons["UG_R1-R1"]
    catastrophic = {seed: bool(row["macro_f1"] < -.03 or row["far"] > .05 or row["early_recall"] < -.05)
                    for seed, row in delta["by_seed"].items()}
    checks = {"mean_macro_f1_preserved": delta["macro_f1"]["mean"] >= -.002,
              "mean_far_preserved": delta["far"]["mean"] <= .002,
              "mean_early_recall_preserved": delta["early_recall"]["mean"] >= -.005,
              "mean_binary_auprc_preserved": delta["binary_auprc"]["mean"] >= -.002,
              "no_catastrophic_seed": not any(catastrophic.values())}
    first = new["seed_results"][str(stage["seeds"][0])]["method"]
    diag = first["augmentation_audit"]["train"]
    uniform_budget = old["seed_results"][str(stage["seeds"][0])]["methods"]["C1"]["augmentation_audit"]["train"]["expected_total_noise_budget"]
    payload = {"stage": "TEP", "supported": all(checks.values()), "checks": checks,
               "catastrophic_by_seed": catastrophic, "methods": method_stats, "comparisons": comparisons,
               "mechanism_audit": _mechanism(new["criticality"], diag, uniform_budget),
               "new_runs": 3, "reused_runs": 6, "paper_final_claim_allowed": False}
    return payload, paired_rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0]); extras = sorted(set().union(*(row for row in rows)) - set(fields)); fields.extend(extras)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def write_reports(config: dict, three: dict, tep: dict) -> dict:
    status = "UG_R1_DUAL_DATASET_GO" if three["supported"] and tep["supported"] else "UG_R1_DUAL_DATASET_NO_GO"
    docs = config["docs"]; a = three["comparisons"]["UG_R1-R1"]; b = tep["comparisons"]["UG_R1-R1"]
    rows3 = "\n".join(f"| {s} | {a['by_seed'][str(s)]['macro_f1']:+.5f} | {a['by_seed'][str(s)]['far']:+.5f} | {a['by_seed'][str(s)]['class_9_recall']:+.5f} |" for s in config["three_w"]["seeds"])
    rowst = "\n".join(f"| {s} | {b['by_seed'][str(s)]['macro_f1']:+.5f} | {b['by_seed'][str(s)]['far']:+.5f} | {b['by_seed'][str(s)]['early_recall']:+.5f} | {b['by_seed'][str(s)]['binary_auprc']:+.5f} |" for s in config["tep"]["seeds"])
    m3 = three["mechanism_audit"]; mt = tep["mechanism_audit"]
    three_text = f"""# UG-R1 3W 报告

结论：`{'3W_UG_R1_GO' if three['supported'] else '3W_UG_R1_NO_GO'}`。

- UG-R1−R1 Macro-F1：`{a['macro_f1']['mean']:+.5f} ± {a['macro_f1']['std']:.5f}`，{a['macro_f1']['nonnegative_seeds']}/3 nonnegative
- FAR：`{a['far']['mean']:+.5f}`；Macro-F1 std R1/UG-R1=`{three['methods']['R1']['macro_f1']['std']:.5f}/{three['methods']['UG_R1']['macro_f1']['std']:.5f}`
- Class 9 Recall/F1 std：R1 `{three['methods']['R1']['class_9_recall']['std']:.5f}/{three['methods']['R1']['class_9_f1']['std']:.5f}`，UG-R1 `{three['methods']['UG_R1']['class_9_recall']['std']:.5f}/{three['methods']['UG_R1']['class_9_f1']['std']:.5f}`

| Seed | Δ Macro-F1 | Δ FAR | Δ Class 9 Recall |
|---:|---:|---:|---:|
{rows3}

完整 WELL bootstrap units=`{m3['bootstrap_unit_count']}`；p median=`{m3['selection_probability']['median']:.4f}`，r median=`{m3['assignment_confidence']['median']:.4f}`；changed bins=`{m3['changed_bins']}`，timestep MAE=`{m3['timestep_mean_absolute_change']:.4f}`；预算误差=`{m3['noise_budget']['absolute_error']:.3g}`。Gate：`{three['checks']}`。
"""
    tep_text = f"""# UG-R1 TEP 报告

结论：`{'TEP_UG_R1_GO' if tep['supported'] else 'TEP_UG_R1_NO_GO'}`。下游保持 binary detection。

- UG-R1−R1 Macro-F1 / FAR：`{b['macro_f1']['mean']:+.5f}` / `{b['far']['mean']:+.5f}`
- Early Recall / Binary AUPRC：`{b['early_recall']['mean']:+.5f}` / `{b['binary_auprc']['mean']:+.5f}`
- Detection Delay / detected rate：`{b['detection_delay']['mean']:+.2f}` / `{b['detected_rate']['mean']:+.5f}`

| Seed | Δ Macro-F1 | Δ FAR | Δ Early Recall | Δ AUPRC |
|---:|---:|---:|---:|---:|
{rowst}

按 faultNumber 分层 run counts=`{mt['stratified_unit_counts']}`；p median=`{mt['selection_probability']['median']:.4f}`，r median=`{mt['assignment_confidence']['median']:.4f}`；changed bins=`{mt['changed_bins']}`，timestep MAE=`{mt['timestep_mean_absolute_change']:.4f}`；预算误差=`{mt['noise_budget']['absolute_error']:.3g}`。Gate：`{tep['checks']}`。
"""
    Path(docs["three_w_report"]).write_text(three_text, encoding="utf-8")
    Path(docs["tep_report"]).write_text(tep_text, encoding="utf-8")
    summary = {"status": status, "three_w_supported": three["supported"], "tep_supported": tep["supported"],
               "new_runs": 6, "reused_runs": 12, "retrained_baselines": 0,
               "stop_uncertainty_direction": status != "UG_R1_DUAL_DATASET_GO", "paper_final_allowed": False}
    decision = ("UG-R1 双数据集 Gate 通过。" if status == "UG_R1_DUAL_DATASET_GO" else
                "UG-R1 未通过双数据集 Gate；按预注册停止该方向，不搜索其他 confidence 映射或 bootstrap 配置。")
    Path(docs["summary"]).write_text(f"# UG-R1 双数据集总结\n\n最终判定：`{status}`。\n\n6 个新增 run 全部完成，12 个 Uniform/R1 baseline 全部复用、无重训。{decision}\n\n结果仍属 exploratory validation，不可作为 paper-final claim。\n", encoding="utf-8")
    Path(docs["audit_json"]).write_text(json.dumps({"three_w": three["mechanism_audit"], "tep": tep["mechanism_audit"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/uncertainty_gated_criticality.yaml")
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    three, rows3 = summarize_three_w(config); tep, rowst = summarize_tep(config); docs = config["docs"]
    Path(docs["three_w_json"]).write_text(json.dumps(three, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(docs["tep_json"]).write_text(json.dumps(tep, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(Path(docs["paired_csv"]), rows3 + rowst)
    profile_rows = []
    for dataset, payload in (("3W", three), ("TEP", tep)):
        audit = payload["mechanism_audit"]
        for row in audit["most_uncertain_bins"]: profile_rows.append({"dataset": dataset, **row})
    _write_csv(Path(docs["uncertainty_csv"]), profile_rows)
    print(json.dumps(write_reports(config, three, tep), ensure_ascii=False))


if __name__ == "__main__": main()
