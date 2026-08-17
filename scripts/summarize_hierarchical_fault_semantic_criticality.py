from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import yaml

from frequency import mask_jaccard
from scripts.run_3w_diffusion_1seed import HFSC_METHOD, METHODS, R2_METHOD
from scripts.summarize_r2_multiclass_criticality import CORE, metric_delta


def _distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {"mean": float(array.mean()), "std": float(array.std()),
            "median": float(np.median(array)), "min": float(array.min()), "max": float(array.max())}


def _paired(by_seed: dict[str, dict]) -> dict:
    result = {"by_seed": by_seed}
    for key in next(iter(by_seed.values())):
        result[key] = {**_distribution([row[key] for row in by_seed.values()]),
                       "positive_seeds": sum(row[key] > 0 for row in by_seed.values()),
                       "nonnegative_seeds": sum(row[key] >= 0 for row in by_seed.values())}
    return result


def _hash(mask: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(mask, dtype=bool)).tobytes()).hexdigest()


def mask_audit(shared: np.ndarray, diagnostic: dict[int, np.ndarray],
               hierarchical: dict[int, np.ndarray]) -> dict:
    shared = np.asarray(shared, dtype=bool); shared_rows = {}
    for kind, value in diagnostic.items():
        value = np.asarray(value, dtype=bool)
        shared_rows[str(kind)] = {"jaccard": mask_jaccard(shared, value),
                                  "changed_bins": int(np.logical_xor(shared, value).sum()),
                                  "diagnostic_mask_sha256": _hash(value),
                                  "hierarchical_mask_sha256": _hash(hierarchical[kind])}
    pairs = {}
    for first, second in itertools.combinations(sorted(diagnostic), 2):
        value = mask_jaccard(diagnostic[first], diagnostic[second])
        pairs[f"{first}-{second}"] = value
    shared_jaccards = [row["jaccard"] for row in shared_rows.values()]
    return {"shared_mask_sha256": _hash(shared), "shared_vs_diagnostic": shared_rows,
            "shared_vs_diagnostic_jaccard_distribution": _distribution(shared_jaccards),
            "diagnostic_pairwise_jaccard": pairs,
            "pairwise_jaccard_distribution": _distribution(list(pairs.values())),
            "class_specific_patterns_confirmed": bool(any(value < .90 for value in pairs.values())),
            "near_identical_pair_count": int(sum(value >= .90 for value in pairs.values()))}


def summarize_three_w(config: dict, manifest_path: Path) -> dict:
    stage = config["three_w"]
    old_manifest = json.loads(Path(stage["existing_r1_manifest"]).read_text(encoding="utf-8"))
    r2_manifest = json.loads(Path(stage["existing_r2_manifest"]).read_text(encoding="utf-8"))
    hfsc_manifest = json.loads(manifest_path.read_text(encoding="utf-8")); records = {}; rows = []
    for seed in map(str, stage["seeds"]):
        old = json.loads(Path(old_manifest["seed_results"][seed]["result_path"]).read_text(encoding="utf-8"))
        r2 = json.loads(Path(r2_manifest["seed_results"][seed]["result_path"]).read_text(encoding="utf-8"))
        hfsc = json.loads(Path(hfsc_manifest["seed_results"][seed]["result_path"]).read_text(encoding="utf-8"))
        for key in ("window_refs_sha256", "initialization_sha256", "supcon_batch_order_sha256"):
            if r2["fairness"][key] != hfsc["fairness"][key]: raise RuntimeError(f"3W HFSC fairness changed: {key}")
        records[seed] = {"old": old, "r2": r2, "hfsc": hfsc}
        choices = (("UNIFORM", old["methods"][METHODS[1]]["metrics"]), ("R1", old["methods"][METHODS[2]]["metrics"]),
                   ("R2", r2["methods"][R2_METHOD]["metrics"]), ("HFSC", hfsc["methods"][HFSC_METHOD]["metrics"]))
        for method, metrics in choices:
            row = {"seed": int(seed), "method": method, **{key: metrics[key] for key in CORE}}
            for item in metrics["per_class"]:
                row[f"class_{item['original_class']}_recall"] = item["recall"]
                row[f"class_{item['original_class']}_f1"] = item["f1"]
            rows.append(row)
    numeric = [key for key in rows[0] if key not in {"seed", "method"}]
    methods = {method: {key: _distribution([row[key] for row in rows if row["method"] == method]) for key in numeric}
               for method in ("UNIFORM", "R1", "R2", "HFSC")}
    comparisons = {}; paired_rows = []
    keys = {"UNIFORM": ("old", METHODS[1]), "R1": ("old", METHODS[2]), "R2": ("r2", R2_METHOD)}
    for baseline, (record_key, method_key) in keys.items():
        name = f"HFSC-{baseline}"; by_seed = {}
        for seed, record in records.items():
            base = record[record_key]["methods"][method_key]["metrics"]
            current = record["hfsc"]["methods"][HFSC_METHOD]["metrics"]
            by_seed[seed] = metric_delta(base, current); paired_rows.append({"comparison": name, "seed": seed, **by_seed[seed]})
        comparisons[name] = _paired(by_seed)
    critical = records[str(stage["seeds"][0])]["hfsc"]["criticality"]
    shared = np.asarray(critical["shared"]["composite_mask"], bool)
    diagnostic = {int(k): np.asarray(v["hard_mask"], bool) for k, v in critical["diagnostic"].items()}
    hierarchical = {int(k): np.asarray(v["hard_mask"], bool) for k, v in critical["hierarchical"].items()}
    masks = mask_audit(shared, diagnostic, hierarchical)
    delta = comparisons["HFSC-R1"]
    checks = {"macro_f1_mean_positive": delta["macro_f1"]["mean"] > 0,
              "macro_f1_at_least_2of3_positive": delta["macro_f1"]["positive_seeds"] >= 2,
              "multiclass_auprc_nonworse": delta["auprc_multiclass_macro"]["mean"] >= 0,
              "far_no_systematic_degradation": delta["far"]["mean"] <= .05,
              "early_no_systematic_degradation": delta["early_recall"]["mean"] >= -.05}
    supported = all(checks.values())
    payload = {"stage": "3W", "supported": supported, "checks": checks, "seeds": stage["seeds"],
               "new_runs": 3, "reused_runs": 9, "methods": methods, "comparisons": comparisons,
               "mask_audit": masks, "paper_final_claim_allowed": False}
    docs = config["docs"]; write_outputs(Path(docs["three_w_results_csv"]), rows,
                                         Path(docs["three_w_paired_csv"]), paired_rows)
    Path(docs["three_w_json"]).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


TEP_KEYS = ("macro_f1", "recall_macro", "multiclass_auprc", "binary_auprc", "fault_recall", "far", "early_recall", "detection_delay")


def _tep_flat(method: dict) -> dict:
    metrics = method["metrics"]; diagnosis = metrics["diagnosis"]; detection = metrics["detection"]
    return {"macro_f1": diagnosis["macro_f1"], "recall_macro": diagnosis["recall_macro"],
            "multiclass_auprc": diagnosis["auprc"], "binary_auprc": detection["auprc"],
            "fault_recall": detection["fault_recall"], "far": detection["far"],
            "early_recall": metrics["early_recall"], "detection_delay": metrics["detection_delay"]}


def summarize_tep(config: dict, result_path: Path) -> dict:
    stage = config["tep"]; result = json.loads(result_path.read_text(encoding="utf-8")); rows = []; metrics = {}
    for seed in map(str, stage["seeds"]):
        record = result["seed_results"][seed]; hashes = {json.dumps(value["fairness"], sort_keys=True) for value in record["methods"].values()}
        if len(hashes) != 1: raise RuntimeError(f"TEP seed {seed} fairness hashes differ")
        metrics[seed] = {method: _tep_flat(record["methods"][method]) for method in result["methods"]}
        rows.extend({"seed": seed, "method": method, **value} for method, value in metrics[seed].items())
    methods = {method: {key: _distribution([metrics[seed][method][key] for seed in metrics]) for key in TEP_KEYS}
               for method in result["methods"]}
    comparisons = {}; paired_rows = []
    for baseline in ("UNIFORM", "R1", "R2"):
        name = f"HFSC-{baseline}"
        by_seed = {seed: {key: float(metrics[seed]["HFSC"][key] - metrics[seed][baseline][key]) for key in TEP_KEYS}
                   for seed in metrics}
        comparisons[name] = _paired(by_seed); paired_rows.extend({"comparison": name, "seed": seed, **row} for seed, row in by_seed.items())
    per_fault = {}; improved_classes = []
    for kind in range(1, 21):
        recalls = {method: [result["seed_results"][seed]["methods"][method]["metrics"]["per_class"][kind]["recall"]
                            for seed in metrics] for method in result["methods"]}
        delta = [current - base for current, base in zip(recalls["HFSC"], recalls["R1"])]
        per_fault[str(kind)] = {method: _distribution(values) for method, values in recalls.items()}
        per_fault[str(kind)]["hfsc_r1_recall_delta"] = _distribution(delta)
        if np.mean(delta) > 0: improved_classes.append(kind)
    audit_source = json.loads(Path(stage["shared_mask_audit_source"]).read_text(encoding="utf-8"))
    shared = np.asarray(audit_source["hard_masks"]["composite"], bool)
    diagnostic = {int(k): np.asarray(v["hard_mask"], bool) for k, v in result["criticality"]["diagnostic"].items()}
    hierarchical = {int(k): np.asarray(v["hard_mask"], bool) for k, v in result["criticality"]["hierarchical"].items()}
    masks = mask_audit(shared, diagnostic, hierarchical)
    if masks["shared_mask_sha256"] != result["criticality"]["shared_mask_sha256"]:
        raise RuntimeError("TEP HFSC shared mask differs from frozen R1 audit")
    delta = comparisons["HFSC-R1"]
    checks = {"macro_f1_mean_positive": delta["macro_f1"]["mean"] > 0,
              "macro_f1_majority_positive": delta["macro_f1"]["positive_seeds"] >= 2,
              "per_fault_improvement_not_sparse": len(improved_classes) >= 8,
              "binary_auprc_preserved": delta["binary_auprc"]["mean"] >= -.03,
              "fault_recall_preserved": delta["fault_recall"]["mean"] >= -.03,
              "far_preserved": delta["far"]["mean"] <= .05,
              "early_recall_preserved": delta["early_recall"]["mean"] >= -.05}
    supported = all(checks.values())
    payload = {"stage": "TEP_21_CLASS_DIAGNOSIS", "supported": supported, "checks": checks,
               "seeds": stage["seeds"], "new_runs": 12, "methods": methods, "comparisons": comparisons,
               "per_fault": per_fault, "fault_classes_with_positive_mean_recall_delta_vs_r1": len(improved_classes),
               "positive_mean_recall_delta_fault_classes": improved_classes,
               "mask_audit": masks, "paper_final_claim_allowed": False}
    docs = config["docs"]; write_outputs(Path(docs["tep_results_csv"]), rows, Path(docs["tep_paired_csv"]), paired_rows)
    Path(docs["tep_json"]).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def write_outputs(results_path: Path, rows: list[dict], paired_path: Path, paired: list[dict]) -> None:
    for path, data in ((results_path, rows), (paired_path, paired)):
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)


def write_reports(config: dict, three_w: dict, tep: dict) -> dict:
    if three_w["supported"] and tep["supported"]: status = "HFSC_DUAL_DATASET_GO"
    elif three_w["supported"] or tep["supported"]: status = "HFSC_PARTIAL_GO"
    else: status = "HFSC_DUAL_DATASET_NO_GO"
    docs = config["docs"]; t3 = three_w["comparisons"]; tt = tep["comparisons"]
    three_seed_rows = "\n".join(
        f"| {seed} | {t3['HFSC-R1']['by_seed'][seed]['macro_f1']:+.5f} | {t3['HFSC-R2']['by_seed'][seed]['macro_f1']:+.5f} |"
        for seed in map(str, three_w["seeds"]))
    tep_seed_rows = "\n".join(
        f"| {seed} | {tt['HFSC-R1']['by_seed'][seed]['macro_f1']:+.5f} | {tt['HFSC-R2']['by_seed'][seed]['macro_f1']:+.5f} | "
        f"{tt['HFSC-R1']['by_seed'][seed]['binary_auprc']:+.5f} | {tt['HFSC-R1']['by_seed'][seed]['far']:+.5f} |"
        for seed in map(str, tep["seeds"]))
    class_rows = "\n".join(
        f"| {kind} | {t3['HFSC-R1'][f'class_{kind}_recall']['mean']:+.5f} | "
        f"{t3['HFSC-R1'][f'class_{kind}_f1']['mean']:+.5f} | {t3['HFSC-R1'][f'class_{kind}_recall']['positive_seeds']}/3 |"
        for kind in (2, 8, 9))
    three_text = f"""# HFSC 3W 报告

结论：`{'3W_HFSC_GO' if three_w['supported'] else '3W_HFSC_NO_GO'}`。

- HFSC−R1 Macro-F1：`{t3['HFSC-R1']['macro_f1']['mean']:+.5f} ± {t3['HFSC-R1']['macro_f1']['std']:.5f}`，{t3['HFSC-R1']['macro_f1']['positive_seeds']}/3 positive
- HFSC−R2 Macro-F1：`{t3['HFSC-R2']['macro_f1']['mean']:+.5f} ± {t3['HFSC-R2']['macro_f1']['std']:.5f}`
- HFSC−R1 Multiclass AUPRC：`{t3['HFSC-R1']['auprc_multiclass_macro']['mean']:+.5f}`
- HFSC−R1 FAR / Early Recall：`{t3['HFSC-R1']['far']['mean']:+.5f}` / `{t3['HFSC-R1']['early_recall']['mean']:+.5f}`

| Seed | HFSC−R1 Macro-F1 | HFSC−R2 Macro-F1 |
|---:|---:|---:|
{three_seed_rows}

| Fault | Δ Recall vs R1 | Δ F1 vs R1 | Recall positive seeds |
|---:|---:|---:|---:|
{class_rows}

Fault 2/8/9 的配对 Recall 变化完整记录在 JSON/CSV。Shared-vs-Diagnostic Jaccard 范围为 `[{three_w['mask_audit']['shared_vs_diagnostic_jaccard_distribution']['min']:.5f}, {three_w['mask_audit']['shared_vs_diagnostic_jaccard_distribution']['max']:.5f}]`；Diagnostic masks pairwise Jaccard 范围为 `[{three_w['mask_audit']['pairwise_jaccard_distribution']['min']:.5f}, {three_w['mask_audit']['pairwise_jaccard_distribution']['max']:.5f}]`，class-specific patterns confirmed=`{three_w['mask_audit']['class_specific_patterns_confirmed']}`。

Gate：`{three_w['checks']}`。HFSC 未达到相对 R1 至少 2/3 seed 正向的稳定性要求。
"""
    tep_text = f"""# HFSC TEP 21 类 Diagnosis 报告

结论：`{'TEP_HFSC_GO' if tep['supported'] else 'TEP_HFSC_NO_GO'}`。

- HFSC−R1 Macro-F1：`{tt['HFSC-R1']['macro_f1']['mean']:+.5f} ± {tt['HFSC-R1']['macro_f1']['std']:.5f}`，{tt['HFSC-R1']['macro_f1']['positive_seeds']}/3 positive
- HFSC−R2 Macro-F1：`{tt['HFSC-R2']['macro_f1']['mean']:+.5f} ± {tt['HFSC-R2']['macro_f1']['std']:.5f}`
- HFSC−R1 Multiclass AUPRC：`{tt['HFSC-R1']['multiclass_auprc']['mean']:+.5f}`
- HFSC−R1 Binary AUPRC / Recall / FAR：`{tt['HFSC-R1']['binary_auprc']['mean']:+.5f}` / `{tt['HFSC-R1']['fault_recall']['mean']:+.5f}` / `{tt['HFSC-R1']['far']['mean']:+.5f}`
- 正 mean Recall 的 fault classes：`{tep['fault_classes_with_positive_mean_recall_delta_vs_r1']}/20`
- 对应 fault：`{tep['positive_mean_recall_delta_fault_classes']}`

| Seed | HFSC−R1 Macro-F1 | HFSC−R2 Macro-F1 | Δ Binary AUPRC vs R1 | Δ FAR vs R1 |
|---:|---:|---:|---:|---:|
{tep_seed_rows}

20 类逐类 Recall/F1 与配对结果见 JSON。Shared-vs-Diagnostic Jaccard median `{tep['mask_audit']['shared_vs_diagnostic_jaccard_distribution']['median']:.5f}`；Diagnostic masks pairwise Jaccard median `{tep['mask_audit']['pairwise_jaccard_distribution']['median']:.5f}`，class-specific patterns confirmed=`{tep['mask_audit']['class_specific_patterns_confirmed']}`。

Gate：`{tep['checks']}`。本协议仍为 exploratory，不能形成 paper-final claim。
"""
    Path(docs["three_w_report"]).write_text(three_text, encoding="utf-8")
    Path(docs["tep_report"]).write_text(tep_text, encoding="utf-8")
    combined_masks = {"three_w": three_w["mask_audit"], "tep": tep["mask_audit"]}
    Path(docs["mask_audit_json"]).write_text(json.dumps(combined_masks, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"status": status, "three_w_supported": three_w["supported"], "tep_supported": tep["supported"],
               "ablation_allowed": status == "HFSC_DUAL_DATASET_GO", "paper_final_allowed": False,
               "new_runs": 15, "reused_runs": 9}
    text = f"""# HFSC 双数据集总结

最终判定：`{status}`。

3W 未形成相对 R1 的稳定 Macro-F1 改善；TEP 21 类 Diagnosis 显示相对 R1 的多数 seed 正向信号。证据只构成部分支持，HFSC 不升级为最终论文主方法，不搜索 λ，不继续添加模块。`ablation_allowed={summary['ablation_allowed']}`，当前结果不得作为 paper-final claim。
"""
    Path(docs["summary"]).write_text(text, encoding="utf-8"); return summary


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/hierarchical_fault_semantic_criticality.yaml")
    parser.add_argument("--three-w-manifest", type=Path, default=Path("outputs/hfsc_3w/result_manifest.json"))
    parser.add_argument("--tep-result", type=Path, default=Path("outputs/hfsc_tep_multiclass/result.json"))
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    three_w = summarize_three_w(config, args.three_w_manifest); tep = summarize_tep(config, args.tep_result)
    print(json.dumps(write_reports(config, three_w, tep), ensure_ascii=False))


if __name__ == "__main__": main()
