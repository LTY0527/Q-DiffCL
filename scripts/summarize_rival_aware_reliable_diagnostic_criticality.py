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
from scripts.run_3w_diffusion_1seed import HFSC_METHOD, METHODS, R2_METHOD, RRDC_METHOD
from scripts.summarize_hierarchical_fault_semantic_criticality import _distribution, _paired, _tep_flat
from scripts.summarize_r2_multiclass_criticality import CORE, metric_delta


def _hash(mask: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(mask, dtype=bool)).tobytes()).hexdigest()


def _mask_audit(criticality: dict) -> dict:
    shared = np.asarray(criticality["shared_hard_mask"] if "shared_hard_mask" in criticality
                        else criticality["shared"]["composite_mask"], bool)
    diagnostic = {int(k): np.asarray(v["hard_mask"], bool) for k, v in criticality["diagnostic"].items()}
    final = {int(k): np.asarray(v["hard_mask"], bool) for k, v in criticality["final"].items()}
    shared_rows = {str(k): {"jaccard": mask_jaccard(shared, value),
                            "changed_bins": int(np.logical_xor(shared, value).sum()),
                            "diagnostic_mask_sha256": _hash(value), "final_mask_sha256": _hash(final[k])}
                   for k, value in diagnostic.items()}
    pairs = {f"{a}-{b}": mask_jaccard(diagnostic[a], diagnostic[b])
             for a, b in itertools.combinations(sorted(diagnostic), 2)}
    reliability = {str(k): {**_distribution(np.asarray(v["reliability"], float).reshape(-1).tolist()),
                             "zero_fraction": float(np.mean(np.asarray(v["reliability"]) == 0)),
                             "one_fraction": float(np.mean(np.asarray(v["reliability"]) == 1))}
                   for k, v in criticality["diagnostic"].items()}
    hardest = {str(k): {"rival": int(v["hardest_rival"]), "score": float(v["hardest_rival_score"])}
               for k, v in criticality["diagnostic"].items()}
    return {"shared_mask_sha256": _hash(shared), "shared_vs_reliable_diagnostic": shared_rows,
            "shared_jaccard_distribution": _distribution([v["jaccard"] for v in shared_rows.values()]),
            "diagnostic_pairwise_jaccard": pairs, "pairwise_jaccard_distribution": _distribution(list(pairs.values())),
            "reliability": reliability, "hardest_rivals": hardest,
            "class_specific_patterns_confirmed": any(value < .90 for value in pairs.values())}


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def summarize_three_w(config: dict) -> dict:
    stage = config["three_w"]
    manifests = {"old": json.loads(Path(stage["existing_r1_manifest"]).read_text(encoding="utf-8")),
                 "r2": json.loads(Path(stage["existing_r2_manifest"]).read_text(encoding="utf-8")),
                 "hfsc": json.loads(Path(stage["existing_hfsc_manifest"]).read_text(encoding="utf-8")),
                 "rrdc": json.loads((Path(stage["output_dir"]) / "result_manifest.json").read_text(encoding="utf-8"))}
    rows = []; paired_rows = []; records = {}
    methods = {"UNIFORM": ("old", METHODS[1]), "R1": ("old", METHODS[2]),
               "R2": ("r2", R2_METHOD), "HFSC": ("hfsc", HFSC_METHOD), "RRDC": ("rrdc", RRDC_METHOD)}
    for seed in map(str, stage["seeds"]):
        records[seed] = {name: json.loads(Path(manifest["seed_results"][seed]["result_path"]).read_text(encoding="utf-8"))
                         for name, manifest in manifests.items()}
        reference = records[seed]["rrdc"]["fairness"]
        for name in ("r2", "hfsc"):
            for key in ("window_refs_sha256", "initialization_sha256", "supcon_batch_order_sha256"):
                if records[seed][name]["fairness"][key] != reference[key]:
                    raise RuntimeError(f"3W RRDC fairness differs from {name}: seed={seed}, key={key}")
        for method, (source, key) in methods.items():
            metrics = records[seed][source]["methods"][key]["metrics"]
            row = {"seed": int(seed), "method": method, **{name: metrics[name] for name in CORE}}
            for item in metrics["per_class"]:
                row[f"class_{item['original_class']}_recall"] = item["recall"]
                row[f"class_{item['original_class']}_f1"] = item["f1"]
            rows.append(row)
    comparisons = {}
    for baseline in ("R1", "R2", "HFSC"):
        by_seed = {}
        base_source, base_key = methods[baseline]
        for seed, record in records.items():
            base = record[base_source]["methods"][base_key]["metrics"]
            current = record["rrdc"]["methods"][RRDC_METHOD]["metrics"]
            by_seed[seed] = metric_delta(base, current)
            paired_rows.append({"comparison": f"RRDC-{baseline}", "seed": seed, **by_seed[seed]})
        comparisons[f"RRDC-{baseline}"] = _paired(by_seed)
    numeric = [key for key in rows[0] if key not in {"seed", "method"}]
    method_stats = {method: {key: _distribution([r[key] for r in rows if r["method"] == method]) for key in numeric}
                    for method in methods}
    criticality = records[str(stage["seeds"][0])]["rrdc"]["criticality"]
    masks = _mask_audit(criticality); delta = comparisons["RRDC-R1"]
    fault_means = {kind: delta[f"class_{kind}_recall"]["mean"] for kind in (2, 8, 9)}
    checks = {"macro_f1_mean_positive": delta["macro_f1"]["mean"] > 0,
              "macro_f1_at_least_2of3_positive": delta["macro_f1"]["positive_seeds"] >= 2,
              "multiclass_auprc_nonworse": delta["auprc_multiclass_macro"]["mean"] >= 0,
              "fault_2_8_9_balanced": sum(v > 0 for v in fault_means.values()) >= 2 and min(fault_means.values()) >= -.05,
              "detection_preserved": delta["auprc_fault_vs_normal"]["mean"] >= -.03 and delta["far"]["mean"] <= .05,
              "early_preserved": delta["early_recall"]["mean"] >= -.05}
    payload = {"stage": "3W", "supported": all(checks.values()), "checks": checks, "seeds": stage["seeds"],
               "new_runs": 3, "reused_runs": 12, "methods": method_stats, "comparisons": comparisons,
               "fault_2_8_9_recall_delta_vs_r1": fault_means, "mask_audit": masks,
               "paper_final_claim_allowed": False}
    docs = config["docs"]; _write_csv(Path(docs["three_w_results_csv"]), rows)
    _write_csv(Path(docs["three_w_paired_csv"]), paired_rows)
    Path(docs["three_w_json"]).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def summarize_tep(config: dict) -> dict:
    stage = config["tep"]
    baseline = json.loads(Path(stage["existing_hfsc_result"]).read_text(encoding="utf-8"))
    result = json.loads((Path(stage["output_dir"]) / "result.json").read_text(encoding="utf-8"))
    rows = []; paired_rows = []; all_metrics = {}; methods = ("UNIFORM", "R1", "R2", "HFSC", "RRDC")
    for seed in map(str, stage["seeds"]):
        rrdc = result["seed_results"][seed]["method"]
        if rrdc["fairness"] != baseline["seed_results"][seed]["fairness"]:
            raise RuntimeError(f"TEP RRDC fairness differs for seed {seed}")
        all_metrics[seed] = {method: _tep_flat(rrdc if method == "RRDC" else baseline["seed_results"][seed]["methods"][method])
                             for method in methods}
        rows.extend({"seed": seed, "method": method, **values} for method, values in all_metrics[seed].items())
    comparisons = {}
    for base in ("R1", "R2", "HFSC"):
        by_seed = {seed: {key: all_metrics[seed]["RRDC"][key] - all_metrics[seed][base][key]
                          for key in all_metrics[seed]["RRDC"]} for seed in all_metrics}
        comparisons[f"RRDC-{base}"] = _paired(by_seed)
        paired_rows.extend({"comparison": f"RRDC-{base}", "seed": seed, **values} for seed, values in by_seed.items())
    method_stats = {method: {key: _distribution([all_metrics[seed][method][key] for seed in all_metrics])
                             for key in all_metrics[next(iter(all_metrics))][method]} for method in methods}
    per_fault = {}; improved = []
    for kind in range(1, 21):
        values = {method: [(result["seed_results"][seed]["method"] if method == "RRDC"
                            else baseline["seed_results"][seed]["methods"][method])["metrics"]["per_class"][kind]["recall"]
                           for seed in all_metrics] for method in methods}
        delta = [current - base for current, base in zip(values["RRDC"], values["R1"])]
        per_fault[str(kind)] = {method: _distribution(current) for method, current in values.items()}
        per_fault[str(kind)]["rrdc_r1_recall_delta"] = _distribution(delta)
        if np.mean(delta) > 0: improved.append(kind)
    masks = _mask_audit(result["criticality"]); delta = comparisons["RRDC-R1"]
    checks = {"macro_f1_mean_positive": delta["macro_f1"]["mean"] > 0,
              "macro_f1_at_least_2of3_positive": delta["macro_f1"]["positive_seeds"] >= 2,
              "improved_fault_count_exceeds_hfsc": len(improved) > 3,
              "improvement_not_sparse": len(improved) >= 8,
              "binary_auprc_preserved": delta["binary_auprc"]["mean"] >= -.03,
              "fault_recall_preserved": delta["fault_recall"]["mean"] >= -.03,
              "far_preserved": delta["far"]["mean"] <= .05,
              "early_recall_preserved": delta["early_recall"]["mean"] >= -.05}
    payload = {"stage": "TEP_21_CLASS_DIAGNOSIS", "supported": all(checks.values()), "checks": checks,
               "seeds": stage["seeds"], "new_runs": 3, "reused_runs": 12, "methods": method_stats,
               "comparisons": comparisons, "per_fault": per_fault,
               "fault_classes_with_positive_mean_recall_delta_vs_r1": len(improved),
               "positive_mean_recall_delta_fault_classes": improved, "mask_audit": masks,
               "paper_final_claim_allowed": False}
    docs = config["docs"]; _write_csv(Path(docs["tep_results_csv"]), rows)
    _write_csv(Path(docs["tep_paired_csv"]), paired_rows)
    Path(docs["tep_json"]).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def write_reports(config: dict, three_w: dict, tep: dict) -> dict:
    status = "RRDC_DUAL_DATASET_GO" if three_w["supported"] and tep["supported"] else "RRDC_DUAL_DATASET_NO_GO"
    docs = config["docs"]; a = three_w["comparisons"]; b = tep["comparisons"]
    seed3 = "\n".join(f"| {s} | {a['RRDC-R1']['by_seed'][str(s)]['macro_f1']:+.5f} | {a['RRDC-R2']['by_seed'][str(s)]['macro_f1']:+.5f} | {a['RRDC-HFSC']['by_seed'][str(s)]['macro_f1']:+.5f} |" for s in three_w["seeds"])
    seedt = "\n".join(f"| {s} | {b['RRDC-R1']['by_seed'][str(s)]['macro_f1']:+.5f} | {b['RRDC-HFSC']['by_seed'][str(s)]['macro_f1']:+.5f} | {b['RRDC-R1']['by_seed'][str(s)]['binary_auprc']:+.5f} |" for s in tep["seeds"])
    fault3 = "\n".join(f"| {k} | {three_w['fault_2_8_9_recall_delta_vs_r1'][k]:+.5f} |" for k in (2, 8, 9))
    hardest3 = ", ".join(f"Fault {k}→{v['rival']} ({v['score']:.5g})" for k, v in three_w["mask_audit"]["hardest_rivals"].items())
    hardestt = ", ".join(f"{k}→{v['rival']}" for k, v in tep["mask_audit"]["hardest_rivals"].items())
    report3 = f"""# RRDC 3W 报告

结论：`{'3W_RRDC_GO' if three_w['supported'] else '3W_RRDC_NO_GO'}`。

- RRDC−R1 Macro-F1：`{a['RRDC-R1']['macro_f1']['mean']:+.5f} ± {a['RRDC-R1']['macro_f1']['std']:.5f}`，{a['RRDC-R1']['macro_f1']['positive_seeds']}/3 positive
- RRDC−R2 / HFSC Macro-F1：`{a['RRDC-R2']['macro_f1']['mean']:+.5f}` / `{a['RRDC-HFSC']['macro_f1']['mean']:+.5f}`
- RRDC−R1 Multiclass AUPRC：`{a['RRDC-R1']['auprc_multiclass_macro']['mean']:+.5f}`
- RRDC−R1 Binary AUPRC / FAR / Early Recall：`{a['RRDC-R1']['auprc_fault_vs_normal']['mean']:+.5f}` / `{a['RRDC-R1']['far']['mean']:+.5f}` / `{a['RRDC-R1']['early_recall']['mean']:+.5f}`

| Seed | Δ R1 Macro-F1 | Δ R2 | Δ HFSC |
|---:|---:|---:|---:|
{seed3}

| Fault | Mean Δ Recall vs R1 |
|---:|---:|
{fault3}

Hardest rivals：{hardest3}。完整 reliability、mask hash、changed bins 与 Jaccard 见 JSON。Gate：`{three_w['checks']}`。
"""
    reportt = f"""# RRDC TEP 21 类 Diagnosis 报告

结论：`{'TEP_RRDC_GO' if tep['supported'] else 'TEP_RRDC_NO_GO'}`。

- RRDC−R1 Macro-F1：`{b['RRDC-R1']['macro_f1']['mean']:+.5f} ± {b['RRDC-R1']['macro_f1']['std']:.5f}`，{b['RRDC-R1']['macro_f1']['positive_seeds']}/3 positive
- RRDC−HFSC Macro-F1：`{b['RRDC-HFSC']['macro_f1']['mean']:+.5f}`
- RRDC−R1 Multiclass / Binary AUPRC：`{b['RRDC-R1']['multiclass_auprc']['mean']:+.5f}` / `{b['RRDC-R1']['binary_auprc']['mean']:+.5f}`
- RRDC−R1 Fault Recall / FAR / Early Recall：`{b['RRDC-R1']['fault_recall']['mean']:+.5f}` / `{b['RRDC-R1']['far']['mean']:+.5f}` / `{b['RRDC-R1']['early_recall']['mean']:+.5f}`
- Mean Recall 改善 fault：`{tep['fault_classes_with_positive_mean_recall_delta_vs_r1']}/20`，类别 `{tep['positive_mean_recall_delta_fault_classes']}`

| Seed | Δ R1 Macro-F1 | Δ HFSC | Δ Binary AUPRC |
|---:|---:|---:|---:|
{seedt}

Hardest-rival pairs：{hardestt}。完整逐类结果与 reliability/mask 审计见 JSON。Gate：`{tep['checks']}`。
"""
    Path(docs["three_w_report"]).write_text(report3, encoding="utf-8")
    Path(docs["tep_report"]).write_text(reportt, encoding="utf-8")
    Path(docs["mask_audit_json"]).write_text(json.dumps({"three_w": three_w["mask_audit"], "tep": tep["mask_audit"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"status": status, "three_w_supported": three_w["supported"], "tep_supported": tep["supported"],
               "new_runs": 6, "reused_runs": 24, "stop_diagnosis_aware_development": status != "RRDC_DUAL_DATASET_GO",
               "paper_final_allowed": False}
    conclusion = ("双数据集均通过，RRDC 可冻结为 Diagnosis-aware 扩展候选。" if status == "RRDC_DUAL_DATASET_GO"
                  else "RRDC 未在双数据集形成稳定 Diagnosis 改善；按预设 Gate 停止继续增加 Diagnosis-aware augmentation 模块，保留 R1 为主方法。")
    Path(docs["summary"]).write_text(f"# RRDC 双数据集总结\n\n最终判定：`{status}`。\n\n{conclusion}\n\n本轮仅新增 6 个 RRDC run，复用 24 个一致 baseline run；仍属 exploratory validation，不作 paper-final claim。\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/rival_aware_reliable_diagnostic_criticality.yaml")
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    three_w = summarize_three_w(config); tep = summarize_tep(config)
    print(json.dumps(write_reports(config, three_w, tep), ensure_ascii=False))


if __name__ == "__main__": main()
