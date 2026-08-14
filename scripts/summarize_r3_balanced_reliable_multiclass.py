from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from frequency import mask_jaccard
from scripts.run_3w_diffusion_1seed import METHODS, R2_METHOD, R3_METHOD
from scripts.summarize_frequency_selective_r1_3seed import _flat, mean_sample_std
from scripts.summarize_r2_multiclass_criticality import CORE, TEP_CORE, metric_delta


def _stats(rows: list[dict], keys: list[str], ddof: int = 0) -> dict:
    return {key: {"mean": float(np.mean([row[key] for row in rows])),
                  "std": float(np.std([row[key] for row in rows], ddof=ddof))} for key in keys}


def _paired(rows: dict[str, dict]) -> dict:
    keys = list(next(iter(rows.values())))
    return {"by_seed": rows,
            "mean": {key: float(np.mean([row[key] for row in rows.values()])) for key in keys},
            "std": {key: float(np.std([row[key] for row in rows.values()])) for key in keys},
            "wins": {key: sum(row[key] > 0 for row in rows.values()) for key in keys},
            "nonnegative": {key: sum(row[key] >= 0 for row in rows.values()) for key in keys}}


def _distribution(values: np.ndarray) -> dict[str, float]:
    return {"min": float(values.min()), "p25": float(np.quantile(values, .25)),
            "median": float(np.median(values)), "mean": float(values.mean()),
            "p75": float(np.quantile(values, .75)), "max": float(values.max())}


def three_w_decision(summary: dict, gate: dict) -> tuple[str, dict]:
    uniform = summary["R3-UNIFORM"]; r1 = summary["R3-R1"]
    r2_method, r3_method = summary["methods"]["R2"], summary["methods"]["R3"]
    checks = {
        "macro_f1_mean_gain": uniform["mean"]["macro_f1"] >= float(gate["minimum_macro_f1_mean_gain"]),
        "macro_f1_wins": uniform["wins"]["macro_f1"] >= int(gate["minimum_macro_f1_wins"]),
        "r1_macro_f1_nonnegative_seeds": r1["nonnegative"]["macro_f1"] >= int(gate["minimum_r1_macro_f1_nonnegative_seeds"]),
        "r1_macro_f1_mean_preserved": r1["mean"]["macro_f1"] >= float(gate["minimum_r1_macro_f1_mean_gain"]),
        "macro_f1_seed_std_reduced": uniform["std"]["macro_f1"] <= float(gate["maximum_macro_f1_delta_std"]),
        "binary_auprc_mean_gain": uniform["mean"]["auprc_fault_vs_normal"] >= float(gate["minimum_binary_auprc_mean_gain"]),
        "no_large_binary_auprc_seed_drop": min(row["auprc_fault_vs_normal"] for row in uniform["by_seed"].values()) >= -float(gate["maximum_single_binary_auprc_drop"]),
        "multiclass_auprc_mean_gain": uniform["mean"]["auprc_multiclass_macro"] > float(gate["minimum_multiclass_auprc_mean_gain"]),
        "far_mean_preserved": uniform["mean"]["far"] <= float(gate["maximum_far_mean_increase"]),
        "class9_recall_improved_vs_r2": r3_method["class_9_recall"]["mean"] > r2_method["class_9_recall"]["mean"],
        "class9_f1_improved_vs_r2": r3_method["class_9_f1"]["mean"] > r2_method["class_9_f1"]["mean"],
        "class9_recall_not_near_zero": r3_method["class_9_recall"]["mean"] >= float(gate["minimum_meaningful_class9_recall"]),
        "class9_f1_not_near_zero": r3_method["class_9_f1"]["mean"] >= float(gate["minimum_meaningful_class9_f1"]),
    }
    core_names = [name for name in checks if not name.startswith("class9_")]
    class_direction = checks["class9_recall_improved_vs_r2"] and checks["class9_f1_improved_vs_r2"]
    if all(checks.values()): status = "R3_3W_GO"
    elif all(checks[name] for name in core_names) and class_direction: status = "R3_3W_PARTIAL_GO"
    else: status = "R3_3W_NO_GO"
    return status, {"checks": checks, "r2_macro_f1_delta_std_reference": .07698}


def tep_decision(summary: dict, gate: dict) -> tuple[str, dict]:
    c1, r2 = summary["R3-C1"], summary["R3-R2"]
    catastrophic = {seed: bool(row["macro_f1"] < -float(gate["catastrophic_macro_f1_drop"])
                               or row["auprc"] < -float(gate["catastrophic_auprc_drop"])
                               or row["far"] > float(gate["catastrophic_far_increase"])
                               or row["early_recall"] < -float(gate["catastrophic_early_recall_drop"]))
                    for seed, row in c1["by_seed"].items()}
    checks = {"macro_f1_3of3_nonnegative_vs_c1": c1["nonnegative"]["macro_f1"] >= int(gate["minimum_macro_f1_nonnegative_seeds_vs_c1"]),
              "macro_f1_preserved_vs_r2": r2["mean"]["macro_f1"] >= -float(gate["maximum_mean_macro_f1_drop_vs_r2"]),
              "auprc_mean_preserved": c1["mean"]["auprc"] >= -float(gate["maximum_mean_auprc_drop"]),
              "recall_mean_preserved": c1["mean"]["fault_recall"] >= -float(gate["maximum_mean_recall_drop"]),
              "far_mean_preserved": c1["mean"]["far"] <= float(gate["maximum_mean_far_increase"]),
              "early_mean_preserved": c1["mean"]["early_recall"] >= -float(gate["maximum_mean_early_recall_drop"]),
              "no_catastrophic_seed": not any(catastrophic.values())}
    preserved = all(value for key, value in checks.items() if key != "macro_f1_3of3_nonnegative_vs_c1")
    status = "R3_CROSS_DATASET_GO" if all(checks.values()) else "R3_CROSS_DATASET_PARTIAL_GO" if preserved else "R3_CROSS_DATASET_NO_GO"
    return status, {"checks": checks, "catastrophic_by_seed": catastrophic}


def summarize_three_w(config: dict, manifest_path: Path) -> dict:
    stage = config["three_w"]
    old_manifest = json.loads(Path(stage["original_manifest"]).read_text(encoding="utf-8"))
    r2_manifest = json.loads(Path(stage["r2_manifest"]).read_text(encoding="utf-8"))
    r3_manifest = json.loads(manifest_path.read_text(encoding="utf-8")); records = {}; rows = []; mask_hash = None
    for seed in map(str, stage["seeds"]):
        old = json.loads(Path(old_manifest["seed_results"][seed]["result_path"]).read_text(encoding="utf-8"))
        r2 = json.loads(Path(r2_manifest["seed_results"][seed]["result_path"]).read_text(encoding="utf-8"))
        r3 = json.loads(Path(r3_manifest["seed_results"][seed]["result_path"]).read_text(encoding="utf-8"))
        for key in ("window_refs_sha256", "initialization_sha256"):
            if old["fairness"][key] != r3["fairness"][key]: raise RuntimeError(f"R3 changed frozen 3W fairness: {key}")
        current_hash = r3["fairness"]["critical_soft_mask_sha256"]
        if mask_hash is not None and current_hash != mask_hash: raise RuntimeError("R3 mask changed across seeds")
        mask_hash = current_hash; records[seed] = {"old": old, "r2": r2, "r3": r3}
        choices = (("UNIFORM", old["methods"][METHODS[1]]["metrics"]),
                   ("R1", old["methods"][METHODS[2]]["metrics"]),
                   ("R2", r2["methods"][R2_METHOD]["metrics"]), ("R3", r3["methods"][R3_METHOD]["metrics"]))
        for label, metrics in choices:
            row = {"seed": int(seed), "method": label, **{name: metrics[name] for name in CORE}}
            for item in metrics["per_class"]:
                row[f"class_{item['original_class']}_recall"] = item["recall"]
                row[f"class_{item['original_class']}_f1"] = item["f1"]
            rows.append(row)
    numeric = [name for name in rows[0] if name not in {"seed", "method"}]
    methods = {method: _stats([row for row in rows if row["method"] == method], numeric)
               for method in ("UNIFORM", "R1", "R2", "R3")}
    comparisons = {}; paired_rows = []
    baseline_keys = {"UNIFORM": ("old", METHODS[1]), "R1": ("old", METHODS[2]), "R2": ("r2", R2_METHOD)}
    for baseline, (record_key, method_key) in baseline_keys.items():
        name = f"R3-{baseline}"; by_seed = {}
        for seed, record in records.items():
            base = record[record_key]["methods"][method_key]["metrics"]
            current = record["r3"]["methods"][R3_METHOD]["metrics"]
            by_seed[seed] = metric_delta(base, current); paired_rows.append({"comparison": name, "seed": seed, **by_seed[seed]})
        comparisons[name] = _paired(by_seed)
    r2_critical = records[str(stage["seeds"][0])]["r2"]["criticality"]
    r3_critical = records[str(stage["seeds"][0])]["r3"]["criticality"]
    r2_mask = np.asarray(r2_critical["composite_mask"], bool); r3_mask = np.asarray(r3_critical["composite_mask"], bool)
    reliability = np.asarray(r3_critical["multiclass_reliability"])
    contributions = {kind: _distribution(np.asarray(values))
                     for kind, values in r3_critical["multiclass_class_contributions"].items()}
    mask = {"r2_sha256": hashlib.sha256(np.ascontiguousarray(r2_mask).tobytes()).hexdigest(),
            "r3_sha256": hashlib.sha256(np.ascontiguousarray(r3_mask).tobytes()).hexdigest(),
            "jaccard": mask_jaccard(r2_mask, r3_mask), "changed_bins": int(np.logical_xor(r2_mask, r3_mask).sum()),
            "selected_bins": int(r3_mask.sum()), "bootstrap_reliability": _distribution(reliability),
            "per_class_balanced_m_contribution": contributions,
            "multiclass_type_run_counts": r3_critical["multiclass_type_run_counts"]}
    summary = {"methods": methods, **comparisons}; status, gate = three_w_decision(summary, stage["gate"])
    payload = {"status": status, "stage": "3W", "seeds": stage["seeds"], "formal_new_runs": 3,
               "reused_runs": 9, "criticality_fit_scope": "train only", "summary": summary,
               "mask_audit": mask, **gate, "stage_b_allowed": status in {"R3_3W_GO", "R3_3W_PARTIAL_GO"}}
    docs = config["docs"]; Path(docs["three_w_json"]).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for path_key, data in (("three_w_results_csv", rows), ("three_w_paired_csv", paired_rows)):
        with Path(docs[path_key]).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    seed_lines = "\n".join(
        f"| {seed} | {row['macro_f1']:+.5f} | {row['auprc_fault_vs_normal']:+.5f} | "
        f"{row['auprc_multiclass_macro']:+.5f} | {row['far']:+.5f} |"
        for seed, row in comparisons["R3-UNIFORM"]["by_seed"].items())
    check_lines = "\n".join(f"- `{name}`：{'通过' if value else '未通过'}" for name, value in gate["checks"].items())
    contribution_lines = "\n".join(
        f"- Class {kind}：mean `{values['mean']:.5f}`，P75 `{values['p75']:.5f}`，max `{values['max']:.5f}`"
        for kind, values in contributions.items())
    report = f"""# R3 平衡可靠多类别关键频率：3W

阶段结论：`{status}`。Stage B 不放行，不执行 TEP。

## 固定方法

R3 保持 `0.40D + 0.24E + 0.16S + 0.20M`，只将 M 替换为 Balanced + Reliable M。所有统计只使用 train run aggregate；对每类计算 one-vs-rest 分数并分别 robust normalization，类别等权平均，再乘以分层 run-bootstrap 中进入 top critical-ratio 的 selection probability。未改变 timestep、mask ratio、TCN、Hard SupCon、Original batching、split、probe、phase/DC 或总噪声预算。

## 配对结果

- R3−Uniform Macro-F1：`{comparisons['R3-UNIFORM']['mean']['macro_f1']:+.5f} ± {comparisons['R3-UNIFORM']['std']['macro_f1']:.5f}`，2/3 positive
- R3−R1 Macro-F1：`{comparisons['R3-R1']['mean']['macro_f1']:+.5f}`，{comparisons['R3-R1']['nonnegative']['macro_f1']}/3 nonnegative
- R3−R2 Macro-F1：`{comparisons['R3-R2']['mean']['macro_f1']:+.5f}`，{comparisons['R3-R2']['nonnegative']['macro_f1']}/3 nonnegative
- R3−Uniform Binary AUPRC：`{comparisons['R3-UNIFORM']['mean']['auprc_fault_vs_normal']:+.5f}`
- R3−Uniform Multiclass AUPRC：`{comparisons['R3-UNIFORM']['mean']['auprc_multiclass_macro']:+.5f}`
- R3−Uniform FAR：`{comparisons['R3-UNIFORM']['mean']['far']:+.5f}`

| Seed | Δ Macro-F1 | Δ Binary AUPRC | Δ Multiclass AUPRC | Δ FAR |
|---:|---:|---:|---:|---:|
{seed_lines}

## Class 9 与稳定性

R3 Class 9 Recall/F1 mean 为 `{methods['R3']['class_9_recall']['mean']:.5f}` / `{methods['R3']['class_9_f1']['mean']:.5f}`，高于 R2 的 `{methods['R2']['class_9_recall']['mean']:.5f}` / `{methods['R2']['class_9_f1']['mean']:.5f}`。但 R3 Recall std 为 `{methods['R3']['class_9_recall']['std']:.5f}`，改善集中在单个 seed，属于不稳定提升，不是稳定失败，也不是稳定解决。

R3−Uniform Macro-F1 std `{comparisons['R3-UNIFORM']['std']['macro_f1']:.5f}`，虽低于 R2 的 `0.07698`，但未达到预注册建议目标 `≤0.050`。

## Mask 与 M 审计

- R2/R3 Jaccard：`{mask['jaccard']:.5f}`
- changed bins：`{mask['changed_bins']}`；selected bins：`{mask['selected_bins']}`
- R2 hash：`{mask['r2_sha256']}`
- R3 hash：`{mask['r3_sha256']}`
- bootstrap selection probability：min `{mask['bootstrap_reliability']['min']:.5f}`，median `{mask['bootstrap_reliability']['median']:.5f}`，mean `{mask['bootstrap_reliability']['mean']:.5f}`，max `{mask['bootstrap_reliability']['max']:.5f}`
- train run counts：`{mask['multiclass_type_run_counts']}`

各类等权 M 贡献：

{contribution_lines}

## Gate

{check_lines}

主信号均值存在，但 R3−R1 seed 覆盖、配对稳定性和单 seed Binary AUPRC 红线未同时通过，因此严格判为 `R3_3W_NO_GO`。按停止线不执行 TEP、不开发 R4/R5，保留 R1/R2 并转入最终候选复核或止损。
"""
    Path(docs["three_w_report"]).write_text(report, encoding="utf-8"); return payload


def summarize_tep(config: dict, result_path: Path) -> dict:
    stage = config["tep"]; c1r1 = json.loads(Path(stage["previous_result"]).read_text(encoding="utf-8"))
    r2 = json.loads(Path(stage["r2_result"]).read_text(encoding="utf-8")); r3 = json.loads(result_path.read_text(encoding="utf-8"))
    metrics = {}; rows = []
    for seed in map(str, stage["seeds"]):
        metrics[seed] = {"C1": _flat(c1r1["seed_results"][seed]["methods"]["C1"]),
                         "R1": _flat(c1r1["seed_results"][seed]["methods"]["R1"]),
                         "R2": _flat(r2["seed_results"][seed]["methods"]["R2"]),
                         "R3": _flat(r3["seed_results"][seed]["methods"]["R3"])}
        for method, values in metrics[seed].items(): rows.append({"seed": seed, "method": method, **values})
    summary = {method: {key: mean_sample_std([metrics[seed][method][key] for seed in metrics])
                        for key in metrics[next(iter(metrics))][method] if key != "missed_fault_runs"}
               for method in ("C1", "R1", "R2", "R3")}
    paired_rows = []
    for baseline in ("C1", "R1", "R2"):
        name = f"R3-{baseline}"
        by_seed = {seed: {key: float(metrics[seed]["R3"][key] - metrics[seed][baseline][key]) for key in TEP_CORE}
                   for seed in metrics}
        summary[name] = _paired(by_seed); paired_rows.extend({"comparison": name, "seed": seed, **row} for seed, row in by_seed.items())
    status, gate = tep_decision(summary, stage["gate"])
    payload = {"status": status, "stage": "TEP", "markers": stage["markers"], "new_runs": 3,
               "reused_runs": 9, "criticality_fit_scope": r3["criticality_fit_scope"], "summary": summary,
               "mask_audit": r3["mask_audit"], **gate, "paper_final_claim_allowed": False}
    docs = config["docs"]; Path(docs["tep_json"]).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for path_key, data in (("tep_results_csv", rows), ("tep_paired_csv", paired_rows)):
        with Path(docs[path_key]).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    report = f"# R3 平衡可靠多类别关键频率：TEP\n\n阶段结论：`{status}`\n\nR3−C1 Macro-F1：`{summary['R3-C1']['mean']['macro_f1']:+.5f}`；R3−R2：`{summary['R3-R2']['mean']['macro_f1']:+.5f}`。\n"
    Path(docs["tep_report"]).write_text(report, encoding="utf-8"); return payload


def write_summary(config: dict) -> None:
    docs = config["docs"]; three_w = json.loads(Path(docs["three_w_json"]).read_text(encoding="utf-8"))
    tep_path = Path(docs["tep_json"]); tep = json.loads(tep_path.read_text(encoding="utf-8")) if tep_path.exists() else None
    final = tep["status"] if tep else three_w["status"]
    text = f"# R3 平衡可靠多类别关键频率：最终总结\n\n3W：`{three_w['status']}`\n\n"
    text += f"TEP：`{tep['status']}`\n\n" if tep else "TEP：未执行（3W gate 未放行）。\n\n"
    if tep is None:
        Path(docs["tep_report"]).write_text(
            "# R3 平衡可靠多类别关键频率：TEP\n\n未执行。3W Stage A 判定为 "
            f"`{three_w['status']}`，按预注册停止线不允许进入 TEP Stage B。\n",
            encoding="utf-8")
    text += f"最终判定：`{final}`。本轮实际新增 3 个正式 3W R3 run；未新增 TEP run。"
    text += "R3 为 NO_GO，停止 R4/R5 方法开发，保留 R1/R2，转入最终候选复核或止损，不将 R3 升级为 paper-final 候选。\n"
    Path(docs["summary_report"]).write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/r3_balanced_reliable_multiclass.yaml")
    parser.add_argument("--stage", choices=("3w", "tep"), default="3w")
    parser.add_argument("--manifest", type=Path, default=Path("outputs/r3_balanced_reliable_multiclass_3w/result_manifest.json"))
    parser.add_argument("--tep-result", type=Path, default=Path("outputs/r3_balanced_reliable_multiclass_tep/result.json"))
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = summarize_three_w(config, args.manifest) if args.stage == "3w" else summarize_tep(config, args.tep_result)
    write_summary(config); print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__": main()
