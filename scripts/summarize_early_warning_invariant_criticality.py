from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from frequency import mask_jaccard
from scripts.summarize_hierarchical_fault_semantic_criticality import _distribution


def _paired(rows: dict[str, dict[str, float]]) -> dict:
    result = {"by_seed": rows}
    for key in next(iter(rows.values())):
        values = [row[key] for row in rows.values()]
        result[key] = {**_distribution(values), "positive_seeds": sum(value > 0 for value in values),
                       "negative_seeds": sum(value < 0 for value in values)}
    return result


def _hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _criticality_audit(record: dict) -> dict:
    r1_hard = np.asarray(record["r1_hard_mask"] if "r1_hard_mask" in record else record["r1"]["composite_mask"], bool)
    hard = np.asarray(record["hard_mask"], bool); reliability = np.asarray(record["early_reliability"], float)
    r1_composite = np.asarray(record["r1_composite"] if "r1_composite" in record else record["r1"]["composite"], float)
    composite = np.asarray(record["composite"], float); soft = np.asarray(record["soft_mask"], float)
    r1_soft = np.asarray(record["r1_soft_mask"] if "r1_soft_mask" in record else record["r1"]["soft_mask"], float)
    horizons = np.asarray(record["horizon_normalized"], float)
    early = horizons[:2].mean(0); late = horizons[-2:].mean(0); difference = early - late
    def bins(values, largest=True):
        flat = values.reshape(-1); indices = np.argsort(flat)[-10:] if largest else np.argsort(flat)[:10]
        return [{"channel": int(np.unravel_index(index, values.shape)[0]),
                 "frequency_bin": int(np.unravel_index(index, values.shape)[1]), "score": float(flat[index])}
                for index in indices[::-1 if largest else 1]]
    lead = np.asarray(record["early_lead"], float); lead_top = lead >= np.partition(lead.reshape(-1), -max(1, int(round(lead.size * .3))))[-max(1, int(round(lead.size * .3)))]
    filtered = lead_top & (reliability < .5)
    return {"r1_mask_sha256": _hash(r1_hard), "ewic_mask_sha256": _hash(hard),
            "hard_mask_jaccard": mask_jaccard(r1_hard, hard), "changed_bins": int(np.logical_xor(r1_hard, hard).sum()),
            "soft_mask_mean_absolute_difference": float(np.mean(np.abs(soft - r1_soft))),
            "composite_correlation": float(np.corrcoef(r1_composite.reshape(-1), composite.reshape(-1))[0, 1]),
            "reliability": {**_distribution(reliability.reshape(-1).tolist()),
                            "below_half_fraction": float(np.mean(reliability < .5)),
                            "lead_top_bins_filtered_below_half": int(filtered.sum())},
            "bootstrap_overlap": _distribution(list(map(float, record["bootstrap_overlap"]))),
            "early_h1_h2_dominant_bins": bins(difference, True), "late_h7_h8_dominant_bins": bins(difference, False),
            "horizon_coverage": record["horizon_coverage"], "lead_weights": record["lead_weights"]}


def _three_flat(value: dict) -> dict[str, float]:
    metrics = value["standard"]["metrics"]
    return {"binary_auprc": metrics["auprc_fault_vs_normal"], "far": metrics["far"],
            "early_recall": metrics["early_recall"], "delay": metrics["mean_detection_delay_seconds"],
            "fault_recall": metrics["fault_recall"], "macro_f1": metrics["macro_f1"],
            "multiclass_auprc": metrics["auprc_multiclass_macro"]}


def _tep_flat(value: dict) -> dict[str, float]:
    metrics = value["standard"]["metrics"]
    return {"binary_auprc": metrics["auprc"], "far": metrics["far"],
            "early_recall": value["standard"]["early_recall"],
            "delay": value["standard"]["detection_delay"]["mean_delay_samples"],
            "fault_recall": metrics["fault_recall"], "macro_f1": metrics["macro_f1"]}


def _fixed_flat(value: dict, dataset: str, op: str) -> dict[str, float]:
    row = value["fixed_far"][op]
    delay = row["mean_detection_delay_seconds"] if dataset == "3w" else row["detection_delay"]["mean_delay_samples"]
    return {"early_recall": row["early_recall"], "delay": delay, "fault_recall": row["fault_recall"],
            "observed_far": row["observed_far"], "threshold": row["threshold"]}


def summarize_three_w(config: dict) -> tuple[dict, list[dict], list[dict]]:
    stage = config["three_w"]; evaluation = json.loads((Path(stage["output_dir"]) / "evaluation.json").read_text(encoding="utf-8"))
    metrics = {seed: {method: _three_flat(value) for method, value in methods.items()}
               for seed, methods in evaluation["evaluations"].items()}
    comparisons = {}; paired_rows = []
    for baseline in ("R1", "UNIFORM"):
        rows = {seed: {key: metrics[seed]["EWIC"][key] - metrics[seed][baseline][key] for key in metrics[seed]["EWIC"]}
                for seed in metrics}
        comparisons[f"EWIC-{baseline}"] = _paired(rows)
        paired_rows.extend({"dataset": "3W", "comparison": f"EWIC-{baseline}", "seed": seed, **row} for seed, row in rows.items())
    fixed = {}; horizon_rows = []
    for op in ("far_1pct", "far_5pct"):
        fixed[op] = {}
        for baseline in ("R1", "UNIFORM"):
            rows = {}
            for seed, methods in evaluation["evaluations"].items():
                current = _fixed_flat(methods["EWIC"], "3w", op); base = _fixed_flat(methods[baseline], "3w", op)
                rows[seed] = {key: current[key] - base[key] for key in ("early_recall", "delay", "fault_recall", "observed_far")}
            fixed[op][f"EWIC-{baseline}"] = _paired(rows)
    per_class = {}; per_well = {}
    for kind in (2, 8, 9):
        early_delta = []; delay_delta = []
        for seed, methods in evaluation["evaluations"].items():
            current = methods["EWIC"]["standard"]["per_class"][str(kind)]
            base = methods["R1"]["standard"]["per_class"][str(kind)]
            early_delta.append(current["early_recall"] - base["early_recall"])
            if current["mean_delay_seconds"] is not None and base["mean_delay_seconds"] is not None:
                delay_delta.append(current["mean_delay_seconds"] - base["mean_delay_seconds"])
        per_class[str(kind)] = {"early_recall_delta": _distribution(early_delta),
                                "delay_delta": _distribution(delay_delta) if delay_delta else None}
    for seed, methods in evaluation["evaluations"].items():
        for well, current in methods["EWIC"]["standard"]["per_well"].items():
            base = methods["R1"]["standard"]["per_well"].get(well, {})
            per_well.setdefault(well, []).append({"seed": seed,
                "early_recall_delta": None if current["early_recall"] is None or base.get("early_recall") is None else current["early_recall"] - base["early_recall"],
                "delay_delta": None if current["mean_delay_seconds"] is None or base.get("mean_delay_seconds") is None else current["mean_delay_seconds"] - base["mean_delay_seconds"]})
    for seed, methods in evaluation["evaluations"].items():
        for method, value in methods.items():
            for horizon, row in value["standard"]["horizon_profile"]["horizons"].items():
                horizon_rows.append({"dataset": "3W", "seed": seed, "method": method, "horizon": horizon, **row})
    delta = comparisons["EWIC-R1"]; fixed_advantage = all(
        fixed[op]["EWIC-R1"]["early_recall"]["positive_seeds"] >= 2
        and fixed[op]["EWIC-R1"]["delay"]["negative_seeds"] >= 2 for op in fixed)
    class_coverage = sum(row["early_recall_delta"]["mean"] > 0 for row in per_class.values())
    checks = {"early_recall_majority_positive": delta["early_recall"]["positive_seeds"] >= 2,
              "delay_majority_shorter": delta["delay"]["negative_seeds"] >= 2,
              "far_not_systematically_worse": delta["far"]["mean"] <= .02,
              "fixed_far_direction_consistent": fixed_advantage,
              "not_single_fault_class": class_coverage >= 2}
    payload = {"stage": "3W", "supported": all(checks.values()), "checks": checks,
               "metrics": metrics, "comparisons": comparisons, "fixed_far": fixed,
               "per_class": per_class, "per_well": per_well, "mask_audit": _criticality_audit(evaluation["criticality"])}
    return payload, paired_rows, horizon_rows


def summarize_tep(config: dict) -> tuple[dict, list[dict], list[dict], list[dict]]:
    stage = config["tep"]; evaluation = json.loads((Path(stage["output_dir"]) / "evaluation.json").read_text(encoding="utf-8"))
    metrics = {seed: {method: _tep_flat(value) for method, value in methods.items()}
               for seed, methods in evaluation["evaluations"].items()}
    comparisons = {}; paired_rows = []
    for baseline in ("R1", "C1"):
        rows = {seed: {key: metrics[seed]["EWIC"][key] - metrics[seed][baseline][key] for key in metrics[seed]["EWIC"]}
                for seed in metrics}
        comparisons[f"EWIC-{baseline}"] = _paired(rows)
        paired_rows.extend({"dataset": "TEP", "comparison": f"EWIC-{baseline}", "seed": seed, **row} for seed, row in rows.items())
    fixed = {}
    for op in ("far_1pct", "far_5pct"):
        fixed[op] = {}
        for baseline in ("R1", "C1"):
            rows = {}
            for seed, methods in evaluation["evaluations"].items():
                current = _fixed_flat(methods["EWIC"], "tep", op); base = _fixed_flat(methods[baseline], "tep", op)
                rows[seed] = {key: current[key] - base[key] for key in ("early_recall", "delay", "fault_recall", "observed_far")}
            fixed[op][f"EWIC-{baseline}"] = _paired(rows)
    per_fault = {}; fault_rows = []; improved_early = []; shorter_delay = []; degraded = []
    for kind in range(1, 21):
        early = []; delay = []
        for seed, methods in evaluation["evaluations"].items():
            current = methods["EWIC"]["standard"]["per_fault"][str(kind)]
            base = methods["R1"]["standard"]["per_fault"][str(kind)]
            early.append(current["early_recall"] - base["early_recall"])
            current_delay_delta = None
            if current["detection_delay"] is not None and base["detection_delay"] is not None:
                current_delay_delta = current["detection_delay"] - base["detection_delay"]
                delay.append(current_delay_delta)
            fault_rows.append({"fault": kind, "seed": seed, "early_recall_delta": early[-1],
                               "delay_delta": current_delay_delta})
        per_fault[str(kind)] = {"early_recall_delta": _distribution(early),
                                "delay_delta": _distribution(delay) if delay else None}
        if np.mean(early) > 0: improved_early.append(kind)
        if delay and np.mean(delay) < 0: shorter_delay.append(kind)
        if np.mean(early) < -.05 or (delay and np.mean(delay) > 16): degraded.append(kind)
    horizon_rows = []
    for seed, methods in evaluation["evaluations"].items():
        for method, value in methods.items():
            for horizon, row in value["standard"]["horizon_profile"]["horizons"].items():
                horizon_rows.append({"dataset": "TEP", "seed": seed, "method": method, "horizon": horizon, **row})
    delta = comparisons["EWIC-R1"]
    fixed_consistent = all(fixed[op]["EWIC-R1"]["early_recall"]["positive_seeds"] >= 2
                           and fixed[op]["EWIC-R1"]["delay"]["negative_seeds"] >= 2 for op in fixed)
    checks = {"early_recall_at_least_2of3_positive": delta["early_recall"]["positive_seeds"] >= 2,
              "delay_at_least_2of3_shorter": delta["delay"]["negative_seeds"] >= 2,
              "far_not_worse": delta["far"]["mean"] <= .01,
              "fault_improvement_broad": len(improved_early) >= 11 and len(shorter_delay) >= 11,
              "fixed_far_direction_consistent": fixed_consistent}
    payload = {"stage": "TEP", "supported": all(checks.values()), "checks": checks,
               "metrics": metrics, "comparisons": comparisons, "fixed_far": fixed, "per_fault": per_fault,
               "early_recall_improved_faults": improved_early, "delay_shortened_faults": shorter_delay,
               "materially_degraded_faults": degraded, "mask_audit": _criticality_audit(evaluation["criticality"])}
    return payload, paired_rows, horizon_rows, fault_rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def write_reports(config: dict, three: dict, tep: dict) -> dict:
    if three["supported"] and tep["supported"]: status = "EWIC_GO"
    elif three["supported"] and tep["checks"]["early_recall_at_least_2of3_positive"]: status = "EWIC_PARTIAL_GO"
    else: status = "EWIC_NO_GO"
    docs = config["docs"]; a = three["comparisons"]["EWIC-R1"]; b = tep["comparisons"]["EWIC-R1"]
    def seeds(comp, seeds):
        return "\n".join(f"| {seed} | {comp['by_seed'][str(seed)]['early_recall']:+.5f} | {comp['by_seed'][str(seed)]['far']:+.5f} | {comp['by_seed'][str(seed)]['delay']:+.2f} |" for seed in seeds)
    fixed3 = "\n".join(f"| {op} | {three['fixed_far'][op]['EWIC-R1']['early_recall']['mean']:+.5f} | {three['fixed_far'][op]['EWIC-R1']['delay']['mean']:+.2f} | {three['fixed_far'][op]['EWIC-R1']['observed_far']['mean']:+.5f} |" for op in ("far_1pct", "far_5pct"))
    fixedt = "\n".join(f"| {op} | {tep['fixed_far'][op]['EWIC-R1']['early_recall']['mean']:+.5f} | {tep['fixed_far'][op]['EWIC-R1']['delay']['mean']:+.2f} | {tep['fixed_far'][op]['EWIC-R1']['observed_far']['mean']:+.5f} |" for op in ("far_1pct", "far_5pct"))
    early_bins_3w = ", ".join(f"c{row['channel']}/f{row['frequency_bin']}" for row in three["mask_audit"]["early_h1_h2_dominant_bins"][:5])
    late_bins_3w = ", ".join(f"c{row['channel']}/f{row['frequency_bin']}" for row in three["mask_audit"]["late_h7_h8_dominant_bins"][:5])
    early_bins_tep = ", ".join(f"c{row['channel']}/f{row['frequency_bin']}" for row in tep["mask_audit"]["early_h1_h2_dominant_bins"][:5])
    late_bins_tep = ", ".join(f"c{row['channel']}/f{row['frequency_bin']}" for row in tep["mask_audit"]["late_h7_h8_dominant_bins"][:5])
    three_text = f"""# EWIC 3W Early Detection 报告

结论：`{'3W_EWIC_GO' if three['supported'] else '3W_EWIC_NO_GO'}`。

## 方法

EWIC 保留 R1 的 run/WELL-level `D` 与 `S`，仅把单一 `E=Fisher(Normal, Early)` 替换为八个 onset-relative `E_h`。每个 `E_h` 独立 Median/IQR robust normalization，按归一化后的 `exp(-0.35(h-1))` 加权得到 `E_lead`；随后用 64 次 train-WELL bootstrap 的 Top-30% selection probability 构造 `E_invariant=E_lead×R_early`。最终仍为 `0.5D+0.3E_invariant+0.2S`。

- EWIC−R1 Early Recall：`{a['early_recall']['mean']:+.5f} ± {a['early_recall']['std']:.5f}`，{a['early_recall']['positive_seeds']}/3 positive
- EWIC−R1 FAR：`{a['far']['mean']:+.5f}`
- EWIC−R1 Detection Delay：`{a['delay']['mean']:+.2f}` 秒，{a['delay']['negative_seeds']}/3 seed 缩短
- Binary AUPRC / Fault Recall：`{a['binary_auprc']['mean']:+.5f}` / `{a['fault_recall']['mean']:+.5f}`

| Seed | Δ Early Recall | Δ FAR | Δ Delay(s) |
|---:|---:|---:|---:|
{seeds(a, config['three_w']['seeds'])}

| Fixed-FAR OP | Δ Early Recall | Δ Delay(s) | Δ observed test FAR |
|---|---:|---:|---:|
{fixed3}

Fault 2/8/9 与 WELL 分布见 JSON。h1–h2 优势最明显的 bins：{early_bins_3w}；只在 h7–h8 更强的 bins：{late_bins_3w}。Reliability 将 `{three['mask_audit']['reliability']['lead_top_bins_filtered_below_half']}` 个原始 lead Top-30% bins 压到 `<0.5`。Mask Jaccard=`{three['mask_audit']['hard_mask_jaccard']:.5f}`，changed bins=`{three['mask_audit']['changed_bins']}`。Gate：`{three['checks']}`。
"""
    tep_text = f"""# EWIC TEP Early Detection 报告

结论：`{'TEP_EWIC_GO' if tep['supported'] else 'TEP_EWIC_NO_GO'}`。

下游始终是 binary fault detection；20-fault 信息仅用于 test profile 分组，没有训练 multiclass classifier。`E_h`、lead weighting、run bootstrap reliability 与 0.5/0.3/0.2 composite 定义和 3W 完全相同。

- EWIC−R1 Early Recall：`{b['early_recall']['mean']:+.5f}`，{b['early_recall']['positive_seeds']}/3 positive
- EWIC−R1 FAR：`{b['far']['mean']:+.5f}`
- EWIC−R1 Detection Delay：`{b['delay']['mean']:+.2f}` samples，{b['delay']['negative_seeds']}/3 seed 缩短
- Binary AUPRC / Fault Recall：`{b['binary_auprc']['mean']:+.5f}` / `{b['fault_recall']['mean']:+.5f}`
- Early Recall 改善 faults：`{tep['early_recall_improved_faults']}`
- Delay 缩短 faults：`{tep['delay_shortened_faults']}`
- 明显退化 faults：`{tep['materially_degraded_faults']}`

| Seed | Δ Early Recall | Δ FAR | Δ Delay(samples) |
|---:|---:|---:|---:|
{seeds(b, config['tep']['seeds'])}

| Fixed-FAR OP | Δ Early Recall | Δ Delay | Δ observed test FAR |
|---|---:|---:|---:|
{fixedt}

h1–h2 优势 bins：{early_bins_tep}；h7–h8 优势 bins：{late_bins_tep}。Reliability 过滤 `{tep['mask_audit']['reliability']['lead_top_bins_filtered_below_half']}` 个 lead Top-30% bins。Mask Jaccard=`{tep['mask_audit']['hard_mask_jaccard']:.5f}`，changed bins=`{tep['mask_audit']['changed_bins']}`。Gate：`{tep['checks']}`。
"""
    Path(docs["three_w_report"]).write_text(three_text, encoding="utf-8")
    Path(docs["tep_report"]).write_text(tep_text, encoding="utf-8")
    summary = {"status": status, "three_w_supported": three["supported"], "tep_supported": tep["supported"],
               "new_runs": 6, "reused_runs": 12, "freeze_r1": status == "EWIC_NO_GO", "paper_final_allowed": False}
    decision = ("EWIC 可进入下一阶段验证。" if status != "EWIC_NO_GO" else
                "两个数据集未形成稳定 Early Detection 收益；停止方法开发，正式冻结 R1，进入 baseline、ablation 与 paper-final protocol。")
    Path(docs["summary"]).write_text(f"# EWIC 双数据集总结\n\n最终判定：`{status}`。\n\n{decision}\n", encoding="utf-8")
    Path(docs["mask_audit"]).write_text(json.dumps({"three_w": three["mask_audit"], "tep": tep["mask_audit"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/early_warning_invariant_criticality.yaml")
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    three, paired3, horizon3 = summarize_three_w(config); tep, pairedt, horizont, faults = summarize_tep(config)
    docs = config["docs"]; Path(docs["three_w_json"]).write_text(json.dumps(three, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(docs["tep_json"]).write_text(json.dumps(tep, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(Path(docs["paired_csv"]), paired3 + pairedt); _write_csv(Path(docs["horizon_csv"]), horizon3 + horizont)
    _write_csv(Path(docs["fault_csv"]), faults)
    print(json.dumps(write_reports(config, three, tep), ensure_ascii=False))


if __name__ == "__main__": main()
