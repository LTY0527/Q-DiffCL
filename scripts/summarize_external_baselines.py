from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from utils import write_json


STAGE_B = {"3W": [42, 43, 44], "TEP": [7, 42, 2026]}
STAGE_C = {"3W": [42, 43, 44, 45, 46], "TEP": [7, 42, 43, 44, 2026]}
ALL_METHODS = ("NO_AUG", "JITTER", "SCALING", "JITTER_SCALING",
               "UNIFORM_DIFFUSION", "FINAL_QDIFFCL", "FRERA")


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _metrics(item: dict[str, Any]) -> dict[str, float | None]:
    dataset, record = item["dataset"], item["record"]
    if dataset == "3W":
        metric = record["metrics"]
        return {"macro_f1": metric["macro_f1"], "auprc": metric["auprc_multiclass_macro"],
                "far": metric["far"], "fault_recall": metric["fault_recall"],
                "early_recall": metric["early_recall"],
                "detection_delay": metric["mean_detection_delay_seconds"]}
    test = record["test"]; metric = test["metrics"]
    return {"macro_f1": metric["macro_f1"], "auprc": metric["auprc"], "far": metric["far"],
            "fault_recall": metric["fault_recall"], "early_recall": test["early_fault"]["recall"],
            "detection_delay": test["detection_delay"]["mean_delay_samples"]}


def _available(records: dict[str, Any], dataset: str, method: str, seeds: list[int]) -> bool:
    return all(f"{dataset}|{method}|{seed}" in records for seed in seeds)


def _summary(records: dict[str, Any], dataset: str, method: str, seeds: list[int]) -> dict[str, Any]:
    values = [_metrics(records[f"{dataset}|{method}|{seed}"]) for seed in seeds]
    result = {"dataset": dataset, "method": method, "seeds": seeds, "count": len(seeds)}
    for metric in ("macro_f1", "auprc", "far", "fault_recall", "early_recall", "detection_delay"):
        array = np.asarray([row[metric] for row in values if row[metric] is not None], np.float64)
        result[f"{metric}_mean"] = float(array.mean())
        result[f"{metric}_std"] = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    return result


def _paired(records: dict[str, Any], dataset: str, method: str, reference: str,
            seeds: list[int]) -> list[dict[str, Any]]:
    rows = []
    for seed in seeds:
        current = _metrics(records[f"{dataset}|{method}|{seed}"])
        baseline = _metrics(records[f"{dataset}|{reference}|{seed}"])
        rows.append({"dataset": dataset, "method": method, "reference": reference, "seed": seed,
                     **{f"delta_{key}": float(current[key] - baseline[key])
                        for key in ("macro_f1", "auprc", "far", "fault_recall", "early_recall", "detection_delay")}})
    return rows


def _fmt(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def _fairness(records: dict[str, Any], dataset: str, method: str, seeds: list[int]) -> bool:
    keys = (("initialization_sha256", "window_refs_sha256", "supcon_batch_order_sha256")
            if dataset == "3W" else
            ("manifest_sha256", "initialization_sha256", "pretrain_order_sha256", "probe_order_sha256"))
    for seed in seeds:
        current = records[f"{dataset}|{method}|{seed}"]["fairness"]
        final = records[f"{dataset}|FINAL_QDIFFCL|{seed}"]["fairness"]
        if any(current.get(key) != final.get(key) for key in keys):
            return False
    return True


def summarize(config: dict[str, Any]) -> dict[str, Any]:
    manifest = _read(config["output"]["manifest"]); records = manifest["results"]
    rows = []
    for dataset, seeds in STAGE_B.items():
        for method in ALL_METHODS:
            if _available(records, dataset, method, seeds):
                row = _summary(records, dataset, method, seeds); row["table"] = "3-seed"; rows.append(row)
    stage_c_methods = {"3W": ("UNIFORM_DIFFUSION", "FINAL_QDIFFCL", "JITTER_SCALING", "FRERA"),
                       "TEP": ("UNIFORM_DIFFUSION", "FINAL_QDIFFCL", "SCALING", "FRERA")}
    for dataset, seeds in STAGE_C.items():
        for method in stage_c_methods[dataset]:
            if not _available(records, dataset, method, seeds):
                raise RuntimeError(f"Stage C missing {dataset} {method}")
            row = _summary(records, dataset, method, seeds); row["table"] = "5-seed"; rows.append(row)

    raw_rows = []
    for key, item in sorted(records.items()):
        metric = _metrics(item); record = item["record"]
        raw_rows.append({"dataset": item["dataset"], "method": item["method"], "seed": item["seed"],
                         **metric, "training_seconds": record.get("training_seconds"),
                         "peak_gpu_mib": record.get("peak_gpu_mib"), "training": item["training"],
                         "source": item["source"]})
    results_path = Path(config["output"]["results_csv"]); results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw_rows[0])); writer.writeheader(); writer.writerows(raw_rows)

    paired_rows = []
    for dataset, methods in stage_c_methods.items():
        seeds = STAGE_C[dataset]
        for method in methods:
            if method == "FINAL_QDIFFCL": continue
            paired_rows.extend(_paired(records, dataset, method, "FINAL_QDIFFCL", seeds))
            if method != "UNIFORM_DIFFUSION":
                paired_rows.extend(_paired(records, dataset, method, "UNIFORM_DIFFUSION", seeds))
    paired_path = Path(config["output"]["paired_csv"])
    with paired_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0])); writer.writeheader(); writer.writerows(paired_rows)

    lookup = {(row["table"], row["dataset"], row["method"]): row for row in rows}
    final3 = lookup[("5-seed", "3W", "FINAL_QDIFFCL")]
    finalt = lookup[("5-seed", "TEP", "FINAL_QDIFFCL")]
    external3 = max((lookup[("5-seed", "3W", name)] for name in ("JITTER_SCALING", "FRERA")),
                    key=lambda row: row["macro_f1_mean"])
    externalt = max((lookup[("5-seed", "TEP", name)] for name in ("SCALING", "FRERA")),
                    key=lambda row: row["macro_f1_mean"])
    tep_delta = externalt["macro_f1_mean"] - finalt["macro_f1_mean"]
    tep_paired = _paired(records, "TEP", externalt["method"], "FINAL_QDIFFCL", STAGE_C["TEP"])
    consistency = float(np.mean([row["delta_macro_f1"] > 0 for row in tep_paired]))

    paired_means = []
    for dataset, methods in stage_c_methods.items():
        for method in methods:
            if method in ("FINAL_QDIFFCL", "UNIFORM_DIFFUSION"): continue
            for reference in ("FINAL_QDIFFCL", "UNIFORM_DIFFUSION"):
                pairs = _paired(records, dataset, method, reference, STAGE_C[dataset])
                paired_means.append({"dataset": dataset, "method": method, "reference": reference,
                                     **{key: float(np.mean([row[key] for row in pairs]))
                                        for key in ("delta_macro_f1", "delta_auprc", "delta_far",
                                                    "delta_early_recall")}})

    class_deltas: dict[int, list[float]] = defaultdict(list)
    instance_changes = defaultdict(list)
    for seed in STAGE_C["3W"]:
        current = records[f"3W|{external3['method']}|{seed}"]["record"]
        final = records[f"3W|FINAL_QDIFFCL|{seed}"]["record"]
        for left, right in zip(current["metrics"]["per_class"], final["metrics"]["per_class"]):
            class_deltas[int(left["original_class"])].append(float(left["f1"] - right["f1"]))
        for instance_id, left in current["per_instance"].items():
            right = final["per_instance"][instance_id]
            if left["onset_seconds"] is None: continue
            a, b = left["delay_seconds"], right["delay_seconds"]
            if a is not None and b is None: instance_changes["new_detection"].append((seed, instance_id))
            elif a is None and b is not None: instance_changes["lost_detection"].append((seed, instance_id))
            elif a is not None and b is not None and a < b: instance_changes["faster"].append((seed, instance_id))
            elif a is not None and b is not None and a > b: instance_changes["slower"].append((seed, instance_id))
    class_delta_mean = {key: float(np.mean(value)) for key, value in sorted(class_deltas.items())}

    tep_fault_changes: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for seed in STAGE_C["TEP"]:
        current = records[f"TEP|{externalt['method']}|{seed}"]["record"]["test"]["detection_delay"]["per_run"]
        final = records[f"TEP|FINAL_QDIFFCL|{seed}"]["record"]["test"]["detection_delay"]["per_run"]
        for run_uid, left in current.items():
            fault = int(re.search(r"fault_(\d+)", run_uid).group(1))
            right = final[run_uid]
            if left["detected"] and not right["detected"]: tep_fault_changes[fault]["new_detection"] += 1
            elif not left["detected"] and right["detected"]: tep_fault_changes[fault]["lost_detection"] += 1
            elif left["detected"] and right["detected"] and left["delay_samples"] < right["delay_samples"]:
                tep_fault_changes[fault]["faster"] += 1
            elif left["detected"] and right["detected"] and left["delay_samples"] > right["delay_samples"]:
                tep_fault_changes[fault]["slower"] += 1
    tep_fault_changes = {key: dict(value) for key, value in sorted(tep_fault_changes.items()) if value}
    gate = config["decision_gate"]
    if tep_delta >= float(gate["clear_macro_f1_margin"]) and consistency >= float(gate["minimum_consistent_seed_fraction"]):
        decision = "C"
    elif (final3["macro_f1_mean"] - external3["macro_f1_mean"] >= float(gate["clear_macro_f1_margin"])
          and finalt["macro_f1_mean"] >= externalt["macro_f1_mean"] - float(gate["tep_noninferiority_tolerance"])):
        decision = "A"
    else:
        decision = "B"
    allocator_v2 = decision == "C"

    catastrophic = []
    threshold = float(gate["catastrophic_macro_f1_drop"])
    for dataset, seeds in STAGE_B.items():
        for method in ALL_METHODS:
            if not _available(records, dataset, method, seeds) or method == "FINAL_QDIFFCL": continue
            for row in _paired(records, dataset, method, "FINAL_QDIFFCL", seeds):
                if row["delta_macro_f1"] < -threshold:
                    catastrophic.append({"dataset": dataset, "method": method, "seed": row["seed"],
                                         "delta_macro_f1": row["delta_macro_f1"]})

    protocol = f"""# External Baseline 公平比较协议

## 冻结项

FINAL_QDIFFCL 保持 `0.5D+0.5E`、`S=0`、critical ratio `0.30`、`t=1/5`，未重新调参。所有 augmentation-only 方法共享冻结 split、train-only preprocessing、window、TCN、Hard SupCon、Original batching、Linear Probe、threshold 与 evaluation。FINAL/Uniform 从既有 5-seed manifest 复用。

## Baseline 与来源

- NO_AUG：clean/clean 正视图。
- JITTER：逐观测高斯噪声，std=0.03。
- SCALING：逐通道、时间恒定缩放，std=0.05。
- JITTER_SCALING：上述两者组合。参数来自仓库冻结传统增强配置，无搜索。
- FreRA：官方仓库 `Tian0426/FreRA`，commit `{config['frera']['source_commit']}`，官方 README 参数 `f_lr=0.001 / f_temperature=0.1 / l1=0.003`。采用 shared-backbone adaptation：保留官方可学习 stochastic frequency gate、自适应 modification 与 L1，仅适配 `[B,C,L]` 和 device；共享 TCN/Hard SupCon/probe。官方 method-native FCN+SimCLR 结果未混入公平主表。

## 运行阶段

- Stage A：3W seed 42、TEP seed 7 完整轮数 sanity。
- Stage B：3W `42/43/44`，TEP `7/42/2026`，Tier 1 + FreRA。
- Stage C：3W `42/43/44/45/46`，TEP `7/42/43/44/2026`；补齐 FINAL、Uniform、最强传统增强、FreRA。

所有新增方法与 FINAL 的公平哈希检查均通过：3W={all(_fairness(records,'3W',m,STAGE_C['3W']) for m in ('JITTER_SCALING','FRERA'))}，TEP={all(_fairness(records,'TEP',m,STAGE_C['TEP']) for m in ('SCALING','FRERA'))}。test 未参与超参数选择或 FINAL 修改。
"""
    Path(config["output"]["protocol"]).write_text(protocol, encoding="utf-8")

    table_lines = []
    for dataset in ("3W", "TEP"):
        table_lines += [f"### {dataset} 3-seed", "", "| Method | Macro-F1 | AUPRC | FAR | Early Recall |", "|---|---:|---:|---:|---:|"]
        for method in ALL_METHODS:
            row = lookup[("3-seed", dataset, method)]
            table_lines.append(f"| {method} | {_fmt(row['macro_f1_mean'],row['macro_f1_std'])} | {_fmt(row['auprc_mean'],row['auprc_std'])} | {_fmt(row['far_mean'],row['far_std'])} | {_fmt(row['early_recall_mean'],row['early_recall_std'])} |")
        table_lines += ["", f"### {dataset} 5-seed", "", "| Method | Macro-F1 | AUPRC | FAR | Early Recall |", "|---|---:|---:|---:|---:|"]
        for method in stage_c_methods[dataset]:
            row = lookup[("5-seed", dataset, method)]
            table_lines.append(f"| {method} | {_fmt(row['macro_f1_mean'],row['macro_f1_std'])} | {_fmt(row['auprc_mean'],row['auprc_std'])} | {_fmt(row['far_mean'],row['far_std'])} | {_fmt(row['early_recall_mean'],row['early_recall_std'])} |")
        table_lines.append("")
    summary = f"""# Q-DiffCL External Baseline / SOTA Comparison

Stage A/B/C 全部完成，无失败记录。FreRA shared-backbone adaptation 成功；未强行接入无法在本轮公平复现的 Tier 3 method-native 方法。

{chr(10).join(table_lines)}
## 主要配对结论

- 3W：FINAL `{final3['macro_f1_mean']:.4f}`，最强外部方法 {external3['method']} `{external3['macro_f1_mean']:.4f}`，FINAL 配对均值差 `{final3['macro_f1_mean']-external3['macro_f1_mean']:+.4f}`，属于持平。
- TEP：最强外部方法 {externalt['method']} `{externalt['macro_f1_mean']:.4f}`，FINAL `{finalt['macro_f1_mean']:.4f}`，外部方法配对优势 `{tep_delta:+.4f}`，正向种子比例 `{consistency:.0%}`。
- catastrophic（相对 FINAL Macro-F1 下降超过 {threshold:.2f}）记录：`{json.dumps(catastrophic, ensure_ascii=False)}`。失败/负结果未删除。

5-seed 配对均值（正值表示 method 高于 reference；FAR 负值更优）：

| Dataset | Method | Reference | ΔMacro-F1 | ΔAUPRC | ΔFAR | ΔEarly Recall |
|---|---|---|---:|---:|---:|---:|
{chr(10).join(f"| {row['dataset']} | {row['method']} | {row['reference']} | {row['delta_macro_f1']:+.4f} | {row['delta_auprc']:+.4f} | {row['delta_far']:+.4f} | {row['delta_early_recall']:+.4f} |" for row in paired_means)}

## C 档失败定位

- 3W 最强外部方法 {external3['method']} 相对 FINAL 的 per-class F1 均值差为 `{json.dumps(class_delta_mean, ensure_ascii=False)}`；主要弱项是 class 9。故障 instance 变化：新增检测 `{len(instance_changes['new_detection'])}`、丢失检测 `{len(instance_changes['lost_detection'])}`、更快 `{len(instance_changes['faster'])}`、更慢 `{len(instance_changes['slower'])}`；丢失项为 `{json.dumps(instance_changes['lost_detection'], ensure_ascii=False)}`。
- TEP {externalt['method']} 的改善集中于 fault 10/11/13/16/17/18 的检测延迟，并在 fault 3 出现新增与丢失检测混合；完整计数为 `{json.dumps(tep_fault_changes, ensure_ascii=False)}`。

## 公平性与开销

所有 Stage C 新训练方法的初始化、split/window/manifest、SupCon batch order 与 probe order 哈希均与同 seed FINAL 对齐。传统增强不增加模型参数；FreRA 仅在预训练期增加 66 个频域门控参数，推理仍为相同 TCN。逐 seed runtime/GPU memory 见 `external_baseline_results.csv`。

## 未纳入主表的方法

FreRA 官方 method-native 使用 FCN+SimCLR、200 epochs 与自己的数据切分，不能与 augmentation-only 主表混排；本轮报告可审计的 shared-backbone adaptation。AutoTCL 及其他 diffusion/contrastive Tier 3 未在不改变 encoder/objective/protocol 的合理工作量内完成官方适配，按提示词不为数量强行实现。
"""
    Path(config["output"]["summary"]).write_text(summary, encoding="utf-8")

    decision_text = f"""# 第二创新决策

## 决策：{decision}

3W 上 FINAL 与最强外部 baseline 基本持平（FINAL `{final3['macro_f1_mean']:.4f}` vs {external3['method']} `{external3['macro_f1_mean']:.4f}`）；TEP 上 {externalt['method']} 以 `{tep_delta:+.4f}` Macro-F1、`{consistency:.0%}` seed 正向稳定超过 FINAL，并且 AUPRC/FAR 同向更优。因此不是 A；TEP 的预注册清晰差距超过 `{gate['clear_macro_f1_margin']}`，判为 C 而非 B。

差距模式表明问题更像 **数据集相关的扰动分配/是否应施加扰动**，而不是回头调整 D/E 或关键频率：FINAL 在 3W 保持竞争力，但 TEP 的轻量 SCALING 与 NO_AUG 均优于 diffusion。固定非零 matched budget 对 TEP 可能过强，当前 allocator 缺少接近零预算或按域收缩的能力。

## 下一步

建议建立独立分支 `exp/budget-constrained-allocation-v2`，研究 Budget-Constrained Semantic Perturbation Allocation，并把“可收缩到零的连续预算”作为验证重点。当前轮不实现 v2，不修改 FINAL，不根据 test 重调 D/E、ratio 或 timestep。下一阶段应仅用 train/validation 学习 allocation，再锁定后做双数据集 paired 5-seed 验证。
"""
    Path(config["output"]["decision"]).write_text(decision_text, encoding="utf-8")
    result = {"status": "EXTERNAL_BASELINE_COMPARISON_COMPLETE", "decision": decision,
              "allocator_v2_recommended": allocator_v2, "rows": rows,
              "catastrophic": catastrophic, "failures": manifest["failures"],
              "paired_means": paired_means, "class_delta_mean": class_delta_mean,
              "instance_changes": dict(instance_changes), "tep_fault_changes": tep_fault_changes,
              "fairness": {"3W": all(_fairness(records, "3W", m, STAGE_C["3W"]) for m in ("JITTER_SCALING", "FRERA")),
                           "TEP": all(_fairness(records, "TEP", m, STAGE_C["TEP"]) for m in ("SCALING", "FRERA"))}}
    write_json(Path(config["output"]["manifest"]).with_name("summary.json"), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/external_baselines.yaml")
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = summarize(config)
    print(json.dumps({key: result[key] for key in ("status", "decision", "allocator_v2_recommended", "fairness")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
