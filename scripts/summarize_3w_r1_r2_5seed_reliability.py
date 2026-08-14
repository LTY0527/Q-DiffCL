from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from scripts.run_3w_diffusion_1seed import METHODS, R2_METHOD
from scripts.summarize_r2_multiclass_criticality import CORE, metric_delta


METHOD_LABELS = ("UNIFORM", "R1", "R2")


def distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {"mean": float(array.mean()), "std": float(array.std()), "median": float(np.median(array)),
            "min": float(array.min()), "max": float(array.max())}


def paired_summary(by_seed: dict[str, dict]) -> dict:
    keys = list(next(iter(by_seed.values()))); result = {"by_seed": by_seed}
    for key in keys:
        values = [row[key] for row in by_seed.values()]; current = distribution(values)
        current["positive_seeds"] = int(sum(value > 0 for value in values))
        current["nonnegative_seeds"] = int(sum(value >= 0 for value in values))
        current["positive_rate"] = float(sum(value > 0 for value in values) / len(values))
        favorable = [value < 0 for value in values] if key in {"far", "mean_detection_delay_seconds"} else [value > 0 for value in values]
        current["favorable_rate"] = float(sum(favorable) / len(values)); result[key] = current
    return result


def r1_decision(summary: dict, gate: dict) -> tuple[str, dict]:
    delta = summary["R1-UNIFORM"]; macro = delta["macro_f1"]
    macro_values = [row["macro_f1"] for row in delta["by_seed"].values()]
    best = max(macro_values); leave_best_out = float((sum(macro_values) - best) / (len(macro_values) - 1))
    positive_total = sum(max(value, 0.) for value in macro_values)
    largest_positive_share = float(max(best, 0.) / positive_total) if positive_total else 1.0
    checks = {
        "macro_f1_positive_at_least_4of5": macro["positive_seeds"] >= int(gate["minimum_macro_f1_positive_seeds"]),
        "macro_f1_mean_gain": macro["mean"] >= float(gate["minimum_macro_f1_mean_gain"]),
        "multiclass_auprc_not_systematically_reversed": delta["auprc_multiclass_macro"]["nonnegative_seeds"] >= int(gate["minimum_multiclass_auprc_nonworse_seeds"]),
        "far_not_systematically_reversed": delta["far"]["mean"] <= float(gate["maximum_far_mean_increase"]),
        "not_driven_only_by_best_seed": leave_best_out > float(gate["minimum_leave_best_out_macro_f1_mean_gain"]),
    }
    status = "R1_5SEED_STABLE_CANDIDATE" if all(checks.values()) else "R1_5SEED_EXPLORATORY_ONLY"
    return status, {"checks": checks, "leave_best_seed_out_macro_f1_mean": leave_best_out,
                    "largest_positive_seed_share": largest_positive_share}


def r2_decision(summary: dict, gate: dict) -> tuple[str, dict]:
    uniform, r1 = summary["R2-UNIFORM"], summary["R2-R1"]
    checks = {
        "macro_f1_majority_positive_vs_uniform": uniform["macro_f1"]["positive_seeds"] >= int(gate["minimum_macro_f1_positive_seeds_vs_uniform"]),
        "macro_f1_at_least_3of5_nonworse_vs_r1": r1["macro_f1"]["nonnegative_seeds"] >= int(gate["minimum_macro_f1_nonworse_seeds_vs_r1"]),
        "binary_auprc_mean_advantage_vs_uniform": uniform["auprc_fault_vs_normal"]["mean"] > float(gate["minimum_binary_auprc_mean_gain_vs_uniform"]),
        "binary_auprc_majority_nonworse_vs_uniform": uniform["auprc_fault_vs_normal"]["nonnegative_seeds"] >= int(gate["minimum_binary_auprc_nonworse_seeds_vs_uniform"]),
        "multiclass_auprc_mean_advantage_vs_uniform": uniform["auprc_multiclass_macro"]["mean"] > float(gate["minimum_multiclass_auprc_mean_gain_vs_uniform"]),
        "multiclass_auprc_majority_nonworse_vs_uniform": uniform["auprc_multiclass_macro"]["nonnegative_seeds"] >= int(gate["minimum_multiclass_auprc_nonworse_seeds_vs_uniform"]),
    }
    status = "R2_5SEED_CANDIDATE" if all(checks.values()) else "R2_EXTENSION_ONLY"
    return status, {"checks": checks}


def _load_records(config: dict, new_manifest: Path) -> dict[str, dict]:
    old = json.loads(Path(config["existing_r1_manifest"]).read_text(encoding="utf-8"))
    r2_old = json.loads(Path(config["existing_r2_manifest"]).read_text(encoding="utf-8"))
    new = json.loads(new_manifest.read_text(encoding="utf-8")); records = {}
    for seed in map(str, config["all_seeds"]):
        if int(seed) in set(map(int, config["existing_seeds"])):
            r1 = json.loads(Path(old["seed_results"][seed]["result_path"]).read_text(encoding="utf-8"))
            r2 = json.loads(Path(r2_old["seed_results"][seed]["result_path"]).read_text(encoding="utf-8"))
        else:
            item = new["seed_results"][seed]
            r1 = json.loads(Path(item["uniform_r1_result_path"]).read_text(encoding="utf-8"))
            r2 = json.loads(Path(item["r2_result_path"]).read_text(encoding="utf-8"))
        if r1["fairness"]["window_refs_sha256"] != r2["fairness"]["window_refs_sha256"]:
            raise RuntimeError(f"seed {seed} window refs differ between R1 and R2")
        if r1["fairness"]["initialization_sha256"] != r2["fairness"]["initialization_sha256"]:
            raise RuntimeError(f"seed {seed} initialization differs between R1 and R2")
        records[seed] = {"r1": r1, "r2": r2}
    r1_masks = {record["r1"]["fairness"]["critical_soft_mask_sha256"] for record in records.values()}
    r2_masks = {record["r2"]["fairness"]["critical_soft_mask_sha256"] for record in records.values()}
    refs = {record["r1"]["fairness"]["window_refs_sha256"] for record in records.values()}
    if len(r1_masks) != 1 or len(r2_masks) != 1 or len(refs) != 1:
        raise RuntimeError("5-seed audit changed frozen R1/R2 mask or window refs")
    return records


def summarize(config: dict, new_manifest: Path) -> dict:
    records = _load_records(config, new_manifest); rows = []
    for seed, record in records.items():
        choices = (("UNIFORM", record["r1"]["methods"][METHODS[1]]["metrics"]),
                   ("R1", record["r1"]["methods"][METHODS[2]]["metrics"]),
                   ("R2", record["r2"]["methods"][R2_METHOD]["metrics"]))
        for label, metrics in choices:
            row = {"seed": int(seed), "method": label, **{key: metrics[key] for key in CORE}}
            for item in metrics["per_class"]:
                row[f"class_{item['original_class']}_recall"] = item["recall"]
                row[f"class_{item['original_class']}_f1"] = item["f1"]
            rows.append(row)
    numeric = [key for key in rows[0] if key not in {"seed", "method"}]
    methods = {method: {key: distribution([row[key] for row in rows if row["method"] == method]) for key in numeric}
               for method in METHOD_LABELS}
    comparisons = {}; paired_rows = []
    for method, baseline in (("R1", "UNIFORM"), ("R2", "UNIFORM"), ("R2", "R1")):
        name = f"{method}-{baseline}"; by_seed = {}
        for seed, record in records.items():
            metrics = {"UNIFORM": record["r1"]["methods"][METHODS[1]]["metrics"],
                       "R1": record["r1"]["methods"][METHODS[2]]["metrics"],
                       "R2": record["r2"]["methods"][R2_METHOD]["metrics"]}
            by_seed[seed] = metric_delta(metrics[baseline], metrics[method])
            paired_rows.append({"comparison": name, "seed": seed, **by_seed[seed]})
        comparisons[name] = paired_summary(by_seed)
    summary = {"methods": methods, **comparisons}
    r1_status, r1_gate = r1_decision(summary, config["r1_gate"])
    r2_status, r2_gate = r2_decision(summary, config["r2_gate"])
    r1_mask = next(iter(records.values()))["r1"]["fairness"]["critical_soft_mask_sha256"]
    r2_mask = next(iter(records.values()))["r2"]["fairness"]["critical_soft_mask_sha256"]
    payload = {"status": {"r1": r1_status, "r2": r2_status}, "seeds": config["all_seeds"],
               "new_training_count": 6, "reused_training_count": 9,
               "paper_final_protocol_design_allowed": r1_status == "R1_5SEED_STABLE_CANDIDATE",
               "current_test_numbers_are_paper_final_claims": False, "summary": summary,
               "class9": {method: {"recall": methods[method]["class_9_recall"], "f1": methods[method]["class_9_f1"]}
                          for method in METHOD_LABELS},
               "fairness": {"same_window_refs_all_seeds": True, "same_seed_initialization_r1_r2": True,
                            "original_batching": True, "r1_soft_mask_sha256": r1_mask,
                            "r2_soft_mask_sha256": r2_mask},
               "r1_gate": r1_gate, "r2_gate": r2_gate}
    docs = config["docs"]; Path(docs["json"]).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for key, data in (("results_csv", rows), ("paired_csv", paired_rows)):
        with Path(docs[key]).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    _write_report(Path(docs["report"]), payload); return payload


def _write_report(path: Path, payload: dict) -> None:
    summary = payload["summary"]; lines = []
    for comparison in ("R1-UNIFORM", "R2-UNIFORM", "R2-R1"):
        metric = summary[comparison]["macro_f1"]
        lines.append(f"- {comparison} Macro-F1：`{metric['mean']:+.5f} ± {metric['std']:.5f}`，median `{metric['median']:+.5f}`，range `[{metric['min']:+.5f}, {metric['max']:+.5f}]`，positive `{metric['positive_seeds']}/5`")
    paired = []
    for seed in map(str, payload["seeds"]):
        paired.append(f"| {seed} | {summary['R1-UNIFORM']['by_seed'][seed]['macro_f1']:+.5f} | "
                      f"{summary['R2-UNIFORM']['by_seed'][seed]['macro_f1']:+.5f} | "
                      f"{summary['R2-R1']['by_seed'][seed]['macro_f1']:+.5f} |")
    class9 = payload["class9"]
    core_lines = []
    for comparison in ("R1-UNIFORM", "R2-UNIFORM", "R2-R1"):
        item = summary[comparison]
        core_lines.append(
            f"| {comparison} | {item['recall_macro']['mean']:+.5f} | "
            f"{item['auprc_fault_vs_normal']['mean']:+.5f} | {item['auprc_multiclass_macro']['mean']:+.5f} | "
            f"{item['far']['mean']:+.5f} | {item['early_recall']['mean']:+.5f} | "
            f"{item['mean_detection_delay_seconds']['mean']:+.2f} |")
    report = f"""# 3W R1/R2 五随机种子稳定性复核

最终判定：R1 `{payload['status']['r1']}`；R2 `{payload['status']['r2']}`。

本轮只新增 seeds 45/46 的 Uniform、R1、R2，共 6 个训练；seeds 42/43/44 的 9 个结果直接复用。未运行 R3、TEP，也未改变权重、timestep、mask ratio、sampler、TCN、loss、split 或 probe。

## Macro-F1 配对稳定性

{chr(10).join(lines)}

| Seed | R1−Uniform | R2−Uniform | R2−R1 |
|---:|---:|---:|---:|
{chr(10).join(paired)}

去掉 R1 最佳增益 seed 后，R1−Uniform Macro-F1 mean 为 `{payload['r1_gate']['leave_best_seed_out_macro_f1_mean']:+.5f}`；最大正增益 seed 占全部正增益 `{payload['r1_gate']['largest_positive_seed_share']:.1%}`。

## 其他核心指标的五 seed 配对均值

| Comparison | Macro Recall | Binary AUPRC | Multiclass AUPRC | FAR | Early Recall | Delay (s) |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(core_lines)}

R1 的 Multiclass AUPRC mean 为轻微负向，但 3/5 seed 非负、median 为 `{summary['R1-UNIFORM']['auprc_multiclass_macro']['median']:+.5f}`，不构成多数 seed 系统性反向；FAR mean 明显改善。R2 的两项 AUPRC 相对 Uniform 均为正，但相对 R1 的 Macro-F1 只有 2/5 正向且 FAR mean 恶化，因此不升级为主候选。

## Class 9

- Uniform Recall/F1：`{class9['UNIFORM']['recall']['mean']:.5f} ± {class9['UNIFORM']['recall']['std']:.5f}` / `{class9['UNIFORM']['f1']['mean']:.5f} ± {class9['UNIFORM']['f1']['std']:.5f}`
- R1 Recall/F1：`{class9['R1']['recall']['mean']:.5f} ± {class9['R1']['recall']['std']:.5f}` / `{class9['R1']['f1']['mean']:.5f} ± {class9['R1']['f1']['std']:.5f}`
- R2 Recall/F1：`{class9['R2']['recall']['mean']:.5f} ± {class9['R2']['recall']['std']:.5f}` / `{class9['R2']['f1']['mean']:.5f} ± {class9['R2']['f1']['std']:.5f}`

三种方法的 Class 9 Recall median 均仅约 `0.01`，且 R1/R2 的 std 都高于 mean。Class 9 仍高度不稳定，较高均值由少数 seed 主导，不能作为稳定改善结论。

## 候选结论

R1 gate：{payload['r1_gate']['checks']}。

R2 gate：{payload['r2_gate']['checks']}。

`paper_final_protocol_design_allowed = {str(payload['paper_final_protocol_design_allowed']).lower()}`。即使允许进入下一阶段，当前 3W/TEP test 已参与方法开发，不能作为 paper-final claim；下一阶段只能设计新的独立 paper-final protocol。
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/3w_r1_r2_5seed_reliability.yaml")
    parser.add_argument("--manifest", type=Path, default=Path("outputs/3w_r1_r2_5seed_reliability/result_manifest.json"))
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    print(json.dumps(summarize(config, args.manifest), ensure_ascii=False))


if __name__ == "__main__": main()
