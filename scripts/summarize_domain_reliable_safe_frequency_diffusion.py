from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from utils import write_json


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std())}


def _reduction(old: float, new: float) -> float:
    return float((old - new) / old) if old > 0 else (1.0 if new == 0 else float("-inf"))


def _class(metrics: dict[str, Any], original_class: int) -> dict[str, Any]:
    return next(item for item in metrics["per_class"] if int(item["original_class"]) == original_class)


def summarize(config: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(Path(config["stage_c"]["output"]).read_text(encoding="utf-8"))
    stage_a = json.loads(Path(config["stage_a"]["output"]).read_text(encoding="utf-8"))
    kill = json.loads(Path(config["stage_b"]["output"]).read_text(encoding="utf-8"))
    manifest = json.loads(Path(config["three_w"]["existing_manifest"]).read_text(encoding="utf-8"))
    three_w_r1 = {}; three_w_drfd = {}
    for seed in (42, 43, 44):
        source = json.loads(Path(manifest["seed_results"][str(seed)]["result_path"]).read_text(encoding="utf-8"))
        three_w_r1[seed] = source["methods"]["FREQUENCY_SELECTIVE_R1"]["metrics"]
        three_w_drfd[seed] = result["three_w"][str(seed)]["method"]["metrics"]
    tep_source = json.loads(Path(config["tep"]["existing_result"]).read_text(encoding="utf-8"))
    tep_r1 = {seed: tep_source["seed_results"][str(seed)]["methods"]["R1"]["test"] for seed in (7, 42, 2026)}
    tep_drfd = {seed: result["tep"][str(seed)]["method"]["test"] for seed in (7, 42, 2026)}

    paired = []
    three_w_metric_names = ("macro_f1", "far", "auprc_fault_vs_normal", "early_recall", "mean_detection_delay_seconds")
    for seed in (42, 43, 44):
        for metric in three_w_metric_names:
            old, new = float(three_w_r1[seed][metric]), float(three_w_drfd[seed][metric])
            paired.append({"dataset": "3W", "seed": seed, "metric": metric, "r1": old, "drfd": new, "delta": new-old})
        for metric in ("recall", "f1"):
            old, new = float(_class(three_w_r1[seed], 9)[metric]), float(_class(three_w_drfd[seed], 9)[metric])
            paired.append({"dataset": "3W", "seed": seed, "metric": f"class9_{metric}", "r1": old, "drfd": new, "delta": new-old})
    for seed in (7, 42, 2026):
        pairs = {
            "macro_f1": (tep_r1[seed]["metrics"]["macro_f1"], tep_drfd[seed]["metrics"]["macro_f1"]),
            "auprc": (tep_r1[seed]["metrics"]["auprc"], tep_drfd[seed]["metrics"]["auprc"]),
            "far": (tep_r1[seed]["metrics"]["far"], tep_drfd[seed]["metrics"]["far"]),
            "early_recall": (tep_r1[seed]["early_fault"]["recall"], tep_drfd[seed]["early_fault"]["recall"]),
            "delay_samples": (tep_r1[seed]["detection_delay"]["mean_delay_samples"], tep_drfd[seed]["detection_delay"]["mean_delay_samples"]),
        }
        for metric, (old, new) in pairs.items():
            paired.append({"dataset": "TEP", "seed": seed, "metric": metric,
                           "r1": float(old), "drfd": float(new), "delta": float(new-old)})

    def series(records, metric): return [float(records[seed][metric]) for seed in records]
    r1_macro = series(three_w_r1, "macro_f1"); drfd_macro = series(three_w_drfd, "macro_f1")
    r1_far = series(three_w_r1, "far"); drfd_far = series(three_w_drfd, "far")
    r1_c9r = [float(_class(three_w_r1[s], 9)["recall"]) for s in three_w_r1]
    drfd_c9r = [float(_class(three_w_drfd[s], 9)["recall"]) for s in three_w_drfd]
    r1_c9f = [float(_class(three_w_r1[s], 9)["f1"]) for s in three_w_r1]
    drfd_c9f = [float(_class(three_w_drfd[s], 9)["f1"]) for s in three_w_drfd]
    macro_std_reduction = _reduction(np.std(r1_macro), np.std(drfd_macro))
    far_std_reduction = _reduction(np.std(r1_far), np.std(drfd_far))
    c9r_reduction = _reduction(np.std(r1_c9r), np.std(drfd_c9r))
    c9f_reduction = _reduction(np.std(r1_c9f), np.std(drfd_c9f))
    three_w_checks = {
        "mean_macro_f1_delta": float(np.mean(drfd_macro)-np.mean(r1_macro)) >= -.005,
        "at_least_two_nonnegative_macro_f1_seeds": int(np.sum(np.asarray(drfd_macro) >= np.asarray(r1_macro))) >= 2,
        "mean_far_delta": float(np.mean(drfd_far)-np.mean(r1_far)) <= .01,
        "main_stability_improvement": max(macro_std_reduction, far_std_reduction) >= .15,
        "class9_stability": ((c9r_reduction >= .15 and c9f_reduction >= -.10)
                             or (c9f_reduction >= .15 and c9r_reduction >= -.10)),
        "worst_seed_macro_f1": min(drfd_macro) >= min(r1_macro)-.01,
        "no_catastrophic_regression": all((three_w_drfd[s]["macro_f1"]-three_w_r1[s]["macro_f1"] >= -.03)
                                          and (three_w_drfd[s]["far"]-three_w_r1[s]["far"] <= .05)
                                          for s in three_w_r1),
    }
    three_w_summary = {
        "r1": {"macro_f1": _mean_std(r1_macro), "far": _mean_std(r1_far),
               "class9_recall": _mean_std(r1_c9r), "class9_f1": _mean_std(r1_c9f)},
        "drfd": {"macro_f1": _mean_std(drfd_macro), "far": _mean_std(drfd_far),
                 "class9_recall": _mean_std(drfd_c9r), "class9_f1": _mean_std(drfd_c9f)},
        "mean_macro_f1_delta": float(np.mean(drfd_macro)-np.mean(r1_macro)),
        "mean_far_delta": float(np.mean(drfd_far)-np.mean(r1_far)),
        "nonnegative_macro_f1_seed_count": int(np.sum(np.asarray(drfd_macro) >= np.asarray(r1_macro))),
        "macro_f1_std_reduction": macro_std_reduction, "far_std_reduction": far_std_reduction,
        "class9_recall_std_reduction": c9r_reduction, "class9_f1_std_reduction": c9f_reduction,
        "worst_seed_macro_f1": {"r1": min(r1_macro), "drfd": min(drfd_macro)},
        "checks": three_w_checks, "passed": all(three_w_checks.values())}

    def tep_values(records, path):
        values = []
        for seed in records:
            value = records[seed]
            for key in path: value = value[key]
            values.append(float(value))
        return values
    tep_paths = {"macro_f1": ("metrics", "macro_f1"), "auprc": ("metrics", "auprc"),
                 "far": ("metrics", "far"), "early_recall": ("early_fault", "recall"),
                 "delay_samples": ("detection_delay", "mean_delay_samples")}
    tep_summary = {"r1": {}, "drfd": {}, "mean_delta": {}}
    for name, path in tep_paths.items():
        old, new = tep_values(tep_r1, path), tep_values(tep_drfd, path)
        tep_summary["r1"][name] = _mean_std(old); tep_summary["drfd"][name] = _mean_std(new)
        tep_summary["mean_delta"][name] = float(np.mean(new)-np.mean(old))
    tep_checks = {"macro_f1": tep_summary["mean_delta"]["macro_f1"] >= -.002,
                  "auprc": tep_summary["mean_delta"]["auprc"] >= -.002,
                  "far": tep_summary["mean_delta"]["far"] <= .002,
                  "early_recall": tep_summary["mean_delta"]["early_recall"] >= -.005,
                  "delay": tep_summary["mean_delta"]["delay_samples"] <= 16,
                  "no_catastrophic_regression": all(
                      tep_drfd[s]["metrics"]["macro_f1"]-tep_r1[s]["metrics"]["macro_f1"] >= -.03
                      and tep_drfd[s]["metrics"]["far"]-tep_r1[s]["metrics"]["far"] <= .05 for s in tep_r1)}
    tep_summary.update({"checks": tep_checks, "passed": all(tep_checks.values())})
    status = "DRFD_DUAL_DATASET_GO" if three_w_summary["passed"] and tep_summary["passed"] else "DRFD_DUAL_DATASET_NO_GO"
    summary = {"status": status, "stage_a": stage_a["status"], "stage_b": kill["status"],
               "new_training_runs": 6, "three_w": three_w_summary, "tep": tep_summary,
               "uncertainty_direction_frozen": status != "DRFD_DUAL_DATASET_GO"}

    paired_path = Path(config["docs"]["paired_csv"]); paired_path.parent.mkdir(parents=True, exist_ok=True)
    with paired_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=("dataset", "seed", "metric", "r1", "drfd", "delta"))
        writer.writeheader(); writer.writerows(paired)
    reliability_path = Path(config["docs"]["reliability_csv"])
    with reliability_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ("dataset", "channel", "frequency_bin", "rank_median", "rank_q25", "rank_q75", "rank_iqr", "category")
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for dataset_key, label in (("three_w", "3W"), ("tep", "TEP")):
            rel = stage_a[dataset_key]["reliability"]
            arrays = {key: np.asarray(rel[key]) for key in ("rank_median", "rank_q25", "rank_q75", "rank_iqr")}
            rc = np.asarray(rel["reliable_critical"], bool); amb = np.asarray(rel["ambiguous"], bool)
            for channel in range(rc.shape[0]):
                for frequency in range(rc.shape[1]):
                    writer.writerow({"dataset": label, "channel": channel, "frequency_bin": frequency,
                                     **{key: arrays[key][channel, frequency] for key in arrays},
                                     "category": "reliable_critical" if rc[channel, frequency] else
                                                 "ambiguous" if amb[channel, frequency] else "reliable_noncritical"})
    write_json(Path("outputs/drfd/final_summary.json"), summary)
    _write_reports(config, summary, stage_a)
    return summary


def _fmt(item: dict[str, float]) -> str:
    return f"{item['mean']:.6f} ± {item['std']:.6f}"


def _write_reports(config: dict[str, Any], summary: dict[str, Any], stage_a: dict[str, Any]) -> None:
    three = summary["three_w"]; tep = summary["tep"]
    Path(config["docs"]["three_w_report"]).write_text(
        "# DRFD 3W 报告\n\n"
        f"Gate：`{'GO' if three['passed'] else 'NO-GO'}`。\n\n"
        f"- Macro-F1：R1 {_fmt(three['r1']['macro_f1'])}；DRFD {_fmt(three['drfd']['macro_f1'])}；均值差 {three['mean_macro_f1_delta']:+.6f}\n"
        f"- FAR：R1 {_fmt(three['r1']['far'])}；DRFD {_fmt(three['drfd']['far'])}；均值差 {three['mean_far_delta']:+.6f}\n"
        f"- Macro-F1/FAR std 降幅：{three['macro_f1_std_reduction']:.2%} / {three['far_std_reduction']:.2%}\n"
        f"- Class 9 Recall/F1 std 降幅：{three['class9_recall_std_reduction']:.2%} / {three['class9_f1_std_reduction']:.2%}\n"
        f"- 各项 Gate：{three['checks']}\n", encoding="utf-8")
    Path(config["docs"]["tep_report"]).write_text(
        "# DRFD TEP 报告\n\n"
        f"Preservation Gate：`{'GO' if tep['passed'] else 'NO-GO'}`。\n\n"
        f"- Macro-F1：R1 {_fmt(tep['r1']['macro_f1'])}；DRFD {_fmt(tep['drfd']['macro_f1'])}\n"
        f"- AUPRC：R1 {_fmt(tep['r1']['auprc'])}；DRFD {_fmt(tep['drfd']['auprc'])}\n"
        f"- FAR：R1 {_fmt(tep['r1']['far'])}；DRFD {_fmt(tep['drfd']['far'])}\n"
        f"- Early Recall：R1 {_fmt(tep['r1']['early_recall'])}；DRFD {_fmt(tep['drfd']['early_recall'])}\n"
        f"- Mean delta：{tep['mean_delta']}\n- 各项 Gate：{tep['checks']}\n", encoding="utf-8")
    no_go = summary["status"] != "DRFD_DUAL_DATASET_GO"
    Path(config["docs"]["summary"]).write_text(
        "# DRFD 双数据集总结\n\n"
        f"最终结论：`{summary['status']}`。Stage A=`{summary['stage_a']}`，Stage B=`{summary['stage_b']}`，新增训练 run=`6`。\n\n"
        "UG-R1 失败源于对称 uncertain→Uniform 会增加部分 R1 protected bins 的扰动；DRFD 通过 `t_r1<=3 => t_safe=t_r1`，并限制所有预算调整只发生在 reliable non-critical bins，避免削弱 semantic protection。\n\n"
        f"3W 可靠 critical / ambiguous / reliable non-critical 数量为 "
        f"{stage_a['three_w']['structure']['categories']['reliable_critical']['count']} / "
        f"{stage_a['three_w']['structure']['categories']['ambiguous']['count']} / "
        f"{stage_a['three_w']['structure']['categories']['reliable_noncritical']['count']}；"
        f"TEP 为 {stage_a['tep']['structure']['categories']['reliable_critical']['count']} / "
        f"{stage_a['tep']['structure']['categories']['ambiguous']['count']} / "
        f"{stage_a['tep']['structure']['categories']['reliable_noncritical']['count']}。Stage A 安全不变量和 2% budget Gate 均通过。\n\n"
        f"3W stability Gate=`{'GO' if three['passed'] else 'NO-GO'}`；TEP preservation Gate=`{'GO' if tep['passed'] else 'NO-GO'}`。"
        + ("按预注册停止 uncertainty 方向，不搜索 rank threshold/confidence function，不开发 DRFD-v2，回退并冻结 R1。" if no_go else "DRFD 可进入论文方法冻结。")
        + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/domain_reliable_safe_frequency_diffusion.yaml")
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = summarize(config); print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
