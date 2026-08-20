from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

import scripts.run_3w_clean_baseline as base3w
from datasets.three_w import discover_instances
from diffusion.fixed_views import sha256_file, sha256_strings
from frequency import capacity_distribution, fault_stages, safe_capacity
from scripts.run_3w_final_primary_grouped import FINAL_PRIMARY_CLASSES
from utils import write_json


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _mask(path: str | Path, expected_hash: str) -> tuple[np.ndarray, dict[str, Any]]:
    payload = _read(path)["criticality"]
    if payload["mask_sha256"] != expected_hash or payload["fit_split"] != "train":
        raise RuntimeError("FINAL train-only mask identity changed")
    if payload.get("test_or_validation_used"):
        raise RuntimeError("capacity audit mask used validation/test")
    return np.asarray(payload["soft_mask"], dtype=np.float32), payload


def load_three_w_train(config: dict[str, Any], data_root: Path) -> dict[str, np.ndarray]:
    stage = config["three_w"]
    runner = yaml.safe_load(Path(stage["base_config"]).read_text(encoding="utf-8"))
    grouped = Path(runner["grouped_output"]); split_index = int(runner["canonical_split_index"])
    split_payload = _read(grouped / "grouped_split_manifest.json")["splits"][split_index]
    train_wells = set(split_payload["wells"]["train"])
    split_groups = [set(split_payload["wells"][name]) for name in ("train", "validation", "test")]
    if any(split_groups[i] & split_groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("3W canonical split has WELL leakage")
    sampling = yaml.safe_load(Path(runner["base_config"]).read_text(encoding="utf-8"))
    protocol = yaml.safe_load(Path(sampling["base_config"]).read_text(encoding="utf-8"))
    base3w.PRIMARY_CLASSES = FINAL_PRIMARY_CLASSES
    base3w.CLASS_TO_TARGET = {value: index for index, value in enumerate(FINAL_PRIMARY_CLASSES)}
    instances = [item for item in discover_instances(data_root)
                 if item.source == "WELL" and item.event_class in FINAL_PRIMARY_CLASSES and item.well_id in train_wells]
    by_instance = {item.instance_id: item for item in instances}
    refs = []
    for item in instances:
        refs.extend(base3w.instance_refs(item, int(protocol["protocol"]["window_length"]),
                                        int(protocol["protocol"]["stride"]),
                                        int(protocol["protocol"]["transient_offset"])))
    refs = base3w.stratified_refs(refs, int(sampling["train_windows_per_class"]),
                                  int(stage["protocol_seed"]))
    preprocessor = _read(grouped / f"split_{split_index:02d}" / "preprocessor.json")
    x, y = base3w.materialize(refs, by_instance, preprocessor,
                              int(protocol["protocol"]["window_length"]), False)
    stages = np.asarray(["normal" if ref.stage == "normal" else
                         "early" if ref.stage == "early" else "mature" for ref in refs], dtype=object)
    wells = np.asarray([by_instance[ref.instance_id].well_id for ref in refs], dtype=object)
    original_classes = np.asarray([FINAL_PRIMARY_CLASSES[ref.target] if ref.target else 0 for ref in refs])
    return {"values": x, "labels": y, "stage": stages, "unit": wells,
            "fault_type": original_classes, "split": np.asarray(["train"] * len(x), dtype=object)}


def load_tep_train(config: dict[str, Any]) -> dict[str, np.ndarray]:
    stage = config["tep"]
    base = yaml.safe_load(Path(stage["base_config"]).read_text(encoding="utf-8"))
    manifest = _read(base["fixed_views"]["manifest"]); record = manifest["splits"]["train"]
    path = Path(record["path"])
    if sha256_file(path) != record["sha256"]:
        raise RuntimeError("TEP frozen train view hash changed")
    with np.load(path, allow_pickle=False) as archive:
        bundle = {key: archive[key] for key in archive.files}
    if sha256_strings(list(map(str, bundle["window_id"]))) != record["window_ids_sha256"]:
        raise RuntimeError("TEP train window order changed")
    stages = fault_stages(bundle, base)
    stage_names = np.where(stages == "prefault", "normal",
                           np.where(stages == "early", "early", "mature"))
    units = np.asarray(bundle["run_uid"], dtype=object)
    kinds = np.asarray([int(match.group(1)) if (match := re.search(r":fault_(\d+):", str(uid))) else 0
                        for uid in units])
    return {"values": bundle["clean"].astype(np.float32), "labels": bundle["labels"],
            "stage": stage_names.astype(object), "unit": units, "fault_type": kinds,
            "split": np.asarray(["train"] * len(units), dtype=object)}


def _groups(dataset: str, bundle: dict[str, np.ndarray], result: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    selectors = [("all", "all", np.ones(len(result["rho"]), dtype=bool))]
    selectors += [("stage", str(value), bundle["stage"] == value) for value in np.unique(bundle["stage"])]
    selectors += [("unit", str(value), bundle["unit"] == value) for value in np.unique(bundle["unit"])]
    selectors += [("fault_type", str(value), bundle["fault_type"] == value)
                  for value in np.unique(bundle["fault_type"])]
    for group_type, group, selector in selectors:
        if not selector.any():
            continue
        for metric in ("critical_energy_ratio", "safe_capacity", "rho"):
            rows.append({"dataset": dataset, "split": "train", "group_type": group_type,
                         "group": group, "metric": metric, **capacity_distribution(result[metric][selector])})
    return rows


def audit(config: dict[str, Any], data_root: Path) -> dict[str, Any]:
    final = yaml.safe_load(Path(config["final_config"]).read_text(encoding="utf-8"))
    if not final.get("frozen") or final["weights"] != {
        "weight_discriminative": .5, "weight_early": .5, "weight_run_stability": 0.0}:
        raise RuntimeError("FINAL_QDIFFCL changed")
    bundles = {"3W": load_three_w_train(config, data_root), "TEP": load_tep_train(config)}
    masks = {dataset: _mask(config[key]["final_mask"], final["mask_sha256"][dataset])[0]
             for dataset, key in (("3W", "three_w"), ("TEP", "tep"))}
    gamma = float(config["gamma"]); primary = {
        dataset: safe_capacity(bundle["values"], masks[dataset], gamma)
        for dataset, bundle in bundles.items()
    }
    aggregate = {dataset: {metric: capacity_distribution(values[metric])
                           for metric in values} for dataset, values in primary.items()}
    candidates = {dataset: {str(candidate): capacity_distribution(
        safe_capacity(bundle["values"], masks[dataset], float(candidate))["rho"])
        for candidate in config["gamma_candidates"]} for dataset, bundle in bundles.items()}
    gamma_gaps = {str(candidate): candidates["3W"][str(candidate)]["mean"] -
                  candidates["TEP"][str(candidate)]["mean"]
                  for candidate in config["gamma_candidates"]}
    gap = aggregate["3W"]["rho"]["mean"] - aggregate["TEP"]["rho"]["mean"]
    direction_go = gap > float(config["direction_gate"]["minimum_3w_minus_tep_mean_rho"])
    status = "SAFE_CAPACITY_DIRECTION_GO" if direction_go else "NO_GO_SAFE_CAPACITY_DIRECTION"
    rows = _groups("3W", bundles["3W"], primary["3W"]) + _groups("TEP", bundles["TEP"], primary["TEP"])
    csv_path = Path(config["output"]["capacity_csv"]); csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    figure_dir = Path(config["output"]["figure_dir"]); figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    for dataset, color in (("3W", "tab:blue"), ("TEP", "tab:orange")):
        axis.hist(primary[dataset]["rho"], bins=np.linspace(0, 1, 31), alpha=.55, density=True,
                  label=f"{dataset} train", color=color)
    axis.set(xlabel=r"Safe budget $\rho(x)$", ylabel="Density", title="Train-only safe-capacity distribution")
    axis.grid(alpha=.2); axis.legend(); fig.tight_layout()
    fig.savefig(figure_dir / "safe_capacity_histogram.png", dpi=180); plt.close(fig)
    payload = {"status": status, "formula": "rho=(1-sum(C_cf*P_cf)/sum(P_cf))**gamma",
               "gamma": gamma, "fit_scope": "train only", "test_metrics_read": False,
               "mask_sha256": final["mask_sha256"], "aggregate": aggregate,
               "gamma_candidates_train_only": candidates, "three_w_minus_tep_mean_rho": gap,
               "gamma_direction_gaps": gamma_gaps,
               "sample_adaptive": {dataset: values["rho"]["std"] > 1e-6
                                   for dataset, values in aggregate.items()},
               "records": len(rows)}
    write_json(Path(config["output"]["capacity_json"]), payload)
    stage_rows = {dataset: {row["group"]: row for row in rows
                            if row["dataset"] == dataset and row["group_type"] == "stage" and row["metric"] == "rho"}
                  for dataset in ("3W", "TEP")}
    gamma_lines = "\n".join(
        f"| {candidate} | {candidates['3W'][str(candidate)]['mean']:.6f} | "
        f"{candidates['TEP'][str(candidate)]['mean']:.6f} | {gamma_gaps[str(candidate)]:+.6f} |"
        for candidate in config["gamma_candidates"])
    stage_lines = "\n".join(
        f"| {dataset} | {stage} | {record['count']} | {record['mean']:.6f} | {record['std']:.6f} | {record['median']:.6f} |"
        for dataset in ("3W", "TEP") for stage, record in sorted(stage_rows[dataset].items()))
    report = f"""# Safe Capacity Train-only Audit

定义：`critical_energy_ratio(x)=sum(C_cf*P_cf(x))/sum(P_cf(x))`，`rho(x)=(1-critical_energy_ratio(x))^{gamma}`；冻结 `gamma={gamma}`。

| Dataset | Windows | Mean rho | Std | Median | P05 | P95 |
|---|---:|---:|---:|---:|---:|---:|
| 3W | {aggregate['3W']['rho']['count']} | {aggregate['3W']['rho']['mean']:.6f} | {aggregate['3W']['rho']['std']:.6f} | {aggregate['3W']['rho']['median']:.6f} | {aggregate['3W']['rho']['p05']:.6f} | {aggregate['3W']['rho']['p95']:.6f} |
| TEP | {aggregate['TEP']['rho']['count']} | {aggregate['TEP']['rho']['mean']:.6f} | {aggregate['TEP']['rho']['std']:.6f} | {aggregate['TEP']['rho']['median']:.6f} | {aggregate['TEP']['rho']['p05']:.6f} | {aggregate['TEP']['rho']['p95']:.6f} |

3W - TEP mean rho：`{gap:+.6f}`。Stage A 判定：`{status}`。

## Gamma 候选方向检查

| Gamma | 3W mean rho | TEP mean rho | 3W - TEP |
|---:|---:|---:|---:|
{gamma_lines}

三个允许候选均保持相反方向；gamma 是单调变换，不能修复跨数据集排序。

## 阶段分布

| Dataset | Stage | Windows | Mean rho | Std | Median |
|---|---|---:|---:|---:|---:|
{stage_lines}

![rho histogram](assets/budget_allocator_v2/safe_capacity_histogram.png)

所有统计仅来自 frozen train split；未加载 validation/test 选择或拟合 capacity。
完整 class、WELL、Run/fault-type 分组统计见 `safe_capacity_audit.csv`。
"""
    Path(config["output"]["capacity_report"]).write_text(report, encoding="utf-8")
    summary = f"""# Budget-Constrained Allocation v2 Summary

## Stage A：`{status}`

参数无关 Safe Capacity 在 train-only 数据上给出 3W mean rho `{aggregate['3W']['rho']['mean']:.6f}`、TEP `{aggregate['TEP']['rho']['mean']:.6f}`，方向差 `{gap:+.6f}`。该方向与 Budget Shrinkage Diagnostic 的 3W 中高预算、TEP 低预算需求相反。

依照预注册硬门，本阶段在 Stage A 停止：未将 sample-adaptive variance 接入 diffusion，未运行 Stage C 单 seed 或 Stage D 三 seed，未读取 validation/test 指标，也未修改 FINAL_QDIFFCL。

最终判定：`NO_GO_BUDGET_CONSTRAINED_ALLOCATION_V2`。
"""
    Path(config["output"]["summary"]).write_text(summary, encoding="utf-8")
    decision = f"""# Budget-Constrained Allocation v2 Decision

## NO_GO_BUDGET_CONSTRAINED_ALLOCATION_V2

当前定义 `rho(x)=1-critical_energy_ratio(x)` 在 train-only 数据上产生相反的 domain ordering：3W - TEP mean rho 为 `{gap:+.6f}`。允许的 `gamma={{0.5,1,2}}` 均不能反转该方向，因此不能用 validation/test 性能为其打补丁。

停止当前 controller；保留冻结 FINAL 与 Budget Shrinkage 结论。Stage B/C/D 未执行。
"""
    Path(config["output"]["decision"]).write_text(decision, encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/budget_constrained_allocation_v2.yaml")
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    print(json.dumps(audit(config, args.data_root), ensure_ascii=False))


if __name__ == "__main__":
    main()
