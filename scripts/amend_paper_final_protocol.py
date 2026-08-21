from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import scripts.run_3w_clean_baseline as base3w
from datasets.three_w import discover_instances
from scripts.audit_paper_final_protocol import _tep
from scripts.run_3w_final_primary_grouped import FINAL_PRIMARY_CLASSES
from utils import write_json


SPLITS = ("train", "validation", "test")


def read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hash_items(items: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(items)).encode("utf-8")).hexdigest()


def window_inventory(data_root: Path, grouped_config: dict[str, Any]) -> tuple[set[str], dict[str, dict[int, int]]]:
    base = yaml.safe_load(Path(grouped_config["base_config"]).read_text(encoding="utf-8"))
    base3w.PRIMARY_CLASSES = FINAL_PRIMARY_CLASSES
    base3w.CLASS_TO_TARGET = {value: index for index, value in enumerate(FINAL_PRIMARY_CLASSES)}
    instances = [item for item in discover_instances(data_root)
                 if item.source == "WELL" and item.event_class in FINAL_PRIMARY_CLASSES]
    wells = {str(item.well_id) for item in instances}; counts = {well: {target: 0 for target in range(len(FINAL_PRIMARY_CLASSES))} for well in wells}
    for item in instances:
        refs = base3w.instance_refs(item, int(base["protocol"]["window_length"]), int(base["protocol"]["stride"]),
                                    int(base["protocol"]["transient_offset"]))
        for ref in refs: counts[str(item.well_id)][int(ref.target)] += 1
    return wells, counts


def coverage(split: dict[str, set[str]], counts: dict[str, dict[int, int]]) -> dict[str, Any]:
    return {name: {str(target): {"windows": int(sum(counts[well][target] for well in wells)),
                                 "wells": int(sum(counts[well][target] > 0 for well in wells))}
                   for target in range(len(FINAL_PRIMARY_CLASSES))}
            for name, wells in split.items()}


def valid(split: dict[str, set[str]], counts: dict[str, dict[int, int]], minimum_fault_wells: dict[str, int]) -> bool:
    current = coverage(split, counts)
    for name in SPLITS:
        for target in range(len(FINAL_PRIMARY_CLASSES)):
            if current[name][str(target)]["windows"] <= 0: return False
            required = 1 if target == 0 else int(minimum_fault_wells[name])
            if current[name][str(target)]["wells"] < required: return False
    return True


def test_jaccard_ok(test: set[str], previous: list[set[str]], maximum: float) -> bool:
    return all(len(test & old) / len(test | old) <= maximum for old in previous)


def regenerate(wells: set[str], counts: dict[str, dict[int, int]], split_counts: dict[str, int],
               minimum_fault_wells: dict[str, int], seed: int, previous: list[set[str]], maximum: float) -> tuple[dict[str, set[str]], int]:
    ordered = np.asarray(sorted(wells), dtype=object); rng = np.random.default_rng(seed)
    for candidate in range(1, 200_001):
        shuffled = ordered[rng.permutation(len(ordered))]; train_end = split_counts["train"]; val_end = train_end + split_counts["validation"]
        split = {"train": set(shuffled[:train_end]), "validation": set(shuffled[train_end:val_end]), "test": set(shuffled[val_end:])}
        if valid(split, counts, minimum_fault_wells) and test_jaccard_ok(split["test"], previous, maximum): return split, candidate
    raise RuntimeError(f"cannot construct WindowRef-coverage-safe split for seed {seed}")


def record(seed: int, split: dict[str, set[str]], counts: dict[str, dict[int, int]], candidates: int, preserved: bool) -> dict[str, Any]:
    overlap = {f"{a}_{b}": sorted(split[a] & split[b]) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))}
    return {"outer_split_seed": seed, "groups": {name: sorted(split[name]) for name in SPLITS},
            "group_hash": {name: hash_items(list(split[name])) for name in SPLITS},
            "windowref_coverage": coverage(split, counts), "overlap": overlap,
            "candidate_assignments_checked": candidates, "preserved_existing_split": preserved,
            "coverage_definition": "usable WindowRef targets after frozen label mapping and transition exclusion"}


def amend(config: dict[str, Any], data_root: Path) -> dict[str, Any]:
    output = config["output"]; dry_path = Path(output["manifest"]); old_snapshot = Path(output["pre_amendment_manifest"])
    if not old_snapshot.exists(): write_json(old_snapshot, read(dry_path))
    old = read(old_snapshot); old_hash = sha256_file(old_snapshot)
    grouped = yaml.safe_load(Path(config["amendment"]["three_w_grouped_config"]).read_text(encoding="utf-8"))
    wells, counts = window_inventory(data_root, grouped); settings = config["datasets"]["three_w"]
    split_counts = {"train": int(settings["wells"]["train"]), "validation": int(settings["wells"]["inner_validation"]), "test": int(settings["wells"]["outer_test"])}
    minimum = {name: int(value) for name, value in config["amendment"]["minimum_fault_wells"].items()}
    old_by_seed = {int(row["outer_split_seed"]): row for row in old["three_w"]}; revised = []; previous: list[set[str]] = []
    for seed in map(int, settings["outer_split_seeds"]):
        old_split = {name: set(old_by_seed[seed]["groups"][name]) for name in SPLITS}
        preserve = valid(old_split, counts, minimum) and test_jaccard_ok(old_split["test"], previous, float(config["amendment"]["maximum_test_jaccard"]))
        if preserve: split, candidate_count = old_split, 0
        else: split, candidate_count = regenerate(wells, counts, split_counts, minimum, seed, previous, float(config["amendment"]["maximum_test_jaccard"]))
        revised.append(record(seed, split, counts, candidate_count, preserve)); previous.append(split["test"])
    tep = _tep(config, Path(config["amendment"]["tep_fixed_manifest"]))
    if tep != old["tep"]: raise RuntimeError("TEP frozen split changed during 3W-only amendment")
    payload = {"amendment": {"version": "windowref-coverage-v2", "old_manifest_sha256": old_hash,
                              "performance_metrics_used": False, "outer_training_run": False, "outer_test_metrics_read": False},
               "three_w": revised, "tep": tep, "outer_metrics": None}
    write_json(dry_path, payload); new_hash = sha256_file(dry_path)
    disjoint = all(not any(row["overlap"].values()) for row in revised + tep)
    coverage_go = all(valid({name: set(row["groups"][name]) for name in SPLITS}, counts, minimum) for row in revised)
    audit = {"status": "PAPER_FINAL_PROTOCOL_AMENDMENT_GO" if disjoint and coverage_go else "PAPER_FINAL_PROTOCOL_AMENDMENT_HOLD",
             "amendment_version": "windowref-coverage-v2", "old_3w_split_manifest_sha256": old_hash,
             "new_split_manifest_sha256": new_hash, "three_w": revised, "tep": tep,
             "group_disjoint": disjoint, "windowref_coverage_go": coverage_go,
             "outer_test_metrics_read": False, "outer_training_run": False, "performance_metrics_used_for_split_selection": False,
             "unchanged": {"methods": True, "hyperparameters": True, "baselines": True, "seeds": True, "metrics": True, "tep_protocol": True},
             "fit_scope": {"scaler": "outer-train", "imputation": "outer-train", "criticality_D_E": "outer-train",
                           "rho": "inner-validation", "threshold": "inner-validation", "outer_test": "evaluation-only"}}
    write_json(Path(output["audit"]), audit); write_json(Path(output["amendment_audit"]), audit)
    if audit["status"] != "PAPER_FINAL_PROTOCOL_AMENDMENT_GO": raise RuntimeError(str(audit))
    return audit


def report(config: dict[str, Any], result: dict[str, Any]) -> None:
    lines = ["# Paper-final Protocol Amendment", "", "状态：`PAPER_FINAL_PROTOCOL_AMENDMENT_GO`。问题在首次 outer training/metric 之前发现并修复；`first_outer_metric_at = null`。", "",
             "## 原因", "", "旧 dry-run 以原始 event/class 文件存在性检查 coverage，而正式 runner 消费的是经过冻结 label mapping、transition exclusion 和 windowization 后的 `WindowRef.target`。因此部分包含 class-9 文件的 WELL 实际没有 target-3 窗口。", "",
             "## Hash", "", f"- Old manifest: `{result['old_3w_split_manifest_sha256']}`", f"- Revised manifest: `{result['new_split_manifest_sha256']}`", "",
             "## Deterministic rule", "", "每个 outer seed 使用独立 NumPy RNG；候选只按 WELL 分组、实际 WindowRef coverage、20/8/8 数量及 test-Jaccard 约束判定；保留仍满足新约束的旧 split，否则接受第一个 valid assignment。没有训练模型、读取 outer-test 或比较性能。", "",
             "| Seed | Preserved | Candidates checked | Train target windows | Validation target windows | Test target windows |", "|---:|---|---:|---|---|---|"]
    for row in result["three_w"]:
        values = lambda name: "/".join(str(row["windowref_coverage"][name][str(target)]["windows"]) for target in range(len(FINAL_PRIMARY_CLASSES)))
        lines.append(f"| {row['outer_split_seed']} | {row['preserved_existing_split']} | {row['candidate_assignments_checked']} | {values('train')} | {values('validation')} | {values('test')} |")
    lines += ["", "TEP split/hash 未改变；FINAL_QDIFFCL、DCBR、baseline、model seeds、metrics 与统计规则均未改变。"]
    Path(config["output"]["amendment_report"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    protocol = ["# Q-DiffCL Paper-final Protocol", "", "状态：`PAPER_FINAL_PROTOCOL_AMENDMENT_GO`。WindowRef coverage 修订发生在任何 outer training/metric 之前。", "",
                "## 3W repeated grouped outer holdout（revised）", "", "| Outer seed | Train WELL | Inner-val WELL | Outer-test WELL | WindowRef targets complete | Candidates checked |", "|---:|---:|---:|---:|---|---:|"]
    for row in result["three_w"]:
        protocol.append(f"| {row['outer_split_seed']} | {len(row['groups']['train'])} | {len(row['groups']['validation'])} | {len(row['groups']['test'])} | True | {row['candidate_assignments_checked']} |")
    protocol += ["", "Coverage 以正式 runner 的冻结 label mapping、transition exclusion 与 `WindowRef.target` 为准；每个 target 在 train/validation/test 均有可用窗口。WELL 完全不相交。", "",
                 "## TEP run-level nested grouped evaluation", "", "TEP 的 248/72/80 Run splits 与修订前逐元素一致，Run 仍为最小分组单位。", "",
                 "## Fit scope", "", "scaler、插补、D/E criticality、frequency statistics 仅 outer-train 拟合；rho、threshold、early stopping 仅 inner-validation；outer-test 只进行冻结评估。", "",
                 "## Frozen boundary", "", "方法、baseline、model seeds、outer seeds、metrics、2,000 次 group bootstrap 均未改变；split 生成未使用模型性能。"]
    Path(config["output"]["report"]).write_text("\n".join(protocol) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/paper_final_protocol.yaml"); parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")); result = amend(config, args.data_root); report(config, result)
    print(json.dumps({"status": result["status"], "old_hash": result["old_3w_split_manifest_sha256"], "new_hash": result["new_split_manifest_sha256"],
                      "splits": [{"seed": row["outer_split_seed"], "preserved": row["preserved_existing_split"], "candidates": row["candidate_assignments_checked"]} for row in result["three_w"]]}, ensure_ascii=False))


if __name__ == "__main__": main()
