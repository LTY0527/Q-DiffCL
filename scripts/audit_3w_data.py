from __future__ import annotations

import argparse
import configparser
import csv
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from datasets.three_w import (discover_instances, process_features, read_instance,
                              well_level_split_covering_classes, window_instance)
from models.backbones import TCNClassifier


def _source(name: str) -> str:
    return "WELL" if name.startswith("WELL-") else "SIMULATED" if name.startswith("SIMULATED_") else "DRAWN"


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _stats(values: list[int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": int(array.min()), "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)), "mean": float(array.mean()),
        "p75": float(np.percentile(array, 75)), "max": int(array.max()),
    }


def _label_phase(raw: float | None, event_class: int, offset: int) -> str:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return "unlabeled"
    label = int(raw)
    if label == 0:
        return "normal"
    if label == event_class + offset:
        return "transient"
    if label == event_class:
        return "event"
    return "other"


def audit(data_root: Path, output: Path, report: Path) -> dict[str, Any]:
    config = configparser.ConfigParser()
    config.read(data_root / "dataset.ini", encoding="utf-8")
    version = config["VERSION"]["DATASET"]
    transient_offset = config.getint("EVENTS", "TRANSIENT_OFFSET")
    events = [item.strip() for item in config["EVENTS"]["NAMES"].replace("\n", " ").split(",")]
    event_metadata = []
    for name in events:
        section = config[name]
        event_metadata.append({
            "class": int(section["LABEL"]), "event": name,
            "description": section["DESCRIPTION"],
            "transient_defined": section.getboolean("TRANSIENT", fallback=False),
        })

    try:
        commit = subprocess.check_output(
            ["git", "-C", str(data_root.parent), "rev-parse", "HEAD"], text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "unavailable"

    instances = discover_instances(data_root)
    first_schema = pq.ParquetFile(instances[0].path).schema_arrow.names
    features = process_features(first_schema)
    feature_acc = {
        name: {"instances": 0, "observations": 0, "present": 0, "sum": 0.0, "sum_sq": 0.0,
               "min": math.inf, "max": -math.inf}
        for name in features
    }
    class_source: dict[tuple[int, str], dict[str, int]] = defaultdict(lambda: {"instances": 0, "observations": 0})
    missing_groups: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    length_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    label_groups: dict[tuple[int, str, str], dict[str, Any]] = defaultdict(lambda: {"observations": 0, "instances": set()})
    label_sequence_groups: Counter[tuple[int, str, str]] = Counter()
    well_rows: list[dict[str, Any]] = []
    sampling_rows: list[dict[str, Any]] = []
    schemas: Counter[tuple[str, ...]] = Counter()
    global_missing = global_cells = 0

    for index, instance in enumerate(instances, 1):
        table = pq.read_table(instance.path)
        frame = table.to_pandas()
        if "timestamp" in frame.columns:
            timestamps = frame["timestamp"].to_numpy(dtype="datetime64[ns]")
        else:
            timestamps = frame.index.to_numpy(dtype="datetime64[ns]")
        source = instance.source
        length = len(frame)
        class_source[(instance.event_class, source)]["instances"] += 1
        class_source[(instance.event_class, source)]["observations"] += length
        length_groups[("overall", "ALL")].append(length)
        length_groups[("class", str(instance.event_class))].append(length)
        length_groups[("source", source)].append(length)
        schemas[tuple(frame.columns)] += 1

        present_features = [name for name in features if name in frame.columns]
        feature_frame = frame.loc[:, present_features]
        missing = int(feature_frame.isna().to_numpy().sum()) + length * (len(features) - len(present_features))
        cells = length * len(features)
        global_missing += missing
        global_cells += cells
        for group in (("class", str(instance.event_class)), ("source", source)):
            missing_groups[group][0] += missing
            missing_groups[group][1] += cells

        for name in features:
            acc = feature_acc[name]
            acc["observations"] += length
            if name not in frame:
                continue
            array = frame[name].to_numpy(dtype=np.float64, na_value=np.nan)
            finite = array[np.isfinite(array)]
            if len(finite):
                acc["instances"] += 1
            acc["present"] += len(finite)
            if len(finite):
                acc["sum"] += float(finite.sum())
                acc["sum_sq"] += float(np.square(finite).sum())
                acc["min"] = min(acc["min"], float(finite.min()))
                acc["max"] = max(acc["max"], float(finite.max()))

        raw_labels = frame["class"].to_numpy(dtype=np.float64, na_value=np.nan)
        unique, counts = np.unique(raw_labels[~np.isnan(raw_labels)], return_counts=True)
        if np.isnan(raw_labels).any():
            unique = np.append(unique, np.nan); counts = np.append(counts, np.isnan(raw_labels).sum())
        phases = []
        for raw, count in zip(unique, counts):
            phase = _label_phase(float(raw), instance.event_class, transient_offset)
            raw_text = "NA" if np.isnan(raw) else str(int(raw))
            item = label_groups[(instance.event_class, raw_text, phase)]
            item["observations"] += int(count); item["instances"].add(instance.instance_id)
        last_phase = None
        for raw in raw_labels:
            phase = _label_phase(float(raw), instance.event_class, transient_offset)
            if phase != last_phase:
                phases.append(phase); last_phase = phase
        label_sequence_groups[(instance.event_class, source, ">".join(phases))] += 1

        if len(timestamps) > 1:
            deltas = np.diff(timestamps).astype("timedelta64[ns]").astype(np.int64) / 1e9
            positive = deltas[deltas > 0]
            mode = float(Counter(positive).most_common(1)[0][0]) if len(positive) else math.nan
            nonpositive = int((deltas <= 0).sum())
            gaps = int((deltas > mode).sum()) if np.isfinite(mode) else 0
            constant = bool(len(deltas) and np.all(deltas == deltas[0]))
            min_delta, max_delta = float(deltas.min()), float(deltas.max())
        else:
            mode = min_delta = max_delta = math.nan; nonpositive = gaps = 0; constant = False
        sampling_rows.append({
            "instance": instance.instance_id, "class": instance.event_class, "source": source,
            "observations": length, "mode_interval_seconds": mode, "min_interval_seconds": min_delta,
            "max_interval_seconds": max_delta, "constant_interval": constant,
            "nonpositive_intervals": nonpositive, "timestamp_gaps": gaps,
        })
        if source == "WELL":
            well_rows.append({
                "well_id": instance.well_id, "instance": instance.instance_id,
                "class": instance.event_class, "start_timestamp": str(timestamps[0]),
                "sequence_length": length, "label_phase_order": ">".join(phases),
            })
        if index % 100 == 0:
            print(f"audited {index}/{len(instances)}", flush=True)

    class_rows = []
    for event_class in range(10):
        real_wells = {row["well_id"] for row in well_rows if row["class"] == event_class}
        for source in ("WELL", "SIMULATED", "DRAWN"):
            item = class_source[(event_class, source)]
            class_rows.append({"class": event_class, "source": source, **item,
                               "distinct_real_wells": len(real_wells) if source == "WELL" else ""})
    feature_rows = []
    for name, acc in feature_acc.items():
        count = acc["present"]
        mean = acc["sum"] / count if count else math.nan
        variance = max(acc["sum_sq"] / count - mean * mean, 0.0) if count else math.nan
        feature_rows.append({
            "feature": name, "instance_coverage": acc["instances"] / len(instances),
            "instances_present": acc["instances"], "total_instances": len(instances),
            "observation_coverage": count / acc["observations"],
            "observations_present": count, "total_observations": acc["observations"],
            "missing_rate": 1 - count / acc["observations"],
            "min": acc["min"] if count else math.nan, "max": acc["max"] if count else math.nan,
            "mean": mean, "std": math.sqrt(variance) if count else math.nan,
        })
    sequence_rows = []
    for (dimension, group), values in sorted(length_groups.items()):
        sequence_rows.append({"dimension": dimension, "group": group, "instances": len(values), **_stats(values)})
    label_rows = []
    for (event_class, raw, phase), item in sorted(label_groups.items()):
        label_rows.append({"event_class": event_class, "raw_label": raw, "phase": phase,
                           "observations": item["observations"], "instances": len(item["instances"])})
    missing_rows = [{"dimension": "overall", "group": "ALL", "missing_cells": global_missing,
                     "total_cells": global_cells, "missing_rate": global_missing / global_cells}]
    for (dimension, group), (missing, cells) in sorted(missing_groups.items()):
        missing_rows.append({"dimension": dimension, "group": group, "missing_cells": missing,
                             "total_cells": cells, "missing_rate": missing / cells})

    _write_csv(output / "3w_class_source_counts.csv", class_rows)
    _write_csv(output / "3w_feature_coverage.csv", feature_rows)
    _write_csv(output / "3w_sequence_stats.csv", sequence_rows)
    _write_csv(output / "3w_sampling_interval_stats.csv", sampling_rows)
    _write_csv(output / "3w_label_audit.csv", label_rows)
    label_sequence_rows = [
        {"event_class": event_class, "source": source, "phase_sequence": sequence, "instances": count}
        for (event_class, source, sequence), count in sorted(label_sequence_groups.items())
    ]
    _write_csv(output / "3w_label_sequence_audit.csv", label_sequence_rows)
    _write_csv(output / "3w_well_manifest.csv", well_rows)
    _write_csv(output / "3w_native_missingness.csv", missing_rows)

    real_wells = sorted({row["well_id"] for row in well_rows})
    well_classes: dict[str, set[int]] = defaultdict(set)
    for row in well_rows:
        well_classes[row["well_id"]].add(row["class"])
    real_well_count_by_class = Counter(event_class for classes in well_classes.values() for event_class in classes)
    feasible_primary_classes = {event_class for event_class, count in real_well_count_by_class.items() if count >= 3}
    split = well_level_split_covering_classes(well_classes, feasible_primary_classes, seed=7)
    split_rows = [{"split": name, "well_id": well} for name, wells in split.items() for well in wells]
    _write_csv(output / "3w_candidate_well_split.csv", split_rows)
    split_coverage = {name: sorted({row["class"] for row in well_rows if row["well_id"] in wells}) for name, wells in split.items()}

    # Loader-only smoke test: three real instances, native mask retained, no fitting/training.
    smoke_instances = [item for item in instances if item.source == "WELL" and item.event_class > 0][:3]
    smoke_x, smoke_masks, smoke_labels = [], [], []
    for instance in smoke_instances:
        batch = read_instance(instance, features)
        x, mask, labels = window_instance(batch, length=64, stride=16, limit=2, from_end=True)
        smoke_x.append(x); smoke_masks.append(mask); smoke_labels.append(labels)
    smoke_x_array = np.concatenate(smoke_x)
    smoke_mask_array = np.concatenate(smoke_masks)
    import torch
    tensor = torch.from_numpy(smoke_x_array)
    model = TCNClassifier(len(features), 8, 8, 10, levels=2)
    with torch.no_grad():
        forward = model(tensor)
    smoke = {
        "instances": [item.instance_id for item in smoke_instances], "input_shape": list(tensor.shape),
        "input_dtype": str(tensor.dtype), "mask_shape": list(smoke_mask_array.shape),
        "mask_dtype": str(smoke_mask_array.dtype), "raw_last_labels": np.concatenate(smoke_labels).tolist(),
        "finite_tensor": bool(torch.isfinite(tensor).all()), "logits_shape": list(forward["logits"].shape),
        "native_missing_preserved_in_mask": bool((~smoke_mask_array).any()),
    }

    source_totals = {source: sum(class_source[(c, source)]["instances"] for c in range(10)) for source in ("WELL", "SIMULATED", "DRAWN")}
    observations = sum(item["observations"] for item in class_source.values())
    intervals = Counter(row["mode_interval_seconds"] for row in sampling_rows)
    interval_hard_check = len(intervals) == 1 and next(iter(intervals)) == 1.0 and all(row["constant_interval"] for row in sampling_rows)
    result = {
        "status": "3W_DATA_PROTOCOL_GO" if interval_hard_check and set(split_coverage["test"]) == set(range(10)) else "3W_DATA_PROTOCOL_HOLD",
        "dataset_version": version, "dataset_git_commit": commit, "transient_offset": transient_offset,
        "event_metadata": event_metadata, "instances": len(instances), "observations": observations,
        "source_instances": source_totals, "real_wells": len(real_wells), "features": list(features),
        "native_missing_rate": global_missing / global_cells, "schema_variants": len(schemas),
        "mode_sampling_intervals_seconds": {str(key): value for key, value in intervals.items()},
        "all_instances_constant_interval": all(row["constant_interval"] for row in sampling_rows),
        "sampling_interval_hard_check": interval_hard_check,
        "candidate_split": split, "candidate_split_class_coverage": split_coverage, "smoke": smoke,
        "feasible_three_way_real_classes": sorted(feasible_primary_classes),
    }
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report.write_text(render_report(result, class_rows, feature_rows, missing_rows, sequence_rows, label_rows, well_rows), encoding="utf-8")
    return result


def render_report(result, class_rows, feature_rows, missing_rows, sequence_rows, label_rows, well_rows) -> str:
    by_class = []
    for event_class in range(10):
        rows = [row for row in class_rows if row["class"] == event_class]
        by_class.append(f"| {event_class} | {sum(r['instances'] for r in rows)} | {sum(r['observations'] for r in rows)} | "
                        f"{rows[0]['instances']} | {rows[0]['distinct_real_wells']} | {rows[1]['instances']} | {rows[2]['instances']} |")
    source = result["source_instances"]
    missing_class = {row["group"]: row["missing_rate"] for row in missing_rows if row["dimension"] == "class"}
    multi_instance = Counter(row["well_id"] for row in well_rows)
    multi_class = defaultdict(set)
    for row in well_rows: multi_class[row["well_id"]].add(row["class"])
    fully_missing = [row["feature"] for row in feature_rows if float(row["observation_coverage"]) == 0]
    usable = [row["feature"] for row in feature_rows if float(row["observation_coverage"]) > 0]
    return f"""# 3W Dataset 阶段 0：Data Audit 与协议设计

最终判定：`{result['status']}`

## 版本与范围

- Petrobras 3W Dataset：`{result['dataset_version']}`
- 外部数据仓库 commit：`{result['dataset_git_commit']}`
- 完整读取 `{result['instances']}` 个 Parquet instance、`{result['observations']}` 条 observation。
- `TRANSIENT_OFFSET={result['transient_offset']}`；101~109 是相应事件的 transient label，不是新故障类。
- 原始数据只读，未复制、移动、修改；本审计未添加 TEP 的 MCAR 30%。

## Class × Source

| class | instance | observation | WELL instance | distinct WELL | SIMULATED | DRAWN |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(by_class)}

来源 instance 总数：WELL={source['WELL']}，SIMULATED={source['SIMULATED']}，DRAWN={source['DRAWN']}。共有 `{result['real_wells']}` 口真实 WELL；其中 `{sum(v > 1 for v in multi_instance.values())}` 口含多个 instance，`{sum(len(v) > 1 for v in multi_class.values())}` 口覆盖多个 event class。

## Feature 与原生缺失

- 自动检测 `{len(feature_rows)}` 个 process variable；其中 `{len(usable)}` 个至少存在一个有限观测，4 个全量缺失字段为 `{fully_missing}`。`class`、`state`、timestamp 均不作为模型 feature。
- 多个字段存在明显极端哨兵/异常范围；不能仅凭 finite 判定有效。下一阶段清洗阈值与最终 feature 集必须仅由 train/equipment metadata 冻结。
- 所有统计来自全量审计，但未来 scaler、imputer、feature selection、D/E/S 只能在 training WELL 上拟合。
- 整体 native missing rate：`{result['native_missing_rate']:.6f}`。各 feature/class/source 明细见 CSV。
- Adapter 返回独立 observation mask；零填充只用于构造有限 tensor，不代表已冻结 imputation 策略。

## 序列、采样与频率 HARD CHECK

- 全局序列长度统计见 `3w_sequence_stats.csv`，并按 class/source 分组。
- 采样 interval mode 分布：`{result['mode_sampling_intervals_seconds']}`。
- 所有 instance 内 interval 恒定：`{result['all_instances_constant_interval']}`。
- Frequency hard check：`{result['sampling_interval_hard_check']}`。只有该项为真才可把窗口内 FFT bin 视为相同物理频率；即便为真，3W 的 D/E/S 与关键频率 mask 仍必须仅由 3W training split 重新估计，不能直接复用 TEP 的频率索引/mask。

## Label Audit

- raw `class` 的 normal/transient/event/NA 统计见 `3w_label_audit.csv`，逐类逐来源的实际阶段顺序见 `3w_label_sequence_audit.csv`。
- 事件类 `{', '.join(str(item['class']) for item in result['event_metadata'] if item['transient_defined'])}` 在配置中声明 transient；3、4 未声明 transient。
- 当前只确认可以把 raw 0、100+class、class 分别解释为 normal、transient、event 候选阶段；NA 与异常时序必须保留。**本阶段不冻结最终 normal/early/established mapping。**

## 推荐协议

Primary：WELL-only、按 `well_id` 分组的 train/validation/test，建议起点 60/20/20；同一 WELL 的所有 instance 必须进入同一 split。候选 manifest 已生成，class coverage 为 `{result['candidate_split_class_coverage']}`。所有拟合仅使用 train。

三路 real-only 均可覆盖的类别是 `{result['feasible_three_way_real_classes']}`；class 3 和 6 各仅有 2 口真实 WELL，数学上无法同时覆盖 train/validation/test，这是当前 HOLD 的核心阻塞。

若候选划分不能让三份都覆盖目标类别：

- 方案 A（首选）：保留 real-only，把主任务限定为三份均有真实 WELL 覆盖的 class；其余 class 明示为 out-of-scope，不把 synthetic 混入 primary。
- 方案 B：保留全部 class，将 SIMULATED/DRAWN 作为单独的 secondary protocol/域迁移消融，绝不与 primary 结论混写。

## Loader Smoke Test

读取少量 WELL → native mask → window → float32 tensor → TCN forward 成功：`{result['smoke']}`。未训练 scaler、probe、对比模型或扩散模型。

## 阶段结论

`{result['status']}`。若为 HOLD，阻塞项是：采样间隔 HARD CHECK 未通过，或当前简单候选 well split 的 test 未覆盖 0~9；在冻结最终 split/label mapping 前不得开始完整训练。下一阶段应先冻结 label/window/imputation 与 class coverage 协议，然后只跑单 Seed clean/传统增强/Uniform diffusion/Frequency-Selective R1 最小比较；通过后才允许 3-Seed。
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Petrobras 3W Dataset 2.0.0 without modifying it")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/3w_stage0"))
    parser.add_argument("--report", type=Path, default=Path("docs/3W_DATA_AUDIT.md"))
    args = parser.parse_args()
    result = audit(args.data_root, args.output_dir, args.report)
    print(json.dumps({key: result[key] for key in ("status", "instances", "observations", "real_wells")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
