from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import yaml

from scripts.run_r1_des_weight_search import DATASETS, validation_metrics


METRICS = ("macro_f1", "far", "early_recall", "auprc")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest(path: Path) -> dict[str, Any]:
    return _read(path)["results"]


def _fmt(values: list[float]) -> str:
    return f"{mean(values):.4f} ± {stdev(values):.4f}"


def audit_records(config: dict[str, Any], manifests: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    expected = {"3W": list(map(int, config["three_w"]["seeds"])),
                "TEP": list(map(int, config["tep"]["seeds"]))}
    selected = ["CURRENT", *selection["top3"]]
    for dataset in DATASETS:
        records = manifests[dataset]
        for record in records.values():
            validation_metrics(dataset, record)
        for variant in selected:
            present = sorted(record["seed"] for record in records.values() if record["variant"] == variant)
            if present != sorted(expected[dataset]):
                raise RuntimeError(f"incomplete Stage 2 {dataset} {variant}: {present}")
            hashes = {record["mask_sha256"] for record in records.values() if record["variant"] == variant}
            if len(hashes) != 1:
                raise RuntimeError(f"mask changed across seeds: {dataset} {variant}")
        for seed in expected[dataset]:
            seed_records = [record for record in records.values()
                            if record["variant"] in selected and int(record["seed"]) == seed]
            if dataset == "3W":
                keys = ("initialization_sha256", "window_refs_sha256", "supcon_batch_order_sha256")
            else:
                keys = ("manifest_sha256", "initialization_sha256", "pretrain_order_sha256", "probe_order_sha256")
            for key in keys:
                values = {json.dumps(record["fairness"].get(key), sort_keys=True) for record in seed_records}
                if len(values) != 1:
                    raise RuntimeError(f"fairness mismatch: {dataset} seed={seed} {key}")
    mask_audit = _read(Path(config["output"]["mask_audit"]))
    max_budget_error = max(float(item["total_budget_error"])
                           for dataset in DATASETS for item in mask_audit[dataset].values())
    if max_budget_error > float(config["selection"]["budget_tolerance"]):
        raise RuntimeError("matched budget audit failed")
    return {"validation_only": True, "test_metrics_read": False, "fixed_mask_across_seeds": True,
            "fairness_hashes_consistent": True, "max_budget_error": max_budget_error}


def result_rows(config: dict[str, Any], manifests: dict[str, Any], selection: dict[str, Any]) -> list[dict[str, Any]]:
    selected = {"CURRENT", *selection["top3"]}
    stage1_seed = {"3W": int(config["three_w"]["stage1_seed"]), "TEP": int(config["tep"]["stage1_seed"])}
    rows = []
    for dataset in DATASETS:
        for record in manifests[dataset].values():
            metrics = validation_metrics(dataset, record)
            rows.append({"dataset": dataset, "variant": record["variant"], "seed": record["seed"],
                         "stage1_screen": int(record["seed"]) == stage1_seed[dataset],
                         "stage2_selected": record["variant"] in selected, **metrics})
    return sorted(rows, key=lambda row: (row["dataset"], row["variant"], int(row["seed"])))


def stage2_rows(config: dict[str, Any], manifests: dict[str, Any], selection: dict[str, Any]) -> list[dict[str, Any]]:
    variants = ["CURRENT", *selection["top3"]]
    rows = []
    for variant in variants:
        row: dict[str, Any] = {"variant": variant}
        for dataset, config_key in (("3W", "three_w"), ("TEP", "tep")):
            records = manifests[dataset]
            current_by_seed = {int(record["seed"]): validation_metrics(dataset, record)
                               for record in records.values() if record["variant"] == "CURRENT"}
            values = {metric: [] for metric in METRICS}
            deltas = {metric: [] for metric in METRICS}
            for seed in map(int, config[config_key]["seeds"]):
                record = records[f"{variant}|{seed}"]
                metrics = validation_metrics(dataset, record)
                for metric in METRICS:
                    values[metric].append(metrics[metric])
                    deltas[metric].append(metrics[metric] - current_by_seed[seed][metric])
            for metric in METRICS:
                prefix = dataset.lower()
                row[f"{prefix}_{metric}"] = _fmt(values[metric])
                row[f"{prefix}_{metric}_paired_delta"] = mean(deltas[metric])
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _markdown_stage2(rows: list[dict[str, Any]]) -> str:
    lines = ["| Variant | 3W Macro-F1 | Δ | 3W FAR | Δ | 3W Early | TEP Macro-F1 | Δ | TEP FAR | Δ | TEP Early |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append("| " + " | ".join((
            row["variant"], row["3w_macro_f1"], f"{row['3w_macro_f1_paired_delta']:+.4f}",
            row["3w_far"], f"{row['3w_far_paired_delta']:+.4f}", row["3w_early_recall"],
            row["tep_macro_f1"], f"{row['tep_macro_f1_paired_delta']:+.4f}",
            row["tep_far"], f"{row['tep_far_paired_delta']:+.4f}", row["tep_early_recall"],
        )) + " |")
    return "\n".join(lines)


def write_final_config(config: dict[str, Any], selection: dict[str, Any], audit: dict[str, Any]) -> None:
    variant = selection["final_variant"]
    weights = config["variants"][variant]
    final = {
        "mode": "FINAL_QDIFFCL",
        "frozen": True,
        "selected_variant": variant,
        "selected_by": "train_plus_validation_only",
        "test_metrics_used_for_selection": False,
        "weights": weights,
        "criticality": config["criticality_base"],
        "spectral_diffusion": config["spectral_diffusion"],
        "training_protocol": {
            "backbone": "TCN", "contrastive": "Hard SupCon", "batching": "original",
            "probe": "frozen linear probe", "three_w_base_config": config["three_w"]["base_config"],
            "tep_base_config": config["tep"]["base_config"],
        },
        "mask_sha256": {dataset: _read(Path(config["output"]["mask_audit"]))[dataset][variant]["mask_sha256"]
                        for dataset in DATASETS},
        "selection_manifest": config["output"]["selection"],
        "fairness_audit": audit,
        "paper_final_protocol": {
            "weights_must_not_change_after_test_or_baseline_results": True,
            "next_experiments": ["external_baselines", "five_seed_reliability", "paper_final"],
            "historical_current_r1_results_preserved": True,
        },
    }
    Path(config["docs"]["final_config"]).write_text(
        yaml.safe_dump(final, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def write_summary(config: dict[str, Any], selection: dict[str, Any], stage2: list[dict[str, Any]], audit: dict[str, Any]) -> None:
    ranking_lines = ["| Rank | Candidate | FAR gate | Relative Macro gain | FAR delta | Early delta | AUPRC delta |",
                     "|---:|---|---|---:|---:|---:|---:|"]
    for index, row in enumerate(selection["ranking"], 1):
        ranking_lines.append(f"| {index} | {row['variant']} | {row['far_gate_pass']} | "
                             f"{row['mean_relative_macro_gain']:+.4%} | {row['mean_far_delta']:+.4f} | "
                             f"{row['mean_early_delta']:+.4f} | {row['mean_auprc_delta']:+.4f} |")
    decision_lines = []
    for row in selection["final_decisions"]:
        decision_lines.append(f"- {row['variant']}：relative Macro gain {row['mean_relative_macro_gain']:+.4%}；"
                              f"Macro floor={row['macro_floor_pass']}，FAR={row['far_gate_pass']}，"
                              f"consistency={row['consistency_pass']}，clear gain={row['clear_gain_pass']}，"
                              f"最终 eligible={row['eligible']}。")
    final = selection["final_variant"]
    weights = selection["final_weights"]
    text = [
        "# R1 DES 权重 Validation 搜索与最终冻结", "",
        "本轮只使用 canonical train split 构建 mask，并且只使用 validation 指标筛选与冻结；未计算或读取 test 指标参与排序。", "",
        "## Stage 1：12 候选单种子筛选", "", *ranking_lines, "",
        f"Top-3：`{selection['top3'][0]}`、`{selection['top3'][1]}`、`{selection['top3'][2]}`。", "",
        "## Stage 2：Top-3 + CURRENT 三随机种子", "", _markdown_stage2(stage2), "",
        "表中 Δ 均为相对 CURRENT 的同 seed paired mean；FAR 越低越好。完整 AUPRC 和每 seed 数据见 CSV。", "",
        "## 一次性最终决策", "", *decision_lines, "",
        f"最终冻结：`{final}`，D/E/S = `{weights[0]:.3f}/{weights[1]:.3f}/{weights[2]:.3f}`。",
        f"决策原因：{selection['final_reason']}。", "",
        "权重自此停止调参。后续 test、external baseline 或 reliability 结果不得触发再次改权重。", "",
        "## 审计", "",
        f"- validation-only：{audit['validation_only']}；test metrics read：{audit['test_metrics_read']}。",
        f"- candidate mask 跨 seed 固定：{audit['fixed_mask_across_seeds']}。",
        f"- initialization / batch order / preprocessing 公平性哈希一致：{audit['fairness_hashes_consistent']}。",
        f"- 最大 matched-budget error：{audit['max_budget_error']:.3e}。", "",
        "现有 R1 test 结果仅保留为历史开发证据，未参与本轮候选排序。下一步进入 external baseline、5-seed reliability 与 paper-final 实验。", "",
    ]
    Path(config["docs"]["summary"]).write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/r1_des_weight_search.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    selection = _read(Path(config["output"]["selection"]))
    if not selection.get("weights_frozen"):
        raise RuntimeError("Stage 2 final freeze has not completed")
    manifests = {"3W": _manifest(Path(config["three_w"]["output_dir"]) / "manifest.json"),
                 "TEP": _manifest(Path(config["tep"]["output_dir"]) / "manifest.json")}
    audit = audit_records(config, manifests, selection)
    rows = result_rows(config, manifests, selection)
    stage2 = stage2_rows(config, manifests, selection)
    _write_csv(Path(config["docs"]["results_csv"]), rows)
    _write_csv(Path(config["docs"]["stage2_table_csv"]), stage2)
    write_final_config(config, selection, audit)
    write_summary(config, selection, stage2, audit)
    print(json.dumps({"stage1_candidates": selection["stage1_candidates"], "top3": selection["top3"],
                      "final_variant": selection["final_variant"], "audit": audit}, ensure_ascii=False))


if __name__ == "__main__":
    main()
