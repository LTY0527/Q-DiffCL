from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from scipy.stats import spearmanr

from scripts.audit_qdiffcl_data_regime import atomic_json


METRICS = ("macro_f1", "auprc", "far", "early_recall", "detection_delay")
CONTRASTS = (
    ("FINAL_QDIFFCL_FIXED", "NO_AUG"),
    ("FINAL_QDIFFCL_FIXED", "UNIFORM_DIFFUSION"),
    ("FINAL_QDIFFCL_FIXED", "JITTER_SCALING"),
    ("CALIBRATED_RHO", "FINAL_QDIFFCL_FIXED"),
    ("CALIBRATED_RHO", "UNIFORM_DIFFUSION"),
    ("CALIBRATED_RHO", "NO_AUG"),
    ("CALIBRATED_RHO", "JITTER_SCALING"),
)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(fields or (rows[0].keys() if rows else []))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def collect_results(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("*/f*/outer_*/model_seed_*/*/result.json")):
        value = json.loads(path.read_text(encoding="utf-8")); value["result_path"] = str(path); rows.append(value)
    return rows


def raw_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for value in records:
        row = {key: value[key] for key in ("cell_id", "dataset", "fraction", "outer_id", "method", "model_seed")}
        row["selected_rho"] = value.get("selected_rho")
        for metric in METRICS:
            row[metric] = value["test_metrics"].get(metric)
        row.update({
            "checkpoint_sha256": value["checkpoint_sha256"], "prediction_sha256": value["prediction_sha256"],
            "fraction_manifest_hash": value["fraction_manifest_hash"], "criticality_mask_sha256": value["criticality_mask_sha256"],
            "result_path": value["result_path"],
        })
        rows.append(row)
    return rows


def split_first_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], float(row["fraction"]), row["method"])].append(row)
    output = []
    for (dataset, fraction, method), current in sorted(grouped.items()):
        record: dict[str, Any] = {"dataset": dataset, "fraction": fraction, "method": method}
        for metric in METRICS:
            by_outer = []
            for outer in sorted({int(row["outer_id"]) for row in current}):
                values = [float(row[metric]) for row in current if int(row["outer_id"]) == outer and row[metric] is not None]
                if values:
                    by_outer.append(float(np.mean(values)))
            record[f"{metric}_mean"] = float(np.mean(by_outer)) if by_outer else None
            record[f"{metric}_sd"] = float(np.std(by_outer, ddof=1)) if len(by_outer) > 1 else None
        record["outer_count"] = len({row["outer_id"] for row in current})
        record["cell_count"] = len(current); output.append(record)
    return output


def _group_delta_rows(records: list[dict[str, Any]], left: str, right: str, fraction: float) -> list[dict[str, Any]]:
    indexed = {(row["dataset"], float(row["fraction"]), int(row["outer_id"]), int(row["model_seed"]), row["method"]): row for row in records}
    output = []
    keys = sorted({key[:4] for key in indexed if key[1] == fraction})
    for dataset, current_fraction, outer, seed in keys:
        a = indexed.get((dataset, current_fraction, outer, seed, left)); b = indexed.get((dataset, current_fraction, outer, seed, right))
        if a is None or b is None:
            continue
        ga = {str(row["group_id"]): row for row in a["groupwise"]}; gb = {str(row["group_id"]): row for row in b["groupwise"]}
        for group in sorted(set(ga) & set(gb)):
            output.append({"dataset": dataset, "fraction": current_fraction, "outer_id": outer, "model_seed": seed,
                           "group_id": group, "delta": float(ga[group]["macro_f1"]) - float(gb[group]["macro_f1"])})
    return output


def cluster_bootstrap_ci(rows: list[dict[str, Any]], repeats: int, seed: int) -> tuple[float | None, float | None]:
    if not rows:
        return None, None
    rng = np.random.default_rng(seed)
    by_outer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_outer[int(row["outer_id"])].append(row)
    outers = sorted(by_outer); estimates = []
    for _ in range(repeats):
        sampled_outer = rng.choice(outers, size=len(outers), replace=True)
        outer_values = []
        for outer in sampled_outer:
            current = by_outer[int(outer)]; groups = sorted({row["group_id"] for row in current})
            sampled_groups = rng.choice(groups, size=len(groups), replace=True)
            values = [row["delta"] for group in sampled_groups for row in current if row["group_id"] == group]
            outer_values.append(float(np.mean(values)))
        estimates.append(float(np.mean(outer_values)))
    low, high = np.quantile(estimates, [.025, .975])
    return float(low), float(high)


def paired_summary(records: list[dict[str, Any]], repeats: int, seed: int) -> list[dict[str, Any]]:
    indexed = {(row["dataset"], float(row["fraction"]), int(row["outer_id"]), int(row["model_seed"]), row["method"]): row for row in records}
    output = []
    for dataset in sorted({row["dataset"] for row in records}):
        for fraction in sorted({float(row["fraction"]) for row in records if row["dataset"] == dataset}, reverse=True):
            for left, right in CONTRASTS:
                cells = []
                for key, a in indexed.items():
                    if key[0] != dataset or key[1] != fraction or key[4] != left:
                        continue
                    b = indexed.get((dataset, fraction, key[2], key[3], right))
                    if b is not None:
                        cells.append((float(a["test_metrics"]["macro_f1"]) - float(b["test_metrics"]["macro_f1"]), key[2], key[3]))
                if not cells:
                    continue
                group_rows = [row for row in _group_delta_rows(records, left, right, fraction) if row["dataset"] == dataset]
                low, high = cluster_bootstrap_ci(group_rows, repeats, seed)
                worst = min(cells)
                output.append({
                    "dataset": dataset, "fraction": fraction, "contrast": f"{left} - {right}",
                    "paired_delta": float(np.mean([item[0] for item in cells])), "ci_low": low, "ci_high": high,
                    "positive_cells": sum(item[0] > 0 for item in cells), "non_worse_cells": sum(item[0] >= 0 for item in cells),
                    "cell_count": len(cells), "worst_delta": worst[0], "worst_outer": worst[1], "worst_seed": worst[2],
                })
    return output


def scarcity_dod(records: list[dict[str, Any]], repeats: int, seed: int) -> list[dict[str, Any]]:
    indexed = {(row["dataset"], float(row["fraction"]), int(row["outer_id"]), int(row["model_seed"]), row["method"]): row for row in records}
    output = []
    for dataset in sorted({row["dataset"] for row in records}):
        fractions = sorted({float(row["fraction"]) for row in records if row["dataset"] == dataset and float(row["fraction"]) < 1})
        for fraction in fractions:
            for left, right in CONTRASTS[:4]:
                cells = []
                for outer in sorted({int(row["outer_id"]) for row in records if row["dataset"] == dataset}):
                    for model_seed in sorted({int(row["model_seed"]) for row in records if row["dataset"] == dataset}):
                        keys = [(dataset, f, outer, model_seed, method) for f, method in ((fraction, left), (fraction, right), (1., left), (1., right))]
                        if all(key in indexed for key in keys):
                            low_delta = float(indexed[keys[0]]["test_metrics"]["macro_f1"]) - float(indexed[keys[1]]["test_metrics"]["macro_f1"])
                            full_delta = float(indexed[keys[2]]["test_metrics"]["macro_f1"]) - float(indexed[keys[3]]["test_metrics"]["macro_f1"])
                            cells.append({"outer_id": outer, "model_seed": model_seed, "delta": low_delta - full_delta})
                if not cells:
                    continue
                group_low = _group_delta_rows(records, left, right, fraction); group_full = _group_delta_rows(records, left, right, 1.)
                full_map = {(row["dataset"], row["outer_id"], row["model_seed"], row["group_id"]): row["delta"] for row in group_full}
                group_dod = [{**row, "delta": row["delta"] - full_map[(dataset, row["outer_id"], row["model_seed"], row["group_id"])]}
                             for row in group_low if row["dataset"] == dataset and (dataset, row["outer_id"], row["model_seed"], row["group_id"]) in full_map]
                low_ci, high_ci = cluster_bootstrap_ci(group_dod, repeats, seed)
                outer_direction = sum(np.mean([row["delta"] for row in cells if row["outer_id"] == outer]) > 0 for outer in {row["outer_id"] for row in cells})
                output.append({
                    "dataset": dataset, "fraction": fraction, "contrast": f"{left} - {right}",
                    "scarcity_dod": float(np.mean([row["delta"] for row in cells])), "ci_low": low_ci, "ci_high": high_ci,
                    "same_direction_outers": outer_direction, "total_outers": len({row["outer_id"] for row in cells}),
                    "same_direction_paired_cells": sum(row["delta"] > 0 for row in cells), "total_paired_cells": len(cells),
                    "paired_direction_win_rate": float(np.mean([row["delta"] > 0 for row in cells])),
                })
    return output


def rho_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("*/f*/outer_*/rho_selection.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        selected = next(row for row in value["candidate_rows"] if float(row["rho"]) == float(value["selected_rho"]))
        rows.append({"dataset": value["dataset"], "fraction": value["fraction"], "outer_id": value["outer_id"],
                     "rho_star": value["selected_rho"], "validation_macro_f1": selected["macro_f1"],
                     "validation_auprc": selected["auprc"], "validation_far": selected["far"],
                     "historical_dcbr_global_rho": value["historical_dcbr_global_rho"]})
    return rows


def mask_stability(root: Path) -> list[dict[str, Any]]:
    output = []
    for dataset_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        outer_ids = sorted({path.name for path in dataset_dir.glob("f*/outer_*")})
        for outer_name in outer_ids:
            entries = {}
            for path in dataset_dir.glob(f"f*/{outer_name}/_context/criticality.npz"):
                fraction = int(path.parents[2].name[1:]) / 100
                audit = json.loads((path.parent / "audit.json").read_text(encoding="utf-8"))
                with np.load(path) as data:
                    entries[fraction] = (data["composite"].reshape(-1), set(audit["top_frequency_flat_indices"]))
            for high, low in ((1., .25), (1., .10), (.25, .10)):
                if high not in entries or low not in entries:
                    continue
                a, atop = entries[high]; b, btop = entries[low]
                union = atop | btop
                output.append({"dataset": dataset_dir.name.upper(), "outer_id": int(outer_name.split("_")[1]),
                               "fraction_a": high, "fraction_b": low,
                               "jaccard": len(atop & btop) / len(union) if union else 1.,
                               "rank_correlation": float(spearmanr(a, b).statistic),
                               "changed_bins": len(atop ^ btop)})
    return output


def write_docs(summary: list[dict[str, Any]], paired: list[dict[str, Any]], dod: list[dict[str, Any]], rho: list[dict[str, Any]], complete: bool) -> None:
    lines = ["# Q-DiffCL Data-Regime Report", "", f"Status: `{'QDIFFCL_DATA_REGIME_V1_COMPLETE' if complete else 'QDIFFCL_DATA_REGIME_V1_RESUMABLE'}`.", "",
             "TEP 10% is excluded from the primary D+E matrix by the preregistered E-identifiability hold.", "", "## Macro-F1 split-first summary", "",
             "| Dataset | Fraction | Method | Mean | SD | Cells |", "|---|---:|---|---:|---:|---:|"]
    for row in summary:
        if row["macro_f1_mean"] is not None:
            lines.append(f"| {row['dataset']} | {row['fraction']:.2f} | {row['method']} | {row['macro_f1_mean']:.6f} | {row['macro_f1_sd'] if row['macro_f1_sd'] is not None else 'NA'} | {row['cell_count']} |")
    lines.extend(["", "Paired contrasts, group-aware CIs, scarcity DoD, direction consistency, and worst-cell fields are stored in the registered analysis CSV files."])
    Path("docs/QDIFFCL_DATA_REGIME_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    rho_lines = ["# Q-DiffCL Data-Regime Rho Curve", "", "All rho values below were selected from validation only.", "",
                 "| Dataset | Fraction | Outer | rho* | Val Macro-F1 | Val AUPRC | Val FAR |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rho:
        rho_lines.append(f"| {row['dataset']} | {row['fraction']:.2f} | {row['outer_id']} | {row['rho_star']:.2f} | {row['validation_macro_f1']:.6f} | {row['validation_auprc']:.6f} | {row['validation_far']:.6f} |")
    Path("docs/QDIFFCL_DATA_REGIME_RHO_CURVE.md").write_text("\n".join(rho_lines) + "\n", encoding="utf-8")


def summarize(config_path: str | Path, allow_incomplete: bool) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")); root = Path(config["output"]["root"]) / config["output"]["namespace"]
    records = collect_results(root); raw = raw_rows(records); summary = split_first_summary(raw)
    repeats = int(config["metrics"]["bootstrap_repeats"]); seed = int(config["metrics"]["bootstrap_seed"])
    paired = paired_summary(records, repeats, seed); dod = scarcity_dod(records, repeats, seed); rho = rho_rows(root)
    masks = mask_stability(root) if root.exists() else []
    run_manifest = json.loads(Path(config["output"]["manifest"]).read_text(encoding="utf-8"))
    expected = int(run_manifest["accounting"]["formal_cells_expected"]); complete = len(records) == expected
    if not complete and not allow_incomplete:
        raise RuntimeError(f"formal matrix incomplete: {len(records)}/{expected}")
    _write_csv(Path(config["output"]["raw_csv"]), raw)
    _write_csv(Path(config["output"]["summary_csv"]), summary)
    _write_csv(Path(config["output"]["paired_csv"]), paired)
    _write_csv(Path(config["output"]["scarcity_dod_csv"]), dod)
    _write_csv(Path(config["output"]["rho_csv"]), rho)
    _write_csv(Path("analysis/results/qdiffcl_data_regime_mask_stability.csv"), masks)
    write_docs(summary, paired, dod, rho, complete)
    status = {"status": "QDIFFCL_DATA_REGIME_V1_COMPLETE" if complete else "QDIFFCL_DATA_REGIME_V1_RESUMABLE",
              "formal_results": len(records), "formal_expected": expected, "rho_selections": len(rho)}
    atomic_json(Path("analysis/results/qdiffcl_data_regime_summary.json"), status)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize locked Q-DiffCL Data-Regime results")
    parser.add_argument("--config", default="configs/qdiffcl_data_regime_v1.yaml")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args(); print(json.dumps(summarize(args.config, args.allow_incomplete), indent=2))


if __name__ == "__main__":
    main()
