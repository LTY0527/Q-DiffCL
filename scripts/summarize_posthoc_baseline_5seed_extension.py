from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.metrics import confusion_matrix

from scripts.run_posthoc_baseline_5seed_extension import (
    CELLS_COMPLETE,
    EVIDENCE_CLASS,
    canonical_hash,
    locked_protocol,
    read_json,
    sha256_file,
    validate_static,
)
from scripts.summarize_posthoc_recent_baselines import METRICS, load_posthoc_records, split_first_summary


COMPLETE = "POSTHOC_BASELINE_5SEED_EXTENSION_COMPLETE"


def extension_cells_complete(status: Any) -> bool:
    return status in {CELLS_COMPLETE, COMPLETE}


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any) -> float | None:
    if value in (None, "", "None", "nan"):
        return None
    return float(value)


def load_extension_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    manifest = read_json(config["output"]["manifest"])
    if not extension_cells_complete(manifest.get("status")):
        raise RuntimeError(f"extension cells are not complete: {manifest.get('status')}")
    if manifest.get("protocol_hash") != canonical_hash(locked_protocol(config)):
        raise RuntimeError("extension manifest protocol hash changed")
    cells = manifest.get("cells", [])
    if len(cells) != 48 or any(cell.get("status") != "complete" for cell in cells):
        raise RuntimeError("extension manifest is not 48/48 complete")
    records: list[dict[str, Any]] = []
    for cell in cells:
        record = read_json(cell["result_path"])
        if record.get("evidence_class") != EVIDENCE_CLASS or record.get("outer_test_evaluated_once") is not True:
            raise RuntimeError(f"extension evidence boundary invalid: {cell['run_id']}")
        if record.get("extension_provenance", {}).get("protocol_hash") != manifest["protocol_hash"]:
            raise RuntimeError(f"extension provenance mismatch: {cell['run_id']}")
        if sha256_file(record["prediction_path"]) != record["prediction_sha256"]:
            raise RuntimeError(f"extension prediction hash mismatch: {cell['run_id']}")
        if sha256_file(record["checkpoint_path"]) != record["checkpoint_sha256"]:
            raise RuntimeError(f"extension checkpoint hash mismatch: {cell['run_id']}")
        records.append(record)
    keys = {(row["dataset"], int(row["outer_seed"]), int(row["model_seed"]), row["method"]) for row in records}
    if len(keys) != 48:
        raise RuntimeError("extension result keys are not unique")
    return records


def load_h1_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    h1_config = yaml.safe_load(Path(config["h1_config"]).read_text(encoding="utf-8"))
    records = load_posthoc_records(h1_config)
    if len(records) != 72:
        raise RuntimeError("H1 reuse is not exactly 72 records")
    return records


def load_paper_final_references(config: dict[str, Any]) -> list[dict[str, Any]]:
    methods = set(config["statistics"]["reference_methods"])
    seeds = {dataset: set(map(int, config["full_seeds"][dataset])) for dataset in ("3W", "TEP")}
    outers = {dataset: set(map(int, config["outer_splits"][dataset])) for dataset in ("3W", "TEP")}
    rows: list[dict[str, Any]] = []
    with Path(config["paper_final_raw_csv"]).open(encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            dataset = raw["dataset"]
            if raw["method"] not in methods:
                continue
            if int(raw["model_seed"]) not in seeds[dataset] or int(raw["outer_seed"]) not in outers[dataset]:
                continue
            row = {
                "run_id": raw["run_id"],
                "dataset": dataset,
                "outer_seed": int(raw["outer_seed"]),
                "model_seed": int(raw["model_seed"]),
                "method": raw["method"],
                "track": "TRACK_A_FROZEN_PAPER_FINAL_REFERENCE",
                "metrics": {metric: as_float(raw.get(metric)) for metric in METRICS},
                "prediction_path": raw["prediction_path"],
                "prediction_sha256": raw["prediction_sha256"],
                "checkpoint_sha256": raw["checkpoint_sha256"],
                "evidence_source": "FROZEN_PAPER_FINAL_5SEED_REUSE",
            }
            if sha256_file(row["prediction_path"]) != row["prediction_sha256"]:
                raise RuntimeError(f"Paper-final prediction hash mismatch: {row['run_id']}")
            rows.append(row)
    expected = 2 * 3 * 5 * len(methods)
    keys = {(row["dataset"], row["outer_seed"], row["model_seed"], row["method"]) for row in rows}
    if len(rows) != expected or len(keys) != expected:
        raise RuntimeError(f"Paper-final five-seed reuse is not {expected} unique rows: {len(rows)}")
    return rows


def combined_raw_rows(
    h1_records: list[dict[str, Any]],
    extension_records: list[dict[str, Any]],
    references: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    source_by_id = {
        **{row["run_id"]: "H1_3SEED_POSTHOC_REUSE" for row in h1_records},
        **{row["run_id"]: "H1_5SEED_EXTENSION_NEW" for row in extension_records},
    }
    for row in [*h1_records, *extension_records, *references]:
        output.append({
            "run_id": row["run_id"],
            "dataset": row["dataset"],
            "outer_seed": int(row["outer_seed"]),
            "model_seed": int(row["model_seed"]),
            "method": row["method"],
            "track": row["track"],
            "evidence_source": (
                row["evidence_source"] if "evidence_source" in row else source_by_id[row["run_id"]]
            ),
            **{metric: row["metrics"].get(metric) for metric in METRICS},
            "prediction_path": row["prediction_path"],
            "prediction_sha256": row["prediction_sha256"],
            "checkpoint_sha256": row["checkpoint_sha256"],
        })
    return sorted(output, key=lambda row: (row["dataset"], row["method"], row["outer_seed"], row["model_seed"]))


def add_worst_cell_identity(summary: list[dict[str, Any]], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in summary:
        if row["level"] != "overall":
            continue
        candidates = [
            item for item in records
            if item["dataset"] == row["dataset"] and item["method"] == row["method"]
        ]
        worst = min(candidates, key=lambda item: float(item["metrics"]["macro_f1"]))
        row["worst_outer_seed"] = int(worst["outer_seed"])
        row["worst_model_seed"] = int(worst["model_seed"])
        row["worst_run_id"] = worst["run_id"]
    return summary


def macro_f1_from_confusions(confusions: np.ndarray) -> np.ndarray:
    values = np.asarray(confusions, dtype=np.float64)
    true_positive = np.diagonal(values, axis1=-2, axis2=-1)
    predicted = values.sum(axis=-2)
    actual = values.sum(axis=-1)
    denominator = predicted + actual
    f1 = np.divide(2 * true_positive, denominator, out=np.zeros_like(true_positive), where=denominator > 0)
    return f1.mean(axis=-1)


def group_confusions(record: dict[str, Any], classes: int) -> tuple[list[str], np.ndarray]:
    with np.load(record["prediction_path"], allow_pickle=False) as archive:
        groups = archive["group_id"].astype(str)
        labels = archive["label"].astype(int)
        prediction = archive["prediction"].astype(int)
    unique = sorted(set(groups))
    matrices = [
        confusion_matrix(labels[groups == group], prediction[groups == group], labels=np.arange(classes))
        for group in unique
    ]
    return unique, np.stack(matrices)


def group_bootstrap(
    records: list[dict[str, Any]],
    repeats: int,
    seed: int,
    comparison_methods: list[str],
) -> list[dict[str, Any]]:
    by_key = {
        (row["dataset"], int(row["outer_seed"]), int(row["model_seed"]), row["method"]): row
        for row in records
    }
    cache: dict[tuple[str, int, int, str], tuple[list[str], np.ndarray]] = {}
    output: list[dict[str, Any]] = []
    for dataset in ("3W", "TEP"):
        classes = 4 if dataset == "3W" else 2
        for method_index, method in enumerate(comparison_methods):
            keys = sorted(key for key in by_key if key[0] == dataset and key[3] == method)
            if len(keys) != 15:
                raise RuntimeError(f"paired matrix for {dataset}/{method} has {len(keys)}/15 cells")
            draws_by_cell: list[np.ndarray] = []
            observed: list[float] = []
            for _, outer, model_seed, _ in keys:
                key_a = (dataset, outer, model_seed, method)
                key_b = (dataset, outer, model_seed, "FINAL_QDIFFCL")
                if key_b not in by_key:
                    raise RuntimeError(f"missing matched FINAL_QDIFFCL cell: {key_b}")
                for key in (key_a, key_b):
                    if key not in cache:
                        cache[key] = group_confusions(by_key[key], classes)
                groups_a, cm_a = cache[key_a]
                groups_b, cm_b = cache[key_b]
                if groups_a != groups_b:
                    raise RuntimeError(f"paired groups differ: {dataset}/{outer}/{model_seed}/{method}")
                rng = np.random.default_rng(seed + method_index * 100000 + outer + model_seed)
                counts = rng.multinomial(len(groups_a), np.full(len(groups_a), 1 / len(groups_a)), size=repeats)
                draws_by_cell.append(
                    macro_f1_from_confusions(np.einsum("rg,gij->rij", counts, cm_a))
                    - macro_f1_from_confusions(np.einsum("rg,gij->rij", counts, cm_b))
                )
                observed.append(
                    float(by_key[key_a]["metrics"]["macro_f1"])
                    - float(by_key[key_b]["metrics"]["macro_f1"])
                )
            draws = np.mean(np.stack(draws_by_cell), axis=0)
            low, high = np.quantile(draws, [0.025, 0.975])
            output.append({
                "dataset": dataset,
                "method": method,
                "track": by_key[keys[0]]["track"],
                "reference": "FINAL_QDIFFCL",
                "metric": "macro_f1",
                "paired_cells": 15,
                "effect": float(np.mean(observed)),
                "ci_low": float(low),
                "ci_high": float(high),
                "positive_count": int(np.sum(np.asarray(observed) > 0)),
                "nonworse_count": int(np.sum(np.asarray(observed) >= 0)),
                "worst_delta": float(np.min(observed)),
                "bootstrap_unit": "WELL" if dataset == "3W" else "Run",
                "bootstrap_repeats": repeats,
            })
    return output


def overall(summary: list[dict[str, Any]], dataset: str, method: str) -> dict[str, Any]:
    return next(
        row for row in summary
        if row["dataset"] == dataset and row["method"] == method and row["level"] == "overall"
    )


def write_report(
    config: dict[str, Any],
    summary: list[dict[str, Any]],
    bootstrap: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    methods = [*config["active_methods"], *config["statistics"]["reference_methods"]]
    lines = [
        "# Recent time-series baselines: unified five-seed extension",
        "",
        f"Status: `{COMPLETE}`",
        "",
        f"Evidence class: `{EVIDENCE_CLASS}`. This is post-hoc seed completion, not preregistered Paper-final evidence.",
        "",
        f"H1 archive commit: `{config['h1_archive_commit']}`. Protocol-lock commit: `{manifest['protocol_lock_commit']}`. Protocol hash: `{manifest['protocol_hash']}`.",
        "",
        "All four H1 baselines were extended independently of their three-seed result direction. The 72 H1 cells were reused and exactly 48 missing-seed cells were added; no completed H1 cell was retrained.",
        "",
        "## Comparison boundary",
        "",
        "- Track A: AutoTCL and SoftCLT are shared-backbone/mechanism adaptations, not official reproductions.",
        "- Track B: TF-C and TS2Vec are method-native representation comparisons and do not identify augmentation-only causality.",
        "- Frozen Paper-final methods are reused without retraining.",
        "",
        "## Split-first five-seed results",
        "",
        "Each split mean first averages five matched model seeds. Mean ± sample SD is then computed across the three frozen outer splits.",
        "",
        "| Dataset | Track | Method | Macro-F1 | AUPRC | FAR | Early recall | Delay | Worst cell (outer/seed) |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for dataset in ("3W", "TEP"):
        for method in methods:
            row = overall(summary, dataset, method)
            show = lambda name: "N/A" if row.get(name) is None else f"{row[name]:.4f}"
            lines.append(
                f"| {dataset} | {row['track']} | {method} | {show('macro_f1_mean')} ± {show('macro_f1_std')} | "
                f"{show('auprc_mean')} | {show('far_mean')} | {show('early_recall_mean')} | "
                f"{show('detection_delay_mean')} | {show('worst_cell_macro_f1')} "
                f"({row['worst_outer_seed']}/{row['worst_model_seed']}) |"
            )
    lines += [
        "",
        "## Paired group-aware bootstrap versus FINAL_QDIFFCL",
        "",
        "Positive Δ means the row method has a higher paired Macro-F1 point estimate. Resampling uses WELL on 3W and Run on TEP; windows are never independent bootstrap units.",
        "",
        "| Dataset | Track | Method | Δ Macro-F1 | 95% CI | Above cells | Non-worse cells | Worst delta |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in bootstrap:
        lines.append(
            f"| {row['dataset']} | {row['track']} | {row['method']} | {row['effect']:+.4f} | "
            f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | {row['positive_count']}/15 | "
            f"{row['nonworse_count']}/15 | {row['worst_delta']:+.4f} |"
        )
    lines += ["", "## Explicit above/below accounting", ""]
    for dataset in ("3W", "TEP"):
        rows = [row for row in bootstrap if row["dataset"] == dataset]
        above = [row["method"] for row in rows if row["effect"] > 0]
        below = [row["method"] for row in rows if row["effect"] < 0]
        lines.append(
            f"- {dataset}: above Q-DiffCL by paired point estimate: {', '.join(above) or 'none'}; "
            f"below: {', '.join(below) or 'none'}. Use the 95% CI, not sign alone, for uncertainty claims."
        )
    lines += [
        "",
        "## Coverage and evidence limits",
        "",
        "- Coverage is two datasets × three grouped outer splits × five matched model seeds.",
        "- The extension does not modify Q-DiffCL, select new baselines, search seeds, or tune on outer-test results.",
        "- Track B remains representation-level context; the extension does not close low-data, third-dataset, or missingness-robustness gaps.",
        "",
    ]
    report_text = "\n".join(lines)
    corrupted = chr(0xFFFD)
    report_text = report_text.replace(f"Mean {corrupted} sample", "Mean ± sample")
    report_text = report_text.replace(f"Positive {corrupted}", "Positive Δ")
    report_text = report_text.replace(f"| {corrupted} Macro-F1", "| Δ Macro-F1")
    report_text = report_text.replace(
        f"two datasets {corrupted} three grouped outer splits {corrupted} five matched model seeds",
        "two datasets × three grouped outer splits × five matched model seeds",
    )
    report_text = report_text.replace(f" {corrupted} ", " ± ")
    Path(config["output"]["report"]).write_text(report_text, encoding="utf-8")


def update_h1_report(config: dict[str, Any]) -> None:
    path = Path("docs/posthoc_recent_baselines.md")
    marker = "## Five-seed extension"
    text = path.read_text(encoding="utf-8")
    section = (
        "\n\n## Five-seed extension\n\n"
        "The four active recent baselines were subsequently completed to the frozen Paper-final five-seed sets under "
        "`POSTHOC_BASELINE_5SEED_EXTENSION`. See `posthoc_recent_baselines_5seed.md`; the original H1 three-seed "
        "results and evidence classification remain unchanged.\n"
    )
    if marker not in text:
        path.write_text(text.rstrip() + section, encoding="utf-8")


def finalize(config: dict[str, Any]) -> dict[str, Any]:
    validate_static(config)
    manifest = read_json(config["output"]["manifest"])
    h1_records = load_h1_records(config)
    extension_records = load_extension_records(config)
    references = load_paper_final_references(config)
    recent = [*h1_records, *extension_records]
    recent_keys = {(row["dataset"], int(row["outer_seed"]), int(row["model_seed"]), row["method"]) for row in recent}
    if len(recent) != 120 or len(recent_keys) != 120:
        raise RuntimeError("recent-baseline five-seed matrix is not 120 unique cells")
    combined = [*recent, *references]
    rows = combined_raw_rows(h1_records, extension_records, references)
    if len(rows) != 360:
        raise RuntimeError(f"combined five-seed raw matrix is not 360 rows: {len(rows)}")
    summary = add_worst_cell_identity(split_first_summary(combined), combined)
    comparison_methods = [
        *config["active_methods"],
        *[method for method in config["statistics"]["reference_methods"] if method != "FINAL_QDIFFCL"],
    ]
    bootstrap = group_bootstrap(
        combined,
        int(config["statistics"]["bootstrap_repeats"]),
        int(config["statistics"]["bootstrap_seed"]),
        comparison_methods,
    )
    write_csv(config["output"]["raw_csv"], rows)
    write_csv(config["output"]["summary_csv"], summary)
    write_csv(config["output"]["bootstrap_csv"], bootstrap)
    write_report(config, summary, bootstrap, manifest)
    update_h1_report(config)
    manifest["status"] = COMPLETE
    manifest["completed_cells"] = 48
    manifest["total_recent_baseline_5seed_cells"] = 120
    manifest["paper_final_5seed_reuse_cells"] = len(references)
    manifest["combined_raw_rows"] = len(rows)
    manifest["summary_rows"] = len(summary)
    manifest["bootstrap_rows"] = len(bootstrap)
    manifest["summary_artifacts"] = {
        key: {"path": config["output"][key], "sha256": sha256_file(config["output"][key])}
        for key in ("raw_csv", "summary_csv", "bootstrap_csv", "report")
    }
    Path(config["output"]["manifest"]).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "status": COMPLETE,
        "h1_reuse_rows": len(h1_records),
        "extension_rows": len(extension_records),
        "paper_final_reuse_rows": len(references),
        "raw_rows": len(rows),
        "summary_rows": len(summary),
        "bootstrap_rows": len(bootstrap),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/posthoc_baseline_5seed_extension.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    print(json.dumps(finalize(config), ensure_ascii=False))


if __name__ == "__main__":
    main()
