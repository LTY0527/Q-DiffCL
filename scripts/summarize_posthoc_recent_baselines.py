from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.metrics import confusion_matrix


POSTHOC_EVIDENCE = "POSTHOC_BASELINE_EVIDENCE"
REFERENCE_METHODS = ("FINAL_QDIFFCL", "NO_AUG", "UNIFORM_DIFFUSION", "FRERA")
METRICS = ("macro_f1", "auprc", "far", "fault_recall", "early_recall", "detection_delay")


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float(value: Any) -> float | None:
    if value in (None, "", "None", "nan", "NaN"):
        return None
    return float(value)


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with destination.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader(); writer.writerows(rows)


def load_posthoc_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    root = Path(config["output"]["root"]) / "benchmark"
    records = [_read_json(path) for path in sorted(root.glob("*/outer_*/seed_*/*/result.json"))]
    expected = {(dataset, int(outer), int(seed), method)
                for dataset in ("3W", "TEP")
                for outer in config["benchmark"]["outer_splits"][dataset]
                for seed in config["benchmark"]["model_seeds"][dataset]
                for method in config["active_methods"]}
    observed = {(row["dataset"], int(row["outer_seed"]), int(row["model_seed"]), row["method"])
                for row in records}
    if observed != expected or len(records) != len(expected):
        raise RuntimeError(f"post-hoc matrix incomplete or duplicated: {len(records)}/{len(expected)}")
    for row in records:
        if row.get("evidence_class") != POSTHOC_EVIDENCE or not row.get("outer_test_evaluated_once"):
            raise RuntimeError(f"invalid post-hoc evidence boundary: {row.get('run_id')}")
        if _sha256(row["prediction_path"]) != row["prediction_sha256"]:
            raise RuntimeError(f"prediction hash mismatch: {row['run_id']}")
        if _sha256(row["checkpoint_path"]) != row["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {row['run_id']}")
    return records


def load_matched_paper_final(config: dict[str, Any]) -> list[dict[str, Any]]:
    seeds = {dataset: set(map(int, config["benchmark"]["model_seeds"][dataset])) for dataset in ("3W", "TEP")}
    outers = {dataset: set(map(int, config["benchmark"]["outer_splits"][dataset])) for dataset in ("3W", "TEP")}
    source = Path("analysis/results/paper_final_outer_raw.csv")
    rows: list[dict[str, Any]] = []
    with source.open(encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            dataset = raw["dataset"]; method = raw["method"]
            if method not in REFERENCE_METHODS or int(raw["model_seed"]) not in seeds[dataset] or int(raw["outer_seed"]) not in outers[dataset]:
                continue
            metrics = {metric: _float(raw.get(metric)) for metric in METRICS}
            row = {"run_id": raw["run_id"], "dataset": dataset, "outer_seed": int(raw["outer_seed"]),
                   "model_seed": int(raw["model_seed"]), "method": method,
                   "track": "TRACK_A_FROZEN_PAPER_FINAL_REFERENCE", "metrics": metrics,
                   "prediction_path": raw["prediction_path"], "prediction_sha256": raw["prediction_sha256"],
                   "checkpoint_sha256": raw["checkpoint_sha256"], "evidence_source": "FROZEN_PAPER_FINAL_REUSE"}
            if _sha256(row["prediction_path"]) != row["prediction_sha256"]:
                raise RuntimeError(f"Paper-final prediction hash mismatch: {row['run_id']}")
            rows.append(row)
    expected = 2 * 3 * 3 * len(REFERENCE_METHODS)
    keys = {(row["dataset"], row["outer_seed"], row["model_seed"], row["method"]) for row in rows}
    if len(rows) != expected or len(keys) != expected:
        raise RuntimeError(f"matched Paper-final reuse is not {expected} unique rows: {len(rows)}")
    return rows


def raw_rows(posthoc: list[dict[str, Any]], references: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in [*posthoc, *references]:
        metrics = row["metrics"]
        output.append({"run_id": row["run_id"], "dataset": row["dataset"], "outer_seed": row["outer_seed"],
                       "model_seed": row["model_seed"], "method": row["method"], "track": row["track"],
                       "evidence_source": row.get("evidence_source", "NEW_POSTHOC_TRAINING"),
                       **{metric: metrics.get(metric) for metric in METRICS},
                       "prediction_path": row["prediction_path"], "prediction_sha256": row["prediction_sha256"],
                       "checkpoint_sha256": row["checkpoint_sha256"]})
    return sorted(output, key=lambda row: (row["dataset"], row["method"], row["outer_seed"], row["model_seed"]))


def split_first_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for dataset, method in sorted({(row["dataset"], row["method"]) for row in records}):
        selected = [row for row in records if row["dataset"] == dataset and row["method"] == method]
        split_rows = []
        for outer in sorted({int(row["outer_seed"]) for row in selected}):
            cells = [row for row in selected if int(row["outer_seed"]) == outer]
            current = {"dataset": dataset, "method": method, "track": cells[0]["track"], "level": "split",
                       "outer_seed": outer, "cells": len(cells)}
            for metric in METRICS:
                values = [float(row["metrics"][metric]) for row in cells if row["metrics"].get(metric) is not None]
                current[f"{metric}_mean"] = float(np.mean(values)) if values else None
                current[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else (0.0 if values else None)
            split_rows.append(current); output.append(current)
        aggregate = {"dataset": dataset, "method": method, "track": selected[0]["track"], "level": "overall",
                     "outer_seed": "ALL", "cells": len(selected)}
        for metric in METRICS:
            values = [row[f"{metric}_mean"] for row in split_rows if row[f"{metric}_mean"] is not None]
            aggregate[f"{metric}_mean"] = float(np.mean(values)) if values else None
            aggregate[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else (0.0 if values else None)
        aggregate["worst_cell_macro_f1"] = float(min(row["metrics"]["macro_f1"] for row in selected))
        output.append(aggregate)
    return output


def _macro_f1_from_confusions(confusions: np.ndarray) -> np.ndarray:
    values = np.asarray(confusions, dtype=np.float64)
    true_positive = np.diagonal(values, axis1=-2, axis2=-1)
    predicted = values.sum(axis=-2); actual = values.sum(axis=-1); denominator = predicted + actual
    f1 = np.divide(2 * true_positive, denominator, out=np.zeros_like(true_positive), where=denominator > 0)
    return f1.mean(axis=-1)


def _group_confusions(record: dict[str, Any], classes: int) -> tuple[list[str], np.ndarray]:
    with np.load(record["prediction_path"], allow_pickle=False) as archive:
        groups = archive["group_id"].astype(str); labels = archive["label"].astype(int); prediction = archive["prediction"].astype(int)
    unique = sorted(set(groups)); matrices = [confusion_matrix(labels[groups == group], prediction[groups == group],
                                                               labels=np.arange(classes)) for group in unique]
    return unique, np.stack(matrices)


def group_bootstrap(records: list[dict[str, Any]], repeats: int, seed: int,
                    comparison_methods: list[str]) -> list[dict[str, Any]]:
    by_key = {(row["dataset"], int(row["outer_seed"]), int(row["model_seed"]), row["method"]): row for row in records}
    cache: dict[tuple[str, int, int, str], tuple[list[str], np.ndarray]] = {}
    output = []
    for dataset in ("3W", "TEP"):
        classes = 4 if dataset == "3W" else 2
        for method_index, method in enumerate(comparison_methods):
            draws_by_cell = []; observed = []
            keys = sorted(key for key in by_key if key[0] == dataset and key[3] == method)
            if len(keys) != 9:
                raise RuntimeError(f"paired matrix for {dataset}/{method} has {len(keys)}/9 cells")
            for _, outer, model_seed, _ in keys:
                key_a = (dataset, outer, model_seed, method); key_b = (dataset, outer, model_seed, "FINAL_QDIFFCL")
                if key_b not in by_key:
                    raise RuntimeError(f"missing matched FINAL_QDIFFCL cell: {key_b}")
                for key in (key_a, key_b):
                    if key not in cache: cache[key] = _group_confusions(by_key[key], classes)
                groups_a, cm_a = cache[key_a]; groups_b, cm_b = cache[key_b]
                if groups_a != groups_b:
                    raise RuntimeError(f"paired groups differ: {dataset}/{outer}/{model_seed}/{method}")
                rng = np.random.default_rng(seed + method_index * 100000 + outer + model_seed)
                counts = rng.multinomial(len(groups_a), np.full(len(groups_a), 1 / len(groups_a)), size=repeats)
                draws_by_cell.append(_macro_f1_from_confusions(np.einsum("rg,gij->rij", counts, cm_a)) -
                                     _macro_f1_from_confusions(np.einsum("rg,gij->rij", counts, cm_b)))
                observed.append(float(by_key[key_a]["metrics"]["macro_f1"]) -
                                float(by_key[key_b]["metrics"]["macro_f1"]))
            draws = np.mean(np.stack(draws_by_cell), axis=0); low, high = np.quantile(draws, [.025, .975])
            track = by_key[keys[0]]["track"]
            output.append({"dataset": dataset, "method": method, "track": track, "reference": "FINAL_QDIFFCL",
                           "metric": "macro_f1", "paired_cells": len(keys), "effect": float(np.mean(observed)),
                           "ci_low": float(low), "ci_high": float(high),
                           "positive_count": int(np.sum(np.asarray(observed) > 0)),
                           "nonworse_count": int(np.sum(np.asarray(observed) >= 0)),
                           "worst_delta": float(np.min(observed)), "bootstrap_unit": "WELL" if dataset == "3W" else "Run",
                           "bootstrap_repeats": repeats})
    return output


def _overall(summary: list[dict[str, Any]], dataset: str, method: str) -> dict[str, Any]:
    return next(row for row in summary if row["dataset"] == dataset and row["method"] == method and row["level"] == "overall")


def write_report(config: dict[str, Any], summary: list[dict[str, Any]], bootstrap: list[dict[str, Any]]) -> None:
    matrix = list(csv.DictReader(Path("analysis/results/posthoc_baseline_candidate_matrix.csv").open(encoding="utf-8-sig")))
    lock = _read_json(config["selection_lock"]); active = set(config["active_methods"])
    replacement = {"TimesURL": "TF-C", "MF-CLR": "SoftCLT", "REBAR": "TS2Vec", "AutoTCL": "retained"}
    lines = ["# Recent Time-series Baselines: Post-hoc Fair Benchmark", "",
             "Status: `POSTHOC_BASELINE_BENCHMARK_COMPLETE`", "",
             f"Evidence class: `{POSTHOC_EVIDENCE}`. These results were produced after the frozen Paper-final evaluation and are not preregistered Paper-final evidence.", "",
             "## Candidate audit and selection lock", "",
             "The selection was performance-blind and completed before candidate outer metrics. The original selection hash remains " + f"`{config['selection_hash']}`.", "",
             "| Rank | Method | Score | Locked | Final disposition |", "|---:|---|---:|---|---|"]
    ranks = {row["method"]: row for row in lock["candidate_ranking"]}
    fallback = {row["method"]: row["rank"] for row in lock["fallback_ranking"]}
    for row in sorted(matrix, key=lambda item: int(ranks[item["method"]]["rank"])):
        method = row["method"]; rank = ranks[method]
        if method in replacement: disposition = replacement[method]
        elif method in active: disposition = "active fallback"
        elif method in fallback: disposition = f"unconsumed fallback rank {fallback[method]}"
        else: disposition = "not selected by frozen ranking/constraints"
        lines.append(f"| {rank['rank']} | {method} | {rank['score']} | {'yes' if rank['selected'] else 'no'} | {disposition} |")
    lines += ["", "The append-only cost amendment replaced TimesURL, MF-CLR, and REBAR by TF-C, SoftCLT, and TS2Vec respectively. Conservative estimates were 63.16 h, 104.23 h, and 74.78 h; no validation or outer score was used. AutoTCL was retained. See `posthoc_baseline_selection_amendment.md` and `posthoc_baseline_failure_log.md`.", "",
              "## Comparison boundary", "", "| Track | Methods | Interpretation |", "|---|---|---|",
              "| Track A | AutoTCL, SoftCLT; frozen Paper-final references | shared-backbone/mechanism comparison; direct paired comparison is permitted |",
              "| Track B | TF-C, TS2Vec | method-native representation context; do not claim augmentation-only causality |", "",
              "## Split-first results", "", "Each outer-split value first averages its three matched model seeds; the table reports mean ± sample SD across the three frozen outer splits.", "",
              "| Dataset | Track | Method | Macro-F1 | AUPRC | FAR | Early recall | Delay | Worst cell |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    methods = [*config["active_methods"], *REFERENCE_METHODS]
    for dataset in ("3W", "TEP"):
        for method in methods:
            row = _overall(summary, dataset, method)
            def show(name: str) -> str:
                value = row.get(name); return "N/A" if value is None else f"{value:.4f}"
            lines.append(f"| {dataset} | {row['track']} | {method} | {show('macro_f1_mean')} ± {show('macro_f1_std')} | {show('auprc_mean')} | {show('far_mean')} | {show('early_recall_mean')} | {show('detection_delay_mean')} | {show('worst_cell_macro_f1')} |")
    lines += ["", "## Paired group-aware bootstrap versus FINAL_QDIFFCL", "",
              "Positive Δ means the row method is above FINAL_QDIFFCL. Resampling uses WELL on 3W and Run on TEP; windows are never treated as independent.", "",
              "| Dataset | Track | Method | Δ Macro-F1 | 95% CI | Above cells | Non-worse cells |", "|---|---|---|---:|---:|---:|---:|"]
    for row in bootstrap:
        lines.append(f"| {row['dataset']} | {row['track']} | {row['method']} | {row['effect']:+.4f} | [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | {row['positive_count']}/{row['paired_cells']} | {row['nonworse_count']}/{row['paired_cells']} |")
    lines += ["", "## Explicit above/below accounting", ""]
    for dataset in ("3W", "TEP"):
        rows = [row for row in bootstrap if row["dataset"] == dataset]
        above = [row["method"] for row in rows if row["effect"] > 0]; below = [row["method"] for row in rows if row["effect"] < 0]
        lines.append(f"- {dataset}: above Q-DiffCL by point estimate: {', '.join(above) or 'none'}; below: {', '.join(below) or 'none'}. Statistical uncertainty is governed by the CI table, not the sign alone.")
    lines += ["", "## Remaining coverage gaps", "",
              "- The active benchmark covers two datasets, three grouped outer splits, and three matched seeds; it is post-hoc evidence, not a new preregistration.",
              "- Track B provides representation-level context and cannot isolate augmentation causality.",
              "- TimesURL, MF-CLR, and REBAR retain successful sanity evidence but lack formal outer results due to the predeclared non-performance cost rule.",
              "- AutoDA-Timeseries remains method-native supplementary only; InfoTS remains audit/fallback coverage.",
              "- No low-data study, third dataset, or broader missingness robustness evaluation was added in H1.", ""]
    Path(config["output"]["report"]).write_text("\n".join(lines), encoding="utf-8")


def finalize(config: dict[str, Any]) -> dict[str, Any]:
    posthoc = load_posthoc_records(config); references = load_matched_paper_final(config); combined = [*posthoc, *references]
    rows = raw_rows(posthoc, references); summary = split_first_summary(combined)
    paper = yaml.safe_load(Path(config["paper_final_config"]).read_text(encoding="utf-8"))
    statistics = paper["statistics"]
    comparisons = [*config["active_methods"], "NO_AUG", "UNIFORM_DIFFUSION", "FRERA"]
    bootstrap = group_bootstrap(combined, int(statistics["bootstrap_repeats"]), int(statistics["bootstrap_seed"]), comparisons)
    _write_csv(config["output"]["raw_csv"], rows); _write_csv(config["output"]["summary_csv"], summary)
    _write_csv(config["output"]["bootstrap_csv"], bootstrap); write_report(config, summary, bootstrap)
    manifest_path = Path(config["output"]["manifest"]); manifest = _read_json(manifest_path)
    if len(manifest["cells"]) != 72 or any(row["status"] != "complete" for row in manifest["cells"]):
        raise RuntimeError("manifest is not 72/72 complete after result audit")
    manifest["status"] = "POSTHOC_BASELINE_BENCHMARK_COMPLETE"; manifest["completed_cells"] = 72
    manifest["summary_artifacts"] = {key: {"path": config["output"][key], "sha256": _sha256(config["output"][key])}
                                     for key in ("raw_csv", "summary_csv", "bootstrap_csv", "report")}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"status": manifest["status"], "posthoc_rows": len(posthoc), "paper_final_reuse_rows": len(references),
            "raw_rows": len(rows), "summary_rows": len(summary), "bootstrap_rows": len(bootstrap)}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/posthoc_recent_baselines.yaml")
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    print(json.dumps(finalize(config), ensure_ascii=False))


if __name__ == "__main__":
    main()
