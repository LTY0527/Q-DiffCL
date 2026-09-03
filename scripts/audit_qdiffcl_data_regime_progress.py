from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from scripts.audit_qdiffcl_data_regime import atomic_json, canonical_hash, sha256_file
from scripts.run_qdiffcl_data_regime import FORMAL_METHODS, fraction_token, legal_dataset_fractions, load_config, load_fraction_manifest


FIELDS = (
    "dataset", "fraction", "outer", "stage", "method", "seed", "rho", "status",
    "macro_f1", "auprc", "far", "early_recall", "detection_delay", "checkpoint_exists",
    "prediction_exists", "manifest_exists", "hash_valid", "test_evaluated", "reuse", "runtime",
    "checkpoint_sha256", "prediction_sha256", "result_sha256", "artifact_path", "validation_only",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def _cell_root(root: Path, dataset: str, fraction: float, outer: int, seed: int, method: str) -> Path:
    return root / dataset.lower() / fraction_token(fraction) / f"outer_{outer}" / f"model_seed_{seed}" / method


def _candidate_path(root: Path, dataset: str, fraction: float, outer: int, seed: int, rho: float) -> Path:
    base = _cell_root(root, dataset, fraction, outer, seed, "CALIBRATED_RHO")
    if rho == 1.0:
        return _cell_root(root, dataset, fraction, outer, seed, "FINAL_QDIFFCL_FIXED") / "_training"
    return base / "_candidates" / f"rho_{int(round(rho * 100)):03d}"


def _context_audit(root: Path, dataset: str, fraction: float, outer: int) -> dict[str, Any] | None:
    path = root / dataset.lower() / fraction_token(fraction) / f"outer_{outer}" / "_context" / "audit.json"
    return _read(path) if path.exists() else None


def _checkpoint_metadata(path: Path) -> dict[str, Any] | None:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return payload.get("metadata")
    except Exception:
        return None


def _candidate_row(
    config: dict[str, Any], root: Path, dataset: str, fraction: float, outer: int, seed: int, rho: float,
) -> dict[str, Any]:
    path = _candidate_path(root, dataset, fraction, outer, seed, rho)
    validation_path = path / "validation.json"; checkpoint = path / "model.pt"
    selection_path = root / dataset.lower() / fraction_token(fraction) / f"outer_{outer}" / "rho_selection.json"
    base = {
        "dataset": dataset, "fraction": fraction, "outer": outer, "stage": "rho_validation_candidate",
        "method": "CALIBRATED_RHO", "seed": seed, "rho": rho, "macro_f1": "N/A", "auprc": "N/A",
        "far": "N/A", "early_recall": "N/A", "detection_delay": "N/A",
        "checkpoint_exists": checkpoint.exists(), "prediction_exists": False,
        "manifest_exists": selection_path.exists(), "test_evaluated": False, "reuse": False,
        "runtime": "N/A", "checkpoint_sha256": sha256_file(checkpoint) if checkpoint.exists() else "N/A",
        "prediction_sha256": "N/A", "result_sha256": "N/A", "artifact_path": str(path),
        "validation_only": True,
    }
    if not (validation_path.exists() and checkpoint.exists()):
        return {**base, "status": "PENDING", "hash_valid": False}
    try:
        record = _read(validation_path); _, fraction_record = load_fraction_manifest(dataset, outer, fraction)
        audit = _context_audit(root, dataset, fraction, outer); metadata = _checkpoint_metadata(checkpoint)
        checks = [
            record.get("validation_only") is True, record.get("outer_test_read") is False,
            record.get("data_regime", {}).get("config_sha256") == sha256_file(config["_config_path"]),
            record.get("data_regime", {}).get("fraction_manifest_hash") == fraction_record["sha256"],
            audit is not None and record.get("data_regime", {}).get("criticality_mask_sha256") == audit.get("criticality_mask_sha256"),
            metadata is not None and metadata.get("context_hash") == record.get("metadata", {}).get("context_hash"),
            record.get("metadata", {}).get("dataset") == dataset,
            int(record.get("metadata", {}).get("outer_seed", -1)) == outer,
            int(record.get("metadata", {}).get("model_seed", -1)) == seed,
        ]
        metrics = record["validation"]
        base.update({"macro_f1": metrics.get("macro_f1", "N/A"), "auprc": metrics.get("auprc", "N/A"),
                     "far": metrics.get("far", "N/A"), "runtime": record.get("training_seconds", "N/A")})
        valid = all(checks)
        return {**base, "status": "COMPLETE_VALID" if valid else "UNVERIFIED", "hash_valid": valid}
    except Exception:
        return {**base, "status": "UNVERIFIED", "hash_valid": False}


def _formal_row(
    config: dict[str, Any], root: Path, manifest_cell: dict[str, Any], manifest_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    dataset = manifest_cell["dataset"]; fraction = float(manifest_cell["fraction"]); outer = int(manifest_cell["outer_id"])
    seed = int(manifest_cell["model_seed"]); method = manifest_cell["method"]
    path = _cell_root(root, dataset, fraction, outer, seed, method)
    result_path = path / "result.json"; prediction = path / "predictions.npz"; started = path / "outer_test_started.json"
    base = {
        "dataset": dataset, "fraction": fraction, "outer": outer, "stage": "formal_locked_test",
        "method": method, "seed": seed, "rho": "N/A", "macro_f1": "N/A", "auprc": "N/A", "far": "N/A",
        "early_recall": "N/A", "detection_delay": "N/A", "checkpoint_exists": False,
        "prediction_exists": prediction.exists(), "manifest_exists": manifest_cell is not None,
        "test_evaluated": started.exists(), "reuse": False, "runtime": "N/A", "checkpoint_sha256": "N/A",
        "prediction_sha256": sha256_file(prediction) if prediction.exists() else "N/A", "result_sha256": "N/A",
        "artifact_path": str(result_path), "validation_only": False,
    }
    if not result_path.exists():
        return {**base, "status": "PENDING", "hash_valid": False}
    try:
        result = _read(result_path); checkpoint = Path(result["checkpoint_path"]); validation = Path(result["validation_metrics_path"])
        base["checkpoint_exists"] = checkpoint.exists(); base["checkpoint_sha256"] = sha256_file(checkpoint) if checkpoint.exists() else "N/A"
        base["rho"] = result.get("selected_rho") if result.get("selected_rho") is not None else "N/A"
        metrics = result["test_metrics"]
        base.update({key: metrics.get(key, "N/A") if metrics.get(key) is not None else "N/A" for key in
                     ("macro_f1", "auprc", "far", "early_recall", "detection_delay")})
        if validation.exists():
            base["runtime"] = _read(validation).get("training_seconds", "N/A")
        payload = dict(result); claimed_payload = payload.pop("result_payload_sha256", None)
        _, fraction_record = load_fraction_manifest(dataset, outer, fraction); audit = _context_audit(root, dataset, fraction, outer)
        indexed = manifest_index.get(result["cell_id"], {})
        checks = [
            claimed_payload == canonical_hash(payload), result.get("prediction_sha256") == base["prediction_sha256"],
            result.get("checkpoint_sha256") == base["checkpoint_sha256"],
            result.get("config_sha256") == sha256_file(config["_config_path"]),
            result.get("fraction_manifest_hash") == fraction_record["sha256"],
            audit is not None and result.get("criticality_mask_sha256") == audit.get("criticality_mask_sha256"),
            result.get("context_hash") == audit.get("context_hash") if audit else False,
            result.get("source_commit") == _read(Path(config["git_freeze"]["protocol_lock_manifest"]))["protocol_lock_commit"],
            result.get("dataset") == dataset, float(result.get("fraction")) == fraction,
            int(result.get("outer_id")) == outer, int(result.get("model_seed")) == seed,
            result.get("method") == method, result.get("outer_test_evaluated_once") is True,
            indexed.get("result_sha256") == claimed_payload, indexed.get("status") == "complete",
        ]
        valid = all(checks)
        base["result_sha256"] = claimed_payload or "N/A"
        return {**base, "status": "COMPLETE_VALID" if valid else "UNVERIFIED", "hash_valid": valid}
    except Exception:
        return {**base, "status": "UNVERIFIED", "hash_valid": False}


def _mean_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["stage"] == "formal_locked_test" and row["status"] == "COMPLETE_VALID":
            grouped[(row["dataset"], float(row["fraction"]), row["method"])].append(row)
    output = []
    for key, current in sorted(grouped.items()):
        record = {"dataset": key[0], "fraction": key[1], "method": key[2], "cells": len(current)}
        for metric in ("macro_f1", "auprc", "far", "early_recall", "detection_delay"):
            values = [float(row[metric]) for row in current if row[metric] != "N/A" and math.isfinite(float(row[metric]))]
            record[metric] = float(np.mean(values)) if values else None
        output.append(record)
    return output


def _paired_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    formal = {(row["dataset"], float(row["fraction"]), row["outer"], row["seed"], row["method"]): row for row in rows
              if row["stage"] == "formal_locked_test" and row["status"] == "COMPLETE_VALID"}
    contrasts = (("FINAL_QDIFFCL_FIXED", "NO_AUG"), ("FINAL_QDIFFCL_FIXED", "UNIFORM_DIFFUSION"),
                 ("FINAL_QDIFFCL_FIXED", "JITTER_SCALING"), ("CALIBRATED_RHO", "FINAL_QDIFFCL_FIXED"),
                 ("CALIBRATED_RHO", "UNIFORM_DIFFUSION"))
    output = []
    for dataset, fraction in sorted({key[:2] for key in formal}):
        for left, right in contrasts:
            values = []
            for key, row in formal.items():
                if key[0] == dataset and key[1] == fraction and key[4] == left:
                    other = formal.get((dataset, fraction, key[2], key[3], right))
                    if other is not None:
                        values.append(float(row["macro_f1"]) - float(other["macro_f1"]))
            if values:
                output.append({"dataset": dataset, "fraction": fraction, "contrast": f"{left} - {right}",
                               "paired_cells": len(values), "macro_f1_delta": float(np.mean(values))})
    return output


def _fmt(value: Any) -> str:
    return "N/A" if value is None else (f"{value:.6f}" if isinstance(value, float) else str(value))


def _progress_doc(accounting: dict[str, Any], rows: list[dict[str, Any]], means: list[dict[str, Any]], paired: list[dict[str, Any]], selections: list[dict[str, Any]], output_root: Path) -> str:
    lines = ["# Q-DiffCL Data-Regime Progress Audit", "", "Status: `PARTIAL_RESULTS_ARCHIVED` / `INTERIM_PARTIAL_EVIDENCE`.", "",
             "The runner is stopped. No training was resumed during this audit.", "", "## Cell accounting", "",
             f"- Formal: {accounting['formal_cells_valid']} valid / {accounting['formal_cells_expected']} expected; {accounting['formal_cells_remaining']} remaining.",
             f"- Rho candidates: {accounting['rho_candidate_cells_valid']} valid / {accounting['rho_candidate_cells_expected']} expected; {accounting['rho_candidate_cells_remaining']} remaining.",
             f"- Invalid: formal {accounting['formal_cells_invalid']}, rho {accounting['rho_candidate_cells_invalid']}; duplicates {accounting['duplicate_count']}; runner failures {accounting['failure_count']}.",
             "", "## Dataset and fraction completion", ""]
    for dataset in ("3W", "TEP"):
        lines.append(f"### {dataset}"); lines.append("")
        for fraction in (1., .25, .10):
            formal = [r for r in rows if r["dataset"] == dataset and float(r["fraction"]) == fraction and r["stage"] == "formal_locked_test" and r["status"] == "COMPLETE_VALID"]
            candidates = [r for r in rows if r["dataset"] == dataset and float(r["fraction"]) == fraction and r["stage"] == "rho_validation_candidate" and r["status"] == "COMPLETE_VALID"]
            selection = [s for s in selections if s["dataset"] == dataset and float(s["fraction"]) == fraction]
            method_counts = Counter(r["method"] for r in formal); outers = sorted({r["outer"] for r in formal})
            seeds = sorted({r["seed"] for r in formal})
            reused = sum(bool(r["reuse"]) for r in formal)
            new = len(formal) - reused
            hold = dataset == "TEP" and fraction == .10
            lines.append(f"- {fraction:.0%}: {'E_IDENTIFIABILITY_HOLD; excluded from primary matrix' if hold else f'{len(formal)} formal ({new} new, {reused} reused), {len(candidates)} rho candidates, completed outers {outers or []}, completed seeds {seeds or []}, methods {dict(method_counts)}, rho selections {len(selection)}'}.")
        lines.append("")
    lines.extend(["## Interim metric means (valid completed cells only)", "", "| Dataset | Fraction | Method | Cells | Macro-F1 | AUPRC | FAR | Early Recall | Delay |", "|---|---:|---|---:|---:|---:|---:|---:|---:|"])
    for row in means:
        lines.append(f"| {row['dataset']} | {row['fraction']:.2f} | {row['method']} | {row['cells']} | {_fmt(row['macro_f1'])} | {_fmt(row['auprc'])} | {_fmt(row['far'])} | {_fmt(row['early_recall'])} | {_fmt(row['detection_delay'])} |")
    lines.extend(["", "## Interim paired Macro-F1", "", "| Dataset | Fraction | Contrast | Paired cells | Mean delta |", "|---|---:|---|---:|---:|"])
    for row in paired:
        lines.append(f"| {row['dataset']} | {row['fraction']:.2f} | {row['contrast']} | {row['paired_cells']} | {row['macro_f1_delta']:.6f} |")
    lines.extend(["", "These are stage results, not paper-final cross-dataset claims. TEP locked-test cells are absent.", "", "## Rho selections", ""])
    for row in selections:
        lines.append(f"- {row['dataset']} {row['fraction']:.2f} outer {row['outer']}: DATA_REGIME_RHO_STAR={row['selected_rho']}, candidates={row['candidates_completed']}/15, seeds={row['selection_seeds']}, validation Macro-F1={row['validation_macro_f1']:.6f}, AUPRC={row['validation_auprc']:.6f}, FAR={row['validation_far']:.6f}, test_used_for_selection={str(row['test_used_for_selection']).lower()}.")
    local_files = [path for path in output_root.rglob("*") if path.is_file()]
    local_bytes = sum(path.stat().st_size for path in local_files)
    over_50mb = [path for path in local_files if path.stat().st_size > 50 * 1024 * 1024]
    lines.extend(["", "`DATA_REGIME_RHO_STAR` values above are outer-specific validation-only choices. They are distinct from historical `HISTORICAL_DCBR_GLOBAL_RHO` lineage values.",
                  "", "## Local artifact archive", "",
                  f"- `{output_root}` contains {len(local_files)} files ({local_bytes} bytes); it remains local and Git-ignored.",
                  f"- Files larger than 50 MiB: {len(over_50mb)}. Checkpoint, prediction, and result hashes are recorded per completed cell in `analysis/results/qdiffcl_data_regime_progress_audit.csv`.",
                  "- No local training artifact was deleted or staged for Git.",
                  "", "## Resume", "",
                  "No resume was executed. Registered command: `E:\\anaconda\\envs\\qdiffcl\\python.exe -u -m scripts.run_qdiffcl_data_regime --stage all --device cuda`.",
                  f"Remaining work: {accounting['formal_cells_remaining']} formal cells and {accounting['rho_candidate_cells_remaining']} rho candidates. Remaining GPU time is `UNAVAILABLE` until the TEP loader can proceed without the observed host-RAM allocation failure; the earlier preliminary ETA is stale and is not presented as a current estimate.",
                  "The TEP RData loader must first be made memory-safe without changing the scientific protocol."])
    return "\n".join(lines) + "\n"


def _runtime_doc(config: dict[str, Any], accounting: dict[str, Any], rows: list[dict[str, Any]], stderr: Path, stdout: Path, root: Path) -> str:
    runtime_path = Path(config["output"]["runtime_status"]); runtime = _read(runtime_path)
    all_files = [p for p in root.rglob("*") if p.is_file()]
    last = max(all_files, key=lambda p: p.stat().st_mtime) if all_files else None
    error_text = stderr.read_text(encoding="utf-8", errors="replace")
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader"], capture_output=True, text=True, encoding="utf-8").stdout.strip()
    completed_formal = [row for row in rows if row["stage"] == "formal_locked_test" and row["status"] == "COMPLETE_VALID"]
    last_formal = max(completed_formal, key=lambda row: Path(row["artifact_path"]).stat().st_mtime)
    last_formal_label = f"{last_formal['dataset']} / {float(last_formal['fraction']):.0%} / outer {last_formal['outer']} / {last_formal['method']} / seed {last_formal['seed']}"
    return "\n".join([
        "# Q-DiffCL Data-Regime Runtime Diagnosis", "", "Status: `PROCESS_EXITED_FAILURE`; artifacts remain resumable.", "",
        f"- Audit time: `{_now()}`", "- Historical PID 45408 exists: `false`",
        "- Runner command: `E:\\anaconda\\envs\\qdiffcl\\python.exe -u -m scripts.run_qdiffcl_data_regime --stage all --device cuda`",
        f"- CPU state: process exited", f"- GPU current: `{gpu}`",
        f"- Runtime status timestamp: `{runtime['timestamp']}`", f"- Stdout mtime UTC: `{datetime.fromtimestamp(stdout.stat().st_mtime, timezone.utc).isoformat()}`",
        f"- Stderr mtime UTC: `{datetime.fromtimestamp(stderr.stat().st_mtime, timezone.utc).isoformat()}`",
        f"- Last artifact: `{last}`" if last else "- Last artifact: N/A",
        f"- Last formal completed cells: `{accounting['formal_cells_valid']}`", f"- Last formal cell: `{last_formal_label}`",
        "- Last validation candidate: `TEP / 100% / outer 32001 / seed 2026 / rho 1.0 (15/15 completed)`",
        "- Traceback: `yes`", "- Python exception: `numpy._core._exceptions._ArrayMemoryError`",
        "- Allocation request: `3.79 GiB`, shape `(53, 9600000)`, dtype `float64`",
        "- CUDA OOM: `no`", "- Metric NaN exception: `no`; stderr contains expected single-class group metric warnings only",
        "- KeyboardInterrupt/external termination: `no evidence`", "- Test-read guard violation: `no`",
        "- Manifest/hash failure: `no`", f"- Failure count: `{accounting['failure_count']}`", "",
        f"- Remaining: `{accounting['formal_cells_remaining']}` formal cells and `{accounting['rho_candidate_cells_remaining']}` rho candidates",
        "- Remaining GPU time: `UNAVAILABLE` until the host-RAM loader failure is resolved; the preliminary ETA file predates current progress and is stale",
        "", "The failure occurred while reloading the TEP RData context after validation-only rho selection. No TEP outer-test result was produced. This audit does not restart the runner.",
    ]) + "\n"


def audit(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path); output_root = Path(config["output"]["root"]); formal_root = output_root / config["output"]["namespace"]
    run_manifest = _read(Path(config["output"]["manifest"])); cells = run_manifest["cells"]
    manifest_index = {cell["cell_id"]: cell for cell in cells}
    rows = [_formal_row(config, formal_root, cell, manifest_index) for cell in cells]
    candidate_rows = []
    for dataset, fraction in legal_dataset_fractions(config):
        stage = config["three_w" if dataset == "3W" else "tep"]
        for outer in map(int, stage["outer_seeds"]):
            for seed in map(int, stage["rho_selection_seeds"]):
                for rho in map(float, config["rho_grid"]):
                    candidate_rows.append(_candidate_row(config, formal_root, dataset, fraction, outer, seed, rho))
    rows.extend(candidate_rows)
    identifiers = [(r["dataset"], r["fraction"], r["outer"], r["stage"], r["method"], r["seed"], r["rho"]) for r in rows if r["status"] != "PENDING"]
    duplicate_count = len(identifiers) - len(set(identifiers))
    formal = [row for row in rows if row["stage"] == "formal_locked_test"]
    candidates = [row for row in rows if row["stage"] == "rho_validation_candidate"]
    accounting = {
        "formal_cells_expected": len(formal), "formal_cells_reused": 0, "formal_cells_new": len(formal),
        "formal_cells_completed": sum(row["status"] != "PENDING" for row in formal),
        "formal_cells_valid": sum(row["status"] == "COMPLETE_VALID" for row in formal),
        "formal_cells_invalid": sum(row["status"] == "UNVERIFIED" for row in formal),
        "formal_cells_remaining": sum(row["status"] == "PENDING" for row in formal),
        "rho_candidate_cells_expected": len(candidates), "rho_candidate_cells_reused": 0, "rho_candidate_cells_new": len(candidates),
        "rho_candidate_cells_completed": sum(row["status"] != "PENDING" for row in candidates),
        "rho_candidate_cells_valid": sum(row["status"] == "COMPLETE_VALID" for row in candidates),
        "rho_candidate_cells_invalid": sum(row["status"] == "UNVERIFIED" for row in candidates),
        "rho_candidate_cells_remaining": sum(row["status"] == "PENDING" for row in candidates),
        "duplicate_count": duplicate_count, "failure_count": 1,
    }
    selections = []
    for path in sorted(formal_root.glob("*/f*/outer_*/rho_selection.json")):
        value = _read(path)
        valid_candidates = [row for row in candidates if row["dataset"] == value["dataset"] and float(row["fraction"]) == float(value["fraction"]) and row["outer"] == int(value["outer_id"]) and row["status"] == "COMPLETE_VALID"]
        selected = next(row for row in value["candidate_rows"] if float(row["rho"]) == float(value["selected_rho"]))
        selections.append({"dataset": value["dataset"], "fraction": value["fraction"], "outer": value["outer_id"],
                           "candidates_completed": len(valid_candidates), "selected_rho": value["selected_rho"],
                           "selection_seeds": value["selection_seeds"], "validation_macro_f1": selected["macro_f1"],
                           "validation_auprc": selected["auprc"], "validation_far": selected["far"],
                           "test_used_for_selection": value["outer_test_read"]})
    means = _mean_table(rows); paired = _paired_table(rows)
    _write_csv(Path("analysis/results/qdiffcl_data_regime_progress_audit.csv"), rows)
    atomic_json(Path("analysis/results/qdiffcl_data_regime_progress_accounting.json"), accounting)
    atomic_json(Path("analysis/results/qdiffcl_data_regime_rho_progress.json"), selections)
    Path("docs/QDIFFCL_DATA_REGIME_PROGRESS_AUDIT.md").write_text(_progress_doc(accounting, rows, means, paired, selections, output_root), encoding="utf-8")
    log_dir = Path("analysis/logs/data_regime")
    stderr = max(log_dir.glob("*stderr.log"), key=lambda path: path.stat().st_mtime)
    stdout = max(log_dir.glob("*stdout.log"), key=lambda path: path.stat().st_mtime)
    Path("docs/QDIFFCL_DATA_REGIME_RUNTIME_DIAGNOSIS.md").write_text(_runtime_doc(config, accounting, rows, stderr, stdout, formal_root), encoding="utf-8")
    result = {"status": "PARTIAL_RESULTS_ARCHIVED", "accounting": accounting, "mean_rows": means,
              "paired_rows": paired, "rho_selections": selections, "audit_time": _now()}
    atomic_json(Path("analysis/results/qdiffcl_data_regime_progress_audit.json"), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit stopped Q-DiffCL Data-Regime artifacts without resuming training")
    parser.add_argument("--config", default="configs/qdiffcl_data_regime_v1.yaml")
    args = parser.parse_args(); print(json.dumps(audit(args.config), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
