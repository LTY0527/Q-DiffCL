from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

import scripts.run_posthoc_recent_baselines as h1
from baselines.posthoc_recent import TRACKS
from utils import select_device, write_json


FORMAL_PYTHON = Path(r"E:\anaconda\envs\qdiffcl\python.exe")
EVIDENCE_CLASS = "POSTHOC_BASELINE_5SEED_EXTENSION"
PREPARED = "POSTHOC_BASELINE_5SEED_EXTENSION_PREPARED"
RUNNING = "POSTHOC_BASELINE_5SEED_EXTENSION_RUNNING"
INTERRUPTED = "POSTHOC_BASELINE_5SEED_EXTENSION_INTERRUPTED"
CELLS_COMPLETE = "POSTHOC_BASELINE_5SEED_EXTENSION_CELLS_COMPLETE"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def git(*args: str, check: bool = True) -> str:
    result = subprocess.run(["git", *args], text=True, capture_output=True, check=False)
    if check and result.returncode:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def build_cells(config: dict[str, Any]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for dataset in ("3W", "TEP"):
        for outer in map(int, config["outer_splits"][dataset]):
            for seed in map(int, config["missing_seeds"][dataset]):
                for method in config["active_methods"]:
                    cells.append({
                        "run_id": h1._cell_id(dataset, outer, seed, method),
                        "dataset": dataset,
                        "outer_seed": outer,
                        "model_seed": seed,
                        "method": method,
                        "track": config["tracks"][method],
                        "status": "pending",
                    })
    return cells


def locked_protocol(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_class": config["evidence_class"],
        "h1_archive_commit": config["h1_archive_commit"],
        "paper_final_source_commit": config["paper_final_source_commit"],
        "selection_hash": config["selection_hash"],
        "active_methods": config["active_methods"],
        "tracks": config["tracks"],
        "h1_completed_seeds": config["h1_completed_seeds"],
        "missing_seeds": config["missing_seeds"],
        "full_seeds": config["full_seeds"],
        "outer_splits": config["outer_splits"],
        "expected_new_cells": config["expected_new_cells"],
        "expected_h1_reuse_cells": config["expected_h1_reuse_cells"],
        "locked_hashes": config["locked_hashes"],
        "official_commits": config["official_commits"],
        "statistics": config["statistics"],
    }


def runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    base = yaml.safe_load(Path(config["h1_config"]).read_text(encoding="utf-8"))
    runtime = copy.deepcopy(base)
    runtime["evidence_class"] = EVIDENCE_CLASS
    runtime["benchmark"]["outer_splits"] = copy.deepcopy(config["outer_splits"])
    runtime["benchmark"]["model_seeds"] = copy.deepcopy(config["missing_seeds"])
    runtime["output"]["root"] = config["output"]["root"]
    runtime["output"]["manifest"] = config["output"]["manifest"]
    # Reuse the already-passed H1 sanity audit; no outer-test data is read here.
    runtime["output"]["sanity_csv"] = base["output"]["sanity_csv"]
    return runtime


def _validate_h1_results(config: dict[str, Any]) -> None:
    manifest = read_json(config["h1_manifest"])
    if manifest.get("status") != "POSTHOC_BASELINE_BENCHMARK_COMPLETE":
        raise RuntimeError("H1 manifest is not complete")
    cells = manifest.get("cells", [])
    if len(cells) != int(config["expected_h1_reuse_cells"]) or any(cell.get("status") != "complete" for cell in cells):
        raise RuntimeError("H1 manifest is not exactly 72/72 complete")
    for cell in cells:
        result = read_json(cell["result_path"])
        if result.get("outer_test_evaluated_once") is not True:
            raise RuntimeError(f"H1 outer-once marker invalid: {result.get('run_id')}")
        if sha256_file(result["prediction_path"]) != result["prediction_sha256"]:
            raise RuntimeError(f"H1 prediction hash invalid: {result['run_id']}")
        if sha256_file(result["checkpoint_path"]) != result["checkpoint_sha256"]:
            raise RuntimeError(f"H1 checkpoint hash invalid: {result['run_id']}")


def validate_static(config: dict[str, Any]) -> dict[str, Any]:
    executable = Path(os.path.abspath(os.sys.executable))
    if executable.resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"H1.5 must use {FORMAL_PYTHON}, observed {executable}")
    if not torch.cuda.is_available():
        raise RuntimeError("formal qdiffcl CUDA environment is unavailable")
    if config["evidence_class"] != EVIDENCE_CLASS:
        raise RuntimeError("extension evidence class changed")
    if config["active_methods"] != ["TF-C", "SoftCLT", "TS2Vec", "AutoTCL"]:
        raise RuntimeError("the four H1 baselines changed")
    if config["missing_seeds"] != {"3W": [45, 46], "TEP": [43, 44]}:
        raise RuntimeError("missing seed lock changed")
    if config["full_seeds"] != {"3W": [42, 43, 44, 45, 46], "TEP": [7, 42, 43, 44, 2026]}:
        raise RuntimeError("full five-seed protocol changed")
    if config["outer_splits"] != {"3W": [31001, 31002, 31003], "TEP": [32001, 32002, 32003]}:
        raise RuntimeError("outer split lock changed")
    cells = build_cells(config)
    if len(cells) != int(config["expected_new_cells"]) or len({cell["run_id"] for cell in cells}) != 48:
        raise RuntimeError("extension is not exactly 48 unique cells")
    if git("branch", "--show-current") != "exp/posthoc-baseline-expansion":
        raise RuntimeError("unexpected branch")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", config["h1_archive_commit"], "HEAD"],
        check=False,
    )
    if ancestor.returncode:
        raise RuntimeError("H1 archive commit is not an ancestor of HEAD")
    paths = {
        "adapter": "baselines/posthoc_recent.py",
        "h1_config": config["h1_config"],
        "h1_runner": "scripts/run_posthoc_recent_baselines.py",
        "h1_manifest": config["h1_manifest"],
        "h1_raw_csv": config["h1_raw_csv"],
        "paper_final_config": config["paper_final_config"],
        "paper_final_raw_csv": config["paper_final_raw_csv"],
    }
    observed = {name: sha256_file(path) for name, path in paths.items()}
    if observed != config["locked_hashes"]:
        raise RuntimeError(f"frozen source hash changed: {observed}")
    h1_config = yaml.safe_load(Path(config["h1_config"]).read_text(encoding="utf-8"))
    if h1_config["active_methods"] != config["active_methods"] or h1_config["selection_hash"] != config["selection_hash"]:
        raise RuntimeError("H1 method/selection boundary changed")
    paper = yaml.safe_load(Path(config["paper_final_config"]).read_text(encoding="utf-8"))
    for dataset, key in (("3W", "three_w"), ("TEP", "tep")):
        if list(map(int, paper[key]["outer_seeds"])) != config["outer_splits"][dataset]:
            raise RuntimeError(f"{dataset} Paper-final outer splits changed")
        if list(map(int, paper[key]["model_seeds"])) != config["full_seeds"][dataset]:
            raise RuntimeError(f"{dataset} Paper-final five seeds changed")
        if set(config["h1_completed_seeds"][dataset]) & set(config["missing_seeds"][dataset]):
            raise RuntimeError(f"{dataset} completed/missing seed sets overlap")
    matrix = list(csv.DictReader(Path("analysis/results/posthoc_baseline_candidate_matrix.csv").open(encoding="utf-8-sig")))
    official = {row["method"]: row["official_commit_sha"] for row in matrix if row["method"] in config["active_methods"]}
    if official != config["official_commits"]:
        raise RuntimeError("official source commit lock changed")
    _validate_h1_results(config)
    return {"paper": paper, "h1_config": h1_config, "protocol_hash": canonical_hash(locked_protocol(config))}


def prepare_manifest(config: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    output_root = Path(config["output"]["root"])
    if output_root.exists() and any(output_root.rglob("result.json")):
        raise RuntimeError("new extension outer metrics already exist before protocol preparation")
    manifest = {
        "status": PREPARED,
        "evidence_class": EVIDENCE_CLASS,
        "protocol_hash": audit["protocol_hash"],
        "h1_archive_commit": config["h1_archive_commit"],
        "paper_final_source_commit": config["paper_final_source_commit"],
        "selection_hash": config["selection_hash"],
        "created_at": now(),
        "outer_test_metrics_read_before_lock": False,
        "expected_new_cells": 48,
        "completed_cells": 0,
        "h1_cells_reused": 72,
        "duplicate_training_cells": 0,
        "provenance": {
            "locked_hashes": config["locked_hashes"],
            "official_commits": config["official_commits"],
            "extension_runner_sha256": sha256_file(__file__),
            "extension_config_sha256": sha256_file("configs/posthoc_baseline_5seed_extension.yaml"),
            "summarizer_sha256": sha256_file("scripts/summarize_posthoc_baseline_5seed_extension.py"),
        },
        "cells": build_cells(config),
        "failures": [],
    }
    write_json(Path(config["output"]["manifest"]), manifest)
    return manifest


def _protocol_files(config: dict[str, Any]) -> list[str]:
    return [
        "configs/posthoc_baseline_5seed_extension.yaml",
        "scripts/run_posthoc_baseline_5seed_extension.py",
        "scripts/summarize_posthoc_baseline_5seed_extension.py",
        "tests/test_posthoc_baseline_5seed_extension.py",
        config["protocol_doc"],
    ]


def validate_committed_lock(config: dict[str, Any], manifest: dict[str, Any], audit: dict[str, Any]) -> str:
    if manifest.get("protocol_hash") != audit["protocol_hash"]:
        raise RuntimeError("extension protocol hash changed")
    if manifest.get("expected_new_cells") != 48 or len(manifest.get("cells", [])) != 48:
        raise RuntimeError("extension manifest cell count changed")
    provenance = manifest["provenance"]
    expected = {
        "extension_runner_sha256": sha256_file(__file__),
        "extension_config_sha256": sha256_file("configs/posthoc_baseline_5seed_extension.yaml"),
        "summarizer_sha256": sha256_file("scripts/summarize_posthoc_baseline_5seed_extension.py"),
    }
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise RuntimeError("extension implementation changed after lock")
    dirty = git("status", "--porcelain", "--", *_protocol_files(config))
    if dirty:
        raise RuntimeError(f"extension protocol files are not committed cleanly:\n{dirty}")
    if git("rev-parse", "HEAD") == config["h1_archive_commit"]:
        raise RuntimeError("protocol lock commit has not been created")
    return git("rev-parse", "HEAD")


def _store_manifest(config: dict[str, Any], manifest: dict[str, Any]) -> None:
    manifest["completed_cells"] = sum(cell.get("status") == "complete" for cell in manifest["cells"])
    write_json(Path(config["output"]["manifest"]), manifest)


def _completed_result(runtime: dict[str, Any], cell: dict[str, Any], protocol_hash: str) -> dict[str, Any] | None:
    result_path = h1._cell_dir(runtime, cell["dataset"], cell["outer_seed"], cell["model_seed"], cell["method"]) / "result.json"
    if not result_path.exists():
        return None
    result = read_json(result_path)
    if result.get("evidence_class") != EVIDENCE_CLASS or result.get("outer_test_evaluated_once") is not True:
        raise RuntimeError(f"extension evidence boundary invalid: {cell['run_id']}")
    if result.get("extension_provenance", {}).get("protocol_hash") != protocol_hash:
        raise RuntimeError(f"extension protocol provenance mismatch: {cell['run_id']}")
    if sha256_file(result["prediction_path"]) != result["prediction_sha256"]:
        raise RuntimeError(f"extension prediction hash invalid: {cell['run_id']}")
    if sha256_file(result["checkpoint_path"]) != result["checkpoint_sha256"]:
        raise RuntimeError(f"extension checkpoint hash invalid: {cell['run_id']}")
    return result


def run_benchmark(
    config: dict[str, Any],
    audit: dict[str, Any],
    method_filter: str | None,
    dataset_filter: str | None,
) -> list[dict[str, Any]]:
    manifest = read_json(config["output"]["manifest"])
    protocol_commit = validate_committed_lock(config, manifest, audit)
    manifest["protocol_lock_commit"] = protocol_commit
    manifest["status"] = RUNNING
    _store_manifest(config, manifest)
    runtime = runtime_config(config)
    sanity = list(csv.DictReader(Path(runtime["output"]["sanity_csv"]).open(encoding="utf-8-sig")))
    relevant = [row for row in sanity if row["method"] in set(config["active_methods"])]
    if len(relevant) != 8 or any(row["status"] != "PASS" or row["outer_test_metric_read"].lower() != "false" for row in relevant):
        raise RuntimeError("the eight frozen H1 sanity cells are not valid")
    h1.EVIDENCE_CLASS = EVIDENCE_CLASS
    device = select_device(str(config["device"]))
    completed_now: list[dict[str, Any]] = []
    for dataset in ("3W", "TEP"):
        if dataset_filter and dataset != dataset_filter:
            continue
        for outer in map(int, config["outer_splits"][dataset]):
            context = h1.prepare_context(audit["paper"], dataset, outer, device, True)
            for seed in map(int, config["missing_seeds"][dataset]):
                for method in config["active_methods"]:
                    if method_filter and method != method_filter:
                        continue
                    run_id = h1._cell_id(dataset, outer, seed, method)
                    cell = next(row for row in manifest["cells"] if row["run_id"] == run_id)
                    provenance = {
                        "protocol_hash": audit["protocol_hash"],
                        "protocol_lock_commit": protocol_commit,
                        "h1_archive_commit": config["h1_archive_commit"],
                        "extension_runner_sha256": manifest["provenance"]["extension_runner_sha256"],
                    }
                    try:
                        result = _completed_result(runtime, cell, audit["protocol_hash"])
                        if result is None:
                            validation, model, probe = h1.train_cell(runtime, context, method, seed, device, False)
                            prior = validation.get("extension_provenance")
                            if prior is not None and prior != provenance:
                                raise RuntimeError(f"validation provenance mismatch: {run_id}")
                            validation["extension_provenance"] = provenance
                            validation_path = h1._cell_dir(runtime, dataset, outer, seed, method) / "validation.json"
                            write_json(validation_path, validation)
                            result = h1.evaluate_cell(runtime, context, method, seed, model, probe, validation, device)
                            result["extension_provenance"] = provenance
                            write_json(h1._cell_dir(runtime, dataset, outer, seed, method) / "result.json", result)
                        cell.update({
                            "status": "complete",
                            "result_path": str(h1._cell_dir(runtime, dataset, outer, seed, method) / "result.json"),
                            "completed_at": result["completed_at"],
                        })
                        completed_now.append(result)
                        _store_manifest(config, manifest)
                    except Exception as error:
                        cell["status"] = "failed"
                        manifest["failures"].append({
                            "run_id": run_id,
                            "type": type(error).__name__,
                            "message": str(error),
                            "at": now(),
                        })
                        manifest["status"] = INTERRUPTED
                        _store_manifest(config, manifest)
                        raise
    if all(cell["status"] == "complete" for cell in manifest["cells"]):
        manifest["status"] = CELLS_COMPLETE
        manifest["cells_completed_at"] = now()
    _store_manifest(config, manifest)
    return completed_now


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/posthoc_baseline_5seed_extension.yaml")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--method", choices=("TF-C", "SoftCLT", "TS2Vec", "AutoTCL"))
    parser.add_argument("--dataset", choices=("3W", "TEP"))
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    audit = validate_static(config)
    print(json.dumps({
        "status": "POSTHOC_BASELINE_5SEED_EXTENSION_PREFLIGHT_GO",
        "python": os.sys.executable,
        "cuda": torch.cuda.is_available(),
        "protocol_hash": audit["protocol_hash"],
    }), flush=True)
    if args.prepare:
        manifest = prepare_manifest(config, audit)
        print(json.dumps({"status": manifest["status"], "cells": len(manifest["cells"])}), flush=True)
    elif args.audit:
        manifest = read_json(config["output"]["manifest"])
        print(json.dumps({"status": manifest["status"], "completed_cells": manifest["completed_cells"]}), flush=True)
    elif args.benchmark:
        rows = run_benchmark(config, audit, args.method, args.dataset)
        print(json.dumps({"completed_this_invocation": len(rows)}), flush=True)
    else:
        raise SystemExit("choose --prepare, --audit, or --benchmark")


if __name__ == "__main__":
    main()
