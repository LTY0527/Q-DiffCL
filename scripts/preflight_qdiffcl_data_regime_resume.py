from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.audit_qdiffcl_data_regime import atomic_json, canonical_hash, sha256_file
from scripts.run_qdiffcl_data_regime import (
    build_run_manifest,
    fraction_token,
    legal_dataset_fractions,
    load_config,
    read_json,
    runtime_implementation_hash,
    scientific_inputs_hash,
    validate_protocol,
    validate_runtime_amendment,
)


OUTPUT_PATH = Path("analysis/results/qdiffcl_data_regime_resume_preflight.json")


def _next_cell(config: dict[str, Any], manifest: dict[str, Any]) -> str:
    root = Path(config["output"]["root"]) / config["output"]["namespace"]
    for dataset, fraction in legal_dataset_fractions(config):
        stage = config["three_w" if dataset == "3W" else "tep"]
        for outer in map(int, stage["outer_seeds"]):
            cells = [
                cell for cell in manifest["cells"]
                if cell["dataset"] == dataset and float(cell["fraction"]) == fraction
                and int(cell["outer_id"]) == outer
            ]
            if cells and not all(cell.get("status") == "complete" for cell in cells):
                selection = root / dataset.lower() / fraction_token(fraction) / f"outer_{outer}" / "rho_selection.json"
                stage_name = "formal" if selection.exists() else "rho-selection"
                return f"{dataset} {fraction:.0%} outer{outer} {stage_name}"
    return "COMPLETE"


def _completed_artifacts_valid(manifest: dict[str, Any]) -> tuple[bool, int]:
    checked = 0
    for cell in manifest["cells"]:
        if cell.get("status") != "complete":
            continue
        result_path = Path(cell["result_path"])
        if not result_path.exists():
            return False, checked
        record = read_json(result_path)
        payload = dict(record)
        claimed = payload.pop("result_payload_sha256", None)
        if claimed != canonical_hash(payload) or cell.get("result_sha256") != claimed:
            return False, checked
        if record.get("checkpoint_sha256") != sha256_file(record["checkpoint_path"]):
            return False, checked
        if record.get("prediction_sha256") != sha256_file(record["prediction_path"]):
            return False, checked
        checked += 1
    return True, checked


def preflight(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    validation = validate_protocol(config, require_lock=True)
    lock = validation["lock"]
    assert lock is not None
    amendment_checks = validate_runtime_amendment(config, lock)
    manifest = build_run_manifest(config)
    artifacts_valid, checked = _completed_artifacts_valid(manifest)
    fraction_hashes_valid = all(
        sha256_file(path) == expected for path, expected in lock["fraction_manifest_hashes"].items()
    )
    payload = {
        "status": "DATA_REGIME_RESUME_PREFLIGHT_GO",
        "scientific_protocol_lock_valid": (
            manifest["protocol_hash"] == lock["protocol_hash"]
            and scientific_inputs_hash(config) == read_json(
                "analysis/results/qdiffcl_data_regime_runtime_amendment.json"
            )["scientific_inputs_hash"]
        ),
        "runtime_amendment_valid": all(amendment_checks.values()),
        "current_runtime_hash_registered": (
            runtime_implementation_hash()
            == read_json("analysis/results/qdiffcl_data_regime_runtime_amendment.json")["runtime_implementation_hash"]
        ),
        "fraction_manifest_hashes_valid": fraction_hashes_valid,
        "completed_artifact_hashes_valid": artifacts_valid,
        "completed_artifacts_checked": checked,
        "formal_completed": manifest["accounting"]["completed"],
        "test_read": False,
        "next_cell": _next_cell(config, manifest),
        "runtime_amendment_checks": amendment_checks,
    }
    required = (
        "scientific_protocol_lock_valid", "runtime_amendment_valid",
        "current_runtime_hash_registered", "fraction_manifest_hashes_valid",
        "completed_artifact_hashes_valid",
    )
    if not all(payload[key] for key in required):
        payload["status"] = "DATA_REGIME_RESUME_PREFLIGHT_HOLD"
    atomic_json(OUTPUT_PATH, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only zero-test-read Data-Regime resume preflight")
    parser.add_argument("--config", default="configs/qdiffcl_data_regime_v1.yaml")
    args = parser.parse_args()
    result = preflight(args.config)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "DATA_REGIME_RESUME_PREFLIGHT_GO":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
