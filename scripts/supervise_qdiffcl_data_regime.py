from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.audit_qdiffcl_data_regime import atomic_json
from scripts.audit_qdiffcl_tep_memory_repair import memory_snapshot
from scripts.run_qdiffcl_data_regime import fraction_token, load_config


STATUS_PATH = Path("analysis/results/qdiffcl_data_regime_supervisor_status.json")
FAILURE_PATH = Path("analysis/results/qdiffcl_data_regime_supervisor_failures.json")
LOG_PATH = Path("analysis/logs/data_regime/supervisor_night.log")


def _read(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _formal_completed(config: dict[str, Any]) -> int:
    manifest = _read(Path(config["output"]["manifest"]), {})
    return sum(row.get("status") == "complete" for row in manifest.get("cells", []))


def _rho_completed(config: dict[str, Any]) -> int:
    root = Path(config["output"]["root"]) / config["output"]["namespace"]
    completed = 0
    for dataset, key in (("3W", "three_w"), ("TEP", "tep")):
        for fraction in config["fractions"]:
            if dataset == "TEP" and float(fraction) == 0.10:
                continue
            for outer in config[key]["outer_seeds"]:
                base = root / dataset.lower() / fraction_token(float(fraction)) / f"outer_{outer}"
                for seed in config[key]["rho_selection_seeds"]:
                    for rho in config["rho_grid"]:
                        path = (
                            base / f"model_seed_{seed}" / "FINAL_QDIFFCL_FIXED" / "_training"
                            if float(rho) == 1.0 else
                            base / f"model_seed_{seed}" / "CALIBRATED_RHO" / "_candidates" /
                            f"rho_{int(round(float(rho) * 100)):03d}"
                        )
                        completed += int((path / "model.pt").exists() and (path / "validation.json").exists())
    return completed


def _gpu() -> tuple[str, str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
        check=False, capture_output=True, text=True, encoding="utf-8",
    )
    fields = [part.strip() for part in result.stdout.strip().split(",")]
    return (fields[0], fields[1]) if len(fields) >= 2 else ("N/A", "N/A")


def _last_artifact(config: dict[str, Any]) -> tuple[str | None, str | None]:
    root = Path(config["output"]["root"])
    files = [path for path in root.rglob("*") if path.is_file()]
    if not files:
        return None, None
    path = max(files, key=lambda item: item.stat().st_mtime)
    return str(path), datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def _commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, encoding="utf-8").strip()


def heartbeat(config: dict[str, Any], process: subprocess.Popen, args: argparse.Namespace,
              start_formal: int, start_rho: int, exit_code: int | None = None) -> dict[str, Any]:
    runtime = _read(Path(config["output"]["runtime_status"]), {})
    formal = _formal_completed(config); rho = _rho_completed(config)
    gpu_util, gpu_memory = _gpu(); memory = memory_snapshot(); artifact, artifact_time = _last_artifact(config)
    failures = _read(FAILURE_PATH, {"failures": []}).get("failures", [])
    payload = {
        "timestamp": _now(), "runner_pid": process.pid, "runner_exit_code": exit_code,
        "dataset": args.dataset, "fraction": args.fraction, "outer": args.outer_id,
        "stage": runtime.get("stage", "launching"),
        "rho_completed": rho, "rho_expected": 225,
        "formal_completed": formal, "formal_expected": 375,
        "session_new_rho": rho - start_rho, "session_new_formal": formal - start_formal,
        "failures": len(failures), "duplicates": 0,
        "gpu_util": gpu_util, "gpu_memory_mb": gpu_memory,
        "host_ram_available_mb": memory["system_available_mb"],
        "last_artifact": artifact, "last_artifact_mtime": artifact_time,
        "last_commit": _commit(),
        "next_action": "audit completed outer" if exit_code == 0 else "continue supervised current outer" if exit_code is None else "stop and diagnose failure",
    }
    atomic_json(STATUS_PATH, payload)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return payload


def record_failure(process: subprocess.Popen, args: argparse.Namespace, exit_code: int) -> None:
    ledger = _read(FAILURE_PATH, {"status": "DATA_REGIME_FAILURE_LEDGER", "failures": []})
    ledger["failures"].append({
        "timestamp": _now(), "runner_pid": process.pid, "exit_code": exit_code,
        "dataset": args.dataset, "fraction": args.fraction, "outer": args.outer_id,
        "stage": args.stage, "classification": "SUPERVISED_RUNNER_FAILURE",
    })
    atomic_json(FAILURE_PATH, ledger)


def main() -> None:
    parser = argparse.ArgumentParser(description="Foreground supervisor for one frozen Data-Regime outer")
    parser.add_argument("--dataset", required=True, choices=("3W", "TEP"))
    parser.add_argument("--fraction", required=True, type=float, choices=(1.0, 0.25, 0.10))
    parser.add_argument("--outer-id", required=True, type=int)
    parser.add_argument("--stage", default="formal", choices=("rho-selection", "formal", "all"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--interval-seconds", type=int, default=60)
    args = parser.parse_args(); config = load_config("configs/qdiffcl_data_regime_v1.yaml")
    memory = memory_snapshot()
    if memory["system_available_mb"] < 2048:
        raise RuntimeError(f"RAM safety hold before outer: {memory['system_available_mb']} MiB available")
    start_formal = _formal_completed(config); start_rho = _rho_completed(config)
    command = [sys.executable, "-u", "-m", "scripts.run_qdiffcl_data_regime", "--stage", args.stage,
               "--dataset", args.dataset, "--fraction", str(args.fraction), "--outer-id", str(args.outer_id),
               "--device", args.device]
    process = subprocess.Popen(command)
    heartbeat(config, process, args, start_formal, start_rho)
    while True:
        try:
            exit_code = process.wait(timeout=args.interval_seconds)
            break
        except subprocess.TimeoutExpired:
            heartbeat(config, process, args, start_formal, start_rho)
    if exit_code != 0:
        record_failure(process, args, exit_code)
    heartbeat(config, process, args, start_formal, start_rho, exit_code)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
