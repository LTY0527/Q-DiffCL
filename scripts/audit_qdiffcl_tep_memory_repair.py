from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.audit_qdiffcl_data_regime import atomic_json
from scripts.run_qdiffcl_data_regime import _mask_hash, load_config, prepare_tep


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def memory_snapshot() -> dict[str, int]:
    status = MEMORYSTATUSEX(); status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError()
    counters = PROCESS_MEMORY_COUNTERS(); counters.cb = ctypes.sizeof(counters)
    get_current_process = ctypes.windll.kernel32.GetCurrentProcess
    get_current_process.restype = ctypes.c_void_p
    get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_process_memory_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), ctypes.c_ulong]
    process = get_current_process()
    if not get_process_memory_info(process, ctypes.byref(counters), counters.cb):
        raise ctypes.WinError()
    mib = 1024 * 1024
    return {
        "system_available_mb": int(status.ullAvailPhys / mib),
        "system_total_mb": int(status.ullTotalPhys / mib),
        "working_set_mb": int(counters.WorkingSetSize / mib),
        "peak_working_set_mb": int(counters.PeakWorkingSetSize / mib),
        "pagefile_usage_mb": int(counters.PagefileUsage / mib),
    }


def _hash_arrays(values: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def context_signature(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "context_hash": context["context_hash"],
        "fraction_manifest_hash": context["fraction_record"]["sha256"],
        "criticality_mask_hash": _mask_hash(context),
        "criticality_input_train_hash": _hash_arrays([context["bundles"]["train"]["clean"]]),
        "train_hash": _hash_arrays([context["train"], context["labels"]["train"]]),
        "validation_hash": _hash_arrays([context["validation"], context["labels"]["validation"]]),
        "test_context_hash": _hash_arrays([
            context["bundles"]["test"]["clean"], context["bundles"]["test"]["labels"],
        ]),
        "ids_hash": _hash_arrays([context["ids"][name] for name in ("train", "validation", "test")]),
        "window_shapes": {
            name: list(context["bundles"][name]["clean"].shape)
            for name in ("train", "validation", "test")
        },
        "finite_counts": {
            name: int(np.isfinite(context["bundles"][name]["clean"]).sum())
            for name in ("train", "validation", "test")
        },
    }


def smoke_fraction(config: dict[str, Any], fraction: float, outer: int, cycles: int) -> dict[str, Any]:
    namespace = "TEP_MEMORY_REPAIR_SMOKE"
    records = []
    for cycle in range(1, cycles + 1):
        before = memory_snapshot()
        if before["system_available_mb"] < 2048:
            return {
                "status": "RAM_SAFETY_HOLD", "fraction": fraction, "outer": outer,
                "cycle": cycle, "reason": "system available RAM below 2048 MiB before context build",
                "memory": before, "records": records,
            }
        context = prepare_tep(config, outer, fraction, namespace)
        during = memory_snapshot(); signature = context_signature(context)
        del context
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        after = memory_snapshot()
        records.append({"cycle": cycle, "before": before, "during": during, "after_release": after,
                        "signature": signature})
    exact_repeat = all(record["signature"] == records[0]["signature"] for record in records[1:])
    post_release_growth = records[-1]["after_release"]["working_set_mb"] - records[0]["after_release"]["working_set_mb"]
    return {
        "status": "RAM_SMOKE_GO" if exact_repeat and post_release_growth < 512 else "RAM_SMOKE_HOLD",
        "fraction": fraction, "outer": outer, "cycles": cycles,
        "exact_repeat": exact_repeat, "post_release_working_set_growth_mb": post_release_growth,
        "records": records, "outer_test_metrics_read": False,
        "large_consolidation_allocation_removed": True,
    }


def archived_equivalence(config: dict[str, Any], result: dict[str, Any], outer: int) -> dict[str, Any]:
    formal_root = Path(config["output"]["root"]) / config["output"]["namespace"]
    smoke_root = Path(config["output"]["root"]) / "TEP_MEMORY_REPAIR_SMOKE"
    relative = Path("tep") / "f100" / f"outer_{outer}" / "_context" / "audit.json"
    old = json.loads((formal_root / relative).read_text(encoding="utf-8"))
    new = json.loads((smoke_root / relative).read_text(encoding="utf-8"))
    signature = next(row for row in result["fractions"] if float(row["fraction"]) == 1.0)["records"][0]["signature"]
    checks = {
        "context_hash_exact": old["context_hash"] == new["context_hash"] == signature["context_hash"],
        "fraction_manifest_hash_exact": old["fraction_manifest_hash"] == new["fraction_manifest_hash"] == signature["fraction_manifest_hash"],
        "criticality_mask_hash_exact": old["criticality_mask_sha256"] == new["criticality_mask_sha256"] == signature["criticality_mask_hash"],
        "criticality_npz_hash_exact": old["criticality_sha256"] == new["criticality_sha256"],
        "scaler_exact": old["scaler"] == new["scaler"],
        "selected_train_ids_exact": old["selected_train_source_ids"] == new["selected_train_source_ids"],
        "validation_ids_exact": old["validation_groups"] == new["validation_groups"],
        "test_ids_exact": old["test_groups"] == new["test_groups"],
        "window_counts_exact": (
            old["train_windows"] == new["train_windows"] and
            old["validation_windows"] == new["validation_windows"]
        ),
    }
    return {
        "status": "TEP_MEMORY_SAFE_LOADER_EQUIVALENCE_GO" if all(checks.values()) else "TEP_MEMORY_SAFE_LOADER_EQUIVALENCE_HOLD",
        "checks": checks, "pre_repair_context_hash": old["context_hash"],
        "post_repair_context_hash": new["context_hash"],
        "pre_repair_criticality_sha256": old["criticality_sha256"],
        "post_repair_criticality_sha256": new["criticality_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Context-only TEP memory-repair smoke; never evaluates test metrics")
    parser.add_argument("--config", default="configs/qdiffcl_data_regime_v1.yaml")
    parser.add_argument("--outer-id", type=int, default=32001)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    output = Path("analysis/results/qdiffcl_tep_memory_smoke.json")
    if args.verify_existing:
        result = json.loads(output.read_text(encoding="utf-8"))
    else:
        result = {
            "status": "TEP_MEMORY_SAFE_LOADER_SMOKE",
            "pid": os.getpid(), "fractions": [], "outer_test_metrics_read": False,
        }
        for fraction in (1.0, 0.25):
            record = smoke_fraction(config, fraction, args.outer_id, args.cycles)
            result["fractions"].append(record)
            if record["status"] != "RAM_SMOKE_GO":
                result["status"] = "TEP_MEMORY_SAFE_LOADER_SMOKE_HOLD"
                break
        if all(record["status"] == "RAM_SMOKE_GO" for record in result["fractions"]):
            result["status"] = "TEP_MEMORY_SAFE_LOADER_SMOKE_GO"
    if result["status"] == "TEP_MEMORY_SAFE_LOADER_SMOKE_GO":
        result["archived_equivalence"] = archived_equivalence(config, result, args.outer_id)
        if result["archived_equivalence"]["status"] != "TEP_MEMORY_SAFE_LOADER_EQUIVALENCE_GO":
            result["status"] = "TEP_MEMORY_SAFE_LOADER_SMOKE_HOLD"
    atomic_json(output, result)
    print(json.dumps(result, indent=2))
    if result["status"] != "TEP_MEMORY_SAFE_LOADER_SMOKE_GO":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
