from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from diffusion.fixed_views import sha256_file, sha256_strings
from frequency import fault_stages
from scripts.run_diffusion_quality_retest import _metrics, _probabilities
from scripts.run_domain_budget_routing import _load_tep_context, _read, run_tep, variant_name
from scripts.run_frequency_selective_r1_3seed import _stage_metrics
from scripts.run_stage_frequency_diffusion_mvp import _configure, _runtime, detection_delays, early_fault_recall
from trainers import build_model
from utils import select_device, write_json


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_test_view(base: dict[str, Any]) -> dict[str, np.ndarray]:
    manifest = _read(base["fixed_views"]["manifest"]); record = manifest["splits"]["test"]
    path = Path(record["path"])
    if sha256_file(path) != record["sha256"]:
        raise RuntimeError("locked TEP test view hash changed")
    with np.load(path, allow_pickle=False) as archive:
        bundle = {key: archive[key] for key in archive.files}
    if sha256_strings(list(map(str, bundle["window_id"]))) != record["window_ids_sha256"]:
        raise RuntimeError("locked TEP test order changed")
    return bundle


def _tep_test(config: dict[str, Any], final: dict[str, Any], device: str) -> dict[str, Any]:
    rho = float(final["domain_rho"]["TEP"]); seeds = list(map(int, final["locked_test_seeds"]["TEP"]))
    run_tep(config, _load_tep_context(config), [rho], seeds, device)
    base = yaml.safe_load(Path(config["tep"]["base_config"]).read_text(encoding="utf-8")); _configure(base)
    test = _load_test_view(base); stages = fault_stages(test, base); clean = test["clean"].astype(np.float32)
    results = {}
    for seed in seeds:
        checkpoint = Path(config["tep"]["output_dir"]) / variant_name(rho) / f"seed_{seed}" / "model.pt"
        metrics_path = checkpoint.parent / "metrics.json"; validation = _read(metrics_path)
        runtime = _runtime(base, seed); runtime["diagnosis"] = {"threshold_band_width": .05, "high_correlation_quantile": .90}
        model = build_model(runtime["model"], clean.shape[1], 2).to(device)
        payload = torch.load(checkpoint, map_location=device, weights_only=True); model.load_state_dict(payload["model_state_dict"])
        probability, _ = _probabilities(model, clean, int(runtime["batch_size"]), device)
        threshold = float(validation["validation_threshold"]); scores = probability[:, 1]; prediction = scores >= threshold
        results[str(seed)] = {"metrics": _metrics(test["labels"], scores, threshold),
                              "early_fault": early_fault_recall(prediction, stages),
                              "detection_delay": detection_delays(test, prediction, runtime),
                              "stages": _stage_metrics(stages, prediction),
                              "validation_threshold": threshold,
                              "checkpoint_sha256": _sha(checkpoint)}
    return results


def run(config: dict[str, Any], final_path: Path) -> dict[str, Any]:
    final = yaml.safe_load(final_path.read_text(encoding="utf-8"))
    if not final.get("frozen") or not final.get("rho_must_not_change_after_test") or final.get("test_used_for_selection"):
        raise RuntimeError("DCBR must be validation-frozen before locked test")
    selection = _read(config["output"]["selection"])
    if selection.get("test_read") or float(final["domain_rho"]["3W"]) not in selection["top2"]["3W"] or float(final["domain_rho"]["TEP"]) not in selection["top2"]["TEP"]:
        raise RuntimeError("frozen rho does not match validation Top-2")
    frozen_hash = _sha(final_path); device = select_device(str(config["device"]))
    tep = _tep_test(config, final, device)
    three_manifest = _read(config["three_w"]["final_test_manifest"])["results"]
    three = {str(seed): three_manifest[f"FINAL_QDIFFCL|{seed}"]["metrics"]
             for seed in map(int, final["locked_test_seeds"]["3W"])}
    payload = {"status": "LOCKED_TEST_COMPLETE", "frozen_config_sha256": frozen_hash,
               "rho": final["domain_rho"], "rho_changed_after_test": False,
               "selection_split": "validation", "test_used_for_selection": False,
               "results": {"3W": three, "TEP": tep}}
    write_json(Path(config["output"]["root"]) / "locked_test.json", payload); return payload


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/domain_calibrated_budget_routing.yaml")
    parser.add_argument("--final", type=Path, default=Path("configs/domain_calibrated_budget_routing_final.yaml"))
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = run(config, args.final); print(json.dumps({"status": result["status"], "rho": result["rho"]}, ensure_ascii=False))


if __name__ == "__main__": main()
