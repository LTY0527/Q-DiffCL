from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch
import yaml

from augmentations import stochastic_view_route
from diffusion import DiffusionSchedule, FrequencyForwardDiffusion, fit_spectral_statistics
from scripts.run_3w_diffusion_1seed import METHODS as THREE_W_METHODS, run as run_three_w_model
from scripts.run_budget_shrinkage_diagnostic import validation_metrics
from scripts.run_diffusion_quality_retest import _state_hash, epoch_orders
from scripts.run_domain_budget_routing import _load_tep_context, _read
from scripts.run_frequency_selective_r1_3seed import _fit_method, file_sha256
from scripts.run_stage_frequency_diffusion_mvp import _runtime, augmentation_mechanism_metrics
from trainers import build_model
from utils import seed_everything, select_device, write_json


def variant_name(p: float) -> str:
    return f"SVR_{int(round(float(p) * 100)):03d}"


def _manifest(path: Path) -> dict[str, Any]:
    return _read(path).get("results", {}) if path.exists() else {}


def _store(path: Path, records: dict[str, Any], key: str, record: dict[str, Any]) -> None:
    records[key] = record
    write_json(path, {"results": records, "evaluation_split": "validation", "test_read": False})


def _validate(config: dict[str, Any]) -> None:
    final = yaml.safe_load(Path(config["final_config"]).read_text(encoding="utf-8"))
    expected = {"weight_discriminative": .5, "weight_early": .5, "weight_run_stability": 0.0}
    if not final.get("frozen") or final.get("weights") != expected:
        raise RuntimeError("FINAL_QDIFFCL weights changed")
    if list(map(float, config["candidates"])) != [0, .25, .5, .75, 1]:
        raise RuntimeError("SVR candidate grid changed")
    if float(config["scaling_std"]) != .05:
        raise RuntimeError("frozen SCALING std changed")
    for dataset, key in (("3W", "three_w"), ("TEP", "tep")):
        criticality = _read(config[key]["final_mask"])["criticality"]
        if criticality["mask_sha256"] != final["mask_sha256"][dataset]:
            raise RuntimeError(f"{dataset} FINAL mask changed")
        if criticality["fit_split"] != "train":
            raise RuntimeError(f"{dataset} mask is not train-only")


def _ids(refs: list[Any]) -> np.ndarray:
    return np.asarray([f"{ref.instance_id}:{ref.start}:{ref.target}" for ref in refs])


def run_three_w(config: dict[str, Any], data_root: Path, probabilities: list[float],
                seeds: list[int], device: str) -> dict[str, Any]:
    stage = config["three_w"]
    output = Path(stage["output_dir"]); path = output / "manifest.json"; records = _manifest(path)
    for probability in probabilities:
        for seed in seeds:
            key = f"{variant_name(probability)}|{seed}"
            if key in records:
                continue
            base = yaml.safe_load(Path(stage["base_config"]).read_text(encoding="utf-8"))
            current = copy.deepcopy(base)
            current.update({
                "seed": seed,
                "protocol_seed": int(stage["protocol_seed"]),
                "criticality_source": str(stage["final_mask"]),
                "methods": [THREE_W_METHODS[2]],
                "evaluation_split": "validation",
                "device": device,
                "output_dir": str(output / variant_name(probability) / f"seed_{seed}"),
                "stochastic_view_routing": {
                    "p": probability, "scaling_std": float(config["scaling_std"]),
                    "router_seed": seed + 71000, "scaling_seed": seed,
                    "validation_router_seed": seed + 71100,
                    "validation_scaling_seed": seed + 100,
                },
            })
            current["training"]["supcon_batching"] = "original"
            result = run_three_w_model(current, data_root)
            if result["evaluation_split"] != "validation":
                raise RuntimeError("3W SVR attempted non-validation evaluation")
            route = result["augmentation_diagnostics"][THREE_W_METHODS[2]]
            record = {
                "variant": variant_name(probability), "p": probability, "seed": seed,
                "evaluation_split": "validation", "method": result["methods"][THREE_W_METHODS[2]],
                "fairness": result["fairness"], "routing_audit": route,
                "mask_sha256": _read(stage["final_mask"])["criticality"]["mask_sha256"],
                "test_metrics_read": False, "training": "new_training",
            }
            _store(path, records, key, record)
    return records


def run_tep(config: dict[str, Any], probabilities: list[float], seeds: list[int], device: str) -> dict[str, Any]:
    stage = config["tep"]
    base, views, clean, stages = _load_tep_context(config)
    mask = _read(stage["final_mask"])["criticality"]
    output = Path(stage["output_dir"]); path = output / "manifest.json"; records = _manifest(path)
    statistics = fit_spectral_statistics(clean["train"], float(config["spectral_diffusion"]["clip_quantile"]), "train")
    schedule = DiffusionSchedule.cosine(int(config["spectral_diffusion"]["diffusion_steps"]), device)
    augmenter = FrequencyForwardDiffusion(
        statistics, schedule.alpha_bars, np.asarray(mask["soft_mask"], np.float32), 3, 1,
        bool(config["spectral_diffusion"]["preserve_phase"]),
        bool(config["spectral_diffusion"]["preserve_dc"]), device)
    for seed in seeds:
        final_views: dict[str, np.ndarray] = {}; diffusion_audits: dict[str, Any] = {}
        for split, offset in (("train", 0), ("validation", 100)):
            sampling_seed = seed + int(config["spectral_diffusion"]["sampling_seed_offset"]) + offset
            final_views[split], diffusion_audits[split] = augmenter.augment(
                clean[split], "selective", sampling_seed,
                int(config["spectral_diffusion"]["t_noncritical"]), int(base["training"]["batch_size"]))
        runtime = _runtime(base, seed)
        runtime["diagnosis"] = {"threshold_band_width": .05, "high_correlation_quantile": .90}
        pretrain_orders = epoch_orders(len(clean["train"]), int(runtime["epochs"]), seed + 10000)
        probe_orders = epoch_orders(len(clean["train"]), int(runtime["probe_epochs"]), seed + 20000)
        seed_everything(seed)
        template = build_model(runtime["model"], clean["train"].shape[1], 2)
        initial = copy.deepcopy(template.state_dict())
        fairness = {
            "manifest_sha256": file_sha256(base["fixed_views"]["manifest"]),
            "initialization_sha256": _state_hash(initial),
            "pretrain_order_sha256": hashlib.sha256("\n".join(','.join(map(str, row)) for row in pretrain_orders).encode()).hexdigest(),
            "probe_order_sha256": hashlib.sha256("\n".join(','.join(map(str, row)) for row in probe_orders).encode()).hexdigest(),
        }
        for probability in probabilities:
            key = f"{variant_name(probability)}|{seed}"
            if key in records:
                continue
            augmented: dict[str, np.ndarray] = {}; audits: dict[str, Any] = {}
            for split, offset in (("train", 0), ("validation", 100)):
                augmented[split], route = stochastic_view_route(
                    clean[split], final_views[split], views[split]["window_id"], probability,
                    seed + 71000 + offset, seed + offset, float(config["scaling_std"]))
                mechanism = augmentation_mechanism_metrics(
                    clean[split], augmented[split], views[split]["labels"], stages[split],
                    np.asarray(mask["hard_mask"], bool), diffusion_audits[split])
                mechanism["stochastic_view_routing"] = route
                audits[split] = mechanism
            checkpoint = output / variant_name(probability) / f"seed_{seed}" / "model.pt"
            metadata = {**fairness, "p": probability, "evaluation_splits": ["validation"],
                        "test_metrics_read": False,
                        "route_fairness_sha256": {split: audits[split]["stochastic_view_routing"]["fairness_sha256"]
                                                   for split in audits}}
            method = _fit_method(
                variant_name(probability), augmented, audits, views, clean, stages, initial,
                pretrain_orders, probe_orders, runtime, device, checkpoint, metadata,
                evaluation_splits=("validation",))
            record = {
                "variant": variant_name(probability), "p": probability, "seed": seed,
                "evaluation_split": "validation", "method": method, "fairness": fairness,
                "routing_audit": audits, "mask_sha256": mask["mask_sha256"],
                "test_metrics_read": False, "training": "new_training",
            }
            _store(path, records, key, record)
    return records


def _records(config: dict[str, Any], dataset: str) -> dict[str, Any]:
    key = "three_w" if dataset == "3W" else "tep"
    return _manifest(Path(config[key]["output_dir"]) / "manifest.json")


def select_top2(config: dict[str, Any], datasets: tuple[str, ...]) -> dict[str, Any]:
    path = Path(config["output"]["selection"])
    payload = _read(path) if path.exists() else {"top2": {}, "stage_a_metrics": {}}
    for dataset in datasets:
        key = "three_w" if dataset == "3W" else "tep"
        seed = int(config[key]["stage_a_seed"]); records = _records(config, dataset)
        metrics = {p: validation_metrics(dataset, records[f"{variant_name(p)}|{seed}"])
                   for p in map(float, config["candidates"])}
        middle = [.25, .5, .75]
        payload["top2"][dataset] = sorted(
            middle, key=lambda p: (metrics[p]["macro_f1"], metrics[p]["auprc"], -metrics[p]["far"], -p),
            reverse=True)[:2]
        payload["stage_a_metrics"][dataset] = metrics
    payload.update({"selection_split": "validation", "test_read": False})
    write_json(path, payload)
    return payload


def _candidate_gate(config: dict[str, Any], dataset: str, p: float, seeds: list[int]) -> dict[str, Any]:
    records = _records(config, dataset)
    metrics = {candidate: [validation_metrics(dataset, records[f"{variant_name(candidate)}|{seed}"])
                           for seed in seeds] for candidate in (0.0, 1.0, p)}
    endpoint = max((0.0, 1.0), key=lambda value: (mean(row["macro_f1"] for row in metrics[value]), -value))
    deltas = [{key: current[key] - reference[key] for key in current}
              for current, reference in zip(metrics[p], metrics[endpoint])]
    gate = config["gate"]
    checks = {
        "macro_f1_gain": mean(row["macro_f1"] for row in deltas) >= float(gate["stage_b_macro_f1_gain"]),
        "positive_seeds": sum(row["macro_f1"] > 0 for row in deltas) >= int(gate["stage_b_positive_seeds"]),
        "auprc_noninferior": mean(row["auprc"] for row in deltas) >= float(gate["stage_b_auprc_min_delta"]),
        "far_safe": mean(row["far"] for row in deltas) <= float(gate["stage_b_far_max_delta"]),
        "no_catastrophic_seed": all(row["macro_f1"] > float(gate["catastrophic_macro_f1_delta"]) for row in deltas),
    }
    return {"p": p, "endpoint_best": endpoint, "deltas": deltas, "checks": checks,
            "passed": all(checks.values())}


def stage_b_decision(config: dict[str, Any], datasets: tuple[str, ...]) -> dict[str, Any]:
    selection = _read(config["output"]["selection"]); result: dict[str, Any] = {}
    for dataset in datasets:
        key = "three_w" if dataset == "3W" else "tep"; seeds = list(map(int, config[key]["stage_b_seeds"]))
        candidates = [_candidate_gate(config, dataset, float(p), seeds) for p in selection["top2"][dataset]]
        passing = [row for row in candidates if row["passed"]]
        if passing:
            best = max(passing, key=lambda row: mean(delta["macro_f1"] for delta in row["deltas"]))
            selected = float(best["p"])
        else:
            selected = None
        result[dataset] = {"seeds": seeds, "candidates": candidates, "selected_p": selected,
                           "passed": selected is not None}
    payload = {**selection, "stage_b": result,
               "stage_b_status": "GO_STAGE_B" if len(result) == 2 and all(row["passed"] for row in result.values())
                                  else "NO_GO_SVR"}
    write_json(Path(config["output"]["selection"]), payload)
    return payload


def stage_c_decision(config: dict[str, Any], datasets: tuple[str, ...]) -> dict[str, Any]:
    selection = _read(config["output"]["selection"]); result: dict[str, Any] = {}
    for dataset in datasets:
        key = "three_w" if dataset == "3W" else "tep"; seeds = list(map(int, config[key]["stage_c_seeds"]))
        p = float(selection["stage_b"][dataset]["selected_p"])
        base_gate = _candidate_gate(config, dataset, p, seeds)
        deltas = base_gate["deltas"]; gate = config["gate"]
        checks = {
            "macro_f1_gain": mean(row["macro_f1"] for row in deltas) >= float(gate["stage_c_macro_f1_gain"]),
            "nonworse_seeds": sum(row["macro_f1"] >= -float(gate["nonworse_tolerance"]) for row in deltas) >= int(gate["stage_c_nonworse_seeds"]),
            "positive_seeds": sum(row["macro_f1"] > 0 for row in deltas) >= int(gate["stage_c_positive_seeds"]),
            "auprc_noninferior": mean(row["auprc"] for row in deltas) >= float(gate["stage_b_auprc_min_delta"]),
            "far_safe": mean(row["far"] for row in deltas) <= float(gate["stage_b_far_max_delta"]),
            "no_catastrophic_seed": all(row["macro_f1"] > float(gate["catastrophic_macro_f1_delta"]) for row in deltas),
        }
        result[dataset] = {**base_gate, "checks": checks, "passed": all(checks.values()), "seeds": seeds}
    status = "GO_STOCHASTIC_VIEW_ROUTING" if len(result) == 2 and all(row["passed"] for row in result.values()) else "NO_GO_SVR"
    payload = {**selection, "stage_c": result, "status": status, "test_read": False}
    write_json(Path(config["output"]["selection"]), payload)
    if status == "GO_STOCHASTIC_VIEW_ROUTING":
        frozen = {"mode": "QDIFFCL_SVR_FINAL", "frozen": True,
                  "domain_p": {dataset: result[dataset]["p"] for dataset in ("3W", "TEP")},
                  "selection_split": "validation", "test_metrics_used_for_selection": False,
                  "seed_rule": "sha256(SVR|router_seed|sample_id)",
                  "source_config": "configs/stochastic_view_routing.yaml"}
        Path("configs/qdiffcl_svr_final.yaml").write_text(yaml.safe_dump(frozen, sort_keys=False), encoding="utf-8")
    return payload


def run(config: dict[str, Any], data_root: Path, stage_name: str, dataset: str) -> dict[str, Any]:
    _validate(config)
    base_tep = yaml.safe_load(Path(config["tep"]["base_config"]).read_text(encoding="utf-8"))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", str(base_tep.get("cublas_workspace_config", ":4096:8")))
    device = select_device(str(config["device"])); names = ("3w", "tep") if dataset == "both" else (dataset,)
    datasets = tuple("3W" if name == "3w" else "TEP" for name in names)
    if stage_name in ("A", "all"):
        probabilities = list(map(float, config["candidates"]))
        if "3w" in names: run_three_w(config, data_root, probabilities, [int(config["three_w"]["stage_a_seed"])], device)
        if "tep" in names: run_tep(config, probabilities, [int(config["tep"]["stage_a_seed"])], device)
        selection = select_top2(config, datasets)
    if stage_name in ("B", "all"):
        selection = _read(config["output"]["selection"])
        for name, label, key in (("3w", "3W", "three_w"), ("tep", "TEP", "tep")):
            if name not in names: continue
            probabilities = sorted({0., 1., *map(float, selection["top2"][label])})
            seeds = list(map(int, config[key]["stage_b_seeds"]))
            if name == "3w": run_three_w(config, data_root, probabilities, seeds, device)
            else: run_tep(config, probabilities, seeds, device)
        selection = stage_b_decision(config, datasets)
        if selection["stage_b_status"] != "GO_STAGE_B":
            write_json(Path(config["output"]["manifest"]), {"stage": "B", "status": "NO_GO_SVR",
                       "validation_only": True, "test_read": False})
            return {"stage": "B", "status": "NO_GO_SVR", "stage_c_run": False, "test_read": False}
    if stage_name in ("C", "all"):
        selection = _read(config["output"]["selection"])
        if selection.get("stage_b_status") != "GO_STAGE_B":
            raise RuntimeError("Stage C is forbidden before Stage B passes")
        for name, label, key in (("3w", "3W", "three_w"), ("tep", "TEP", "tep")):
            if name not in names: continue
            probabilities = [0., float(selection["stage_b"][label]["selected_p"]), 1.]
            seeds = list(map(int, config[key]["stage_c_seeds"]))
            if name == "3w": run_three_w(config, data_root, probabilities, seeds, device)
            else: run_tep(config, probabilities, seeds, device)
        selection = stage_c_decision(config, datasets)
    status = selection.get("status", selection.get("stage_b_status", "STAGE_A_COMPLETE"))
    write_json(Path(config["output"]["manifest"]), {"stage": stage_name, "status": status,
               "validation_only": True, "test_read": False})
    return {"stage": stage_name, "status": status, "test_read": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stochastic_view_routing.yaml")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("A", "B", "C", "all"), default="all")
    parser.add_argument("--dataset", choices=("3w", "tep", "both"), default="both")
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    print(json.dumps(run(config, args.data_root, args.stage, args.dataset), ensure_ascii=False))


if __name__ == "__main__":
    main()
