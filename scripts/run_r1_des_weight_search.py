from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
import torch
import yaml

from diffusion import DiffusionSchedule, FrequencyForwardDiffusion, fit_spectral_statistics
from scripts.run_3w_diffusion_1seed import METHODS as THREE_W_METHODS, run as run_three_w_model
from scripts.run_diffusion_quality_retest import _state_hash, epoch_orders
from scripts.run_frequency_selective_r1_3seed import (
    _fit_method as fit_tep_method, file_sha256, sha256_strings,
)
from scripts.run_r1_des_ablation import build_masks
from scripts.run_stage_frequency_diffusion_mvp import _runtime, augmentation_mechanism_metrics
from trainers import build_model
from utils import seed_everything, select_device, write_json


DATASETS = ("3W", "TEP")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest(path: Path) -> dict[str, Any]:
    return _read(path).get("results", {}) if path.exists() else {}


def _weights(config: dict[str, Any], name: str) -> tuple[float, float, float]:
    item = config["variants"][name]
    value = tuple(float(item[key]) for key in (
        "weight_discriminative", "weight_early", "weight_run_stability"
    ))
    if any(weight < 0 for weight in value) or not np.isclose(sum(value), 1.0):
        raise ValueError(f"invalid weights for {name}: {value}")
    return value


def _validate_config(config: dict[str, Any]) -> None:
    if len(config["variants"]) > 15 or "CURRENT" not in config["variants"]:
        raise ValueError("weight search must contain CURRENT and at most 15 candidates")
    for name in config["variants"]:
        _weights(config, name)
    frozen = config["spectral_diffusion"]
    expected = {"t_uniform": 3, "t_critical": 1, "t_noncritical": 5,
                "preserve_phase": True, "preserve_dc": True, "noise_structure": "iid"}
    if any(frozen[key] != value for key, value in expected.items()):
        raise ValueError("frozen diffusion protocol changed")
    if float(config["criticality_base"]["critical_ratio"]) != 0.30:
        raise ValueError("critical_ratio changed")


def _verify_current_mask(audit: dict[str, Any]) -> None:
    historical = _read(Path("docs/r1_des_mask_audit.json"))
    for dataset in DATASETS:
        if audit[dataset]["CURRENT"]["mask_sha256"] != historical[dataset]["FULL_DES"]["mask_sha256"]:
            raise RuntimeError(f"CURRENT no longer reproduces historical R1 mask on {dataset}")


def run_three_w(config: dict[str, Any], masks: dict[str, Any], variants: list[str], seeds: list[int]) -> dict[str, Any]:
    stage = config["three_w"]
    base = yaml.safe_load(Path(stage["base_config"]).read_text(encoding="utf-8"))
    manifest_path = Path(stage["output_dir"]) / "manifest.json"
    completed = _manifest(manifest_path)
    for variant in variants:
        source = Path(config["output"]["mask_dir"]) / f"3w_{variant}.json"
        for seed in map(int, seeds):
            key = f"{variant}|{seed}"
            if key in completed and Path(completed[key]["result_path"]).exists():
                continue
            current = copy.deepcopy(base)
            current.update({
                "seed": seed,
                "protocol_seed": int(stage["protocol_seed"]),
                "criticality_source": str(source),
                "methods": [THREE_W_METHODS[2]],
                "evaluation_split": "validation",
                "output_dir": str(Path(stage["output_dir"]) / variant / f"seed_{seed}"),
            })
            current["training"]["supcon_batching"] = "original"
            result = run_three_w_model(current, Path(args_data_root(config)))
            if result.get("evaluation_split") != "validation":
                raise RuntimeError("3W weight search evaluated a non-validation split")
            method = result["methods"][THREE_W_METHODS[2]]
            completed[key] = {
                "variant": variant, "seed": seed, "evaluation_split": "validation",
                "result_path": str(Path(current["output_dir"]) / "result.json"),
                "method": method, "fairness": result["fairness"],
                "mask_sha256": masks[variant]["mask_sha256"], "test_metrics_read": False,
            }
            write_json(manifest_path, {"results": completed})
    return completed


def args_data_root(config: dict[str, Any]) -> str:
    value = config.get("runtime_data_root")
    if not value:
        raise RuntimeError("runtime_data_root was not supplied by CLI")
    return str(value)


def run_tep(config: dict[str, Any], context: tuple[Any, ...], masks: dict[str, Any],
            variants: list[str], seeds: list[int]) -> dict[str, Any]:
    stage = config["tep"]
    base_config, views, clean, stages, _ = context
    device = select_device(str(config["device"]))
    output = Path(stage["output_dir"])
    manifest_path = output / "manifest.json"
    completed = _manifest(manifest_path)
    statistics = fit_spectral_statistics(clean["train"], float(stage["spectral_diffusion"]["clip_quantile"]), "train")
    schedule = DiffusionSchedule.cosine(int(config["spectral_diffusion"]["diffusion_steps"]), device)
    tolerance = float(config["selection"]["budget_tolerance"])
    for variant in variants:
        mask = np.asarray(masks[variant]["soft_mask"], np.float32)
        augmenter = FrequencyForwardDiffusion(statistics, schedule.alpha_bars, mask, 3, 1, True, True, device)
        for seed in map(int, seeds):
            key = f"{variant}|{seed}"
            if key in completed:
                continue
            augmented, audits = {}, {}
            for split, offset in (("train", 0), ("validation", 100)):
                sampling_seed = seed + int(stage["spectral_diffusion"]["sampling_seed_offset"]) + offset
                augmented[split], diagnostics = augmenter.augment(
                    clean[split], "selective", sampling_seed, 5,
                    int(base_config["training"]["batch_size"]), noise_structure="iid"
                )
                audits[split] = augmentation_mechanism_metrics(
                    clean[split], augmented[split], views[split]["labels"], stages[split],
                    np.asarray(masks[variant]["hard_mask"], bool), diagnostics,
                )
            if float(config["mask_audit"]["TEP"][variant]["total_budget_error"]) > tolerance:
                raise RuntimeError(f"TEP matched-budget failure: {variant}")
            runtime = _runtime(base_config, seed)
            runtime["diagnosis"] = {"threshold_band_width": .05, "high_correlation_quantile": .90}
            pretrain_orders = epoch_orders(len(clean["train"]), int(runtime["epochs"]), seed + 10000)
            probe_orders = epoch_orders(len(clean["train"]), int(runtime["probe_epochs"]), seed + 20000)
            seed_everything(seed)
            template = build_model(runtime["model"], clean["train"].shape[1], 2)
            initial_state = copy.deepcopy(template.state_dict())
            fairness = {
                "manifest_sha256": file_sha256(base_config["fixed_views"]["manifest"]),
                "initialization_sha256": _state_hash(initial_state),
                "pretrain_order_sha256": sha256_strings([','.join(map(str, order)) for order in pretrain_orders]),
                "probe_order_sha256": sha256_strings([','.join(map(str, order)) for order in probe_orders]),
            }
            metadata = {
                **fairness, "method": variant, "seed": seed,
                "mask_sha256": masks[variant]["mask_sha256"],
                "evaluation_splits": ["validation"], "test_metrics_read": False,
            }
            checkpoint = output / variant / f"seed_{seed}" / "model.pt"
            record = fit_tep_method(
                variant, augmented, audits, views, clean, stages, initial_state,
                pretrain_orders, probe_orders, runtime, device, checkpoint, metadata,
                evaluation_splits=("validation",),
            )
            if "test" in record or record.get("evaluation_splits") != ["validation"]:
                raise RuntimeError("TEP weight search evaluated test")
            completed[key] = {
                "variant": variant, "seed": seed, "evaluation_split": "validation",
                "method": record, "fairness": fairness,
                "mask_sha256": masks[variant]["mask_sha256"], "test_metrics_read": False,
            }
            write_json(manifest_path, {"results": completed})
    return completed


def validation_metrics(dataset: str, record: dict[str, Any]) -> dict[str, float]:
    if record.get("evaluation_split") != "validation" or record.get("test_metrics_read") is not False:
        raise RuntimeError(f"non-validation record in selection: {dataset}")
    if dataset == "3W":
        metrics = record["method"]["metrics"]
        return {"macro_f1": float(metrics["macro_f1"]), "far": float(metrics["far"]),
                "early_recall": float(metrics["early_recall"]),
                "auprc": float(metrics["auprc_multiclass_macro"])}
    method = record["method"]
    if "test" in method:
        raise RuntimeError("test metrics present in TEP selection record")
    validation = method["validation"]
    return {"macro_f1": float(validation["metrics"]["macro_f1"]),
            "far": float(validation["metrics"]["far"]),
            "early_recall": float(validation["early_fault"]["recall"]),
            "auprc": float(validation["metrics"]["auprc"])}


def select_top_three(config: dict[str, Any]) -> dict[str, Any]:
    manifests = {dataset: _manifest(Path(config[dataset.lower().replace("3w", "three_w")]["output_dir"]) / "manifest.json")
                 for dataset in DATASETS}
    stage1_seed = {"3W": int(config["three_w"]["stage1_seed"]), "TEP": int(config["tep"]["stage1_seed"])}
    values: dict[str, dict[str, dict[str, float]]] = {dataset: {} for dataset in DATASETS}
    for dataset in DATASETS:
        for variant in config["variants"]:
            key = f"{variant}|{stage1_seed[dataset]}"
            if key not in manifests[dataset]:
                raise RuntimeError(f"missing Stage 1 record: {dataset} {key}")
            values[dataset][variant] = validation_metrics(dataset, manifests[dataset][key])
    current = {dataset: values[dataset]["CURRENT"] for dataset in DATASETS}
    ranking = []
    for variant in config["variants"]:
        if variant == "CURRENT":
            continue
        far_pass = all(values[dataset][variant]["far"] <= current[dataset]["far"] +
                       float(config["selection"]["far_tolerance"][dataset]) for dataset in DATASETS)
        macro_relative_gain = mean((values[dataset][variant]["macro_f1"] - current[dataset]["macro_f1"])
                                   / max(current[dataset]["macro_f1"], 1e-12) for dataset in DATASETS)
        far_delta = mean(values[dataset][variant]["far"] - current[dataset]["far"] for dataset in DATASETS)
        early_delta = mean(values[dataset][variant]["early_recall"] - current[dataset]["early_recall"] for dataset in DATASETS)
        auprc_delta = mean(values[dataset][variant]["auprc"] - current[dataset]["auprc"] for dataset in DATASETS)
        ranking.append({"variant": variant, "far_gate_pass": far_pass,
                        "mean_relative_macro_gain": macro_relative_gain, "mean_far_delta": far_delta,
                        "mean_early_delta": early_delta, "mean_auprc_delta": auprc_delta,
                        "validation": {dataset: values[dataset][variant] for dataset in DATASETS}})
    ranking.sort(key=lambda row: (not row["far_gate_pass"], -row["mean_relative_macro_gain"],
                                  row["mean_far_delta"], -row["mean_early_delta"], -row["mean_auprc_delta"], row["variant"]))
    eligible = [row["variant"] for row in ranking if row["far_gate_pass"]]
    remaining = [row["variant"] for row in ranking if not row["far_gate_pass"]]
    top = (eligible + remaining)[:int(config["selection"]["top_k"])]
    payload = {"selection_split": "validation", "test_metrics_read": False,
               "stage1_candidates": len(config["variants"]), "top3": top,
               "ranking": ranking, "current_validation": current}
    write_json(Path(config["output"]["selection"]), payload)
    return payload


def _sample_std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def freeze_final(config: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    manifests = {"3W": _manifest(Path(config["three_w"]["output_dir"]) / "manifest.json"),
                 "TEP": _manifest(Path(config["tep"]["output_dir"]) / "manifest.json")}
    variants = ["CURRENT", *selection["top3"]]
    summaries: dict[str, Any] = {}
    for variant in variants:
        summaries[variant] = {}
        for dataset, config_key in (("3W", "three_w"), ("TEP", "tep")):
            per_seed = []
            for seed in map(int, config[config_key]["seeds"]):
                key = f"{variant}|{seed}"
                if key not in manifests[dataset]:
                    raise RuntimeError(f"missing Stage 2 record: {dataset} {key}")
                per_seed.append({"seed": seed, **validation_metrics(dataset, manifests[dataset][key])})
            summaries[variant][dataset] = {
                "per_seed": per_seed,
                **{f"{metric}_mean": mean(row[metric] for row in per_seed) for metric in ("macro_f1", "far", "early_recall", "auprc")},
                **{f"{metric}_std": _sample_std([row[metric] for row in per_seed]) for metric in ("macro_f1", "far", "early_recall", "auprc")},
            }
    current = summaries["CURRENT"]
    decisions = []
    for variant in selection["top3"]:
        relative_gain = mean((summaries[variant][dataset]["macro_f1_mean"] - current[dataset]["macro_f1_mean"])
                             / max(current[dataset]["macro_f1_mean"], 1e-12) for dataset in DATASETS)
        macro_floor = all(summaries[variant][dataset]["macro_f1_mean"] >= current[dataset]["macro_f1_mean"] -
                          float(config["selection"]["final_max_dataset_macro_drop"]) for dataset in DATASETS)
        far_pass = all(summaries[variant][dataset]["far_mean"] <= current[dataset]["far_mean"] +
                       float(config["selection"]["far_tolerance"][dataset]) for dataset in DATASETS)
        consistency = all(summaries[variant][dataset]["macro_f1_std"] <= current[dataset]["macro_f1_std"] *
                          float(config["selection"]["final_max_std_ratio"]) + 1e-12 for dataset in DATASETS)
        clear_gain = relative_gain >= float(config["selection"]["final_min_mean_relative_macro_gain"])
        decisions.append({"variant": variant, "mean_relative_macro_gain": relative_gain,
                          "macro_floor_pass": macro_floor, "far_gate_pass": far_pass,
                          "consistency_pass": consistency, "clear_gain_pass": clear_gain,
                          "eligible": macro_floor and far_pass and consistency and clear_gain})
    eligible = [row for row in decisions if row["eligible"]]
    eligible.sort(key=lambda row: -row["mean_relative_macro_gain"])
    final_variant = eligible[0]["variant"] if eligible else "CURRENT"
    reason = "validation gates passed" if eligible else "no candidate had a clear stable validation advantage"
    selection.update({"stage2": summaries, "final_decisions": decisions,
                      "final_variant": final_variant, "final_weights": _weights(config, final_variant),
                      "final_reason": reason, "weights_frozen": True,
                      "future_test_driven_weight_changes_forbidden": True})
    write_json(Path(config["output"]["selection"]), selection)
    return selection


def _write_global_manifest(config: dict[str, Any], stage: str, selection: dict[str, Any] | None) -> None:
    three = _manifest(Path(config["three_w"]["output_dir"]) / "manifest.json")
    tep = _manifest(Path(config["tep"]["output_dir"]) / "manifest.json")
    write_json(Path(config["output"]["manifest"]), {
        "stage": stage, "evaluation_split": "validation", "test_metrics_read": False,
        "records": {"3W": len(three), "TEP": len(tep)},
        "selection": selection, "weights_only_search": True,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/r1_des_weight_search.yaml")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("stage1", "select", "stage2", "all"), default="all")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    config["runtime_data_root"] = str(args.data_root)
    _validate_config(config)
    three_masks, tep_context, audit = build_masks(config, args.data_root)
    config["mask_audit"] = audit
    _verify_current_mask(audit)
    variants = list(config["variants"])
    selection = None
    if args.stage in ("stage1", "all"):
        run_three_w(config, three_masks, variants, [int(config["three_w"]["stage1_seed"])])
        run_tep(config, tep_context, tep_context[-1], variants, [int(config["tep"]["stage1_seed"])])
    if args.stage in ("select", "all"):
        selection = select_top_three(config)
    if args.stage in ("stage2", "all"):
        selection = selection or _read(Path(config["output"]["selection"]))
        selected = ["CURRENT", *selection["top3"]]
        run_three_w(config, three_masks, selected, list(map(int, config["three_w"]["seeds"])))
        run_tep(config, tep_context, tep_context[-1], selected, list(map(int, config["tep"]["seeds"])))
        selection = freeze_final(config, selection)
    _write_global_manifest(config, args.stage, selection)
    print(json.dumps({"stage": args.stage, "selection": selection and selection.get("top3"),
                      "final": selection and selection.get("final_variant")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
