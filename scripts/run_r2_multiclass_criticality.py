from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from frequency import fault_stages, mask_jaccard
from scripts.audit_semantic_diffusion_augmentation import bases
from scripts.diagnose_frequency_selective_far import correlation_drift
from scripts.run_3w_diffusion_1seed import R2_METHOD, run as run_three_w
from scripts.run_diffusion_quality_retest import _state_hash, epoch_orders, load_fixed_views
from scripts.run_frequency_selective_r1_3seed import (_fit_method, array_sha256,
                                                       file_sha256, sha256_strings,
                                                       validate_frozen_config)
from scripts.run_stage_frequency_diffusion_mvp import (_build_frequency_components,
                                                       _configure, _runtime,
                                                       augmentation_mechanism_metrics)
from trainers import build_model
from utils import seed_everything, write_json


R2_WEIGHTS = {"weight_discriminative": .40, "weight_early": .24,
              "weight_run_stability": .16, "weight_multiclass": .20}


def validate_r2_weights(weights: dict) -> None:
    actual = {key: float(weights[key]) for key in R2_WEIGHTS}
    if actual != R2_WEIGHTS or abs(sum(actual.values()) - 1.0) > 1e-12:
        raise ValueError("R2 weights must remain frozen at D/E/S/M=0.40/0.24/0.16/0.20")


def run_three_w_stage(config: dict, data_root: Path, selected_seeds: list[int] | None = None) -> dict:
    stage = config["three_w"]; validate_r2_weights(stage["criticality_weights"])
    if list(map(int, stage["seeds"])) != [42, 43, 44] or stage["method"] != R2_METHOD:
        raise ValueError("3W R2 stage must freeze seeds 42/43/44 and the R2 method")
    base = yaml.safe_load(Path(stage["base_config"]).read_text(encoding="utf-8"))
    output = Path(stage["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "result_manifest.json"
    completed = json.loads(manifest_path.read_text(encoding="utf-8")).get("seed_results", {}) if manifest_path.exists() else {}
    seeds = list(map(int, selected_seeds if selected_seeds is not None else stage["seeds"]))
    for seed in seeds:
        current = copy.deepcopy(base); current["seed"] = seed; current["protocol_seed"] = int(stage["protocol_seed"])
        current.pop("criticality_source", None); current["methods"] = [R2_METHOD]
        current["training"]["supcon_batching"] = "original"
        current["criticality"].update({key: float(value) for key, value in stage["criticality_weights"].items()})
        current["output_dir"] = str(output / f"seed_{seed}")
        result = run_three_w(current, data_root); path = Path(current["output_dir"]) / "result.json"
        completed[str(seed)] = {"result_path": str(path), "methods": list(result["methods"]), "status": "complete"}
    manifest = {"stage": "3W", "seeds": sorted(map(int, completed)), "protocol_seed": int(stage["protocol_seed"]),
                "criticality_fit_scope": "train labels and train fault types only", "weights": R2_WEIGHTS,
                "multiclass_classes": [0, 2, 8, 9],
                "seed_results": completed}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def run_tep_stage(config: dict, data_root: Path, selected_seeds: list[int] | None = None) -> dict:
    stage = config["tep"]; validate_r2_weights(stage["criticality_weights"])
    stage_a = json.loads(Path(config["docs"]["three_w_json"]).read_text(encoding="utf-8"))
    if stage_a["status"] not in {"R2_3W_GO", "R2_3W_PARTIAL_GO"} or not stage_a["stage_b_allowed"]:
        raise RuntimeError("3W Stage A did not enable TEP Stage B")
    if not data_root.is_dir(): raise FileNotFoundError(f"TEP data root does not exist: {data_root}")
    r1_config = yaml.safe_load(Path(stage["base_config"]).read_text(encoding="utf-8"))
    base_config = yaml.safe_load(Path(r1_config["base_config"]).read_text(encoding="utf-8"))
    far_fix = yaml.safe_load(Path(r1_config["far_fix_config"]).read_text(encoding="utf-8"))
    validate_frozen_config(r1_config, base_config, far_fix); _configure(base_config)
    views, _ = load_fixed_views(base_config); clean = bases(views)
    stages = {split: fault_stages(views[split], base_config) for split in views}
    r2_config = copy.deepcopy(base_config); r2_config["criticality"].update(stage["criticality_weights"])
    r1_critical, _ = _build_frequency_components(base_config, views, clean, stages, str(r1_config["device"]))
    r2_critical, augmenter = _build_frequency_components(r2_config, views, clean, stages, str(r1_config["device"]))
    r1_mask, r2_mask = r1_critical["masks"]["composite"], r2_critical["masks"]["composite"]
    mask_audit = {"r1_mask_sha256": array_sha256(r1_mask), "r2_mask_sha256": array_sha256(r2_mask),
                  "jaccard": mask_jaccard(r1_mask, r2_mask), "changed_bins": int(np.logical_xor(r1_mask, r2_mask).sum()),
                  "selected_bins": int(r2_mask.sum()), "r2_component_weights": r2_critical["component_weights"],
                  "multiclass_type_run_counts": r2_critical["multiclass_type_run_counts"],
                  "fit_split": r2_critical["fit_split"]}
    output = Path(stage["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    final_path = output / "result.json"
    previous = json.loads(Path(stage["previous_result"]).read_text(encoding="utf-8")); results = {}
    if final_path.exists(): results = json.loads(final_path.read_text(encoding="utf-8")).get("seed_results", {})
    seeds = list(map(int, selected_seeds if selected_seeds is not None else stage["seeds"]))
    for seed in seeds:
        runtime = _runtime(base_config, seed); runtime["diagnosis"] = r1_config["diagnosis"]
        pretrain_orders = epoch_orders(len(clean["train"]), int(runtime["epochs"]), seed + 10_000)
        probe_orders = epoch_orders(len(clean["train"]), int(runtime["probe_epochs"]), seed + 20_000)
        seed_everything(seed); template = build_model(runtime["model"], clean["train"].shape[1], 2)
        initial_state = copy.deepcopy(template.state_dict())
        hashes = {"manifest_sha256": file_sha256(r1_config["fixed_views_manifest"]),
                  "mask_sha256": array_sha256(r2_mask), "initialization_sha256": _state_hash(initial_state),
                  "pretrain_order_sha256": sha256_strings([','.join(map(str, order)) for order in pretrain_orders]),
                  "probe_order_sha256": sha256_strings([','.join(map(str, order)) for order in probe_orders]),
                  "r2_weights_sha256": hashlib.sha256(json.dumps(R2_WEIGHTS, sort_keys=True).encode()).hexdigest()}
        old_hashes = previous["seed_results"][str(seed)]["fairness"]
        for key in ("manifest_sha256", "initialization_sha256", "pretrain_order_sha256", "probe_order_sha256"):
            if hashes[key] != old_hashes[key]: raise RuntimeError(f"TEP R2 changed frozen fairness hash: {key}")
        augmented = {}; audits = {}
        for split, offset in (("train", 0), ("validation", 100), ("test", 200)):
            sampling_seed = seed + int(base_config["spectral_diffusion"]["sampling_seed_offset"]) + offset
            augmented[split], diagnostic = augmenter.augment(
                clean[split], "selective", sampling_seed, 5, int(runtime["batch_size"]), noise_structure="iid")
            audit = augmentation_mechanism_metrics(clean[split], augmented[split], views[split]["labels"], stages[split], r2_mask, diagnostic)
            audit["correlation_drift"] = correlation_drift(clean[split], augmented[split], views[split]["labels"],
                                                             float(r1_config["diagnosis"]["high_correlation_quantile"]))
            audits[split] = audit
        old_budget = previous["seed_results"][str(seed)]["methods"]["C1"]["augmentation_audit"]["train"]["expected_total_noise_budget"]
        if abs(float(audits["train"]["expected_total_noise_budget"]) - float(old_budget)) > 1e-6:
            raise RuntimeError("TEP C1/R2 total spectral noise budgets differ")
        checkpoint = output / f"seed_{seed}" / "R2" / "model.pt"
        metadata = {**hashes, "method": "R2", "seed": seed, "augmentation": "selective_iid_t5_r2_multiclass",
                    "criticality_fit_scope": "train-only faultNumber including normal type 0"}
        method = _fit_method("R2", augmented, audits, views, clean, stages, initial_state, pretrain_orders,
                             probe_orders, runtime, str(r1_config["device"]), checkpoint, metadata)
        record = {"seed": seed, "methods": {"R2": method}, "fairness": hashes, "same_fixed_views": True,
                  "same_initialization": True, "same_pretrain_order": True, "same_probe_order": True,
                  "train_only_fault_type_m": True, "c1_r2_same_noise_seed": True}
        write_json(output / f"seed_{seed}" / "result.json", record); results[str(seed)] = record
    payload = {"stage": "TEP", "markers": stage["markers"], "seeds": sorted(map(int, results)),
               "criticality_fit_scope": "train-only faultNumber/fault type including normal type 0",
               "weights": R2_WEIGHTS, "mask_audit": mask_audit, "seed_results": results,
               "test_used_for_tuning_or_selection": False}
    write_json(final_path, payload); return payload


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/r2_multiclass_criticality.yaml")
    parser.add_argument("--stage", choices=("3w", "tep"), default="3w")
    parser.add_argument("--data-root", type=Path, required=True); parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.stage == "3w": result = run_three_w_stage(config, args.data_root, args.seeds)
    else: result = run_tep_stage(config, args.data_root, args.seeds)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__": main()
