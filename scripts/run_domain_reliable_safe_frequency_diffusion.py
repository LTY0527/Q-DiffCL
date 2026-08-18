from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import yaml

import scripts.run_3w_clean_baseline as base3w
from datasets.three_w import discover_instances
from diffusion import (DiffusionSchedule, FrequencyForwardDiffusion,
                       constrained_safe_variance, fit_spectral_statistics)
from frequency import (build_tep_stratified_run_bootstrap,
                       build_three_w_leave_one_well_out, fault_stages,
                       fit_frequency_scaler, log_amplitude_phase)
from scripts.audit_semantic_diffusion_augmentation import bases
from scripts.run_3w_final_primary_grouped import FINAL_PRIMARY_CLASSES
from scripts.run_3w_diffusion_1seed import (DRFD_METHOD, run as run_three_w)
from scripts.run_diffusion_quality_retest import _state_hash, epoch_orders
from scripts.run_diffusion_quality_retest import load_fixed_views
from scripts.run_frequency_selective_r1_3seed import (_fit_method as fit_tep_method,
                                                       file_sha256, sha256_strings)
from scripts.run_stage_frequency_diffusion_mvp import (_configure, _runtime,
                                                        augmentation_mechanism_metrics)
from trainers import build_model
from utils import seed_everything, select_device, write_json


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [_jsonable(item) for item in value]
    return value


def _load_three_w_train(stage: dict[str, Any], data_root: Path):
    config = yaml.safe_load(Path(stage["base_config"]).read_text(encoding="utf-8"))
    stability = json.loads(Path(config["stability_result"]).read_text(encoding="utf-8"))
    if stability["status"] != "3W_FINAL_PRIMARY_STABILITY_GO" or not stability["diffusion_allowed"]:
        raise RuntimeError("frozen 3W primary protocol is not GO")
    grouped = Path(config["grouped_output"]); split_index = int(config["canonical_split_index"])
    manifest = json.loads((grouped / "grouped_split_manifest.json").read_text(encoding="utf-8"))
    frozen = manifest["splits"][split_index]["wells"]
    split = {name: set(wells) for name, wells in frozen.items()}
    grouped_result = json.loads((grouped / f"split_{split_index:02d}" / "result.json").read_text(encoding="utf-8"))
    if grouped_result["split"] != frozen: raise RuntimeError("3W canonical split differs from frozen result")
    base_config = yaml.safe_load(Path(config["base_config"]).read_text(encoding="utf-8"))
    base = yaml.safe_load(Path(base_config["base_config"]).read_text(encoding="utf-8"))
    base3w.PRIMARY_CLASSES = FINAL_PRIMARY_CLASSES
    base3w.CLASS_TO_TARGET = {value: index for index, value in enumerate(FINAL_PRIMARY_CLASSES)}
    instances = [item for item in discover_instances(data_root)
                 if item.source == "WELL" and item.event_class in FINAL_PRIMARY_CLASSES]
    by_instance = {item.instance_id: item for item in instances}
    train_instances = [item for item in instances if item.well_id in split["train"]]
    refs = []
    for item in train_instances:
        refs.extend(base3w.instance_refs(item, int(base["protocol"]["window_length"]),
                                         int(base["protocol"]["stride"]),
                                         int(base["protocol"]["transient_offset"])))
    refs = base3w.stratified_refs(refs, int(base_config["train_windows_per_class"]),
                                  int(stage["protocol_seed"]))
    preprocessor = json.loads((grouped / f"split_{split_index:02d}" / "preprocessor.json").read_text(encoding="utf-8"))
    values, labels = base3w.materialize(refs, by_instance, preprocessor,
                                        int(base["protocol"]["window_length"]), False)
    def run_uid(ref):
        item = by_instance[ref.instance_id]
        original = FINAL_PRIMARY_CLASSES[ref.target] if ref.target else 0
        return f"training:fault_{original}:{item.well_id}"
    bundle = {"run_uid": np.asarray([run_uid(ref) for ref in refs]), "labels": labels}
    stages = np.asarray([ref.stage for ref in refs])
    wells = np.asarray([by_instance[ref.instance_id].well_id for ref in refs], dtype=object)
    return values, bundle, stages, wells, frozen


def _build_three_w_reliability(config: dict[str, Any], data_root: Path):
    stage = config["three_w"]
    values, bundle, stages, wells, split = _load_three_w_train(stage, data_root)
    train_log = log_amplitude_phase(values)[0]
    scaler = fit_frequency_scaler(train_log, "train")
    reliability = build_three_w_leave_one_well_out(
        scaler.transform(train_log), bundle, stages, wells, stage["criticality"], train_log)
    return reliability, {"train_windows": len(values), "train_wells": sorted(set(map(str, wells))),
                         "frozen_split": split, "test_used_for_reliability": False}


def _build_tep_reliability(config: dict[str, Any]):
    stage = config["tep"]
    base_config = yaml.safe_load(Path(stage["base_config"]).read_text(encoding="utf-8"))
    _configure(base_config); views, _ = load_fixed_views(base_config); clean = bases(views)
    stages = fault_stages(views["train"], base_config)
    train_log = log_amplitude_phase(clean["train"])[0]
    scaler = fit_frequency_scaler(train_log, "train")
    reliability = build_tep_stratified_run_bootstrap(
        scaler.transform(train_log), views["train"], stages, stage["criticality"], train_log)
    return reliability, {"train_windows": len(clean["train"]),
                         "train_runs": reliability["unit_ids"],
                         "test_used_for_reliability": False}


def _category_stats(reliability: dict[str, Any], allocation: dict[str, Any]) -> dict[str, Any]:
    r1 = np.asarray(allocation["r1_timestep"]); safe = np.asarray(allocation["safe_timestep"])
    masks = {name: np.asarray(reliability[name], dtype=bool) for name in
             ("reliable_critical", "ambiguous", "reliable_noncritical")}
    result = {}
    for name, mask in masks.items():
        result[name] = {"count": int(mask.sum()), "fraction": float(mask.mean()),
                        "mean_r1_timestep": float(r1[mask].mean()) if mask.any() else None,
                        "mean_drfd_timestep": float(safe[mask].mean()) if mask.any() else None}
    changed = np.abs(safe - r1) > 1e-7
    flat_iqr = np.asarray(reliability["rank_iqr"]).reshape(-1)
    top = np.argsort(flat_iqr)[::-1][:20]
    channels, frequencies = r1.shape
    similarity_r1 = float(np.dot(r1.reshape(-1), safe.reshape(-1)) /
                          max(np.linalg.norm(r1) * np.linalg.norm(safe), 1e-12))
    uniform = np.full_like(safe, 3.0)
    similarity_uniform = float(np.dot(uniform.reshape(-1), safe.reshape(-1)) /
                              max(np.linalg.norm(uniform) * np.linalg.norm(safe), 1e-12))
    return {"categories": result, "changed_bins": int(changed.sum()),
            "mean_absolute_timestep_change": float(np.abs(safe - r1).mean()),
            "r1_drfd_timestep_cosine_similarity": similarity_r1,
            "uniform_drfd_timestep_cosine_similarity": similarity_uniform,
            "top_20_unstable_bins": [{"channel": int(index // frequencies),
                                       "frequency_bin": int(index % frequencies),
                                       "rank_iqr": float(flat_iqr[index])} for index in top]}


def _audit_dataset(name: str, reliability: dict[str, Any], stage: dict[str, Any],
                   device: str) -> dict[str, Any]:
    spectral = stage["spectral_diffusion"]
    schedule = DiffusionSchedule.cosine(int(spectral["diffusion_steps"]), device)
    _, allocation = constrained_safe_variance(
        schedule.alpha_bars, reliability["r1"]["soft_mask"], reliability["rank_q25"],
        reliability["rank_q75"], reliability["reliable_noncritical"], reliability["ambiguous"],
        bool(spectral["preserve_dc"]), int(spectral["t_critical"]), int(spectral["t_uniform"]),
        int(spectral["t_noncritical"]))
    invariants = {key: allocation[key] for key in (
        "protected_timestep_not_increased", "protected_variance_not_increased",
        "ambiguous_variance_not_increased", "extra_only_reliable_noncritical",
        "budget_adjustment_only_reliable_noncritical", "maximum_variance_respected", "finite")}
    budget_ok = allocation["budget_error_fraction"] <= float(spectral["maximum_budget_error"])
    structure = _category_stats(reliability, allocation)
    nondegenerate = 0 < structure["changed_bins"] < np.asarray(allocation["r1_timestep"]).size
    passed = all(invariants.values()) and budget_ok and nondegenerate
    return {"dataset": name, "passed": bool(passed), "invariants": invariants,
            "budget_ok": bool(budget_ok), "nondegenerate": bool(nondegenerate),
            "allocation": _jsonable(allocation), "structure": structure,
            "reliability": _jsonable({key: value for key, value in reliability.items()
                                      if key not in {"r1", "composites", "ranks"}}),
            "rank_profiles": _jsonable(reliability["ranks"]),
            "r1": {"soft_mask": _jsonable(reliability["r1"]["soft_mask"]),
                   "composite": _jsonable(reliability["r1"]["composite"])}}


def _write_report(path: Path, result: dict[str, Any]) -> None:
    lines = ["# DRFD Stage A 机制审计", "", f"最终状态：`{result['status']}`。", "",
             "Stage A 仅使用训练域统计，未训练 encoder/probe，也未读取 test 数据参与可靠性拟合。", ""]
    for key in ("three_w", "tep"):
        item = result[key]; allocation = item["allocation"]
        lines.extend([f"## {item['dataset']}", "", f"- Gate：`{'GO' if item['passed'] else 'NO-GO'}`",
                      f"- 可靠 critical / ambiguous / 可靠 non-critical："
                      f"{item['structure']['categories']['reliable_critical']['count']} / "
                      f"{item['structure']['categories']['ambiguous']['count']} / "
                      f"{item['structure']['categories']['reliable_noncritical']['count']}",
                      f"- changed bins：{item['structure']['changed_bins']}",
                      f"- mean |t_DRFD-t_R1|：{item['structure']['mean_absolute_timestep_change']:.6f}",
                      f"- budget error：{allocation['budget_error_fraction']:.6%}",
                      f"- 安全不变量：{item['invariants']}", ""])
    if result["status"] != "DRFD_MECHANISM_GO":
        lines.extend(["## 停止线", "", "Stage A 未通过，因此按预注册协议停止：不执行 Stage B/C，不新增训练 run。", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_stage_a(config: dict[str, Any], data_root: Path) -> dict[str, Any]:
    device = select_device(str(config["device"]))
    three_w_reliability, three_w_scope = _build_three_w_reliability(config, data_root)
    tep_reliability, tep_scope = _build_tep_reliability(config)
    three_w = _audit_dataset("3W", three_w_reliability, config["three_w"], device)
    tep = _audit_dataset("TEP", tep_reliability, config["tep"], device)
    status = "DRFD_MECHANISM_GO" if three_w["passed"] and tep["passed"] else "DRFD_MECHANISM_NO_GO"
    result = {"status": status, "stage": "A", "new_training_runs": 0,
              "three_w": {**three_w, "scope": three_w_scope},
              "tep": {**tep, "scope": tep_scope},
              "phase_rule_changed": False, "dc_rule_changed": False}
    write_json(Path(config["stage_a"]["output"]), result)
    _write_report(Path(config["stage_a"]["report"]), result)
    return result


def _require_stage_a(config: dict[str, Any]) -> dict[str, Any]:
    path = Path(config["stage_a"]["output"])
    if not path.exists(): raise RuntimeError("DRFD Stage A result is missing")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result["status"] != "DRFD_MECHANISM_GO":
        raise RuntimeError("DRFD Stage A is not GO; training is forbidden")
    return result


def _run_three_w_seed(config: dict[str, Any], data_root: Path, seed: int) -> dict[str, Any]:
    stage = config["three_w"]
    current = yaml.safe_load(Path(stage["base_config"]).read_text(encoding="utf-8"))
    current["seed"] = int(seed); current["protocol_seed"] = 42
    current.pop("criticality_source", None)
    current["domain_reliable_safe_frequency_diffusion"] = True
    current["methods"] = [DRFD_METHOD]
    current["training"]["supcon_batching"] = "original"
    current["criticality"] = copy.deepcopy(stage["criticality"])
    current["output_dir"] = str(Path(stage["output_dir"]) / f"seed_{seed}")
    result = run_three_w(current, data_root)
    return {"result_path": str(Path(current["output_dir"]) / "result.json"),
            "method": result["methods"][DRFD_METHOD], "fairness": result["fairness"]}


def _run_tep_seed(config: dict[str, Any], seed: int) -> dict[str, Any]:
    stage = config["tep"]
    base_config = yaml.safe_load(Path(stage["base_config"]).read_text(encoding="utf-8"))
    _configure(base_config); views, _ = load_fixed_views(base_config); clean = bases(views)
    stages = {split: fault_stages(views[split], base_config) for split in views}
    train_log = log_amplitude_phase(clean["train"])[0]
    scaler = fit_frequency_scaler(train_log, "train")
    reliability = build_tep_stratified_run_bootstrap(
        scaler.transform(train_log), views["train"], stages["train"], stage["criticality"], train_log)
    statistics = fit_spectral_statistics(clean["train"], float(stage["spectral_diffusion"]["clip_quantile"]), "train")
    device = select_device(str(config["device"]))
    schedule = DiffusionSchedule.cosine(int(stage["spectral_diffusion"]["diffusion_steps"]), device)
    augmenter = FrequencyForwardDiffusion(statistics, schedule.alpha_bars, reliability["r1"]["soft_mask"],
        int(stage["spectral_diffusion"]["t_uniform"]), int(stage["spectral_diffusion"]["t_critical"]),
        bool(stage["spectral_diffusion"]["preserve_phase"]), bool(stage["spectral_diffusion"]["preserve_dc"]), device)
    variance, allocation = constrained_safe_variance(
        schedule.alpha_bars, reliability["r1"]["soft_mask"], reliability["rank_q25"], reliability["rank_q75"],
        reliability["reliable_noncritical"], reliability["ambiguous"],
        bool(stage["spectral_diffusion"]["preserve_dc"]), int(stage["spectral_diffusion"]["t_critical"]),
        int(stage["spectral_diffusion"]["t_uniform"]), int(stage["spectral_diffusion"]["t_noncritical"]))
    if allocation["budget_error_fraction"] > .02: raise RuntimeError("TEP DRFD mechanism Gate is not GO")
    baseline = json.loads(Path(stage["existing_result"]).read_text(encoding="utf-8"))
    seed = int(seed); augmented = {}; audits = {}
    for split, offset in (("train", 0), ("validation", 100), ("test", 200)):
        sampling_seed = seed + int(stage["spectral_diffusion"]["sampling_seed_offset"]) + offset
        augmented[split], diag = augmenter.augment(
            clean[split], "domain_reliable_safe", sampling_seed,
            int(stage["spectral_diffusion"]["t_noncritical"]), int(base_config["training"]["batch_size"]),
            noise_structure="iid", variance_override=variance)
        audits[split] = augmentation_mechanism_metrics(clean[split], augmented[split], views[split]["labels"],
            stages[split], reliability["r1"]["masks"]["composite"], diag)
        old_budget = baseline["seed_results"][str(seed)]["methods"]["R1"]["augmentation_audit"][split]["expected_total_noise_budget"]
        if abs(audits[split]["expected_total_noise_budget"] - old_budget) > 1e-6:
            raise RuntimeError(f"TEP DRFD/R1 {split} noise budgets differ")
    runtime = _runtime(base_config, seed); runtime["diagnosis"] = {"threshold_band_width": .05, "high_correlation_quantile": .90}
    pretrain_orders = epoch_orders(len(clean["train"]), int(runtime["epochs"]), seed + 10_000)
    probe_orders = epoch_orders(len(clean["train"]), int(runtime["probe_epochs"]), seed + 20_000)
    seed_everything(seed); template = build_model(runtime["model"], clean["train"].shape[1], 2)
    initial_state = copy.deepcopy(template.state_dict()); old = baseline["seed_results"][str(seed)]["fairness"]
    fairness = {"manifest_sha256": file_sha256(base_config["fixed_views"]["manifest"]),
                "initialization_sha256": _state_hash(initial_state),
                "pretrain_order_sha256": sha256_strings([','.join(map(str, order)) for order in pretrain_orders]),
                "probe_order_sha256": sha256_strings([','.join(map(str, order)) for order in probe_orders])}
    for key, value in fairness.items():
        if value != old[key]: raise RuntimeError(f"TEP DRFD fairness differs for seed {seed}: {key}")
    output = Path(stage["output_dir"]) / f"seed_{seed}" / "DRFD"
    metadata = {**old, "method": "DRFD", "seed": seed,
                "augmentation": "domain_reliable_safe_frequency_diffusion_iid_t5"}
    record = fit_tep_method("DRFD", augmented, audits, views, clean, stages, initial_state,
        pretrain_orders, probe_orders, runtime, device, output / "model.pt", metadata)
    payload = {"seed": seed, "method": record, "fairness": fairness,
               "allocation": _jsonable(allocation), "test_used_for_reliability_or_fit": False}
    write_json(Path(stage["output_dir"]) / f"seed_{seed}" / "result.json", payload)
    return payload


def run_stage_b(config: dict[str, Any], data_root: Path) -> dict[str, Any]:
    _require_stage_a(config)
    three_w = _run_three_w_seed(config, data_root, 42)
    tep = _run_tep_seed(config, 7)
    three_w_r1 = json.loads(Path("outputs/3w_diffusion_1seed_seed42/result.json").read_text(encoding="utf-8"))["methods"]["FREQUENCY_SELECTIVE_R1"]
    tep_r1 = json.loads(Path(config["tep"]["existing_result"]).read_text(encoding="utf-8"))["seed_results"]["7"]["methods"]["R1"]
    comparisons = {
        "3W": {"macro_f1_delta": float(three_w["method"]["metrics"]["macro_f1"] - three_w_r1["metrics"]["macro_f1"]),
               "far_delta": float(three_w["method"]["metrics"]["far"] - three_w_r1["metrics"]["far"])},
        "TEP": {"macro_f1_delta": float(tep["method"]["test"]["metrics"]["macro_f1"] - tep_r1["test"]["metrics"]["macro_f1"]),
                "far_delta": float(tep["method"]["test"]["metrics"]["far"] - tep_r1["test"]["metrics"]["far"])},
    }
    passed = all(item["macro_f1_delta"] >= -.03 and item["far_delta"] <= .05
                 for item in comparisons.values())
    result = {"status": "DRFD_KILL_TEST_GO" if passed else "DRFD_KILL_TEST_NO_GO",
              "new_training_runs": 2, "comparisons": comparisons,
              "three_w": three_w, "tep": tep}
    write_json(Path(config["stage_b"]["output"]), result)
    return result


def run_stage_c(config: dict[str, Any], data_root: Path) -> dict[str, Any]:
    kill_path = Path(config["stage_b"]["output"])
    if not kill_path.exists() or json.loads(kill_path.read_text(encoding="utf-8"))["status"] != "DRFD_KILL_TEST_GO":
        raise RuntimeError("DRFD Kill Test is not GO; Stage C is forbidden")
    kill = json.loads(kill_path.read_text(encoding="utf-8"))
    three_w = {"42": kill["three_w"]}
    tep = {"7": kill["tep"]}
    for seed in (43, 44):
        three_w[str(seed)] = _run_three_w_seed(config, data_root, seed)
    for seed in (42, 2026):
        tep[str(seed)] = _run_tep_seed(config, seed)
    result = {"status": "DRFD_STAGE_C_COMPLETE", "new_training_runs": 4,
              "total_drfd_training_runs": 6, "three_w": three_w, "tep": tep}
    write_json(Path(config["stage_c"]["output"]), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/domain_reliable_safe_frequency_diffusion.yaml")
    parser.add_argument("--stage", choices=("a", "b", "c"), default="a")
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.stage == "a": result = run_stage_a(config, args.data_root)
    elif args.stage == "b": result = run_stage_b(config, args.data_root)
    else: result = run_stage_c(config, args.data_root)
    print(json.dumps({"status": result["status"], "new_training_runs": result["new_training_runs"]}, ensure_ascii=False))


if __name__ == "__main__": main()
