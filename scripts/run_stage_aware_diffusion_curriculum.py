from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml

from diffusion import StageAwareTimestepScheduler
from frequency import fault_stages
from metrics import representation_diagnostics
from scripts.audit_semantic_diffusion_augmentation import bases
from scripts.diagnose_frequency_selective_far import score_profile
from scripts.run_diffusion_quality_retest import (
    _contrastive_loss, _fit_probe, _metrics, _probabilities, _state_hash,
    best_probe_record, epoch_orders, load_fixed_views,
)
from scripts.run_frequency_selective_r1_3seed import (
    _stage_metrics, array_sha256, file_sha256, sha256_strings,
)
from scripts.run_stage_frequency_diffusion_mvp import (
    _build_frequency_components, _configure, _runtime, detection_delays,
)
from trainers import build_model
from utils import environment_metadata, seed_everything, write_json


METHODS = ("R1", "C3-E", "C3-S")


def validate_config(config: dict[str, Any], base_config: dict[str, Any], r1_config: dict[str, Any]) -> None:
    if config["seeds"] != [7, 42, 2026] or config["methods"] != list(METHODS):
        raise ValueError("C3 seeds/methods are frozen")
    frozen = config["frozen"]
    expected = {"t_critical": 1, "r1_t_noncritical": 5, "noise_structure": "iid",
                "preserve_phase": True, "preserve_dc": True, "critical_ratio": .3,
                "weight_discriminative": .5, "weight_early": .3, "weight_run_stability": .2}
    if frozen != expected: raise ValueError("C3 frozen R1 configuration changed")
    critical = base_config["criticality"]
    if any(critical[key] != expected[key] for key in ("critical_ratio", "weight_discriminative", "weight_early", "weight_run_stability")):
        raise ValueError("D/E/S or critical ratio changed")
    if r1_config["recorded_result"]["status"] != "FREQUENCY_SELECTIVE_R1_3SEED_GO":
        raise RuntimeError("R1 3-seed GO is required")
    if config["curriculum"] != {"t_start": 2,
                                 "epoch_only_targets": {"normal": 5, "early": 5, "middle": 5, "stable": 5},
                                 "stage_aware_targets": {"normal": 5, "early": 3, "middle": 4, "stable": 5}}:
        raise ValueError("C3 curriculum targets changed")


def training_stage_names(labels: np.ndarray, stages: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels); stages = np.asarray(stages).astype(str)
    names = np.where(labels == 0, "normal", stages)
    if not np.isin(names, ("normal", "early", "middle", "stable")).all():
        raise ValueError("training contains an unsupported fault stage")
    return names


def _strength_audit(base: np.ndarray, changed: np.ndarray, stage_names: np.ndarray,
                    timesteps: dict[str, int], critical_mask: np.ndarray) -> dict[str, Any]:
    base_scale = np.maximum(base.std(axis=(1, 2)), 1e-6)
    sample_l1 = np.abs(changed - base).mean(axis=(1, 2)) / base_scale
    base_log = np.log1p(np.abs(np.fft.rfft(base, axis=-1)))
    changed_log = np.log1p(np.abs(np.fft.rfft(changed, axis=-1)))
    difference = np.abs(changed_log - base_log)
    critical = np.asarray(critical_mask, dtype=bool)
    stage_records = {}
    for stage in ("normal", "early", "middle", "stable"):
        selected = stage_names == stage
        stage_records[stage] = {"count": int(selected.sum()), "effective_t": int(timesteps[stage]),
                                "normalized_l1": float(sample_l1[selected].mean()) if selected.any() else None}
    return {"mean_effective_t": float(np.mean([timesteps[stage] for stage in stage_names])),
            "stages": stage_records, "normalized_l1": float(sample_l1.mean()),
            "critical_frequency_l1": float(difference[:, critical].mean()),
            "noncritical_frequency_l1": float(difference[:, ~critical].mean()),
            "finite": bool(np.isfinite(changed).all())}


def build_epoch_views(augmenter, base_train: np.ndarray, stage_names: np.ndarray,
                      scheduler: StageAwareTimestepScheduler | None, epochs: int, sampling_seed: int,
                      batch_size: int, critical_mask: np.ndarray) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    views, audits = [], []
    for epoch in range(epochs):
        timesteps = ({stage: 5 for stage in ("normal", "early", "middle", "stable")} if scheduler is None
                     else scheduler.epoch_timesteps(epoch, epochs))
        candidates = {}
        for timestep in sorted(set(timesteps.values())):
            candidates[timestep], _ = augmenter.augment(base_train, "selective", sampling_seed, timestep,
                                                         batch_size, noise_structure="iid")
        changed = np.empty_like(base_train)
        for stage, timestep in timesteps.items():
            selected = stage_names == stage; changed[selected] = candidates[timestep][selected]
        views.append(changed)
        audits.append(_strength_audit(base_train, changed, stage_names, timesteps, critical_mask))
    return views, audits


def _fit_dynamic_supcon(model, base_train, labels, epoch_views, validation, orders, runtime, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=float(runtime["learning_rate"]))
    quality = np.ones(len(labels), np.float32); val_q = np.ones(len(validation["labels"]), np.float32)
    validation_order = np.arange(len(validation["labels"])); history = []; best_state = None; best_loss = float("inf"); stale = 0
    for epoch, order in enumerate(orders):
        bundle = {"clean": base_train, "restored": epoch_views[epoch], "labels": labels}
        loss, diagnostics = _contrastive_loss(model, bundle, quality, order, runtime, device, optimizer)
        val_loss, _ = _contrastive_loss(model, validation, val_q, validation_order, runtime, device, None)
        history.append({"epoch": epoch, "loss": loss, "validation_supcon_loss": val_loss, **diagnostics})
        if val_loss < best_loss - 1e-6:
            best_loss, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= int(runtime["early_stopping_patience"]): break
    if best_state is not None: model.load_state_dict(best_state)
    return history


def _fit_method(name, epoch_views, epoch_audits, validation_augmented, test_augmented, views, base, stages,
                initial_state, pretrain_orders, probe_orders, runtime, device, checkpoint, metadata):
    metrics_path = checkpoint.parent / "metrics.json"
    if checkpoint.exists() and metrics_path.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        result = json.loads(metrics_path.read_text(encoding="utf-8"))
        if payload.get("metadata") != metadata or result.get("metadata") != metadata: raise RuntimeError("resume metadata mismatch")
        return result
    if checkpoint.exists() or metrics_path.exists(): raise RuntimeError("incomplete method output")
    seed_everything(int(runtime["random_seed"])); started = time.perf_counter()
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    model = build_model(runtime["model"], base["train"].shape[1], 2).to(device); model.load_state_dict(initial_state)
    validation = {"clean": base["validation"], "restored": validation_augmented, "labels": views["validation"]["labels"]}
    pretrain = _fit_dynamic_supcon(model, base["train"], views["train"]["labels"], epoch_views,
                                    validation, pretrain_orders, runtime, device)
    seed_everything(int(runtime["random_seed"]) + 1)
    probe = _fit_probe(model, {"clean": base["train"], "labels": views["train"]["labels"]},
                       {"restored": base["validation"], "labels": views["validation"]["labels"]},
                       probe_orders, runtime, device)
    best = best_probe_record(probe); threshold = float(best["validation_threshold"]); split_records = {}
    for split, augmented in (("validation", validation_augmented), ("test", test_augmented)):
        probability, embedding = _probabilities(model, base[split], int(runtime["batch_size"]), device)
        _, augmented_embedding = _probabilities(model, augmented, int(runtime["batch_size"]), device)
        scores = probability[:, 1]; prediction = scores >= threshold
        split_records[split] = {"metrics": _metrics(views[split]["labels"], scores, threshold),
                                "score_profile": score_profile(views[split]["labels"], scores, threshold, .05),
                                "stages": _stage_metrics(stages[split], prediction),
                                "detection_delay": detection_delays(views[split], prediction, runtime),
                                "representation": representation_diagnostics(embedding, augmented_embedding, views[split]["labels"])}
    result = {"method": name, "seed": int(runtime["random_seed"]), "validation_threshold": threshold,
              "best_pretrain_epoch": int(min(pretrain, key=lambda row: row["validation_supcon_loss"])["epoch"]),
              "best_probe_epoch": int(best["epoch"]), "pretrain_history": pretrain, "probe_history": probe,
              "initialization_sha256": _state_hash(initial_state), "validation": split_records["validation"],
              "test": split_records["test"], "effective_timestep_history": epoch_audits, "metadata": metadata,
              "training_seconds": time.perf_counter() - started,
              "peak_gpu_mib": torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0}
    checkpoint.parent.mkdir(parents=True, exist_ok=False)
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, checkpoint); write_json(metrics_path, result)
    return result


def run_seed(config, seed, views, base, stages, critical, augmenter, fingerprints):
    output = Path(config["output_dir"]) / f"seed_{seed}"; result_path = output / "result.json"
    if result_path.exists(): return json.loads(result_path.read_text(encoding="utf-8"))
    runtime_base = yaml.safe_load(Path(config["base_config"]).read_text(encoding="utf-8")); runtime = _runtime(runtime_base, seed)
    epochs = int(runtime["epochs"]); stage_names = training_stage_names(views["train"]["labels"], stages["train"])
    schedulers = {"R1": None,
                  "C3-E": StageAwareTimestepScheduler("epoch_only", 2, config["curriculum"]["epoch_only_targets"]),
                  "C3-S": StageAwareTimestepScheduler("stage_aware", 2, config["curriculum"]["stage_aware_targets"])}
    sampling_seed = seed + int(runtime_base["spectral_diffusion"]["sampling_seed_offset"])
    validation_augmented, _ = augmenter.augment(base["validation"], "selective", sampling_seed + 100, 5,
                                                int(runtime["batch_size"]), noise_structure="iid")
    test_augmented, _ = augmenter.augment(base["test"], "selective", sampling_seed + 200, 5,
                                          int(runtime["batch_size"]), noise_structure="iid")
    pretrain_orders = epoch_orders(len(base["train"]), epochs, seed + 10_000)
    probe_orders = epoch_orders(len(base["train"]), int(runtime["probe_epochs"]), seed + 20_000)
    seed_everything(seed); template = build_model(runtime["model"], base["train"].shape[1], 2); initial_state = copy.deepcopy(template.state_dict())
    common = {**fingerprints, "seed": seed, "initialization_sha256": _state_hash(initial_state),
              "pretrain_order_sha256": sha256_strings([','.join(map(str, x)) for x in pretrain_orders]),
              "probe_order_sha256": sha256_strings([','.join(map(str, x)) for x in probe_orders]),
              "stage_only_used_by_training_augmentation": True}
    methods = {}
    for name in METHODS:
        epoch_views, epoch_audits = build_epoch_views(
            augmenter, base["train"], stage_names, schedulers[name], epochs, sampling_seed,
            int(runtime["batch_size"]), critical["masks"]["composite"])
        methods[name] = _fit_method(
            name, epoch_views, epoch_audits, validation_augmented, test_augmented, views, base, stages,
            initial_state, pretrain_orders, probe_orders, runtime, str(config["device"]),
            output / name / "model.pt", {**common, "method": name})
        del epoch_views, epoch_audits
    result = {"seed": seed, "methods": methods, "fairness": common, "same_initialization": True,
              "same_orders": True, "same_mask": True, "same_validation_objective": True,
              "stage_not_used_by_encoder_probe_or_inference": True}
    write_json(result_path, result); return result


def run(config: dict[str, Any]) -> dict[str, Any]:
    final_path = Path(config["output_dir"]) / "result.json"
    if final_path.exists(): return json.loads(final_path.read_text(encoding="utf-8"))
    base_config = yaml.safe_load(Path(config["base_config"]).read_text(encoding="utf-8")); r1_config = yaml.safe_load(Path(config["r1_config"]).read_text(encoding="utf-8"))
    validate_config(config, base_config, r1_config); _configure(base_config)
    views, _ = load_fixed_views(base_config); base = bases(views); stages = {split: fault_stages(views[split], base_config) for split in views}
    critical, augmenter = _build_frequency_components(base_config, views, base, stages, str(config["device"]))
    code_files = ["diffusion/stage_curriculum.py", "scripts/run_stage_aware_diffusion_curriculum.py",
                  "scripts/run_frequency_selective_r1_3seed.py", "diffusion/frequency_selective.py"]
    fingerprints = {"manifest_sha256": file_sha256(config["fixed_views_manifest"]),
                    "mask_sha256": array_sha256(critical["masks"]["composite"]),
                    "r1_config_sha256": file_sha256(config["r1_config"]),
                    "training_code_sha256": sha256_strings([f"{p}:{file_sha256(p)}" for p in code_files])}
    r1_result = json.loads(Path(config["r1_result"]).read_text(encoding="utf-8"))
    if fingerprints["mask_sha256"] != r1_result["fingerprints"]["mask_sha256"]: raise RuntimeError("C3 mask differs from R1")
    seed_results = {"7": run_seed(config, 7, views, base, stages, critical, augmenter, fingerprints)}
    from scripts.summarize_stage_aware_diffusion_curriculum import summarize
    seed7 = summarize(config, seed_results, fingerprints)
    if seed7["seed7_status"] == "STAGE_AWARE_CURRICULUM_SEED7_GO":
        for seed in (42, 2026): seed_results[str(seed)] = run_seed(config, seed, views, base, stages, critical, augmenter, fingerprints)
    result = summarize(config, seed_results, fingerprints); result.update(environment_metadata())
    final_path.parent.mkdir(parents=True, exist_ok=True); write_json(final_path, result)
    summarize(config, seed_results, fingerprints, result=result, report_path=config["report"])
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/stage_aware_diffusion_curriculum.yaml")
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")); result = run(config)
    print(json.dumps({"status": result["status"], "seed7_status": result["seed7_status"], "seeds": list(result["seed_results"])}, ensure_ascii=False))


if __name__ == "__main__": main()
