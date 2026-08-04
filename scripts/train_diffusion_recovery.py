from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from diffusion import DiffusionSchedule, ddpm_restore
from models import MinimalConditionalDiffusion1D
from scripts.common import prepare_real
from scripts.run_rapid_diffusion_gates import (_classifier_view, _correlation_error,
                                               _masked_rmse)
from scripts.run_rapid_idea_validation import (_kept_ids, _loader, _masked_mae,
                                                _view_bundle)
from trainers import build_model
from utils import environment_metadata, seed_everything, write_json


MARKERS = ["DIFFUSION_DEBUG", "SINGLE_SEED", "NOT_FOR_PAPER_CLAIMS"]


def diffusion_objective(model: torch.nn.Module, schedule: DiffusionSchedule,
                        clean: torch.Tensor, degraded: torch.Tensor,
                        observation: torch.Tensor, timesteps: torch.Tensor,
                        noise: torch.Tensor, lambda_rec: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    noisy = schedule.q_sample(clean, timesteps, noise)
    noisy = torch.where(observation, degraded, noisy)
    predicted_noise = model(noisy, degraded, observation, timesteps)
    missing = ~observation
    diffusion_loss = ((predicted_noise - noise).square() * missing).sum() / missing.sum().clamp_min(1)
    predicted_clean = schedule.predict_x0(noisy, timesteps, predicted_noise)
    reconstruction_loss = (F.smooth_l1_loss(predicted_clean, clean, reduction="none") * missing).sum() / missing.sum().clamp_min(1)
    return diffusion_loss + lambda_rec * reconstruction_loss, diffusion_loss, reconstruction_loss, predicted_clean


@torch.no_grad()
def restore_array(model: torch.nn.Module, degraded: np.ndarray, observation: np.ndarray,
                  schedule: DiffusionSchedule, batch_size: int, device: str, seed: int,
                  clip_min: np.ndarray | None = None, clip_max: np.ndarray | None = None) -> np.ndarray:
    model.eval(); generator = torch.Generator(device=device).manual_seed(seed); restored = []
    minimum = None if clip_min is None else torch.from_numpy(clip_min.astype(np.float32)).to(device)[None, :, None]
    maximum = None if clip_max is None else torch.from_numpy(clip_max.astype(np.float32)).to(device)[None, :, None]
    for degraded_b, observation_b in _loader(degraded.astype(np.float32), observation.astype(bool), batch_size=batch_size, shuffle=False):
        restored.append(ddpm_restore(model, degraded_b.to(device), observation_b.to(device), schedule, generator, clip_min=minimum, clip_max=maximum).cpu().numpy())
    return np.concatenate(restored)


def first_difference_error(clean: np.ndarray, restored: np.ndarray) -> float:
    return float(np.mean(np.abs(np.diff(clean, axis=-1) - np.diff(restored, axis=-1))))


def recovery_metrics(clean: np.ndarray, restored: np.ndarray, observation: np.ndarray) -> dict[str, float]:
    missing = ~observation
    return {
        "masked_mae": float(np.mean(np.abs(clean[missing] - restored[missing]))),
        "masked_rmse": _masked_rmse(clean, restored, observation),
        "observed_mae": float(np.mean(np.abs(clean[observation] - restored[observation]))),
        "full_window_mae": float(np.mean(np.abs(clean - restored))),
        "correlation_matrix_error": _correlation_error(clean, restored),
        "first_difference_error": first_difference_error(clean, restored),
    }


def tiny_gate_checks(initial_mae: float, methods: dict[str, dict[str, float]],
                     sampling_maes: list[float], sample_checkpoints: list[dict[str, float]]) -> dict[str, bool]:
    diffusion = methods["diffusion"]; simple = methods["simple"]; degraded = methods["degraded"]
    return {
        "masked_mae_sustained_decline": bool(sample_checkpoints and sample_checkpoints[-1]["masked_mae"] < sample_checkpoints[0]["masked_mae"]),
        "meets_suggested_simple_mae_target": diffusion["masked_mae"] <= simple["masked_mae"],
        "observed_mae_near_zero": diffusion["observed_mae"] < 1e-7,
        "finite_training_and_sampling": bool(np.isfinite(sampling_maes).all() and np.isfinite(diffusion["masked_mae"])),
        "restoration_not_random_noise": diffusion["masked_mae"] < initial_mae * 0.7,
        "multiple_samples_stable": float(np.std(sampling_maes)) < max(float(np.mean(sampling_maes)) * 0.05, 1e-6),
        "fixed_windows_memorized_better_than_zero_fill": diffusion["masked_mae"] < degraded["masked_mae"],
    }


def small_subset_gate_checks(methods: dict[str, dict[str, float]], task_metrics: dict[str, Any],
                             sample_checkpoints: list[dict[str, float]]) -> dict[str, bool]:
    simple = methods["simple"]; diffusion = methods["diffusion"]
    simple_task = task_metrics["simple"]; diffusion_task = task_metrics["diffusion"]
    return {
        "validation_masked_mae_converged": bool(sample_checkpoints and sample_checkpoints[-1]["masked_mae"] < sample_checkpoints[0]["masked_mae"] * 0.5),
        "test_masked_mae_close_to_simple_within_10_percent": diffusion["masked_mae"] <= simple["masked_mae"] * 1.10,
        "fault_recall_not_lower_than_simple_by_over_2_points": diffusion_task["metrics"]["fault_recall"] >= simple_task["metrics"]["fault_recall"] - 0.02,
        "far_not_catastrophic_vs_simple_within_10_points": diffusion_task["metrics"]["far"] <= simple_task["metrics"]["far"] + 0.10,
        "semantic_consistency_materially_above_old_0_4356": diffusion_task["teacher_prediction_consistency"] >= 0.70,
        "at_least_one_task_metric_better_than_simple": bool(
            diffusion_task["metrics"]["auprc"] > simple_task["metrics"]["auprc"]
            or diffusion_task["metrics"]["fault_recall"] > simple_task["metrics"]["fault_recall"]
            or diffusion_task["metrics"]["far"] < simple_task["metrics"]["far"]
            or diffusion_task["teacher_prediction_consistency"] > simple_task["teacher_prediction_consistency"]
        ),
    }


def prepare_bundles(config: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], Any, dict[str, Any]]:
    clean_data, manifest, window_stats = prepare_real(config, degrade=False); bundles = {}
    for split in ("train", "validation", "test"):
        clean, labels = clean_data[split]; ids = _kept_ids(window_stats[split])
        degraded, simple, observation, _, degraded_mae, simple_mae = _view_bundle(clean, ids, config)
        bundles[split] = {"clean": clean, "labels": labels, "ids": ids, "degraded": degraded,
                          "simple": simple, "observation": observation,
                          "degraded_mae": degraded_mae, "simple_mae": simple_mae}
    return bundles, manifest, window_stats


def balanced_tiny(bundle: dict[str, Any], maximum: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed); half = maximum // 2
    normal = rng.choice(np.flatnonzero(bundle["labels"] == 0), half, replace=False)
    fault = rng.choice(np.flatnonzero(bundle["labels"] == 1), maximum - half, replace=False)
    indices = np.sort(np.r_[normal, fault])
    return {key: value[indices] if isinstance(value, np.ndarray) and len(value) == len(bundle["labels"]) else value for key, value in bundle.items()}


def plot_debug(path: Path, history: list[dict[str, float]], clean: np.ndarray,
               degraded: np.ndarray, restored: np.ndarray, observation: np.ndarray) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(10, 7))
    axes[0].plot([item["epoch"] for item in history], [item["total_loss"] for item in history], label="total")
    axes[0].plot([item["epoch"] for item in history], [item["diffusion_loss"] for item in history], label="diffusion")
    axes[0].plot([item["epoch"] for item in history], [item["reconstruction_loss"] for item in history], label="masked rec")
    axes[0].legend(); axes[0].set_title("DIFFUSION_DEBUG training curves")
    channel = 0; axes[1].plot(clean[0, channel], label="clean"); axes[1].plot(degraded[0, channel], label="degraded")
    axes[1].plot(restored[0, channel], label="DDPM restored"); axes[1].scatter(np.flatnonzero(~observation[0, channel]), restored[0, channel, ~observation[0, channel]], s=8, label="missing")
    axes[1].legend(); axes[1].set_title("NOT_FOR_PAPER_CLAIMS")
    figure.tight_layout(); figure.savefig(path); plt.close(figure)


def train(config: dict[str, Any]) -> dict[str, Any]:
    output = Path(config["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    seed = int(config["random_seed"]); seed_everything(seed); device = str(config["device"]); started = time.perf_counter()
    bundles, manifest, window_stats = prepare_bundles(config)
    if config["stage"] == "tiny_overfit":
        bundles["train"] = balanced_tiny(bundles["train"], int(config["diffusion"]["max_windows"]), seed)
    channels = bundles["train"]["clean"].shape[1]
    model = MinimalConditionalDiffusion1D(channels, int(config["diffusion"]["hidden_channels"]),
                                          int(config["diffusion"]["hidden_channels"]), int(config["diffusion"]["residual_blocks"])).to(device)
    schedule = DiffusionSchedule.cosine(int(config["diffusion"]["steps"]), device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["diffusion"]["learning_rate"]))
    lambda_rec = float(config["diffusion"]["lambda_rec"]); history = []; samples = []
    clip_min = bundles["train"]["clean"].min(axis=(0, 2)); clip_max = bundles["train"]["clean"].max(axis=(0, 2))
    best_state, best_metric, stale = None, float("inf"), 0
    torch.cuda.reset_peak_memory_stats(); training_started = time.perf_counter()
    initial_restored = restore_array(model, bundles["train"]["degraded"], bundles["train"]["observation"], schedule, int(config["batch_size"]), device, seed + 1000, clip_min, clip_max)
    initial_mae = recovery_metrics(bundles["train"]["clean"], initial_restored, bundles["train"]["observation"])["masked_mae"]
    for epoch in range(1, int(config["diffusion"]["epochs"]) + 1):
        model.train(); totals = []; diffusion_values = []; reconstruction_values = []
        for _ in range(int(config["diffusion"].get("loader_repeats_per_epoch", 1))):
            for clean_b, degraded_b, observation_b in _loader(
                bundles["train"]["clean"].astype(np.float32), bundles["train"]["degraded"].astype(np.float32), bundles["train"]["observation"].astype(bool),
                batch_size=int(config["batch_size"]), shuffle=True,
            ):
                clean_b, degraded_b, observation_b = clean_b.to(device), degraded_b.to(device), observation_b.to(device)
                timesteps = torch.randint(0, len(schedule.betas), (len(clean_b),), device=device); noise = torch.randn_like(clean_b)
                optimizer.zero_grad(); total, diffusion_loss, reconstruction_loss, _ = diffusion_objective(model, schedule, clean_b, degraded_b, observation_b, timesteps, noise, lambda_rec)
                total.backward(); optimizer.step(); totals.append(float(total.detach())); diffusion_values.append(float(diffusion_loss.detach())); reconstruction_values.append(float(reconstruction_loss.detach()))
        record = {"epoch": epoch, "total_loss": float(np.mean(totals)), "diffusion_loss": float(np.mean(diffusion_values)), "reconstruction_loss": float(np.mean(reconstruction_values))}
        evaluate = epoch == 1 or epoch % int(config["diffusion"]["evaluation_interval"]) == 0 or epoch == int(config["diffusion"]["epochs"])
        if evaluate:
            split = "train" if config["stage"] == "tiny_overfit" else "validation"
            restored = restore_array(model, bundles[split]["degraded"], bundles[split]["observation"], schedule, int(config["batch_size"]), device, seed + 2000, clip_min, clip_max)
            sampled_metrics = recovery_metrics(bundles[split]["clean"], restored, bundles[split]["observation"])
            record["sampled_masked_mae"] = sampled_metrics["masked_mae"]; samples.append({"epoch": epoch, **sampled_metrics})
            metric = sampled_metrics["masked_mae"]
            if metric < best_metric - 1e-6:
                best_metric, best_state, stale = metric, copy.deepcopy(model.state_dict()), 0
            else:
                stale += 1
            patience = config["diffusion"].get("early_stopping_patience")
            if patience is not None and stale >= int(patience): history.append(record); break
        history.append(record)
    if best_state is not None: model.load_state_dict(best_state)
    training_seconds = time.perf_counter() - training_started; checkpoint = output / "best_diffusion.pt"; torch.save(model.state_dict(), checkpoint)
    evaluation_split = "train" if config["stage"] == "tiny_overfit" else "test"
    evaluation = bundles[evaluation_split]
    restored_samples = [restore_array(model, evaluation["degraded"], evaluation["observation"], schedule, int(config["batch_size"]), device, seed + 3000 + index, clip_min, clip_max) for index in range(3)]
    restored = restored_samples[0]
    methods = {"degraded": recovery_metrics(evaluation["clean"], evaluation["degraded"], evaluation["observation"]),
               "simple": recovery_metrics(evaluation["clean"], evaluation["simple"], evaluation["observation"]),
               "diffusion": recovery_metrics(evaluation["clean"], restored, evaluation["observation"])}
    sampling_maes = [recovery_metrics(evaluation["clean"], value, evaluation["observation"])["masked_mae"] for value in restored_samples]
    np.savez_compressed(output / "evaluation_batch.npz", clean=evaluation["clean"], degraded=evaluation["degraded"], simple=evaluation["simple"], observation=evaluation["observation"], labels=evaluation["labels"], restored=restored, clip_min=clip_min, clip_max=clip_max)
    task_metrics = None
    if config["stage"] == "small_subset":
        teacher = build_model(config["model"], channels, 2).to(device)
        teacher.load_state_dict(torch.load(config["teacher_checkpoint"], map_location=device, weights_only=True)); teacher.eval()
        task_metrics = {name: _classifier_view(teacher, evaluation["clean"], value, evaluation["labels"], config, device)
                        for name, value in (("degraded", evaluation["degraded"]), ("simple", evaluation["simple"]), ("diffusion", restored))}
    simple_mae = methods["simple"]["masked_mae"]; diffusion_mae = methods["diffusion"]["masked_mae"]
    if config["stage"] == "tiny_overfit":
        checks = tiny_gate_checks(initial_mae, methods, sampling_maes, samples)
        passed = sum(checks.values()) >= 4
        status = "TINY_OVERFIT_PASSED" if passed else "TINY_OVERFIT_FAILED"
    else:
        checks = small_subset_gate_checks(methods, task_metrics, samples)
        status = "DIFFUSION_RECOVERY_READY_FOR_IDEA_RETEST" if sum(checks.values()) >= 4 else "TINY_OVERFIT_PASSED_BUT_SMALL_SUBSET_NO_GO"
    result = {"markers": MARKERS, "stage": config["stage"], "status": status, **environment_metadata(),
              "initial_sampled_masked_mae": initial_mae, "methods": methods, "task_metrics": task_metrics,
              "sampling_masked_maes": sampling_maes, "history": history, "sample_checkpoints": samples,
              "tiny_gate_checks": checks if config["stage"] == "tiny_overfit" else None,
              "small_subset_gate_checks": checks if config["stage"] == "small_subset" else None,
              "training_seconds": training_seconds, "total_seconds": time.perf_counter() - started,
              "peak_gpu_mib": torch.cuda.max_memory_allocated() / 1024 ** 2, "split_manifest": manifest.__dict__,
              "train_windows": len(bundles["train"]["clean"]), "evaluation_windows": len(evaluation["clean"]),
              "terminal_alpha_bar": float(schedule.alpha_bars[-1])}
    write_json(output / "result.json", result); write_json(output / "config.json", config)
    plot_debug(output / "debug_curves.png", history, evaluation["clean"], evaluation["degraded"], restored, evaluation["observation"])
    print(json.dumps({"status": status, "diffusion_masked_mae": diffusion_mae, "simple_masked_mae": simple_mae, "output": str(output)}, ensure_ascii=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if config["stage"] == "small_subset":
        tiny_result = json.loads(Path(config["tiny_result_path"]).read_text(encoding="utf-8"))
        checks = tiny_gate_checks(tiny_result["initial_sampled_masked_mae"], tiny_result["methods"], tiny_result["sampling_masked_maes"], tiny_result["sample_checkpoints"])
        if sum(checks.values()) < 4: raise RuntimeError("Tiny Overfit 未通过多数条件，禁止小型真实子集训练")
    train(config)


if __name__ == "__main__": main()
