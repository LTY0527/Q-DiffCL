from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from diffusion import DiffusionSchedule
from diffusion.semantic_augmentation import (SemanticPartialDiffusion1D,
                                             partial_diffusion_objective,
                                             partial_reverse_sample)
from losses import freeze_teacher, semantic_consistency_losses
from scripts.run_diffusion_quality_retest import epoch_orders, load_fixed_views
from scripts.run_rapid_idea_validation import _simple_interpolate
from trainers import build_model
from utils import deterministic_seed, environment_metadata, seed_everything, write_json


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()): digest.update(name.encode()); digest.update(value.cpu().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def teacher_arrays(teacher: torch.nn.Module, values: np.ndarray, batch_size: int,
                   device: str) -> tuple[np.ndarray, np.ndarray]:
    probabilities = []; features = []
    for start in range(0, len(values), batch_size):
        output = teacher(torch.from_numpy(values[start:start + batch_size]).float().to(device))
        probabilities.append(torch.softmax(output["logits"], 1).cpu().numpy()); features.append(output["embedding"].cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(features)


def audit_indices(views: dict[str, dict[str, np.ndarray]], manifest_path: str) -> dict[str, np.ndarray]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8")); result = {}
    for split in ("train", "validation", "test"):
        selected = [record["window_id"] for record in manifest["records"] if record["split"] == split]
        lookup = {str(value): index for index, value in enumerate(views[split]["window_id"])}
        if any(value not in lookup for value in selected): raise RuntimeError(f"audit subset mismatch: {split}")
        result[split] = np.asarray([lookup[value] for value in selected], dtype=np.int64)
    return result


def bases(views: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {split: _simple_interpolate(bundle["degraded"], bundle["observation"]).astype(np.float32)
            for split, bundle in views.items()}


def train_generator(
    name: str, model: SemanticPartialDiffusion1D, teacher: torch.nn.Module,
    views: dict[str, dict[str, np.ndarray]], base: dict[str, np.ndarray],
    semantic: dict[str, np.ndarray], schedule: DiffusionSchedule,
    config: dict[str, Any], device: str,
) -> tuple[list[dict[str, float]], float, float]:
    settings = config["generator"]; seed = int(config["random_seed"]); batch_size = int(config["batch_size"])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(settings["learning_rate"])); history = []
    orders = epoch_orders(len(base["train"]), int(settings["epochs"]), seed + 30_000)
    started = time.perf_counter()
    for epoch, order in enumerate(orders):
        model.train(); values = {key: [] for key in ("total", "diff", "prob", "feat")}
        generator = torch.Generator(device=device).manual_seed(seed + 40_000 + epoch)
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            base_b = torch.from_numpy(base["train"][indices]).to(device)
            observation_b = torch.from_numpy(views["train"]["observation"][indices]).to(device)
            semantic_b = torch.from_numpy(semantic["train"][indices]).to(device)
            timesteps = torch.randint(0, int(settings["train_max_timestep"]) + 1, (len(indices),), device=device, generator=generator)
            noise = torch.randn(base_b.shape, device=device, generator=generator)
            optimizer.zero_grad(); diffusion_loss, predicted = partial_diffusion_objective(
                model, schedule, base_b, observation_b, semantic_b, timesteps, noise,
            )
            probability_loss = feature_loss = diffusion_loss.new_zeros(())
            if name == "G1":
                probability_loss, feature_loss = semantic_consistency_losses(teacher, base_b, predicted)
            total = diffusion_loss + (float(settings["lambda_prob"]) * probability_loss +
                                      float(settings["lambda_feat"]) * feature_loss if name == "G1" else 0)
            total.backward(); optimizer.step()
            for key, value in (("total", total), ("diff", diffusion_loss), ("prob", probability_loss), ("feat", feature_loss)):
                values[key].append(float(value.detach()))
        history.append({"epoch": epoch, **{key: float(np.mean(value)) for key, value in values.items()}})
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0
    return history, elapsed, peak


@torch.no_grad()
def generate_repeats(
    model: SemanticPartialDiffusion1D, base: np.ndarray, observation: np.ndarray,
    semantic: np.ndarray, schedule: DiffusionSchedule, t_aug: int, repeats: int,
    batch_size: int, device: str, seed: int, clip_min: np.ndarray, clip_max: np.ndarray,
) -> np.ndarray:
    result = np.empty((repeats, *base.shape), dtype=np.float32)
    minimum = torch.from_numpy(clip_min).float().to(device)[None, :, None]
    maximum = torch.from_numpy(clip_max).float().to(device)[None, :, None]
    model.eval()
    for repeat in range(repeats):
        generator = torch.Generator(device=device).manual_seed(seed + t_aug * 100 + repeat)
        for start in range(0, len(base), batch_size):
            stop = min(start + batch_size, len(base))
            result[repeat, start:stop] = partial_reverse_sample(
                model, schedule, torch.from_numpy(base[start:stop]).to(device),
                torch.from_numpy(observation[start:stop]).to(device),
                torch.from_numpy(semantic[start:stop]).to(device), t_aug, generator, minimum, maximum,
            ).cpu().numpy()
    return result


def traditional_augmentation(values: np.ndarray, window_ids: np.ndarray,
                             config: dict[str, float], seed: int) -> np.ndarray:
    result = np.empty_like(values)
    for index, (value, window_id) in enumerate(zip(values, window_ids)):
        rng = np.random.default_rng(deterministic_seed(seed, str(window_id), "traditional_augmentation"))
        scale = rng.normal(1, float(config["scaling_std"]), size=(value.shape[0], 1))
        jitter = rng.normal(0, float(config["jitter_std"]), size=value.shape)
        result[index] = value * scale + jitter
    return result


def _fault_type(run_uid: str) -> int:
    match = re.search(r":fault_(\d+):", str(run_uid)); return int(match.group(1)) if match else 0


def augmentation_metrics(
    base: np.ndarray, augmented: np.ndarray, labels: np.ndarray, run_uids: np.ndarray,
    base_probability: np.ndarray, base_feature: np.ndarray, teacher: torch.nn.Module,
    batch_size: int, device: str,
) -> dict[str, Any]:
    repeats, count = augmented.shape[:2]; flat = augmented.reshape(-1, *augmented.shape[-2:])
    probability, feature = teacher_arrays(teacher, flat, batch_size, device)
    probability = probability.reshape(repeats, count, -1); feature = feature.reshape(repeats, count, -1)
    base_class = base_probability.argmax(1); generated_class = probability.argmax(2)
    consistency = generated_class == base_class[None, :]
    normal = labels == 0; fault = labels != 0
    probability_kl = (base_probability[None] * (np.log(np.maximum(base_probability[None], 1e-12)) -
                                                 np.log(np.maximum(probability, 1e-12)))).sum(2)
    feature_cosine = (feature * base_feature[None]).sum(2) / np.maximum(
        np.linalg.norm(feature, axis=2) * np.linalg.norm(base_feature[None], axis=2), 1e-12)
    scale = np.maximum(base.std(axis=(1, 2)), 1e-6)[None, :]
    difference = augmented - base[None]
    normalized_l1 = np.abs(difference).mean(axis=(2, 3)) / scale
    normalized_l2 = np.sqrt((difference ** 2).mean(axis=(2, 3))) / scale
    derivative = np.abs(np.diff(augmented, axis=-1) - np.diff(base[None], axis=-1)).mean(axis=(2, 3)) / scale
    pairwise = [np.abs(augmented[first] - augmented[second]).mean(axis=(1, 2)) / scale[0]
                for first in range(repeats) for second in range(first + 1, repeats)]
    types = np.asarray([_fault_type(value) for value in run_uids]); type_consistency = {}
    for kind in sorted(set(types) - {0}): type_consistency[str(kind)] = float(consistency[:, types == kind].mean())
    base_std = base.std(axis=(0, 2)); aug_std = flat.std(axis=(0, 2))
    spike_limit = np.maximum(10.0, np.abs(base).mean() + 10 * base.std())
    return {
        "teacher_consistency": float(consistency.mean()),
        "normal_consistency": float(consistency[:, normal].mean()), "fault_consistency": float(consistency[:, fault].mean()),
        "normal_to_fault_flip": float((generated_class[:, normal] == 1).mean()),
        "fault_to_normal_flip": float((generated_class[:, fault] == 0).mean()),
        "teacher_probability_kl": float(probability_kl.mean()), "teacher_feature_cosine": float(feature_cosine.mean()),
        "normalized_l1": float(normalized_l1.mean()), "normalized_l2": float(normalized_l2.mean()),
        "first_difference_distance": float(derivative.mean()),
        "feature_space_distance": float((1 - feature_cosine).mean()),
        "pairwise_diversity": float(np.mean(pairwise)) if pairwise else 0.0,
        "finite": bool(np.isfinite(augmented).all()), "amplitude_min": float(augmented.min()), "amplitude_max": float(augmented.max()),
        "channel_variance_ratio_mean": float(np.mean(aug_std / np.maximum(base_std, 1e-6))),
        "abnormal_spike_fraction": float(np.mean(np.abs(augmented) > spike_limit)),
        "fault_type_consistency": type_consistency,
    }


def audit(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config["random_seed"]); device = str(config["device"]); settings = config["generator"]
    seed_everything(seed); views, fixed_manifest = load_fixed_views(config); base = bases(views)
    indices = audit_indices(views, config["audit_subset_manifest"])
    teacher = freeze_teacher(build_model(config["teacher_model"], base["train"].shape[1], 2).to(device))
    teacher.load_state_dict(torch.load(config["teacher_checkpoint"], map_location=device, weights_only=True)); teacher.eval()
    teacher_semantic = {split: teacher_arrays(teacher, values, int(config["batch_size"]), device)[1] for split, values in base.items()}
    schedule = DiffusionSchedule.cosine(int(settings["diffusion_steps"]), device)
    template = SemanticPartialDiffusion1D(base["train"].shape[1], int(settings["semantic_dimension"]),
                                          int(settings["hidden_channels"]), int(settings["hidden_channels"]),
                                          int(settings["residual_blocks"])).to(device)
    initial_state = copy.deepcopy(template.state_dict()); initial_hash = _state_hash(initial_state)
    output = Path(config["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    models = {}; training = {}
    for name in ("G0", "G1"):
        seed_everything(seed); model = SemanticPartialDiffusion1D(
            base["train"].shape[1], int(settings["semantic_dimension"]), int(settings["hidden_channels"]),
            int(settings["hidden_channels"]), int(settings["residual_blocks"]),
        ).to(device); model.load_state_dict(initial_state)
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        history, seconds, peak = train_generator(name, model, teacher, views, base, teacher_semantic, schedule, config, device)
        torch.save(model.state_dict(), output / f"{name.lower()}_semantic_partial_diffusion.pt")
        models[name] = model; training[name] = {"history": history, "training_seconds": seconds, "peak_gpu_mib": peak}
    clip_min = base["train"].min(axis=(0, 2)); clip_max = base["train"].max(axis=(0, 2))
    audits: dict[str, Any] = {"train": {}, "validation": {}}
    for split in ("train", "validation"):
        chosen = indices[split]; base_subset = base[split][chosen]; bundle = views[split]
        base_probability, base_feature = teacher_arrays(teacher, base_subset, int(config["batch_size"]), device)
        traditional = traditional_augmentation(base_subset, bundle["window_id"][chosen], config["traditional_augmentation"], seed)
        traditional_metrics = augmentation_metrics(base_subset, traditional[None], bundle["labels"][chosen], bundle["run_uid"][chosen],
                                                   base_probability, base_feature, teacher, int(config["batch_size"]), device)
        audits[split]["traditional"] = traditional_metrics
        for name, model in models.items():
            audits[split][name] = {}
            for t_aug in settings["t_aug"]:
                generated = generate_repeats(model, base_subset, bundle["observation"][chosen], teacher_semantic[split][chosen],
                                             schedule, int(t_aug), int(settings["audit_repeats"]), int(config["batch_size"]),
                                             device, seed, clip_min, clip_max)
                audits[split][name][str(t_aug)] = augmentation_metrics(
                    base_subset, generated, bundle["labels"][chosen], bundle["run_uid"][chosen],
                    base_probability, base_feature, teacher, int(config["batch_size"]), device,
                )
    validation_g1 = audits["validation"]["G1"]; feasible = []
    for timestep, metrics in validation_g1.items():
        if (metrics["teacher_consistency"] >= float(config["gate"]["teacher_consistency"])
                and metrics["teacher_feature_cosine"] >= float(config["gate"]["feature_cosine"])
                and float(config["gate"]["minimum_normalized_l1"]) <= metrics["normalized_l1"] <= float(config["gate"]["maximum_normalized_l1"])
                and metrics["finite"] and metrics["abnormal_spike_fraction"] == 0):
            feasible.append(timestep)
    candidates = feasible or list(validation_g1)
    selected_t = max(candidates, key=lambda value: (validation_g1[value]["teacher_consistency"], validation_g1[value]["normalized_l1"]))
    g0 = audits["validation"]["G0"][selected_t]; g1 = validation_g1[selected_t]
    checks = {
        "g1_fault_consistency_improved": g1["fault_consistency"] >= g0["fault_consistency"] + float(config["gate"]["minimum_fault_consistency_gain"]),
        "g1_fault_flip_reduced": g1["fault_to_normal_flip"] < g0["fault_to_normal_flip"],
        "g1_nonzero_moderate_diversity": float(config["gate"]["minimum_normalized_l1"]) <= g1["normalized_l1"] <= float(config["gate"]["maximum_normalized_l1"]),
        "normal_fault_both_preserved": g1["normal_consistency"] >= .9 and g1["fault_consistency"] >= .9,
        "selected_t_in_feasible_region": selected_t in feasible,
    }
    passed = sum(checks.values()) >= 4 and checks["selected_t_in_feasible_region"]
    test_audit = {}
    chosen = indices["test"]; base_subset = base["test"][chosen]; bundle = views["test"]
    base_probability, base_feature = teacher_arrays(teacher, base_subset, int(config["batch_size"]), device)
    for name, model in models.items():
        generated = generate_repeats(model, base_subset, bundle["observation"][chosen], teacher_semantic["test"][chosen], schedule,
                                     int(selected_t), int(settings["audit_repeats"]), int(config["batch_size"]), device,
                                     seed, clip_min, clip_max)
        test_audit[name] = augmentation_metrics(base_subset, generated, bundle["labels"][chosen], bundle["run_uid"][chosen],
                                                base_probability, base_feature, teacher, int(config["batch_size"]), device)
    result = {"markers": config["markers"], "status": "SEMANTIC_DIFFUSION_FEASIBLE_REGION_GO" if passed else "SEMANTIC_DIFFUSION_AUGMENTATION_NO_GO",
              **environment_metadata(), "fixed_view_manifest": fixed_manifest, "teacher_checkpoint": config["teacher_checkpoint"],
              "initialization_sha256": initial_hash, "same_batch_order": True,
              "same_training_noise": True, "same_audit_sampling_noise": True,
              "semantic_injection": "observation mask at input; teacher embedding projected into every residual block",
              "training": training, "audits": audits, "selected_t_aug": int(selected_t), "feasible_timesteps": list(map(int, feasible)),
              "generator_checkpoints": {name: str(output / f"{name.lower()}_semantic_partial_diffusion.pt") for name in models},
              "semantic_gate_thresholds": {
                  "teacher_consistency": float(config["gate"]["teacher_consistency"]),
                  "feature_cosine": float(config["gate"]["feature_cosine"]),
                  "minimum_normalized_l1": float(config["gate"]["minimum_normalized_l1"]),
                  "maximum_normalized_l1": float(config["gate"]["maximum_normalized_l1"]),
              },
              "selection_split": "validation", "gate_checks": checks, "downstream_retest_allowed": passed,
              "test_selected_t_only": test_audit}
    write_json(Path(config["result_path"]), result); return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/semantic_diffusion_augmentation_audit.yaml")
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = audit(config); print(json.dumps({"status": result["status"], "selected_t": result["selected_t_aug"],
                                             "feasible": result["feasible_timesteps"], "checks": result["gate_checks"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
