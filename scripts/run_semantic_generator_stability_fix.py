from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from diffusion import DiffusionSchedule
from diffusion.semantic_augmentation import (
    SemanticPartialDiffusion1D, partial_diffusion_objective, sample_training_timesteps,
)
from losses import balanced_semantic_consistency_loss, freeze_teacher
from scripts.audit_semantic_diffusion_augmentation import (
    audit_indices, augmentation_metrics, bases, generate_repeats, teacher_arrays,
)
from scripts.diagnose_semantic_generator_seeds import gradient_norm
from scripts.run_diffusion_quality_retest import epoch_orders, load_fixed_views, _state_hash
from trainers import build_model
from utils import environment_metadata, seed_everything, write_json


def update_ema(ema_state: dict[str, torch.Tensor], state: dict[str, torch.Tensor], decay: float) -> None:
    with torch.no_grad():
        for name, value in state.items():
            if torch.is_floating_point(value):
                ema_state[name].mul_(decay).add_(value.detach(), alpha=1 - decay)
            else:
                ema_state[name].copy_(value)


def diversity_penalty(value: float, minimum: float, maximum: float) -> float:
    return max(0.0, minimum - value) + max(0.0, value - maximum)


def generator_score(metrics: dict[str, float], config: dict[str, Any]) -> float:
    limits = config["validation"]; weights = limits["score_weights"]
    balanced = .5 * metrics["normal_to_fault_flip"] + .5 * metrics["fault_to_normal_flip"]
    penalty = diversity_penalty(metrics["normalized_l1"], float(limits["normalized_l1_minimum"]),
                                float(limits["normalized_l1_maximum"]))
    return (float(weights["balanced_flip_rate"]) * balanced
            + float(weights["probability_distance"]) * metrics["teacher_probability_kl"]
            + float(weights["diversity_penalty"]) * penalty)


def select_best_candidate(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("candidate records must not be empty")
    return min(records, key=lambda row: (row["score"], row["epoch"], row["variant"]))


def _configure(config: dict[str, Any]) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(config["cublas_workspace_config"])
    enabled = bool(config["deterministic_algorithms"])
    torch.use_deterministic_algorithms(enabled)
    torch.backends.cudnn.deterministic = enabled; torch.backends.cudnn.benchmark = False


def _model(config: dict[str, Any], channels: int, device: str) -> SemanticPartialDiffusion1D:
    settings = config["generator"]
    return SemanticPartialDiffusion1D(channels, int(settings["semantic_dimension"]),
                                      int(settings["hidden_channels"]), int(settings["hidden_channels"]),
                                      int(settings["residual_blocks"])).to(device)


def _validation_losses(model, teacher, schedule, base, observation, semantic, labels,
                       seed: int, config: dict[str, Any], method: str) -> dict[str, float]:
    generator = torch.Generator(device=base.device).manual_seed(seed + 88000)
    timestep = torch.full((len(base),), int(config["generator"]["t_aug"]), device=base.device, dtype=torch.long)
    noise = torch.randn(base.shape, device=base.device, generator=generator)
    with torch.enable_grad():
        diff, predicted = partial_diffusion_objective(model, schedule, base, observation, semantic, timestep, noise)
        sem = balanced_semantic_consistency_loss(teacher, base, predicted, labels)["total"] if method == "G1-fixed" else diff.new_zeros(())
    return {"diff": float(diff.detach()), "semantic": float(sem.detach())}


def train_candidate(
    method: str, seed: int, lambda_sem: float, candidate_dir: Path,
    config: dict[str, Any], views: dict[str, dict[str, np.ndarray]], base: dict[str, np.ndarray],
    semantic: dict[str, np.ndarray], teacher: torch.nn.Module, schedule: DiffusionSchedule,
    selected: np.ndarray, initial_state: dict[str, torch.Tensor], device: str,
) -> dict[str, Any]:
    result_path = candidate_dir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    candidate_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed); model = _model(config, base["train"].shape[1], device); model.load_state_dict(initial_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["generator"]["learning_rate"]))
    ema_state = copy.deepcopy(model.state_dict()); decay = float(config["generator"]["ema"]["decay"])
    orders = epoch_orders(len(base["train"]), int(config["generator"]["max_epochs"]), seed + 30_000)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    history = []; candidates = []; best_score = float("inf"); stale = 0
    val_base_np = base["validation"][selected]
    val_base = torch.from_numpy(val_base_np).to(device)
    val_observation = torch.from_numpy(views["validation"]["observation"][selected]).to(device)
    val_semantic = torch.from_numpy(semantic["validation"][selected]).to(device)
    val_labels = torch.from_numpy(views["validation"]["labels"][selected]).long().to(device)
    val_probability, val_feature = teacher_arrays(teacher, val_base_np, int(config["batch_size"]), device)
    clip_min, clip_max = base["train"].min((0, 2)), base["train"].max((0, 2))
    started_all = time.perf_counter()
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()

    for epoch, order in enumerate(orders):
        epoch_started = time.perf_counter(); model.train()
        values = {key: [] for key in ("total", "diff", "semantic", "js", "margin", "feature")}
        norms = None; composition = {"normal": 0, "fault": 0}; histogram = {str(step): 0 for step in config["generator"]["train_timesteps"]}
        generator = torch.Generator(device=device).manual_seed(seed + 40_000 + epoch)
        for start in range(0, len(order), int(config["batch_size"])):
            indices = order[start:start + int(config["batch_size"])]
            base_b = torch.from_numpy(base["train"][indices]).to(device)
            observation_b = torch.from_numpy(views["train"]["observation"][indices]).to(device)
            semantic_b = torch.from_numpy(semantic["train"][indices]).to(device)
            labels_b = torch.from_numpy(views["train"]["labels"][indices]).long().to(device)
            timestep = sample_training_timesteps(config["generator"]["train_timesteps"], len(indices), generator, device)
            for step, count in zip(*torch.unique(timestep, return_counts=True)):
                histogram[str(int(step))] += int(count)
            noise = torch.randn(base_b.shape, device=device, generator=generator)
            optimizer.zero_grad(); diff, predicted = partial_diffusion_objective(
                model, schedule, base_b, observation_b, semantic_b, timestep, noise,
            )
            if method == "G1-fixed":
                sem = balanced_semantic_consistency_loss(teacher, base_b, predicted, labels_b)
            else:
                zero = diff.new_zeros(())
                sem = {"total": zero, "js": zero, "margin": zero, "feature": zero,
                       "normal_count": int((labels_b == 0).sum()), "fault_count": int((labels_b == 1).sum())}
            semantic_loss = sem["total"]
            total = diff + float(lambda_sem) * semantic_loss
            composition["normal"] += sem["normal_count"]; composition["fault"] += sem["fault_count"]
            if norms is None:
                norms = {"diff": gradient_norm(diff, parameters), "semantic_raw": 0.0,
                         "semantic_weighted": 0.0, "js": 0.0, "margin": 0.0, "feature": 0.0}
                if method == "G1-fixed":
                    norms.update({"semantic_raw": gradient_norm(sem["total"], parameters),
                                  "semantic_weighted": gradient_norm(float(lambda_sem) * sem["total"], parameters),
                                  "js": gradient_norm(sem["js"], parameters),
                                  "margin": gradient_norm(sem["margin"], parameters),
                                  "feature": gradient_norm(sem["feature"], parameters)})
            total.backward(); optimizer.step()
            if bool(config["generator"]["ema"]["enabled"]): update_ema(ema_state, model.state_dict(), decay)
            for key, value in (("total", total), ("diff", diff), ("semantic", semantic_loss),
                               ("js", sem["js"]), ("margin", sem["margin"]), ("feature", sem["feature"])):
                values[key].append(float(value.detach()))

        epoch_record = {"epoch": epoch, **{key: float(np.mean(value)) for key, value in values.items()},
                        "gradient_norms_first_batch": norms, "class_composition": composition,
                        "timestep_histogram": histogram, "epoch_seconds": time.perf_counter() - epoch_started,
                        "validation": {}}
        state_variants = {"raw": copy.deepcopy(model.state_dict()), "ema": copy.deepcopy(ema_state)}
        for variant, state in state_variants.items():
            evaluation_model = _model(config, base["train"].shape[1], device); evaluation_model.load_state_dict(state); evaluation_model.eval()
            losses = _validation_losses(evaluation_model, teacher, schedule, val_base, val_observation,
                                        val_semantic, val_labels, seed, config, method)
            for alpha in config["generator"]["alpha_candidates"]:
                generated = generate_repeats(
                    evaluation_model, val_base_np, views["validation"]["observation"][selected],
                    semantic["validation"][selected], schedule, int(config["generator"]["t_aug"]), 1,
                    int(config["batch_size"]), device, seed + int(config["validation"]["sampling_seed_offset"]),
                    clip_min, clip_max, float(alpha),
                )
                metrics = augmentation_metrics(
                    val_base_np, generated, views["validation"]["labels"][selected], views["validation"]["run_uid"][selected],
                    val_probability, val_feature, teacher, int(config["batch_size"]), device,
                )
                metrics["balanced_flip_rate"] = .5 * metrics["normal_to_fault_flip"] + .5 * metrics["fault_to_normal_flip"]
                record = {"epoch": epoch, "variant": variant, "alpha": float(alpha), "metrics": metrics,
                          "validation_losses": losses, "score": generator_score(metrics, config), "state": state}
                candidates.append(record)
                epoch_record["validation"][f"{variant}_alpha_{alpha}"] = {key: value for key, value in record.items() if key != "state"}
        history.append(epoch_record)
        best_epoch_record = select_best_candidate([row for row in candidates if row["epoch"] == epoch])
        if best_epoch_record["score"] < best_score - 1e-8:
            best_score, stale = best_epoch_record["score"], 0
        else:
            stale += 1
            if stale >= int(config["generator"]["early_stopping_patience"]): break

    best = select_best_candidate(candidates)
    torch.save(best["state"], candidate_dir / "best_generator.pt")
    best_raw = select_best_candidate([row for row in candidates if row["variant"] == "raw"])
    torch.save(best_raw["state"], candidate_dir / "best_non_ema_generator.pt")
    torch.save(model.state_dict(), candidate_dir / "last_generator.pt")
    serial_best = {key: value for key, value in best.items() if key != "state"}
    serial_raw = {key: value for key, value in best_raw.items() if key != "state"}
    last_candidates = [row for row in candidates if row["epoch"] == history[-1]["epoch"]]
    last = {key: value for key, value in select_best_candidate(last_candidates).items() if key != "state"}
    best_by_alpha = {}; last_by_alpha = {}
    for alpha in config["generator"]["alpha_candidates"]:
        alpha_key = str(float(alpha))
        alpha_best = select_best_candidate([row for row in candidates if row["alpha"] == float(alpha)])
        alpha_last = select_best_candidate([row for row in last_candidates if row["alpha"] == float(alpha)])
        torch.save(alpha_best["state"], candidate_dir / f"best_alpha_{alpha}.pt")
        best_by_alpha[alpha_key] = {key: value for key, value in alpha_best.items() if key != "state"}
        last_by_alpha[alpha_key] = {key: value for key, value in alpha_last.items() if key != "state"}
    result = {"markers": config["markers"], "status": "COMPLETE", "seed": seed, "method": method,
              "lambda_sem": float(lambda_sem), "best": serial_best, "best_non_ema": serial_raw, "last": last,
              "best_by_alpha": best_by_alpha, "last_by_alpha": last_by_alpha,
              "history": history, "initialization_sha256": _state_hash(initial_state),
              "training_seconds": time.perf_counter() - started_all,
              "peak_gpu_mib": torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0,
              **environment_metadata()}
    write_json(result_path, result); return result


def run(config: dict[str, Any]) -> dict[str, Any]:
    _configure(config); device = str(config["device"])
    views, _ = load_fixed_views(config); base = bases(views)
    selected = audit_indices(views, config["validation_subset_manifest"])["validation"]
    teacher = freeze_teacher(build_model(config["teacher_model"], base["train"].shape[1], 2).to(device))
    teacher.load_state_dict(torch.load(config["teacher_checkpoint"], map_location=device, weights_only=True))
    semantic = {split: teacher_arrays(teacher, values, int(config["batch_size"]), device)[1] for split, values in base.items()}
    schedule = DiffusionSchedule.cosine(int(config["generator"]["diffusion_steps"]), device)
    output = Path(config["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    results = {}; selected_lambda = None; selected_alpha = None
    for seed in map(int, config["seeds"]):
        seed_everything(seed); template = _model(config, base["train"].shape[1], device); initial_state = copy.deepcopy(template.state_dict())
        g0 = train_candidate("G0", seed, 0.0, output / f"seed_{seed}" / "G0", config, views, base,
                             semantic, teacher, schedule, selected, initial_state, device)
        results[f"{seed}:G0"] = g0
        lambdas = config["generator"]["lambda_candidates"] if seed == int(config["seeds"][0]) else [selected_lambda]
        seed_candidates = []
        for value in lambdas:
            result = train_candidate("G1-fixed", seed, float(value), output / f"seed_{seed}" / f"G1_lambda_{value}",
                                     config, views, base, semantic, teacher, schedule, selected, initial_state, device)
            seed_candidates.append(result)
        if selected_lambda is None:
            choices = [(item["best_by_alpha"][str(float(alpha))]["score"], item, float(alpha))
                       for item in seed_candidates for alpha in config["generator"]["alpha_candidates"]]
            _, chosen, selected_alpha = min(choices, key=lambda row: (row[0], row[1]["lambda_sem"], row[2]))
            selected_lambda = chosen["lambda_sem"]
        chosen = next(item for item in seed_candidates if item["lambda_sem"] == selected_lambda)
        results[f"{seed}:G1-fixed"] = chosen
    from scripts.summarize_semantic_generator_stability import summarize
    return summarize(config, results, selected_lambda, selected_alpha)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/semantic_generator_stability_fix.yaml")
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = run(config); print(json.dumps({"status": result["status"], "selected": result["selected_configuration"]}, ensure_ascii=False))


if __name__ == "__main__": main()
