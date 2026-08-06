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
from diffusion.semantic_augmentation import SemanticPartialDiffusion1D, partial_diffusion_objective
from losses import freeze_teacher, semantic_consistency_losses
from scripts.audit_semantic_diffusion_augmentation import (
    audit_indices, augmentation_metrics, bases, generate_repeats, teacher_arrays,
)
from scripts.run_diffusion_quality_retest import epoch_orders, load_fixed_views
from trainers import build_model
from utils import environment_metadata, seed_everything, write_json


def gradient_norm(loss: torch.Tensor, parameters: list[torch.nn.Parameter], retain_graph: bool = True) -> float:
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    squares = [gradient.detach().square().sum() for gradient in gradients if gradient is not None]
    return float(torch.stack(squares).sum().sqrt()) if squares else 0.0


def configure_determinism(config: dict[str, Any]) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(config["cublas_workspace_config"])
    enabled = bool(config["deterministic_algorithms"])
    torch.use_deterministic_algorithms(enabled)
    torch.backends.cudnn.deterministic = enabled
    torch.backends.cudnn.benchmark = False


def diagnose(config: dict[str, Any]) -> dict[str, Any]:
    configure_determinism(config)
    device = str(config["device"]); settings = config["generator"]
    views, manifest = load_fixed_views(config); base = bases(views)
    selected = audit_indices(views, config["validation_subset_manifest"])["validation"]
    teacher = freeze_teacher(build_model(config["teacher_model"], base["train"].shape[1], 2).to(device))
    teacher.load_state_dict(torch.load(config["teacher_checkpoint"], map_location=device, weights_only=True))
    semantic = {split: teacher_arrays(teacher, values, int(config["batch_size"]), device)[1]
                for split, values in base.items()}
    val_base = base["validation"][selected]
    val_probability, val_feature = teacher_arrays(teacher, val_base, int(config["batch_size"]), device)
    schedule = DiffusionSchedule.cosine(int(settings["diffusion_steps"]), device)
    clip_min, clip_max = base["train"].min((0, 2)), base["train"].max((0, 2))
    output = Path(config["diagnosis_output_dir"]); output.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}

    for seed in map(int, config["seeds"]):
        path = output / f"seed_{seed}" / "result.json"
        if path.exists():
            results[str(seed)] = json.loads(path.read_text(encoding="utf-8")); continue
        seed_everything(seed)
        model = SemanticPartialDiffusion1D(
            base["train"].shape[1], int(settings["semantic_dimension"]), int(settings["hidden_channels"]),
            int(settings["hidden_channels"]), int(settings["residual_blocks"]),
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(settings["learning_rate"]))
        orders = epoch_orders(len(base["train"]), int(settings["old_epochs"]), seed + 30_000)
        history = []; parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        for epoch, order in enumerate(orders):
            started = time.perf_counter(); model.train()
            values = {key: [] for key in ("total", "diff", "prob", "feat")}
            norms = None; histogram = np.zeros(int(settings["old_train_max_timestep"]) + 1, dtype=np.int64)
            generator = torch.Generator(device=device).manual_seed(seed + 40_000 + epoch)
            for start in range(0, len(order), int(config["batch_size"])):
                indices = order[start:start + int(config["batch_size"])]
                base_b = torch.from_numpy(base["train"][indices]).to(device)
                observation_b = torch.from_numpy(views["train"]["observation"][indices]).to(device)
                semantic_b = torch.from_numpy(semantic["train"][indices]).to(device)
                timesteps = torch.randint(0, int(settings["old_train_max_timestep"]) + 1,
                                           (len(indices),), device=device, generator=generator)
                histogram += np.bincount(timesteps.cpu().numpy(), minlength=len(histogram))
                noise = torch.randn(base_b.shape, device=device, generator=generator)
                optimizer.zero_grad()
                diff, predicted = partial_diffusion_objective(
                    model, schedule, base_b, observation_b, semantic_b, timesteps, noise,
                )
                prob, feat = semantic_consistency_losses(teacher, base_b, predicted)
                total = diff + .1 * prob + .1 * feat
                if norms is None:
                    norms = {"diff": gradient_norm(diff, parameters), "prob_raw": gradient_norm(prob, parameters),
                             "feat_raw": gradient_norm(feat, parameters),
                             "prob_weighted": gradient_norm(.1 * prob, parameters),
                             "feat_weighted": gradient_norm(.1 * feat, parameters)}
                total.backward(); optimizer.step()
                for key, value in (("total", total), ("diff", diff), ("prob", prob), ("feat", feat)):
                    values[key].append(float(value.detach()))
            model.eval()
            generated = generate_repeats(
                model, val_base, views["validation"]["observation"][selected], semantic["validation"][selected],
                schedule, int(settings["t_aug"]), 1, int(config["batch_size"]), device,
                seed + int(config["validation"]["sampling_seed_offset"]), clip_min, clip_max,
            )
            metrics = augmentation_metrics(
                val_base, generated, views["validation"]["labels"][selected], views["validation"]["run_uid"][selected],
                val_probability, val_feature, teacher, int(config["batch_size"]), device,
            )
            history.append({"epoch": epoch, **{key: float(np.mean(value)) for key, value in values.items()},
                            "gradient_norms_first_batch": norms, "timestep_histogram": histogram.tolist(),
                            "validation": metrics, "epoch_seconds": time.perf_counter() - started})
        result = {"markers": config["markers"], "mode": "OLD_GENERATOR_ONLY_DIAGNOSIS", "seed": seed,
                  "old_logs_missing": ["per_epoch_validation", "component_gradient_norms"],
                  "history": history, "peak_gpu_mib": torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0,
                  "fixed_view_manifest": manifest, **environment_metadata()}
        write_json(path, result); results[str(seed)] = result
    combined = {"markers": config["markers"], "results": results}
    write_json(output / "summary.json", combined); return combined


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/semantic_generator_stability_fix.yaml")
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = diagnose(config); print(json.dumps({"seeds": list(result["results"])}, ensure_ascii=False))


if __name__ == "__main__": main()
