from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from diffusion import DiffusionSchedule
from diffusion.fixed_views import sha256_file, sha256_strings
from diffusion.semantic_augmentation import SemanticPartialDiffusion1D
from losses import freeze_teacher
from scripts.audit_semantic_diffusion_augmentation import (
    augmentation_metrics,
    bases,
    generate_repeats,
    teacher_arrays,
    traditional_augmentation,
    train_generator,
)
from scripts.run_diffusion_quality_retest import epoch_orders, load_fixed_views, _state_hash
from scripts.run_semantic_diffusion_retest import _fit_method
from trainers import build_model
from utils import environment_metadata, seed_everything, write_json


METHODS = ("B0", "B1", "B2")
MARKERS = ("SEMANTIC_DIFFUSION_3SEED_VALIDATION", "FIXED_CONFIG", "SUBSET_DATA", "NOT_FOR_PAPER_FINAL_CLAIMS")


def configure_determinism(enabled: bool, cublas_workspace_config: str = ":4096:8") -> dict[str, Any]:
    if enabled:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_workspace_config
    torch.use_deterministic_algorithms(enabled)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = enabled
        torch.backends.cudnn.benchmark = False
    return {
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "known_limit": "CUDA/library kernels outside PyTorch may not be bitwise deterministic",
    }


def validate_frozen_config(config: dict[str, Any]) -> None:
    if list(map(int, config["seeds"])) != [7, 42, 2026]:
        raise ValueError("the frozen seeds must be [7, 42, 2026]")
    if tuple(config["run_methods"]) != METHODS or bool(config.get("run_b3", True)):
        raise ValueError("only B0/B1/B2 are allowed; B3 must remain disabled")
    if int(config["t_aug"]) != 5:
        raise ValueError("frozen t_aug must remain 5")
    generator = config["generator"]
    if float(generator["lambda_prob"]) != .1 or float(generator["lambda_feat"]) != .1:
        raise ValueError("semantic loss weights are frozen at 0.1")


def frozen_fingerprints(config_path: Path, config: dict[str, Any]) -> dict[str, str]:
    return {
        "config_sha256": sha256_file(config_path),
        "split_manifest_sha256": sha256_file(config["fixed_views"]["manifest"]),
        "teacher_checkpoint_sha256": sha256_file(config["teacher_checkpoint"]),
    }


def completed_result_is_reusable(result: dict[str, Any], seed: int, method: str,
                                  fingerprints: dict[str, str]) -> bool:
    return (int(result.get("seed", -1)) == seed and result.get("method") == method
            and result.get("fingerprints") == fingerprints
            and result.get("markers") == list(MARKERS)
            and result.get("status") == "COMPLETE")


def initialization_hash(config: dict[str, Any], channels: int, seed: int) -> str:
    seed_everything(seed)
    return _state_hash(build_model(config["model"], channels, 2).state_dict())


def _generator_model(config: dict[str, Any], channels: int, device: str) -> SemanticPartialDiffusion1D:
    settings = config["generator"]
    return SemanticPartialDiffusion1D(
        channels, int(settings["semantic_dimension"]), int(settings["hidden_channels"]),
        int(settings["hidden_channels"]), int(settings["residual_blocks"]),
    ).to(device)


def _save_generator_training(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != value:
            raise FileExistsError(f"refusing to overwrite generator metadata: {path}")
        return
    write_json(path, value)


def _train_or_load_generator(
    method: str, method_dir: Path, initial_state: dict[str, torch.Tensor],
    teacher: torch.nn.Module, views: dict[str, dict[str, np.ndarray]], base: dict[str, np.ndarray],
    semantic: dict[str, np.ndarray], schedule: DiffusionSchedule, runtime: dict[str, Any], device: str,
) -> tuple[SemanticPartialDiffusion1D, dict[str, Any]]:
    checkpoint = method_dir / "generator.pt"
    metadata_path = method_dir / "generator_training.json"
    model = _generator_model(runtime, base["train"].shape[1], device)
    if checkpoint.exists() and metadata_path.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        return model.eval(), json.loads(metadata_path.read_text(encoding="utf-8"))
    if checkpoint.exists() or metadata_path.exists():
        raise RuntimeError(f"incomplete generator state requires manual inspection: {method_dir}")
    model.load_state_dict(initial_state)
    name = "G0" if method == "B1" else "G1"
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    history, seconds, peak = train_generator(
        name, model, teacher, views, base, semantic, schedule, runtime, device,
    )
    torch.save(model.state_dict(), checkpoint)
    metadata = {"name": name, "history": history, "training_seconds": seconds, "peak_gpu_mib": peak}
    _save_generator_training(metadata_path, metadata)
    return model.eval(), metadata


def run(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_frozen_config(config)
    fingerprints = frozen_fingerprints(config_path, config)
    deterministic = configure_determinism(bool(config["deterministic_algorithms"]), str(config["cublas_workspace_config"]))
    device = str(config["device"])
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    views, _ = load_fixed_views(config)
    base = bases(views)
    teacher = freeze_teacher(build_model(config["teacher_model"], base["train"].shape[1], 2).to(device))
    teacher.load_state_dict(torch.load(config["teacher_checkpoint"], map_location=device, weights_only=True))
    teacher.eval()
    semantic = {split: teacher_arrays(teacher, values, int(config["batch_size"]), device)[1]
                for split, values in base.items()}
    test_probability, test_feature = teacher_arrays(teacher, base["test"], int(config["batch_size"]), device)
    schedule = DiffusionSchedule.cosine(int(config["generator"]["diffusion_steps"]), device)
    clip_min, clip_max = base["train"].min((0, 2)), base["train"].max((0, 2))
    completed: dict[str, dict[str, Any]] = {}

    for seed in map(int, config["seeds"]):
        runtime = copy.deepcopy(config)
        runtime["random_seed"] = seed
        pretrain_orders = epoch_orders(len(base["train"]), int(config["epochs"]), seed + 10_000)
        probe_orders = epoch_orders(len(base["train"]), int(config["probe_epochs"]), seed + 20_000)
        pretrain_hash = sha256_strings([",".join(map(str, value.tolist())) for value in pretrain_orders])
        probe_hash = sha256_strings([",".join(map(str, value.tolist())) for value in probe_orders])
        seed_everything(seed)
        classifier_template = build_model(config["model"], base["train"].shape[1], 2)
        classifier_state = copy.deepcopy(classifier_template.state_dict())
        classifier_hash = _state_hash(classifier_state)
        seed_everything(seed)
        generator_template = _generator_model(config, base["train"].shape[1], device)
        generator_state = copy.deepcopy(generator_template.state_dict())
        generator_hash = _state_hash(generator_state)

        for method in METHODS:
            method_dir = output / f"seed_{seed}" / method
            method_dir.mkdir(parents=True, exist_ok=True)
            result_path = method_dir / "result.json"
            if result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if not completed_result_is_reusable(result, seed, method, fingerprints):
                    raise RuntimeError(f"completed result fingerprint mismatch: {result_path}")
                completed[f"{seed}:{method}"] = result
                continue

            generator_training = None
            augmented: dict[str, np.ndarray] = {}
            if method == "B0":
                for split in ("train", "validation", "test"):
                    augmented[split] = traditional_augmentation(
                        base[split], views[split]["window_id"], config["traditional_augmentation"], seed,
                    )
            else:
                generator, generator_training = _train_or_load_generator(
                    method, method_dir, generator_state, teacher, views, base, semantic, schedule, runtime, device,
                )
                for split in ("train", "validation", "test"):
                    augmented[split] = generate_repeats(
                        generator, base[split], views[split]["observation"], semantic[split], schedule,
                        int(config["t_aug"]), 1, int(config["batch_size"]), device, seed, clip_min, clip_max,
                    )[0]

            classifier_checkpoint = method_dir / "classifier.pt"
            result = _fit_method(
                method, augmented, base, views, classifier_state, pretrain_orders, probe_orders,
                runtime, device, classifier_checkpoint,
            )
            result["augmentation_audit"] = augmentation_metrics(
                base["test"], augmented["test"][None], views["test"]["labels"], views["test"]["run_uid"],
                test_probability, test_feature, teacher, int(config["batch_size"]), device,
            )
            result.update({
                "status": "COMPLETE", "markers": list(MARKERS), "seed": seed, "method": method,
                "fingerprints": fingerprints, "determinism": deterministic,
                "fairness": {"classifier_initialization_sha256": classifier_hash,
                             "generator_initialization_sha256": generator_hash if method != "B0" else None,
                             "pretrain_batch_order_sha256": pretrain_hash, "probe_batch_order_sha256": probe_hash,
                             "sample_total_weight": 1.0, "test_used_for_selection": False},
                "generator_training": generator_training, **environment_metadata(),
            })
            if not np.isfinite([result["metrics"][key] for key in ("macro_f1", "auprc", "fault_recall", "far", "auroc")]).all():
                raise FloatingPointError(f"non-finite metrics for seed={seed}, method={method}")
            write_json(result_path, result)
            completed[f"{seed}:{method}"] = result

    from scripts.summarize_semantic_diffusion_3seed import summarize
    summary = summarize(config_path, config, completed, fingerprints)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/semantic_diffusion_3seed.yaml")
    args = parser.parse_args()
    summary = run(Path(args.config))
    print(json.dumps({"status": summary["status"], "counts": summary["direction_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
