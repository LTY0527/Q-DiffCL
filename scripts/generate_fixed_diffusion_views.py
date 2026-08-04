from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from diffusion import DiffusionSchedule
from diffusion.fixed_views import (SPLITS, mask_id, per_sample_masked_mae,
                                   sha256_file, sha256_strings, split_window_id,
                                   validate_view_splits)
from models import MinimalConditionalDiffusion1D
from scripts.train_diffusion_recovery import prepare_bundles, restore_array
from utils import environment_metadata, seed_everything, write_json


def _load_manifest(path: Path) -> dict[str, list[str]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return {split: list(value[split]) for split in SPLITS}


def _metadata(ids: list[str], observations: np.ndarray) -> dict[str, np.ndarray]:
    parsed = [split_window_id(value) for value in ids]
    return {
        "window_id": np.asarray(ids),
        "run_uid": np.asarray([value[0] for value in parsed]),
        "start_sample": np.asarray([value[1] for value in parsed], dtype=np.int64),
        "end_sample": np.asarray([value[2] for value in parsed], dtype=np.int64),
        "clean_index": np.arange(len(ids), dtype=np.int64),
        "mask_id": np.asarray([mask_id(value) for value in observations]),
    }


def generate(config: dict[str, Any]) -> dict[str, Any]:
    fixed = config["fixed_views"]
    output = Path(fixed["output_dir"])
    paths = {split: output / f"{split}_views.npz" for split in SPLITS}
    manifest_path = Path(fixed["manifest"])
    tracked_manifest = Path(fixed["tracked_manifest"])
    existing = [str(path) for path in (*paths.values(), manifest_path, tracked_manifest) if path.exists()]
    if existing:
        raise FileExistsError("fixed views are immutable; refusing to overwrite: " + ", ".join(existing))

    checkpoint = Path(fixed["checkpoint"])
    source_manifest_path = Path(fixed["source_manifest"])
    if not checkpoint.is_file() or not source_manifest_path.is_file():
        raise FileNotFoundError("fixed checkpoint or source split manifest is missing")
    source_manifest = _load_manifest(source_manifest_path)

    seed = int(config["random_seed"])
    seed_everything(seed)
    started = time.perf_counter()
    bundles, generated_manifest, _ = prepare_bundles(config)
    generated = {split: list(getattr(generated_manifest, split)) for split in SPLITS}
    if generated != source_manifest:
        raise RuntimeError("generated data split does not exactly match the frozen source manifest")

    device = str(config["device"])
    channels = bundles["train"]["clean"].shape[1]
    model = MinimalConditionalDiffusion1D(
        channels, int(fixed["hidden_channels"]), int(fixed["hidden_channels"]),
        int(fixed["residual_blocks"]),
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    schedule = DiffusionSchedule.cosine(int(fixed["diffusion_steps"]), device)
    clip_min = bundles["train"]["clean"].min(axis=(0, 2)).astype(np.float32)
    clip_max = bundles["train"]["clean"].max(axis=(0, 2)).astype(np.float32)
    output.mkdir(parents=True, exist_ok=True)

    views: dict[str, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    sampling_seed = seed + int(fixed["sampling_seed_offset"])
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    for split in SPLITS:
        bundle = bundles[split]
        split_started = time.perf_counter()
        restored = restore_array(
            model, bundle["degraded"], bundle["observation"], schedule,
            int(config["batch_size"]), device, sampling_seed, clip_min, clip_max,
        ).astype(np.float32)
        metadata = _metadata(bundle["ids"], bundle["observation"])
        payload = {
            "clean": bundle["clean"].astype(np.float32),
            "degraded": bundle["degraded"].astype(np.float32),
            "restored": restored,
            "observation": bundle["observation"].astype(bool),
            "labels": bundle["labels"].astype(np.int64),
            "clip_min": clip_min, "clip_max": clip_max,
            **metadata,
        }
        np.savez_compressed(paths[split], **payload)
        error = per_sample_masked_mae(payload["clean"], restored, payload["observation"])
        views[split] = payload
        summaries[split] = {
            "path": paths[split].as_posix(), "sha256": sha256_file(paths[split]),
            "count": len(restored), "shape": list(restored.shape),
            "run_count": len(set(map(str, metadata["run_uid"]))),
            "run_uids": sorted(set(map(str, metadata["run_uid"]))),
            "window_ids_sha256": sha256_strings(list(map(str, metadata["window_id"]))),
            "mask_ids_sha256": sha256_strings(list(map(str, metadata["mask_id"]))),
            "actual_missing_ratio": float((~payload["observation"]).mean()),
            "masked_mae": float(error.mean()),
            "generation_seconds": time.perf_counter() - split_started,
        }
    validate_view_splits(views, source_manifest)

    result = {
        "markers": config["markers"], "status": "FIXED_DIFFUSION_VIEWS_FROZEN",
        **environment_metadata(), "checkpoint": checkpoint.as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint),
        "source_manifest": source_manifest_path.as_posix(),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "seed": seed, "sampling_seed": sampling_seed,
        "degradation": config["degradation"], "mask_ratio": float(config["degradation_severity"]),
        "diffusion_steps": int(fixed["diffusion_steps"]),
        "clip_min_sha256": sha256_strings([format(float(x), ".9g") for x in clip_min]),
        "clip_max_sha256": sha256_strings([format(float(x), ".9g") for x in clip_max]),
        "splits": summaries, "total_seconds": time.perf_counter() - started,
        "peak_gpu_mib": torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0,
    }
    write_json(manifest_path, result)
    write_json(tracked_manifest, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/diffusion_quality_retest.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = generate(config)
    print(json.dumps({"status": result["status"], "splits": {k: v["count"] for k, v in result["splits"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
