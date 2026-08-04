from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from diffusion import DiffusionSchedule
from diffusion.candidate_generation import (candidate_seed_matrix,
                                            restore_candidates,
                                            select_balanced_audit_indices,
                                            validate_shared_context)
from diffusion.fixed_views import SPLITS, sha256_file, sha256_strings
from models import MinimalConditionalDiffusion1D
from scripts.run_diffusion_quality_retest import load_fixed_views
from utils import environment_metadata, seed_everything, write_json


def _fault_type(run_uid: str) -> int:
    match = re.search(r":fault_(\d+):", str(run_uid)); return int(match.group(1)) if match else 0


def validate_candidate_trace(records: list[dict[str, Any]], k: int) -> None:
    required = {"split", "run_uid", "window_id", "mask_id", "candidate_seeds"}
    if any(not required.issubset(record) for record in records): raise ValueError("candidate trace is incomplete")
    if any(len(record["candidate_seeds"]) != k for record in records): raise ValueError("candidate seed trace length mismatch")
    keys = [(record["split"], record["window_id"]) for record in records]
    if len(keys) != len(set(keys)): raise ValueError("duplicate candidate window trace")


def generate(config: dict[str, Any]) -> dict[str, Any]:
    output = Path(config["output_dir"] if "output_dir" in config else config["candidate_dir"]); output.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(config["manifest"] if "manifest" in config else config["candidate_manifest"])
    tracked_manifest = Path(config["tracked_manifest"])
    paths = {split: output / f"{split}_candidates.npz" for split in SPLITS}
    existing = [str(path) for path in (*paths.values(), manifest_path, tracked_manifest) if path.exists()]
    if existing: raise FileExistsError("candidate archives are immutable; refusing to overwrite: " + ", ".join(existing))
    views, fixed_manifest = load_fixed_views(config)
    checkpoint = Path(config["checkpoint"]); device = str(config["device"]); seed = int(config["random_seed"])
    seed_everything(seed); started = time.perf_counter()
    model = MinimalConditionalDiffusion1D(
        views["train"]["clean"].shape[1], int(config["diffusion"]["hidden_channels"]),
        int(config["diffusion"]["hidden_channels"]), int(config["diffusion"]["residual_blocks"]),
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)); model.eval()
    schedule = DiffusionSchedule.cosine(int(config["diffusion"]["steps"]), device)
    clip_min = views["train"]["clip_min"]; clip_max = views["train"]["clip_max"]
    k = int(config["k_candidates"]); summaries = {}; all_records = []
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    for split in SPLITS:
        split_started = time.perf_counter(); bundle = views[split]
        if "subset_counts" in config:
            indices = select_balanced_audit_indices(bundle["labels"], int(config["subset_counts"][split]), seed, split)
        else:
            indices = np.arange(len(bundle["labels"]), dtype=np.int64)
        window_ids = list(map(str, bundle["window_id"][indices])); seeds = candidate_seed_matrix(seed, split, window_ids, k)
        candidates = restore_candidates(
            model, bundle["degraded"][indices], bundle["observation"][indices], seeds, schedule,
            int(config["batch_size"]), device, clip_min, clip_max,
        )
        validate_shared_context(candidates, bundle["degraded"][indices], bundle["observation"][indices])
        np.savez_compressed(paths[split], candidates=candidates, fixed_indices=indices, candidate_seeds=seeds)
        records = [{"split": split, "run_uid": str(bundle["run_uid"][index]),
                    "window_id": str(bundle["window_id"][index]), "mask_id": str(bundle["mask_id"][index]),
                    "candidate_ids": list(range(k)), "candidate_seeds": [int(value) for value in seeds[position]]}
                   for position, index in enumerate(indices)]
        validate_candidate_trace(records, k); all_records.extend(records)
        types, counts = np.unique([_fault_type(value) for value in bundle["run_uid"][indices]], return_counts=True)
        summaries[split] = {
            "path": paths[split].as_posix(), "sha256": sha256_file(paths[split]),
            "count": int(len(indices)), "k": k, "shape": list(candidates.shape),
            "fixed_indices_sha256": sha256_strings(list(map(str, indices))),
            "window_ids_sha256": sha256_strings(window_ids),
            "mask_ids_sha256": sha256_strings(list(map(str, bundle["mask_id"][indices]))),
            "candidate_seeds_sha256": sha256_strings([",".join(map(str, row)) for row in seeds]),
            "class_counts": np.bincount(bundle["labels"][indices], minlength=2).tolist(),
            "fault_type_counts": {str(int(key)): int(value) for key, value in zip(types, counts)},
            "run_uids": sorted(set(map(str, bundle["run_uid"][indices]))),
            "generation_seconds": time.perf_counter() - split_started,
        }
    validate_candidate_trace(all_records, k)
    run_sets = {split: set(summaries[split]["run_uids"]) for split in SPLITS}
    if any(run_sets[a] & run_sets[b] for index, a in enumerate(SPLITS) for b in SPLITS[index + 1:]):
        raise RuntimeError("candidate subset run leakage")
    result = {
        "markers": config["markers"], "status": "INTRA_SAMPLE_CANDIDATES_FROZEN", **environment_metadata(),
        "checkpoint": checkpoint.as_posix(), "checkpoint_sha256": sha256_file(checkpoint),
        "fixed_view_manifest_sha256": sha256_file(config["fixed_views"]["manifest"]),
        "master_seed": seed, "candidate_seed_rule": "sha256(master|split|window_id|candidate_id|ddpm_candidate)",
        "diffusion_steps": int(config["diffusion"]["steps"]), "k": k,
        "splits": summaries, "records": all_records,
        "total_seconds": time.perf_counter() - started,
        "peak_gpu_mib": torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0,
    }
    write_json(manifest_path, result); write_json(tracked_manifest, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True)
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = generate(config)
    print(json.dumps({"status": result["status"], "counts": {key: value["count"] for key, value in result["splits"].items()},
                      "k": result["k"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
