from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from diffusion import DiffusionSchedule, ddpm_restore
from utils import deterministic_seed


def candidate_seed(master_seed: int, split: str, window_id: str, candidate_id: int) -> int:
    if candidate_id < 0: raise ValueError("candidate_id must be non-negative")
    return deterministic_seed(master_seed, f"{split}|{window_id}|candidate_{candidate_id}", "ddpm_candidate") % (2 ** 63 - 1)


def candidate_seed_matrix(master_seed: int, split: str, window_ids: Sequence[str], k: int) -> np.ndarray:
    if k < 1: raise ValueError("K must be positive")
    return np.asarray([[candidate_seed(master_seed, split, str(window_id), candidate) for candidate in range(k)]
                       for window_id in window_ids], dtype=np.int64)


def select_balanced_audit_indices(labels: np.ndarray, count: int, seed: int, split: str) -> np.ndarray:
    labels = np.asarray(labels); normal = np.flatnonzero(labels == 0); fault = np.flatnonzero(labels != 0)
    normal_count = count // 2; fault_count = count - normal_count
    if len(normal) < normal_count or len(fault) < fault_count: raise ValueError("not enough normal/fault windows")
    normal_rng = np.random.default_rng(deterministic_seed(seed, split, "audit_normal"))
    fault_rng = np.random.default_rng(deterministic_seed(seed, split, "audit_fault"))
    chosen = np.r_[normal_rng.choice(normal, normal_count, replace=False), fault_rng.choice(fault, fault_count, replace=False)]
    return np.sort(chosen.astype(np.int64))


def _noise_bank(seeds: np.ndarray, steps: int, shape: tuple[int, int], device: str) -> torch.Tensor:
    banks = []
    for seed in seeds:
        generator = torch.Generator(device=device).manual_seed(int(seed))
        banks.append(torch.randn((steps, *shape), device=device, generator=generator))
    return torch.stack(banks, dim=1)


@torch.no_grad()
def restore_candidates(
    model: torch.nn.Module, degraded: np.ndarray, observation: np.ndarray,
    seed_matrix: np.ndarray, schedule: DiffusionSchedule, batch_size: int,
    device: str, clip_min: np.ndarray, clip_max: np.ndarray,
) -> np.ndarray:
    degraded = np.asarray(degraded, dtype=np.float32); observation = np.asarray(observation, dtype=bool)
    seeds = np.asarray(seed_matrix, dtype=np.int64)
    if degraded.shape != observation.shape or degraded.ndim != 3: raise ValueError("invalid degraded/mask shapes")
    if seeds.ndim != 2 or len(seeds) != len(degraded): raise ValueError("seed matrix must be [N,K]")
    minimum = torch.from_numpy(np.asarray(clip_min, dtype=np.float32)).to(device)[None, :, None]
    maximum = torch.from_numpy(np.asarray(clip_max, dtype=np.float32)).to(device)[None, :, None]
    result = np.empty((len(degraded), seeds.shape[1], *degraded.shape[1:]), dtype=np.float32)
    model.eval()
    for candidate in range(seeds.shape[1]):
        for start in range(0, len(degraded), batch_size):
            stop = min(start + batch_size, len(degraded))
            degraded_b = torch.from_numpy(degraded[start:stop]).to(device)
            observation_b = torch.from_numpy(observation[start:stop]).to(device)
            bank = _noise_bank(seeds[start:stop, candidate], len(schedule.betas), degraded.shape[1:], device)
            cursor = 0
            def noise_factory(shape: torch.Size) -> torch.Tensor:
                nonlocal cursor
                if tuple(shape) != tuple(bank[cursor].shape): raise ValueError("noise bank shape mismatch")
                value = bank[cursor]; cursor += 1; return value
            dummy = torch.Generator(device=device)
            restored = ddpm_restore(model, degraded_b, observation_b, schedule, dummy, noise_factory,
                                    clip_min=minimum, clip_max=maximum)
            result[start:stop, candidate] = restored.cpu().numpy()
    return result


def validate_shared_context(candidates: np.ndarray, degraded: np.ndarray, observation: np.ndarray) -> None:
    values = np.asarray(candidates); degraded = np.asarray(degraded); observation = np.asarray(observation, dtype=bool)
    if values.ndim != 4 or values.shape[0] != len(degraded) or values.shape[2:] != degraded.shape[1:]:
        raise ValueError("candidate/context shape mismatch")
    expected = np.broadcast_to(degraded[:, None], values.shape)
    mask = np.broadcast_to(observation[:, None], values.shape)
    if not np.array_equal(values[mask], expected[mask]): raise ValueError("observed values must be shared and clamped")
