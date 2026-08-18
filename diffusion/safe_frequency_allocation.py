from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .frequency_selective import continuous_alpha_bar


def asymmetric_safe_timestep(soft_mask: np.ndarray, rank_q75: np.ndarray,
                             t_critical: int = 1, t_uniform: int = 3,
                             t_noncritical: int = 5) -> tuple[np.ndarray, np.ndarray]:
    soft_mask = np.asarray(soft_mask, dtype=np.float64)
    rank_q75 = np.asarray(rank_q75, dtype=np.float64)
    if soft_mask.shape != rank_q75.shape or not np.isfinite(soft_mask).all() or not np.isfinite(rank_q75).all():
        raise ValueError("DRFD soft mask and rank_q75 must be aligned and finite")
    if np.any((soft_mask < 0) | (soft_mask > 1) | (rank_q75 < 0) | (rank_q75 > 1)):
        raise ValueError("DRFD masks and ranks must lie in [0,1]")
    r1 = float(t_critical) + (1 - soft_mask) * (float(t_noncritical) - float(t_critical))
    reliability = np.clip((0.70 - rank_q75) / 0.70, 0, 1)
    safe = np.where(r1 <= float(t_uniform), r1,
                    float(t_uniform) + reliability * (r1 - float(t_uniform)))
    return r1.astype(np.float32), safe.astype(np.float32)


def constrained_safe_variance(
    alpha_bars: torch.Tensor,
    soft_mask: np.ndarray,
    rank_q25: np.ndarray,
    rank_q75: np.ndarray,
    reliable_noncritical: np.ndarray,
    ambiguous: np.ndarray,
    preserve_dc: bool = True,
    t_critical: int = 1,
    t_uniform: int = 3,
    t_noncritical: int = 5,
    tolerance: float = 1e-7,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Allocate Uniform budget without increasing protected or ambiguous bins."""
    device = alpha_bars.device
    q25 = np.asarray(rank_q25, dtype=np.float64)
    q75 = np.asarray(rank_q75, dtype=np.float64)
    reliable_noncritical = np.asarray(reliable_noncritical, dtype=bool)
    ambiguous = np.asarray(ambiguous, dtype=bool)
    if not (np.asarray(soft_mask).shape == q25.shape == q75.shape == reliable_noncritical.shape == ambiguous.shape):
        raise ValueError("DRFD allocation arrays must have one [C,F] shape")
    r1_timestep, safe_timestep = asymmetric_safe_timestep(
        soft_mask, q75, t_critical, t_uniform, t_noncritical)
    r1_t = torch.as_tensor(r1_timestep, dtype=torch.float32, device=device)
    safe_t = torch.as_tensor(safe_timestep, dtype=torch.float32, device=device)
    initial = 1 - continuous_alpha_bar(alpha_bars, safe_t)
    maximum = torch.full_like(initial, 1 - alpha_bars[int(t_noncritical)])
    if preserve_dc:
        initial[:, 0] = 0
        maximum[:, 0] = 0
    final = initial.clone()
    eligible = torch.as_tensor(reliable_noncritical, dtype=torch.bool, device=device) & (r1_t > float(t_uniform))
    if preserve_dc:
        eligible[:, 0] = False
    uniform = torch.full_like(final, 1 - alpha_bars[int(t_uniform)])
    if preserve_dc:
        uniform[:, 0] = 0
    target_total = uniform.sum()
    deficit = target_total - final.sum()
    if float(deficit) > tolerance and eligible.any():
        room = (maximum - final).clamp_min(0) * eligible
        addition_total = torch.minimum(deficit, room.sum())
        remaining = addition_total
        active = eligible.clone()
        for _ in range(final.numel() + 1):
            if float(remaining) <= tolerance or not active.any():
                break
            share = remaining / active.sum()
            addition = torch.minimum(torch.full_like(final, share), (maximum - final).clamp_min(0)) * active
            final += addition
            remaining = addition_total - (final - initial).sum()
            active = active & ((maximum - final) > tolerance)
    elif float(deficit) < -tolerance and eligible.any():
        # The raw safe map can exceed Uniform before budget matching.  Remove
        # that excess only from the same reliable non-critical carrier set;
        # protected and ambiguous bins remain exactly at their safe variance.
        removal_total = torch.minimum(-deficit, (final * eligible).sum())
        remaining = removal_total
        active = eligible.clone()
        for _ in range(final.numel() + 1):
            if float(remaining) <= tolerance or not active.any():
                break
            share = remaining / active.sum()
            removal = torch.minimum(torch.full_like(final, share), final) * active
            final -= removal
            remaining = removal_total - (initial - final).sum()
            active = active & (final > tolerance)
    protected = r1_t <= float(t_uniform)
    ambiguous_t = torch.as_tensor(ambiguous, dtype=torch.bool, device=device)
    extra = final - initial
    residual = torch.abs(target_total - final.sum())
    target = target_total.clamp_min(torch.finfo(final.dtype).eps)
    budget_error = residual / target
    max_variance = float(1 - alpha_bars[int(t_noncritical)])
    diagnostics = {
        "target_total_variance": float(target_total),
        "initial_total_variance": float(initial.sum()),
        "final_total_variance": float(final.sum()),
        "residual_budget_mismatch": float(residual),
        "budget_error_fraction": float(budget_error),
        "eligible_bin_count": int(eligible.sum()),
        "changed_bin_count": int((torch.abs(safe_t - r1_t) > tolerance).sum()),
        "compensated_bin_count": int((extra > tolerance).sum()),
        "reduced_bin_count": int((extra < -tolerance).sum()),
        "protected_timestep_not_increased": bool(torch.allclose(safe_t[protected], r1_t[protected], atol=tolerance, rtol=0)),
        "protected_variance_not_increased": bool(torch.all(extra[protected] <= tolerance)),
        "ambiguous_variance_not_increased": bool(torch.all(extra[ambiguous_t] <= tolerance)),
        "extra_only_reliable_noncritical": bool(torch.all(~(extra > tolerance) | eligible)),
        "budget_adjustment_only_reliable_noncritical": bool(torch.all((torch.abs(extra) <= tolerance) | eligible)),
        "maximum_variance_respected": bool(torch.all(final <= max_variance + tolerance)),
        "finite": bool(torch.isfinite(final).all()),
        "r1_timestep": r1_timestep,
        "safe_timestep": safe_timestep,
        "initial_variance": initial.detach().cpu().numpy(),
        "final_variance": final.detach().cpu().numpy(),
    }
    return final, diagnostics
