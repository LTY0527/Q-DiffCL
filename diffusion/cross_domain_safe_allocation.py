from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .frequency_selective import continuous_alpha_bar


def cross_domain_safe_timestep(soft_mask: np.ndarray, safe_prob: np.ndarray,
                               t_critical: int = 1, t_uniform: int = 3,
                               t_noncritical: int = 5) -> tuple[np.ndarray, np.ndarray]:
    soft = np.asarray(soft_mask, dtype=np.float64); safe_prob = np.asarray(safe_prob, dtype=np.float64)
    if soft.shape != safe_prob.shape or not np.isfinite(soft).all() or not np.isfinite(safe_prob).all():
        raise ValueError("CDVS mask and safe_prob must be finite and aligned")
    if np.any((soft < 0) | (soft > 1) | (safe_prob < 0) | (safe_prob > 1)):
        raise ValueError("CDVS inputs must lie in [0,1]")
    r1 = t_critical + (1-soft) * (t_noncritical-t_critical)
    cdvs = np.where(r1 <= t_uniform, r1, t_uniform + safe_prob * (r1-t_uniform))
    return r1.astype(np.float32), cdvs.astype(np.float32)


def cross_domain_safe_variance(alpha_bars: torch.Tensor, soft_mask: np.ndarray, safe_prob: np.ndarray,
                               preserve_dc: bool = True, t_critical: int = 1,
                               t_uniform: int = 3, t_noncritical: int = 5,
                               tolerance: float = 1e-7) -> tuple[torch.Tensor, dict[str, Any]]:
    r1_np, cdvs_np = cross_domain_safe_timestep(soft_mask, safe_prob, t_critical, t_uniform, t_noncritical)
    device = alpha_bars.device; r1 = torch.as_tensor(r1_np, device=device); timestep = torch.as_tensor(cdvs_np, device=device)
    probability = torch.as_tensor(safe_prob, dtype=torch.float32, device=device)
    initial = 1-continuous_alpha_bar(alpha_bars, timestep); maximum = torch.full_like(initial, 1-alpha_bars[t_noncritical])
    if preserve_dc: initial[:, 0] = 0; maximum[:, 0] = 0
    final = initial.clone(); eligible = (r1 > t_uniform) & (probability > 0)
    if preserve_dc: eligible[:, 0] = False
    uniform = torch.full_like(final, 1-alpha_bars[t_uniform])
    if preserve_dc: uniform[:, 0] = 0
    target = uniform.sum(); difference = target-final.sum()
    if abs(float(difference)) > tolerance and eligible.any():
        capacity = ((maximum-final).clamp_min(0) if float(difference) > 0 else final) * eligible
        amount = torch.minimum(torch.abs(difference), capacity.sum()); remaining = amount; active = eligible.clone()
        for _ in range(final.numel()+1):
            if float(remaining) <= tolerance or not active.any(): break
            weights = probability * active
            if float(weights.sum()) <= 0: break
            proposal = remaining * weights / weights.sum()
            change = torch.minimum(proposal, (maximum-final).clamp_min(0) if float(difference) > 0 else final) * active
            final = final + change if float(difference) > 0 else final-change
            remaining = amount-torch.abs(final-initial).sum()
            active = active & (((maximum-final) > tolerance) if float(difference) > 0 else (final > tolerance))
    adjustment = final-initial; protected = r1 <= t_uniform; unsafe = probability == 0
    residual = torch.abs(target-final.sum()); error = residual/target.clamp_min(torch.finfo(final.dtype).eps)
    audit = {"r1_timestep": r1_np, "cdvs_timestep": cdvs_np,
             "initial_variance": initial.cpu().numpy(), "final_variance": final.cpu().numpy(),
             "budget_error_fraction": float(error), "residual_budget_mismatch": float(residual),
             "protected_timestep_not_increased": bool(torch.allclose(timestep[protected], r1[protected], atol=tolerance, rtol=0)),
             "protected_variance_not_increased": bool(torch.all(adjustment[protected] <= tolerance)),
             "unsafe_variance_not_increased": bool(torch.all(adjustment[unsafe] <= tolerance)),
             "budget_adjustment_only_safe_noncritical": bool(torch.all((torch.abs(adjustment) <= tolerance) | eligible)),
             "maximum_variance_respected": bool(torch.all(final <= (1-alpha_bars[t_noncritical])+tolerance)),
             "eligible_bin_count": int(eligible.sum()), "changed_bin_count": int((torch.abs(timestep-r1)>tolerance).sum()),
             "finite": bool(torch.isfinite(final).all())}
    return final, audit
