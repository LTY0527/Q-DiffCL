from __future__ import annotations

import torch
import torch.nn.functional as F


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    return quality_weighted_supervised_contrastive_loss(features, labels, None, temperature)


def quality_weighted_supervised_contrastive_loss(
    features: torch.Tensor, labels: torch.Tensor, anchor_weights: torch.Tensor | None = None,
    temperature: float = 0.1,
) -> torch.Tensor:
    if temperature <= 0: raise ValueError("temperature must be positive")
    if features.ndim == 3:
        features = features.reshape(-1, features.shape[-1])
        labels = labels.repeat_interleave(features.shape[0] // labels.shape[0])
    features = F.normalize(features, dim=-1)
    logits = torch.clamp(features @ features.T / temperature, -100.0, 100.0)
    count = logits.shape[0]
    self_mask = torch.eye(count, dtype=torch.bool, device=logits.device)
    positive = labels[:, None].eq(labels[None, :]) & ~self_mask
    denominator_mask = ~self_mask
    stable = logits - logits.masked_fill(~denominator_mask, float("-inf")).max(dim=1, keepdim=True).values
    log_denominator = torch.logsumexp(stable.masked_fill(~denominator_mask, float("-inf")), dim=1)
    log_probability = stable - log_denominator[:, None]
    if anchor_weights is None:
        positive_weights = positive.to(features.dtype)
    else:
        weights = anchor_weights.reshape(-1).to(features.device, features.dtype).clamp(0, 1)
        if len(weights) != len(log_probability): raise ValueError("anchor_weights must match flattened features")
        positive_weights = positive.to(features.dtype) * weights[None, :]
    positive_count = positive_weights.sum(dim=1)
    valid = positive_count > 0
    if not valid.any():
        return features.sum() * 0.0
    losses = -((log_probability * positive_weights).sum(dim=1) / positive_count.clamp_min(1e-12))
    loss = losses[valid].mean()
    if not torch.isfinite(loss): raise FloatingPointError("non-finite SupCon loss")
    return loss


def joint_ce_supcon(logits: torch.Tensor, projections: torch.Tensor, labels: torch.Tensor,
                    weight: float, temperature: float) -> torch.Tensor:
    return F.cross_entropy(logits, labels) + weight * supervised_contrastive_loss(projections, labels, temperature)
