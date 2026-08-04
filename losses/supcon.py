from __future__ import annotations

import torch
import torch.nn.functional as F


def normalized_positive_weights(
    positive_mask: torch.Tensor, candidate_weights: torch.Tensor, dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize positive weights to mean one per valid anchor.

    This is equivalent to dividing each anchor's weighted positive sum by its
    positive-weight sum, so the overall SupCon scale does not follow mean(q).
    """
    target_dtype = dtype or candidate_weights.dtype
    weights = candidate_weights.to(device=positive_mask.device, dtype=target_dtype)
    if not torch.isfinite(weights).all(): raise ValueError("positive weights must be finite")
    raw = positive_mask.to(target_dtype) * weights[None, :]
    positive_count = positive_mask.sum(dim=1)
    raw_mass = raw.sum(dim=1)
    valid = (positive_count > 0) & (raw_mass > 0)
    raw_mean = raw_mass / positive_count.clamp_min(1).to(target_dtype)
    normalized = torch.where(valid[:, None], raw / raw_mean.clamp_min(1e-12)[:, None], torch.zeros_like(raw))
    return normalized, valid


def _weighted_positive_mean(
    log_probability: torch.Tensor,
    positive_mask: torch.Tensor,
    candidate_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-anchor positive means and the anchors that have positive mass."""
    positive_weights, valid = normalized_positive_weights(
        positive_mask, candidate_weights, log_probability.dtype,
    )
    positive_count = positive_mask.sum(dim=1).to(log_probability.dtype)
    means = (log_probability * positive_weights).sum(dim=1) / positive_count.clamp_min(1)
    return means, valid


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
    weights = torch.ones(count, device=features.device, dtype=features.dtype)
    if anchor_weights is not None:
        weights = anchor_weights.reshape(-1).to(features.device, features.dtype)
        if len(weights) != len(log_probability): raise ValueError("anchor_weights must match flattened features")
        weights = weights.clamp(0, 1)
    positive_mean, valid = _weighted_positive_mean(log_probability, positive, weights)
    if not valid.any():
        return features.sum() * 0.0
    loss = -positive_mean[valid].mean()
    if not torch.isfinite(loss): raise FloatingPointError("non-finite SupCon loss")
    return loss


def joint_ce_supcon(logits: torch.Tensor, projections: torch.Tensor, labels: torch.Tensor,
                    weight: float, temperature: float) -> torch.Tensor:
    return F.cross_entropy(logits, labels) + weight * supervised_contrastive_loss(projections, labels, temperature)
