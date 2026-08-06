from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def freeze_teacher(teacher: torch.nn.Module) -> torch.nn.Module:
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    return teacher


def _balanced_mean(values: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    labels = labels.reshape(-1).long()
    classes = []
    per_class = {}
    for label, name in ((0, "normal"), (1, "fault")):
        selector = labels == label
        if selector.any():
            value = values[selector].mean()
            classes.append(value); per_class[name] = value
        else:
            per_class[name] = None
    if not classes:
        raise ValueError("semantic batch must not be empty")
    total = torch.stack(classes).mean()
    return total, {"normal_count": int((labels == 0).sum()), "fault_count": int((labels == 1).sum()),
                   "normal_loss": per_class["normal"], "fault_loss": per_class["fault"]}


def balanced_semantic_consistency_loss(
    teacher: torch.nn.Module, base: torch.Tensor, generated: torch.Tensor, labels: torch.Tensor,
) -> dict[str, Any]:
    """Balanced JS + logit-margin + feature loss with frozen teacher targets."""
    with torch.no_grad():
        target = teacher(base)
        target_logits = target["logits"]
        target_probability = torch.softmax(target_logits, dim=1)
        target_feature = target["embedding"]
        target_margin = target_logits[:, 1] - target_logits[:, 0]
    output = teacher(generated)
    probability = torch.softmax(output["logits"], dim=1)
    midpoint = .5 * (target_probability + probability)
    js_per_sample = .5 * (
        (target_probability * (target_probability.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())).sum(1)
        + (probability * (probability.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())).sum(1)
    )
    margin = output["logits"][:, 1] - output["logits"][:, 0]
    margin_per_sample = F.smooth_l1_loss(margin, target_margin, reduction="none")
    feature_per_sample = 1 - F.cosine_similarity(output["embedding"], target_feature, dim=1)
    combined = js_per_sample + margin_per_sample + feature_per_sample
    total, composition = _balanced_mean(combined, labels)
    js, _ = _balanced_mean(js_per_sample, labels)
    margin_loss, _ = _balanced_mean(margin_per_sample, labels)
    feature, _ = _balanced_mean(feature_per_sample, labels)
    for value in (total, js, margin_loss, feature):
        if not torch.isfinite(value):
            raise FloatingPointError("non-finite balanced semantic loss")
    return {"total": total, "js": js, "margin": margin_loss, "feature": feature, **composition}


def semantic_consistency_losses(
    teacher: torch.nn.Module, base: torch.Tensor, predicted: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Legacy finite KL/feature pair retained for historical diagnostics."""
    with torch.no_grad():
        target = teacher(base)
        target_probability = torch.softmax(target["logits"], dim=1)
        target_feature = target["embedding"]
    generated = teacher(predicted)
    probability_loss = F.kl_div(torch.log_softmax(generated["logits"], dim=1),
                                target_probability, reduction="batchmean")
    feature_loss = (1 - F.cosine_similarity(generated["embedding"], target_feature, dim=1)).mean()
    if not torch.isfinite(probability_loss) or not torch.isfinite(feature_loss):
        raise FloatingPointError("non-finite semantic consistency loss")
    return probability_loss, feature_loss
