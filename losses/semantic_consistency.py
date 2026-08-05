from __future__ import annotations

import torch
import torch.nn.functional as F


def freeze_teacher(teacher: torch.nn.Module) -> torch.nn.Module:
    teacher.eval()
    for parameter in teacher.parameters(): parameter.requires_grad = False
    return teacher


def semantic_consistency_losses(
    teacher: torch.nn.Module, base: torch.Tensor, predicted: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """KL and feature cosine losses; teacher parameters remain frozen."""
    with torch.no_grad():
        target = teacher(base)
        target_probability = torch.softmax(target["logits"], dim=1)
        target_feature = target["embedding"]
    generated = teacher(predicted)
    probability_loss = F.kl_div(
        torch.log_softmax(generated["logits"], dim=1), target_probability,
        reduction="batchmean",
    )
    feature_loss = (1 - F.cosine_similarity(generated["embedding"], target_feature, dim=1)).mean()
    if not torch.isfinite(probability_loss) or not torch.isfinite(feature_loss):
        raise FloatingPointError("non-finite semantic consistency loss")
    return probability_loss, feature_loss
