from __future__ import annotations

import torch
from torch import nn


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        self.left_padding = dilation * (kernel_size - 1)
        super().__init__(in_channels, out_channels, kernel_size, padding=self.left_padding, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        return y[..., :-self.left_padding] if self.left_padding else y


class _Heads(nn.Module):
    def __init__(self, dimension: int, projection_dim: int, num_classes: int):
        super().__init__()
        self.projection_head = nn.Sequential(nn.Linear(dimension, dimension), nn.ReLU(), nn.Linear(dimension, projection_dim))
        self.classification_head = nn.Linear(dimension, num_classes)

    def output(self, feature_map: torch.Tensor, projection: bool, classification: bool) -> dict[str, torch.Tensor | None]:
        embedding = feature_map.mean(dim=-1)
        return {
            "feature_map": feature_map,
            "embedding": embedding,
            "projection": self.projection_head(embedding) if projection else None,
            "logits": self.classification_head(embedding) if classification else None,
        }


class TCNClassifier(_Heads):
    def __init__(self, in_channels: int, hidden_channels: int, projection_dim: int, num_classes: int, levels: int = 3):
        super().__init__(hidden_channels, projection_dim, num_classes)
        layers: list[nn.Module] = []
        current = in_channels
        for level in range(levels):
            layers += [CausalConv1d(current, hidden_channels, 3, 2 ** level), nn.ReLU()]
            current = hidden_channels
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, projection: bool = True, classification: bool = True):
        if x.ndim != 3: raise ValueError("input must have shape [batch, channels, length]")
        return self.output(self.encoder(x), projection, classification)


class CNN1DClassifier(_Heads):
    def __init__(self, in_channels: int, hidden_channels: int, projection_dim: int, num_classes: int):
        super().__init__(hidden_channels, projection_dim, num_classes)
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 5, padding="same"), nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, 3, padding="same"), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, projection: bool = True, classification: bool = True):
        if x.ndim != 3: raise ValueError("input must have shape [batch, channels, length]")
        return self.output(self.encoder(x), projection, classification)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

