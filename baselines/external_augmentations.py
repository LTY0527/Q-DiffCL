from __future__ import annotations

import hashlib
from typing import Literal

import numpy as np
import torch
from torch import nn


TraditionalMethod = Literal["NO_AUG", "JITTER", "SCALING", "JITTER_SCALING"]


def _sample_seed(seed: int, sample_id: str, method: str) -> int:
    payload = f"{seed}|{sample_id}|{method}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def traditional_view(
    values: np.ndarray,
    sample_ids: np.ndarray,
    method: TraditionalMethod,
    seed: int,
    jitter_std: float,
    scaling_std: float,
) -> np.ndarray:
    """Create deterministic per-window views without changing the clean input.

    Scaling follows the repository's frozen traditional baseline: one factor per
    channel, constant over time. Jitter is independent per observation.
    """
    if method not in {"NO_AUG", "JITTER", "SCALING", "JITTER_SCALING"}:
        raise ValueError(f"unsupported traditional method: {method}")
    if len(values) != len(sample_ids):
        raise ValueError("sample ids must align with values")
    result = np.asarray(values, dtype=np.float32).copy()
    if method == "NO_AUG":
        return result
    for index, (value, sample_id) in enumerate(zip(values, sample_ids)):
        rng = np.random.default_rng(_sample_seed(seed, str(sample_id), method))
        current = np.asarray(value, dtype=np.float32)
        if method in {"SCALING", "JITTER_SCALING"}:
            scale = rng.normal(1.0, float(scaling_std), size=(current.shape[0], 1))
            current = current * scale
        if method in {"JITTER", "JITTER_SCALING"}:
            current = current + rng.normal(0.0, float(jitter_std), size=current.shape)
        result[index] = current
    if not np.isfinite(result).all():
        raise RuntimeError(f"{method} generated non-finite values")
    return result


class FreRAAdapter(nn.Module):
    """Shared-backbone adaptation of the official KDD 2025 FreRA gate.

    Source: Tian0426/FreRA, commit 7236fbfc1c665f83ed5f4364cad59093ee283c14,
    ``autoaug/fourier.py``. The adapter preserves the learnable stochastic gate,
    detached self-adaptive modification and L1 regularizer. It replaces the
    official hard-coded ``.cuda()`` and maps this repository's [B,C,L] tensors
    to the intended time-axis FFT. The empty selected-bin guard only prevents
    the official division-by-empty failure and contributes zero modification.
    """

    OFFICIAL_COMMIT = "7236fbfc1c665f83ed5f4364cad59093ee283c14"

    def __init__(self, window_length: int) -> None:
        super().__init__()
        self.window_length = int(window_length)
        self.weight = nn.Parameter(torch.empty(self.window_length // 2 + 1, 2))
        nn.init.normal_(self.weight, mean=0.0, std=0.10)
        self.last_gate: torch.Tensor | None = None

    def _gate(self, temperature: float) -> torch.Tensor:
        if self.training:
            bias = 0.0001
            eps = (bias - (1.0 - bias)) * torch.rand_like(self.weight) + (1.0 - bias)
            gate = torch.sigmoid((torch.log(eps) - torch.log1p(-eps) + self.weight) / temperature)
        else:
            gate = torch.sigmoid(self.weight)
        self.last_gate = gate
        return gate

    def forward(self, values: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        if values.ndim != 3 or values.shape[-1] != self.window_length:
            raise ValueError("FreRA expects [batch, channel, window_length]")
        gate = self._gate(float(temperature))
        noise = -self.weight.detach().clone()
        threshold = max(0.0, float(noise[:, 0].mean().detach()))
        noise[noise < threshold] = 0.0
        selected = noise[:, 0] != 0
        adaptive = torch.zeros_like(noise[:, 0])
        if bool(selected.any()):
            adaptive[selected] = noise[selected, 0] / noise[selected, 0].mean()
        multiplier = gate[:, 0] + adaptive
        spectrum = torch.fft.rfft(values, dim=-1)
        return torch.fft.irfft(spectrum * multiplier[None, None, :], n=self.window_length, dim=-1)

    def l1_regularizer(self) -> torch.Tensor:
        if self.last_gate is None:
            raise RuntimeError("FreRA must run before requesting its regularizer")
        return torch.norm(self.last_gate[:, 0], p=1)
