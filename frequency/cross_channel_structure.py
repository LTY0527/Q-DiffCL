from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CrossChannelSpectralStructure:
    covariance: np.ndarray
    factors: np.ndarray
    shrinkage_to_diagonal: float
    eigenvalue_floor: float
    marginal_variance_matching: bool
    fit_split: str = "train"


def fit_cross_channel_spectral_structure(
    train_values: np.ndarray,
    shrinkage_to_diagonal: float = 0.25,
    eigenvalue_floor: float = 1e-5,
    marginal_variance_matching: bool = True,
    split: str = "train",
) -> CrossChannelSpectralStructure:
    if split != "train":
        raise ValueError("cross-channel structure may only be fitted on train")
    if not 0 <= shrinkage_to_diagonal <= 1:
        raise ValueError("shrinkage_to_diagonal must be in [0, 1]")
    if eigenvalue_floor <= 0:
        raise ValueError("eigenvalue_floor must be positive")
    values = np.asarray(train_values, dtype=np.float64)
    if values.ndim != 3 or len(values) < 2:
        raise ValueError("train_values must have shape [samples, channels, time] with at least two samples")
    log_amplitude = np.log1p(np.abs(np.fft.rfft(values, axis=-1)))
    channels, frequencies = log_amplitude.shape[1:]
    covariance = np.empty((frequencies, channels, channels), dtype=np.float64)
    factors = np.empty_like(covariance)
    identity = np.eye(channels, dtype=np.float64)
    for frequency in range(frequencies):
        empirical = np.atleast_2d(np.cov(log_amplitude[:, :, frequency], rowvar=False, ddof=1))
        empirical = np.nan_to_num(empirical, nan=0.0, posinf=0.0, neginf=0.0)
        diagonal = np.diag(np.maximum(np.diag(empirical), eigenvalue_floor))
        shrunk = (1 - shrinkage_to_diagonal) * empirical + shrinkage_to_diagonal * diagonal
        shrunk = 0.5 * (shrunk + shrunk.T) + eigenvalue_floor * identity
        eigenvalues, eigenvectors = np.linalg.eigh(shrunk)
        stabilized = (eigenvectors * np.maximum(eigenvalues, eigenvalue_floor)) @ eigenvectors.T
        if marginal_variance_matching:
            scale = np.sqrt(np.maximum(np.diag(stabilized), eigenvalue_floor))
            stabilized = stabilized / np.outer(scale, scale)
            stabilized = 0.5 * (stabilized + stabilized.T)
            np.fill_diagonal(stabilized, 1.0)
        eigenvalues, eigenvectors = np.linalg.eigh(stabilized)
        eigenvalues = np.maximum(eigenvalues, eigenvalue_floor)
        stabilized = (eigenvectors * eigenvalues) @ eigenvectors.T
        if marginal_variance_matching:
            scale = np.sqrt(np.maximum(np.diag(stabilized), eigenvalue_floor))
            stabilized = stabilized / np.outer(scale, scale)
            stabilized = (1 - eigenvalue_floor) * stabilized + eigenvalue_floor * identity
        covariance[frequency] = 0.5 * (stabilized + stabilized.T)
        factor_values, factor_vectors = np.linalg.eigh(covariance[frequency])
        factors[frequency] = factor_vectors @ np.diag(np.sqrt(np.maximum(factor_values, 0.0)))
    if not np.isfinite(covariance).all() or not np.isfinite(factors).all():
        raise FloatingPointError("non-finite fitted cross-channel structure")
    return CrossChannelSpectralStructure(
        covariance.astype(np.float32), factors.astype(np.float32), float(shrinkage_to_diagonal),
        float(eigenvalue_floor), bool(marginal_variance_matching), split,
    )
