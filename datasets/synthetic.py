from __future__ import annotations

import numpy as np

from .protocol import Run


WARNING = "DEBUG | SYNTHETIC | NOT FOR SCIENTIFIC COMPARISON"


def make_synthetic_runs(n_runs: int = 12, length: int = 48, channels: int = 4, seed: int = 7) -> list[Run]:
    rng = np.random.default_rng(seed)
    runs: list[Run] = []
    samples = np.arange(length)
    for i in range(n_runs):
        fault = 0 if i % 3 == 0 else 1 + i % 2
        onset = None if fault == 0 else float(length // 2)
        values = rng.normal(size=(length, channels))
        if onset is not None:
            values[int(onset):, fault % channels] += 1.5
        runs.append(Run(f"synthetic-{i:03d}", values, samples.copy(), fault, onset))
    return runs
