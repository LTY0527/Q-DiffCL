from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Sequence

import numpy as np


LABEL_COLUMNS = frozenset({"class", "state"})
SOURCE_PATTERN = re.compile(r"^(WELL-\d{5}|SIMULATED_\d{5}|DRAWN_\d{5})")
WELL_PATTERN = re.compile(r"^(WELL-\d{5})_(\d{14})$")


@dataclass(frozen=True)
class ThreeWInstance:
    path: Path
    event_class: int
    source: str
    instance_id: str
    well_id: str | None


@dataclass(frozen=True)
class ThreeWBatch:
    values: np.ndarray
    observation_mask: np.ndarray
    labels: np.ndarray
    timestamps: np.ndarray
    feature_names: tuple[str, ...]


def discover_instances(data_root: Path | str) -> list[ThreeWInstance]:
    root = Path(data_root)
    if not (root / "dataset.ini").is_file():
        raise FileNotFoundError(f"3W dataset.ini not found below {root}")
    instances: list[ThreeWInstance] = []
    for path in sorted(root.glob("[0-9]/*.parquet")):
        event_class = int(path.parent.name)
        match = SOURCE_PATTERN.match(path.stem)
        if match is None:
            raise ValueError(f"unrecognized 3W instance filename: {path.name}")
        prefix = match.group(1)
        source = "WELL" if prefix.startswith("WELL-") else prefix.split("_")[0]
        well_id = prefix if source == "WELL" else None
        instances.append(ThreeWInstance(path, event_class, source, path.stem, well_id))
    if not instances:
        raise FileNotFoundError(f"no class-directory Parquet instances found below {root}")
    return instances


def process_features(columns: Iterable[str]) -> tuple[str, ...]:
    return tuple(column for column in columns if column not in LABEL_COLUMNS and column != "timestamp")


def read_instance(instance: ThreeWInstance, feature_names: Sequence[str]) -> ThreeWBatch:
    """Read one instance without imputing or interpreting transient labels.

    Missing values are returned as a separate mask and replaced by zero only in
    ``values`` so downstream tensors remain finite. Imputation/scaling must be
    fitted later using training wells only.
    """
    import pyarrow.parquet as pq

    requested = list(feature_names) + ["class", "timestamp"]
    frame = pq.read_table(instance.path, columns=requested).to_pandas()
    if "timestamp" in frame.columns:
        timestamps = frame.pop("timestamp").to_numpy()
    else:
        timestamps = frame.index.to_numpy()
    labels = frame.pop("class").astype("Int64").to_numpy(dtype=np.float64, na_value=np.nan)
    raw = frame.loc[:, feature_names].to_numpy(dtype=np.float64)
    mask = np.isfinite(raw)
    values = np.where(mask, raw, 0.0).astype(np.float32)
    return ThreeWBatch(values, mask, labels, timestamps, tuple(feature_names))


def well_level_split(
    well_ids: Sequence[str], ratios: tuple[float, float, float] = (0.6, 0.2, 0.2), seed: int = 7,
) -> dict[str, tuple[str, ...]]:
    """Create a deterministic group split; a WELL can occur in exactly one split."""
    if not np.isclose(sum(ratios), 1.0) or any(value <= 0 for value in ratios):
        raise ValueError("split ratios must be positive and sum to one")
    unique = np.asarray(sorted(set(well_ids)), dtype=object)
    if len(unique) < 3:
        raise ValueError("at least three distinct WELL groups are required")
    shuffled = unique[np.random.default_rng(seed).permutation(len(unique))]
    n_train = min(max(1, round(len(unique) * ratios[0])), len(unique) - 2)
    n_validation = min(max(1, round(len(unique) * ratios[1])), len(unique) - n_train - 1)
    result = {
        "train": tuple(sorted(shuffled[:n_train])),
        "validation": tuple(sorted(shuffled[n_train:n_train + n_validation])),
        "test": tuple(sorted(shuffled[n_train + n_validation:])),
    }
    groups = [set(result[name]) for name in ("train", "validation", "test")]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("WELL leakage detected")
    return result


def well_level_split_covering_classes(
    well_classes: dict[str, set[int]], required_classes: set[int],
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2), seed: int = 7,
    attempts: int = 100_000,
) -> dict[str, tuple[str, ...]]:
    """Search deterministic WELL-group splits that cover required classes in all splits."""
    base = well_level_split(list(well_classes), ratios, seed)
    sizes = tuple(len(base[name]) for name in ("train", "validation", "test"))
    wells = np.asarray(sorted(well_classes), dtype=object)
    rng = np.random.default_rng(seed)
    best: tuple[int, dict[str, tuple[str, ...]]] | None = None
    for _ in range(attempts):
        shuffled = wells[rng.permutation(len(wells))]
        groups = (shuffled[:sizes[0]], shuffled[sizes[0]:sizes[0] + sizes[1]], shuffled[sizes[0] + sizes[1]:])
        split = {name: tuple(sorted(group)) for name, group in zip(("train", "validation", "test"), groups)}
        coverage = [set().union(*(well_classes[well] for well in group)) for group in groups]
        missing = sum(len(required_classes - classes) for classes in coverage)
        if best is None or missing < best[0]:
            best = (missing, split)
        if missing == 0:
            return split
    assert best is not None
    raise ValueError(f"cannot cover required classes in every WELL split; minimum missing assignments={best[0]}")


def window_instance(
    batch: ThreeWBatch, length: int, stride: int, limit: int | None = None, from_end: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Window one instance while preserving native missingness masks and raw labels."""
    if length <= 0 or stride <= 0:
        raise ValueError("window length and stride must be positive")
    starts = list(range(0, len(batch.values) - length + 1, stride))
    if from_end:
        starts = list(reversed(starts))
    if limit is not None:
        starts = starts[:limit]
    if not starts:
        raise ValueError("instance is shorter than the requested window")
    values = np.stack([batch.values[start:start + length].T for start in starts]).astype(np.float32)
    masks = np.stack([batch.observation_mask[start:start + length].T for start in starts])
    labels = np.asarray([batch.labels[start + length - 1] for start in starts], dtype=np.float64)
    return values, masks, labels
