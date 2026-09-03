from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import numpy as np


REQUIRED_FILES = (
    "TEP_FaultFree_Training.RData", "TEP_FaultFree_Testing.RData",
    "TEP_Faulty_Training.RData", "TEP_Faulty_Testing.RData",
)


class DatasetProtocolError(RuntimeError):
    pass


def validate_tep_config(config: dict[str, Any]) -> None:
    dataset = config.get("dataset", {})
    required = ("version", "source", "doi", "root", "run_id_column", "sample_column", "fault_id_column")
    missing = [key for key in required if dataset.get(key) in (None, "")]
    boundary = config.get("fault_boundary", {})
    for split_name in ("training", "testing"):
        item = boundary.get(split_name, {})
        if item.get("last_normal_sample") is None or item.get("first_faulty_sample") is None:
            missing.append(f"fault_boundary.{split_name}")
        elif int(item["first_faulty_sample"]) != int(item["last_normal_sample"]) + 1:
            raise DatasetProtocolError(f"ambiguous fault boundary for {split_name}")
    if missing:
        raise DatasetProtocolError(
            "BLOCKED: DATASET_PROTOCOL_UNCONFIRMED; configure evidence-backed fields: " + ", ".join(missing)
        )


def inspect_rdata_files(root: Path) -> dict[str, int]:
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        raise DatasetProtocolError("BLOCKED: DATASET_NOT_AVAILABLE; missing: " + ", ".join(missing))
    return {name: (root / name).stat().st_size for name in REQUIRED_FILES}


def load_tep_rdata(root: Path, config: dict[str, Any]) -> Any:
    inspect_rdata_files(root)
    validate_tep_config(config)
    try:
        import pyreadr
    except ImportError as exc:
        raise DatasetProtocolError("pyreadr is required to read RData; install project[data]") from exc
    return {name: pyreadr.read_r(str(root / name)) for name in REQUIRED_FILES}


def _read_rdata_frame(path: Path, *, column_manager: bool) -> Any:
    try:
        import pandas as pd
        import pyreadr
    except ImportError as exc:
        raise DatasetProtocolError("pyreadr is required to read RData; install project[data]") from exc
    if column_manager:
        # pyreadr constructs a DataFrame from one ndarray per R column. Pandas'
        # BlockManager then consolidates all float columns with np.vstack,
        # transiently requiring another multi-GiB full-table copy for TEP.
        # ArrayManager keeps those already-decoded columns separate. Values and
        # dtypes are unchanged, and selected Runs are materialized below.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            with pd.option_context("mode.data_manager", "array"):
                objects = pyreadr.read_r(str(path))
    else:
        objects = pyreadr.read_r(str(path))
    if len(objects) != 1:
        raise DatasetProtocolError(f"expected exactly one data frame in {path.name}")
    return next(iter(objects.values()))


def read_rdata_frame(path: Path) -> Any:
    """Read TEP without pandas' redundant full-table consolidation copy."""
    return _read_rdata_frame(path, column_manager=True)


def read_rdata_frame_legacy(path: Path) -> Any:
    """Pre-amendment reader retained only for small-subset equivalence tests."""
    return _read_rdata_frame(path, column_manager=False)


def frame_to_runs(
    frame: Any, config: dict[str, Any], source_split: str,
    max_runs_per_fault: int | dict[str, int] | None = None, *, prefilter: bool = True,
) -> list[Any]:
    """Convert a validated RData frame to globally identifiable Runs."""
    from .protocol import Run, make_run_uid
    validate_tep_config(config)
    dataset = config["dataset"]
    run_column = dataset["run_id_column"]
    sample_column = dataset["sample_column"]
    fault_column = dataset["fault_id_column"]
    feature_columns = [column for column in frame.columns if column not in {run_column, sample_column, fault_column}]
    first_faulty_sample = config["fault_boundary"][source_split]["first_faulty_sample"]
    runs = []
    if prefilter and max_runs_per_fault is not None:
        if isinstance(max_runs_per_fault, dict):
            normal_limit = int(max_runs_per_fault["normal"])
            fault_limit = int(max_runs_per_fault["fault"])
            keep = ((frame[fault_column] == 0) & (frame[run_column] <= normal_limit)) | (
                (frame[fault_column] != 0) & (frame[run_column] <= fault_limit)
            )
        else:
            keep = frame[run_column] <= int(max_runs_per_fault)
        frame = frame.loc[keep]
    grouped = frame.groupby([fault_column, run_column], sort=True)
    for (fault_value, simulation_value), group in grouped:
        fault_id = int(fault_value)
        simulation_run = int(simulation_value)
        if isinstance(max_runs_per_fault, dict):
            limit = int(max_runs_per_fault["normal" if fault_id == 0 else "fault"])
        else:
            limit = max_runs_per_fault
        if limit is not None and simulation_run > limit:
            continue
        run_uid = make_run_uid(source_split, fault_id, simulation_run)
        boundary = None if fault_id == 0 else float(first_faulty_sample)
        # BlockManager historically exposed each selected Run as an F-contiguous
        # view. Preserve that layout explicitly: NumPy reduction order depends
        # on strides, so C-contiguous ArrayManager output can differ by a few
        # floating-point ulps even when every element is identical.
        values = np.asfortranarray(group[feature_columns].to_numpy())
        runs.append(Run(run_uid, values, group[sample_column].to_numpy(), fault_id, boundary))
    return runs
