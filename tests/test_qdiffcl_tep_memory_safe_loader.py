from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from datasets.tep import frame_to_runs, read_rdata_frame, read_rdata_frame_legacy


DATA_CONFIG = Path("configs/diffusion_quality_retest.yaml")
REGIME_CONFIG = Path("configs/qdiffcl_data_regime_v1.yaml")


def _configs() -> tuple[dict, dict]:
    data = yaml.safe_load(DATA_CONFIG.read_text(encoding="utf-8"))
    regime = yaml.safe_load(REGIME_CONFIG.read_text(encoding="utf-8"))
    return data, regime


def _array_hash(values: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _assert_runs_exact(old: list, new: list) -> None:
    assert [run.run_uid for run in old] == [run.run_uid for run in new]
    assert len(old) == len(new)
    for left, right in zip(old, new):
        assert left.fault_id == right.fault_id
        assert left.first_faulty_sample == right.first_faulty_sample
        assert left.values.shape == right.values.shape
        assert left.values.dtype == right.values.dtype
        assert left.values.flags.f_contiguous == right.values.flags.f_contiguous
        assert left.samples.dtype == right.samples.dtype
        assert np.array_equal(left.values, right.values, equal_nan=True)
        assert np.array_equal(left.samples, right.samples, equal_nan=True)


def test_real_small_rdata_old_vs_memory_safe_exact():
    data, regime = _configs()
    root = Path(regime["tep"]["data_root"])
    path = root / "TEP_FaultFree_Training.RData"
    if not path.is_file():
        pytest.skip("registered TEP RData is not available")

    old_frame = read_rdata_frame_legacy(path)
    new_frame = read_rdata_frame(path)
    assert type(old_frame._mgr).__name__ == "BlockManager"
    assert type(new_frame._mgr).__name__ == "ArrayManager"
    assert old_frame.shape == new_frame.shape
    assert old_frame.index.equals(new_frame.index)
    assert list(old_frame.columns) == list(new_frame.columns)
    assert all(old_frame[column].dtype == new_frame[column].dtype for column in old_frame.columns)
    assert all(
        np.array_equal(old_frame[column].to_numpy(), new_frame[column].to_numpy(), equal_nan=True)
        for column in old_frame.columns
    )

    limits = regime["tep"]["selected_run_limits"]["training"]
    old_runs = frame_to_runs(old_frame, data, "training", limits, prefilter=False)
    new_runs = frame_to_runs(new_frame, data, "training", limits, prefilter=True)
    _assert_runs_exact(old_runs, new_runs)
    assert _array_hash([run.values for run in old_runs]) == _array_hash([run.values for run in new_runs])
    assert _array_hash([run.samples for run in old_runs]) == _array_hash([run.samples for run in new_runs])


def test_prefilter_preserves_mixed_fault_run_semantics_exact():
    data, _ = _configs()
    rows = []
    for fault in (0, 1, 2):
        for simulation in (1, 2, 3):
            for sample in (1, 2, 3):
                rows.append({
                    "faultNumber": fault,
                    "simulationRun": simulation,
                    "sample": sample,
                    "xmeas_1": fault * 100 + simulation * 10 + sample,
                    "xmeas_2": np.nan if sample == 2 else sample / 10,
                })
    frame = pd.DataFrame(rows)
    limits = {"normal": 2, "fault": 1}
    old = frame_to_runs(frame, data, "testing", limits, prefilter=False)
    new = frame_to_runs(frame, data, "testing", limits, prefilter=True)
    _assert_runs_exact(old, new)
    assert [run.run_uid for run in new] == [
        "testing:normal:0001", "testing:normal:0002",
        "testing:fault_01:0001", "testing:fault_02:0001",
    ]
    assert [run.first_faulty_sample for run in new] == [None, None, 161.0, 161.0]


def test_formal_runner_reuses_prepared_context_without_second_full_load():
    import scripts.run_qdiffcl_data_regime as runner

    source = inspect.getsource(runner.main)
    assert "run_formal_context(config, run_manifest, dataset, fraction, outer_id, args.device, context)" in source
    formal_source = inspect.getsource(runner.run_formal_context)
    assert "if context is None:" in formal_source


def test_frozen_scientific_protocol_is_unchanged():
    _, regime = _configs()
    assert regime["algorithm"]["criticality_weights"] == {
        "weight_discriminative": 0.5,
        "weight_early": 0.5,
        "weight_run_stability": 0.0,
    }
    assert regime["algorithm"]["critical_ratio"] == 0.30
    assert regime["algorithm"]["t_uniform"] == 3
    assert regime["algorithm"]["t_critical"] == 1
    assert regime["algorithm"]["t_noncritical"] == 5
    assert regime["rho_grid"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert ("TEP", 0.10) not in __import__(
        "scripts.run_qdiffcl_data_regime", fromlist=["legal_dataset_fractions"]
    ).legal_dataset_fractions(__import__(
        "scripts.run_qdiffcl_data_regime", fromlist=["load_config"]
    ).load_config(REGIME_CONFIG))
