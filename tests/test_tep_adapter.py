import gc
from pathlib import Path

import numpy as np
import pyreadr
import pytest
import yaml

from datasets.tep import (DatasetProtocolError, REQUIRED_FILES, frame_to_runs,
                          inspect_rdata_files, validate_tep_config)
from datasets.protocol import make_run_uid


ROOT = Path("E:/Datasets/TEP_Rieth2017/raw")


def real_config() -> dict:
    return yaml.safe_load(Path("configs/tep_template.yaml").read_text(encoding="utf-8"))


def test_unconfirmed_protocol_blocks_real_training():
    with pytest.raises(DatasetProtocolError, match="BLOCKED: DATASET_PROTOCOL_UNCONFIRMED"):
        validate_tep_config({"dataset": {"name": "tep"}})


def test_confirmed_protocol_and_compound_run_uid():
    config = real_config(); validate_tep_config(config)
    frame = {
        "faultNumber": [0, 0, 1, 1], "simulationRun": [1, 1, 1, 1],
        "sample": [1, 2, 20, 21], "xmeas_1": [0.0, 0.0, 1.0, 2.0],
    }
    import pandas as pd
    runs = frame_to_runs(pd.DataFrame(frame), config, "training")
    assert [run.run_uid for run in runs] == ["training:normal:0001", "training:fault_01:0001"]
    assert runs[0].first_faulty_sample is None
    assert runs[1].first_faulty_sample == 21


@pytest.mark.integration
def test_real_rieth_rdata_protocol():
    expected = {
        "TEP_FaultFree_Training.RData": ("fault_free_training", (250000, 55), (0, 0), 500),
        "TEP_FaultFree_Testing.RData": ("fault_free_testing", (480000, 55), (0, 0), 960),
        "TEP_Faulty_Training.RData": ("faulty_training", (5000000, 55), (1, 20), 500),
        "TEP_Faulty_Testing.RData": ("faulty_testing", (9600000, 55), (1, 20), 960),
    }
    assert set(inspect_rdata_files(ROOT)) == set(REQUIRED_FILES)
    required_columns = {"faultNumber", "simulationRun", "sample"}
    all_run_uids: set[str] = set()
    for filename, (object_name, shape, fault_range, sample_max) in expected.items():
        objects = pyreadr.read_r(str(ROOT / filename))
        assert list(objects) == [object_name]
        frame = objects[object_name]
        assert frame.shape == shape
        assert required_columns <= set(frame.columns)
        assert (int(frame.faultNumber.min()), int(frame.faultNumber.max())) == fault_range
        assert (int(frame.simulationRun.min()), int(frame.simulationRun.max())) == (1, 500)
        assert (int(frame["sample"].min()), int(frame["sample"].max())) == (1, sample_max)
        per_run = frame.groupby(["faultNumber", "simulationRun"])["sample"].agg(["count", "nunique", "min", "max"])
        expected_groups = 500 * (fault_range[1] - fault_range[0] + 1)
        assert len(per_run) == expected_groups
        assert (per_run["count"] == sample_max).all()
        assert (per_run["nunique"] == sample_max).all()
        assert (per_run["min"] == 1).all() and (per_run["max"] == sample_max).all()
        assert frame.isna().to_numpy().sum() == 0
        numeric = frame.select_dtypes(include=[np.number])
        assert sum(np.isinf(numeric[column].to_numpy(copy=False)).sum() for column in numeric.columns) == 0
        source_split = "training" if "Training" in filename else "testing"
        for fault_id, simulation_run in per_run.index:
            uid = make_run_uid(source_split, int(fault_id), int(simulation_run))
            assert uid not in all_run_uids
            all_run_uids.add(uid)
        del numeric, per_run, frame, objects
        gc.collect()
    assert len(all_run_uids) == 21_000
