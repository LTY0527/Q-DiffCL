from pathlib import Path

import numpy as np
import pytest

from datasets.three_w import process_features, well_level_split, well_level_split_covering_classes


def test_process_features_excludes_labels_and_timestamp():
    assert process_features(["P-PDG", "class", "state", "timestamp", "T-PDG"]) == ("P-PDG", "T-PDG")


def test_well_level_split_is_deterministic_and_disjoint():
    wells = [f"WELL-{index:05d}" for index in range(1, 11)]
    first = well_level_split(wells, seed=7)
    second = well_level_split(list(reversed(wells)), seed=7)
    assert first == second
    assert set(first["train"]).isdisjoint(first["validation"])
    assert set(first["train"]).isdisjoint(first["test"])
    assert set(first["validation"]).isdisjoint(first["test"])
    assert set().union(*map(set, first.values())) == set(wells)


@pytest.mark.parametrize("ratios", [(0.5, 0.5, 0.5), (0.8, 0.2, 0.0)])
def test_well_level_split_rejects_invalid_ratios(ratios):
    with pytest.raises(ValueError):
        well_level_split(["WELL-1", "WELL-2", "WELL-3"], ratios=ratios)


def test_coverage_split_covers_requested_classes_without_well_leakage():
    mapping = {f"WELL-{index}": {index % 2, (index + 1) % 2} for index in range(9)}
    split = well_level_split_covering_classes(mapping, {0, 1}, seed=3, attempts=100)
    for wells in split.values():
        assert set().union(*(mapping[well] for well in wells)) == {0, 1}
    assert len(set().union(*map(set, split.values()))) == 9
