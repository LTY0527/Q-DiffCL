from __future__ import annotations

from scripts.amend_paper_final_protocol import coverage, regenerate, valid
import json
from pathlib import Path


def _counts():
    return {f"W{i}": {target: int((i + target) % 4 != 0) for target in range(4)} for i in range(12)}


def test_windowref_validator_checks_every_target_without_class9_special_case():
    counts = _counts(); split = {"train": {f"W{i}" for i in range(6)}, "validation": {"W6", "W7", "W8"}, "test": {"W9", "W10", "W11"}}
    minimum = {"train": 1, "validation": 1, "test": 1}
    assert valid(split, counts, minimum)
    for well in split["validation"]: counts[well][2] = 0
    assert not valid(split, counts, minimum)
    assert coverage(split, counts)["validation"]["2"]["windows"] == 0


def test_regeneration_is_deterministic_and_data_only():
    counts = _counts(); wells = set(counts); sizes = {"train": 6, "validation": 3, "test": 3}; minimum = {name: 1 for name in sizes}
    first, first_count = regenerate(wells, counts, sizes, minimum, 31001, [], 1.0)
    second, second_count = regenerate(wells, counts, sizes, minimum, 31001, [], 1.0)
    assert first == second
    assert first_count == second_count
    assert valid(first, counts, minimum)


def test_revised_manifest_has_complete_windowref_coverage_and_no_outer_access():
    audit = json.loads(Path("outputs/paper_final_protocol/protocol_amendment_audit.json").read_text(encoding="utf-8"))
    manifest = json.loads(Path("outputs/paper_final_protocol/dry_run_manifest.json").read_text(encoding="utf-8"))
    assert audit["status"] == "PAPER_FINAL_PROTOCOL_AMENDMENT_GO"
    assert audit["windowref_coverage_go"] is True
    assert audit["unchanged"]["tep_protocol"] is True
    assert audit["performance_metrics_used_for_split_selection"] is False
    assert manifest["outer_metrics"] is None
    for row in audit["three_w"]:
        assert not any(row["overlap"].values())
        for split in ("train", "validation", "test"):
            assert all(row["windowref_coverage"][split][str(target)]["windows"] > 0 for target in range(4))
