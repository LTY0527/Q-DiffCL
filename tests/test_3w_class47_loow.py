from scripts.run_3w_class47_loow import constrained_fold_split


def test_constrained_loow_excludes_heldout_and_preserves_coverage():
    wells = {f"W{i}" for i in range(10)}
    targets = {well: set(range(6)) for well in wells}
    split = constrained_fold_split(wells, "W0", targets, validation_count=3, seed=42)
    assert split["test"] == {"W0"}
    assert "W0" not in split["train"] and "W0" not in split["validation"]
    assert split["train"].isdisjoint(split["validation"])
    assert len(split["validation"]) == 3
