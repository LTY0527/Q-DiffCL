from scripts.run_3w_final_primary_grouped import FINAL_PRIMARY_CLASSES, grouped_split, split_coverage
from scripts.summarize_3w_final_primary_grouped import METRICS, build_summary
from scripts.summarize_3w_diffusion_1seed import classify


def test_grouped_split_is_disjoint_reproducible_and_covered():
    wells = {f"W{i:02d}" for i in range(12)}
    targets = {well: {0, 1, 2, 3} for well in wells}
    kwargs = ({"train": 6, "validation": 3, "test": 3}, {"train": 1, "validation": 1, "test": 1}, 42)
    first = grouped_split(wells, targets, *kwargs)
    second = grouped_split(wells, targets, *kwargs)
    assert first == second
    assert set.union(*first.values()) == wells
    assert all(not first[a] & first[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")))
    coverage = split_coverage(first, targets)
    assert all(set(item) == set(range(len(FINAL_PRIMARY_CLASSES))) for item in coverage.values())


def test_summary_hard_gate_blocks_any_zero_recall():
    classes = [0, 2, 8, 9]
    rows = []
    for index in range(3):
        row = {name: 0.5 for name in METRICS}
        row.update({f"class_{original}_recall": 0.0 if (index == 1 and original == 8) else 0.5 for original in classes})
        rows.append(row)
    result = build_summary(rows, classes)
    assert result["status"] == "3W_FINAL_PRIMARY_STABILITY_HOLD"
    assert result["zero_recall_split_counts"]["8"] == 1
    assert not result["diffusion_allowed"]


def test_diffusion_gate_requires_r1_improvement_without_operational_regression():
    per_class = [{"original_class": original, "recall": 0.5, "f1": 0.4} for original in (0, 2, 8, 9)]
    uniform = {"macro_f1": 0.5, "recall_macro": 0.5, "auprc_fault_vs_normal": 0.7, "auprc_multiclass_macro": 0.6, "far": 0.2, "early_recall": 0.8, "mean_detection_delay_seconds": 100.0, "per_class": per_class}
    r1 = {"macro_f1": 0.51, "recall_macro": 0.51, "auprc_fault_vs_normal": 0.71, "auprc_multiclass_macro": 0.61, "far": 0.19, "early_recall": 0.795, "mean_detection_delay_seconds": 105.0, "per_class": per_class}
    gate = {
        "maximum_early_recall_drop": 0.01, "maximum_delay_ratio": 1.10, "maximum_delay_absolute_increase_seconds": 60.0,
        "catastrophic_macro_f1_drop": 0.03, "catastrophic_auprc_drop": 0.03, "catastrophic_far_increase": 0.05,
        "catastrophic_early_recall_drop": 0.05, "catastrophic_delay_ratio": 1.25, "catastrophic_delay_increase_seconds": 300.0,
    }
    status, _, checks = classify(uniform, r1, gate)
    assert status == "3W_FREQUENCY_SELECTIVE_R1_1SEED_GO"
    assert all(checks.values())
