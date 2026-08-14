import numpy as np

from scripts.run_3w_diffusion_1seed import supcon_orders
from scripts.summarize_3w_cross_well_supcon import stability_decision
from trainers.balanced import CrossWellPositiveSafeBatchSampler


def test_cross_well_sampler_is_deterministic_balanced_and_uses_every_well():
    labels = np.repeat(np.arange(2), 12)
    wells = np.asarray((["A"] * 4 + ["B"] * 4 + ["C"] * 4) * 2, dtype=object)
    sampler = CrossWellPositiveSafeBatchSampler(labels, wells, 2, 6, 2, 42, 3.0)
    first = list(sampler); second = list(sampler)
    assert first == second
    for batch in first:
        for target in (0, 1):
            selected = np.asarray(batch)[labels[np.asarray(batch)] == target]
            assert len(selected) == 6 and set(wells[selected]) == {"A", "B", "C"}


def test_cross_well_orders_record_batch_epoch_and_positive_audits():
    labels = np.repeat(np.arange(2), 12); wells = np.asarray((["A"] * 4 + ["B"] * 4 + ["C"] * 4) * 2, dtype=object)
    training = {"epochs": 2, "batch_size": 12, "supcon_batching": "cross_well_positive_safe",
                "cross_well_sampler": {"classes_per_batch": 2, "samples_per_class": 6, "batches_per_epoch": 2, "max_oversampling": 3.0}}
    orders, audit = supcon_orders(labels, training, 42, wells)
    assert len(orders) == 2 and all(len(order) == 24 for order in orders)
    cross = audit["cross_well"]
    assert cross["class_well_counts"] == {0: 3, 1: 3} and not cross["classes_without_cross_well_support"]
    assert cross["minimum_wells_for_multiwell_class_in_any_batch"] == 3
    assert cross["duplicate_window_rate"] == 0
    assert cross["clean_cross_well_positive_ratio"] > .7 and cross["paired_view_cross_well_positive_ratio"] > .6
    assert len(cross["per_epoch"]) == 2 and len(cross["per_epoch"][0]["batches"]) == 2
    assert sum(cross["per_epoch"][0]["per_class_well_sample_counts"][0].values()) == 12


def test_cross_well_sampler_does_not_repeat_a_tiny_well_window_each_batch():
    labels = np.repeat(np.arange(2), 8)
    wells = np.asarray(["TINY"] + ["LARGE"] * 7 + ["A"] * 4 + ["B"] * 4, dtype=object)
    batches = list(CrossWellPositiveSafeBatchSampler(labels, wells, 2, 4, 2, 9, 3.0))
    assert all(len(batch) == len(set(batch)) for batch in batches)
    assert sum(batch.count(0) for batch in batches) == 1


def test_cross_well_audit_explicitly_marks_single_well_class():
    labels = np.repeat(np.arange(2), 8); wells = np.asarray(["ONLY"] * 8 + ["A"] * 4 + ["B"] * 4, dtype=object)
    training = {"epochs": 1, "batch_size": 8, "supcon_batching": "cross_well_positive_safe",
                "cross_well_sampler": {"classes_per_batch": 2, "samples_per_class": 4, "batches_per_epoch": 1, "max_oversampling": 3.0}}
    _, audit = supcon_orders(labels, training, 7, wells)
    assert audit["cross_well"]["classes_without_cross_well_support"] == [0]


def test_cross_well_gate_returns_partial_only_with_stability_gain_and_no_catastrophe():
    metric = {"macro_f1": .5, "auprc_fault_vs_normal": .7, "far": .3, "early_recall": .8, "mean_detection_delay_seconds": 100.}
    common = {"r1_metric_mean": metric, "paired_macro_improved_seeds": 2,
              "paired_binary_auprc_by_seed": {"42": 0., "43": 0., "44": 0.},
              "finite_training": True, "all_positive_pairs": True, "all_classes_have_cross_well_support": True}
    summary = {"ORIGINAL": {**common, "r1_class9_recall_std": .2, "r1_class9_f1_std": .1, "paired_view_cross_well_positive_ratio": .4},
               "BALANCED": {**common, "r1_class9_recall_std": .3, "r1_class9_f1_std": .2, "paired_view_cross_well_positive_ratio": .5},
               "CROSS_WELL": {**common, "r1_class9_recall_std": .1, "r1_class9_f1_std": .09, "paired_view_cross_well_positive_ratio": .7}}
    gate = {"minimum_macro_improved_seeds": 2, "maximum_any_seed_binary_auprc_drop": .03,
            "minimum_class9_std_reduction_ratio": .2, "minimum_cross_well_ratio_increase": .05,
            "maximum_far_mean_increase": .05, "maximum_macro_mean_drop": .03, "maximum_binary_auprc_mean_drop": .03,
            "maximum_early_mean_drop": .01, "maximum_delay_ratio": 1.1, "maximum_delay_absolute_increase_seconds": 60}
    status, detail = stability_decision(summary, gate)
    assert status == "CROSS_WELL_SUPCON_PARTIAL_GO"
    assert detail["checks"]["class9_recall_std_reduced"] and not detail["checks"]["class9_f1_std_reduced"]
