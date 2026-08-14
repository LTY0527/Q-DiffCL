import numpy as np

from scripts.run_3w_diffusion_1seed import supcon_orders
from scripts.summarize_3w_balanced_supcon_stability import stability_decision


def test_balanced_orders_are_reproducible_balanced_and_positive_safe():
    labels = np.concatenate([np.full(count, target) for target, count in enumerate((4000, 495, 4000, 3660))])
    training = {"epochs": 2, "supcon_batching": "balanced_positive_safe",
                "balanced_sampler": {"classes_per_batch": 4, "samples_per_class": 64, "batches_per_epoch": 23, "max_oversampling": 3.0}}
    first, audit = supcon_orders(labels, training, 42); second, other = supcon_orders(labels, training, 42)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert audit["batch_order_sha256"] == other["batch_order_sha256"]
    assert audit["minimum_classes_in_any_batch"] == 4 and audit["minimum_clean_samples_per_present_class"] == 64
    assert audit["clean_positive_anchor_rate"] == 1 and audit["all_classes_retain_positive_pairs"]
    assert set(audit["planned_sample_counts_per_epoch"].values()) == {1472}
    assert audit["oversampling_factors"][1] == 1472 / 495
    assert max(audit["oversampling_factors"].values()) < 3.0


def test_stability_gate_can_return_partial_go():
    metric = {"macro_f1": .5, "auprc_fault_vs_normal": .7, "far": .3, "early_recall": .8, "mean_detection_delay_seconds": 100.}
    summary = {"ORIGINAL": {"r1_class9_recall_std": .2, "r1_class9_f1_std": .1, "r1_metric_mean": metric},
               "BALANCED": {"r1_class9_recall_std": .1, "r1_class9_f1_std": .09, "r1_metric_mean": metric,
                            "paired_macro_improved_seeds": 2, "seed44_binary_auprc_delta": -.01,
                            "finite_training": True, "all_positive_pairs": True}}
    gate = {"minimum_macro_improved_seeds": 2, "maximum_seed44_binary_auprc_drop": .03, "minimum_class9_std_reduction_ratio": .2,
            "maximum_far_mean_increase": .05, "maximum_macro_mean_drop": .03, "maximum_binary_auprc_mean_drop": .03,
            "maximum_early_mean_drop": .01, "maximum_delay_ratio": 1.1, "maximum_delay_absolute_increase_seconds": 60}
    status, detail = stability_decision(summary, gate)
    assert status == "BALANCED_SUPCON_PARTIAL_GO"
    assert detail["checks"]["class9_recall_std_reduced"] and not detail["checks"]["class9_f1_std_reduced"]
