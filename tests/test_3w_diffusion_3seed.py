from scripts.run_3w_diffusion_1seed import METHODS
from scripts.summarize_3w_diffusion_3seed import three_seed_decision


def metrics(macro, auprc, far, early, delay, recalls=(0.7, 0.6, 0.8, 0.5)):
    return {
        "macro_f1": macro, "recall_macro": 0.6, "auprc_fault_vs_normal": auprc,
        "auprc_multiclass_macro": auprc, "far": far, "early_recall": early,
        "mean_detection_delay_seconds": delay,
        "per_class": [{"original_class": c, "recall": r, "f1": r} for c, r in zip((0, 2, 8, 9), recalls)],
    }


def test_three_seed_gate_requires_paired_majority_and_no_systematic_zero():
    records = []
    for gain in (0.02, 0.01, -0.001):
        uniform = metrics(0.5, 0.6, 0.3, 0.8, 100)
        r1 = metrics(0.5 + gain, 0.605, 0.28, 0.795, 95)
        records.append({"methods": {METHODS[1]: {"metrics": uniform}, METHODS[2]: {"metrics": r1}}})
    single_gate = {
        "maximum_early_recall_drop": 0.01, "maximum_delay_ratio": 1.10, "maximum_delay_absolute_increase_seconds": 60,
        "catastrophic_macro_f1_drop": 0.03, "catastrophic_auprc_drop": 0.03, "catastrophic_far_increase": 0.05,
        "catastrophic_early_recall_drop": 0.05, "catastrophic_delay_ratio": 1.25, "catastrophic_delay_increase_seconds": 300,
    }
    status, summary = three_seed_decision(records, single_gate, {
        "minimum_macro_nonworse_seeds": 2, "minimum_far_improved_seeds": 2, "minimum_auprc_nonworse_seeds": 2,
    })
    assert status == "3W_FREQUENCY_SELECTIVE_R1_3SEED_GO"
    assert summary["wins"]["macro_f1_improved"] == 2
    assert not summary["systematic_zero_recall_classes"]
