from scripts.run_r2_multiclass_criticality import R2_WEIGHTS, validate_r2_weights
from scripts.summarize_r2_multiclass_criticality import tep_decision, three_w_decision


def _method(recall_std=.1, f1_std=.05):
    return {"class_9_recall_std": recall_std, "class_9_f1_std": f1_std}


def _comparison(macro=.025, multi=.01, binary=0., far=0.):
    values = {"macro_f1": macro, "auprc_multiclass_macro": multi,
              "auprc_fault_vs_normal": binary, "far": far}
    return {"mean": values, "by_seed": {str(seed): values for seed in (42, 43, 44)},
            "wins": {"macro_f1": 3}}


def _gate():
    return {"minimum_macro_f1_mean_gain": .02, "minimum_macro_f1_wins": 2,
            "minimum_multiclass_auprc_mean_gain": 0., "maximum_binary_auprc_mean_drop": .005,
            "maximum_single_binary_auprc_drop": .03, "maximum_far_mean_increase": .05,
            "maximum_class9_std_increase_ratio": .1}


def test_r2_weights_are_frozen_and_not_searchable():
    validate_r2_weights(R2_WEIGHTS)
    changed = dict(R2_WEIGHTS); changed["weight_multiclass"] = .21
    try: validate_r2_weights(changed)
    except ValueError: pass
    else: raise AssertionError("changed multiclass weight must be rejected")


def test_3w_gate_go_partial_and_no_go():
    base = {"R2-UNIFORM": _comparison(), "methods": {"R1": _method(), "R2": _method(0.105, .052)}}
    assert three_w_decision(base, _gate())[0] == "R2_3W_GO"
    partial = {**base, "methods": {"R1": _method(), "R2": _method(.12, .052)}}
    assert three_w_decision(partial, _gate())[0] == "R2_3W_PARTIAL_GO"
    failed = {**base, "R2-UNIFORM": _comparison(macro=.005)}
    assert three_w_decision(failed, _gate())[0] == "R2_3W_NO_GO"


def test_tep_gate_go_partial_and_no_go():
    gate = {"minimum_macro_f1_wins_vs_c1": 2, "minimum_macro_f1_mean_gain_for_go": .005,
            "maximum_mean_auprc_drop": .005,
            "maximum_mean_recall_drop": .01, "maximum_mean_far_increase": .03,
            "maximum_mean_early_recall_drop": .01, "catastrophic_macro_f1_drop": .02,
            "catastrophic_auprc_drop": .03, "catastrophic_far_increase": .05,
            "catastrophic_early_recall_drop": .05}
    row = {"macro_f1": .01, "auprc": 0., "fault_recall": 0., "far": -.01, "early_recall": 0., "mean_delay": 0.}
    summary = {"R2-C1": {"mean": row, "by_seed": {str(seed): row for seed in (7, 42, 2026)}, "wins": {"macro_f1": 3}}}
    assert tep_decision(summary, gate)[0] == "R2_CROSS_DATASET_GO"
    flat = {**row, "macro_f1": 0.}; summary["R2-C1"] = {"mean": flat, "by_seed": {str(seed): flat for seed in (7, 42, 2026)}, "wins": {"macro_f1": 0}}
    assert tep_decision(summary, gate)[0] == "R2_CROSS_DATASET_PARTIAL_GO"
    bad = {**flat, "far": .06}; summary["R2-C1"] = {"mean": bad, "by_seed": {str(seed): bad for seed in (7, 42, 2026)}, "wins": {"macro_f1": 0}}
    assert tep_decision(summary, gate)[0] == "R2_CROSS_DATASET_NO_GO"
