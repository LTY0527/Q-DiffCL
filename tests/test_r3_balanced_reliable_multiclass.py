import copy

from scripts.run_r3_balanced_reliable_multiclass import R3_MODE, R3_WEIGHTS, validate_r3_settings
from scripts.summarize_r3_balanced_reliable_multiclass import tep_decision, three_w_decision


def _comparison(macro=.03, std=.03, wins=3, binary=.01, multi=.01, far=0.):
    mean = {"macro_f1": macro, "auprc_fault_vs_normal": binary,
            "auprc_multiclass_macro": multi, "far": far}
    seed = {**mean}; return {"mean": mean, "std": {"macro_f1": std},
                            "wins": {"macro_f1": wins}, "nonnegative": {"macro_f1": wins},
                            "by_seed": {str(value): seed for value in (42, 43, 44)}}


def _gate():
    return {"minimum_macro_f1_mean_gain": .02, "minimum_macro_f1_wins": 2,
            "minimum_r1_macro_f1_nonnegative_seeds": 2, "minimum_r1_macro_f1_mean_gain": -.005,
            "maximum_macro_f1_delta_std": .05, "minimum_multiclass_auprc_mean_gain": 0.,
            "minimum_binary_auprc_mean_gain": 0., "maximum_single_binary_auprc_drop": .03,
            "maximum_far_mean_increase": .05, "minimum_meaningful_class9_recall": .02,
            "minimum_meaningful_class9_f1": .01}


def test_r3_formula_and_mode_are_frozen():
    settings = {**R3_WEIGHTS, "multiclass_mode": R3_MODE}; validate_r3_settings(settings)
    changed = copy.deepcopy(settings); changed["weight_multiclass"] = .21
    try: validate_r3_settings(changed)
    except ValueError: pass
    else: raise AssertionError("changed R3 weights must be rejected")


def test_3w_gate_distinguishes_go_partial_and_no_go():
    methods = {"R2": {"class_9_recall": {"mean": .007}, "class_9_f1": {"mean": .005}},
               "R3": {"class_9_recall": {"mean": .03}, "class_9_f1": {"mean": .02}}}
    summary = {"methods": methods, "R3-UNIFORM": _comparison(), "R3-R1": _comparison(macro=0.)}
    assert three_w_decision(summary, _gate())[0] == "R3_3W_GO"
    partial = copy.deepcopy(summary); partial["methods"]["R3"]["class_9_recall"]["mean"] = .01
    assert three_w_decision(partial, _gate())[0] == "R3_3W_PARTIAL_GO"
    failed = copy.deepcopy(summary); failed["R3-UNIFORM"] = _comparison(macro=.005)
    assert three_w_decision(failed, _gate())[0] == "R3_3W_NO_GO"


def test_tep_gate_requires_preservation_and_three_nonnegative_seeds():
    gate = {"minimum_macro_f1_nonnegative_seeds_vs_c1": 3, "maximum_mean_macro_f1_drop_vs_r2": .005,
            "maximum_mean_auprc_drop": .005, "maximum_mean_recall_drop": .01,
            "maximum_mean_far_increase": .03, "maximum_mean_early_recall_drop": .01,
            "catastrophic_macro_f1_drop": .02, "catastrophic_auprc_drop": .03,
            "catastrophic_far_increase": .05, "catastrophic_early_recall_drop": .05}
    row = {"macro_f1": .001, "auprc": 0., "fault_recall": 0., "far": 0., "early_recall": 0., "mean_delay": 0.}
    comp = {"mean": row, "nonnegative": {"macro_f1": 3}, "by_seed": {str(s): row for s in (7, 42, 2026)}}
    assert tep_decision({"R3-C1": comp, "R3-R2": comp}, gate)[0] == "R3_CROSS_DATASET_GO"
    bad = copy.deepcopy(comp); bad["mean"] = {**row, "far": .06}
    assert tep_decision({"R3-C1": bad, "R3-R2": comp}, gate)[0] == "R3_CROSS_DATASET_NO_GO"
