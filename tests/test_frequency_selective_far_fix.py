from scripts.run_frequency_selective_far_fix import (
    evaluate_seed7_gate, select_repair_variant,
)


def _record(macro, far, auprc=.9, recall=.9, early=.8, drift=.1):
    return {"validation": {"metrics": {"macro_f1": macro, "far": far, "auprc": auprc,
                                         "fault_recall": recall}, "early_fault": {"recall": early}},
            "validation_structure": {"normal": {"corr_drift": drift}}}


def test_variant_selection_uses_validation_constraints_and_lexicographic_rule():
    records = {"R0": _record(.90, .05), "R1": _record(.901, .03),
               "R2": _record(.91, .2), "R3": _record(.9005, .02)}
    selected, audit = select_repair_variant(
        records, {"far": .04, "auprc": .9, "fault_recall": .9},
        {"macro_f1_tolerance": .001, "maximum_far_above_c1": .005,
         "maximum_auprc_drop": .005, "maximum_recall_drop": .01})
    assert selected == "R3"
    assert not audit["decisions"]["R0"]["eligible"]
    assert not audit["decisions"]["R2"]["eligible"]
    assert audit["selection_split"] == "validation" and not audit["test_used"]


def test_seed7_gate_requires_every_fixed_condition():
    c1 = {"metrics": {"macro_f1": .8, "far": .04, "auprc": .9, "fault_recall": .8},
          "early_fault": {"recall": .5}, "detection_delay": {"mean_delay_samples": 100}}
    c2 = {"metrics": {}}
    c2s = {"metrics": {"macro_f1": .81, "far": .04, "auprc": .9, "fault_recall": .8},
           "early_fault": {"recall": .51}, "detection_delay": {"mean_delay_samples": 99}}
    audit = {"critical_fisher_retention": 1., "time_normalized_l1": .05, "finite": True}
    checks, passed = evaluate_seed7_gate(
        c1, c2, c2s, audit, audit, .2, .1,
        {"maximum_far_above_c1": .005, "maximum_auprc_drop": .005, "maximum_recall_drop": .01})
    assert passed and all(checks.values())
    c2s["metrics"]["macro_f1"] = .79
    _, passed = evaluate_seed7_gate(c1, c2, c2s, audit, audit, .2, .1,
                                    {"maximum_far_above_c1": .005, "maximum_auprc_drop": .005,
                                     "maximum_recall_drop": .01})
    assert not passed
