import copy

from scripts.run_3w_r1_r2_5seed_reliability import validate_config
from scripts.summarize_3w_r1_r2_5seed_reliability import paired_summary, r1_decision, r2_decision


def _paired(values):
    rows = {str(index): {"macro_f1": value, "auprc_fault_vs_normal": .01,
                         "auprc_multiclass_macro": .01, "far": -.01}
            for index, value in enumerate(values)}
    return paired_summary(rows)


def test_config_freezes_only_two_new_seeds_and_r2_weights():
    config = {"existing_seeds": [42, 43, 44], "new_seeds": [45, 46],
              "all_seeds": [42, 43, 44, 45, 46],
              "r2_weights": {"weight_discriminative": .4, "weight_early": .24,
                             "weight_run_stability": .16, "weight_multiclass": .2}}
    validate_config(config)
    bad = copy.deepcopy(config); bad["new_seeds"] = [45, 47]
    try: validate_config(bad)
    except ValueError: pass
    else: raise AssertionError("seed set must remain frozen")


def test_r1_candidate_requires_four_wins_and_survives_best_seed_removal():
    gate = {"minimum_macro_f1_positive_seeds": 4, "minimum_macro_f1_mean_gain": .015,
            "minimum_multiclass_auprc_nonworse_seeds": 3, "maximum_far_mean_increase": 0.,
            "minimum_leave_best_out_macro_f1_mean_gain": 0.}
    summary = {"R1-UNIFORM": _paired([.02, .02, .02, .02, -.001])}
    assert r1_decision(summary, gate)[0] == "R1_5SEED_STABLE_CANDIDATE"
    summary["R1-UNIFORM"] = _paired([.20, -.01, -.01, -.01, -.01])
    assert r1_decision(summary, gate)[0] == "R1_5SEED_EXPLORATORY_ONLY"


def test_r2_candidate_requires_majority_vs_uniform_and_nonworse_vs_r1():
    gate = {"minimum_macro_f1_positive_seeds_vs_uniform": 3,
            "minimum_macro_f1_nonworse_seeds_vs_r1": 3,
            "minimum_binary_auprc_nonworse_seeds_vs_uniform": 3,
            "minimum_multiclass_auprc_nonworse_seeds_vs_uniform": 3,
            "minimum_binary_auprc_mean_gain_vs_uniform": 0.,
            "minimum_multiclass_auprc_mean_gain_vs_uniform": 0.}
    summary = {"R2-UNIFORM": _paired([.02, .02, .02, -.01, -.01]),
               "R2-R1": _paired([.01, 0., .01, -.01, -.01])}
    assert r2_decision(summary, gate)[0] == "R2_5SEED_CANDIDATE"
    summary["R2-R1"] = _paired([.01, -.01, -.01, -.01, -.01])
    assert r2_decision(summary, gate)[0] == "R2_EXTENSION_ONLY"
