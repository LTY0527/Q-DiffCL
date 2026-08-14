from scripts.summarize_3w_seed_source_audit import mechanism_decision


def test_mechanism_decision_requires_clear_margin():
    source = {
        "A_DIFFUSION": {"binary_auprc_std": 0.10, "class_9_recall_std": 0.08},
        "B_ENCODER": {"binary_auprc_std": 0.04, "class_9_recall_std": 0.03},
        "C_PROBE": {"binary_auprc_std": 0.03, "class_9_recall_std": 0.02},
    }
    status, detail = mechanism_decision(source, 1.25)
    assert status == "DIFFUSION_RANDOMNESS_DOMINANT"
    assert detail["class9_primary_source"] == "A_DIFFUSION"


def test_mechanism_decision_reports_mixed_when_scores_are_close():
    source = {
        "A_DIFFUSION": {"binary_auprc_std": 0.05, "class_9_recall_std": 0.04},
        "B_ENCODER": {"binary_auprc_std": 0.045, "class_9_recall_std": 0.04},
        "C_PROBE": {"binary_auprc_std": 0.04, "class_9_recall_std": 0.04},
    }
    status, _ = mechanism_decision(source, 1.25)
    assert status == "MIXED_SEED_INSTABILITY"
