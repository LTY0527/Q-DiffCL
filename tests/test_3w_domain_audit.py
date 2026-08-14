import numpy as np

from scripts.audit_3w_domain_shift_mask_shortcut import stable_sample, summary_views


def test_summary_views_separate_process_and_mask():
    raw = np.asarray([[1.0, np.nan], [3.0, 4.0]])
    process = np.asarray([[1.0, 0.0], [3.0, 4.0]])
    mask = np.asarray([[True, False], [True, True]])
    raw_s, process_s, mask_s, combined = summary_views(raw, process, mask, 0, 2)
    assert len(process_s) == 4 and len(mask_s) == 2 and len(combined) == 6
    assert np.array_equal(mask_s, np.asarray([1.0, 0.5]))
    assert not np.array_equal(process_s[:2], mask_s)


def test_stable_window_sampling_is_reproducible():
    class Ref:
        def __init__(self, start): self.start = start
    refs = [Ref(i) for i in range(20)]
    assert [x.start for x in stable_sample(refs, 5, "WELL-1")] == [x.start for x in stable_sample(refs, 5, "WELL-1")]
