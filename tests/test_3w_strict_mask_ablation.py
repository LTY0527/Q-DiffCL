import json
from pathlib import Path

import numpy as np

from scripts.run_3w_strict_mask_ablation import STRICT_CLASSES, load_strict_manifest


def test_strict_manifest_is_disjoint_and_classes_are_audit_selected():
    split = load_strict_manifest(Path("configs/3w_strict_split_manifest.json"))
    assert set(split["train"]).isdisjoint(split["validation"])
    assert set(split["train"]).isdisjoint(split["test"])
    assert set(split["validation"]).isdisjoint(split["test"])
    assert STRICT_CLASSES == (0, 2, 4, 7, 8, 9)


def test_process_only_is_exact_prefix_and_mask_channels_are_separate():
    combined = np.arange(4 * 44 * 3).reshape(4, 44, 3)
    process = combined[:, :22]
    mask = combined[:, 22:]
    assert process.shape[1] == 22 and mask.shape[1] == 22
    assert np.array_equal(process, combined[:, :22])
    assert not np.shares_memory(process.copy(), mask)
