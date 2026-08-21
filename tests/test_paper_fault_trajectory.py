import pytest

from scripts.build_paper_fault_trajectory import relative_window


def test_relative_window_alignment_excludes_transition():
    assert relative_window(145,160,161,16)==-1
    assert relative_window(161,224,161,16)==0
    assert relative_window(177,240,161,16)==1
    with pytest.raises(ValueError):relative_window(145,208,161,16)
