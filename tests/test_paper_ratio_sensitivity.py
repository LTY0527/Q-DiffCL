import numpy as np

from scripts.run_paper_ratio_sensitivity import ratio_mask


def test_ratio_mask_preserves_soft_allocation_and_requested_count():
    source={"criticality":{"composite":[[0.,1.,2.,3.,4.]],"hard_mask":[[0,0,0,1,1]],"soft_mask":[[0,0,0,1,1]]}}
    result=ratio_mask(source,.4)["criticality"]
    assert np.asarray(result["hard_mask"]).sum()==2
    soft=np.asarray(result["soft_mask"])
    assert np.all((soft>0)&(soft<1))
    assert result["critical_ratio"]==.4
