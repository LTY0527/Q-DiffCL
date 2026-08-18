import numpy as np
import pytest

from diffusion import DiffusionSchedule, cross_domain_safe_timestep, cross_domain_safe_variance
from frequency import (build_tep_cross_domain_safety,
                       build_three_w_cross_domain_safety, stratified_run_folds)


SETTINGS = {"critical_ratio": .3, "weight_discriminative": .5, "weight_early": .3,
            "weight_run_stability": .2, "bootstrap_repeats": 64, "bootstrap_seed": 19}


def _fixture():
    rng=np.random.default_rng(3); x=[]; y=[]; stages=[]; wells=[]; runs=[]
    for kind in (0,1,2):
        for unit in range(8):
            well=f"W{unit}"; uid=f"training:fault_{kind:02d}:{well}"
            for window in range(5):
                label=int(kind>0 and window>=2); value=rng.normal(0,.2,(2,5))
                if label: value[kind-1,kind]+=2
                x.append(value); y.append(label); stages.append("early" if label and window<4 else "stable" if label else "prefault")
                wells.append(well); runs.append(uid)
    return np.asarray(x,np.float32), {"labels":np.asarray(y),"run_uid":np.asarray(runs)}, np.asarray(stages), np.asarray(wells)


def test_pseudo_unseen_wells_are_disjoint_and_safe_probability_is_bounded():
    x,bundle,stages,wells=_fixture(); result=build_three_w_cross_domain_safety(x,bundle,stages,wells,SETTINGS)
    assert result["fit_split"]=="train" and result["test_or_validation_used"] is False
    assert len(result["valid_unit_ids"])==8
    assert np.all((result["safe_prob"]>=0)&(result["safe_prob"]<=1))
    assert np.allclose(result["safe_prob"],1-result["unsafe_rate"])


def test_tep_folds_are_stratified_disjoint_and_frozen_to_eight():
    x,bundle,stages,_=_fixture(); folds=stratified_run_folds(bundle["run_uid"],8,7)
    flattened=np.concatenate(folds); assert len(flattened)==len(np.unique(flattened))
    result=build_tep_cross_domain_safety(x,bundle,stages,SETTINGS,8,7)
    assert result["fold_count"]==8 and result["test_or_validation_used"] is False
    with pytest.raises(ValueError): build_tep_cross_domain_safety(x,bundle,stages,SETTINGS,4,7)


def test_support_below_three_is_conservatively_zero():
    x,bundle,stages,wells=_fixture(); keep=np.isin(wells,["W0","W1"])
    result=build_three_w_cross_domain_safety(x[keep],{k:v[keep] for k,v in bundle.items()},stages[keep],wells[keep],SETTINGS)
    assert np.all(result["safe_prob"]==0)


def test_cdvs_protects_r1_and_unsafe_bins_and_matches_budget():
    soft=np.asarray([[1.,.8,.5,.2,0.]],np.float32); safe=np.asarray([[0.,0.,0.,.5,1.]],np.float32)
    r1,cdvs=cross_domain_safe_timestep(soft,safe)
    assert np.allclose(cdvs[r1<=3],r1[r1<=3])
    schedule=DiffusionSchedule.cosine(50,"cpu")
    variance,audit=cross_domain_safe_variance(schedule.alpha_bars,soft,safe,False)
    assert audit["protected_timestep_not_increased"] and audit["protected_variance_not_increased"]
    assert audit["unsafe_variance_not_increased"] and audit["budget_adjustment_only_safe_noncritical"]
    assert audit["maximum_variance_respected"] and audit["finite"]
    assert audit["budget_error_fraction"]<.02 and np.isfinite(variance.numpy()).all()
