import numpy as np

from diffusion import DiffusionSchedule, domain_shortcut_timestep, matched_domain_shortcut_variance
from frequency import build_domain_shortcut_score


def test_domain_score_is_train_normal_prefault_only_and_bounded():
    rng=np.random.default_rng(2);x=rng.normal(size=(12,2,4));labels=np.asarray([0,0,1,1]*3);stages=np.asarray(["prefault","prefault","early","stable"]*3);domains=np.repeat(["W1","W2","W3"],4)
    x[domains=="W2",0,1]+=3;x[domains=="W3",0,1]-=3
    first=build_domain_shortcut_score(x,labels,stages,domains)
    changed=x.copy();changed[labels==1]+=100
    second=build_domain_shortcut_score(changed,labels,stages,domains)
    assert first["fit_split"]=="train" and first["normal_prefault_only"] and first["test_or_validation_used"] is False
    assert np.all((first["domain_score"]>=0)&(first["domain_score"]<=1))
    assert np.allclose(first["domain_score"],second["domain_score"])
    assert set(first["valid_domain_ids"])=={"W1","W2","W3"}


def test_dsfd_fixed_formula_endpoints_and_bounds():
    fault=np.asarray([[1.,0.,0.]]);domain=np.asarray([[1.,1.,0.]])
    t=domain_shortcut_timestep(fault,domain)
    assert np.allclose(t,[[1.,5.,3.]]) and t.min()>=1 and t.max()<=5


def test_budget_matching_does_not_increase_protected_variance():
    fault=np.asarray([[1.,.8,.2,0.,0.]],np.float32);domain=np.asarray([[1.,.9,.9,.2,1.]],np.float32)
    schedule=DiffusionSchedule.cosine(50,"cpu");variance,audit=matched_domain_shortcut_variance(schedule.alpha_bars,fault,domain,False)
    assert audit["protected_variance_not_increased"] and audit["budget_adjustment_only_low_fault"]
    assert audit["maximum_variance_respected"] and audit["minimum_timestep_respected"] and audit["maximum_timestep_respected"]
    assert audit["budget_error_fraction"]<.02 and np.isfinite(variance.numpy()).all()
