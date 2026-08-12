import pytest
from diffusion import StageAwareTimestepScheduler


def test_epoch_only_schedule_ignores_stage_and_reaches_target():
    scheduler=StageAwareTimestepScheduler("epoch_only",2,{s:5 for s in ("normal","early","middle","stable")})
    assert {scheduler.timestep(s,0,8) for s in ("normal","early","middle","stable")} == {2}
    assert {scheduler.timestep(s,7,8) for s in ("normal","early","middle","stable")} == {5}


def test_stage_aware_targets_order_and_integer_timesteps():
    scheduler=StageAwareTimestepScheduler("stage_aware",2,{"normal":5,"early":3,"middle":4,"stable":5})
    final=scheduler.epoch_timesteps(7,8)
    assert final == {"normal":5,"early":3,"middle":4,"stable":5}
    assert final["early"] <= final["middle"] <= final["stable"]
    assert all(isinstance(v,int) and v>=1 for epoch in range(8) for v in scheduler.epoch_timesteps(epoch,8).values())
    assert scheduler.t_critical == 1


def test_invalid_stage_or_frozen_start_rejected():
    with pytest.raises(ValueError): StageAwareTimestepScheduler("stage_aware",3,{"normal":5,"early":3,"middle":4,"stable":5})
