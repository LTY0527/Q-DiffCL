import numpy as np
import yaml
from pathlib import Path

from frequency import build_criticality
from scripts.run_r1_des_ablation import STAGE_A_VARIANTS,STAGE_B_VARIANTS,variant_settings


def _config():return yaml.safe_load(Path("configs/r1_des_ablation.yaml").read_text(encoding="utf-8"))


def test_des_weights_are_frozen_normalized_and_components_zeroed():
    config=_config();expected={"W/O_D":(0,.6,.4),"W/O_E":(5/7,0,2/7),"W/O_S":(.625,.375,0),"D_ONLY":(1,0,0),"E_ONLY":(0,1,0),"S_ONLY":(0,0,1)}
    for name,weights in expected.items():
        value=variant_settings(config,name);actual=tuple(value[k] for k in ("weight_discriminative","weight_early","weight_run_stability"))
        assert np.allclose(actual,weights) and np.isclose(sum(actual),1)
    assert len(STAGE_A_VARIANTS)==3 and len(STAGE_B_VARIANTS)==3


def test_variant_masks_are_rebuilt_train_only_and_can_differ():
    rng=np.random.default_rng(4);features=[];labels=[];stages=[];uids=[]
    for kind in (0,1):
        for run in range(4):
            uid=f"training:fault_{kind:02d}:r{run}"
            for window in range(5):
                label=int(kind and window>=2);x=rng.normal(size=(2,5));x[0,kind]+=label*(1+run)
                features.append(x);labels.append(label);stages.append("early" if label and window<4 else "stable" if label else "prefault");uids.append(uid)
    bundle={"labels":np.asarray(labels),"run_uid":np.asarray(uids)};config=_config();masks=[]
    for name in ("FULL_DES","W/O_D","W/O_E","W/O_S"):
        record=build_criticality(np.asarray(features),bundle,np.asarray(stages),variant_settings(config,name));assert record["fit_split"]=="train";masks.append(record["masks"]["composite"])
    assert any(not np.array_equal(masks[0],mask) for mask in masks[1:])
