from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from diffusion import DiffusionSchedule, spectral_noise_variance
from utils import write_json


def read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); fields=sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w",newline="",encoding="utf-8-sig") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _top_mask(score: np.ndarray, ratio: float) -> np.ndarray:
    flat=score.ravel(); count=max(1,int(round(len(flat)*ratio))); selected=np.argpartition(flat,-count)[-count:]
    result=np.zeros(len(flat),bool); result[selected]=True; return result.reshape(score.shape)


def _jaccard(a: np.ndarray,b: np.ndarray) -> float:
    return float(np.sum(a&b)/max(np.sum(a|b),1))


def audit(config: dict[str, Any]) -> dict[str, Any]:
    rows=[]; schedule=DiffusionSchedule.cosine(50,"cpu")
    masks={"3W":read("outputs/qdiffcl_final_5seed/masks/3w_FINAL_DE.json")["criticality"],
           "TEP":read("outputs/qdiffcl_final_5seed/masks/tep_FINAL_DE.json")["criticality"]}
    for dataset,item in masks.items():
        composite=np.asarray(item["composite"],float); d=np.asarray(item["discriminative"],float); e=np.asarray(item["early"],float)
        reference=_top_mask(composite,.3)
        for ratio in (.2,.3,.4):
            hard=_top_mask(composite,ratio); soft=np.asarray(item["soft_mask"],np.float32) if ratio==.3 else hard.astype(np.float32)
            variance=spectral_noise_variance(schedule.alpha_bars,*soft.shape,"selective",3,True,torch.as_tensor(soft),1,5)
            rows.append({"audit":"critical_ratio","dataset":dataset,"setting":ratio,"groups_or_bins":int(hard.sum()),
                         "jaccard_to_frozen_030":_jaccard(hard,reference),"D_selected_mean":float(d[hard].mean()),
                         "E_selected_mean":float(e[hard].mean()),"total_noise_budget":float(variance.mean()),
                         "downstream_performance_support":"SUPPORTED" if ratio==.3 else "UNSUPPORTED / DO NOT CLAIM"})
        soft=np.asarray(item["soft_mask"],np.float32)
        for timestep in (3,5,8):
            variance=spectral_noise_variance(schedule.alpha_bars,*soft.shape,"selective",3,True,torch.as_tensor(soft),1,timestep)
            rows.append({"audit":"timestep","dataset":dataset,"setting":timestep,"total_noise_budget":float(variance.mean()),
                         "critical_budget":float(variance[torch.as_tensor(soft>=.5)].mean()),
                         "noncritical_budget":float(variance[torch.as_tensor(soft<.5)].mean()),
                         "downstream_performance_support":"SUPPORTED" if timestep==5 else "UNSUPPORTED / DO NOT CLAIM"})
    protocol=read("outputs/paper_final_protocol/dry_run_manifest.json")
    rng=np.random.default_rng(26082026)
    for dataset,records in (("3W",protocol["three_w"]),("TEP",protocol["tep"])):
        for outer in records:
            groups=outer["groups"]["train"]
            for ratio in (.25,.5,1.0):
                count=max(1,int(round(len(groups)*ratio))); chosen=sorted(rng.choice(groups,count,replace=False).tolist()) if ratio<1 else sorted(groups)
                rows.append({"audit":"limited_data_group_dry_run","dataset":dataset,"setting":ratio,
                             "outer_split_seed":outer["outer_split_seed"],"groups_or_bins":len(chosen),
                             "group_sha256":hashlib.sha256("\n".join(chosen).encode()).hexdigest(),
                             "downstream_performance_support":"UNSUPPORTED / DO NOT CLAIM"})
    for dataset,native in (("3W","native missingness; no added MCAR"),("TEP","fixed MCAR 30%")):
        for ratio in (.1,.3):
            supported=dataset=="TEP" and ratio==.3
            rows.append({"audit":"missingness","dataset":dataset,"setting":ratio,"protocol":native,
                         "downstream_performance_support":"SUPPORTED" if supported else "UNSUPPORTED / DO NOT CLAIM"})
    output=Path("docs/paper_evidence/robustness_sensitivity.csv"); write_csv(output,rows)
    result={"status":"ROBUSTNESS_PROTOCOL_AUDIT_COMPLETE","rows":rows,"frozen_setting_changed":False,
            "performance_gaps":["limited-data 25/50%","MCAR 10%","critical ratio 0.20/0.40","noncritical timestep alternatives"],
            "paper_final_outer_test_run":False}
    write_json(Path("outputs/paper_evidence_chain/robustness_audit.json"),result)
    lines=["# Robustness / Sensitivity Audit","",
           "冻结设置仍为 critical ratio `0.30`、timesteps `1/5`；本审计不根据结果更换设置。","",
           "- critical-ratio 与 timestep 表仅证明 mask/budget 计算可复现；除冻结点外没有同协议下游训练结果，不能声称性能敏感性。",
           "- limited-data 25/50/100% 已完成 grouped sampling dry-run 与 group hash，但尚无模型性能结果。",
           "- TEP 现有结果覆盖固定 MCAR 30%；3W 保持 native missingness。MCAR 10% 仍缺失。",
           "- 所有缺失格均在 CSV 标记 `UNSUPPORTED / DO NOT CLAIM`，paper-final outer test 未运行。","",
           f"原始审计表：`{output.as_posix()}`。"]
    Path("docs/paper_evidence/robustness_sensitivity.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return result


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="configs/paper_evidence_chain.yaml")
    args=parser.parse_args(); config=yaml.safe_load(Path(args.config).read_text(encoding="utf-8")); result=audit(config); print(json.dumps({"status":result["status"],"gaps":result["performance_gaps"]},ensure_ascii=False))


if __name__=="__main__": main()
