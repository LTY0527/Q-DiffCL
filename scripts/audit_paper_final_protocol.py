from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from datasets.three_w import discover_instances
from scripts.run_3w_final_primary_grouped import FINAL_PRIMARY_CLASSES, grouped_split, split_coverage
from utils import write_json


def _hash(items: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(items)).encode("utf-8")).hexdigest()


def _three_w(config: dict[str, Any], data_root: Path) -> list[dict[str, Any]]:
    instances = [item for item in discover_instances(data_root)
                 if item.source == "WELL" and item.event_class in FINAL_PRIMARY_CLASSES]
    targets: dict[str, set[int]] = defaultdict(set)
    for item in instances:
        targets[item.well_id].add(FINAL_PRIMARY_CLASSES.index(item.event_class))
    settings = config["datasets"]["three_w"]; counts = settings["wells"]
    expected = {"train": int(counts["train"]), "validation": int(counts["inner_validation"]),
                "test": int(counts["outer_test"])}
    previous: list[set[str]] = []; records = []
    minimum = {"train": 3, "validation": 2, "test": 2}
    for seed in settings["outer_split_seeds"]:
        split = grouped_split(set(targets), targets, expected, minimum, int(seed), previous, .5); previous.append(split["test"])
        overlap = {f"{a}_{b}": sorted(split[a] & split[b]) for a, b in (("train","validation"),("train","test"),("validation","test"))}
        records.append({"outer_split_seed": int(seed), "groups": {name: sorted(value) for name,value in split.items()},
                        "group_hash": {name:_hash(list(value)) for name,value in split.items()},
                        "coverage": split_coverage(split,targets), "overlap": overlap})
    return records


def _tep(config: dict[str, Any], fixed_manifest: Path) -> list[dict[str, Any]]:
    manifest=json.loads(fixed_manifest.read_text(encoding="utf-8")); groups: dict[str,list[str]]=defaultdict(list)
    for split in manifest["splits"].values():
        with np.load(split["path"],allow_pickle=False) as archive:
            for run in np.unique(archive["run_uid"].astype(str)):
                match = run.split(":")[1]; groups[match].append(run)
    groups={kind:sorted(set(values)) for kind,values in groups.items()}; settings=config["datasets"]["tep"]; records=[]
    for seed in settings["outer_split_seeds"]:
        rng=np.random.default_rng(int(seed)); outer=[]; validation=[]; train=[]
        for kind,values in sorted(groups.items()):
            order=np.asarray(values,dtype=object)[rng.permutation(len(values))].tolist()
            n_outer=max(1,int(round(len(order)*float(settings["outer_fraction"]))))
            remainder=order[n_outer:]; n_val=max(1,int(round(len(remainder)*float(settings["inner_validation_fraction_of_remainder"]))))
            outer.extend(order[:n_outer]); validation.extend(remainder[:n_val]); train.extend(remainder[n_val:])
        split={"train":set(train),"validation":set(validation),"test":set(outer)}
        overlap={f"{a}_{b}":sorted(split[a]&split[b]) for a,b in (("train","validation"),("train","test"),("validation","test"))}
        coverage={name:{kind:sum(run.split(":")[1]==kind for run in values) for kind in groups} for name,values in split.items()}
        records.append({"outer_split_seed":int(seed),"groups":{name:sorted(value) for name,value in split.items()},
                        "group_hash":{name:_hash(list(value)) for name,value in split.items()},"coverage":coverage,"overlap":overlap})
    return records


def audit(config: dict[str, Any], data_root: Path) -> dict[str, Any]:
    three_w=_three_w(config,data_root); tep=_tep(config,Path("outputs/fixed_diffusion_views/views_manifest.json"))
    all_records=three_w+tep; disjoint=all(not any(record["overlap"].values()) for record in all_records)
    frozen_checks={"weights":config["algorithm"]["criticality_weights"]=={"D":.5,"E":.5,"S":0.0},
                   "critical_ratio":float(config["algorithm"]["critical_ratio"])==.3,
                   "timesteps":(int(config["algorithm"]["t_critical"]),int(config["algorithm"]["t_noncritical"]))==(1,5),
                   "rho_grid":list(map(float,config["algorithm"]["rho_candidates"]))==[0,.25,.5,.75,1],
                   "outer_test_selection_forbidden":bool(config["selection"]["forbidden_outer_test_selection"])}
    result={"status":"PAPER_FINAL_PROTOCOL_DRY_RUN_GO" if disjoint and all(frozen_checks.values()) else "PAPER_FINAL_PROTOCOL_HOLD",
            "three_w":three_w,"tep":tep,"group_disjoint":disjoint,"frozen_checks":frozen_checks,
            "outer_test_metrics_read":False,"outer_training_run":False,
            "fit_scope":{"scaler":"outer-train","imputation":"outer-train","criticality_D_E":"outer-train",
                         "rho":"inner-validation","threshold":"inner-validation","outer_test":"evaluation-only"}}
    write_json(Path(config["output"]["manifest"]),{"three_w":three_w,"tep":tep,"outer_metrics":None})
    write_json(Path(config["output"]["audit"]),result)
    return result


def report(config: dict[str, Any], result: dict[str, Any]) -> None:
    def table(records):
        lines=["| Outer seed | Train groups | Inner-val groups | Outer-test groups | Disjoint |","|---:|---:|---:|---:|---|"]
        for row in records: lines.append(f"| {row['outer_split_seed']} | {len(row['groups']['train'])} | {len(row['groups']['validation'])} | {len(row['groups']['test'])} | {not any(row['overlap'].values())} |")
        return "\n".join(lines)
    text=f"""# Q-DiffCL Paper-final Protocol

状态：`{result['status']}`。本阶段仅完成协议锁定与 dry-run；没有训练 outer 模型，也没有读取 outer-test 指标。

## 冻结方法

- FINAL_QDIFFCL：`0.5D + 0.5E`，critical ratio `0.30`，timesteps `1/5`，soft allocation、TCN、Hard SupCon、Original batching、frozen Linear Probe。
- DCBR：domain-level validation-calibrated `rho ∈ {{0,.25,.5,.75,1}}`；不学习 controller，推理新增参数为 0。
- primary metric：Macro-F1；secondary：AUPRC、FAR、Early Recall、Detection Delay、per-group performance。

## 3W repeated grouped outer holdout

{table(result['three_w'])}

同一 WELL 严禁跨 outer-train、inner-validation、outer-test。每个 outer split 使用 20/8/8 WELL，inner validation 只校准 rho、threshold 和 early stopping。

## TEP run-level nested grouped evaluation

{table(result['tep'])}

Run 是最小分组单位；同一 Run 的窗口绝不跨 split。各 fault type 与 normal 分层分配。

## Fit scope 与 leakage rule

scaler、插补、feature/criticality D/E、frequency statistics 仅由 outer-train 拟合；rho、threshold、early stopping 仅使用 inner validation；outer-test 只进行一次冻结评估。任何 outer-test 后的算法、候选网格或阈值修改均禁止。

## Seeds 与统计

- 3W model seeds：`{config['datasets']['three_w']['model_seeds']}`。
- TEP model seeds：`{config['datasets']['tep']['model_seeds']}`。
- 报告 mean±std、paired delta、positive/non-worse count、worst seed、LOSO 与 2,000 次 WELL/Run bootstrap 95% CI。

## Final freeze statement

当前长期开发 test 不得称 untouched。只有此 manifest 中预注册、未参与任何后续选择的 outer groups 才可作为 paper-final evaluation；outer 结果产生后项目进入只分析、不改算法状态。
"""
    Path(config["output"]["report"]).write_text(text,encoding="utf-8")


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="configs/paper_final_protocol.yaml"); parser.add_argument("--data-root",type=Path,required=True)
    args=parser.parse_args(); config=yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if config.get("amendment", {}).get("version") == "windowref-coverage-v2":
        from scripts.amend_paper_final_protocol import amend, report as amendment_report
        result=amend(config,args.data_root); amendment_report(config,result)
    else:
        result=audit(config,args.data_root); report(config,result)
    print(json.dumps({"status":result["status"],"outer_test_metrics_read":False},ensure_ascii=False))


if __name__=="__main__": main()
