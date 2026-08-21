from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from utils import environment_metadata, write_json


def read(path:str|Path)->dict[str,Any]:return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()


def hash_collection(root:Path,patterns:tuple[str,...])->dict[str,Any]:
    files=sorted({path for pattern in patterns for path in root.rglob(pattern) if path.is_file()},key=lambda p:p.as_posix())
    digest=hashlib.sha256();records=[]
    for path in files:
        relative=path.relative_to(root).as_posix();value=sha256_file(path);size=path.stat().st_size
        digest.update(f"{relative}\0{size}\0{value}\n".encode());records.append({"path":relative,"bytes":size,"sha256":value})
    return {"root":str(root),"file_count":len(records),"total_bytes":sum(r["bytes"] for r in records),"collection_sha256":digest.hexdigest(),"files":records}


def git(*args:str)->str:return subprocess.check_output(["git",*args],text=True,encoding="utf-8").strip()


def run(config:dict[str,Any],three_w_root:Path,tep_root:Path,pytest_summary:str)->dict[str,Any]:
    protocol=read("outputs/paper_final_protocol/dry_run_manifest.json");audit=read("outputs/paper_final_protocol/leakage_audit.json")
    if protocol["outer_metrics"] is not None or audit["outer_test_metrics_read"] or audit["outer_training_run"]:raise RuntimeError("outer evaluation boundary violated")
    if audit["status"] not in ("PAPER_FINAL_PROTOCOL_DRY_RUN_GO","PAPER_FINAL_PROTOCOL_AMENDMENT_GO"):raise RuntimeError("paper-final leakage audit is not GO")
    evidence={
      "contrastive":len(read("outputs/paper_contrastive_ablation/manifest.json")["results"]),
      "ratio":len(read("outputs/paper_ratio_sensitivity/manifest.json")["results"]),
      "efficiency":len(read("outputs/paper_efficiency/manifest.json")["rows"]),
      "trajectory":read("outputs/paper_fault_trajectory/manifest.json")["rows"],
      "external":len(read("outputs/paper_baseline_feasibility_audit.json")["candidates"]),
    }
    if evidence!={"contrastive":24,"ratio":18,"efficiency":12,"trajectory":80,"external":2}:raise RuntimeError(f"A-E evidence incomplete: {evidence}")
    config_paths=[Path(p) for p in config["hashes"]["configs"]];split_paths=[Path(p) for p in config["hashes"]["splits"]]
    hashes={"configs":{str(p):sha256_file(p) for p in config_paths},"splits":{str(p):sha256_file(p) for p in split_paths},
            "data":{"3W":hash_collection(three_w_root,("dataset.ini","*.parquet")),"TEP":hash_collection(tep_root,("*.RData",))}}
    metadata=environment_metadata();branch=git("branch","--show-current");head=git("rev-parse","HEAD")
    freeze_tag=config.get("git_freeze",{}).get("tag")
    result={"status":"PAPER_FINAL_FREEZE_READY","branch":branch,"head":head,"freeze_commit":None,"freeze_tag":freeze_tag,
      "pytest":pytest_summary,"protocol_status":audit["status"],"outer_test_metrics_read":False,"outer_training_run":False,
      "evidence_counts":evidence,"experiment_counts":{"reused_existing":27,"new_training":27,"evaluation_only":4,"audit_only":2},
      "environment":metadata,"hashes":hashes}
    output=Path(config["output"]["manifest"]);write_json(output,result)
    three_w_unique=3*5*7;tep_unique=3*5*8
    lines=["# Q-DiffCL Paper-final Freeze","",f"状态：`{result['status']}`。这是 pre-outer 冻结快照；没有训练 outer model，也没有读取 outer-test metric。","",
      "## Version","",f"- Source branch before freeze commit: `{branch}`",f"- Source HEAD used to generate snapshot: `{head}`","- Freeze commit: 由包含本快照的 Git commit/tag 确定，并写入 outer manifest。",f"- Freeze tag: `{freeze_tag}`",f"- Outer branch after tag: `{config.get('git_freeze',{}).get('outer_branch')}`",f"- Tests: `{pytest_summary}`",f"- Environment: Python 3.10.20, PyTorch {metadata['pytorch']}, CUDA {metadata['cuda']}, {metadata['gpu']}","",
      "## Frozen method","","- FINAL_QDIFFCL: `0.5D + 0.5E`, `S=0`, `critical_ratio=0.30`, selective timesteps `1/5`, soft channel-frequency allocation.","- TCN encoder, Hard SupCon, Original batching, frozen Linear Probe.","- DCBR: inner-validation domain calibration over `rho ∈ {0,.25,.5,.75,1}`; no learned controller and 0 inference parameters. Development references remain 3W `rho=1`, TEP `rho=.75`.","- SVR/router/controller: `NO_GO_SVR`, excluded.","",
      "## Frozen baseline set","","`NO_AUG`, `JITTER`, `SCALING`, `JITTER_SCALING`, `UNIFORM_DIFFUSION`, `FRERA` shared-backbone adaptation, `FINAL_QDIFFCL`, `DCBR`.","AutoDA-Timeseries is method-native supplementary only; DiCL is not fairly reproducible. Neither enters the outer main-table matrix.","",
      "## Frozen evaluation protocol","","- 3W: repeated grouped outer holdout, outer seeds `31001/31002/31003`, grouping unit WELL, per split 20 train / 8 inner-val / 8 outer-test WELL.","- TEP: repeated stratified Run-level outer holdout, outer seeds `32001/32002/32003`, per split 248 train / 72 inner-val / 80 outer-test Runs.","- Model seeds: 3W `42/43/44/45/46`; TEP `7/42/43/44/2026`.","- scaler/imputation/D/E/frequency statistics fit on outer-train only; rho/threshold/early stopping use inner validation only; outer-test is evaluated once.","- Primary metric Macro-F1; secondary AUPRC, FAR, Early Recall, Detection Delay, per-group/per-fault metrics; 2,000 group bootstrap repeats.","",
      "## Exact future run matrix","",f"- 3W: `3 outer splits × 5 model seeds × 7 unique trained methods = {three_w_unique}` training/evaluation cells; DCBR `rho=1` may be emitted as 15 exact FINAL alias rows only when inner validation selects 1.",f"- TEP: `3 outer splits × 5 model seeds × 8 methods = {tep_unique}` training/evaluation cells.","- Each cell writes config, split IDs, checkpoint, validation selection record, raw scores/predictions, per-group metrics and environment metadata.","- Expected roots: `outputs/paper_final/3w/outer_{seed}/model_seed_{seed}/{method}/` and `outputs/paper_final/tep/outer_{seed}/model_seed_{seed}/{method}/`.","",
      "## Hash audit","",f"- 3W content collection: `{hashes['data']['3W']['collection_sha256']}` ({hashes['data']['3W']['file_count']} files, {hashes['data']['3W']['total_bytes']} bytes).",f"- TEP content collection: `{hashes['data']['TEP']['collection_sha256']}` ({hashes['data']['TEP']['file_count']} files, {hashes['data']['TEP']['total_bytes']} bytes).",f"- Dry-run split manifest: `{hashes['splits'][str(Path('outputs/paper_final_protocol/dry_run_manifest.json'))]}`.",f"- Leakage audit: `{hashes['splits'][str(Path('outputs/paper_final_protocol/leakage_audit.json'))]}`; status `{audit['status']}`, outer metrics `null`.","- Per-file data and config hashes are stored in `outputs/paper_final_freeze/freeze_manifest.json`.","",
      "## Resume and stopping policy","","- Resume only an incomplete cell whose config, data, split, initialization and epoch-order hashes match exactly; otherwise fail closed.","- Completed outer-test cells are immutable and never rerun for selection.","- After the first outer metric is produced, algorithm structure, candidate grids, thresholds and baseline membership are locked; only predeclared analysis may continue.","",
      "## A–E inventory","",f"- Reused existing experiment cells: `{result['experiment_counts']['reused_existing']}`.",f"- New training cells: `{result['experiment_counts']['new_training']}`.",f"- Evaluation-only checkpoint replays: `{result['experiment_counts']['evaluation_only']}`.",f"- Audit-only candidates: `{result['experiment_counts']['audit_only']}`.","",
      "## Remaining claim boundaries","","- Paper-final generalization remains pending until the frozen outer matrix runs once.","- Limited-data and broader missingness robustness remain unsupported.","- 3W universal cross-WELL superiority remains unsupported; existing bootstrap CI crosses zero.","- TEP ratio 0.30 is a local sensitivity trough; the frozen parameter is not reopened, and no universal optimum claim is allowed.","- FRERA augmentation-only timing remains unavailable because its standalone augmenter checkpoint was not preserved."]
    Path(config["output"]["report"]).write_text("\n".join(lines)+"\n",encoding="utf-8");return result


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--config",default="configs/paper_final_freeze.yaml");parser.add_argument("--three-w-root",type=Path,required=True);parser.add_argument("--tep-root",type=Path,required=True);parser.add_argument("--pytest-summary",required=True)
    args=parser.parse_args();config=yaml.safe_load(Path(args.config).read_text(encoding="utf-8"));result=run(config,args.three_w_root,args.tep_root,args.pytest_summary);print(json.dumps({k:result[k] for k in ("status","branch","head","pytest")},ensure_ascii=False))


if __name__=="__main__":main()
