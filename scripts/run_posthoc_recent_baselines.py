from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score

import scripts.run_3w_clean_baseline as base3w
from baselines.posthoc_recent import (AutoTCLAdaptation, FrozenLinearProbe, NativeRecentModel,
                                      SoftCLTAdaptation, TFCRepresentation, TRACKS, fit_autotcl,
                                      fit_linear_probe, fit_softclt, fit_tfc, probe_probabilities,
                                      verify_external_commit)
from datasets.protocol import Run, Standardizer
from frequency import fault_stages
from scripts.run_paper_final_outer import (_canonical_hash, _criticality, _load_selected_tep_runs, _metrics,
                                           _multiclass_metrics, _window_bundle, evaluate_three_w,
                                           split_record)
from scripts.run_paper_final_outer import prepare_three_w, prepare_tep
from scripts.run_stage_frequency_diffusion_mvp import _configure, _runtime
from utils import environment_metadata, seed_everything, select_device, write_json


FORMAL_PYTHON = Path(r"E:\anaconda\envs\qdiffcl\python.exe")
EVIDENCE_CLASS = "POSTHOC_BASELINE_EVIDENCE"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader(); writer.writerows(rows)


def validate_protocol(config: dict[str, Any]) -> dict[str, Any]:
    executable = Path(os.path.abspath(os.sys.executable))
    if executable.resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"H1 must use {FORMAL_PYTHON}, observed {executable}")
    if not torch.cuda.is_available():
        raise RuntimeError("formal qdiffcl CUDA environment is unavailable")
    lock = read_json(config["selection_lock"])
    checks = {
        "evidence_class": config["evidence_class"] == EVIDENCE_CLASS,
        "selection_status": lock["status"] == "POSTHOC_BASELINE_SELECTION_LOCKED",
        "selection_hash": lock["selection_hash"] == config["selection_hash"],
        "methods": [row["method"] for row in lock["selected_methods"]] == config["selected_methods"],
        "outer_unread_at_lock": lock["outer_test_metrics_read_before_lock"] is False,
        "paper_final_source": lock["paper_final_source_commit"] == config["paper_final_source_commit"],
        "expected_cells": lock["expected_total_runs"] == 72,
        "active_replacements": config["active_methods"] == ["TF-C", "SoftCLT", "TS2Vec", "AutoTCL"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"posthoc selection lock changed: {checks}")
    paper = yaml.safe_load(Path(config["paper_final_config"]).read_text(encoding="utf-8"))
    for dataset, key in (("3W", "three_w"), ("TEP", "tep")):
        if list(map(int, paper[key]["outer_seeds"])) != list(map(int, config["benchmark"]["outer_splits"][dataset])):
            raise RuntimeError(f"{dataset} outer splits differ from Paper-final")
    return {"checks": checks, "lock": lock, "paper": paper}


def _prepare_tep_train_validation(paper: dict[str, Any], outer_seed: int) -> dict[str, Any]:
    """Sanity preparation that never windows/materializes the outer-test Runs."""
    stage = paper["tep"]; base = yaml.safe_load(Path(stage["base_config"]).read_text(encoding="utf-8")); _configure(base)
    row = split_record(paper, "TEP", outer_seed); by_uid = {run.run_uid: run for run in _load_selected_tep_runs(paper)}
    registered = set().union(*(set(row["groups"][name]) for name in ("train", "validation", "test")))
    if set(by_uid) != registered:
        raise RuntimeError("TEP Run universe changed")
    groups = {name: [by_uid[uid] for uid in row["groups"][name]] for name in ("train", "validation")}
    scaler = Standardizer().fit_many([run.values for run in groups["train"]])
    bundles = {name: _window_bundle(groups[name], scaler, base) for name in ("train", "validation")}
    stages = {name: fault_stages(bundles[name], base) for name in ("train", "validation")}
    critical = _criticality(bundles["train"]["clean"], bundles["train"], stages["train"], paper["algorithm"])
    runtime = _runtime(base, 0); runtime["diagnosis"] = {"threshold_band_width": .05, "high_correlation_quantile": .90}
    scaler_payload = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                      "fit_groups": sorted(row["groups"]["train"])}
    return {"dataset": "TEP", "outer_seed": outer_seed, "base": base,
            "split": {name: set(row["groups"][name]) for name in ("train", "validation", "test")},
            "bundles": bundles, "stages": stages, "critical": critical, "runtime": runtime,
            "train": bundles["train"]["clean"], "validation": bundles["validation"]["clean"],
            "labels": {name: bundles[name]["labels"] for name in ("train", "validation")},
            "ids": {name: bundles[name]["window_id"] for name in ("train", "validation")},
            "context_hash": _canonical_hash({"dataset": "TEP", "outer_seed": outer_seed, "groups": row["groups"],
                                               "scaler": scaler_payload,
                                               "critical_soft_mask": critical["soft_mask"].tolist()}),
            "sanity_outer_test_materialized": False}


def prepare_context(paper: dict[str, Any], dataset: str, outer_seed: int,
                    device: str, include_test: bool) -> dict[str, Any]:
    if dataset == "TEP" and not include_test:
        return _prepare_tep_train_validation(paper, outer_seed)
    context = prepare_three_w(paper, outer_seed, device) if dataset == "3W" else prepare_tep(paper, outer_seed, device)
    context["sanity_outer_test_materialized"] = include_test
    return context


def stratified_subset(labels: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if limit >= len(labels): return np.arange(len(labels))
    rng = np.random.default_rng(seed); selected = []
    classes = np.unique(labels); each = max(1, limit // len(classes))
    for label in classes:
        candidates = np.flatnonzero(labels == label); selected.extend(rng.choice(candidates, min(each, len(candidates)), replace=False))
    remaining = np.setdiff1d(np.arange(len(labels)), np.asarray(selected), assume_unique=False)
    if len(selected) < limit: selected.extend(rng.choice(remaining, limit - len(selected), replace=False))
    return np.asarray(selected[:limit], dtype=np.int64)


def classes_for(context: dict[str, Any]) -> int:
    return 4 if context["dataset"] == "3W" else 2


def _probe_validation(probability: np.ndarray, labels: np.ndarray, dataset: str) -> tuple[dict[str, float], float | None]:
    if dataset == "3W":
        prediction = probability.argmax(1); fault = labels != 0
        return {"macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
                "auprc": float(average_precision_score(fault.astype(int), 1 - probability[:, 0])),
                "far": float(np.mean(prediction[labels == 0] != 0))}, None
    from metrics import select_binary_threshold
    threshold = float(select_binary_threshold(labels, probability[:, 1])); metrics = _metrics(labels, probability[:, 1], threshold)
    return {key: float(metrics[key]) for key in ("macro_f1", "auprc", "far")}, threshold


def _cell_id(dataset: str, outer_seed: int, model_seed: int, method: str) -> str:
    return f"{dataset.lower()}-outer{outer_seed}-seed{model_seed}-{method.lower().replace('-', '_')}"


def _cell_dir(config: dict[str, Any], dataset: str, outer_seed: int, model_seed: int, method: str,
              sanity: bool = False) -> Path:
    scope = "sanity" if sanity else "benchmark"
    return Path(config["output"]["root"]) / scope / dataset.lower() / f"outer_{outer_seed}" / f"seed_{model_seed}" / method


def _encode(model: Any, method: str, values: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    if method in ("AutoTCL", "TF-C", "SoftCLT"): return model.encode(values, batch_size, device)
    return model.encode(values, batch_size)


def _finite_history(history: list[Any]) -> bool:
    values = []
    for row in history:
        values.append(row.get("loss", 0.0) if isinstance(row, dict) else row)
    return bool(np.isfinite(np.asarray(values, dtype=np.float64)).all())


def train_cell(config: dict[str, Any], context: dict[str, Any], method: str, model_seed: int,
               device: str, sanity: bool) -> tuple[dict[str, Any], Any, Any]:
    output = _cell_dir(config, context["dataset"], context["outer_seed"], model_seed, method, sanity)
    checkpoint = output / "checkpoint.pt"; validation_path = output / "validation.json"
    scope = config["sanity"] if sanity else config["benchmark"]
    epochs = int(scope["epochs"] if sanity else scope["epochs"][context["dataset"]])
    probe_epochs = int(scope["probe_epochs"] if sanity else scope["probe_epochs"][context["dataset"]])
    batch_size = int(config["benchmark"]["batch_size"][context["dataset"]])
    train_idx = stratified_subset(context["labels"]["train"], int(scope["train_windows"]), model_seed) if sanity else np.arange(len(context["train"]))
    val_idx = stratified_subset(context["labels"]["validation"], int(scope["validation_windows"]), model_seed + 1) if sanity else np.arange(len(context["validation"]))
    train_x, train_y = context["train"][train_idx], context["labels"]["train"][train_idx]
    val_x, val_y = context["validation"][val_idx], context["labels"]["validation"][val_idx]
    metadata = {"evidence_class": EVIDENCE_CLASS, "selection_hash": config["selection_hash"],
                "dataset": context["dataset"], "outer_seed": context["outer_seed"], "model_seed": model_seed,
                "method": method, "track": TRACKS[method], "context_hash": context["context_hash"],
                "epochs": epochs, "probe_epochs": probe_epochs, "sanity": sanity,
                "train_indices_sha256": hashlib.sha256(train_idx.tobytes()).hexdigest(),
                "validation_indices_sha256": hashlib.sha256(val_idx.tobytes()).hexdigest(),
                "config_sha256": sha256_file("configs/posthoc_recent_baselines.yaml"),
                "runner_sha256": sha256_file(__file__),
                "adapter_sha256": sha256_file("baselines/posthoc_recent.py")}
    if checkpoint.exists() and validation_path.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False); record = read_json(validation_path)
        if payload["metadata"] != metadata or record["metadata"] != metadata: raise RuntimeError("resume metadata mismatch")
        if method == "AutoTCL":
            model = AutoTCLAdaptation(train_x.shape[1], classes_for(context), context["base"]["training"]["model"] if context["dataset"] == "3W" else context["runtime"]["model"]).to(device)
            model.load_state_dict(payload["model_state"])
        elif method == "TF-C":
            model=TFCRepresentation(train_x.shape[-1],int(config["representation_dim"])).to(device);model.load_state_dict(payload["model_state"])
        elif method == "SoftCLT":
            model_cfg=context["base"]["training"]["model"] if context["dataset"]=="3W" else context["runtime"]["model"]
            model=SoftCLTAdaptation(train_x.shape[1],classes_for(context),model_cfg).to(device);model.load_state_dict(payload["model_state"])
        else:
            model = NativeRecentModel(method, train_x.shape[1], classes_for(context), config, device, output)
            model.load_state_dict(payload["model_state"], train_x, val_x)
        probe = FrozenLinearProbe(payload["representation_dim"], classes_for(context)).to(device)
        probe.load_state_dict(payload["probe_state"]); probe.eval(); record["resumed"] = True
        return record, model, probe
    if checkpoint.exists() or validation_path.exists(): raise RuntimeError(f"incomplete checkpoint pair: {output}")
    seed_everything(model_seed); started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    max_batches = int(scope["max_batches"]) if sanity else None
    if method == "AutoTCL":
        model_cfg = context["base"]["training"]["model"] if context["dataset"] == "3W" else context["runtime"]["model"]
        model = AutoTCLAdaptation(train_x.shape[1], classes_for(context), model_cfg).to(device)
        pretrain = fit_autotcl(model, train_x, train_y, val_x, val_y, epochs, batch_size,
                               float(config["learning_rate"]), float(config["temperature"]), model_seed, device, max_batches)
        adapter_audit = {"implementation": "independent shared-TCN mechanism adaptation", "official_code_copied": False}
    elif method == "TF-C":
        commit=verify_external_commit(config["external_roots"]["TF-C"],"TF-C");model=TFCRepresentation(train_x.shape[-1],int(config["representation_dim"])).to(device)
        pretrain=fit_tfc(model,train_x,val_x,epochs,batch_size,float(config["learning_rate"]),model_seed,device,max_batches)
        adapter_audit={"official_commit":commit,"implementation":"method-native dual time/frequency Transformer with multivariate channel pooling","official_objective":"time/frequency NT-Xent"}
    elif method == "SoftCLT":
        commit=verify_external_commit(config["external_roots"]["SoftCLT"],"SoftCLT");model_cfg=context["base"]["training"]["model"] if context["dataset"]=="3W" else context["runtime"]["model"]
        model=SoftCLTAdaptation(train_x.shape[1],classes_for(context),model_cfg).to(device);pretrain=fit_softclt(model,train_x,val_x,epochs,batch_size,float(config["learning_rate"]),float(config["temperature"]),model_seed,device,max_batches)
        adapter_audit={"source_commit":commit,"implementation":"independent shared-TCN soft-instance-target mechanism adaptation","official_code_copied":False}
    else:
        model = NativeRecentModel(method, train_x.shape[1], classes_for(context), config, device, output)
        pretrain = model.fit(train_x, val_x, epochs); adapter_audit = model.audit
    train_z = _encode(model, method, train_x, batch_size, device); val_z = _encode(model, method, val_x, batch_size, device)
    probe, probe_history = fit_linear_probe(train_z, train_y, val_z, val_y, classes_for(context), probe_epochs,
                                            batch_size, float(config["learning_rate"]), model_seed + 1, device)
    probability = probe_probabilities(probe, val_z, batch_size, device); metrics, threshold = _probe_validation(probability, val_y, context["dataset"])
    model_state = model.state_dict(); representation_dim = int(train_z.shape[1])
    # Method-native repositories (notably REBAR) may create their own audited
    # sub-checkpoints below the cell directory before our wrapper checkpoint.
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": metadata, "model_state": model_state, "probe_state": probe.state_dict(),
                "representation_dim": representation_dim, "threshold": threshold}, checkpoint)
    # Immediate round-trip validation uses the just-written payload without touching outer-test data.
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if loaded["metadata"] != metadata: raise RuntimeError("checkpoint round-trip metadata changed")
    record = {"metadata": metadata, "validation": metrics, "threshold": threshold, "validation_only": True,
              "outer_test_read": False, "outer_test_materialized": bool(context["sanity_outer_test_materialized"]),
              "representation_shape": [int(len(val_z)), representation_dim], "finite_loss": _finite_history(pretrain),
              "pretrain_history": pretrain, "probe_history": probe_history, "adapter_audit": adapter_audit,
              "training_seconds": time.perf_counter() - started,
              "peak_gpu_mib": torch.cuda.max_memory_allocated() / 1024 ** 2, "checkpoint_roundtrip": True, "resumed": False}
    write_json(validation_path, record); return record, model, probe


def _evaluate_three_w_representation(context: dict[str, Any], model: Any, probe: Any, method: str,
                                     batch_size: int, device: str) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]]]:
    ys=[]; predictions=[]; probabilities=[]; groups=[]; faults=[]; instances=[]; stages=[]; starts=[]; ends=[]; delay_by_instance={}
    length = int(context["base"]["protocol"]["window_length"])
    for item in context["by_split"]["test"]:
        refs = context["refs_by_instance"].get(item.instance_id, [])
        if not refs: continue
        x,y = base3w.materialize(refs, {item.instance_id:item}, context["preprocessor"], length, False)
        z = _encode(model, method, x, batch_size, device); prob = probe_probabilities(probe, z, batch_size, device); pred=prob.argmax(1)
        ys.append(y); predictions.append(pred); probabilities.append(prob); groups.extend([item.well_id]*len(y)); faults.extend([item.event_class]*len(y)); instances.extend([item.instance_id]*len(y)); stages.extend([r.stage for r in refs]); starts.extend([r.start for r in refs]); ends.extend([r.start+length-1 for r in refs])
        onset=refs[0].onset_seconds; detections=[r.end_seconds-float(onset) for r,v in zip(refs,pred) if onset is not None and r.target!=0 and r.end_seconds>=onset and v!=0]
        delay_by_instance[item.instance_id]=float(detections[0]) if detections else None
    raw={"label":np.concatenate(ys),"prediction":np.concatenate(predictions),"probability":np.concatenate(probabilities),"group_id":np.asarray(groups),"fault_id":np.asarray(faults),"instance_id":np.asarray(instances),"stage":np.asarray(stages),"start":np.asarray(starts),"end":np.asarray(ends)}
    metrics=_multiclass_metrics(raw["label"],raw["prediction"],raw["probability"]); early=raw["stage"]=="early"; metrics["early_recall"]=float(np.mean(raw["prediction"][early]!=0)) if early.any() else None
    delays=[v for v in delay_by_instance.values() if v is not None]; metrics["detection_delay"]=float(np.mean(delays)) if delays else None; groupwise=[]
    for group in sorted(set(groups)):
        sel=raw["group_id"]==group; row=_multiclass_metrics(raw["label"][sel],raw["prediction"][sel],raw["probability"][sel]); row.update({"group_id":group,"windows":int(sel.sum()),"early_recall":float(np.mean(raw["prediction"][sel&early]!=0)) if np.any(sel&early) else None}); gd=[v for k,v in delay_by_instance.items() if k in set(raw["instance_id"][sel]) and v is not None]; row["detection_delay"]=float(np.mean(gd)) if gd else None; groupwise.append(row)
    return metrics,raw,groupwise


def _evaluate_tep_representation(context: dict[str, Any], model: Any, probe: Any, method: str, threshold: float,
                                 batch_size: int, device: str) -> tuple[dict[str, Any], dict[str,np.ndarray], list[dict[str,Any]]]:
    bundle=context["bundles"]["test"]; z=_encode(model,method,bundle["clean"],batch_size,device); probability=probe_probabilities(probe,z,batch_size,device); scores=probability[:,1]; prediction=(scores>=threshold).astype(np.int64); labels=bundle["labels"]
    raw={"label":labels,"prediction":prediction,"score":scores,"probability":probability,"group_id":bundle["run_uid"],"fault_id":bundle["faultNumber"],"stage":context["stages"]["test"],"start":bundle["start_sample"],"end":bundle["end_sample"]}; metrics=_metrics(labels,scores,threshold); early=raw["stage"]=="early"; metrics["early_recall"]=float(prediction[early].mean()) if early.any() else None; groupwise=[]; delays=[]
    for group in sorted(set(map(str,raw["group_id"]))):
        sel=raw["group_id"]==group; row=_metrics(labels[sel],scores[sel],threshold); fault=int(raw["fault_id"][sel][0]); delay=None
        if fault:
            post=sel&(raw["stage"]!="prefault"); detected=np.flatnonzero(post&(prediction==1))
            if len(detected): onset=int(context["base"]["protocol"]["fault_onset"][str(group).split(":",1)[0]]); delay=float(raw["end"][detected[0]]-onset); delays.append(delay)
        row.update({"group_id":group,"fault_id":fault,"windows":int(sel.sum()),"early_recall":float(prediction[sel&early].mean()) if np.any(sel&early) else None,"detection_delay":delay}); groupwise.append(row)
    metrics["detection_delay"]=float(np.mean(delays)) if delays else None; return metrics,raw,groupwise


def evaluate_cell(config: dict[str, Any], context: dict[str, Any], method: str, model_seed: int,
                  model: Any, probe: Any, validation: dict[str, Any], device: str) -> dict[str, Any]:
    output=_cell_dir(config,context["dataset"],context["outer_seed"],model_seed,method); result_path=output/"result.json"; prediction_path=output/"predictions.npz"
    if result_path.exists() and prediction_path.exists(): return read_json(result_path)
    if result_path.exists() or prediction_path.exists(): raise RuntimeError(f"incomplete outer result {output}")
    batch=int(config["benchmark"]["batch_size"][context["dataset"]])
    if context["dataset"]=="3W": metrics,raw,groupwise=_evaluate_three_w_representation(context,model,probe,method,batch,device)
    else:
        if validation["threshold"] is None: raise RuntimeError("TEP requires validation-only threshold")
        metrics,raw,groupwise=_evaluate_tep_representation(context,model,probe,method,float(validation["threshold"]),batch,device)
    np.savez_compressed(prediction_path,**raw); checkpoint=output/"checkpoint.pt"
    record={"run_id":_cell_id(context["dataset"],context["outer_seed"],model_seed,method),"evidence_class":EVIDENCE_CLASS,"dataset":context["dataset"],"outer_seed":context["outer_seed"],"model_seed":model_seed,"method":method,"track":TRACKS[method],"metrics":metrics,"groupwise":groupwise,"threshold":validation["threshold"],"prediction_path":str(prediction_path),"prediction_sha256":sha256_file(prediction_path),"checkpoint_path":str(checkpoint),"checkpoint_sha256":sha256_file(checkpoint),"outer_test_evaluated_once":True,"completed_at":now()}
    write_json(result_path,record); return record


def run_sanity(config: dict[str, Any], audit: dict[str, Any], device: str, method: str | None = None) -> list[dict[str, Any]]:
    sanity_path = Path(config["output"]["sanity_csv"])
    rows = list(csv.DictReader(sanity_path.open(encoding="utf-8-sig"))) if sanity_path.exists() else []
    methods=[method] if method else list(config["active_methods"])
    for dataset in ("3W","TEP"):
        outer=int(config["sanity"]["outer_splits"][dataset]); seed=int(config["sanity"]["model_seeds"][dataset]); context=prepare_context(audit["paper"],dataset,outer,device,False)
        train_groups=context["split"]["train"]; val_groups=context["split"]["validation"]; test_groups=context["split"]["test"]
        if train_groups&val_groups or train_groups&test_groups or val_groups&test_groups: raise RuntimeError("group leakage")
        for current in methods:
            started=time.perf_counter()
            try:
                record,_,_=train_cell(config,context,current,seed,device,True); status="PASS" if record["finite_loss"] and record["checkpoint_roundtrip"] else "FAIL"
                new_row={"dataset":dataset,"outer_seed":outer,"model_seed":seed,"method":current,"track":TRACKS[current],"status":status,"input_shape":list(context["train"].shape),"representation_shape":record["representation_shape"],"finite_loss":record["finite_loss"],"gpu":torch.cuda.get_device_name(0),"checkpoint_roundtrip":record["checkpoint_roundtrip"],"group_leakage":False,"linear_probe":True,"outer_test_materialized":record["outer_test_materialized"],"outer_test_metric_read":record["outer_test_read"],"validation_macro_f1":record["validation"]["macro_f1"],"runtime_seconds":record["training_seconds"],"peak_gpu_mib":record["peak_gpu_mib"],"failure":""}
                rows=[row for row in rows if not (row["dataset"]==dataset and row["method"]==current)]; rows.append(new_row)
            except Exception as error:
                new_row={"dataset":dataset,"outer_seed":outer,"model_seed":seed,"method":current,"track":TRACKS[current],"status":"FAIL","input_shape":list(context["train"].shape),"representation_shape":"","finite_loss":False,"gpu":torch.cuda.get_device_name(0),"checkpoint_roundtrip":False,"group_leakage":False,"linear_probe":False,"outer_test_materialized":False,"outer_test_metric_read":False,"validation_macro_f1":"","runtime_seconds":time.perf_counter()-started,"peak_gpu_mib":torch.cuda.max_memory_allocated()/1024**2,"failure":f"{type(error).__name__}: {error}"}
                rows=[row for row in rows if not (row["dataset"]==dataset and row["method"]==current)]; rows.append(new_row)
                write_csv(config["output"]["sanity_csv"],rows); raise
            write_csv(config["output"]["sanity_csv"],rows)
    return rows


def _manifest(config: dict[str, Any]) -> dict[str, Any]:
    path=Path(config["output"]["manifest"])
    if path.exists(): return read_json(path)
    cells=[]
    for dataset in ("3W","TEP"):
        for outer in config["benchmark"]["outer_splits"][dataset]:
            for seed in config["benchmark"]["model_seeds"][dataset]:
                for method in config["active_methods"]: cells.append({"run_id":_cell_id(dataset,int(outer),int(seed),method),"dataset":dataset,"outer_seed":int(outer),"model_seed":int(seed),"method":method,"status":"pending"})
    payload={"status":"POSTHOC_BASELINE_BENCHMARK_PREPARED","evidence_class":EVIDENCE_CLASS,"selection_hash":config["selection_hash"],"created_at":now(),"expected_cells":72,"cells":cells,"failures":[]}; write_json(path,payload); return payload


def _store_manifest(config: dict[str, Any], manifest: dict[str, Any]) -> None:
    manifest["completed_cells"]=sum(row["status"]=="complete" for row in manifest["cells"]); write_json(Path(config["output"]["manifest"]),manifest)


def _completed_result(config: dict[str, Any], dataset: str, outer: int, seed: int, method: str) -> dict[str, Any] | None:
    path=_cell_dir(config,dataset,outer,seed,method)/"result.json"
    if not path.exists(): return None
    record=read_json(path)
    if not record.get("outer_test_evaluated_once") or sha256_file(record["prediction_path"])!=record["prediction_sha256"] or sha256_file(record["checkpoint_path"])!=record["checkpoint_sha256"]:
        raise RuntimeError(f"completed result provenance check failed: {path}")
    return record


def run_benchmark(config: dict[str, Any], audit: dict[str, Any], device: str,
                  method_filter: str | None = None, dataset_filter: str | None = None) -> list[dict[str, Any]]:
    sanity=list(csv.DictReader(Path(config["output"]["sanity_csv"]).open(encoding="utf-8-sig")))
    active=set(config["active_methods"]); relevant=[row for row in sanity if row["method"] in active]
    if len(relevant)!=8 or any(row["status"]!="PASS" or row["outer_test_metric_read"].lower()!="false" for row in relevant): raise RuntimeError("all eight active sanity cells must pass before formal evaluation")
    manifest=_manifest(config); results=[]
    for dataset in ("3W","TEP"):
        if dataset_filter and dataset!=dataset_filter: continue
        for outer in map(int,config["benchmark"]["outer_splits"][dataset]):
            context=prepare_context(audit["paper"],dataset,outer,device,True)
            for seed in map(int,config["benchmark"]["model_seeds"][dataset]):
                for method in config["active_methods"]:
                    if method_filter and method!=method_filter: continue
                    cell=next(row for row in manifest["cells"] if row["run_id"]==_cell_id(dataset,outer,seed,method))
                    try:
                        result=_completed_result(config,dataset,outer,seed,method)
                        if result is None:
                            validation,model,probe=train_cell(config,context,method,seed,device,False); result=evaluate_cell(config,context,method,seed,model,probe,validation,device)
                        cell.update({"status":"complete","result_path":str(_cell_dir(config,dataset,outer,seed,method)/"result.json"),"completed_at":result["completed_at"]}); results.append(result); _store_manifest(config,manifest)
                    except Exception as error:
                        cell["status"]="failed"; manifest["failures"].append({"run_id":cell["run_id"],"type":type(error).__name__,"message":str(error),"at":now()}); manifest["status"]="POSTHOC_BASELINE_BENCHMARK_INTERRUPTED"; _store_manifest(config,manifest); raise
    return results


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="configs/posthoc_recent_baselines.yaml"); parser.add_argument("--sanity",action="store_true"); parser.add_argument("--benchmark",action="store_true"); parser.add_argument("--method"); parser.add_argument("--dataset",choices=("3W","TEP")); args=parser.parse_args()
    config=yaml.safe_load(Path(args.config).read_text(encoding="utf-8")); audit=validate_protocol(config); device=select_device(str(config["device"])); print(json.dumps({"status":"POSTHOC_BASELINE_PREFLIGHT_GO","python":os.sys.executable,"device":device,"selection_hash":config["selection_hash"]}),flush=True)
    if args.sanity: rows=run_sanity(config,audit,device,args.method); print(json.dumps({"sanity_rows":len(rows),"passed":sum(row["status"]=="PASS" for row in rows)}),flush=True)
    elif args.benchmark: rows=run_benchmark(config,audit,device,args.method,args.dataset); print(json.dumps({"completed_this_invocation":len(rows)}),flush=True)
    else: raise SystemExit("choose --sanity or --benchmark")


if __name__=="__main__": main()
