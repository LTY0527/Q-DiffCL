from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from datasets.protocol import Standardizer, split_runs, split_training_runs_stratified, window_runs
from datasets.synthetic import WARNING, make_synthetic_runs
from degradations import apply_degradation
from utils import configure_logging, environment_metadata, seed_everything, select_device, write_json


def load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def prepare_synthetic(config: dict, degrade: bool = False):
    runs = make_synthetic_runs(seed=int(config["random_seed"]))
    ratios = config["split"]
    manifest = split_runs([r.run_id for r in runs], (ratios["train"], ratios["validation"], ratios["test"]), int(config["random_seed"]))
    by_id = {r.run_id: r for r in runs}
    groups = {name: [by_id[x] for x in getattr(manifest, name)] for name in ("train", "validation", "test")}
    scaler = Standardizer().fit(np.concatenate([r.values for r in groups["train"]]))
    for group in groups.values():
        for run in group:
            run.values[:] = scaler.transform(run.values)
    prepared = {}
    stats = {}
    for name, group in groups.items():
        x, y, ids, group_stats = window_runs(group, int(config["window_length"]), int(config["stride"]), config["transition_policy"], config.get("fault_ratio_threshold"), config.get("task", "binary_fault_detection"))
        if degrade:
            degraded = [apply_degradation(sample, config["degradation"], config["degradation_severity"], int(config["random_seed"]), sid, config["degradation_space"]) for sample, sid in zip(x, ids)]
            x = np.stack([item.data for item in degraded])
        prepared[name] = (x, y)
        stats[name] = group_stats
    return prepared, manifest, stats


def prepare_real(config: dict, degrade: bool = False):
    from datasets.tep import REQUIRED_FILES, frame_to_runs, inspect_rdata_files, read_rdata_frame, validate_tep_config
    validate_tep_config(config)
    root = Path(config["dataset"]["root"]); inspect_rdata_files(root)
    training_runs, testing_runs = [], []
    for filename in REQUIRED_FILES:
        source_split = "testing" if "Testing" in filename else "training"
        run_limit = config.get("subset", {}).get(
            source_split, config.get("smoke", {}).get("max_runs_per_fault")
        )
        runs = frame_to_runs(read_rdata_frame(root / filename), config, source_split, run_limit)
        (testing_runs if source_split == "testing" else training_runs).extend(runs)
    manifest = split_training_runs_stratified(
        training_runs, float(config["split"]["validation"]), int(config["split"]["seed"]),
        [run.run_uid for run in testing_runs],
    )
    by_uid = {run.run_uid: run for run in training_runs}
    train_runs = [by_uid[uid] for uid in manifest.train]
    validation_runs = [by_uid[uid] for uid in manifest.validation]
    scaler = Standardizer().fit_many(run.values for run in train_runs)
    prepared, stats = {}, {}
    for name, group in (("train", train_runs), ("validation", validation_runs), ("test", testing_runs)):
        normalized = [type(run)(run.run_uid, scaler.transform(run.values), run.samples, run.fault_id, run.first_faulty_sample) for run in group]
        x, y, ids, group_stats = window_runs(normalized, int(config["window_length"]), int(config["stride"]), config["transition_policy"], config.get("fault_ratio_threshold"), config["task"])
        if degrade:
            items = [apply_degradation(sample, config["degradation"], config["degradation_severity"], int(config["random_seed"]), sid, config["degradation_space"]) for sample, sid in zip(x, ids)]
            x = np.stack([item.data for item in items])
        prepared[name] = (x, y); stats[name] = group_stats
    return prepared, manifest, stats


def run_experiment(config: dict, mode: str = "ce", degrade: bool = False) -> Path:
    from trainers import ExperimentTrainer, build_model
    import torch
    is_debug = config.get("mode") == "debug"
    total_started = time.perf_counter()
    started = datetime.now(timezone.utc)
    seed_everything(int(config["random_seed"]))
    condition = f"{config['degradation']}-{config['degradation_severity']}" if degrade else "clean"
    output = Path(config["output_dir"]) / f"{mode}-{condition}-seed-{config['random_seed']}"
    logger = configure_logging(output)
    if is_debug: logger.info(WARNING)
    data, manifest, window_stats = prepare_synthetic(config, degrade) if is_debug else prepare_real(config, degrade)
    channels, classes = data["train"][0].shape[1], int(max(v[1].max() for v in data.values()) + 1)
    model = build_model(config["model"], channels, classes)
    trainer = ExperimentTrainer(model, select_device(config["device"]), float(config["learning_rate"]))
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    training_started = time.perf_counter()
    train_x, train_y = data["train"]; val_x, val_y = data["validation"]; test_x, test_y = data["test"]
    pretrain_history = None
    if mode in {"linear_probe", "fine_tune"}:
        pretrain_history = trainer.pretrain_supcon(train_x, train_y, val_x, val_y, int(config["epochs"]), int(config["batch_size"]), float(config["temperature"]))
        if mode == "linear_probe": trainer.freeze_encoder()
        else: trainer.unfreeze_all()
        result = trainer.fit(train_x, train_y, val_x, val_y, int(config["epochs"]), int(config["batch_size"]), "ce")
    else:
        result = trainer.fit(train_x, train_y, val_x, val_y, int(config["epochs"]), int(config["batch_size"]), mode, float(config["supcon_weight"]), float(config["temperature"]))
    test_metrics = trainer.evaluate(test_x, test_y, int(config["batch_size"]))
    training_seconds = time.perf_counter() - training_started
    torch.save(model.state_dict(), output / "model.pt")
    metadata = {"markers": config.get("markers", []), "training_mode": mode, "started_at": started.isoformat(), "ended_at": datetime.now(timezone.utc).isoformat(), **environment_metadata(), "best_epoch": result.best_epoch, "validation_metrics": result.validation_metrics, "test_metrics": test_metrics, "window_stats": window_stats, "split_manifest": manifest.__dict__, "pretrain_history": pretrain_history, "training_seconds": training_seconds, "total_seconds": time.perf_counter() - total_started, "peak_gpu_mib": torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0}
    write_json(output / "metadata.json", metadata); write_json(output / "config.json", config); write_json(output / "split_manifest.json", manifest.__dict__)
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "loss", "validation_macro_f1"]); writer.writeheader(); writer.writerows(result.history)
    logger.info("completed engineering smoke test: %s", output)
    return output


def parser(description: str) -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=description)
    value.add_argument("--config", default="configs/debug.yaml")
    return value
