from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from diffusion import DiffusionSchedule, FrequencyForwardDiffusion, fit_spectral_statistics
from frequency import fault_stages
from scripts.run_3w_diffusion_1seed import METHODS as THREE_W_METHODS, run as run_three_w
from scripts.run_diffusion_quality_retest import _state_hash, epoch_orders, load_fixed_views
from scripts.run_frequency_selective_r1_3seed import _fit_method, file_sha256
from scripts.run_stage_frequency_diffusion_mvp import _runtime, augmentation_mechanism_metrics
from trainers import build_model
from utils import seed_everything, select_device, write_json


CELLS = (("CE_REP", "NO_AUG"), ("CE_REP", "FINAL_QDIFFCL"),
         ("HARD_SUPCON", "NO_AUG"), ("HARD_SUPCON", "FINAL_QDIFFCL"))


def read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def store(path: Path, records: dict[str, Any], key: str, record: dict[str, Any]) -> None:
    records[key] = record
    write_json(path, {"results": records, "outer_test_run": False,
                      "evidence_scope": "development_test"})


def _metric_record(dataset: str, payload: dict[str, Any]) -> dict[str, float | None]:
    if dataset == "3W":
        metrics = payload["metrics"] if "metrics" in payload and "macro_f1" not in payload else payload
        return {"macro_f1": float(metrics["macro_f1"]),
                "auprc": float(metrics["auprc_multiclass_macro"]),
                "far": float(metrics["far"]), "early_recall": float(metrics["early_recall"]),
                "detection_delay": float(metrics["mean_detection_delay_seconds"])}
    test = payload["test"]
    return {"macro_f1": float(test["metrics"]["macro_f1"]),
            "auprc": float(test["metrics"]["auprc"]), "far": float(test["metrics"]["far"]),
            "early_recall": float(test["early_fault"]["recall"]),
            "detection_delay": float(test["detection_delay"]["mean_delay_samples"])}


def reuse_hard(config: dict[str, Any], dataset: str, records: dict[str, Any], path: Path) -> None:
    stage = config["three_w" if dataset == "3W" else "tep"]
    no_aug = read(stage["hard_no_aug_manifest"])["results"]
    final = read(stage["hard_final_manifest"])["results"]
    for seed in map(int, stage["seeds"]):
        key = f"{dataset}|HARD_SUPCON|NO_AUG|{seed}"
        if key not in records:
            item = no_aug[f"{dataset}|NO_AUG|{seed}"]
            store(path, records, key, {"dataset": dataset, "objective": "HARD_SUPCON",
                  "augmentation": "NO_AUG", "seed": seed,
                  "metrics": _metric_record(dataset, item["record"]),
                  "action": "REUSE_EXISTING", "source": stage["hard_no_aug_manifest"]})
        key = f"{dataset}|HARD_SUPCON|FINAL_QDIFFCL|{seed}"
        if key not in records:
            item = final[f"FINAL_QDIFFCL|{seed}"]
            store(path, records, key, {"dataset": dataset, "objective": "HARD_SUPCON",
                  "augmentation": "FINAL_QDIFFCL", "seed": seed,
                  "metrics": _metric_record(dataset, item["metrics"]),
                  "action": "REUSE_EXISTING", "source": stage["hard_final_manifest"]})


def run_three_w_ce(config: dict[str, Any], data_root: Path, records: dict[str, Any], path: Path) -> None:
    stage = config["three_w"]
    for seed in map(int, stage["seeds"]):
        wanted = [("NO_AUG", THREE_W_METHODS[0]), ("FINAL_QDIFFCL", THREE_W_METHODS[2])]
        missing = [(name, method) for name, method in wanted
                   if f"3W|CE_REP|{name}|{seed}" not in records]
        if not missing: continue
        base = yaml.safe_load(Path(stage["base_config"]).read_text(encoding="utf-8"))
        current = copy.deepcopy(base); current.update({"seed": seed, "protocol_seed": int(stage["protocol_seed"]),
            "criticality_source": stage["final_mask"], "methods": [method for _, method in missing],
            "evaluation_split": "test", "representation_objective": "ce_rep",
            "output_dir": str(Path(stage["output_dir"]) / f"seed_{seed}")})
        current["training"]["supcon_batching"] = "original"
        result = run_three_w(current, data_root)
        for name, method in missing:
            store(path, records, f"3W|CE_REP|{name}|{seed}", {"dataset": "3W",
                  "objective": "CE_REP", "augmentation": name, "seed": seed,
                  "metrics": _metric_record("3W", result["methods"][method]),
                  "action": "NEW_TRAINING_REQUIRED", "source": str(Path(current["output_dir"]) / "result.json"),
                  "fairness": result["fairness"]})


def run_tep_ce(config: dict[str, Any], device: str, records: dict[str, Any], path: Path) -> None:
    stage = config["tep"]; base = yaml.safe_load(Path(stage["base_config"]).read_text(encoding="utf-8"))
    views, _ = load_fixed_views(base); clean = {split: views[split]["clean"].astype(np.float32) for split in views}
    stages = {split: fault_stages(views[split], base) for split in views}; mask = read(stage["final_mask"])["criticality"]
    spectral = config["spectral_diffusion"]
    statistics = fit_spectral_statistics(clean["train"], float(spectral["clip_quantile"]), "train")
    schedule = DiffusionSchedule.cosine(int(spectral["diffusion_steps"]), device)
    augmenter = FrequencyForwardDiffusion(statistics, schedule.alpha_bars,
        np.asarray(mask["soft_mask"], np.float32), int(spectral["t_uniform"]),
        int(spectral["t_critical"]), bool(spectral["preserve_phase"]), bool(spectral["preserve_dc"]), device)
    for seed in map(int, stage["seeds"]):
        runtime = _runtime(base, seed); runtime["diagnosis"] = {"threshold_band_width": .05, "high_correlation_quantile": .90}
        pretrain = epoch_orders(len(clean["train"]), int(runtime["epochs"]), seed + 10000)
        probe = epoch_orders(len(clean["train"]), int(runtime["probe_epochs"]), seed + 20000)
        seed_everything(seed); template = build_model(runtime["model"], clean["train"].shape[1], 2)
        initial = copy.deepcopy(template.state_dict())
        fairness = {"manifest_sha256": file_sha256(base["fixed_views"]["manifest"]),
                    "initialization_sha256": _state_hash(initial),
                    "pretrain_order_sha256": hashlib.sha256("\n".join(','.join(map(str, row)) for row in pretrain).encode()).hexdigest(),
                    "probe_order_sha256": hashlib.sha256("\n".join(','.join(map(str, row)) for row in probe).encode()).hexdigest()}
        final_augmented = {"test": clean["test"]}; audits: dict[str, Any] = {}
        for split, offset in (("train", 0), ("validation", 100)):
            changed, diagnostic = augmenter.augment(clean[split], "selective",
                seed + int(spectral["sampling_seed_offset"]) + offset,
                int(spectral["t_noncritical"]), int(runtime["batch_size"]))
            final_augmented[split] = changed
            audits[split] = augmentation_mechanism_metrics(clean[split], changed, views[split]["labels"],
                stages[split], np.asarray(mask["hard_mask"], bool), diagnostic)
        for augmentation, augmented in (("NO_AUG", clean), ("FINAL_QDIFFCL", final_augmented)):
            key = f"TEP|CE_REP|{augmentation}|{seed}"
            if key in records: continue
            output = Path(stage["output_dir"]) / augmentation / f"seed_{seed}"
            metadata = {**fairness, "objective": "ce_rep", "augmentation": augmentation,
                        "evaluation_splits": ["test"], "outer_test_run": False}
            record = _fit_method(augmentation, augmented, audits if augmentation == "FINAL_QDIFFCL" else {},
                views, clean, stages, initial, pretrain, probe, runtime, device, output / "model.pt", metadata,
                evaluation_splits=("test",), representation_objective="ce_rep")
            store(path, records, key, {"dataset": "TEP", "objective": "CE_REP",
                  "augmentation": augmentation, "seed": seed, "metrics": _metric_record("TEP", record),
                  "action": "NEW_TRAINING_REQUIRED", "source": str(output / "metrics.json"), "fairness": fairness})


def _bootstrap_ci(values: np.ndarray, repeats: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed); means = np.asarray([rng.choice(values, len(values), replace=True).mean() for _ in range(repeats)])
    return tuple(map(float, np.quantile(means, [.025, .975])))


def summarize(config: dict[str, Any], records: dict[str, Any]) -> None:
    rows = []
    for item in records.values(): rows.append({k: item[k] for k in ("dataset", "objective", "augmentation", "seed", "action", "source")} | item["metrics"])
    output = Path(config["output"]["results_csv"]); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(sorted(rows, key=lambda x:(x["dataset"],x["objective"],x["augmentation"],x["seed"])))
    lines = ["# Contrastive Learning Necessity Ablation", "", "所有结果均为既有 development test 协议或同协议新增训练；不是 Paper-final outer evidence。", ""]
    repeats = int(config["statistics"]["bootstrap_repeats"]); bootstrap_seed = int(config["statistics"]["bootstrap_seed"])
    for dataset in ("3W", "TEP"):
        lines += [f"## {dataset}", "", "| Objective | Augmentation | Macro-F1 | AUPRC | FAR | Early Recall | Delay |", "|---|---|---:|---:|---:|---:|---:|"]
        grid = {}
        for objective, augmentation in CELLS:
            selected = [r for r in rows if r["dataset"] == dataset and r["objective"] == objective and r["augmentation"] == augmentation]
            grid[objective, augmentation] = {int(r["seed"]): r for r in selected}
            means = {m: float(np.mean([r[m] for r in selected])) for m in ("macro_f1", "auprc", "far", "early_recall", "detection_delay")}
            std = float(np.std([r["macro_f1"] for r in selected], ddof=1))
            lines.append(f"| {objective} | {augmentation} | {means['macro_f1']:.4f} ± {std:.4f} | {means['auprc']:.4f} | {means['far']:.4f} | {means['early_recall']:.4f} | {means['detection_delay']:.2f} |")
        ce = np.asarray([grid["CE_REP","FINAL_QDIFFCL"][s]["macro_f1"]-grid["CE_REP","NO_AUG"][s]["macro_f1"] for s in grid["CE_REP","NO_AUG"]])
        sup = np.asarray([grid["HARD_SUPCON","FINAL_QDIFFCL"][s]["macro_f1"]-grid["HARD_SUPCON","NO_AUG"][s]["macro_f1"] for s in grid["HARD_SUPCON","NO_AUG"]])
        interaction = sup-ce; low, high = _bootstrap_ci(interaction, repeats, bootstrap_seed + (0 if dataset == "3W" else 1))
        dz = float(interaction.mean()/interaction.std(ddof=1)) if interaction.std(ddof=1) > 0 else None
        lines += ["", f"Macro-F1 paired augmentation delta：CE `{ce.mean():+.4f}`，Hard SupCon `{sup.mean():+.4f}`。",
                  f"Interaction `(FINAL-NO_AUG)_SupCon-(FINAL-NO_AUG)_CE` = `{interaction.mean():+.4f}`，95% bootstrap CI `[{low:+.4f}, {high:+.4f}]`，positive/non-worse `{int((interaction>0).sum())}/{int((interaction>=0).sum())}` / `{len(interaction)}`，Cohen dz `{('N/A' if dz is None else f'{dz:.3f}')}`。", ""]
    Path(config["output"]["report"]).write_text("\n".join(lines)+"\n", encoding="utf-8")


def run(config: dict[str, Any], data_root: Path, dataset: str) -> dict[str, Any]:
    if config["audit"] != {"final_method_frozen": True, "outer_test_run": False, "existing_development_test_only": True}:
        raise RuntimeError("contrastive audit boundary changed")
    path = Path(config["output"]["manifest"]); records = read(path).get("results", {}) if path.exists() else {}
    device = select_device(str(config["device"])); os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    datasets = ("3W", "TEP") if dataset == "both" else (dataset.upper(),)
    for current in datasets:
        reuse_hard(config, current, records, path)
        if current == "3W": run_three_w_ce(config, data_root, records, path)
        else: run_tep_ce(config, device, records, path)
    expected = sum(4 * len(config["three_w" if d == "3W" else "tep"]["seeds"]) for d in datasets)
    if len([r for r in records.values() if r["dataset"] in datasets]) == expected and set(datasets) == {"3W", "TEP"}:
        summarize(config, records)
    return {"status": "PAPER_CONTRASTIVE_ABLATION_COMPLETE", "records": len(records), "outer_test_run": False}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/paper_contrastive_ablation.yaml")
    parser.add_argument("--data-root", type=Path, required=True); parser.add_argument("--dataset", choices=("3w","tep","both"), default="both")
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    print(json.dumps(run(config, args.data_root, args.dataset), ensure_ascii=False))


if __name__ == "__main__": main()
