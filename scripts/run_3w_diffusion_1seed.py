from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

import scripts.run_3w_clean_baseline as base3w
from datasets.three_w import discover_instances
from diffusion import DiffusionSchedule, FrequencyForwardDiffusion, fit_spectral_statistics
from frequency import build_criticality, fit_frequency_scaler, log_amplitude_phase
from scripts.run_3w_clean_collapse_diagnosis import train_probe
from scripts.run_3w_final_primary_grouped import FINAL_PRIMARY_CLASSES, build_model
from scripts.run_diffusion_quality_retest import _fit_supcon, epoch_orders
from trainers.balanced import sqrt_inverse_frequency_weights
from utils import seed_everything, select_device


METHODS = ("CLEAN_HARD_SUPCON", "UNIFORM_DIFFUSION", "FREQUENCY_SELECTIVE_R1")


def state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def json_ready_criticality(record: dict) -> dict:
    return {
        "fit_split": record["fit_split"],
        "run_counts": record["run_counts"],
        "discriminative": record["discriminative"].tolist(),
        "early": record["early"].tolist(),
        "stability": record["stability"].tolist(),
        "composite": record["composite"].tolist(),
        "soft_mask": record["soft_mask"].tolist(),
        "composite_mask": record["masks"]["composite"].astype(int).tolist(),
        "bootstrap_overlap": record["bootstrap_overlap"].tolist(),
    }


def run(config: dict, data_root: Path) -> dict:
    stability = json.loads(Path(config["stability_result"]).read_text(encoding="utf-8"))
    if stability["status"] != "3W_FINAL_PRIMARY_STABILITY_GO" or not stability["diffusion_allowed"]:
        raise RuntimeError("Stage A HARD GATE is not GO; diffusion is forbidden")
    grouped = Path(config["grouped_output"])
    manifest = json.loads((grouped / "grouped_split_manifest.json").read_text(encoding="utf-8"))
    split_index = int(config["canonical_split_index"])
    manifest_split = manifest["splits"][split_index]
    split = {name: set(wells) for name, wells in manifest_split["wells"].items()}
    grouped_result = json.loads((grouped / f"split_{split_index:02d}" / "result.json").read_text(encoding="utf-8"))
    if grouped_result["split"] != manifest_split["wells"]:
        raise RuntimeError("canonical grouped result does not match frozen manifest")

    base_config = yaml.safe_load(Path(config["base_config"]).read_text(encoding="utf-8"))
    base = yaml.safe_load(Path(base_config["base_config"]).read_text(encoding="utf-8"))
    seed = int(config["seed"])
    protocol_seed = int(config.get("protocol_seed", 42))
    device = select_device(str(config["device"]))
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    base3w.PRIMARY_CLASSES = FINAL_PRIMARY_CLASSES
    base3w.CLASS_TO_TARGET = {value: index for index, value in enumerate(FINAL_PRIMARY_CLASSES)}
    instances = [item for item in discover_instances(data_root) if item.source == "WELL" and item.event_class in FINAL_PRIMARY_CLASSES]
    by_instance = {item.instance_id: item for item in instances}
    by_split = {name: [item for item in instances if item.well_id in wells] for name, wells in split.items()}
    preprocessor = json.loads((grouped / f"split_{split_index:02d}" / "preprocessor.json").read_text(encoding="utf-8"))

    refs_by_split = {}
    refs_by_instance = {}
    for name, items in by_split.items():
        refs = []
        for item in items:
            current = base3w.instance_refs(
                item,
                int(base["protocol"]["window_length"]),
                int(base["protocol"]["stride"]),
                int(base["protocol"]["transient_offset"]),
            )
            refs.extend(current)
            refs_by_instance[item.instance_id] = current
        refs_by_split[name] = refs
    train_refs = base3w.stratified_refs(
        refs_by_split["train"], int(base_config["train_windows_per_class"]), protocol_seed
    )
    validation_refs = base3w.stratified_refs(
        refs_by_split["validation"], int(base_config["validation_windows_per_class"]), protocol_seed + 1
    )
    length = int(base["protocol"]["window_length"])
    train_x, train_y = base3w.materialize(train_refs, by_instance, preprocessor, length, False)
    validation_x, validation_y = base3w.materialize(validation_refs, by_instance, preprocessor, length, False)

    def run_uid(ref) -> str:
        item = by_instance[ref.instance_id]
        original = FINAL_PRIMARY_CLASSES[ref.target] if ref.target else 0
        return f"training:fault_{original}:{item.well_id}"

    criticality_source = config.get("criticality_source")
    if criticality_source:
        source = json.loads(Path(criticality_source).read_text(encoding="utf-8"))
        criticality_payload = source["criticality"]
        critical_soft_mask = np.asarray(criticality_payload["soft_mask"], dtype=np.float32)
    else:
        train_bundle = {"run_uid": np.asarray([run_uid(ref) for ref in train_refs]), "labels": train_y}
        train_stages = np.asarray([ref.stage for ref in train_refs])
        train_log = log_amplitude_phase(train_x)[0]
        scaler = fit_frequency_scaler(train_log, "train")
        criticality = build_criticality(
            scaler.transform(train_log), train_bundle, train_stages, config["criticality"], train_log
        )
        criticality_payload = json_ready_criticality(criticality)
        critical_soft_mask = criticality["soft_mask"]
    statistics = fit_spectral_statistics(train_x, float(config["spectral_diffusion"]["clip_quantile"]), "train")
    schedule = DiffusionSchedule.cosine(int(config["spectral_diffusion"]["diffusion_steps"]), device)
    augmenter = FrequencyForwardDiffusion(
        statistics,
        schedule.alpha_bars,
        critical_soft_mask,
        int(config["spectral_diffusion"]["t_uniform"]),
        int(config["spectral_diffusion"]["t_key"]),
        bool(config["spectral_diffusion"]["preserve_phase"]),
        bool(config["spectral_diffusion"]["preserve_dc"]),
        device,
    )
    sampling_seed = seed + int(config["spectral_diffusion"]["sampling_seed_offset"])
    uniform_train, uniform_train_diagnostics = augmenter.augment(
        train_x, "uniform", sampling_seed, batch_size=int(config["training"]["batch_size"])
    )
    r1_train, r1_train_diagnostics = augmenter.augment(
        train_x,
        "selective",
        sampling_seed,
        int(config["spectral_diffusion"]["t_nonkey"]),
        int(config["training"]["batch_size"]),
    )
    uniform_validation, uniform_validation_diagnostics = augmenter.augment(
        validation_x, "uniform", sampling_seed + 100, batch_size=int(config["training"]["batch_size"])
    )
    r1_validation, r1_validation_diagnostics = augmenter.augment(
        validation_x,
        "selective",
        sampling_seed + 100,
        int(config["spectral_diffusion"]["t_nonkey"]),
        int(config["training"]["batch_size"]),
    )
    if abs(uniform_train_diagnostics["expected_total_noise_budget"] - r1_train_diagnostics["expected_total_noise_budget"]) > 1e-6:
        raise RuntimeError("Uniform/R1 total perturbation budgets are not comparable")

    restored = {
        METHODS[0]: (train_x, validation_x),
        METHODS[1]: (uniform_train, uniform_validation),
        METHODS[2]: (r1_train, r1_validation),
    }
    training = dict(config["training"])
    pretrain_orders = epoch_orders(len(train_y), int(training["epochs"]), seed + 10_000)
    weights = sqrt_inverse_frequency_weights(train_y)
    seed_everything(seed)
    template = build_model(base["training"]["model"], train_x.shape[1], device)
    initial_state = copy.deepcopy(template.state_dict())
    initialization_sha256 = state_hash(initial_state)
    results = {}
    for method in METHODS:
        method_result_path = output / f"{method}_result.json"
        if method_result_path.exists():
            results[method] = json.loads(method_result_path.read_text(encoding="utf-8"))
            print("skip", method, flush=True)
            continue
        print("start", method, flush=True)
        seed_everything(seed)
        model = build_model(base["training"]["model"], train_x.shape[1], device)
        model.load_state_dict(initial_state)
        checkpoint_path = output / f"{method}_model.pt"
        if checkpoint_path.exists():
            model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
            pretrain_history = None
            probe_history = None
            recovered_from_checkpoint = True
        else:
            train_view, validation_view = restored[method]
            pretrain_history = _fit_supcon(
                model,
                {"clean": train_x, "restored": train_view, "labels": train_y},
                {"clean": validation_x, "restored": validation_view, "labels": validation_y},
                np.ones(len(train_y), np.float32),
                np.ones(len(validation_y), np.float32),
                pretrain_orders,
                training,
                device,
            )
            probe_history = train_probe(
                model,
                train_x,
                train_y,
                validation_x,
                validation_y,
                weights,
                int(training["probe_epochs"]),
                float(training["learning_rate"]),
                int(training["batch_size"]),
                seed,
                device,
            )
            recovered_from_checkpoint = False
        evaluation_config = copy.deepcopy(base)
        evaluation_config["protocol"]["append_missing_mask"] = False
        evaluation_config["training"]["batch_size"] = int(training["batch_size"])
        metrics, per_instance = base3w.evaluate_stream(
            model, by_split["test"], refs_by_instance, preprocessor, evaluation_config, device
        )
        torch.save(model.state_dict(), checkpoint_path)
        results[method] = {
            "metrics": metrics,
            "per_instance": per_instance,
            "pretrain_history": pretrain_history,
            "probe_history": probe_history,
            "initialization_sha256": initialization_sha256,
            "recovered_from_checkpoint": recovered_from_checkpoint,
        }
        method_result_path.write_text(json.dumps(results[method], ensure_ascii=False, indent=2), encoding="utf-8")
        print("done", method, metrics["macro_f1"], metrics["auprc_multiclass_macro"], metrics["far"], flush=True)

    payload = {
        "stage_a_status": stability["status"],
        "seed": seed,
        "protocol_seed": protocol_seed,
        "canonical_split_index": split_index,
        "primary_classes": list(FINAL_PRIMARY_CLASSES),
        "methods": results,
        "criticality": criticality_payload,
        "augmentation_diagnostics": {
            METHODS[1]: {"train": uniform_train_diagnostics, "validation": uniform_validation_diagnostics},
            METHODS[2]: {"train": r1_train_diagnostics, "validation": r1_validation_diagnostics},
        },
        "fairness": {
            "same_grouped_split": True,
            "same_train_only_preprocessor": True,
            "same_clean_windows_and_labels": True,
            "same_initialization": True,
            "same_pretrain_batch_order": True,
            "same_balanced_probe": True,
            "uniform_r1_total_budget_matched": True,
            "initialization_sha256": initialization_sha256,
            "probe_weights_train_only": weights.tolist(),
            "window_refs_sha256": hashlib.sha256(
                "\n".join(f"{ref.instance_id}:{ref.start}:{ref.target}" for ref in train_refs + validation_refs).encode()
            ).hexdigest(),
            "critical_soft_mask_sha256": hashlib.sha256(
                np.ascontiguousarray(critical_soft_mask).tobytes()
            ).hexdigest(),
        },
    }
    (output / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/3w_diffusion_1seed.yaml")
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = run(config, args.data_root)
    print(json.dumps({name: value["metrics"]["macro_f1"] for name, value in result["methods"].items()}))


if __name__ == "__main__":
    main()
