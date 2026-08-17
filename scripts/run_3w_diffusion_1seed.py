from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

import scripts.run_3w_clean_baseline as base3w
from datasets.three_w import discover_instances
from diffusion import DiffusionSchedule, FrequencyForwardDiffusion, fit_spectral_statistics
from frequency import (build_criticality, build_hierarchical_criticality,
                       build_rival_aware_criticality, fit_frequency_scaler,
                       log_amplitude_phase)
from scripts.run_3w_clean_collapse_diagnosis import train_probe
from scripts.run_3w_final_primary_grouped import FINAL_PRIMARY_CLASSES, build_model
from scripts.run_diffusion_quality_retest import _fit_supcon, epoch_orders
from trainers.balanced import (CrossWellPositiveSafeBatchSampler,
                               PositiveSafeBatchSampler,
                               sqrt_inverse_frequency_weights)
from utils import seed_everything, select_device


METHODS = ("CLEAN_HARD_SUPCON", "UNIFORM_DIFFUSION", "FREQUENCY_SELECTIVE_R1")
R2_METHOD = "FREQUENCY_SELECTIVE_R2"
R3_METHOD = "FREQUENCY_SELECTIVE_R3"
HFSC_METHOD = "HIERARCHICAL_FAULT_SEMANTIC_CRITICALITY"
RRDC_METHOD = "RIVAL_AWARE_RELIABLE_DIAGNOSTIC_CRITICALITY"


def state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _cross_well_order_audit(orders: list[np.ndarray], labels: np.ndarray, well_ids: np.ndarray,
                            batch_size: int, include_batches: bool) -> dict:
    well_ids = np.asarray(well_ids, dtype=object)
    if len(well_ids) != len(labels): raise ValueError("WELL ids must align with SupCon labels")
    available = {int(c): sorted(set(map(str, well_ids[labels == c]))) for c in np.unique(labels)}
    available_counts = {c: {well: int(((labels == c) & (well_ids == well)).sum()) for well in wells}
                        for c, wells in available.items()}
    epoch_rows = []; total_clean = total_paired = cross_clean = cross_paired = 0
    duplicate_windows = sampled_windows = 0
    per_class = {c: {"clean_cross": 0, "clean_total": 0, "paired_cross": 0, "paired_total": 0} for c in available}
    minimum_wells = min((len(wells) for wells in available.values() if len(wells) > 1), default=1)
    for order in orders:
        well_counts = {c: Counter() for c in available}; batch_rows = []
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]; current_y = labels[indices]; current_wells = well_ids[indices]
            duplicates = len(indices) - len(np.unique(indices)); duplicate_windows += duplicates; sampled_windows += len(indices)
            class_counts = Counter(map(int, current_y)); class_well_counts = {}; clean_num = clean_den = paired_num = paired_den = 0
            for c, count in sorted(class_counts.items()):
                counts = Counter(map(str, current_wells[current_y == c])); class_well_counts[c] = len(counts)
                well_counts[c].update(counts); minimum_wells = min(minimum_wells, len(counts)) if len(available[c]) > 1 else minimum_wells
                clean_total = count * (count - 1); clean_cross = clean_total - sum(n * (n - 1) for n in counts.values())
                paired_total = 2 * count * (2 * count - 1); paired_cross = paired_total - sum(2 * n * (2 * n - 1) for n in counts.values())
                clean_num += clean_cross; clean_den += clean_total; paired_num += paired_cross; paired_den += paired_total
                item = per_class[c]; item["clean_cross"] += clean_cross; item["clean_total"] += clean_total
                item["paired_cross"] += paired_cross; item["paired_total"] += paired_total
            cross_clean += clean_num; total_clean += clean_den; cross_paired += paired_num; total_paired += paired_den
            if include_batches:
                batch_rows.append({"class_counts": dict(class_counts), "class_well_counts": class_well_counts,
                                   "duplicate_window_count": duplicates,
                                   "clean_cross_well_positive_ratio": clean_num / clean_den if clean_den else 0.0,
                                   "paired_view_cross_well_positive_ratio": paired_num / paired_den if paired_den else 0.0})
        epoch_rows.append({"per_class_well_sample_counts": {c: dict(sorted(counts.items())) for c, counts in well_counts.items()},
                           "per_class_well_oversampling_factors": {
                               c: {well: count / available_counts[c][well] for well, count in sorted(counts.items())}
                               for c, counts in well_counts.items()},
                           **({"batches": batch_rows} if include_batches else {})})
    return {"class_available_wells": available, "class_well_window_counts": available_counts,
            "class_well_counts": {c: len(wells) for c, wells in available.items()},
            "classes_without_cross_well_support": [c for c, wells in available.items() if len(wells) < 2],
            "minimum_wells_for_multiwell_class_in_any_batch": minimum_wells,
            "duplicate_window_count": duplicate_windows,
            "duplicate_window_rate": duplicate_windows / sampled_windows if sampled_windows else 0.0,
            "clean_cross_well_positive_ratio": cross_clean / total_clean if total_clean else 0.0,
            "paired_view_cross_well_positive_ratio": cross_paired / total_paired if total_paired else 0.0,
            "per_class_cross_well_positive_ratio": {
                c: {"clean": item["clean_cross"] / item["clean_total"] if item["clean_total"] else 0.0,
                    "paired_view": item["paired_cross"] / item["paired_total"] if item["paired_total"] else 0.0}
                for c, item in per_class.items()},
            "per_epoch": epoch_rows}


def supcon_orders(labels: np.ndarray, training: dict, seed: int, well_ids: np.ndarray | None = None,
                  include_batch_audit: bool = True) -> tuple[list[np.ndarray], dict]:
    epochs = int(training["epochs"]); mode = str(training.get("supcon_batching", "original"))
    if mode == "original":
        orders = epoch_orders(len(labels), epochs, seed + 10_000)
        audit = {"mode": mode, "sampler_fit_scope": "train labels only"}
    elif mode == "balanced_positive_safe":
        spec = training["balanced_sampler"]
        sampler = PositiveSafeBatchSampler(labels, int(spec["classes_per_batch"]), int(spec["samples_per_class"]),
                                            int(spec["batches_per_epoch"]), seed + 10_000, float(spec["max_oversampling"]))
        orders = []; epoch_counts = []; minimum_classes = len(np.unique(labels)); minimum_samples = int(spec["samples_per_class"])
        valid = total = 0
        for epoch in range(epochs):
            sampler.set_epoch(epoch); batches = list(sampler); orders.append(np.asarray([index for batch in batches for index in batch], dtype=np.int64))
            counts = np.bincount(labels[orders[-1]], minlength=len(np.unique(labels))); epoch_counts.append(counts.tolist())
            for batch in batches:
                current = np.bincount(labels[np.asarray(batch)], minlength=len(counts)); present = current[current > 0]
                minimum_classes = min(minimum_classes, len(present)); minimum_samples = min(minimum_samples, int(present.min()))
                valid += int(present[present >= 2].sum()); total += len(batch)
        audit = {"mode": mode, "sampler_fit_scope": "train labels only", "classes_per_batch": int(spec["classes_per_batch"]),
                 "samples_per_class": int(spec["samples_per_class"]), "batches_per_epoch": int(spec["batches_per_epoch"]),
                 "planned_sample_counts_per_epoch": sampler.planned_sample_counts,
                 "oversampling_factors": sampler.oversampling_factors, "actual_sample_counts_per_epoch": epoch_counts,
                 "minimum_classes_in_any_batch": minimum_classes, "minimum_clean_samples_per_present_class": minimum_samples,
                 "clean_positive_anchor_rate": valid / total, "paired_view_positive_anchor_rate": 1.0,
                 "all_classes_retain_positive_pairs": bool(valid == total)}
    elif mode == "cross_well_positive_safe":
        if well_ids is None: raise ValueError("cross-WELL SupCon batching requires training WELL ids")
        spec = training["cross_well_sampler"]
        sampler = CrossWellPositiveSafeBatchSampler(labels, well_ids, int(spec["classes_per_batch"]),
                                                     int(spec["samples_per_class"]), int(spec["batches_per_epoch"]),
                                                     seed + 10_000, float(spec["max_oversampling"]))
        orders = []; epoch_counts = []
        for epoch in range(epochs):
            sampler.set_epoch(epoch); batches = list(sampler)
            orders.append(np.asarray([index for batch in batches for index in batch], dtype=np.int64))
            epoch_counts.append(np.bincount(labels[orders[-1]], minlength=len(np.unique(labels))).tolist())
        audit = {"mode": mode, "sampler_fit_scope": "train labels and train WELL ids only",
                 "classes_per_batch": int(spec["classes_per_batch"]), "samples_per_class": int(spec["samples_per_class"]),
                 "batches_per_epoch": int(spec["batches_per_epoch"]), "planned_sample_counts_per_epoch": sampler.planned_sample_counts,
                 "oversampling_factors": sampler.oversampling_factors, "actual_sample_counts_per_epoch": epoch_counts,
                 "clean_positive_anchor_rate": 1.0, "paired_view_positive_anchor_rate": 1.0,
                 "all_classes_retain_positive_pairs": True}
    else:
        raise ValueError(f"unknown SupCon batching mode: {mode}")
    if well_ids is not None:
        audit["cross_well"] = _cross_well_order_audit(
            orders, labels, np.asarray(well_ids, dtype=object), int(training["batch_size"]), include_batch_audit
        )
    audit["batch_order_sha256"] = hashlib.sha256(
        "\n".join(",".join(map(str, order.tolist())) for order in orders).encode()
    ).hexdigest()
    return orders, audit


def json_ready_criticality(record: dict) -> dict:
    return {
        "fit_split": record["fit_split"],
        "run_counts": record["run_counts"],
        "discriminative": record["discriminative"].tolist(),
        "early": record["early"].tolist(),
        "stability": record["stability"].tolist(),
        "multiclass_fisher": record["multiclass_fisher"].tolist(),
        "multiclass_mode": record["multiclass_mode"],
        "multiclass_reliability": record["multiclass_reliability"].tolist(),
        "multiclass_class_contributions": {
            str(kind): values.tolist() for kind, values in record["multiclass_class_contributions"].items()
        },
        "composite": record["composite"].tolist(),
        "soft_mask": record["soft_mask"].tolist(),
        "composite_mask": record["masks"]["composite"].astype(int).tolist(),
        "multiclass_mask": record["masks"]["multiclass"].astype(int).tolist(),
        "bootstrap_overlap": record["bootstrap_overlap"].tolist(),
        "component_weights": record["component_weights"],
        "multiclass_type_run_counts": record["multiclass_type_run_counts"],
    }


def json_ready_hierarchical_criticality(record: dict) -> dict:
    def item_ready(item: dict) -> dict:
        return {"score": item["score"].tolist(), "hard_mask": item["hard_mask"].astype(int).tolist(),
                "soft_mask": item["soft_mask"].tolist()}
    return {"fit_split": record["fit_split"], "shared": json_ready_criticality(record["shared"]),
            "diagnostic": {str(kind): item_ready(item) for kind, item in record["diagnostic"].items()},
            "hierarchical": {str(kind): item_ready(item) for kind, item in record["hierarchical"].items()},
            "fault_run_counts": record["fault_run_counts"], "diagnostic_classes": record["diagnostic_classes"],
            "shared_weight": record["shared_weight"], "diagnostic_weight": record["diagnostic_weight"]}


def json_ready_rival_aware_criticality(record: dict) -> dict:
    def item_ready(item: dict) -> dict:
        payload = {key: item[key].tolist() for key in ("score", "hard_mask", "soft_mask")}
        payload["hard_mask"] = item["hard_mask"].astype(int).tolist()
        return payload
    diagnostic = {}
    for kind, item in record["diagnostic"].items():
        diagnostic[str(kind)] = {
            "score": item["score"].tolist(), "reliability": item["reliability"].tolist(),
            "reliable_score": item["reliable_score"].tolist(),
            "hard_mask": item["hard_mask"].astype(int).tolist(), "soft_mask": item["soft_mask"].tolist(),
            "pairwise": {str(rival): {name: values.tolist() for name, values in pair.items()}
                         for rival, pair in item["pairwise"].items()},
            "pairwise_summary": {str(rival): summary for rival, summary in item["pairwise_summary"].items()},
            "hardest_rival": item["hardest_rival"], "hardest_rival_score": item["hardest_rival_score"],
        }
    return {"fit_split": record["fit_split"], "shared": json_ready_criticality(record["shared"]),
            "diagnostic": diagnostic, "final": {str(k): item_ready(v) for k, v in record["final"].items()},
            "fault_run_counts": record["fault_run_counts"], "diagnostic_classes": record["diagnostic_classes"],
            "hard_rival_quantile": record["hard_rival_quantile"],
            "bootstrap_repeats": record["bootstrap_repeats"], "combination": record["combination"]}


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
    diffusion_seed = int(config.get("diffusion_seed", seed))
    validation_diffusion_seed = int(config.get("validation_diffusion_seed", diffusion_seed))
    encoder_seed = int(config.get("encoder_seed", seed))
    probe_seed = int(config.get("probe_seed", seed))
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

    hierarchical_enabled = bool(config.get("hierarchical_criticality", False))
    rival_aware_enabled = bool(config.get("rival_aware_criticality", False))
    if hierarchical_enabled and rival_aware_enabled:
        raise ValueError("HFSC and RRDC criticality modes are mutually exclusive")
    class_conditional_enabled = hierarchical_enabled or rival_aware_enabled
    criticality_source = config.get("criticality_source")
    hierarchical_soft_masks = None
    train_original_y = np.asarray([FINAL_PRIMARY_CLASSES[ref.target] if ref.target else 0 for ref in train_refs])
    validation_original_y = np.asarray([FINAL_PRIMARY_CLASSES[ref.target] if ref.target else 0 for ref in validation_refs])
    if class_conditional_enabled and criticality_source:
        raise ValueError("class-conditional criticality must be fitted train-only in the current protocol")
    if criticality_source:
        source = json.loads(Path(criticality_source).read_text(encoding="utf-8"))
        criticality_payload = source["criticality"]
        critical_soft_mask = np.asarray(criticality_payload["soft_mask"], dtype=np.float32)
    else:
        train_bundle = {"run_uid": np.asarray([run_uid(ref) for ref in train_refs]), "labels": train_y}
        train_stages = np.asarray([ref.stage for ref in train_refs])
        train_log = log_amplitude_phase(train_x)[0]
        scaler = fit_frequency_scaler(train_log, "train")
        if class_conditional_enabled:
            criticality = (build_rival_aware_criticality if rival_aware_enabled else build_hierarchical_criticality)(
                scaler.transform(train_log), train_bundle, train_stages, config["criticality"], train_log)
            criticality_payload = (json_ready_rival_aware_criticality(criticality) if rival_aware_enabled
                                   else json_ready_hierarchical_criticality(criticality))
            critical_soft_mask = criticality["shared"]["soft_mask"]
            hierarchical_soft_masks = criticality["soft_masks"]
        else:
            criticality = build_criticality(
                scaler.transform(train_log), train_bundle, train_stages, config["criticality"], train_log)
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
    sampling_seed = diffusion_seed + int(config["spectral_diffusion"]["sampling_seed_offset"])
    validation_sampling_seed = validation_diffusion_seed + int(config["spectral_diffusion"]["sampling_seed_offset"]) + 100
    uniform_train, uniform_train_diagnostics = augmenter.augment(
        train_x, "uniform", sampling_seed, batch_size=int(config["training"]["batch_size"])
    )
    if class_conditional_enabled:
        r1_train, r1_train_diagnostics = augmenter.augment_hierarchical(
            train_x, train_original_y, hierarchical_soft_masks, sampling_seed,
            int(config["spectral_diffusion"]["t_nonkey"]), int(config["training"]["batch_size"]))
    else:
        r1_train, r1_train_diagnostics = augmenter.augment(
            train_x, "selective", sampling_seed, int(config["spectral_diffusion"]["t_nonkey"]),
            int(config["training"]["batch_size"]))
    uniform_validation, uniform_validation_diagnostics = augmenter.augment(
        validation_x, "uniform", validation_sampling_seed, batch_size=int(config["training"]["batch_size"])
    )
    if class_conditional_enabled:
        r1_validation, r1_validation_diagnostics = augmenter.augment_hierarchical(
            validation_x, validation_original_y, hierarchical_soft_masks, validation_sampling_seed,
            int(config["spectral_diffusion"]["t_nonkey"]), int(config["training"]["batch_size"]))
    else:
        r1_validation, r1_validation_diagnostics = augmenter.augment(
            validation_x, "selective", validation_sampling_seed,
            int(config["spectral_diffusion"]["t_nonkey"]), int(config["training"]["batch_size"]))
    if abs(uniform_train_diagnostics["expected_total_noise_budget"] - r1_train_diagnostics["expected_total_noise_budget"]) > 1e-6:
        raise RuntimeError("Uniform/R1 total perturbation budgets are not comparable")

    restored = {
        METHODS[0]: (train_x, validation_x),
        METHODS[1]: (uniform_train, uniform_validation),
        METHODS[2]: (r1_train, r1_validation),
        R2_METHOD: (r1_train, r1_validation),
        R3_METHOD: (r1_train, r1_validation),
        HFSC_METHOD: (r1_train, r1_validation),
        RRDC_METHOD: (r1_train, r1_validation),
    }
    training = dict(config["training"])
    train_well_ids = np.asarray([by_instance[ref.instance_id].well_id for ref in train_refs], dtype=object)
    pretrain_orders, sampler_audit = supcon_orders(train_y, training, encoder_seed, train_well_ids)
    batching_comparison = {}
    if "comparison_balanced_sampler" in config:
        for name, mode in (("ORIGINAL", "original"), ("BALANCED", "balanced_positive_safe"),
                           ("CROSS_WELL", "cross_well_positive_safe")):
            if mode == training.get("supcon_batching", "original"):
                current_audit = sampler_audit
            else:
                comparison_training = dict(training); comparison_training["supcon_batching"] = mode
                comparison_training["balanced_sampler"] = copy.deepcopy(config["comparison_balanced_sampler"])
                _, current_audit = supcon_orders(
                    train_y, comparison_training, encoder_seed, train_well_ids, include_batch_audit=False
                )
            cross = {key: value for key, value in current_audit["cross_well"].items() if key != "per_epoch"}
            batching_comparison[name] = {"batch_order_sha256": current_audit["batch_order_sha256"], **cross}
    weights = sqrt_inverse_frequency_weights(train_y)
    seed_everything(encoder_seed)
    template = build_model(base["training"]["model"], train_x.shape[1], device)
    initial_state = copy.deepcopy(template.state_dict())
    initialization_sha256 = state_hash(initial_state)
    results = {}
    selected_methods = tuple(config.get("methods", METHODS))
    if any(method not in (*METHODS, R2_METHOD, R3_METHOD, HFSC_METHOD, RRDC_METHOD) for method in selected_methods):
        raise ValueError("unknown 3W diffusion comparison method")
    for method in selected_methods:
        method_result_path = output / f"{method}_result.json"
        if method_result_path.exists():
            results[method] = json.loads(method_result_path.read_text(encoding="utf-8"))
            print("skip", method, flush=True)
            continue
        print("start", method, flush=True)
        seed_everything(encoder_seed)
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
                probe_seed,
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
        "diffusion_seed": diffusion_seed,
        "validation_diffusion_seed": validation_diffusion_seed,
        "encoder_seed": encoder_seed,
        "probe_seed": probe_seed,
        "canonical_split_index": split_index,
        "primary_classes": list(FINAL_PRIMARY_CLASSES),
        "methods": results,
        "criticality": criticality_payload,
        "augmentation_diagnostics": {
            METHODS[1]: {"train": uniform_train_diagnostics, "validation": uniform_validation_diagnostics},
            METHODS[2]: {"train": r1_train_diagnostics, "validation": r1_validation_diagnostics},
            R2_METHOD: {"train": r1_train_diagnostics, "validation": r1_validation_diagnostics},
            R3_METHOD: {"train": r1_train_diagnostics, "validation": r1_validation_diagnostics},
            HFSC_METHOD: {"train": r1_train_diagnostics, "validation": r1_validation_diagnostics},
            RRDC_METHOD: {"train": r1_train_diagnostics, "validation": r1_validation_diagnostics},
        },
        "supcon_sampler": sampler_audit,
        "supcon_batching_comparison": batching_comparison,
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
            "critical_soft_mask_sha256": hashlib.sha256(np.ascontiguousarray(
                np.stack([hierarchical_soft_masks[kind] for kind in sorted(hierarchical_soft_masks)])
                if hierarchical_soft_masks is not None else critical_soft_mask).tobytes()).hexdigest(),
            "supcon_batch_order_sha256": sampler_audit["batch_order_sha256"],
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
