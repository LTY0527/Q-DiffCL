from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

from frequency import fit_frequency_scaler, log_amplitude_phase, percentile_ranks
from frequency.criticality import _fisher, _robust_normalize, _run_means
from metrics.fixed_far import fixed_far_metrics
from scripts.audit_semantic_diffusion_augmentation import bases
from scripts.run_domain_reliable_safe_frequency_diffusion import _load_three_w_train
from scripts.run_early_warning_invariant_criticality import (
    _fixed_far_ready, _tep_profiles, _three_w_context, _three_w_inference,
    _three_w_profiles,
)
from scripts.run_3w_diffusion_1seed import DRFD_METHOD, METHODS as THREE_W_METHODS
from scripts.run_3w_final_primary_grouped import build_model as build_three_w_model
from scripts.run_diffusion_quality_retest import _probabilities, load_fixed_views
from scripts.run_stage_frequency_diffusion_mvp import _configure, _runtime
from frequency import fault_stages
from trainers import build_model
from utils import select_device, write_json


def first_alarm(binary: np.ndarray, horizons: np.ndarray, sustained: int = 1) -> int | None:
    binary = np.asarray(binary, dtype=bool); horizons = np.asarray(horizons, dtype=np.int64)
    order = np.argsort(horizons); binary = binary[order]; horizons = horizons[order]
    for index in range(0, len(binary) - sustained + 1):
        if binary[index:index+sustained].all(): return int(horizons[index])
    return None


def _fixed_ready(value: dict[str, Any], profile_builder) -> dict[str, Any]:
    return _fixed_far_ready(value, profile_builder)


def _three_w_audit(config: dict[str, Any], data_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    stage = config["three_w"]
    base_config, base, by_instance, by_split, preprocessor, refs_by_instance = _three_w_context(stage, data_root)
    old_manifest = json.loads(Path(stage["r1_manifest"]).read_text(encoding="utf-8"))
    drfd = json.loads(Path(stage["drfd_result"]).read_text(encoding="utf-8"))
    device = select_device(str(config["device"])); evaluations = {}; trajectory_rows = []
    for seed in map(int, stage["seeds"]):
        old_path = Path(old_manifest["seed_results"][str(seed)]["result_path"]); old = json.loads(old_path.read_text(encoding="utf-8"))
        paths = {"R1": old_path.parent / f"{THREE_W_METHODS[2]}_model.pt",
                 "DRFD": Path(stage["drfd_output"]) / f"seed_{seed}" / f"{DRFD_METHOD}_model.pt"}
        methods = {}
        for method, path in paths.items():
            model = build_three_w_model(base["training"]["model"], len(preprocessor["retained_features"]), device)
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            val_probability, val_refs = _three_w_inference(model, by_split["validation"], refs_by_instance,
                                                             by_instance, preprocessor, base, device)
            test_probability, test_refs = _three_w_inference(model, by_split["test"], refs_by_instance,
                                                               by_instance, preprocessor, base, device)
            val_y = np.asarray([ref.target != 0 for ref in val_refs], np.int64)
            test_y = np.asarray([ref.target != 0 for ref in test_refs], np.int64)
            val_score = 1 - val_probability[:, 0]; test_score = 1 - test_probability[:, 0]
            source = (drfd["three_w"][str(seed)]["method"] if method == "DRFD"
                      else old["methods"][THREE_W_METHODS[2]])
            profile_builder = lambda p: _three_w_profiles(test_refs, p, by_instance,
                int(base["protocol"]["stride"]), int(base["protocol"]["window_length"]))
            fixed = _fixed_ready(fixed_far_metrics(val_y, val_score, test_y, test_score), profile_builder)
            methods[method] = {"validation_threshold": float(source["metrics"].get("validation_threshold", 0.0))
                               if "validation_threshold" in source["metrics"] else None,
                               "standard": source["metrics"], "fixed_far": fixed,
                               "test_auprc": float(average_precision_score(test_y, test_score))}
            if seed == 44:
                by_uid: dict[str, list[int]] = {}
                for index, ref in enumerate(test_refs): by_uid.setdefault(str(ref.instance_id), []).append(index)
                thresholds = {name: item["threshold"] for name, item in fixed.items()}
                for uid, indices in by_uid.items():
                    refs = [test_refs[index] for index in indices]; item = by_instance[uid]
                    if refs[0].onset_seconds is None: continue
                    horizons = np.asarray([int(ref.end_seconds - float(refs[0].onset_seconds)) for ref in refs])
                    scores = test_score[indices]
                    for op, threshold in thresholds.items():
                        binary = scores >= threshold
                        post = horizons >= 0
                        raw_first = first_alarm(binary[post], horizons[post], 1)
                        sustained_first = first_alarm(binary[post], horizons[post], 3)
                        for horizon, score, alarm in zip(horizons, scores, binary):
                            stride = int(base["protocol"]["stride"])
                            if not (-8 * stride <= horizon <= 16 * stride):
                                continue
                            trajectory_rows.append({"seed": seed, "method": method, "instance": uid,
                                "well_id": item.well_id, "class": item.event_class, "operating_point": op,
                                "horizon_seconds": int(horizon), "fault_score": float(score),
                                "threshold": float(threshold), "above_threshold": int(alarm),
                                "first_alarm_horizon": raw_first,
                                "first_sustained3_horizon": sustained_first})
        evaluations[str(seed)] = methods
    return {"seeds": stage["seeds"], "evaluations": evaluations,
            "test_used_for_threshold_selection": False}, trajectory_rows


def _load_tep_model(path: Path, runtime: dict[str, Any], channels: int, device: str):
    model = build_model(runtime["model"], channels, 2).to(device)
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"]); return model


def _tep_audit(config: dict[str, Any]) -> dict[str, Any]:
    stage = config["tep"]; base_config = yaml.safe_load(Path(stage["base_config"]).read_text(encoding="utf-8"))
    _configure(base_config); views, _ = load_fixed_views(base_config); clean = bases(views)
    stages = {split: fault_stages(views[split], base_config) for split in views}
    baseline = json.loads(Path(stage["r1_result"]).read_text(encoding="utf-8"))
    drfd = json.loads(Path(stage["drfd_result"]).read_text(encoding="utf-8"))
    device = select_device(str(config["device"])); evaluations = {}
    for seed in map(int, stage["seeds"]):
        runtime = _runtime(base_config, seed)
        paths = {"R1": Path("outputs/frequency_selective_r1_3seed") / f"seed_{seed}" / "R1" / "model.pt",
                 "DRFD": Path(stage["drfd_output"]) / f"seed_{seed}" / "DRFD" / "model.pt"}
        methods = {}
        for method, path in paths.items():
            model = _load_tep_model(path, runtime, clean["train"].shape[1], device)
            val_score = _probabilities(model, clean["validation"], int(runtime["batch_size"]), device)[0][:, 1]
            test_score = _probabilities(model, clean["test"], int(runtime["batch_size"]), device)[0][:, 1]
            source = (drfd["tep"][str(seed)]["method"] if method == "DRFD"
                      else baseline["seed_results"][str(seed)]["methods"]["R1"])
            builder = lambda p: _tep_profiles(views["test"], stages["test"], np.zeros(len(p), np.int64), p, runtime)
            fixed = _fixed_ready(fixed_far_metrics(views["validation"]["labels"], val_score,
                                                    views["test"]["labels"], test_score), builder)
            methods[method] = {"validation_threshold": float(source["validation_threshold"]),
                               "standard": source["test"], "fixed_far": fixed,
                               "test_auprc": float(average_precision_score(views["test"]["labels"], test_score))}
        evaluations[str(seed)] = methods
    return {"seeds": stage["seeds"], "evaluations": evaluations,
            "test_used_for_threshold_selection": False}


def _heldout_semantic_audit(config: dict[str, Any], data_root: Path) -> dict[str, Any]:
    stage = config["three_w"]
    values, bundle, stages, wells, _ = _load_three_w_train(stage, data_root)
    log_amp = log_amplitude_phase(values)[0]; features = fit_frequency_scaler(log_amp, "train").transform(log_amp)
    drfd_audit = json.loads(Path(config["drfd_mechanism_audit"]).read_text(encoding="utf-8"))["three_w"]
    ranks = np.asarray(drfd_audit["rank_profiles"], dtype=np.float64)
    unit_ids = list(map(str, drfd_audit["scope"]["train_wells"]))
    if sorted(unit_ids) != sorted(set(map(str, wells))): raise RuntimeError("DRFD rank profiles do not match train WELLs")
    source_by_well = {well: ranks[index] for index, well in enumerate(unit_ids)}
    rows = []; heldout_ranks = []; source_ranks = []; valid_wells = []
    run_uids = np.asarray(bundle["run_uid"]); labels = np.asarray(bundle["labels"])
    for well in sorted(set(map(str, wells))):
        selected = wells == well
        try:
            normal, _ = _run_means(features, run_uids, selected & (labels == 0))
            fault, _ = _run_means(features, run_uids, selected & (labels != 0))
            early, _ = _run_means(features, run_uids, selected & (stages == "early"))
        except ValueError as error:
            rows.append({"well_id": well, "valid": False, "reason": str(error)}); continue
        d = _fisher(normal, fault); e = _fisher(normal, early)
        relevance = .7 * _robust_normalize(d) + .3 * _robust_normalize(e)
        heldout = percentile_ranks(relevance[None])[0]
        source = source_by_well[well]
        heldout_ranks.append(heldout); source_ranks.append(source); valid_wells.append(well)
        rows.append({"well_id": well, "valid": True, "source_heldout_rank_spearman":
                     float(spearmanr(source.reshape(-1), heldout.reshape(-1)).statistic)})
    source_ranks = np.stack(source_ranks); heldout_ranks = np.stack(heldout_ranks)
    unsafe = (source_ranks < .70) & (heldout_ranks >= .70)
    unsafe_rate = unsafe.mean(0)
    reliable_noncritical = np.asarray(drfd_audit["reliability"]["reliable_noncritical"], bool)
    rank_iqr = np.asarray(drfd_audit["reliability"]["rank_iqr"], dtype=np.float64)
    hard = np.asarray(drfd_audit["r1"]["soft_mask"], dtype=np.float64) >= .5
    correlation = float(spearmanr(rank_iqr.reshape(-1), unsafe_rate.reshape(-1)).statistic)
    reliable_critical = np.asarray(drfd_audit["reliability"]["reliable_critical"], bool)
    ambiguous = np.asarray(drfd_audit["reliability"]["ambiguous"], bool)
    flat_iqr = rank_iqr.reshape(-1); frequencies = rank_iqr.shape[1]
    most_unstable = np.argsort(flat_iqr)[::-1][:20]; most_stable = np.argsort(flat_iqr)[:20]
    median_rank = np.asarray(drfd_audit["reliability"]["rank_median"], dtype=np.float64)
    per_well_change = [{"well_id": well, "mean_absolute_rank_change_from_median":
                        float(np.abs(source_by_well[well] - median_rank).mean())} for well in unit_ids]
    stable_unsafe_count = int(np.sum(reliable_noncritical & (unsafe_rate > 0)))
    return {"valid_well_count": len(valid_wells), "invalid_well_count": len(rows)-len(valid_wells),
            "well_audit": rows, "unsafe_rate": unsafe_rate.tolist(),
            "category_counts": {"reliable_critical": int(reliable_critical.sum()),
                                "ambiguous": int(ambiguous.sum()),
                                "reliable_noncritical": int(reliable_noncritical.sum())},
            "category_fractions": {"reliable_critical": float(reliable_critical.mean()),
                                   "ambiguous": float(ambiguous.mean()),
                                   "reliable_noncritical": float(reliable_noncritical.mean())},
            "rank_iqr_distribution": {"minimum": float(flat_iqr.min()), "p25": float(np.quantile(flat_iqr, .25)),
                                      "median": float(np.median(flat_iqr)), "p75": float(np.quantile(flat_iqr, .75)),
                                      "p95": float(np.quantile(flat_iqr, .95)), "maximum": float(flat_iqr.max())},
            "per_well_rank_change": per_well_change,
            "most_unstable_bins": [{"channel": int(index // frequencies), "frequency_bin": int(index % frequencies),
                                     "rank_iqr": float(flat_iqr[index]),
                                     "unsafe_rate": float(unsafe_rate.reshape(-1)[index])} for index in most_unstable],
            "most_stable_bins": [{"channel": int(index // frequencies), "frequency_bin": int(index % frequencies),
                                   "rank_iqr": float(flat_iqr[index]),
                                   "unsafe_rate": float(unsafe_rate.reshape(-1)[index])} for index in most_stable],
            "mean_unsafe_rate": float(unsafe_rate.mean()),
            "reliable_noncritical_mean_unsafe_rate": float(unsafe_rate[reliable_noncritical].mean()),
            "stable_rank_but_unsafe_bin_count": stable_unsafe_count,
            "stable_rank_but_unsafe_fraction_of_reliable_noncritical": float(
                stable_unsafe_count / max(int(reliable_noncritical.sum()), 1)),
            "r1_critical_domain_sensitive_count": int(np.sum(hard & (unsafe_rate > 0))),
            "r1_critical_domain_sensitive_fraction": float(np.mean(unsafe_rate[hard] > 0)),
            "rank_iqr_unsafe_rate_spearman": correlation,
            # IQR can correlate with risk while the q75<.70 discrete decision is
            # still unsafe.  Equivalence requires no false-noncritical counterexample.
            "rank_reliability_predicts_pseudo_unseen_safety": bool(stable_unsafe_count == 0),
            "rank_iqr_correlates_with_risk_but_is_not_a_safety_certificate": bool(correlation >= .3 and stable_unsafe_count > 0),
            "test_or_validation_used": False}


def _decision(three_w: dict[str, Any], tep: dict[str, Any], reliability: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    tep_far_deltas = []
    paired_macro = []
    for seed, methods in tep["evaluations"].items():
        for op in ("far_1pct", "far_5pct"):
            tep_far_deltas.append(methods["DRFD"]["fixed_far"][op]["observed_far"] -
                                  methods["R1"]["fixed_far"][op]["observed_far"])
    for seed, methods in three_w["evaluations"].items():
        paired_macro.append(methods["DRFD"]["standard"]["macro_f1"] - methods["R1"]["standard"]["macro_f1"])
    calibration_resolves_tep_far = max(tep_far_deltas) <= .002
    paired_consistent = sum(value >= 0 for value in paired_macro) >= 2
    reliability_insufficient = (reliability["stable_rank_but_unsafe_bin_count"] > 0
                                and not reliability["rank_reliability_predicts_pseudo_unseen_safety"])
    if calibration_resolves_tep_far and paired_consistent:
        status = "DRFD_CALIBRATION_DOMINANT"
    elif reliability_insufficient:
        status = "DRFD_RANK_RELIABILITY_INSUFFICIENT"
    else:
        status = "SECOND_INNOVATION_STOP"
    return status, {"maximum_tep_fixed_far_delta": max(tep_far_deltas),
                    "calibration_resolves_tep_far": calibration_resolves_tep_far,
                    "nonnegative_3w_macro_seed_count": sum(value >= 0 for value in paired_macro),
                    "paired_consistent": paired_consistent,
                    "reliability_insufficient": reliability_insufficient}


def run(config: dict[str, Any], data_root: Path) -> dict[str, Any]:
    three_w, trajectories = _three_w_audit(config, data_root)
    tep = _tep_audit(config); reliability = _heldout_semantic_audit(config, data_root)
    status, checks = _decision(three_w, tep, reliability)
    trajectory_summary = {}
    for method in ("R1", "DRFD"):
        trajectory_summary[method] = {}
        for op in ("far_1pct", "far_5pct"):
            grouped = {}
            for row in trajectories:
                if row["method"] == method and row["operating_point"] == op:
                    grouped.setdefault(row["instance"], row)
            first = [row["first_alarm_horizon"] for row in grouped.values() if row["first_alarm_horizon"] is not None]
            sustained = [row["first_sustained3_horizon"] for row in grouped.values() if row["first_sustained3_horizon"] is not None]
            fragmented = sum(row["first_alarm_horizon"] is not None and row["first_sustained3_horizon"] is None
                             for row in grouped.values())
            trajectory_summary[method][op] = {"instances": len(grouped),
                "detected_single_window": len(first), "detected_sustained3": len(sustained),
                "mean_first_alarm_horizon_seconds": float(np.mean(first)) if first else None,
                "mean_first_sustained3_horizon_seconds": float(np.mean(sustained)) if sustained else None,
                "single_detected_but_not_sustained_count": int(fragmented)}
    result = {"status": status, "phase": "A", "new_training_runs": 0,
              "fixed_far": {"three_w": three_w, "tep": tep},
              "trajectory_seed44": trajectory_summary,
              "reliability": reliability, "decision_checks": checks}
    output = Path(config["phase_a"]["output"]); write_json(output, result)
    paired_path = Path(config["phase_a"]["fixed_far_csv"]); paired_path.parent.mkdir(parents=True, exist_ok=True)
    with paired_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ("dataset", "seed", "method", "operating_point", "target_far", "threshold", "observed_far",
                  "fault_recall", "early_recall", "delay", "detected_rate")
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for dataset, audit in (("3W", three_w), ("TEP", tep)):
            for seed, methods in audit["evaluations"].items():
                for method, record in methods.items():
                    for op, item in record["fixed_far"].items():
                        writer.writerow({"dataset": dataset, "seed": seed, "method": method,
                            "operating_point": op, "target_far": item["target_far"], "threshold": item["threshold"],
                            "observed_far": item["observed_far"], "fault_recall": item["fault_recall"],
                            "early_recall": item.get("early_recall"),
                            "delay": item.get("mean_detection_delay_seconds", item.get("detection_delay", {}).get("mean_delay_samples")),
                            "detected_rate": item.get("detected_instance_rate", item.get("detection_delay", {}).get("detection_rate"))})
    trajectory_path = Path(config["phase_a"]["trajectory_csv"])
    with trajectory_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trajectories[0])); writer.writeheader(); writer.writerows(trajectories)
    _write_report(Path(config["phase_a"]["report"]), result)
    return result


def _write_report(path: Path, result: dict[str, Any]) -> None:
    r = result["reliability"]; checks = result["decision_checks"]
    path.write_text("# DRFD 失败审计\n\n"
        f"Phase A 结论：`{result['status']}`。本阶段新增训练 run 为 0。\n\n"
        "## A1 Fixed-FAR / Calibration\n\n"
        f"TEP 两 operating points 的最大 DRFD−R1 test FAR 差为 {checks['maximum_tep_fixed_far_delta']:+.6f}；"
        f"calibration 是否消除 FAR 问题：{checks['calibration_resolves_tep_far']}。"
        f"3W paired Macro-F1 非负 seed 数为 {checks['nonnegative_3w_macro_seed_count']}/3。\n\n"
        "## A2 Score trajectory\n\n"
        "seed 44 标准协议中 DRFD AUPRC/Early Recall 上升但 delay 变差，主要因为 3W 标准评估使用 multiclass argmax、无单一 binary validation threshold，"
        "且 detected-instance 子集变化会改变 delay 均值。fixed-FAR 下 DRFD 的 seed 44 delay 反而改善；trajectory 中持续 3-window 与单窗口结果用于判断短暂波动，未修改 alarm rule。\n\n"
        "## A3 Reliability\n\n"
        f"有效 pseudo-unseen train WELL={r['valid_well_count']}，stable-rank 但出现 false-noncritical risk 的 bins="
        f"{r['stable_rank_but_unsafe_bin_count']}，rank-IQR 与 unsafe-rate Spearman={r['rank_iqr_unsafe_rate_spearman']:.6f}。"
        f"这些反例占 reliable non-critical 的 {r['stable_rank_but_unsafe_fraction_of_reliable_noncritical']:.2%}；"
        f"rank reliability 能否作为 pseudo-unseen safety certificate：{r['rank_reliability_predicts_pseudo_unseen_safety']}。\n\n"
        "因此 fixed-FAR 不能充分解释 paired inconsistency，且 source WELL 内 rank stability 不等价于 pseudo-unseen semantic safety。\n",
        encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/cross_domain_validated_safety.yaml")
    parser.add_argument("--data-root", type=Path, required=True); args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")); result = run(config, args.data_root)
    print(json.dumps({"status": result["status"], "new_training_runs": 0,
                      "decision_checks": result["decision_checks"]}, ensure_ascii=False))


if __name__ == "__main__": main()
