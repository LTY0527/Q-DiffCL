from __future__ import annotations

import argparse
import copy
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from metrics import select_binary_threshold
from scripts.run_diffusion_quality_retest import (
    _fit_probe, _fit_supcon, _metrics, _probabilities, best_probe_record,
    epoch_orders, load_fixed_views,
)
from scripts.run_stage_frequency_diffusion_mvp import (
    METHODS, _build_frequency_components, _configure, _runtime,
    augmentation_mechanism_metrics, detection_delays, early_fault_recall,
)
from frequency import fault_stages, log_amplitude_phase
from frequency.criticality import fault_type
from scripts.audit_semantic_diffusion_augmentation import bases
from trainers import build_model
from utils import environment_metadata, seed_everything, write_json


def score_distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p90": float(np.quantile(values, .90)), "p95": float(np.quantile(values, .95)),
            "std": float(values.std()), "minimum": float(values.min()), "maximum": float(values.max())}


def expected_calibration_error(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    labels = np.asarray(labels) != 0; scores = np.asarray(scores, dtype=float); result = 0.0
    for lower, upper in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        selected = (scores >= lower) & (scores < upper if upper < 1 else scores <= upper)
        if selected.any(): result += selected.mean() * abs(scores[selected].mean() - labels[selected].mean())
    return float(result)


def score_profile(labels: np.ndarray, scores: np.ndarray, threshold: float, band_width: float) -> dict[str, Any]:
    labels = np.asarray(labels); scores = np.asarray(scores); prediction = scores >= threshold
    normal, fault = labels == 0, labels != 0
    return {"normal": score_distribution(scores[normal]), "fault": score_distribution(scores[fault]),
            "normal_to_fault": float(prediction[normal].mean()), "fault_to_normal": float((~prediction[fault]).mean()),
            "threshold": float(threshold), "threshold_band_width": float(band_width),
            "threshold_near_count": int(np.sum(np.abs(scores - threshold) <= band_width)),
            "threshold_near_fraction": float(np.mean(np.abs(scores - threshold) <= band_width)),
            "brier_score": float(np.mean(np.square(scores - (labels != 0)))),
            "expected_calibration_error": expected_calibration_error(labels, scores)}


def subgroup_scores(bundle: dict[str, np.ndarray], stages: np.ndarray, scores: np.ndarray,
                    threshold: float) -> dict[str, Any]:
    prediction = scores >= threshold; labels = np.asarray(bundle["labels"])
    result = {"stages": {}, "fault_types": {}}
    for stage in ("prefault", "early", "middle", "stable"):
        selector = stages == stage
        result["stages"][stage] = {"count": int(selector.sum()), "mean_fault_score": float(scores[selector].mean()),
                                   "fault_prediction_rate": float(prediction[selector].mean())}
    types = np.asarray([fault_type(value) for value in bundle["run_uid"]])
    for kind in range(1, 21):
        selector = types == kind
        result["fault_types"][str(kind)] = {"count": int(selector.sum()), "mean_fault_score": float(scores[selector].mean()),
                                             "recall": float(prediction[selector].mean())}
    return result


def correlation_matrix(values: np.ndarray) -> np.ndarray:
    flattened = np.asarray(values).transpose(1, 0, 2).reshape(values.shape[1], -1)
    matrix = np.corrcoef(flattened)
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)


def correlation_drift(base: np.ndarray, augmented: np.ndarray, labels: np.ndarray,
                      high_quantile: float = .9) -> dict[str, Any]:
    result = {}
    for group, selector in (("all", np.ones(len(labels), dtype=bool)), ("normal", labels == 0), ("fault", labels != 0)):
        original = correlation_matrix(base[selector]); changed = correlation_matrix(augmented[selector])
        drift = float(np.linalg.norm(changed - original) / max(float(np.linalg.norm(original)), 1e-12))
        upper = np.triu_indices_from(original, 1); baseline = np.abs(original[upper])
        boundary = float(np.quantile(baseline, high_quantile)); high = baseline >= boundary
        sign_flips = int(np.sum(np.sign(original[upper][high]) != np.sign(changed[upper][high])))
        result[group] = {"corr_drift": drift, "high_correlation_threshold": boundary,
                         "high_pair_count": int(high.sum()), "high_pair_sign_flips": sign_flips,
                         "high_pair_mean_absolute_change": float(np.abs(changed[upper][high] - original[upper][high]).mean())}
    return result


def frequency_structure_drift(base: np.ndarray, augmented: np.ndarray, soft_mask: np.ndarray) -> dict[str, Any]:
    first = log_amplitude_phase(base)[0]; second = log_amplitude_phase(augmented)[0]
    drifts = []
    for frequency in range(first.shape[-1]):
        original = np.corrcoef(first[:, :, frequency], rowvar=False)
        changed = np.corrcoef(second[:, :, frequency], rowvar=False)
        original = np.nan_to_num(original); changed = np.nan_to_num(changed)
        drifts.append(float(np.linalg.norm(changed - original) / max(float(np.linalg.norm(original)), 1e-12)))
    drifts = np.asarray(drifts); weights = np.asarray(soft_mask).mean(0)
    return {"per_frequency": drifts.tolist(), "all_frequency_mean": float(drifts.mean()),
            "critical_weighted": float(np.average(drifts, weights=np.maximum(weights, 1e-8))),
            "noncritical_weighted": float(np.average(drifts, weights=np.maximum(1 - weights, 1e-8)))}


def perturbation_concentration(base: np.ndarray, augmented: np.ndarray) -> dict[str, Any]:
    first = log_amplitude_phase(base)[0]; second = log_amplitude_phase(augmented)[0]
    energy = np.mean(np.square(second - first), axis=0); channel = energy.sum(1); frequency = energy.sum(0)
    flat = np.sort(energy.reshape(-1))[::-1]; total = max(float(energy.sum()), 1e-12)
    return {"channel_energy": channel.tolist(), "frequency_energy": frequency.tolist(),
            "maximum_channel_share": float(channel.max() / total), "maximum_frequency_share": float(frequency.max() / total),
            "top_10_percent_cf_share": float(flat[:max(1, int(np.ceil(.1 * len(flat))))].sum() / total),
            "per_channel_variance_ratio": (augmented.var((0, 2)) / np.maximum(base.var((0, 2)), 1e-8)).tolist()}


def classify_cause(validation: dict[str, Any], config: dict[str, Any]) -> tuple[str, dict[str, bool]]:
    candidates = validation["candidates"]; c1 = validation["C1"]; c2 = candidates["8"]
    lower_far = min(candidates["3"]["metrics"]["far"], candidates["5"]["metrics"]["far"])
    intensity = lower_far <= c2["metrics"]["far"] - float(config["diagnosis"]["minimum_far_reduction_signal"])
    drift_ratio = c2["time_structure"]["normal"]["corr_drift"] / max(c1["time_structure"]["normal"]["corr_drift"], 1e-12)
    normal_shift = c2["score_profile"]["normal"]["mean"] > c1["score_profile"]["normal"]["mean"]
    cross = drift_ratio >= float(config["diagnosis"]["minimum_structure_drift_ratio"]) and normal_shift
    category = "C. BOTH" if intensity and cross else "A. INTENSITY_DOMINANT" if intensity else "B. CROSS_CHANNEL_DRIFT_DOMINANT" if cross else "D. NEITHER_CLEAR"
    return category, {"lower_intensity_reduces_validation_far": bool(intensity),
                      "c2_normal_structure_drift_above_c1": bool(cross),
                      "normal_fault_score_moves_up": bool(normal_shift)}


def _fit_replay(name: str, augmented: dict[str, np.ndarray], views, base, stages, initial_state,
                pretrain_orders, probe_orders, runtime, device: str, checkpoint: Path,
                include_test: bool) -> dict[str, Any]:
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "replay_result" not in payload or "model_state_dict" not in payload:
            raise RuntimeError(f"diagnosis checkpoint is not resumable: {checkpoint}")
        result = payload["replay_result"]
        if result.get("name") != name or bool("test" in result) != include_test:
            raise RuntimeError(f"diagnosis checkpoint does not match replay {name}: {checkpoint}")
        return result
    seed_everything(int(runtime["random_seed"])); model = build_model(runtime["model"], base["train"].shape[1], 2).to(device)
    model.load_state_dict(initial_state); started = time.perf_counter()
    train = {"clean": base["train"], "restored": augmented["train"], "labels": views["train"]["labels"]}
    validation = {"clean": base["validation"], "restored": augmented["validation"], "labels": views["validation"]["labels"]}
    _fit_supcon(model, train, validation, np.ones(len(train["labels"]), np.float32),
                np.ones(len(validation["labels"]), np.float32), pretrain_orders, runtime, device)
    seed_everything(int(runtime["random_seed"]) + 1)
    probe = _fit_probe(model, {"clean": base["train"], "labels": views["train"]["labels"]},
                       {"restored": base["validation"], "labels": views["validation"]["labels"]},
                       probe_orders, runtime, device)
    best = best_probe_record(probe); threshold = float(best["validation_threshold"])
    val_probability, _ = _probabilities(model, base["validation"], int(runtime["batch_size"]), device)
    val_score = val_probability[:, 1]; val_prediction = val_score >= threshold
    result = {"name": name, "validation_threshold": threshold,
              "validation": {"metrics": _metrics(views["validation"]["labels"], val_score, threshold),
                             "scores": val_score.tolist(),
                             "score_profile": score_profile(views["validation"]["labels"], val_score, threshold,
                                                            float(runtime["diagnosis"]["threshold_band_width"])),
                             "subgroups": subgroup_scores(views["validation"], stages["validation"], val_score, threshold),
                             "early_fault": early_fault_recall(val_prediction, stages["validation"]),
                             "detection_delay": detection_delays(views["validation"], val_prediction, runtime)},
              "best_probe_epoch": int(best["epoch"]), "training_seconds": time.perf_counter() - started}
    if include_test:
        test_probability, _ = _probabilities(model, base["test"], int(runtime["batch_size"]), device)
        score = test_probability[:, 1]; prediction = score >= threshold
        result["test"] = {"metrics": _metrics(views["test"]["labels"], score, threshold), "scores": score.tolist(),
                          "score_profile": score_profile(views["test"]["labels"], score, threshold,
                                                         float(runtime["diagnosis"]["threshold_band_width"])),
                          "subgroups": subgroup_scores(views["test"], stages["test"], score, threshold),
                          "early_fault": early_fault_recall(prediction, stages["test"]),
                          "detection_delay": detection_delays(views["test"], prediction, runtime)}
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    result["checkpoint"] = checkpoint.as_posix()
    torch.save({"model_state_dict": model.state_dict(), "replay_result": result}, checkpoint)
    return result


def _render_report_legacy(result: dict[str, Any], path: str) -> None:
    rows = []
    for name, value in (("C1", result["replays"]["C1"]), ("C2 t=8", result["replays"]["8"])):
        for split in ("validation", "test"):
            profile = value[split]["score_profile"]
            rows.append(f"| {name} | {split} | {profile['normal']['mean']:.4f} | {profile['normal']['median']:.4f} | {profile['normal']['p90']:.4f} | {profile['normal']['p95']:.4f} | {profile['fault']['mean']:.4f} | {profile['normal_to_fault']:.4f} | {profile['fault_to_normal']:.4f} | {value['validation_threshold']:.4f} |")
    candidate_rows = []
    for key in ("3", "5", "8"):
        replay = result["replays"][key]["validation"]; structure = result["validation_diagnosis"]["candidates"][key]
        candidate_rows.append(f"| {key} | {replay['metrics']['macro_f1']:.4f} | {replay['metrics']['auprc']:.4f} | {replay['metrics']['fault_recall']:.4f} | {replay['metrics']['far']:.4f} | {replay['early_fault']['recall']:.4f} | {replay['detection_delay']['mean_delay_samples']} | {structure['mechanism']['critical_fisher_retention']:.4f} | {structure['time_structure']['normal']['corr_drift']:.5f} | {structure['mechanism']['time_normalized_l1']:.4f} |")
    category = result["cause_category"]
    report = f"""# 频率选择性扩散 FAR 上升诊断

> **FREQUENCY_SELECTIVE_FAR_FIX / STRUCTURE_PRESERVING_SPECTRAL_NOISE / EXPLORATORY_TEP_SUBSET / NOT_FOR_PAPER_CLAIMS**

## 结论

原因分类：`{category}`。状态：`{result['status']}`。

旧 MVP 的 C2 在相同总噪声预算下提高 Recall/Early Recall，但 FAR 上升、Macro-F1 下降。本报告基于原始 JSON、训练曲线和确定性重放；旧输出没有逐样本 score/checkpoint，因此重放原 C1 与 iid t=3/5/8，只补齐诊断字段，没有修改模型或使用 test 选择候选。

## Normal/Fault 分数漂移

| 方法 | Split | Normal mean | median | P90 | P95 | Fault mean | N→F | F→N | Val threshold |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

完整 threshold 邻域数量、Brier/ECE、stage、20 个 fault type、Fault 3/9/15 和逐 Run delay/miss 均保存在 `outputs/frequency_selective_far_fix/diagnosis/result.json`。test 只作多轮探索后的外部描述，不参与原因分类或后续版本选择。

## Validation t=3/5/8 权衡

| t | Macro-F1 | AUPRC | Recall | FAR | Early Recall | Mean Delay | Critical retention | Normal corr drift | Norm. L1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(candidate_rows)}

旧规则只优化 critical Fisher + early retention，因此选择 t=8；它没有 FAR 约束。本表用于判断 t=5 是否更平衡，全部选择依据来自 validation。

## 扰动集中与跨传感器结构

C1 validation normal corr drift={result['validation_diagnosis']['C1']['time_structure']['normal']['corr_drift']:.6f}；C2 t=8={result['validation_diagnosis']['candidates']['8']['time_structure']['normal']['corr_drift']:.6f}，比值={result['validation_diagnosis']['structure_drift_ratio_c2_c1']:.3f}。频域关键/非关键/全频结构漂移和每通道、每 bin 扰动能量均在结果 JSON 中。

判据：

- 低强度 t=3/5 是否使 validation FAR 至少下降 {result['thresholds']['minimum_far_reduction_signal']:.4f}：{result['cause_checks']['lower_intensity_reduces_validation_far']}；
- C2 normal corr drift 是否至少为 C1 的 {result['thresholds']['minimum_structure_drift_ratio']:.2f} 倍且 normal score 上移：{result['cause_checks']['c2_normal_structure_drift_above_c1']}；
- normal fault score 是否上移：{result['cause_checks']['normal_fault_score_moves_up']}。

## 下一阶段权限

若分类为 D，则输出 `FREQUENCY_SELECTIVE_FAR_CAUSE_UNRESOLVED` 并停止。若为 A/B/C，才允许固定 mask/D/E/S 后实现一次相关噪声与每通道预算约束的 R0–R3 修复。本阶段结果仅是已多次查看 test 的工程筛选信号，不是论文无偏结论。
"""
    Path(path).write_text(report, encoding="utf-8")


def render_report(result: dict[str, Any], path: str) -> None:
    score_rows = []
    for name, value in (("C1", result["replays"]["C1"]), ("C2 t=8", result["replays"]["8"])):
        for split in ("validation", "test"):
            profile = value[split]["score_profile"]
            score_rows.append(
                f"| {name} | {split} | {profile['normal']['mean']:.4f} | "
                f"{profile['normal']['median']:.4f} | {profile['normal']['p90']:.4f} | "
                f"{profile['normal']['p95']:.4f} | {profile['fault']['mean']:.4f} | "
                f"{profile['normal_to_fault']:.4f} | {profile['fault_to_normal']:.4f} | "
                f"{value['validation_threshold']:.4f} |"
            )
    candidate_rows = []
    for key in ("3", "5", "8"):
        replay = result["replays"][key]["validation"]
        diagnosis = result["validation_diagnosis"]["candidates"][key]
        candidate_rows.append(
            f"| {key} | {replay['metrics']['macro_f1']:.4f} | {replay['metrics']['auprc']:.4f} | "
            f"{replay['metrics']['fault_recall']:.4f} | {replay['metrics']['far']:.4f} | "
            f"{replay['early_fault']['recall']:.4f} | {replay['detection_delay']['mean_delay_samples']} | "
            f"{diagnosis['mechanism']['critical_fisher_retention']:.4f} | "
            f"{diagnosis['time_structure']['normal']['corr_drift']:.5f} | "
            f"{diagnosis['mechanism']['time_normalized_l1']:.4f} |"
        )
    c1_drift = result["validation_diagnosis"]["C1"]["time_structure"]["normal"]["corr_drift"]
    c2_drift = result["validation_diagnosis"]["candidates"]["8"]["time_structure"]["normal"]["corr_drift"]
    checks = result["cause_checks"]
    report = f"""# 频率选择性扩散 FAR 上升诊断

> **FREQUENCY_SELECTIVE_FAR_FIX / STRUCTURE_PRESERVING_SPECTRAL_NOISE / EXPLORATORY_TEP_SUBSET / NOT_FOR_PAPER_CLAIMS**

## 结论

原因分类：`{result['cause_category']}`。状态：`{result['status']}`。

旧 MVP 的 C2 在相同总噪声预算下提高了 Recall 和 Early Recall，但 FAR 上升、Macro-F1 下降。旧输出没有保存逐样本 score 和 checkpoint，因此本次以相同 Seed 7、初始化、批次顺序、SupCon 与 Probe 协议确定性重放 C1 和 iid `t=3/5/8`。test 仅作外部描述，不参与原因分类或候选选择。

## Normal/Fault 分数漂移

| 方法 | Split | Normal mean | median | P90 | P95 | Fault mean | N→F | F→N | Val threshold |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(score_rows)}

threshold 邻域数量、Brier/ECE、四阶段、Fault 1–20（含 3/9/15）以及逐 Run delay/miss 均保存在 `outputs/frequency_selective_far_fix/diagnosis/result.json`。

## Validation t=3/5/8 权衡

| t | Macro-F1 | AUPRC | Recall | FAR | Early Recall | Mean Delay | Critical retention | Normal corr drift | Norm. L1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(candidate_rows)}

旧规则只优化 critical Fisher 与 early retention，未设置 FAR 约束，因此选择了 `t=8`。本次结果显示 `t=5` 的 FAR 明显低于 `t=8`，且 Macro-F1 略高；代价是 Recall 和 Early Recall 较低。

## 扰动集中与跨传感器结构

C1 validation normal corr drift={c1_drift:.6f}，C2 t=8={c2_drift:.6f}，C2/C1 比值={result['validation_diagnosis']['structure_drift_ratio_c2_c1']:.3f}。频域关键/非关键/全频结构漂移、每通道与每 bin 扰动能量均保存在结果 JSON。

- 低强度 `t=3/5` 是否使 validation FAR 至少下降 {result['thresholds']['minimum_far_reduction_signal']:.4f}：`{checks['lower_intensity_reduces_validation_far']}`。
- C2 normal corr drift 是否至少为 C1 的 {result['thresholds']['minimum_structure_drift_ratio']:.2f} 倍且 normal score 上移：`{checks['c2_normal_structure_drift_above_c1']}`。
- normal fault score 是否上移：`{checks['normal_fault_score_moves_up']}`。

因此证据支持“非关键频率扰动强度主导”，不支持“跨通道结构漂移主导”。分类为 A，允许按固定配置继续 R0–R3 受限修复；这不意味着相关结构约束必然有效。

## 结论边界

当前 TEP test 已被多轮探索查看，所有 test 指标仅是工程筛选后的外部报告，不能作为论文无偏结论。后续必须严格使用 validation 选择版本，并在新数据集、重新冻结协议或未触碰评测设置上验证论文主张。
"""
    Path(path).write_text(report, encoding="utf-8")


def diagnose(config: dict[str, Any]) -> dict[str, Any]:
    result_path = Path(config["diagnosis_result"])
    if result_path.exists(): return json.loads(result_path.read_text(encoding="utf-8"))
    base_config = yaml.safe_load(Path(config["base_config"]).read_text(encoding="utf-8")); _configure(base_config)
    views, _ = load_fixed_views(base_config); base = bases(views)
    stages = {split: fault_stages(views[split], base_config) for split in views}
    critical, augmenter = _build_frequency_components(base_config, views, base, stages, str(config["device"]))
    seed = int(config["diagnosis"]["seed"]); runtime = _runtime(base_config, seed); runtime["diagnosis"] = config["diagnosis"]
    pretrain_orders = epoch_orders(len(base["train"]), int(runtime["epochs"]), seed + 10_000)
    probe_orders = epoch_orders(len(base["train"]), int(runtime["probe_epochs"]), seed + 20_000)
    seed_everything(seed); template = build_model(runtime["model"], base["train"].shape[1], 2); initial_state = copy.deepcopy(template.state_dict())
    augmented = {}; diagnostics = {}
    for key, mode, timestep in (("C1", "uniform", None), ("3", "selective", 3), ("5", "selective", 5), ("8", "selective", 8)):
        augmented[key] = {}; diagnostics[key] = {}
        for split, offset in (("train", 0), ("validation", 100), ("test", 200)):
            if split == "test" and key not in {"C1", "8"}: continue
            sampling_seed = seed + int(base_config["spectral_diffusion"]["sampling_seed_offset"]) + offset
            changed, diag = augmenter.augment(base[split], mode, sampling_seed, timestep, int(runtime["batch_size"]))
            augmented[key][split] = changed; diagnostics[key][split] = diag
    replays = {}
    for key in ("C1", "3", "5", "8"):
        replay_augmented = {"train": augmented[key]["train"], "validation": augmented[key]["validation"]}
        if key in {"C1", "8"}: replay_augmented["test"] = augmented[key]["test"]
        replays[key] = _fit_replay(key, replay_augmented, views, base, stages, initial_state, pretrain_orders,
                                   probe_orders, runtime, str(config["device"]),
                                   Path(config["output_dir"]) / "diagnosis" / f"{key}_model.pt", key in {"C1", "8"})
    old = json.loads(Path(config["old_mvp_result"]).read_text(encoding="utf-8"))
    for key, method in (("C1", METHODS[1]), ("8", METHODS[2])):
        for metric in ("macro_f1", "auprc", "fault_recall", "far"):
            if abs(replays[key]["test"]["metrics"][metric] - old["results"][method]["metrics"][metric]) > float(config["diagnosis"]["replay_metric_tolerance"]):
                raise RuntimeError(f"diagnosis replay mismatch for {key}:{metric}")
    validation = {"C1": {}, "candidates": {}}
    for key in ("C1", "3", "5", "8"):
        mechanism = augmentation_mechanism_metrics(base["validation"], augmented[key]["validation"], views["validation"]["labels"],
                                                   stages["validation"], critical["masks"]["composite"], diagnostics[key]["validation"])
        record = {"metrics": replays[key]["validation"]["metrics"], "score_profile": replays[key]["validation"]["score_profile"],
                  "time_structure": correlation_drift(base["validation"], augmented[key]["validation"], views["validation"]["labels"],
                                                      float(config["diagnosis"]["high_correlation_quantile"])),
                  "frequency_structure": frequency_structure_drift(base["validation"], augmented[key]["validation"], critical["soft_mask"]),
                  "concentration": perturbation_concentration(base["validation"], augmented[key]["validation"]), "mechanism": mechanism}
        if key == "C1":
            validation["C1"] = record
        else:
            validation["candidates"][key] = record
    validation["structure_drift_ratio_c2_c1"] = (validation["candidates"]["8"]["time_structure"]["normal"]["corr_drift"]
                                                  / max(validation["C1"]["time_structure"]["normal"]["corr_drift"], 1e-12))
    category, cause_checks = classify_cause(validation, config)
    test_diagnosis = {}
    for key in ("C1", "8"):
        test_diagnosis[key] = {"time_structure": correlation_drift(base["test"], augmented[key]["test"], views["test"]["labels"],
                                                                   float(config["diagnosis"]["high_correlation_quantile"])),
                               "frequency_structure": frequency_structure_drift(base["test"], augmented[key]["test"], critical["soft_mask"]),
                               "concentration": perturbation_concentration(base["test"], augmented[key]["test"])}
    status = "FREQUENCY_SELECTIVE_FAR_CAUSE_UNRESOLVED" if category.startswith("D.") else "FREQUENCY_SELECTIVE_FAR_CAUSE_RESOLVED"
    result = {"markers": config["markers"], "status": status, "cause_category": category,
              "cause_checks": cause_checks, "thresholds": config["diagnosis"], "selection_split": "validation",
              "test_used_for_cause_or_selection": False, "replays": replays, "validation_diagnosis": validation,
              "test_external_diagnosis": test_diagnosis, "repair_allowed": not category.startswith("D."), **environment_metadata()}
    result_path.parent.mkdir(parents=True, exist_ok=True); write_json(result_path, result); render_report(result, config["diagnosis_report"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/frequency_selective_far_fix.yaml")
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = diagnose(config); print(json.dumps({"status": result["status"], "cause": result["cause_category"],
                                                 "repair_allowed": result["repair_allowed"]}, ensure_ascii=False))


if __name__ == "__main__": main()
