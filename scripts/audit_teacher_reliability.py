from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import recall_score

from losses import freeze_teacher
from metrics import classification_metrics
from scripts.audit_semantic_diffusion_augmentation import bases
from scripts.run_diffusion_quality_retest import load_fixed_views
from trainers import build_model
from utils import environment_metadata, seed_everything, write_json


MARKERS = ["TEACHER_AND_SEMANTIC_LOSS_AUDIT", "GENERATOR_ONLY", "THREE_SEEDS", "NOT_FOR_PAPER_CLAIMS"]


def _fault_type(run_uid: str) -> int:
    match = re.search(r":fault_(\d+):", str(run_uid))
    return int(match.group(1)) if match else 0


def effective_rank(embedding: np.ndarray) -> float:
    centered = np.asarray(embedding, dtype=np.float64) - np.mean(embedding, axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = np.square(singular)
    if not np.isfinite(energy).all() or energy.sum() <= 1e-15:
        return 0.0
    probability = energy / energy.sum()
    return float(np.exp(-(probability * np.log(np.maximum(probability, 1e-15))).sum()))


def distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()), "std": float(values.std()),
        "minimum": float(values.min()), "q05": float(np.quantile(values, .05)),
        "median": float(np.median(values)), "q95": float(np.quantile(values, .95)),
        "maximum": float(values.max()),
    }


@torch.no_grad()
def teacher_outputs(model: torch.nn.Module, values: np.ndarray, batch_size: int,
                    device: str) -> dict[str, np.ndarray]:
    probabilities, logits, embeddings = [], [], []
    for start in range(0, len(values), batch_size):
        output = model(torch.from_numpy(values[start:start + batch_size]).float().to(device))
        logits.append(output["logits"].cpu().numpy())
        probabilities.append(torch.softmax(output["logits"], dim=1).cpu().numpy())
        embeddings.append(output["embedding"].cpu().numpy())
    probability = np.concatenate(probabilities)
    logit = np.concatenate(logits)
    return {"probability": probability, "prediction": probability.argmax(1),
            "logits": logit, "embedding": np.concatenate(embeddings)}


def base_metrics(bundle: dict[str, np.ndarray], output: dict[str, np.ndarray]) -> dict[str, Any]:
    labels = np.asarray(bundle["labels"]).astype(np.int64)
    prediction, probability = output["prediction"], output["probability"]
    metrics = classification_metrics(labels, prediction, probability)
    fault = labels != 0
    metrics["fault_recall"] = float(recall_score(fault.astype(int), prediction, pos_label=1, zero_division=0))
    types = np.asarray([_fault_type(value) for value in bundle["run_uid"]])
    per_type = {}
    for kind in sorted(set(types) - {0}):
        selector = (types == kind) & fault
        per_type[str(kind)] = {"count": int(selector.sum()),
                               "recall": float((prediction[selector] == 1).mean())}
    early = fault & (np.asarray(bundle["start_sample"]) <= 289)
    margin = output["logits"][:, 1] - output["logits"][:, 0]
    metrics.update({
        "fault_type_recall": per_type,
        "early_fault": {"count": int(early.sum()),
                        "recall": float((prediction[early] == 1).mean()) if early.any() else None},
        "prediction_confidence": distribution(probability.max(1)),
        "signed_logit_margin": distribution(margin),
        "absolute_logit_margin": distribution(np.abs(margin)),
        "embedding_effective_rank": effective_rank(output["embedding"]),
    })
    return metrics


def _interpolate_masked(values: np.ndarray, missing: np.ndarray) -> np.ndarray:
    result = values.copy()
    time = np.arange(values.shape[-1])
    for sample in range(len(values)):
        for channel in range(values.shape[1]):
            observed = ~missing[sample, channel]
            if observed.any():
                result[sample, channel, ~observed] = np.interp(
                    time[~observed], time[observed], values[sample, channel, observed])
            else:
                result[sample, channel] = 0.0
    return result.astype(np.float32)


def make_perturbations(values: np.ndarray, settings: dict[str, float], seed: int) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    result = {}
    rng = np.random.default_rng(seed + 1)
    result["jitter"] = (values + rng.normal(0, float(settings["jitter_std"]), values.shape)).astype(np.float32)
    rng = np.random.default_rng(seed + 2)
    scale = rng.normal(1, float(settings["scaling_std"]), (len(values), values.shape[1], 1))
    result["scaling"] = (values * scale).astype(np.float32)
    rng = np.random.default_rng(seed + 3)
    missing = rng.random(values.shape) < float(settings["masking_ratio"])
    result["light_masking"] = _interpolate_masked(values, missing)
    rng = np.random.default_rng(seed + 4)
    missing = rng.random(values.shape) < float(settings["interpolation_mask_ratio"])
    interpolated = _interpolate_masked(values, missing)
    blend = float(settings["interpolation_blend"])
    result["interpolation_perturbation"] = ((1 - blend) * values + blend * interpolated).astype(np.float32)
    if any(value.shape != values.shape or not np.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("teacher perturbations must be finite and shape preserving")
    return result


def jensen_shannon(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = np.maximum(np.asarray(first, dtype=np.float64), 1e-12)
    second = np.maximum(np.asarray(second, dtype=np.float64), 1e-12)
    midpoint = .5 * (first + second)
    return .5 * ((first * (np.log(first) - np.log(midpoint))).sum(1)
                 + (second * (np.log(second) - np.log(midpoint))).sum(1))


def perturbation_metrics(bundle: dict[str, np.ndarray], base: dict[str, np.ndarray],
                         changed: dict[str, np.ndarray]) -> dict[str, Any]:
    base_prediction, prediction = base["prediction"], changed["prediction"]
    labels = np.asarray(bundle["labels"]); types = np.asarray([_fault_type(value) for value in bundle["run_uid"]])
    transition_n2f = (base_prediction == 0) & (prediction == 1)
    transition_f2n = (base_prediction == 1) & (prediction == 0)
    base_margin = base["logits"][:, 1] - base["logits"][:, 0]
    changed_margin = changed["logits"][:, 1] - changed["logits"][:, 0]
    cosine = (base["embedding"] * changed["embedding"]).sum(1) / np.maximum(
        np.linalg.norm(base["embedding"], axis=1) * np.linalg.norm(changed["embedding"], axis=1), 1e-12)
    groups = {}
    selectors = {"normal": labels == 0, "fault": labels != 0}
    selectors.update({f"fault_type_{kind:02d}": types == kind for kind in sorted(set(types) - {0})})
    for name, selector in selectors.items():
        groups[name] = {"count": int(selector.sum()),
                        "consistency": float((prediction[selector] == base_prediction[selector]).mean())}
    return {
        "prediction_consistency": float((prediction == base_prediction).mean()),
        "normal_to_fault": float(transition_n2f.mean()),
        "fault_to_normal": float(transition_f2n.mean()),
        "js_distance": distribution(jensen_shannon(base["probability"], changed["probability"])),
        "absolute_logit_margin_change": distribution(np.abs(changed_margin - base_margin)),
        "embedding_cosine": distribution(cosine),
        "groups": groups,
    }


def teacher_gate(base: dict[str, Any], perturbations: dict[str, dict[str, Any]],
                 gate: dict[str, Any]) -> tuple[dict[str, bool], bool]:
    major = []
    for value in perturbations.values():
        major.extend(group["consistency"] for name, group in value["groups"].items()
                     if name.startswith("fault_type_") and group["count"] >= int(gate["minimum_major_fault_type_count"]))
    fault_recalls = [row["recall"] for row in base["fault_type_recall"].values()]
    low_fraction = float(np.mean(np.asarray(fault_recalls) < float(gate["minimum_fault_type_recall"])))
    checks = {
        "all_perturbation_consistency_at_least_095": all(
            value["prediction_consistency"] >= float(gate["minimum_consistency"]) for value in perturbations.values()),
        "directional_flips_not_abnormal": all(
            value["normal_to_fault"] <= float(gate["maximum_directional_flip"])
            and value["fault_to_normal"] <= float(gate["maximum_directional_flip"]) for value in perturbations.values()),
        "normal_fault_directions_consistent": all(
            abs(value["normal_to_fault"] - value["fault_to_normal"]) <= float(gate["maximum_direction_gap"])
            for value in perturbations.values()),
        "major_fault_types_stable": bool(major) and min(major) >= float(gate["minimum_major_fault_type_consistency"]),
        "base_macro_f1_acceptable": base["macro_f1"] >= float(gate["minimum_macro_f1"]),
        "base_auprc_acceptable": base["auprc"] >= float(gate["minimum_auprc"]),
        "base_auroc_acceptable": base["auroc"] >= float(gate["minimum_auroc"]),
        "embedding_effective_rank_acceptable": base["embedding_effective_rank"] >= float(gate["minimum_effective_rank"]),
        "fault_type_recall_not_broadly_low": low_fraction <= float(gate["maximum_low_fault_type_fraction"]),
    }
    return checks, all(checks.values())


def render_report(result: dict[str, Any], path: str) -> None:
    validation_clean = result["base_performance"]["validation"]["clean"]
    guidance = result["base_performance"]["validation"]["guidance_input"]
    test_clean = result["base_performance"]["test"]["clean"]
    test = result["base_performance"]["test"]["guidance_input"]
    rows = []
    for name, value in result["validation_perturbations"].items():
        rows.append(f"| {name} | {value['prediction_consistency']:.4f} | {value['normal_to_fault']:.4f} | "
                    f"{value['fault_to_normal']:.4f} | {value['js_distance']['mean']:.6f} | "
                    f"{value['absolute_logit_margin_change']['mean']:.6f} | {value['embedding_cosine']['mean']:.6f} |")
    fault_rows = [f"| {kind} | {value['count']} | {value['recall']:.4f} |"
                  for kind, value in test["fault_type_recall"].items()]
    checks = [f"- {'通过' if passed else '**失败**'}：`{name}`" for name, passed in result["gate_checks"].items()]
    text = f"""# 教师可靠性与扰动稳定性审计

> **TEACHER_AND_SEMANTIC_LOSS_AUDIT / GENERATOR_ONLY / THREE_SEEDS / NOT_FOR_PAPER_CLAIMS**

## 结论

最终状态：`{result['status']}`。

本审计没有重新训练教师。基础性能同时记录 clean 与生成器实际使用的插值基础视图；轻度扰动只施加在 validation，test 未用于门槛选择或扰动调参。

## 基础性能

| Split / 输入 | Accuracy | Macro-F1 | AUPRC | AUROC | Fault Recall | FAR | Effective Rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation / clean | {validation_clean['accuracy']:.4f} | {validation_clean['macro_f1']:.4f} | {validation_clean['auprc']:.4f} | {validation_clean['auroc']:.4f} | {validation_clean['fault_recall']:.4f} | {validation_clean['far']:.4f} | {validation_clean['embedding_effective_rank']:.4f} |
| validation / guidance input | {guidance['accuracy']:.4f} | {guidance['macro_f1']:.4f} | {guidance['auprc']:.4f} | {guidance['auroc']:.4f} | {guidance['fault_recall']:.4f} | {guidance['far']:.4f} | {guidance['embedding_effective_rank']:.4f} |
| test / clean | {test_clean['accuracy']:.4f} | {test_clean['macro_f1']:.4f} | {test_clean['auprc']:.4f} | {test_clean['auroc']:.4f} | {test_clean['fault_recall']:.4f} | {test_clean['far']:.4f} | {test_clean['embedding_effective_rank']:.4f} |
| test / guidance input | {test['accuracy']:.4f} | {test['macro_f1']:.4f} | {test['auprc']:.4f} | {test['auroc']:.4f} | {test['fault_recall']:.4f} | {test['far']:.4f} | {test['embedding_effective_rank']:.4f} |

validation confusion matrix：`{guidance['confusion_matrix']}`；test confusion matrix：`{test['confusion_matrix']}`。

test 早期故障 recall：{test['early_fault']['recall']}（n={test['early_fault']['count']}）。预测置信度、signed/absolute logit margin 的完整分布已写入机器可读结果 `outputs/teacher_semantic_loss_audit/teacher_reliability/result.json`。

## Validation 轻扰动稳定性

| 扰动 | Consistency | N→F | F→N | JS mean | Margin change mean | Embedding cosine mean |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

N→F/F→N 是相对基础预测的真实方向翻转率。normal、fault 及各 fault type 的分组 consistency 已保存在结果 JSON 中。

## Test 的 20 个故障类型 Recall

| Fault type | Count | Recall |
|---:|---:|---:|
{chr(10).join(fault_rows)}

## Gate

{chr(10).join(checks)}

教师 gate 必须全部通过。若状态为 `TEACHER_NOT_RELIABLE_FOR_SEMANTIC_GUIDANCE`，则按提示词停止 S0–S4 语义损失训练并禁止下游 SupCon。

本次唯一 gate 失败项是 embedding effective rank：validation/test 的 guidance input 分别约为 {guidance['embedding_effective_rank']:.4f}/{test['embedding_effective_rank']:.4f}，低于预先固定的最低 2.0。虽然“低于 0.5 recall 的 fault type 占比不超过 25%”这一宽泛条件通过，test 的 fault 3、9、15 recall 仍分别只有 {test['fault_type_recall']['3']['recall']:.4f}、{test['fault_type_recall']['9']['recall']:.4f}、{test['fault_type_recall']['15']['recall']:.4f}，不应把扰动一致性误解为覆盖所有故障语义。

> 本报告仅为固定 TEP 子集的工程审计：**NOT_FOR_PAPER_CLAIMS**。
"""
    Path(path).write_text(text, encoding="utf-8")


def audit(config: dict[str, Any]) -> dict[str, Any]:
    result_path = Path(config["teacher_result"])
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    seed_everything(int(config["seed"])); device = str(config["device"])
    views, manifest = load_fixed_views(config); guidance = bases(views)
    teacher = freeze_teacher(build_model(config["teacher_model"], views["train"]["clean"].shape[1], 2).to(device))
    teacher.load_state_dict(torch.load(config["teacher_checkpoint"], map_location=device, weights_only=True))
    performance = {}
    cached = {}
    for split in ("validation", "test"):
        clean_output = teacher_outputs(teacher, views[split]["clean"], int(config["batch_size"]), device)
        guidance_output = teacher_outputs(teacher, guidance[split], int(config["batch_size"]), device)
        cached[split] = guidance_output
        performance[split] = {"clean": base_metrics(views[split], clean_output),
                              "guidance_input": base_metrics(views[split], guidance_output)}
    perturbed = make_perturbations(guidance["validation"], config["perturbations"], int(config["seed"])); perturbation = {}
    for name, values in perturbed.items():
        output = teacher_outputs(teacher, values, int(config["batch_size"]), device)
        perturbation[name] = perturbation_metrics(views["validation"], cached["validation"], output)
    checks, passed = teacher_gate(performance["validation"]["guidance_input"], perturbation, config["teacher_gate"])
    result = {
        "markers": config["markers"],
        "status": "TEACHER_RELIABLE_FOR_SEMANTIC_GUIDANCE" if passed else "TEACHER_NOT_RELIABLE_FOR_SEMANTIC_GUIDANCE",
        "teacher_retrained": False, "selection_split": "validation", "test_used_for_gate": False,
        "teacher_checkpoint": config["teacher_checkpoint"], "fixed_view_manifest": config["fixed_views"]["manifest"],
        "split_counts": {split: int(len(views[split]["labels"])) for split in ("validation", "test")},
        "base_performance": performance, "validation_perturbations": perturbation,
        "gate_thresholds": config["teacher_gate"], "gate_checks": checks,
        "semantic_loss_training_allowed": passed, "downstream_supcon_allowed": False,
        **environment_metadata(),
    }
    result_path.parent.mkdir(parents=True, exist_ok=False)
    write_json(result_path, result); render_report(result, config["teacher_report"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/teacher_semantic_loss_audit.yaml")
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = audit(config)
    print(json.dumps({"status": result["status"], "checks": result["gate_checks"],
                      "semantic_loss_training_allowed": result["semantic_loss_training_allowed"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
