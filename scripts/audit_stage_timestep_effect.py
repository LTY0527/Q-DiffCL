from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from frequency import fault_stages, log_amplitude_phase
from scripts.audit_semantic_diffusion_augmentation import bases
from scripts.run_diffusion_quality_retest import _probabilities, load_fixed_views
from scripts.run_frequency_selective_r1_3seed import array_sha256, file_sha256
from scripts.run_stage_aware_diffusion_curriculum import training_stage_names
from scripts.run_stage_frequency_diffusion_mvp import _build_frequency_components, _configure, _runtime
from trainers import build_model
from utils import environment_metadata, write_json


STAGES = ("normal", "early", "middle", "stable")


def effect_ratio(high: float, low: float, epsilon: float = 1e-12) -> float:
    return float(high / max(low, epsilon))


def classify_effect(stage_records: dict[str, Any], threshold: float = 1.10,
                    minimum_strong_layers: int = 2) -> tuple[str, dict[str, Any]]:
    ratios = {
        "time": float(np.median([stage_records[s]["ratios"]["time_t5_t3"] for s in STAGES])),
        "frequency": float(np.median([stage_records[s]["ratios"]["noncritical_frequency_t5_t3"] for s in STAGES])),
        "representation": float(np.median([stage_records[s]["ratios"]["representation_t5_t3"] for s in STAGES])),
    }
    strong = {key: bool(value >= threshold) for key, value in ratios.items()}
    monotonic = {stage: bool(stage_records[stage]["monotonic"]["time"]
                              and stage_records[stage]["monotonic"]["noncritical_frequency"]
                              and stage_records[stage]["monotonic"]["representation"]) for stage in STAGES}
    present = sum(strong.values()) >= int(minimum_strong_layers)
    status = "STAGE_TIMESTEP_EFFECT_PRESENT_BUT_TASK_NO_GAIN" if present else "STAGE_TIMESTEP_EFFECT_WEAK"
    return status, {"median_ratios": ratios, "strong_layers": strong,
                    "strong_layer_count": sum(strong.values()), "stage_monotonic_all_layers": monotonic}


def _time_metrics(base: np.ndarray, changed: np.ndarray) -> dict[str, float]:
    delta = changed - base; scale = np.maximum(base.std(axis=(1, 2)), 1e-6)
    channel_variance = np.mean(np.abs(changed.var((0, 2)) - base.var((0, 2))) / np.maximum(base.var((0, 2)), 1e-8))
    return {"normalized_l1": float(np.mean(np.abs(delta).mean((1, 2)) / scale)),
            "normalized_l2": float(np.mean(np.sqrt(np.mean(delta ** 2, axis=(1, 2))) / scale)),
            "mse": float(np.mean(delta ** 2)), "per_channel_variance_change": float(channel_variance),
            "peak_absolute_change": float(np.max(np.abs(delta)))}


def _frequency_metrics(base: np.ndarray, changed: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    base_log = log_amplitude_phase(base)[0]; changed_log = log_amplitude_phase(changed)[0]
    delta = changed_log - base_log; base_energy = np.square(np.expm1(base_log)); changed_energy = np.square(np.expm1(changed_log))
    result = {}
    for name, selected in (("critical", mask), ("noncritical", ~mask), ("all", np.ones_like(mask, bool))):
        b = base_log[:, selected]; d = delta[:, selected]
        signal = float(np.mean(b ** 2)); noise = float(np.mean(d ** 2))
        result[name] = {"relative_log_l1": float(np.mean(np.abs(d)) / max(float(np.mean(np.abs(b))), 1e-12)),
                        "relative_log_l2": float(np.sqrt(np.mean(d ** 2)) / max(float(np.sqrt(np.mean(b ** 2))), 1e-12)),
                        "spectral_snr_db": float(10 * np.log10(max(signal, 1e-12) / max(noise, 1e-12))),
                        "energy_retention": float(changed_energy[:, selected].mean() / max(float(base_energy[:, selected].mean()), 1e-12))}
    return result


def _fisher(values: np.ndarray, labels: np.ndarray) -> float:
    first, second = values[labels == 0], values[labels != 0]
    if not len(first) or not len(second): return 1.0
    return float(np.mean(np.square(first.mean(0) - second.mean(0)) / (first.var(0) + second.var(0) + 1e-8)))


def run(config: dict[str, Any]) -> dict[str, Any]:
    result_path = Path(config["result"])
    if result_path.exists(): return json.loads(result_path.read_text(encoding="utf-8"))
    if list(map(int, config["timesteps"])) != [3, 4, 5]: raise ValueError("stage effect audit only allows t=3/4/5")
    base_config = yaml.safe_load(Path(config["base_config"]).read_text(encoding="utf-8")); _configure(base_config)
    views, _ = load_fixed_views(base_config); base = bases(views); stages = fault_stages(views["validation"], base_config)
    stage_names = training_stage_names(views["validation"]["labels"], stages)
    all_stages = {split: fault_stages(views[split], base_config) for split in views}
    critical, augmenter = _build_frequency_components(base_config, views, base, all_stages, str(config["device"]))
    runtime = _runtime(base_config, int(config["seed"])); payload = torch.load(config["r1_checkpoint"], map_location=config["device"], weights_only=False)
    model = build_model(runtime["model"], base["train"].shape[1], 2).to(config["device"]); model.load_state_dict(payload["model_state_dict"]); model.eval()
    sampling_seed = int(config["seed"]) + int(base_config["spectral_diffusion"]["sampling_seed_offset"]) + 100
    augmented = {}; embeddings = {}; diagnostics = {}
    _, base_embedding = _probabilities(model, base["validation"], int(runtime["batch_size"]), str(config["device"]))
    for timestep in map(int, config["timesteps"]):
        augmented[timestep], diagnostics[timestep] = augmenter.augment(
            base["validation"], "selective", sampling_seed, timestep, int(runtime["batch_size"]), noise_structure="iid")
        _, embeddings[timestep] = _probabilities(model, augmented[timestep], int(runtime["batch_size"]), str(config["device"]))
    records = {}
    mask = critical["masks"]["composite"].astype(bool); labels = views["validation"]["labels"]
    base_fisher = _fisher(log_amplitude_phase(base["validation"])[0][:, mask], labels)
    for stage in STAGES:
        selected = stage_names == stage; stage_result = {"count": int(selected.sum()), "timesteps": {}}
        for timestep in (3, 4, 5):
            time = _time_metrics(base["validation"][selected], augmented[timestep][selected])
            frequency = _frequency_metrics(base["validation"][selected], augmented[timestep][selected], mask)
            z0, zt = base_embedding[selected], embeddings[timestep][selected]
            representation = {"cosine_base_augmented": float(np.mean(np.sum(z0 * zt, 1) / (np.linalg.norm(z0, axis=1) * np.linalg.norm(zt, axis=1) + 1e-12))),
                              "l2_base_augmented": float(np.mean(np.linalg.norm(z0 - zt, axis=1)))}
            frequency["critical"]["fisher_retention"] = float(_fisher(log_amplitude_phase(augmented[timestep])[0][:, mask], labels) / max(base_fisher, 1e-12))
            stage_result["timesteps"][str(timestep)] = {"time": time, "frequency": frequency, "representation": representation,
                                                        "mechanism": diagnostics[timestep]}
        t3, t4, t5 = (stage_result["timesteps"][str(t)] for t in (3, 4, 5))
        stage_result["representation_pair"] = {"cosine_t3_t5": float(np.mean(np.sum(embeddings[3][selected] * embeddings[5][selected], 1) /
                                                            (np.linalg.norm(embeddings[3][selected], axis=1) * np.linalg.norm(embeddings[5][selected], axis=1) + 1e-12))),
                                                "l2_t3_t5": float(np.mean(np.linalg.norm(embeddings[3][selected] - embeddings[5][selected], axis=1)))}
        stage_result["ratios"] = {"time_t4_t3": effect_ratio(t4["time"]["normalized_l1"], t3["time"]["normalized_l1"]),
                                  "time_t5_t3": effect_ratio(t5["time"]["normalized_l1"], t3["time"]["normalized_l1"]),
                                  "noncritical_frequency_t4_t3": effect_ratio(t4["frequency"]["noncritical"]["relative_log_l1"], t3["frequency"]["noncritical"]["relative_log_l1"]),
                                  "noncritical_frequency_t5_t3": effect_ratio(t5["frequency"]["noncritical"]["relative_log_l1"], t3["frequency"]["noncritical"]["relative_log_l1"]),
                                  "representation_t4_t3": effect_ratio(t4["representation"]["l2_base_augmented"], t3["representation"]["l2_base_augmented"]),
                                  "representation_t5_t3": effect_ratio(t5["representation"]["l2_base_augmented"], t3["representation"]["l2_base_augmented"])}
        stage_result["monotonic"] = {"time": t3["time"]["normalized_l1"] <= t4["time"]["normalized_l1"] <= t5["time"]["normalized_l1"],
                                     "noncritical_frequency": t3["frequency"]["noncritical"]["relative_log_l1"] <= t4["frequency"]["noncritical"]["relative_log_l1"] <= t5["frequency"]["noncritical"]["relative_log_l1"],
                                     "representation": t3["representation"]["l2_base_augmented"] <= t4["representation"]["l2_base_augmented"] <= t5["representation"]["l2_base_augmented"]}
        records[stage] = stage_result
    status, decision = classify_effect(records, float(config["effect_ratio_threshold"]), int(config["minimum_strong_layers"]))
    result = {"markers": config["markers"], "status": status, "budget_mvp_allowed": status == "STAGE_TIMESTEP_EFFECT_WEAK",
              "selection_split": "validation", "test_used": False, "timesteps": config["timesteps"], "stages": records,
              "decision": decision, "fingerprints": {"manifest_sha256": file_sha256(config["fixed_views_manifest"]),
              "mask_sha256": array_sha256(mask), "r1_checkpoint_sha256": file_sha256(config["r1_checkpoint"])}, **environment_metadata()}
    result_path.parent.mkdir(parents=True, exist_ok=True); write_json(result_path, result); render_report(result, config["report"]); return result


def render_report(result: dict[str, Any], path: str) -> None:
    rows=[]
    for stage in STAGES:
        for t in (3,4,5):
            v=result["stages"][stage]["timesteps"][str(t)]
            rows.append(f"| {stage} | {t} | {v['time']['normalized_l1']:.5f} | {v['time']['normalized_l2']:.5f} | {v['time']['mse']:.6f} | {v['frequency']['noncritical']['relative_log_l1']:.5f} | {v['frequency']['noncritical']['spectral_snr_db']:.2f} | {v['frequency']['critical']['energy_retention']:.4f} | {v['frequency']['critical']['fisher_retention']:.4f} | {v['representation']['cosine_base_augmented']:.5f} | {v['representation']['l2_base_augmented']:.5f} |")
    ratios=[]
    for stage in STAGES:
        v=result["stages"][stage]
        ratios.append(f"| {stage} | {v['ratios']['time_t4_t3']:.4f} | {v['ratios']['time_t5_t3']:.4f} | {v['ratios']['noncritical_frequency_t4_t3']:.4f} | {v['ratios']['noncritical_frequency_t5_t3']:.4f} | {v['ratios']['representation_t4_t3']:.4f} | {v['ratios']['representation_t5_t3']:.4f} | {v['representation_pair']['cosine_t3_t5']:.5f} | {v['representation_pair']['l2_t3_t5']:.5f} |")
    report=f"""# C3 Stage Effect 审计

> **STAGE_EFFECT_AUDIT / STAGE_PERTURBATION_BUDGET_MVP / FIXED_R1_BASELINE / EXPLORATORY_TEP_SUBSET / NOT_FOR_PAPER_FINAL_CLAIMS**

## 结论

状态：`{result['status']}`。允许进入固定 Budget MVP：`{result['budget_mvp_allowed']}`。本审计只使用固定 validation、train-only mask、同一噪声 realization 和冻结 Seed 7 R1 encoder；唯一变量为 `t_noncritical∈{{3,4,5}}`，critical t 始终为 1，phase/DC 保持。

## 分 Stage 时域、频域与表征

| Stage | t | Time L1 | Time L2 | MSE | Noncritical rel-L1 | SNR dB | Critical energy retention | Fisher retention | Repr cosine | Repr L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

完整 per-channel variance change、peak change、critical/noncritical/all 的 relative L1/L2、SNR、energy retention 保存在结果 JSON。

## 效应比与单调性

| Stage | Time 4/3 | Time 5/3 | Freq 4/3 | Freq 5/3 | Repr 4/3 | Repr 5/3 | cos(t3,t5) | L2(t3,t5) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(ratios)}

决策：`{result['decision']}`。规则预先固定为四 stage 比值取中位数，时域/非关键频域/表征三层中至少两层达到 1.10 才认定 timestep effect 明确。若 effect weak，仅允许 beta=1.0/0.6/0.8/1.0 的一次 Budget MVP；否则直接停止 C3。

本结果是已多轮查看 TEP test 背景下的 validation 机制审计，不声称统计显著或论文最终无偏结论。
"""
    Path(path).write_text(report,encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="configs/stage_effect_audit.yaml")
    args=parser.parse_args(); config=yaml.safe_load(Path(args.config).read_text(encoding="utf-8")); result=run(config)
    print(json.dumps({"status":result["status"],"budget_mvp_allowed":result["budget_mvp_allowed"],"ratios":result["decision"]["median_ratios"]},ensure_ascii=False))


if __name__=="__main__": main()
