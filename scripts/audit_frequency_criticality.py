from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

from frequency import (build_criticality, fault_stages, fit_frequency_scaler,
                       log_amplitude_phase, mask_jaccard)
from frequency.criticality import STAGES, fault_type
from scripts.audit_semantic_diffusion_augmentation import bases
from scripts.run_diffusion_quality_retest import load_fixed_views
from utils import environment_metadata, write_json


def _distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "std": float(values.std()),
            "minimum": float(values.min()), "median": float(np.median(values)),
            "maximum": float(values.max())}


def _array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _stage_summary(bundle: dict[str, np.ndarray], stages: np.ndarray) -> dict[str, Any]:
    return {stage: {"samples": int(np.sum(stages == stage)),
                    "runs": int(len(np.unique(bundle["run_uid"][stages == stage])))}
            for stage in STAGES}


def _stage_spectra(features: np.ndarray, stages: np.ndarray) -> dict[str, list[float]]:
    return {stage: features[stages == stage].mean(axis=(0, 1)).tolist()
            for stage in STAGES if np.any(stages == stage)}


def _fault_focus(features: np.ndarray, bundle: dict[str, np.ndarray], kinds=(3, 9, 15)) -> dict[str, Any]:
    run_types = np.asarray([fault_type(value) for value in bundle["run_uid"]])
    normal = features[np.asarray(bundle["labels"]) == 0].mean(0)
    result = {}
    for kind in kinds:
        selector = run_types == kind
        if not selector.any():
            result[str(kind)] = {"count": 0, "top_channel_frequency": []}; continue
        difference = np.abs(features[selector].mean(0) - normal)
        flat = np.argsort(difference.reshape(-1))[-10:][::-1]
        result[str(kind)] = {"count": int(selector.sum()), "top_channel_frequency": [
            {"channel": int(index // difference.shape[1]), "frequency_bin": int(index % difference.shape[1]),
             "absolute_standardized_difference": float(difference.reshape(-1)[index])} for index in flat]}
    return result


def _validation_direction(train_direction: np.ndarray, validation_features: np.ndarray,
                          validation_bundle: dict[str, np.ndarray], mask: np.ndarray) -> float:
    labels = np.asarray(validation_bundle["labels"])
    difference = validation_features[labels != 0].mean(0) - validation_features[labels == 0].mean(0)
    meaningful = mask & (np.abs(train_direction) > np.median(np.abs(train_direction[mask])))
    if not meaningful.any(): meaningful = mask
    return float(np.mean(np.sign(train_direction[meaningful]) == np.sign(difference[meaningful])))


def _top_frequencies(composite: np.ndarray, count: int = 5) -> dict[str, list[dict[str, float]]]:
    result = {}
    for channel, row in enumerate(composite):
        indices = np.argsort(row)[-count:][::-1]
        result[str(channel)] = [{"frequency_bin": int(index), "score": float(row[index])} for index in indices]
    return result


def _plot_audit(result: dict[str, Any], critical: dict[str, Any], figure_dir: Path) -> list[str]:
    if figure_dir.exists() and any(figure_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite frequency figures: {figure_dir}")
    figure_dir.mkdir(parents=True, exist_ok=True); paths = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for axis, (name, values) in zip(axes, (("Energy Top-K", critical["masks"]["energy"]),
                                                   ("Fisher Top-K", critical["masks"]["fisher"]),
                                                   ("Composite soft mask", critical["soft_mask"]))):
        image = axis.imshow(values, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0, vmax=1)
        axis.set_title(name); axis.set_xlabel("rFFT bin"); axis.set_ylabel("channel")
    fig.colorbar(image, ax=axes, shrink=.8); path = figure_dir / "mask_comparison.png"
    fig.savefig(path, dpi=140); plt.close(fig); paths.append(path.as_posix())

    spectra = result["stage_spectra"]["train"]
    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for stage, values in spectra.items(): axis.plot(values, label=stage)
    axis.set_xlabel("rFFT bin"); axis.set_ylabel("mean standardized log amplitude")
    axis.set_title("Train stage frequency profiles"); axis.legend(); path = figure_dir / "stage_frequency_profiles.png"
    fig.savefig(path, dpi=140); plt.close(fig); paths.append(path.as_posix())

    fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    axis.hist(result["bootstrap_overlap_values"], bins=12, color="#356A8A")
    axis.axvline(result["bootstrap_overlap"]["mean"], color="#B84A39", linestyle="--", label="mean")
    axis.set_xlabel("Jaccard overlap with full-train composite mask"); axis.set_ylabel("count")
    axis.set_title("Train-run bootstrap stability"); axis.legend(); path = figure_dir / "bootstrap_overlap.png"
    fig.savefig(path, dpi=140); plt.close(fig); paths.append(path.as_posix())
    return paths


def render_report(result: dict[str, Any], path: str) -> None:
    split_rows = [f"| {split} | {value['samples']} | {value['runs']} |"
                  for split, value in result["split_counts"].items()]
    stage_rows = []
    for split, summary in result["stage_distribution"].items():
        stage_rows.append(f"| {split} | " + " | ".join(str(summary[name]["samples"]) for name in STAGES) + " |")
    checks = [f"- {'通过' if passed else '**失败**'}：`{name}`" for name, passed in result["gate_checks"].items()]
    top_rows = []
    for channel, values in result["top_frequencies_by_channel"].items():
        top_rows.append(f"| {channel} | " + ", ".join(str(value["frequency_bin"]) for value in values) + " |")
    focus_rows = []
    for split, kinds in result["fault_focus"].items():
        for kind, value in kinds.items():
            pairs = ", ".join(f"c{x['channel']}/f{x['frequency_bin']}" for x in value["top_channel_frequency"][:5])
            focus_rows.append(f"| {split} | {kind} | {value['count']} | {pairs} |")
    report = f"""# 故障阶段与关键频率审计

> **STAGE_FREQUENCY_DIFFUSION_MVP / FORWARD_DIFFUSION_ONLY / FIXED_TEP_SUBSET / NOT_FOR_PAPER_CLAIMS**

## 结论

状态：`{result['status']}`。频率统计、标准化、D/E/S、三种 mask 与 bootstrap 均只由 train split 拟合；validation/test 仅用于外部方向验证和报告。

旧质量评价、候选排序和教师语义路线已连续得到 NO-GO，本主线不再训练反向恢复器或教师约束，而是检验工业故障是否存在跨 Run 可重复、早期敏感且可选择性保护的频带。

## 数据与阶段协议

| Split | 窗口数 | Run 数 |
|---|---:|---:|
{chr(10).join(split_rows)}

| Split | prefault | early | middle | stable |
|---|---:|---:|---:|---:|
{chr(10).join(stage_rows)}

training/testing 的真实 fault onset 分别为 21/161。保存原始 `delta=end_sample-onset`；为适配长度 64 的完整非过渡窗口，阶段进度定义为 `delta-(window_length-1)`。进度 `[0,4*stride)` 为 early、`[4*stride,12*stride)` 为 middle，其后为 stable。这样 training 与 testing 均严格覆盖 onset 后首 4 个完整故障窗口，不使用固定绝对 `start_sample`。

## 频率表示与关键性

`x_base [N,52,64]` 经 `torch.fft.rfft` 得到 33 个 bin，使用 `log1p(abs(X))` 并保留 phase。频谱 scaler 只在 train 拟合。

- D：先按 Run 聚合，再计算 normal/fault 类间—类内 Fisher 比。
- E：按 Run 聚合的 early fault 与 train normal Fisher 比。
- S：fault Run 相对 normal Run 中位参考的方向一致性，并以稳健变异系数惩罚。
- Composite：train-only median/IQR robust normalization 后按 `0.5D + 0.3E + 0.2S` 组合。

三种相同比例（30%）mask 的重合：energy/composite Jaccard={result['mask_comparison']['energy_composite_jaccard']:.4f}，fisher/composite Jaccard={result['mask_comparison']['fisher_composite_jaccard']:.4f}。Composite bootstrap overlap 为 {result['bootstrap_overlap']['mean']:.4f}±{result['bootstrap_overlap']['std']:.4f}。

![三种 mask 对比](assets/stage_frequency_diffusion_mvp/mask_comparison.png)

![阶段频谱](assets/stage_frequency_diffusion_mvp/stage_frequency_profiles.png)

![Bootstrap 稳定性](assets/stage_frequency_diffusion_mvp/bootstrap_overlap.png)

## 每通道 Top-5 Composite 频率

| Channel（0-based） | rFFT bins |
|---:|---|
{chr(10).join(top_rows)}

## Fault 3/9/15 重点频率

| Split | Fault | 窗口数 | Top channel/bin |
|---|---:|---:|---|
{chr(10).join(focus_rows)}

这些 validation/test 结果只验证 train 拟合方向，没有反向修改 mask。

## Gate

- 关键/非关键 Fisher 均值比：{result['mechanism_metrics']['critical_noncritical_fisher_ratio']:.4f}
- 关键频带 E / 随机同规模频带 E：{result['mechanism_metrics']['critical_random_early_ratio']:.4f}
- validation 关键频率方向一致率：{result['mechanism_metrics']['validation_direction_agreement']:.4f}
- 选中频率中 bin>2 的比例：{result['mechanism_metrics']['nonlow_frequency_fraction']:.4f}

{chr(10).join(checks)}

通过条件采用 6 项中的至少 {result['gate_minimum_passes']} 项。若状态为 `FREQUENCY_CRITICALITY_AUDIT_NO_GO`，则停止 C0/C1/C2；若为 `FREQUENCY_CRITICALITY_AUDIT_GO`，才允许进入前向频谱扩散 MVP。

> 本报告仅为小型固定 TEP 子集工程审计：**NOT_FOR_PAPER_CLAIMS**。
"""
    Path(path).write_text(report, encoding="utf-8")


def audit(config: dict[str, Any]) -> dict[str, Any]:
    result_path = Path(config["frequency_audit_result"])
    if result_path.exists(): return json.loads(result_path.read_text(encoding="utf-8"))
    views, manifest = load_fixed_views(config); base = bases(views)
    stages = {split: fault_stages(views[split], config) for split in ("train", "validation", "test")}
    representations = {split: log_amplitude_phase(base[split])[0] for split in ("train", "validation", "test")}
    scaler = fit_frequency_scaler(representations["train"], "train")
    standardized = {split: scaler.transform(value) for split, value in representations.items()}
    critical = build_criticality(standardized["train"], views["train"], stages["train"],
                                 config["criticality"], representations["train"])
    mask = critical["masks"]["composite"]; settings = config["criticality"]
    rng = np.random.default_rng(int(settings["bootstrap_seed"]) + 1)
    random_early = []
    for _ in range(64):
        random_mask = np.zeros(mask.size, dtype=bool)
        random_mask[rng.choice(mask.size, int(mask.sum()), replace=False)] = True
        random_early.append(float(critical["early"].reshape(-1)[random_mask].mean()))
    fisher_ratio = float(critical["discriminative"][mask].mean()
                         / max(float(critical["discriminative"][~mask].mean()), 1e-12))
    early_ratio = float(critical["early"][mask].mean() / max(float(np.mean(random_early)), 1e-12))
    energy_overlap = mask_jaccard(critical["masks"]["energy"], mask)
    direction = _validation_direction(critical["train_direction"], standardized["validation"], views["validation"], mask)
    selected_frequency = np.indices(mask.shape)[1][mask]
    nonlow = float(np.mean(selected_frequency > 2))
    mechanism = {"critical_noncritical_fisher_ratio": fisher_ratio,
                 "critical_random_early_ratio": early_ratio,
                 "validation_direction_agreement": direction,
                 "nonlow_frequency_fraction": nonlow}
    checks = {
        "train_run_bootstrap_reproducible": float(critical["bootstrap_overlap"].mean()) >= float(settings["minimum_bootstrap_overlap"]),
        "critical_fisher_above_noncritical": fisher_ratio >= float(settings["minimum_fisher_ratio"]),
        "critical_early_above_random": early_ratio >= float(settings["minimum_early_ratio"]),
        "energy_not_equivalent_to_composite": energy_overlap <= float(settings["maximum_energy_composite_overlap"]),
        "validation_direction_preserved": direction >= float(settings["minimum_validation_direction_agreement"]),
        "mask_not_only_dc_or_lowest_bins": nonlow >= float(settings["minimum_nonlow_frequency_fraction"]),
    }
    passed = sum(checks.values()) >= int(settings["gate_minimum_passes"])
    result = {
        "markers": config["markers"],
        "status": "FREQUENCY_CRITICALITY_AUDIT_GO" if passed else "FREQUENCY_CRITICALITY_AUDIT_NO_GO",
        "fit_split": "train", "validation_test_used_for_fit": False,
        "fixed_view_manifest": config["fixed_views"]["manifest"], "fixed_view_manifest_metadata": manifest,
        "split_counts": {split: {"samples": int(len(views[split]["labels"])),
                                  "runs": int(len(np.unique(views[split]["run_uid"])))} for split in views},
        "stage_distribution": {split: _stage_summary(views[split], stages[split]) for split in stages},
        "fft_shape": {split: list(representations[split].shape) for split in representations},
        "frequency_bins": int(representations["train"].shape[-1]),
        "scaler": {"fit_split": scaler.fit_split, "mean_sha256": _array_hash(scaler.mean),
                   "scale_sha256": _array_hash(scaler.scale)},
        "criticality": {key: critical[key].tolist() for key in
                         ("discriminative", "early", "stability", "composite", "soft_mask", "energy", "multiclass_fisher")},
        "hard_masks": {key: value.astype(int).tolist() for key, value in critical["masks"].items()},
        "mask_sha256": {key: _array_hash(value) for key, value in critical["masks"].items()},
        "mask_comparison": {"energy_composite_jaccard": energy_overlap,
                            "fisher_composite_jaccard": mask_jaccard(critical["masks"]["fisher"], mask)},
        "bootstrap_overlap": _distribution(critical["bootstrap_overlap"]),
        "bootstrap_overlap_values": critical["bootstrap_overlap"].tolist(),
        "criticality_run_counts": critical["run_counts"], "mechanism_metrics": mechanism,
        "stage_spectra": {split: _stage_spectra(standardized[split], stages[split]) for split in stages},
        "top_frequencies_by_channel": _top_frequencies(critical["composite"]),
        "fault_focus": {split: _fault_focus(standardized[split], views[split]) for split in views},
        "gate_checks": checks, "gate_minimum_passes": int(settings["gate_minimum_passes"]),
        "training_allowed": passed, **environment_metadata(),
    }
    result_path.parent.mkdir(parents=True, exist_ok=False)
    result["figures"] = _plot_audit(result, critical, Path(config["frequency_figure_dir"]))
    write_json(result_path, result); render_report(result, config["frequency_audit_report"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/stage_frequency_diffusion_mvp.yaml")
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = audit(config)
    print(json.dumps({"status": result["status"], "checks": result["gate_checks"],
                      "training_allowed": result["training_allowed"]}, ensure_ascii=False))


if __name__ == "__main__": main()
