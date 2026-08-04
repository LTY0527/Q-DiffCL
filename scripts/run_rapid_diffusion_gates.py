from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from models import MinimalConditionalDiffusion1D
from scripts.common import prepare_real
from scripts.run_rapid_idea_validation import (_evaluate, _fit_ce, _fit_supcon,
                                                _kept_ids, _loader, _masked_mae,
                                                _quality, _view_bundle)
from trainers import build_model
from utils import environment_metadata, seed_everything, write_json


MARKERS = ["RAPID_IDEA_VALIDATION", "SINGLE_SEED", "SUBSET_DATA", "NOT_FOR_PAPER_CLAIMS"]


def _schedule(steps: int, device: str) -> torch.Tensor:
    betas = torch.linspace(1e-4, 0.02, steps, device=device)
    return torch.cumprod(1.0 - betas, dim=0)


def _diffusion_loss(model: torch.nn.Module, clean: np.ndarray, degraded: np.ndarray,
                    observation: np.ndarray, config: dict[str, Any], device: str,
                    optimizer: torch.optim.Optimizer | None, seed: int | None = None) -> float:
    training = optimizer is not None; model.train(training); losses = []
    steps = int(config["diffusion"]["steps"]); alpha_bars = _schedule(steps, device)
    generator = None if seed is None else torch.Generator(device=device).manual_seed(seed)
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for clean_b, degraded_b, observation_b in _loader(
            clean.astype(np.float32), degraded.astype(np.float32), observation.astype(bool),
            batch_size=int(config["batch_size"]), shuffle=training,
        ):
            clean_b, degraded_b, observation_b = clean_b.to(device), degraded_b.to(device), observation_b.to(device)
            t = torch.randint(0, steps, (len(clean_b),), device=device, generator=generator)
            noise = torch.randn(clean_b.shape, device=device, generator=generator)
            alpha = alpha_bars[t][:, None, None]
            noisy = alpha.sqrt() * clean_b + (1 - alpha).sqrt() * noise
            noisy = torch.where(observation_b, degraded_b, noisy)
            if optimizer is not None: optimizer.zero_grad()
            predicted = model(noisy, degraded_b, observation_b, t)
            missing = ~observation_b
            loss = ((predicted - noise).square() * missing).sum() / missing.sum().clamp_min(1)
            if optimizer is not None: loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
    return float(np.mean(losses))


def _fit_diffusion(model: torch.nn.Module, train: dict[str, Any], validation: dict[str, Any],
                   config: dict[str, Any], device: str) -> tuple[list[dict[str, float]], float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["diffusion"]["learning_rate"]))
    best_state, best_loss, stale, history = None, float("inf"), 0, []
    started = time.perf_counter()
    for epoch in range(int(config["diffusion"]["epochs"])):
        train_loss = _diffusion_loss(model, train["clean"], train["degraded"], train["observation"], config, device, optimizer)
        val_loss = _diffusion_loss(model, validation["clean"], validation["degraded"], validation["observation"], config, device, None, 10_000 + epoch)
        history.append({"epoch": epoch, "train_masked_noise_mse": train_loss, "validation_masked_noise_mse": val_loss})
        if val_loss < best_loss - 1e-6:
            best_loss, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= int(config["diffusion"]["early_stopping_patience"]): break
    if best_state is not None: model.load_state_dict(best_state)
    return history, time.perf_counter() - started


@torch.no_grad()
def _restore(model: torch.nn.Module, degraded: np.ndarray, observation: np.ndarray,
             config: dict[str, Any], device: str, seed: int) -> np.ndarray:
    model.eval(); steps = int(config["diffusion"]["steps"]); alpha_bars = _schedule(steps, device)
    generator = torch.Generator(device=device).manual_seed(seed); restored = []
    for degraded_b, observation_b in _loader(degraded.astype(np.float32), observation.astype(bool), batch_size=int(config["batch_size"]), shuffle=False):
        degraded_b, observation_b = degraded_b.to(device), observation_b.to(device)
        x = torch.randn(degraded_b.shape, device=device, generator=generator)
        x = torch.where(observation_b, degraded_b, x)
        for step in reversed(range(steps)):
            t = torch.full((len(x),), step, device=device, dtype=torch.long)
            predicted_noise = model(x, degraded_b, observation_b, t)
            alpha = alpha_bars[step]
            predicted_clean = (x - (1 - alpha).sqrt() * predicted_noise) / alpha.sqrt().clamp_min(1e-8)
            if step > 0:
                previous_alpha = alpha_bars[step - 1]
                x = previous_alpha.sqrt() * predicted_clean + (1 - previous_alpha).sqrt() * predicted_noise
            else:
                x = predicted_clean
            x = torch.where(observation_b, degraded_b, x)
        restored.append(x.cpu().numpy())
    return np.concatenate(restored)


def _masked_rmse(clean: np.ndarray, restored: np.ndarray, observation: np.ndarray) -> float:
    missing = ~observation
    return float(np.sqrt(np.mean((clean[missing] - restored[missing]) ** 2)))


def _correlation_error(clean: np.ndarray, restored: np.ndarray) -> float:
    clean_flat = clean.transpose(1, 0, 2).reshape(clean.shape[1], -1)
    restored_flat = restored.transpose(1, 0, 2).reshape(restored.shape[1], -1)
    clean_corr = np.nan_to_num(np.corrcoef(clean_flat)); restored_corr = np.nan_to_num(np.corrcoef(restored_flat))
    return float(np.linalg.norm(clean_corr - restored_corr, ord="fro") / clean.shape[1])


def _classifier_view(teacher: torch.nn.Module, clean: np.ndarray, view: np.ndarray, labels: np.ndarray,
                     config: dict[str, Any], device: str) -> dict[str, Any]:
    clean_metrics, clean_prediction, _, _ = _evaluate(teacher, clean, labels, int(config["batch_size"]), device)
    metrics, prediction, _, _ = _evaluate(teacher, view, labels, int(config["batch_size"]), device)
    return {"metrics": metrics, "teacher_prediction_consistency": float(np.mean(clean_prediction == prediction)),
            "fault_recall_retention": None if clean_metrics["fault_recall"] == 0 else metrics["fault_recall"] / clean_metrics["fault_recall"]}


def main() -> None:
    raise RuntimeError(
        "该历史入口使用已审计为不一致的旧扩散采样器，已禁用。"
        "请使用 scripts.train_diffusion_recovery 和 scripts.sample_diffusion_recovery。"
    )
    config = yaml.safe_load(Path("configs/rapid_idea_validation.yaml").read_text(encoding="utf-8"))
    output = Path(config["output_dir"]); gate1 = json.loads((output / "gate1_results.json").read_text(encoding="utf-8"))
    if gate1["gate_one"] != "GO": raise RuntimeError("Gate 1 is not GO; diffusion execution is forbidden")
    seed_everything(int(config["random_seed"])); device = "cuda"; total_started = time.perf_counter()
    clean_data, manifest, window_stats = prepare_real(config, degrade=False); bundles = {}
    for split in ("train", "validation", "test"):
        clean, labels = clean_data[split]; ids = _kept_ids(window_stats[split])
        degraded, simple, observation, degraded_q, degraded_mae, simple_mae = _view_bundle(clean, ids, config)
        bundles[split] = {"clean": clean, "labels": labels, "degraded": degraded, "simple": simple,
                          "observation": observation, "degraded_mae": degraded_mae, "simple_mae": simple_mae}
    channels = bundles["train"]["clean"].shape[1]
    diffusion = MinimalConditionalDiffusion1D(channels, int(config["diffusion"]["hidden_channels"]),
                                               int(config["diffusion"]["hidden_channels"]), int(config["diffusion"]["residual_blocks"])).to(device)
    torch.cuda.reset_peak_memory_stats(); history, training_seconds = _fit_diffusion(diffusion, bundles["train"], bundles["validation"], config, device)
    peak_gpu_mib = torch.cuda.max_memory_allocated() / 1024 ** 2
    torch.save(diffusion.state_dict(), output / "minimal_diffusion.pt")
    for index, split in enumerate(("train", "validation", "test")):
        bundles[split]["diffusion"] = _restore(diffusion, bundles[split]["degraded"], bundles[split]["observation"], config, device, int(config["random_seed"]) + 100 + index)
    teacher = build_model(config["model"], channels, 2).to(device)
    teacher.load_state_dict(torch.load(output / "G1_0.pt", map_location=device, weights_only=True)); teacher.eval()
    view_results = {}
    test = bundles["test"]
    for view in ("degraded", "simple", "diffusion"):
        result = _classifier_view(teacher, test["clean"], test[view], test["labels"], config, device)
        result.update({"masked_mae": _masked_mae(test["clean"], test[view], test["observation"])[0],
                       "masked_rmse": _masked_rmse(test["clean"], test[view], test["observation"]),
                       "correlation_matrix_error": _correlation_error(test["clean"], test[view])})
        view_results[view] = result
    diffusion_better_error = view_results["diffusion"]["masked_mae"] <= view_results["simple"]["masked_mae"] * 1.05
    diffusion_better_downstream = (view_results["diffusion"]["metrics"]["auprc"] > max(view_results["degraded"]["metrics"]["auprc"], view_results["simple"]["metrics"]["auprc"])
                                   and view_results["diffusion"]["metrics"]["fault_recall"] >= max(view_results["degraded"]["metrics"]["fault_recall"], view_results["simple"]["metrics"]["fault_recall"]) - 0.01)
    gate_two = "GO" if diffusion_better_error and diffusion_better_downstream else "NO-GO"
    result: dict[str, Any] = {"markers": MARKERS, **environment_metadata(), "gate_one": "GO", "gate_two": gate_two,
                              "diffusion_training_seconds": training_seconds, "diffusion_peak_gpu_mib": peak_gpu_mib,
                              "diffusion_history": history, "view_results": view_results, "gate_three": "NOT_RUN",
                              "total_seconds": time.perf_counter() - total_started}
    if gate_two == "GO":
        for split in bundles:
            _, per_sample = _masked_mae(bundles[split]["clean"], bundles[split]["diffusion"], bundles[split]["observation"])
            bundles[split]["diffusion_q"] = _quality(per_sample)
        g3 = {}
        for name, weighted in (("G3-0 Diffusion + Hard SupCon", False), ("G3-1 Diffusion + Oracle Quality SupCon", True)):
            seed_everything(int(config["random_seed"])); started = time.perf_counter()
            model = build_model(config["model"], channels, 2).to(device)
            train_q = bundles["train"]["diffusion_q"] if weighted else np.ones(len(bundles["train"]["labels"]), np.float32)
            val_q = bundles["validation"]["diffusion_q"] if weighted else np.ones(len(bundles["validation"]["labels"]), np.float32)
            pretrain = _fit_supcon(model, bundles["train"]["clean"], bundles["train"]["diffusion"], bundles["train"]["labels"], train_q,
                                   bundles["validation"]["clean"], bundles["validation"]["diffusion"], bundles["validation"]["labels"], val_q, config, device)
            for parameter in model.encoder.parameters(): parameter.requires_grad = False
            probe = _fit_ce(model, bundles["train"]["clean"], bundles["train"]["labels"], bundles["validation"]["diffusion"], bundles["validation"]["labels"], config, device, True)
            metrics, _, _, _ = _evaluate(model, bundles["test"]["diffusion"], bundles["test"]["labels"], int(config["batch_size"]), device)
            g3[name] = {"metrics": metrics, "seconds": time.perf_counter() - started, "pretrain_history": pretrain, "probe_history": probe}
            torch.save(model.state_dict(), output / ("G3_1.pt" if weighted else "G3_0.pt"))
        base, weighted = g3["G3-0 Diffusion + Hard SupCon"]["metrics"], g3["G3-1 Diffusion + Oracle Quality SupCon"]["metrics"]
        gain = max(weighted["macro_f1"] - base["macro_f1"], weighted["auprc"] - base["auprc"])
        recall_safe = weighted["fault_recall"] >= base["fault_recall"] - 0.01
        result.update({"gate_three": "GO" if gain >= 0.015 and recall_safe else "NO-GO", "g3_results": g3, "g3_gain_signal": gain})
    write_json(output / "gate2_gate3_results.json", result)
    print(json.dumps({"gate_two": result["gate_two"], "gate_three": result["gate_three"], "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
