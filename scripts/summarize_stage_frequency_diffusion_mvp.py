from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scripts.run_stage_frequency_diffusion_mvp import METHODS
from utils import environment_metadata, write_json


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {"mean": float(array.mean()), "std": float(array.std())}


def _method_summary(seed_results: dict[str, Any], method: str) -> dict[str, Any]:
    keys = ("accuracy", "macro_f1", "auprc", "auroc", "fault_recall", "far")
    result = {key: _mean_std([value["results"][method]["metrics"][key] for value in seed_results.values()]) for key in keys}
    result["early_fault_recall"] = _mean_std([value["results"][method]["early_fault"]["recall"] for value in seed_results.values()])
    delays = [value["results"][method]["detection_delay"]["mean_delay_samples"] for value in seed_results.values()]
    detected = [value["results"][method]["detection_delay"]["detection_rate"] for value in seed_results.values()]
    result["detection_delay"] = _mean_std([float(value) for value in delays if value is not None]) if any(value is not None for value in delays) else None
    result["detection_rate"] = _mean_std(detected)
    result["training_seconds"] = _mean_std([value["results"][method]["training_seconds"] for value in seed_results.values()])
    result["peak_gpu_mib"] = _mean_std([value["results"][method]["peak_gpu_mib"] for value in seed_results.values()])
    return result


def three_seed_gate(seed_results: dict[str, Any], summary: dict[str, Any], config: dict[str, Any]) -> tuple[dict[str, bool], bool]:
    c0, c1, c2 = (summary[method] for method in METHODS)
    macro_wins = sum(value["results"][METHODS[2]]["metrics"]["macro_f1"]
                     > value["results"][METHODS[1]]["metrics"]["macro_f1"] for value in seed_results.values())
    far_wins = sum(value["results"][METHODS[2]]["metrics"]["far"]
                   < value["results"][METHODS[1]]["metrics"]["far"] for value in seed_results.values())
    delay_better = (c2["detection_delay"] is not None and c1["detection_delay"] is not None
                    and c2["detection_delay"]["mean"] < c1["detection_delay"]["mean"])
    catastrophic = any(
        value["results"][METHODS[2]]["metrics"]["macro_f1"]
        < value["results"][METHODS[1]]["metrics"]["macro_f1"] - .05
        or value["results"][METHODS[2]]["metrics"]["far"]
        > value["results"][METHODS[1]]["metrics"]["far"] + .10
        for value in seed_results.values())
    checks = {
        "mean_macro_f1_above_c1": c2["macro_f1"]["mean"] > c1["macro_f1"]["mean"],
        "macro_f1_wins_2_of_3": macro_wins >= 2,
        "mean_far_below_c1": c2["far"]["mean"] < c1["far"]["mean"],
        "far_wins_2_of_3": far_wins >= 2,
        "mean_early_or_delay_improved": (c2["early_fault_recall"]["mean"] > c1["early_fault_recall"]["mean"] or delay_better),
        "recall_auprc_maintained": (c2["fault_recall"]["mean"] >= c1["fault_recall"]["mean"] - float(config["mvp_gate"]["maximum_recall_drop"])
                                     and c2["auprc"]["mean"] >= c1["auprc"]["mean"] - float(config["mvp_gate"]["maximum_auprc_drop"])),
        "c2_at_least_close_to_c0": (c2["macro_f1"]["mean"] >= c0["macro_f1"]["mean"] - .005
                                     and c2["auprc"]["mean"] >= c0["auprc"]["mean"] - .005),
        "no_catastrophic_reverse_seed": not catastrophic,
    }
    return checks, all(checks.values())


def summarize(config: dict[str, Any], seed_results: dict[str, Any], selected_t: int) -> dict[str, Any]:
    path = Path(config["output_dir"]) / "summary.json"
    if path.exists(): return json.loads(path.read_text(encoding="utf-8"))
    method_summary = {method: _method_summary(seed_results, method) for method in METHODS}
    deltas = {}
    for seed, value in seed_results.items():
        deltas[seed] = {}
        for comparison, reference in (("C2-C1", METHODS[1]), ("C2-C0", METHODS[0])):
            deltas[seed][comparison] = {
                key: float(value["results"][METHODS[2]]["metrics"][key] - value["results"][reference]["metrics"][key])
                for key in ("macro_f1", "auprc", "fault_recall", "far")}
            deltas[seed][comparison]["early_fault_recall"] = float(
                value["results"][METHODS[2]]["early_fault"]["recall"]
                - value["results"][reference]["early_fault"]["recall"])
    first = seed_results[str(config["training"]["selection_seed"])]
    if not first["three_seed_allowed"]:
        status, checks = "FREQUENCY_SELECTIVE_DIFFUSION_MVP_NO_GO", {"three_seed_skipped_by_single_seed_gate": True}
    else:
        if len(seed_results) != 3: raise RuntimeError("single-seed GO requires complete 3-seed audit")
        checks, passed = three_seed_gate(seed_results, method_summary, config)
        status = "FREQUENCY_SELECTIVE_DIFFUSION_3SEED_GO" if passed else "FREQUENCY_SELECTIVE_DIFFUSION_3SEED_NO_GO"
    result = {"markers": config["markers"], "status": status, "selected_t_noncritical": int(selected_t),
              "seed_results": seed_results, "method_summary": method_summary, "per_seed_deltas": deltas,
              "gate_checks": checks, "three_seed_ran": len(seed_results) == 3,
              "next_step": "加入故障阶段扩散课程 C3" if status == "FREQUENCY_SELECTIVE_DIFFUSION_3SEED_GO" else "停止当前主线",
              **environment_metadata()}
    write_json(path, result); return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/stage_frequency_diffusion_mvp.yaml")
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output = Path(config["output_dir"]); seed_results = {}
    for seed in config["training"]["seeds"]:
        path = output / f"seed_{seed}" / "result.json"
        if path.exists(): seed_results[str(seed)] = json.loads(path.read_text(encoding="utf-8"))
    if not seed_results: raise FileNotFoundError("no frequency MVP seed results")
    first = next(iter(seed_results.values())); result = summarize(config, seed_results, int(first["selected_t_noncritical"]))
    print(json.dumps({"status": result["status"], "seeds": list(seed_results)}, ensure_ascii=False))


if __name__ == "__main__": main()

