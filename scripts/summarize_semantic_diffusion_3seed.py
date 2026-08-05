from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from utils import environment_metadata, write_json


METHODS = ("B0", "B1", "B2")
METRICS = ("macro_f1", "auprc", "fault_recall", "far", "auroc")
PAIRS = (("B2", "B1"), ("B2", "B0"), ("B1", "B0"))


def mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) != 3 or not np.isfinite(array).all():
        raise ValueError("exactly three finite values are required")
    return {"mean": float(array.mean()), "std": float(array.std(ddof=1))}


def paired_deltas(results: dict[str, dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for left, right in PAIRS:
        name = f"{left}-{right}"
        output[name] = {}
        for seed in seeds:
            first, second = results[f"{seed}:{left}"]["metrics"], results[f"{seed}:{right}"]["metrics"]
            output[name][str(seed)] = {metric: float(first[metric] - second[metric]) for metric in METRICS}
    return output


def stability_status(summary: dict[str, dict[str, dict[str, float]]], deltas: dict[str, Any],
                     seeds: list[int], gate: dict[str, float]) -> tuple[str, dict[str, bool], dict[str, int]]:
    b21 = deltas["B2-B1"]
    counts = {
        "macro_f1_above_b1": sum(b21[str(seed)]["macro_f1"] > 0 for seed in seeds),
        "far_below_b1": sum(b21[str(seed)]["far"] < 0 for seed in seeds),
        "auprc_not_below_b1": sum(b21[str(seed)]["auprc"] >= 0 for seed in seeds),
        "recall_drop_within_1_point": sum(b21[str(seed)]["fault_recall"] >= -float(gate["maximum_mean_recall_drop"]) for seed in seeds),
    }
    catastrophic = any(
        b21[str(seed)]["far"] > float(gate["catastrophic_far_increase"])
        or b21[str(seed)]["fault_recall"] < -float(gate["catastrophic_recall_drop"])
        for seed in seeds
    )
    checks = {
        "mean_macro_f1_above_b1": summary["B2"]["macro_f1"]["mean"] > summary["B1"]["macro_f1"]["mean"],
        "macro_f1_wins_at_least_two": counts["macro_f1_above_b1"] >= 2,
        "mean_far_below_b1": summary["B2"]["far"]["mean"] < summary["B1"]["far"]["mean"],
        "far_wins_at_least_two": counts["far_below_b1"] >= 2,
        "mean_recall_drop_within_limit": summary["B2"]["fault_recall"]["mean"] >= summary["B1"]["fault_recall"]["mean"] - float(gate["maximum_mean_recall_drop"]),
        "mean_auprc_drop_within_limit": summary["B2"]["auprc"]["mean"] >= summary["B1"]["auprc"]["mean"] - float(gate["maximum_mean_auprc_drop"]),
        "no_catastrophic_reverse_seed": not catastrophic,
        "b2_at_least_b0_overall": summary["B2"]["macro_f1"]["mean"] >= summary["B0"]["macro_f1"]["mean"] and summary["B2"]["auprc"]["mean"] >= summary["B0"]["auprc"]["mean"] - float(gate["maximum_mean_auprc_drop"]),
    }
    if all(checks.values()):
        status = "SEMANTIC_DIFFUSION_3SEED_GO"
    elif counts["macro_f1_above_b1"] <= 1 and counts["far_below_b1"] <= 1 and not checks["mean_macro_f1_above_b1"] and not checks["mean_far_below_b1"]:
        status = "SEMANTIC_DIFFUSION_3SEED_NO_GO"
    else:
        status = "SEMANTIC_DIFFUSION_3SEED_UNSTABLE"
    return status, checks, counts


def summarize(config_path: Path, config: dict[str, Any], results: dict[str, dict[str, Any]],
              fingerprints: dict[str, str]) -> dict[str, Any]:
    seeds = list(map(int, config["seeds"]))
    expected = {f"{seed}:{method}" for seed in seeds for method in METHODS}
    if set(results) != expected:
        raise ValueError(f"incomplete result grid: missing={sorted(expected - set(results))}")
    method_summary = {
        method: {metric: mean_std([results[f"{seed}:{method}"]["metrics"][metric] for seed in seeds])
                 for metric in METRICS}
        for method in METHODS
    }
    deltas = paired_deltas(results, seeds)
    status, checks, counts = stability_status(method_summary, deltas, seeds, config["stability_gate"])
    summary = {
        "markers": config["markers"], "status": status, "seeds": seeds,
        "fingerprints": fingerprints, "method_summary": method_summary,
        "paired_deltas": deltas, "gate_checks": checks, "direction_counts": counts,
        "seed_results": results, **environment_metadata(),
    }
    output = Path(config["output_dir"])
    result_path = output / "summary.json"
    if result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        comparable = {key: value for key, value in existing.items() if key not in ("git_commit",)}
        current = {key: value for key, value in summary.items() if key not in ("git_commit",)}
        if comparable != current:
            raise FileExistsError(f"refusing to overwrite a different summary: {result_path}")
    else:
        write_json(result_path, summary)
    csv_path = output / "summary.csv"
    rows = [{"method": method, **{f"{metric}_{part}": method_summary[method][metric][part]
                                  for metric in METRICS for part in ("mean", "std")}}
            for method in METHODS]
    columns = list(rows[0])
    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader(); writer.writerows(rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/semantic_diffusion_3seed.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = Path(config["output_dir"]); results = {}
    for seed in config["seeds"]:
        for method in METHODS:
            path = output / f"seed_{seed}" / method / "result.json"
            results[f"{seed}:{method}"] = json.loads(path.read_text(encoding="utf-8"))
    from scripts.run_semantic_diffusion_3seed import frozen_fingerprints
    value = summarize(config_path, config, results, frozen_fingerprints(config_path, config))
    print(json.dumps({"status": value["status"], "counts": value["direction_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
