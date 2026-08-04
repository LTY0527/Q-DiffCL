from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy.stats import kendalltau, spearmanr

from diffusion.fixed_views import per_sample_masked_mae, sha256_file
from quality import center_scores, combined_scores, semantic_scores
from scripts.run_diffusion_quality_retest import load_fixed_views
from scripts.run_rapid_idea_validation import _simple_interpolate
from trainers import build_model
from utils import deterministic_seed, environment_metadata, write_json


def oracle_candidate_errors(clean: np.ndarray, candidates: np.ndarray,
                            observation: np.ndarray) -> dict[str, np.ndarray]:
    """Oracle-only metrics. Never called by no-reference scoring functions."""
    clean = np.asarray(clean)[:, None]; candidates = np.asarray(candidates); missing = (~np.asarray(observation, dtype=bool))[:, None]
    count = missing.sum(axis=(2, 3)); difference = candidates - clean
    mae = (np.abs(difference) * missing).sum(axis=(2, 3)) / count
    rmse = np.sqrt(((difference ** 2) * missing).sum(axis=(2, 3)) / count)
    first_difference = np.abs(np.diff(candidates, axis=-1) - np.diff(clean, axis=-1)).mean(axis=(2, 3))
    correlation = np.empty(mae.shape, dtype=np.float64)
    for sample in range(len(candidates)):
        clean_corr = np.nan_to_num(np.corrcoef(clean[sample, 0]))
        for candidate in range(candidates.shape[1]):
            restored_corr = np.nan_to_num(np.corrcoef(candidates[sample, candidate]))
            correlation[sample, candidate] = np.linalg.norm(clean_corr - restored_corr, ord="fro") / clean.shape[2]
    return {"masked_mae": mae, "masked_rmse": rmse, "first_difference_error": first_difference,
            "correlation_error": correlation}


@torch.no_grad()
def teacher_probabilities(model: torch.nn.Module, values: np.ndarray, batch_size: int,
                          device: str) -> np.ndarray:
    original_shape = values.shape[:-2]; flat = values.reshape(-1, *values.shape[-2:]); outputs = []
    for start in range(0, len(flat), batch_size):
        logits = model(torch.from_numpy(flat[start:start + batch_size]).float().to(device))["logits"]
        outputs.append(torch.softmax(logits, 1).cpu().numpy())
    return np.concatenate(outputs).reshape(*original_shape, -1)


def _rank_value(function, score: np.ndarray, utility: np.ndarray) -> float:
    value = function(score, utility).statistic
    return 0.0 if not np.isfinite(value) else float(value)


def ranking_metrics(scores: np.ndarray, mae: np.ndarray, semantic_distance: np.ndarray) -> dict[str, float]:
    oracle_utility = -mae; oracle = mae.argmin(1); selected = scores.argmax(1); k = scores.shape[1]
    spearman = [_rank_value(spearmanr, scores[index], oracle_utility[index]) for index in range(len(scores))]
    kendall = [_rank_value(kendalltau, scores[index], oracle_utility[index]) for index in range(len(scores))]
    top_two = np.argpartition(scores, -min(2, k), axis=1)[:, -min(2, k):]
    return {
        "spearman": float(np.mean(spearman)), "kendall_tau": float(np.mean(kendall)),
        "top1_hit_rate": float(np.mean(selected == oracle)),
        "top2_recall": float(np.mean([oracle[index] in top_two[index] for index in range(len(oracle))])),
        "mean_regret": float(np.mean(mae[np.arange(len(mae)), selected] - mae.min(1))),
        "teacher_semantic_regret": float(np.mean(semantic_distance[np.arange(len(mae)), selected] - semantic_distance.min(1))),
        "selected_mae": float(np.mean(mae[np.arange(len(mae)), selected])),
    }


def group_summary(selector: np.ndarray, errors: dict[str, np.ndarray], scores: dict[str, np.ndarray],
                  fixed_mae: np.ndarray, simple_mae: np.ndarray, semantic_distance: np.ndarray,
                  semantic_consistency: np.ndarray) -> dict[str, Any]:
    mae = errors["masked_mae"][selector]; fixed = fixed_mae[selector]; simple = simple_mae[selector]
    semantic = semantic_distance[selector]; consistency = semantic_consistency[selector]
    best = mae.min(1); random_expected = mae.mean(1); oracle_index = mae.argmin(1)
    return {
        "count": int(len(mae)),
        "candidate_mae_mean": float(mae.mean()),
        "candidate_mae_std_within_sample": float(mae.std(1).mean()),
        "candidate_mae_range_within_sample": float(np.ptp(mae, axis=1).mean()),
        "oracle_best_mae": float(best.mean()), "random_expected_mae": float(random_expected.mean()),
        "oracle_improvement_vs_random": float((random_expected.mean() - best.mean()) / random_expected.mean()),
        "fixed_single_mae": float(fixed.mean()),
        "oracle_improvement_vs_fixed": float((fixed.mean() - best.mean()) / fixed.mean()),
        "simple_mae": float(simple.mean()), "oracle_gap_vs_simple": float(best.mean() - simple.mean()),
        "oracle_teacher_distance": float(semantic[np.arange(len(mae)), oracle_index].mean()),
        "random_teacher_distance": float(semantic.mean()),
        "oracle_teacher_consistency": float(consistency[np.arange(len(mae)), oracle_index].mean()),
        "random_teacher_consistency": float(consistency.mean()),
        "oracle_metrics": {name: float(values[selector].min(1).mean()) for name, values in errors.items()},
        "rankings": {name: ranking_metrics(value[selector], mae, semantic) for name, value in scores.items()},
    }


def _fault_type(run_uid: str) -> int:
    match = re.search(r":fault_(\d+):", str(run_uid)); return int(match.group(1)) if match else 0


def audit(config: dict[str, Any]) -> dict[str, Any]:
    views, _ = load_fixed_views(config); manifest = json.loads(Path(config["manifest"]).read_text(encoding="utf-8"))
    device = str(config["device"]); teacher = build_model(config["model"], views["train"]["clean"].shape[1], 2).to(device)
    teacher.load_state_dict(torch.load(config["teacher_checkpoint"], map_location=device, weights_only=True)); teacher.eval()
    split_results = {}
    for split in ("train", "validation", "test"):
        record = manifest["splits"][split]; path = Path(record["path"])
        if sha256_file(path) != record["sha256"]: raise RuntimeError(f"candidate hash mismatch: {split}")
        with np.load(path, allow_pickle=False) as archive:
            candidates, indices = archive["candidates"], archive["fixed_indices"]
        bundle = {key: value[indices] if isinstance(value, np.ndarray) and len(value) == len(views[split]["labels"]) else value
                  for key, value in views[split].items()}
        errors = oracle_candidate_errors(bundle["clean"], candidates, bundle["observation"])
        candidate_probability = teacher_probabilities(teacher, candidates, int(config["batch_size"]), device)
        clean_probability = teacher_probabilities(teacher, bundle["clean"], int(config["batch_size"]), device)
        semantic_distance = np.abs(candidate_probability - clean_probability[:, None]).sum(2) / 2
        semantic_consistency = candidate_probability.argmax(2) == clean_probability.argmax(1)[:, None]
        h1 = center_scores(candidates, bundle["observation"]); h2 = semantic_scores(candidate_probability)
        preliminary = {"h1_center": h1, "h2_semantic": h2}
        preliminary_metrics = {name: ranking_metrics(score, errors["masked_mae"], semantic_distance) for name, score in preliminary.items()}
        allow_h3 = any(value["spearman"] > 0 or value["top1_hit_rate"] > 1 / int(config["k_candidates"]) for value in preliminary_metrics.values())
        scores = dict(preliminary)
        if allow_h3: scores["h3_combined"] = combined_scores(h1, h2, float(config["lambda_sem"]))
        fixed_mae = per_sample_masked_mae(bundle["clean"], bundle["restored"], bundle["observation"])
        simple = _simple_interpolate(bundle["degraded"], bundle["observation"])
        simple_mae = per_sample_masked_mae(bundle["clean"], simple, bundle["observation"])
        labels = bundle["labels"]; types = np.asarray([_fault_type(value) for value in bundle["run_uid"]])
        groups = {"overall": np.ones(len(labels), dtype=bool), "normal": labels == 0, "fault": labels != 0}
        groups.update({f"fault_type_{kind:02d}": types == kind for kind in sorted(set(types) - {0})})
        split_results[split] = {"h3_allowed": allow_h3, "groups": {
            name: group_summary(selector, errors, scores, fixed_mae, simple_mae, semantic_distance, semantic_consistency)
            for name, selector in groups.items() if selector.any()
        }}
    train = split_results["train"]["groups"]; overall = train["overall"]
    best_name, best_ranking = max(overall["rankings"].items(), key=lambda item: (item[1]["spearman"], item[1]["top1_hit_rate"]))
    checks = {
        "nonzero_stable_candidate_difference": overall["candidate_mae_std_within_sample"] > 1e-4 and train["normal"]["candidate_mae_std_within_sample"] > 1e-4 and train["fault"]["candidate_mae_std_within_sample"] > 1e-4,
        "oracle_improves_random_by_5_percent": overall["oracle_improvement_vs_random"] >= float(config["gate"]["oracle_random_mae_improvement"]),
        "oracle_semantics_not_worse_than_random": overall["oracle_teacher_distance"] <= overall["random_teacher_distance"] + 1e-12,
        "no_reference_rank_signal": best_ranking["spearman"] >= float(config["gate"]["minimum_spearman"]),
        "no_reference_topk_above_random": best_ranking["top1_hit_rate"] > float(config["gate"]["random_top1"]) or best_ranking["top2_recall"] > float(config["gate"]["random_top2"]),
        "normal_fault_same_oracle_direction": train["normal"]["oracle_improvement_vs_random"] > 0 and train["fault"]["oracle_improvement_vs_random"] > 0,
    }
    engineering = checks["oracle_improves_random_by_5_percent"] and (checks["no_reference_rank_signal"] or checks["no_reference_topk_above_random"])
    passed = sum(checks.values()) >= 4 and engineering
    result = {"markers": config["markers"], "status": "INTRA_SAMPLE_CANDIDATE_RANKING_GO" if passed else "INTRA_SAMPLE_CANDIDATE_RANKING_NO_GO",
              **environment_metadata(), "candidate_manifest": manifest, "splits": split_results,
              "best_no_reference_score": best_name, "gate_checks": checks, "engineering_gate": engineering,
              "downstream_retest_allowed": passed}
    write_json(Path(config["audit_result"]), result)
    csv_path = Path(config["audit_result"]).with_suffix(".csv")
    rows = ["split,group,score,spearman,kendall_tau,top1_hit_rate,top2_recall,mean_regret"]
    for split, split_value in split_results.items():
        for group, group_value in split_value["groups"].items():
            for score, metrics in group_value["rankings"].items():
                rows.append(",".join(map(str, [split, group, score, metrics["spearman"], metrics["kendall_tau"], metrics["top1_hit_rate"], metrics["top2_recall"], metrics["mean_regret"]])))
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/intra_sample_candidate_audit.yaml")
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = audit(config)
    print(json.dumps({"status": result["status"], "best_score": result["best_no_reference_score"],
                      "checks": result["gate_checks"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
