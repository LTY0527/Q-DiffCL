from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from utils import environment_metadata, write_json


def summarize(config: dict[str, Any], results: dict[str, dict[str, Any]], selected_lambda: float,
              selected_alpha: float) -> dict[str, Any]:
    seeds = list(map(int, config["seeds"])); chosen = {}
    for seed in seeds:
        for method in ("G0", "G1-fixed"):
            chosen[f"{seed}:{method}"] = results[f"{seed}:{method}"]["best_by_alpha"][str(float(selected_alpha))]
    wins = {"consistency": 0, "normal_flip": 0, "fault_flip": 0, "last_not_better": 0}
    for seed in seeds:
        g0 = chosen[f"{seed}:G0"]["metrics"]; g1 = chosen[f"{seed}:G1-fixed"]["metrics"]
        wins["consistency"] += g1["teacher_consistency"] > g0["teacher_consistency"]
        wins["normal_flip"] += g1["normal_to_fault_flip"] <= g0["normal_to_fault_flip"]
        wins["fault_flip"] += g1["fault_to_normal_flip"] <= g0["fault_to_normal_flip"]
        wins["last_not_better"] += chosen[f"{seed}:G1-fixed"]["score"] <= results[f"{seed}:G1-fixed"]["last_by_alpha"][str(float(selected_alpha))]["score"]
    mean_balanced = {method: float(np.mean([chosen[f"{seed}:{method}"]["metrics"]["balanced_flip_rate"] for seed in seeds])) for method in ("G0", "G1-fixed")}
    diversity_ok = all(.10 <= chosen[f"{seed}:G1-fixed"]["metrics"]["normalized_l1"] <= .20 for seed in seeds)
    no_normal_explosion = all(chosen[f"{seed}:G1-fixed"]["metrics"]["normal_to_fault_flip"] <= .30 for seed in seeds)
    checks = {"consistency_wins_2_of_3": wins["consistency"] >= 2, "normal_flip_wins_2_of_3": wins["normal_flip"] >= 2,
              "fault_flip_wins_2_of_3": wins["fault_flip"] >= 2,
              "mean_balanced_flip_below_g0": mean_balanced["G1-fixed"] < mean_balanced["G0"],
              "diversity_in_target_range": diversity_ok, "no_normal_flip_above_30_percent": no_normal_explosion,
              "best_not_worse_than_last_all_seeds": wins["last_not_better"] == 3}
    status = "SEMANTIC_GENERATOR_FIX_READY_FOR_DOWNSTREAM_RETEST" if all(checks.values()) else "SEMANTIC_GENERATOR_FIX_NO_GO"
    summary = {"markers": config["markers"], "status": status,
               "selected_configuration": {"lambda_sem": selected_lambda, "alpha": selected_alpha},
               "results": results, "chosen": chosen, "wins": wins, "mean_balanced_flip_rate": mean_balanced,
               "gate_checks": checks, **environment_metadata()}
    write_json(Path(config["output_dir"]) / "summary.json", summary); return summary
