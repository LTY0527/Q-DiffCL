from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from scripts.run_3w_diffusion_1seed import METHODS, R2_METHOD, run
from scripts.run_r2_multiclass_criticality import R2_WEIGHTS, validate_r2_weights
from utils import write_json


def validate_config(config: dict) -> None:
    if list(map(int, config["existing_seeds"])) != [42, 43, 44]:
        raise ValueError("frozen existing seeds must be 42/43/44")
    if list(map(int, config["new_seeds"])) != [45, 46] or list(map(int, config["all_seeds"])) != [42, 43, 44, 45, 46]:
        raise ValueError("reliability audit must add only seeds 45/46")
    validate_r2_weights(config["r2_weights"])


def _frozen_config(base: dict, seed: int, protocol_seed: int, source: str,
                   methods: list[str], output_dir: Path) -> dict:
    current = copy.deepcopy(base); current["seed"] = int(seed); current["protocol_seed"] = int(protocol_seed)
    current["criticality_source"] = source; current["methods"] = methods
    current["training"]["supcon_batching"] = "original"; current["output_dir"] = str(output_dir)
    return current


def run_audit(config: dict, data_root: Path, selected_seeds: list[int] | None = None) -> dict:
    validate_config(config); base = yaml.safe_load(Path(config["base_config"]).read_text(encoding="utf-8"))
    output = Path(config["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "result_manifest.json"
    completed = json.loads(manifest_path.read_text(encoding="utf-8")).get("seed_results", {}) if manifest_path.exists() else {}
    seeds = list(map(int, selected_seeds if selected_seeds is not None else config["new_seeds"]))
    if any(seed not in {45, 46} for seed in seeds): raise ValueError("only new seeds 45/46 may be trained")
    for seed in seeds:
        r1_config = _frozen_config(base, seed, config["protocol_seed"], config["r1_criticality_source"],
                                   [METHODS[1], METHODS[2]], output / f"seed_{seed}" / "uniform_r1")
        r1_result = run(r1_config, data_root)
        r2_config = _frozen_config(base, seed, config["protocol_seed"], config["r2_criticality_source"],
                                   [R2_METHOD], output / f"seed_{seed}" / "r2")
        r2_result = run(r2_config, data_root)
        if r1_result["fairness"]["window_refs_sha256"] != r2_result["fairness"]["window_refs_sha256"]:
            raise RuntimeError(f"seed {seed} R1/R2 window references differ")
        if r1_result["fairness"]["initialization_sha256"] != r2_result["fairness"]["initialization_sha256"]:
            raise RuntimeError(f"seed {seed} R1/R2 initialization differs")
        completed[str(seed)] = {
            "uniform_r1_result_path": str(Path(r1_config["output_dir"]) / "result.json"),
            "r2_result_path": str(Path(r2_config["output_dir"]) / "result.json"),
            "trained_methods": [METHODS[1], METHODS[2], R2_METHOD], "status": "complete"}
        write_json(manifest_path, {"new_seeds": sorted(map(int, completed)), "seed_results": completed})
    payload = {"stage": "3W_R1_R2_5SEED_RELIABILITY", "existing_seeds": config["existing_seeds"],
               "new_seeds": sorted(map(int, completed)), "all_seeds": config["all_seeds"],
               "protocol_seed": int(config["protocol_seed"]), "supcon_batching": "original",
               "r1_criticality_source": config["r1_criticality_source"],
               "r2_criticality_source": config["r2_criticality_source"], "new_training_count": 3 * len(completed),
               "seed_results": completed}
    write_json(manifest_path, payload); return payload


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/3w_r1_r2_5seed_reliability.yaml")
    parser.add_argument("--data-root", type=Path, required=True); parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    print(json.dumps(run_audit(config, args.data_root, args.seeds), ensure_ascii=False))


if __name__ == "__main__": main()
