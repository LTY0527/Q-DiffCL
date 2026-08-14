from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from scripts.run_3w_diffusion_1seed import run


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/3w_balanced_supcon_stability.yaml")
    parser.add_argument("--data-root", type=Path, required=True); parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base = yaml.safe_load(Path(config["base_config"]).read_text(encoding="utf-8")); output = Path(config["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "result_manifest.json"
    completed = json.loads(manifest_path.read_text(encoding="utf-8")).get("seed_results", {}) if manifest_path.exists() else {}
    selected = list(map(int, args.seeds if args.seeds is not None else config["seeds"]))
    for seed in selected:
        current = copy.deepcopy(base); current["seed"] = seed; current["protocol_seed"] = int(config["protocol_seed"])
        current["criticality_source"] = config["criticality_source"]; current["methods"] = list(config["methods"])
        current["training"]["supcon_batching"] = "balanced_positive_safe"
        current["training"]["balanced_sampler"] = copy.deepcopy(config["balanced_sampler"])
        current["output_dir"] = str(output / f"seed_{seed}")
        result = run(current, args.data_root); path = Path(current["output_dir"]) / "result.json"
        completed[str(seed)] = {"result_path": str(path), "methods": list(result["methods"]), "status": "complete"}
    manifest = {"seeds": sorted(map(int, completed)), "protocol_seed": int(config["protocol_seed"]),
                "criticality_source": config["criticality_source"], "seed_results": completed}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__": main()
