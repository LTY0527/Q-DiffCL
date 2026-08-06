from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def generator_allows_downstream(summary: dict) -> bool:
    return summary.get("status") == "SEMANTIC_GENERATOR_FIX_READY_FOR_DOWNSTREAM_RETEST"


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/semantic_generator_stability_fix.yaml")
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    summary = json.loads((Path(config["output_dir"]) / "summary.json").read_text(encoding="utf-8"))
    if not generator_allows_downstream(summary):
        print(json.dumps({"status": "SKIPPED_BY_GENERATOR_GATE", "training_skipped": True}, ensure_ascii=False)); return
    raise NotImplementedError("downstream retest is only implemented after the generator gate passes")


if __name__ == "__main__": main()
