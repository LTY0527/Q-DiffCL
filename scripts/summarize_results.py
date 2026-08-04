from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from metrics import drop_rate, performance_retention, supcon_gain
from utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("root"); parser.add_argument("--output", default="outputs/summary.json")
    args = parser.parse_args(); rows = []
    for path in Path(args.root).rglob("metadata.json"):
        value = json.loads(path.read_text(encoding="utf-8")); value["path"] = str(path.parent); rows.append(value)
    metrics = [float(row["test_metrics"]["macro_f1"]) for row in rows]
    summary = {"count": len(rows), "macro_f1_mean": float(np.mean(metrics)) if metrics else None, "macro_f1_std": float(np.std(metrics)) if metrics else None, "rows": rows, "formulas": {"performance_retention": "degraded/clean", "drop_rate": "(clean-degraded)/clean", "supcon_gain": "supcon-ce"}}
    write_json(Path(args.output), summary)


if __name__ == "__main__": main()

