from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from degradations import apply_degradation
from scripts.common import load_config, prepare_synthetic


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/debug.yaml"); parser.add_argument("--output", default="outputs/debug/degradation.png")
    args = parser.parse_args(); config = load_config(args.config); data, _, _ = prepare_synthetic(config)
    sample = data["train"][0][0]; result = apply_degradation(sample, config["degradation"], config["degradation_severity"], config["random_seed"], "visual-check", config["degradation_space"])
    figure, axes = plt.subplots(3, 1, sharex=True); axes[0].plot(sample.T); axes[0].set_title("Clean SYNTHETIC"); axes[1].plot(result.data.T); axes[1].set_title("Degraded DEBUG"); axes[2].imshow(result.corruption_mask, aspect="auto"); axes[2].set_title("Corruption mask — NOT FOR SCIENTIFIC COMPARISON")
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); figure.tight_layout(); figure.savefig(output); plt.close(figure)
