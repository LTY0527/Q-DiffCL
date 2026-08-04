from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from diffusion import DiffusionSchedule
from models import MinimalConditionalDiffusion1D
from scripts.train_diffusion_recovery import recovery_metrics, restore_array
from utils import write_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--checkpoint", required=True); parser.add_argument("--input", required=True); parser.add_argument("--output", required=True); args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")); device = config["device"]
    batch = np.load(args.input); channels = batch["clean"].shape[1]
    model = MinimalConditionalDiffusion1D(channels, int(config["diffusion"]["hidden_channels"]), int(config["diffusion"]["hidden_channels"]), int(config["diffusion"]["residual_blocks"])).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    schedule = DiffusionSchedule.cosine(int(config["diffusion"]["steps"]), device)
    restored = restore_array(model, batch["degraded"], batch["observation"], schedule, int(config["batch_size"]), device, int(config["random_seed"]) + 9000, batch["clip_min"], batch["clip_max"])
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); np.save(output, restored)
    write_json(output.with_suffix(".json"), {"markers": config["markers"], **recovery_metrics(batch["clean"], restored, batch["observation"])})
