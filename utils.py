from __future__ import annotations

import hashlib
import json
import logging
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def deterministic_seed(master_seed: int, sample_id: str | int, kind: str) -> int:
    payload = f"{master_seed}|{sample_id}|{kind}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def select_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def git_commit() -> str:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        return value or "UNCOMMITTED_INITIAL_WORKSPACE"
    except (OSError, subprocess.CalledProcessError):
        return "UNCOMMITTED_INITIAL_WORKSPACE"


def environment_metadata() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version,
        "git_commit": git_commit(),
        "cuda": None,
        "gpu": None,
    }
    try:
        import torch

        result["pytorch"] = torch.__version__
        result["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            result["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        result["pytorch"] = "NOT_INSTALLED"
    return result


def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("q_diffcl")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (
        logging.StreamHandler(),
        logging.FileHandler(output_dir / "run.log", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
