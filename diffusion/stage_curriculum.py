from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


CurriculumMode = Literal["epoch_only", "stage_aware"]
STAGES = ("normal", "early", "middle", "stable")


@dataclass(frozen=True)
class StageAwareTimestepScheduler:
    mode: CurriculumMode
    t_start: int
    targets: dict[str, int]
    t_critical: int = 1

    def __post_init__(self) -> None:
        if self.mode not in {"epoch_only", "stage_aware"}:
            raise ValueError("unknown curriculum mode")
        if set(self.targets) != set(STAGES):
            raise ValueError("curriculum targets must define normal/early/middle/stable")
        if self.t_critical != 1 or self.t_start != 2:
            raise ValueError("C3 freezes t_critical=1 and t_start=2")
        if any(int(value) < self.t_critical for value in self.targets.values()):
            raise ValueError("stage target is below the critical timestep")

    def timestep(self, stage: str, epoch: int, total_epochs: int) -> int:
        if epoch < 0 or epoch >= total_epochs or total_epochs < 1:
            raise ValueError("invalid epoch")
        selected = "normal" if self.mode == "epoch_only" else stage
        if selected not in self.targets:
            raise ValueError(f"unknown stage: {stage}")
        progress = epoch / max(total_epochs - 1, 1)
        return int(round(self.t_start + progress * (int(self.targets[selected]) - self.t_start)))

    def epoch_timesteps(self, epoch: int, total_epochs: int) -> dict[str, int]:
        return {stage: self.timestep(stage, epoch, total_epochs) for stage in STAGES}

