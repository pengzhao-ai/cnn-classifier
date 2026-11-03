"""Optional TensorBoard tracker."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class TensorBoardTracker:
    def __init__(self, log_dir: str) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise RuntimeError("TensorBoard tracking requires the 'tracking' extra") from exc
        self.writer: Any = SummaryWriter(log_dir)

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)

    def close(self) -> None:
        self.writer.close()
