"""Console tracker."""

from __future__ import annotations

from collections.abc import Mapping


class ConsoleTracker:
    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        rendered = " | ".join(f"{key}={value:.4f}" for key, value in metrics.items())
        print(f"step {step}: {rendered}")

    def close(self) -> None:
        return None
