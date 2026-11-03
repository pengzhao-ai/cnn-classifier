"""Run directories, metadata, and atomic artifact writes."""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torchvision
import yaml

from cnn_classifier.config.schema import RunConfig
from cnn_classifier.core.device import device_report
from cnn_classifier.core.hashing import stable_hash


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return cleaned or "run"


def _git_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None}
    return {"commit": commit, "branch": branch, "dirty": dirty}


@dataclass
class RunContext:
    """A self-describing directory for one experiment run."""

    directory: Path
    config: RunConfig
    config_hash: str

    @classmethod
    def create(cls, config: RunConfig, root: Path = Path("runs")) -> RunContext:
        config_hash = stable_hash(config.model_dump(mode="json"))
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        base = root / f"{timestamp}_{_slug(config.experiment.name)}_{config_hash[:8]}"
        directory = base
        suffix = 1
        while directory.exists():
            directory = base.with_name(f"{base.name}_{suffix}")
            suffix += 1
        for child in ("data", "checkpoints", "predictions", "figures"):
            (directory / child).mkdir(parents=True, exist_ok=True)
        run = cls(directory=directory, config=config, config_hash=config_hash)
        run.write_text(
            "config.resolved.yaml",
            yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
        )
        run.write_json(
            "environment.json",
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "device": device_report(),
            },
        )
        run.write_json("git.json", _git_metadata())
        run.set_status("running")
        return run

    def _atomic_write(self, relative_path: str, data: bytes) -> Path:
        path = self.directory / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary_path = Path(temporary)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
            os.replace(temporary_path, path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
        return path

    def write_json(self, relative_path: str, payload: Any) -> Path:
        text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        return self._atomic_write(relative_path, text.encode("utf-8"))

    def write_text(self, relative_path: str, text: str) -> Path:
        return self._atomic_write(relative_path, text.encode("utf-8"))

    def set_status(self, status: str, error: str | None = None) -> None:
        self.write_json(
            "status.json",
            {
                "status": status,
                "updated_at": datetime.now(UTC).isoformat(),
                "error": error,
            },
        )

    def close(self, status: str = "completed") -> None:
        self.set_status(status)
