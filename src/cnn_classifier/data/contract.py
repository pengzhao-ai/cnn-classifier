"""Model input contract derivation and serialization."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

from cnn_classifier.core.types import DatasetBundle, ModelInputContract


def build_model_contract(
    bundle: DatasetBundle,
    *,
    model_name: str,
    stem_adaptation: str | None = None,
) -> ModelInputContract:
    """Return the exact model-input contract for a selected model."""
    _ = model_name
    return replace(bundle.model_contract, stem_adaptation=stem_adaptation)


def write_model_contract(contract: ModelInputContract, path: Path) -> Path:
    """Write the contract as deterministic JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(contract)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
