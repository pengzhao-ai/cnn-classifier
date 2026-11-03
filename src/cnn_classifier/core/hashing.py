"""Stable hashing for configs, manifests, and run fingerprints."""

from __future__ import annotations

import hashlib
import json


def stable_hash(payload: object) -> str:
    """Return a short canonical SHA-256 hash of a JSON-compatible payload."""
    text = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        allow_nan=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
