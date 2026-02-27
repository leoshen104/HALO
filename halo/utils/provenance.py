from __future__ import annotations

import hashlib
import json
import pathlib
from datetime import datetime, timezone
from typing import Any


def config_hash(config_dict: dict[str, Any]) -> str:
    payload = json.dumps(config_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_manifest(path: str | pathlib.Path, *, config: dict[str, Any], seed: int) -> pathlib.Path:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "config_hash": config_hash(config),
        "config": config,
    }
    p.write_text(json.dumps(manifest, indent=2))
    return p
