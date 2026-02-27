from __future__ import annotations

import json
import pathlib
from typing import Any

from .schema import AppConfig, DataConfig, RuntimeConfig


def load_config(path: str | pathlib.Path) -> AppConfig:
    p = pathlib.Path(path)
    data: dict[str, Any] = json.loads(p.read_text()) if p.exists() else {}

    runtime = RuntimeConfig(**data.get("runtime", {}))
    raw_data = data.get("data", {})
    cfg_data = DataConfig(**raw_data)
    if cfg_data.resample_hz < 1:
        raise ValueError("resample_hz must be >= 1")

    return AppConfig(runtime=runtime, data=cfg_data)
