from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal


@dataclass
class DataConfig:
    source: Literal["vitaldb", "csv"] = "vitaldb"
    subset_case_ids: list[int] = field(default_factory=list)
    resample_hz: int = 1
    cache_dir: str = "artifacts/cache"


@dataclass
class RuntimeConfig:
    seed: int = 7
    mode: Literal["research", "clinician"] = "research"


@dataclass
class AppConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def to_dict(self) -> dict:
        return asdict(self)
