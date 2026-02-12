from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class VitalDBRequest:
    case_ids: list[int]
    tracks: list[str]


def fetch_vitaldb_subset(req: VitalDBRequest) -> pd.DataFrame:
    """Placeholder for Phase 1 integration.

    The implementation intentionally returns an empty canonical frame so the
    scaffold can run deterministically before full VitalDB wiring.
    """
    return pd.DataFrame(columns=["Time", "HR", "SpO2", "MAP", "EtCO2", "RR"])
