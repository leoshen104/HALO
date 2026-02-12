from __future__ import annotations

import pandas as pd


def fetch_vitaldb_via_api(*, case_id: int, track_names: list[str]) -> pd.DataFrame:
    """Web API fallback stub for deterministic scaffolding phase."""
    _ = (case_id, track_names)
    return pd.DataFrame(columns=["Time", "HR", "SpO2", "MAP", "EtCO2", "RR"])
