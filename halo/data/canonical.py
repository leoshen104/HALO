from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CanonicalSignalMap:
    time: str = "Time"
    hr: str = "HR"
    map: str = "MAP"
    spo2: str = "SpO2"
    etco2: str = "EtCO2"
    rr: str = "RR"


@dataclass(frozen=True)
class SignalAvailability:
    hr: bool
    map: bool
    spo2: bool
    etco2: bool
    rr: bool


def availability_from_columns(columns: list[str]) -> SignalAvailability:
    c = set(columns)
    return SignalAvailability(
        hr="HR" in c,
        map="MAP" in c,
        spo2="SpO2" in c,
        etco2="EtCO2" in c,
        rr="RR" in c,
    )


def maybe_value(value: Optional[float]) -> str:
    return "Not available" if value is None else f"{value:.2f}"
