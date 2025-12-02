# HALO — Hypothesis & Alarm Logic Orchestrator
# v4.x — Live Fusion Prototype
# Educational prototype only. NOT for clinical use.

# ================== IMPORTS ==================
import io
import time
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import textwrap

# ================== PAGE / BRANDING ==================
st.set_page_config(page_title="HALO v4", page_icon="🛡️", layout="wide")
DEMO_MODE = False  # toggle extra controls in sidebar

HALO_DARK = """
<style>
:root {
    --halo-bg: #050609;
    --halo-surface: #111318;
    --halo-surface-soft: #181b22;
    --halo-border: rgba(255,255,255,0.06);
    --halo-border-strong: rgba(255,255,255,0.12);
    --halo-text-main: #F7F7F7;
    --halo-text-soft: #C8CCD2;
    --halo-text-muted: #8B9097;
    --halo-blue: #4DA8FF;
    --halo-blue-soft: rgba(77,168,255,0.18);
    --halo-teal: #59E3C2;
    --halo-amber: #E6C87A;
    --halo-red: #E05A68;
    --halo-purple: #B798FF;
    --halo-radius: 18px;
    --halo-pill: 999px;
    --halo-shadow: 0 8px 24px rgba(0,0,0,0.55);
}

/* Global dark background */
html, body, [data-testid="stAppViewContainer"], .stApp {
    background-color: var(--halo-bg) !important;
    color: var(--halo-text-main) !important;
    font-family: "Inter", -apple-system, BlinkMacSystemFont, system-ui, sans-serif !important;
}

/* Flatten the header */
[data-testid="stHeader"] {
    background: transparent !important;
    box-shadow: none !important;
}

/* Main container flush and wide */
.block-container {
    padding: 0.8rem 1.8rem 2rem 1.8rem !important;
    max-width: 100% !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #07090d !important;
    border-right: 1px solid var(--halo-border-strong) !important;
}
[data-testid="stSidebar"] * {
    color: var(--halo-text-soft) !important;
}

/* Text inputs */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: var(--halo-surface-soft) !important;
    color: var(--halo-text-main) !important;
    border-radius: 12px !important;
    border: 1px solid var(--halo-border) !important;
}

/* Buttons */
.stButton > button {
    background: var(--halo-surface) !important;
    border: 1px solid var(--halo-blue) !important;
    border-radius: var(--halo-pill) !important;
    color: var(--halo-blue) !important;
    padding: 0.3rem 1.1rem !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.45);
    transition: all 0.12s ease-in-out;
}
.stButton > button:hover {
    background: var(--halo-blue-soft) !important;
    transform: translateY(-1px);
}

/* Cards */
.halo-card,
.halo-card-soft,
.halo-response-card {
    background: var(--halo-surface) !important;
    border-radius: var(--halo-radius) !important;
    padding: 10px 12px !important;
    border: 1px solid var(--halo-border) !important;
    box-shadow: var(--halo-shadow);
}
.halo-card-soft {
    background: var(--halo-surface-soft) !important;
}

/* Summary card (used for trajectory + state model) */
.halo-summary-card {
    background: var(--halo-surface-soft) !important;
    border-left: 5px solid var(--halo-blue);
    border-radius: var(--halo-radius);
    padding: 12px 14px;
    border: 1px solid var(--halo-border);
}

/* Alarm cards */
.halo-alarm-card {
    border-radius: 14px;
    padding: 8px 12px;
    margin-bottom: 6px;
}
.halo-alarm-advisory {
    background: rgba(77,168,255,0.07);
    border-left: 4px solid var(--halo-blue);
}
.halo-alarm-warning {
    background: rgba(230,200,122,0.10);
    border-left: 4px solid var(--halo-amber);
}
.halo-alarm-critical {
    background: rgba(224,90,104,0.12);
    border-left: 4px solid var(--halo-red);
}

/* OK banner */
.halo-ok-banner {
    border-left: 4px solid var(--halo-teal);
    background: rgba(89,227,194,0.10);
    border-radius: 14px;
    padding: 10px 14px;
}

/* Gauge */
.halo-gauge-outer {
    width: 100%;
    height: 16px;
    border-radius: 999px;
    background: #1a1d23;
    border: 1px solid var(--halo-border-strong);
    overflow: hidden;
}
.halo-gauge-inner {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #2f6b4b, #e6c87a, #e05a68);
}

/* Voice card */
.halo-voice-card {
    background: rgba(230,200,122,0.10) !important;
    border-left: 4px solid var(--halo-amber) !important;
    border-radius: 14px !important;
    padding: 8px 12px !important;
}

/* Data quality badge */
.halo-dq-badge {
    background: var(--halo-surface-soft);
    color: var(--halo-text-soft);
    padding: 8px 10px;
    border-radius: 12px;
    border: 1px solid var(--halo-border);
    font-size: 0.9rem;
}

/* Vit legend */
.halo-legend {
    display:flex;
    gap:12px;
    flex-wrap:wrap;
    font-size:0.8rem;
    margin:4px 0 10px 0;
}
.halo-legend-item {
    display:flex;
    align-items:center;
    gap:6px;
}
.halo-legend-swatch {
    width:10px;
    height:10px;
    border-radius:999px;
}

/* Top pill */
.halo-top-pill {
    font-size: 0.9rem;
    color: var(--halo-text-muted);
    padding: 6px 12px;
    border-radius: 999px;
    border: 1px solid var(--halo-border);
    background: var(--halo-surface-soft);
}

/* Charts container soft framing */
[data-testid="stVerticalBlock"] .element-container:has(canvas),
[data-testid="stVerticalBlock"] .element-container:has(svg) {
    background: var(--halo-surface);
    border-radius: var(--halo-radius);
    border: 1px solid var(--halo-border);
    padding: 4px 6px;
    margin-top: 6px;
}

/* General text */
.stMarkdown, .stCaption, p, span {
    font-size: 0.94rem !important;
    color: var(--halo-text-soft) !important;
}
h2, h3, h4 {
    color: var(--halo-text-main) !important;
    letter-spacing: -0.01em !important;
}

/* Voice recorder widget */
.audio-recorder,
.audio-recorder button {
    border-radius: 999px !important;
    background: var(--halo-surface-soft) !important;
    border: 1px solid var(--halo-blue) !important;
    color: var(--halo-blue) !important;
}

</style>
"""
st.markdown(HALO_DARK, unsafe_allow_html=True)

# ================== SMALL UTILS ==================
def _setdefault(key, value):
    if key not in st.session_state:
        st.session_state[key] = value

# ================== SESSION DEFAULTS ==================
_setdefault("history", pd.DataFrame(columns=["Time", "HR", "SpO2", "MAP", "EtCO2", "RR"]))
_setdefault("running", False)
_setdefault("mode", "Live")  # "Live" or "Replay"
_setdefault("replay_df", None)
_setdefault("replay_idx", 0)
_setdefault("replay_speed", 1)
_setdefault("sim_hz", 5)
_setdefault("events", [])
_setdefault("audit", [])
_setdefault("conversation_log", [])
_setdefault("voice_status", "Idle")

# Patient context
_setdefault("age", 62)
_setdefault("sex", "Unknown")
_setdefault("asa_class", "III")
_setdefault("bmi", 27.0)
_setdefault("case_type", "Elective laparotomy")
_setdefault("emergent", False)

# Simulation state
_setdefault("sim_val", {"HR": 80, "SpO2": 97, "MAP": 80, "EtCO2": 37, "RR": 12})
_setdefault("sim_time", 0)

# Pharmacology effects (active)
_setdefault("vaso_effect", 0.0)
_setdefault("fluid_effect", 0.0)

# Thresholds
_setdefault("low_spo2", 92)
_setdefault("tachy_hr", 120)
_setdefault("low_map", 65)
_setdefault("high_et", 50)
_setdefault("low_et", 30)
_setdefault("low_rr", 8)
_setdefault("high_rr", 28)

# Persistence windows (sec)
_setdefault("win_spo2", 8)
_setdefault("win_hr", 8)
_setdefault("win_map", 10)
_setdefault("win_resp", 12)

# Hysteresis & cooldown
_setdefault("hys_spo2", 2)
_setdefault("hys_map", 5)
_setdefault("cooldown", 30)

# Noise / artifact
_setdefault("enable_noise", True)
_setdefault("artifact_pct", 5)

# Voice transcript
_setdefault("last_voice_transcript", "")

# ================== VOICE HELPERS ==================
def voice_available() -> bool:
    try:
        import speech_recognition  # type: ignore
        from audio_recorder_streamlit import audio_recorder  # type: ignore
        return True
    except Exception:
        return False

def voice_widget(label: str = "🎤 Click to speak") -> str:
    """Capture short voice input and update last_voice_transcript + status."""
    try:
        from audio_recorder_streamlit import audio_recorder  # type: ignore
        import speech_recognition as sr  # type: ignore

        status_key = "voice_status"
        if status_key not in st.session_state:
            st.session_state[status_key] = "Idle"

        audio_bytes = audio_recorder(text=label, icon_size="3x")
        if not audio_bytes:
            if st.session_state[status_key] != "Listening…":
                st.session_state[status_key] = "Idle"
            return ""

        st.session_state[status_key] = "Listening…"

        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 200
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.8

        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio, language="en-US")
        except sr.UnknownValueError:
            st.session_state[status_key] = "Error: speech not clear"
            st.warning("Voice not clear enough to transcribe. Try again a bit closer/slower.")
            return ""
        except Exception as e:
            st.session_state[status_key] = "Error"
            st.warning(f"Transcription error: {e}")
            return ""

        text = text.strip()
        st.session_state["last_voice_transcript"] = text
        st.session_state[status_key] = "Idle"
        return text
    except Exception as e:
        st.session_state["voice_status"] = f"Error: {e}"
        st.info("Voice capture not available (missing optional dependencies or microphone permissions).")
        return ""

# ================== COLUMN RESOLUTION ==================
def _resolve_cols(df: pd.DataFrame) -> dict:
    def pick(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None
    return {
        "Time": pick(["Time", "time", "t"]),
        "MAP": pick(["MAP", "Map", "map"]),
        "HR": pick(["HR", "Heart Rate", "heart_rate", "Hr", "hr"]),
        "SpO2": pick(["SpO2", "SpO₂", "spo2", "SPO2"]),
        "EtCO2": pick(["EtCO2", "EtCO₂", "etco2", "ETCO2", "ETCO₂"]),
        "RR": pick(["RR", "Resp Rate", "Respiration", "resp_rate", "rr"]),
    }

# ================== SMALL TIME/SLICE HELPERS ==================
def last_n(df_: pd.DataFrame, seconds: int) -> pd.DataFrame:
    if df_ is None or df_.empty or "Time" not in df_.columns:
        return df_
    end = df_["Time"].iloc[-1]
    start = max(0, end - seconds + 1)
    return df_[df_["Time"].between(start, end)]

def _time_now_index() -> int:
    if st.session_state.history.empty:
        return 0
    return int(st.session_state.history["Time"].iloc[-1])

def events_dataframe() -> pd.DataFrame:
    if "events" not in st.session_state or not st.session_state.events:
        return pd.DataFrame(columns=["Time", "name", "type"])

    df_e = pd.DataFrame(st.session_state.events)
    df_e = df_e.rename(columns={"t": "Time"})

    def infer_type(name: str) -> str:
        if "Fluid" in name:
            return "Fluids"
        if "Vasopressor" in name:
            return "Vasopressor"
        if "Incision" in name:
            return "Incision"
        if "Position" in name:
            return "Position"
        return "Other"

    df_e["type"] = df_e["name"].apply(infer_type)
    return df_e

def _age_since_event(name: str, horizon: int) -> Optional[float]:
    t_now = _time_now_index()
    recent = [
        e for e in st.session_state.events
        if e["name"] == name and 0 <= (t_now - e["t"]) <= horizon
    ]
    if not recent:
        return None
    return float(min(t_now - e["t"] for e in recent))

def _decay_linear(age: Optional[float], horizon: int) -> float:
    if age is None:
        return 0.0
    return max(0.0, (horizon - age) / float(horizon))

# ================== SCENARIO MODIFIERS ==================
def _scenario_mods():
    name = getattr(st.session_state, "scenario_name", None) if "scenario_name" in st.session_state else None
    end = getattr(st.session_state, "scenario_end", 0.0) if "scenario_end" in st.session_state else 0.0
    active = (name is not None) and (time.time() <= end)
    if not active:
        return dict(spo2=0, hr=0, map=0, et_high=0, rr_low=0)

    if name == "Bleed":
        return dict(spo2=0, hr=-10, map=+5, et_high=0, rr_low=0)
    if name == "Bronchospasm":
        return dict(spo2=+2, hr=0, map=0, et_high=-4, rr_low=+2)
    if name == "Vasodilation":
        return dict(spo2=0, hr=0, map=+5, et_high=0, rr_low=0)
    if name == "Pain/Light":
        return dict(spo2=0, hr=-10, map=0, et_high=0, rr_low=0)
    if name in ["OB_Hemorrhage", "Sepsis_Laparotomy"]:
        return dict(spo2=0, hr=-5, map=+5, et_high=0, rr_low=0)
    if name in ["Thoracic_OneLung", "Craniotomy_Hypertensive"]:
        return dict(spo2=0, hr=-5, map=0, et_high=0, rr_low=0)
    return dict(spo2=0, hr=0, map=0, et_high=0, rr_low=0)

def _event_mods():
    H_INCISION = 90
    H_FLUIDS = 120
    H_PRESSOR = 180
    H_POSN = 120

    age_incision = _age_since_event("Incision", H_INCISION)
    age_fluids = _age_since_event("Fluids 250 mL", H_FLUIDS)
    age_pressor = _age_since_event("Vasopressor", H_PRESSOR)
    age_posn = _age_since_event("Position change", H_POSN)

    w_inc = _decay_linear(age_incision, H_INCISION)
    w_flu = _decay_linear(age_fluids, H_FLUIDS)
    w_pre = _decay_linear(age_pressor, H_PRESSOR)
    w_pos = _decay_linear(age_posn, H_POSN)

    spo2 = -2.0 * w_pos
    hr = +10.0 * w_inc
    map_ = (-5.0 * w_flu) + (+5.0 * w_pre)
    et_h = 0.0
    rr_l = 0.0

    return dict(spo2=spo2, hr=hr, map=map_, et_high=et_h, rr_low=rr_l)

def effective_thresholds():
    base = dict(
        low_spo2=st.session_state.low_spo2,
        tachy_hr=st.session_state.tachy_hr,
        low_map=st.session_state.low_map,
        high_et=st.session_state.high_et,
        low_rr=st.session_state.low_rr,
    )
    scen = _scenario_mods()
    ev = _event_mods()
    eff = dict(
        low_spo2=int(round(np.clip(base["low_spo2"] + scen["spo2"] + ev["spo2"], 80, 99))),
        tachy_hr=int(round(np.clip(base["tachy_hr"] + scen["hr"] + ev["hr"], 80, 180))),
        low_map=int(round(np.clip(base["low_map"] + scen["map"] + ev["map"], 45, 90))),
        high_et=int(round(np.clip(base["high_et"] + scen["et_high"] + ev["et_high"], 35, 60))),
        low_rr=int(round(np.clip(base["low_rr"] + scen["rr_low"] + ev["rr_low"], 4, 20))),
    )
    eff["exit_spo2"] = eff["low_spo2"] + st.session_state.hys_spo2
    eff["exit_map"] = eff["low_map"] + st.session_state.hys_map
    return eff

# ================== SCENARIO TARGETS (PHYSIOLOGIC) ==================
def scenario_targets_for_sim(base: Dict[str, float]) -> Dict[str, float]:
    name = getattr(st.session_state, "scenario_name", None) if "scenario_name" in st.session_state else None
    end = getattr(st.session_state, "scenario_end", 0.0) if "scenario_end" in st.session_state else 0.0
    active = (name is not None) and (time.time() <= end)
    if not active:
        return dict(HR=base["HR"], SpO2=base["SpO2"], MAP=base["MAP"],
                    EtCO2=base["EtCO2"], RR=base["RR"])

    t = dict(HR=base["HR"], SpO2=base["SpO2"], MAP=base["MAP"],
             EtCO2=base["EtCO2"], RR=base["RR"])

    if name == "Bleed":
        t["MAP"] = base["MAP"] - 15
        t["HR"] = base["HR"] + 15
        t["SpO2"] = base["SpO2"] - 1
    elif name == "Bronchospasm":
        t["SpO2"] = base["SpO2"] - 6
        t["EtCO2"] = base["EtCO2"] + 8
        t["RR"] = base["RR"] + 6
    elif name == "Vasodilation":
        t["MAP"] = base["MAP"] - 12
        t["HR"] = base["HR"] - 3
    elif name == "Pain/Light":
        t["HR"] = base["HR"] + 12
        t["MAP"] = base["MAP"] + 5
        t["RR"] = base["RR"] + 4
    elif name == "OB_Hemorrhage":
        t["MAP"] = base["MAP"] - 20
        t["HR"] = base["HR"] + 20
        t["SpO2"] = base["SpO2"] - 2
        t["RR"] = base["RR"] + 2
    elif name == "Sepsis_Laparotomy":
        t["MAP"] = base["MAP"] - 15
        t["HR"] = base["HR"] + 15
        t["EtCO2"] = base["EtCO2"] + 3
        t["RR"] = base["RR"] + 3
        t["SpO2"] = base["SpO2"] - 1
    elif name == "Thoracic_OneLung":
        t["SpO2"] = base["SpO2"] - 5
        t["EtCO2"] = base["EtCO2"] + 5
        t["RR"] = base["RR"] + 4
        t["HR"] = base["HR"] + 5
        t["MAP"] = base["MAP"] + 3
    elif name == "Craniotomy_Hypertensive":
        t["MAP"] = base["MAP"] + 15
        t["HR"] = base["HR"] + 10
    return t

# ================== SIMULATION ENGINE ==================
def __step(val, target, sigma, lo, hi):
    v = val + random.gauss(0, sigma) + (target - val) * random.uniform(0.02, 0.08)
    return max(lo, min(hi, v))

def _tick_live(sim_hz: int):
    for _ in range(sim_hz):
        base = st.session_state.sim_val
        targets = scenario_targets_for_sim(base)

        hr = __step(base["HR"], targets["HR"], 1.2, 30, 180)
        spo2 = __step(base["SpO2"], targets["SpO2"], 0.6, 70, 100)
        map_ = __step(base["MAP"], targets["MAP"], 1.5, 40, 120)
        et = __step(base["EtCO2"], targets["EtCO2"], 0.8, 20, 60)
        rr = __step(base["RR"], targets["RR"], 0.8, 6, 40)

        vaso = float(st.session_state.vaso_effect)
        fluid = float(st.session_state.fluid_effect)

        map_ += vaso * 0.5
        hr -= vaso * 0.1

        map_ += fluid * 0.25
        if hr > 90:
            hr -= fluid * 0.1

        hr = max(30, min(180, hr))
        spo2 = max(70, min(100, spo2))
        map_ = max(40, min(120, map_))
        et = max(20, min(60, et))
        rr = max(6, min(40, rr))

        if st.session_state.enable_noise and random.random() < (
            st.session_state.artifact_pct / 100.0
        ):
            pick = random.random()
            if pick < 0.25:
                hr += random.choice([-15, 15])
            elif pick < 0.50:
                spo2 += random.choice([-8, 8])
            elif pick < 0.75:
                map_ += random.choice([-10, 10])
            else:
                et += random.choice([-6, 6])

        st.session_state.vaso_effect *= 0.90
        st.session_state.fluid_effect *= 0.92

        st.session_state.sim_val = {
            "HR": hr,
            "SpO2": spo2,
            "MAP": map_,
            "EtCO2": et,
            "RR": rr,
        }
        st.session_state.sim_time += 1
        row = {
            "Time": len(st.session_state.history),
            "HR": int(round(hr)),
            "SpO2": int(round(spo2)),
            "MAP": int(round(map_)),
            "EtCO2": int(round(et)),
            "RR": int(round(rr)),
        }
        st.session_state.history.loc[len(st.session_state.history)] = row

def _tick_replay():
    df_r = st.session_state.replay_df
    if df_r is None:
        return
    i = st.session_state.replay_idx
    if i >= len(df_r):
        return
    step = int(st.session_state.replay_speed)
    end = min(i + step, len(df_r))
    chunk = df_r.iloc[i:end].copy()
    base = len(st.session_state.history)
    chunk["Time"] = np.arange(base, base + len(chunk))
    st.session_state.history = pd.concat(
        [st.session_state.history, chunk], ignore_index=True
    )
    st.session_state.replay_idx = end

# ================== PHYSIOLOGY STORY ==================
def physiology_story(df: pd.DataFrame, window_sec: int = 90, sim_hz: int = 5) -> str:
    if df is None or df.empty:
        return "Not enough data for physiologic pattern analysis."
    n_samples = window_sec * max(sim_hz, 1)
    window = df.tail(n_samples).copy()
    if window.empty:
        window = df.copy()

    def slope(series: pd.Series) -> float:
        if series is None or len(series) < 2:
            return 0.0
        return float(series.iloc[-1] - series.iloc[0]) / max(len(series), 1)

    def col(name: str) -> pd.Series:
        return window[name] if name in window.columns else pd.Series(dtype=float)

    map_s = slope(col("MAP"))
    hr_s = slope(col("HR"))
    spo2_s = slope(col("SpO2"))
    et_s = slope(col("EtCO2"))
    rr_s = slope(col("RR"))

    def arrow(v: float) -> str:
        return "↑" if v > 0.1 else ("↓" if v < -0.1 else "→")

    try:
        if "SpO2" in window.columns and "EtCO2" in window.columns and len(window) >= 3:
            corr_val = float(window["SpO2"].corr(window["EtCO2"]))
        else:
            corr_val = None
    except Exception:
        corr_val = None

    if corr_val is None:
        corr_txt = "insufficient data"
    elif corr_val > 0.2:
        corr_txt = f"positive (r≈{corr_val:.2f})"
    elif corr_val < -0.2:
        corr_txt = f"negative (r≈{corr_val:.2f})"
    else:
        corr_txt = f"weak (r≈{corr_val:.2f})"

    story = (
        f"Over the last {window_sec} seconds, MAP is {arrow(map_s)} ({map_s:.2f}/sample), "
        f"HR is {arrow(hr_s)} ({hr_s:.2f}/sample), SpO₂ is {arrow(spo2_s)} ({spo2_s:.2f}/sample), "
        f"EtCO₂ is {arrow(et_s)} ({et_s:.2f}/sample), and RR is {arrow(rr_s)} ({rr_s:.2f}/sample). "
        f"SpO₂–EtCO₂ correlation appears {corr_txt}. "
    )

    if map_s < -0.2:
        if "MAP" in window.columns:
            map_now = float(window["MAP"].iloc[-1])
            if map_now <= 65:
                story += "MAP is already below a typical safety target of 65 mmHg. "
            else:
                delta = map_now - 65.0
                samples_to_cross = delta / abs(map_s)
                seconds_to_cross = samples_to_cross / max(sim_hz, 1)
                if seconds_to_cross < 4:
                    story += "MAP may breach 65 mmHg imminently. "
                elif seconds_to_cross < 900:
                    story += f"If the trend continues, MAP may fall below 65 mmHg in ≈{seconds_to_cross:.0f} s. "

    if all(abs(x) < 0.1 for x in [map_s, hr_s, spo2_s, et_s, rr_s]):
        story += "Overall, physiology appears relatively stable with no rapid deterioration detected."
    else:
        story += "Overall, physiology is dynamic, with evolving trends that merit ongoing attention."

    return story

# ================== RISK SCORING (NEWS-LIKE) ==================
def _news_score_single(value: Optional[float], bands: List[Tuple[float, float, int]]) -> int:
    if value is None:
        return 0
    for lo, hi, score in bands:
        if lo <= value <= hi:
            return score
    return 0

def compute_composite_risk(row: pd.Series) -> Tuple[int, str]:
    if row is None or row.empty:
        return 0, "Insufficient data"

    rr = float(row.get("RR", np.nan))
    spo2 = float(row.get("SpO2", np.nan))
    map_ = float(row.get("MAP", np.nan))
    hr = float(row.get("HR", np.nan))
    etco2 = float(row.get("EtCO2", np.nan))

    rr_score = _news_score_single(rr, [
        (0, 8, 3), (9, 11, 1), (12, 20, 0), (21, 24, 2), (25, 100, 3)
    ])
    spo2_score = _news_score_single(spo2, [
        (0, 90, 3), (91, 93, 2), (94, 95, 1), (96, 100, 0)
    ])
    map_score = _news_score_single(map_, [
        (0, 60, 3), (61, 65, 2), (66, 75, 1), (76, 110, 0), (111, 120, 1), (121, 1000, 2)
    ])
    hr_score = _news_score_single(hr, [
        (0, 40, 3), (41, 50, 1), (51, 90, 0), (91, 110, 1), (111, 130, 2), (131, 300, 3)
    ])
    et_score = _news_score_single(etco2, [
        (0, 25, 2), (26, 50, 0), (51, 60, 2), (61, 999, 3)
    ])

    total = rr_score + spo2_score + map_score + hr_score + et_score

    if total <= 2:
        band = "Low risk"
    elif total <= 4:
        band = "Borderline risk"
    elif total <= 6:
        band = "Moderate risk"
    else:
        band = "High risk"

    return int(total), band

def risk_index_to_gauge(total: int) -> Tuple[float, str]:
    idx = max(0.0, min(1.0, total / 12.0))
    if total <= 2:
        label = "Stable"
    elif total <= 4:
        label = "Watch closely"
    elif total <= 6:
        label = "At risk"
    else:
        label = "Unstable"
    return idx, label

# ================== ANSWER ENGINE (LOCAL LANGUAGE) ==================
def answer_query(q: str, df: pd.DataFrame, sim_hz: int = 5) -> Dict[str, str]:
    if df is None or df.empty:
        return {
            "text": (
                "Short answer: There is not enough data yet to interpret physiology.\n\n"
                "Start the live stream or load a replay, then ask again."
            )
        }

    cols = _resolve_cols(df)
    if not cols["Time"]:
        return {
            "text": "Short answer: This stream has no time index, so it cannot be interpreted safely."
        }

    look_secs = 90
    keep_cols = [
        c for c in
        [cols["Time"], cols["MAP"], cols["HR"], cols["SpO2"], cols["EtCO2"], cols["RR"]]
        if c
    ]
    window = last_n(df[keep_cols], look_secs)
    if window is None or window.empty:
        window = df[keep_cols]

    def series_for(name: str) -> Optional[pd.Series]:
        c = cols.get(name)
        return window[c] if c and c in window.columns else None

    def last_safe(s: Optional[pd.Series]) -> Optional[float]:
        if s is None or s.empty:
            return None
        try:
            return float(s.iloc[-1])
        except Exception:
            return None

    def slope(s: Optional[pd.Series]) -> float:
        if s is None or len(s) < 2:
            return 0.0
        x = np.arange(len(s), dtype=float)
        y = s.to_numpy(dtype=float)
        xm, ym = x.mean(), y.mean()
        denom = ((x - xm) ** 2).sum()
        if denom == 0:
            return 0.0
        return float(((x - xm) * (y - ym)).sum() / denom)

    vals = {v: last_safe(series_for(v)) for v in ["MAP", "HR", "SpO2", "EtCO2", "RR"]}
    slopes = {v: slope(series_for(v)) for v in ["MAP", "HR", "SpO2", "EtCO2", "RR"]}

    score, band = compute_composite_risk(window.iloc[-1])
    idx, stability_label = risk_index_to_gauge(score)

    map_s, hr_s, spo2_s, et_s, rr_s = (
        slopes["MAP"],
        slopes["HR"],
        slopes["SpO2"],
        slopes["EtCO2"],
        slopes["RR"],
    )

    interpretations = []

    if map_s < -0.3 and hr_s > 0.3:
        interpretations.append(
            "MAP is falling while heart rate is rising, a pattern compatible with a compensatory response "
            "to relative hypovolemia or early bleeding."
        )
    if map_s < -0.3 and hr_s <= 0.1:
        interpretations.append(
            "MAP is declining without a strong heart rate response, which fits more with vasodilation or "
            "anesthetic drug effect than pure blood loss."
        )
    if spo2_s < -0.2 and et_s > 0.2:
        interpretations.append(
            "Oxygen saturation is drifting down while EtCO₂ is rising, consistent with hypoventilation or an "
            "evolving airway/ventilation problem."
        )
    if not interpretations:
        interpretations.append(
            "No single catastrophic pattern dominates, but the trajectory still requires attention over time."
        )

    q_low = (q or "").lower()
    if "stable" in q_low and ("is the" in q_low or "is patient" in q_low):
        if stability_label == "Stable":
            short = "Short answer: In this window the physiology is acceptably stable."
        elif stability_label in ["Watch closely", "At risk"]:
            short = (
                "Short answer: Physiology is borderline. It is not frankly unstable, "
                "but it does not meet a comfortable definition of stable."
            )
        else:
            short = "Short answer: The patient would be described as unstable in this window."
    elif any(w in q_low for w in ["why", "cause", "reason"]):
        short = "Short answer: Here is the most likely physiologic explanation for the current pattern."
    elif any(w in q_low for w in ["what", "status", "trend", "happening"]):
        short = "Short answer: Here is the current physiological status based on the recent data."
    else:
        short = "Short answer: Here is how the current physiology looks over the last minute or so."

    next_steps = []
    if map_s < -0.3 and vals["MAP"] is not None and vals["MAP"] < 65:
        next_steps.append(
            "Reassess volume status, anesthetic depth, and vasopressor support to keep MAP around or above 65 mmHg."
        )
    if spo2_s < -0.2:
        next_steps.append(
            "Confirm airway patency, FiO₂ source, and ventilation settings; rule out probe artifact before escalating."
        )
    if et_s > 0.3:
        next_steps.append(
            "Evaluate minute ventilation and circuit; consider increasing tidal volume or rate if clinically appropriate."
        )
    if not next_steps:
        next_steps.append("Continue close monitoring and re-evaluate the trend in 30–60 seconds.")

    risks = []
    if vals["MAP"] is not None and vals["MAP"] < 65:
        risks.append("Sustained hypotension risks reduced organ perfusion.")
    if vals["SpO2"] is not None and vals["SpO2"] < 92:
        risks.append("Ongoing desaturation risks tissue hypoxia if uncorrected.")
    if not risks:
        risks.append("No immediate high-risk trajectory is dominant, but vigilance is still required.")

    status_lines = []
    if vals["MAP"] is not None:
        status_lines.append(f"MAP ≈ {int(round(vals['MAP']))} mmHg")
    if vals["HR"] is not None:
        status_lines.append(f"HR ≈ {int(round(vals['HR']))} bpm")
    if vals["SpO2"] is not None:
        status_lines.append(f"SpO₂ ≈ {int(round(vals['SpO2']))}%")
    if vals["EtCO2"] is not None:
        status_lines.append(f"EtCO₂ ≈ {int(round(vals['EtCO2']))} mmHg")
    if vals["RR"] is not None:
        status_lines.append(f"RR ≈ {int(round(vals['RR']))} breaths/min")

    interpretations_block = "- " + "\n- ".join(interpretations)
    next_steps_block = "- " + "\n- ".join(next_steps)
    risks_block = "- " + "\n- ".join(risks)

    text = f"""
Short answer: {short.replace("Short answer: ", "")}

Clinical status:
- Composite risk score: {score} ({stability_label})
- Key vitals: {", ".join(status_lines)}

Physiologic interpretation:
{interpretations_block}

Recommended next steps (for teaching only, not prescriptive):
{next_steps_block}

Anticipated risks:
{risks_block}

Educational note:
This explanation is generated for pattern-recognition training only and is not a clinical decision or order.
    """
    return {"text": textwrap.dedent(text).strip()}

# ================== 12s MICRO-CHARTS + FORECASTING ==================
def _forecast_points(df_: pd.DataFrame, field: str, horizon_sec: int, sim_hz: int) -> Optional[pd.DataFrame]:
    if df_ is None or df_.empty or field not in df_.columns or "Time" not in df_.columns:
        return None
    base = last_n(df_[["Time", field]], 30)
    if base is None or len(base) < 5:
        return None
    x = base["Time"].to_numpy(dtype=float)
    y = base[field].to_numpy(dtype=float)
    if np.allclose(y.max(), y.min()):
        return None
    try:
        m, b = np.polyfit(x, y, 1)
    except Exception:
        return None
    last_t = float(df_["Time"].iloc[-1])
    steps = max(1, horizon_sec * max(sim_hz, 1))
    future_t = np.arange(last_t + 1, last_t + 1 + steps)
    future_v = m * future_t + b
    forecast_df = pd.DataFrame({"Time": future_t, field: future_v})
    return forecast_df

VITAL_COLORS = {
    "HR": "#F59E0B",
    "MAP": "#EF4444",
    "SpO2": "#3B82F6",
    "EtCO2": "#10B981",
    "RR": "#A855F7",
}

def main_chart_with_forecast(df_: pd.DataFrame, field: str, title: str, sim_hz: int) -> Optional[alt.Chart]:
    if df_ is None or df_.empty or field not in df_.columns:
        return None

    base = last_n(df_[["Time", field]], 120)
    if base is None or base.empty:
        base = df_[["Time", field]]

    color = VITAL_COLORS.get(field, "#C8CCD2")

    base_chart = alt.Chart(base).mark_line(point=False, strokeWidth=2).encode(
        x=alt.X("Time:Q", axis=alt.Axis(title="Time (index)")),
        y=alt.Y(f"{field}:Q", axis=alt.Axis(title=field)),
        color=alt.value(color),
        tooltip=["Time", field],
    ).properties(title=title, height=200)

    forecast_df = _forecast_points(df_, field, horizon_sec=60, sim_hz=sim_hz)
    if forecast_df is not None:
        forecast_chart = alt.Chart(forecast_df).mark_line(
            strokeDash=[4, 4], strokeWidth=1.5
        ).encode(
            x="Time:Q",
            y=f"{field}:Q",
            color=alt.value(color),
            tooltip=["Time", field],
        )
    else:
        forecast_chart = None

    ev_df = events_dataframe()
    event_layer = None
    if not ev_df.empty:
        event_layer = alt.Chart(ev_df).mark_rule(strokeWidth=1.5).encode(
            x="Time:Q",
            color=alt.Color(
                "type:N",
                scale=alt.Scale(
                    domain=["Incision", "Fluids", "Vasopressor", "Position", "Other"],
                    range=[
                        "#F97316",
                        "#22C55E",
                        "#EC4899",
                        "#8B5CF6",
                        "#E5E7EB",
                    ],
                ),
                legend=alt.Legend(title="Events"),
            ),
            tooltip=["name", "type", "Time"],
        )

    chart = base_chart
    if forecast_chart is not None:
        chart = chart + forecast_chart
    if event_layer is not None:
        chart = chart + event_layer

    chart = chart.configure_axis(
        labelColor="#C8CCD2",
        titleColor="#C8CCD2",
        gridColor="#2A2E36",
    ).properties(background="transparent")

    return chart

def interactive_12s_chart(df_: pd.DataFrame, field: str, title: str) -> Optional[alt.Chart]:
    if df_ is None or df_.empty or field not in df_.columns:
        return None
    w = last_n(df_, 12)
    if w is None or w.empty:
        w = df_.iloc[-min(12, len(df_)):]
    chart_df = w[["Time", field]].copy()
    color = VITAL_COLORS.get(field, "#C8CCD2")

    base = (
        alt.Chart(chart_df)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X(
                "Time:Q",
                axis=alt.Axis(title="(last 12s)", labelColor="#C8CCD2", titleColor="#C8CCD2"),
            ),
            y=alt.Y(
                f"{field}:Q",
                axis=alt.Axis(title=field, labelColor="#C8CCD2", titleColor="#C8CCD2"),
            ),
            color=alt.value(color),
            tooltip=["Time", field],
        )
        .properties(title=title, height=160, background="transparent")
        .configure_axis(gridColor="#2A2E36")
    )

    ev_df = events_dataframe()
    if not ev_df.empty:
        t_min, t_max = chart_df["Time"].min(), chart_df["Time"].max()
        ev_df_w = ev_df[(ev_df["Time"] >= t_min) & (ev_df["Time"] <= t_max)]
        if not ev_df_w.empty:
            ev_layer = alt.Chart(ev_df_w).mark_rule(strokeWidth=1.5).encode(
                x="Time:Q",
                color=alt.Color(
                    "type:N",
                    scale=alt.Scale(
                        domain=["Incision", "Fluids", "Vasopressor", "Position", "Other"],
                        range=[
                            "#F97316",
                            "#22C55E",
                            "#EC4899",
                            "#8B5CF6",
                            "#E5E7EB",
                        ],
                    ),
                    legend=None,
                ),
                tooltip=["name", "type", "Time"],
            )
            base = base + ev_layer

    return base

# ================== ARTIFACT DETECTION ==================
def is_artifact_series(series: pd.Series) -> bool:
    if series is None or len(series) < 6:
        return False
    recent = series.iloc[-6:]
    diffs = np.abs(np.diff(recent.values))
    if len(diffs) == 0:
        return False
    return diffs[-1] > (np.std(recent.values) * 4 + 4)

def recent_artifact(df_: pd.DataFrame) -> dict:
    if df_ is None or df_.empty:
        return {"HR": False, "SpO2": False, "MAP": False, "EtCO2": False, "RR": False}
    return {
        "HR": is_artifact_series(df_["HR"]),
        "SpO2": is_artifact_series(df_["SpO2"]),
        "MAP": is_artifact_series(df_["MAP"]),
        "EtCO2": is_artifact_series(df_["EtCO2"]),
        "RR": is_artifact_series(df_["RR"]),
    }

def persistent_low(df_, signal, thresh, sec):
    w = last_n(df_, sec)
    return (
        (w is not None)
        and (len(w) >= max(3, min(sec, len(w))))
        and (w[signal].min() < thresh)
    )

def persistent_high(df_, signal, thresh, sec):
    w = last_n(df_, sec)
    return (
        (w is not None)
        and (len(w) >= max(3, min(sec, len(w))))
        and (w[signal].max() > thresh)
    )

# ================== TOP SUGGESTED ACTION (LIMITED) ==================
def top_suggested_action(df_: pd.DataFrame) -> Dict[str, str]:
    if df_ is None or df_.empty:
        return {}
    w = last_n(df_, 20)
    if w is None or w.empty:
        w = df_

    def slope_local(s: pd.Series) -> float:
        if s is None or len(s) < 2:
            return 0.0
        x = np.arange(len(s), dtype=float)
        y = s.to_numpy(dtype=float)
        xm, ym = x.mean(), y.mean()
        denom = ((x - xm) ** 2).sum()
        if denom == 0:
            return 0.0
        return float(((x - xm) * (y - ym)).sum() / denom)

    stats = {}
    for k in ["MAP", "HR", "SpO2", "EtCO2", "RR"]:
        if k in w.columns:
            stats[k] = {"mean": float(np.mean(w[k])), "trend": slope_local(w[k])}
    if not stats:
        return {}

    action = "Observe."
    rationale = "Signals stable within current educational thresholds."
    confidence = 0.5
    anticipatory = "No major deterioration projected in the immediate window."
    alts = ["Reassess in 30–60 seconds", "Confirm sensor positions"]
    within = "Within your current safety rails."

    if "MAP" in stats and "HR" in stats:
        if stats["MAP"]["mean"] < 65 or stats["MAP"]["trend"] < -0.3:
            if stats["HR"]["trend"] > 0.2:
                action = "Evaluate volume status and consider fluid resuscitation if appropriate."
                rationale = "Low or falling MAP with rising HR suggests a compensatory hypovolemic pattern."
                confidence = 0.7
                anticipatory = "Without support, blood pressure is likely to remain below target."
                alts = [
                    "Assess the need for vasopressor support",
                    "Review anesthetic depth and recent drug dosing",
                ]
            else:
                action = "Consider modest vasopressor support and review anesthetic depth."
                rationale = "Low or falling MAP with limited HR response suggests vasodilation or drug effect."
                confidence = 0.65
                anticipatory = "Hypotension may persist without adjustment."
                alts = [
                    "Check volume status and bleeding risk",
                    "Reduce anesthetic dose if clinically safe",
                ]

    if "SpO2" in stats and "EtCO2" in stats:
        if stats["SpO2"]["trend"] < -0.2 and stats["EtCO2"]["trend"] > 0.2:
            action = "Check airway and ventilation; increase support if needed."
            rationale = "Falling oxygen saturation with rising EtCO₂ is compatible with hypoventilation or airway compromise."
            confidence = 0.75
            anticipatory = "Oxygenation may worsen if ventilation is not corrected."
            alts = [
                "Re-seat the oximeter probe to rule out artifact",
                "Increase minute ventilation",
                "Inspect the circuit and ETT position",
            ]

    return {
        "Suggested": action,
        "Rationale": rationale,
        "Within": within,
        "Confidence": f"{int(round(confidence * 100))}%",
        "Anticipatory": anticipatory,
        "Alternatives": alts,
    }

# ================== CURRENT STATE MODEL (PATTERN CLASSIFIER) ==================
def current_state_model(df: pd.DataFrame, sim_hz: int = 5) -> dict:
    """
    Lightweight rule-based classifier that summarizes what is most likely happening
    in the last ~60–90 seconds, with a confidence estimate and supporting signals.

    Returns a dict with:
      - label: short human-readable state description
      - confidence: float in [0,1]
      - supporting_signals: list of bullet strings explaining why
      - risk_score, risk_band: reuse NEWS-like composite risk for context
    """
    result = {
        "label": "No data available",
        "confidence": 0.0,
        "supporting_signals": [],
        "risk_score": 0,
        "risk_band": "Insufficient data",
    }

    if df is None or df.empty:
        return result

    # Use the last ~90 seconds or all available data if shorter
    window_sec = 90
    n_samples = window_sec * max(sim_hz, 1)
    window = df.tail(n_samples).copy()
    if window.empty:
        window = df.copy()

    latest = window.iloc[-1]
    risk_score, risk_band = compute_composite_risk(latest)
    result["risk_score"] = int(risk_score)
    result["risk_band"] = risk_band

    def slope(series: pd.Series) -> float:
        if series is None or len(series) < 2:
            return 0.0
        x = np.arange(len(series), dtype=float)
        y = series.to_numpy(dtype=float)
        xm, ym = x.mean(), y.mean()
        denom = ((x - xm) ** 2).sum()
        if denom == 0:
            return 0.0
        return float(((x - xm) * (y - ym)).sum() / denom)

    def change_string(name: str) -> str:
        if name not in window.columns or len(window[name]) < 2:
            return ""
        start = float(window[name].iloc[0])
        end = float(window[name].iloc[-1])
        dt_samples = float(window["Time"].iloc[-1] - window["Time"].iloc[0]) if "Time" in window.columns else len(window) - 1
        dt_sec = dt_samples / max(sim_hz, 1)
        dt_sec = max(1.0, dt_sec)
        return f"{name} {int(round(start))} → {int(round(end))} over ~{int(round(dt_sec))} s"

    vals = {
        "MAP": float(latest.get("MAP", np.nan)) if "MAP" in latest else np.nan,
        "HR": float(latest.get("HR", np.nan)) if "HR" in latest else np.nan,
        "SpO2": float(latest.get("SpO2", np.nan)) if "SpO2" in latest else np.nan,
        "EtCO2": float(latest.get("EtCO2", np.nan)) if "EtCO2" in latest else np.nan,
        "RR": float(latest.get("RR", np.nan)) if "RR" in latest else np.nan,
    }

    slopes = {
        "MAP": slope(window["MAP"]) if "MAP" in window.columns else 0.0,
        "HR": slope(window["HR"]) if "HR" in window.columns else 0.0,
        "SpO2": slope(window["SpO2"]) if "SpO2" in window.columns else 0.0,
        "EtCO2": slope(window["EtCO2"]) if "EtCO2" in window.columns else 0.0,
        "RR": slope(window["RR"]) if "RR" in window.columns else 0.0,
    }

    label = "Physiology relatively stable"
    confidence = 0.4
    supporting = []

    # Pattern 1: hypotension with compensatory tachycardia
    if not np.isnan(vals["MAP"]) and vals["MAP"] < 65 and slopes["MAP"] < -0.3 and slopes["HR"] > 0.2:
        label = "Hypotension with compensatory tachycardia (possible bleeding or hypovolemia)"
        confidence = 0.75
        supporting.append("MAP is low and falling while HR is rising, suggesting a compensatory response.")
        cs_map = change_string("MAP")
        cs_hr = change_string("HR")
        if cs_map:
            supporting.append(cs_map)
        if cs_hr:
            supporting.append(cs_hr)

    # Pattern 2: hypotension with limited HR response (vasodilation / drug effect)
    elif not np.isnan(vals["MAP"]) and vals["MAP"] < 65 and slopes["MAP"] < -0.3 and slopes["HR"] <= 0.1:
        label = "Hypotension with limited heart-rate response (possible vasodilation or drug effect)"
        confidence = 0.7
        supporting.append("MAP is low and drifting down without a strong HR response.")
        cs_map = change_string("MAP")
        if cs_map:
            supporting.append(cs_map)

    # Pattern 3: worsening oxygenation with rising CO2
    elif (not np.isnan(vals["SpO2"]) and vals["SpO2"] < 92 and slopes["SpO2"] < -0.2
          and slopes["EtCO2"] > 0.2):
        label = "Worsening oxygenation with rising CO₂ (possible hypoventilation or airway problem)"
        confidence = 0.8
        supporting.append("SpO₂ is drifting down while EtCO₂ is rising, consistent with hypoventilation or airway compromise.")
        cs_spo2 = change_string("SpO2")
        cs_et = change_string("EtCO2")
        if cs_spo2:
            supporting.append(cs_spo2)
        if cs_et:
            supporting.append(cs_et)

    # Pattern 4: recovery after prior instability
    elif slopes["MAP"] > 0.2 or slopes["SpO2"] > 0.2:
        label = "Recovering after recent instability"
        confidence = 0.65
        supporting.append("Key signals are trending up from previously lower levels, suggesting early recovery.")
        cs_map = change_string("MAP")
        cs_spo2 = change_string("SpO2")
        if cs_map:
            supporting.append(cs_map)
        if cs_spo2:
            supporting.append(cs_spo2)

    else:
        label = "No dominant instability pattern detected"
        confidence = 0.45
        supporting.append("Slopes of MAP, HR, SpO₂, EtCO₂, and RR are small in this window.")

    if risk_score >= 6:
        confidence = min(1.0, confidence + 0.1)
    elif risk_score <= 2:
        confidence = max(0.3, confidence - 0.05)

    result["label"] = label
    result["confidence"] = float(round(confidence, 2))
    result["supporting_signals"] = supporting

    return result

# ================== TREND HELPERS FOR COCKPIT ==================
def compute_trends(df_: pd.DataFrame) -> Dict[str, float]:
    if df_ is None or df_.empty:
        return {k: 0.0 for k in ["MAP", "HR", "SpO2", "EtCO2", "RR"]}
    w = last_n(df_, 60)
    if w is None or w.empty:
        w = df_
    trends = {}
    for col in ["MAP", "HR", "SpO2", "EtCO2", "RR"]:
        if col not in w.columns or len(w[col]) < 2:
            trends[col] = 0.0
        else:
            vals = w[col].astype(float).values
            trends[col] = float(vals[-1] - vals[0]) / max(len(vals), 1)
    return trends

def arrow_for_trend(v: float) -> str:
    if v > 0.3:
        return "↑"
    if v < -0.3:
        return "↓"
    return "→"

# ================== LIVE SUMMARY BLOCK ==================
def live_summary_block(df_: pd.DataFrame) -> str:
    if df_ is None or df_.empty:
        return "No data yet to summarize."
    window = df_.tail(min(len(df_), 90))

    def slope_of(col: str):
        if col not in window.columns or len(window[col]) < 2:
            return 0.0
        vals = window[col].astype(float).values
        if np.allclose(vals.max(), vals.min()):
            return 0.0
        return float(vals[-1] - vals[0]) / max(len(vals), 1)

    s_map, s_hr, s_spo2, s_et, s_rr = map(
        slope_of, ["MAP", "HR", "SpO2", "EtCO2", "RR"]
    )
    parts = []
    parts.append(
        "MAP falling — possible hypoperfusion."
        if s_map < -0.3
        else ("MAP rising — recovery or vasopressor response." if s_map > 0.3 else "MAP relatively stable.")
    )
    parts.append(
        "HR increasing — possible pain/stress or compensation."
        if s_hr > 0.3
        else ("HR decreasing — drug or depth effect." if s_hr < -0.3 else "HR stable.")
    )
    parts.append(
        "SpO₂ trending down — check airway, oxygenation, or probe."
        if s_spo2 < -0.2
        else ("SpO₂ improving." if s_spo2 > 0.2 else "SpO₂ stable.")
    )
    parts.append(
        "EtCO₂ falling — hyperventilation or perfusion drop."
        if s_et < -0.3
        else ("EtCO₂ rising — hypoventilation or CO₂ retention." if s_et > 0.3 else "EtCO₂ stable.")
    )
    parts.append(
        "RR rising — compensatory hyperventilation."
        if s_rr > 0.3
        else ("RR decreasing — sedation or airway depression." if s_rr < -0.3 else "RR stable.")
    )
    return " · ".join(parts)

# ================== DATA QUALITY HELPERS ==================
def artifact_rate(series: pd.Series, window=60) -> float:
    if series is None or len(series) < 8:
        return 0.0
    w = series.iloc[-min(window, len(series)):]
    jumps = np.abs(np.diff(w.values))
    thr = np.std(w.values) * 4 + 4
    return float(np.mean(jumps > thr))

def dropout_rate(series: pd.Series, window=60) -> float:
    if series is None or len(series) < 2:
        return 0.0
    w = series.iloc[-min(window, len(series)):]
    return float(w.isna().mean())

def flatline_rate(series: pd.Series, window=60) -> float:
    if series is None or len(series) < 4:
        return 0.0
    w = series.iloc[-min(window, len(series)):]
    diffs = np.abs(np.diff(w.values))
    return float(np.mean(diffs < 1e-6))

# ================== TOP STRIP: BANNER ==================
st.markdown(
    """
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;">
        <div class="halo-top-pill">⚠ Demo build · Educational only · Not for clinical use</div>
        <div class="halo-top-pill">HALO — Surgical Physiology Cockpit</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ================== SIDEBAR ==================
with st.sidebar:
    st.header("Mode & Patient")
    st.session_state.mode = st.radio(
        "Input source",
        ["Live", "Replay"],
        index=0 if st.session_state.mode == "Live" else 1,
        key="mode_radio",
    )

    st.subheader("Patient")
    st.session_state.age = st.number_input("Age (years)", min_value=0, max_value=120, value=int(st.session_state.age), step=1)
    st.session_state.sex = st.selectbox("Sex", ["Unknown", "Male", "Female"], index=["Unknown", "Male", "Female"].index(st.session_state.sex))
    st.session_state.asa_class = st.selectbox("ASA class", ["I", "II", "III", "IV", "V"], index=["I", "II", "III", "IV", "V"].index(str(st.session_state.asa_class)))
    st.session_state.bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=float(st.session_state.bmi), step=0.1)

    case_options = [
        "Elective laparotomy",
        "Emergency laparotomy",
        "Trauma laparotomy",
        "Cesarean section",
        "Craniotomy",
        "Thoracic surgery",
        "Orthopedic surgery",
        "Other",
    ]
    try:
        default_idx = case_options.index(st.session_state.case_type)
    except ValueError:
        default_idx = 0

    selected_case = st.selectbox(
        "Case Type",
        case_options,
        index=default_idx,
        help="Select the broad case category for this simulated patient.",
    )
    if selected_case == "Other":
        custom = st.text_input("Custom Case Description", value=st.session_state.case_type)
        st.session_state.case_type = custom or selected_case
    else:
        st.session_state.case_type = selected_case

    st.session_state.emergent = st.checkbox(
        "Emergency case",
        value=st.session_state.emergent,
        help="Check if this case is being managed as an emergency / urgent case rather than elective.",
    )

    if st.session_state.mode == "Replay":
        st.subheader("Replay input")
        up = st.file_uploader("Upload CSV (Time, HR, SpO2, MAP, EtCO2, RR)", type=["csv"], key="replay_csv")
        if up is not None:
            try:
                df_up = pd.read_csv(up)
                if "Time" not in df_up.columns:
                    df_up["Time"] = np.arange(len(df_up))
                st.session_state.replay_df = df_up
                st.session_state.replay_idx = 0
                st.success("Replay file loaded.")
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
        st.slider(
            "Replay speed (rows / tick)",
            1,
            20,
            int(st.session_state.get("replay_speed", 1)),
            key="replay_speed",
        )

    show_details = (
        True
        if not DEMO_MODE
        else st.checkbox("Show researcher details", value=False, key="show_details")
    )

    if show_details:
        st.header("Thresholds")
        st.slider("Low SpO₂ (%)", 85, 96, int(st.session_state.low_spo2), key="low_spo2")
        st.slider("Tachycardia HR (bpm)", 90, 160, int(st.session_state.tachy_hr), key="tachy_hr")
        st.slider("Low MAP (mmHg)", 50, 80, int(st.session_state.low_map), key="low_map")

        st.subheader("Respiratory")
        st.slider("High EtCO₂ (mmHg)", 40, 60, int(st.session_state.high_et), key="high_et")
        st.slider("Low EtCO₂ (mmHg)", 20, 40, int(st.session_state.low_et), key="low_et")
        st.slider("Low RR (bpm)", 4, 20, int(st.session_state.low_rr), key="low_rr")
        st.slider("High RR (bpm)", 18, 40, int(st.session_state.high_rr), key="high_rr")

        st.header("Persistence (sec)")
        st.slider("SpO₂ persistence", 5, 30, int(st.session_state.win_spo2), key="win_spo2")
        st.slider("HR persistence", 5, 30, int(st.session_state.win_hr), key="win_hr")
        st.slider("MAP persistence", 5, 30, int(st.session_state.win_map), key="win_map")
        st.slider("Resp persistence", 5, 30, int(st.session_state.win_resp), key="win_resp")

        st.header("Hysteresis & cooldown")
        st.slider("SpO₂ hysteresis (+%)", 1, 6, int(st.session_state.hys_spo2), key="hys_spo2")
        st.slider("MAP hysteresis (+mmHg)", 2, 12, int(st.session_state.hys_map), key="hys_map")
        st.slider("Alarm cooldown (s)", 0, 120, int(st.session_state.cooldown), key="cooldown")

        st.header("Noise / artifacts (live)")
        st.checkbox("Inject artifact / noise", value=st.session_state.enable_noise, key="enable_noise")
        st.slider("Artifact chance (%)", 0, 20, int(st.session_state.artifact_pct), key="artifact_pct")

    st.divider()
    if st.button("Reset buffers", key="reset_btn"):
        st.session_state.history = pd.DataFrame(columns=["Time", "HR", "SpO2", "MAP", "EtCO2", "RR"])
        st.session_state.replay_idx = 0
        st.session_state.events = []
        st.session_state.audit = []
        st.session_state.sim_time = 0
        st.session_state.vaso_effect = 0.0
        st.session_state.fluid_effect = 0.0
        st.success("Simulation history cleared.")

    st.header("Surgery scenarios")
    st.caption("Select a predefined teaching scenario. Each runs for about 60 seconds in simulation.")

    scenario_descriptions = {
        "Bleed": "Intraoperative bleeding with falling MAP and rising HR.",
        "Bronchospasm": "Airway resistance increase, desaturation with EtCO₂ changes.",
        "Vasodilation": "Drug-induced vasodilation with hypotension.",
        "Pain/Light": "Inadequate anesthesia with tachycardia and hypertension.",
    }

    scenario_buttons = [
        ("Bleed", "Bleed"),
        ("Bronchospasm", "Bronchospasm"),
        ("Vasodilation", "Vasodilation"),
        ("Pain / Light", "Pain/Light"),
    ]

    if "scenario_name" not in st.session_state:
        st.session_state.scenario_name = None
    if "scenario_end" not in st.session_state:
        st.session_state.scenario_end = 0.0

    for label, name in scenario_buttons:
        if st.button(label, key=f"sc_{name}"):
            st.session_state.scenario_name = name
            st.session_state.scenario_end = time.time() + 60

    if st.button("End scenario", key="sc_end"):
        st.session_state.scenario_name = None
        st.session_state.scenario_end = 0.0

    if st.session_state.scenario_name:
        desc = scenario_descriptions.get(st.session_state.scenario_name, "Active teaching scenario.")
        st.info(f"Active scenario: {st.session_state.scenario_name} — {desc}")

    st.header("Events")
    if st.button("Mark: Incision", key="mark_incision"):
        st.session_state.events.append({"t": len(st.session_state.history), "name": "Incision"})
    if st.button("Mark: Position change", key="mark_pos"):
        st.session_state.events.append({"t": len(st.session_state.history), "name": "Position change"})
    if st.button("Fluids 250 mL", key="mark_fluid"):
        st.session_state.events.append({"t": len(st.session_state.history), "name": "Fluids 250 mL"})
        st.session_state.fluid_effect = float(st.session_state.fluid_effect) + 5.0
    if st.button("Vasopressor bolus", key="mark_press"):
        st.session_state.events.append({"t": len(st.session_state.history), "name": "Vasopressor"})
        st.session_state.vaso_effect = float(st.session_state.vaso_effect) + 8.0

# ================== TICK (SIM OR REPLAY) ==================
if st.session_state.running:
    if st.session_state.mode == "Live":
        _tick_live(int(st.session_state.sim_hz))
    else:
        _tick_replay()

df = st.session_state.history
latest_row = df.iloc[-1] if not df.empty else None
risk_score, risk_band = compute_composite_risk(latest_row) if latest_row is not None else (0, "Insufficient data")
gauge_idx, gauge_label = risk_index_to_gauge(risk_score)
arts = recent_artifact(df) if not df.empty else {"HR": False, "SpO2": False, "MAP": False, "EtCO2": False, "RR": False}
trends = compute_trends(df) if not df.empty else {"MAP": 0.0, "HR": 0.0, "SpO2": 0.0, "EtCO2": 0.0, "RR": 0.0}
eff = effective_thresholds()

# ================== TOP STRIP ROW (HALO + PATIENT + CONTROL) ==================
top_c1, top_c2, top_c3 = st.columns([1.6, 1.2, 1.4])

with top_c1:
    st.markdown("#### HALO — Live Case")
    st.markdown(
        f"""
        <div class="halo-card-soft">
            <div style="font-weight:700;color:var(--halo-blue);margin-bottom:4px;">Simulated patient</div>
            <div style="font-size:0.9rem;">
                Age: {st.session_state.age} · Sex: {st.session_state.sex}<br>
                ASA class: {st.session_state.asa_class} · BMI: {st.session_state.bmi:.1f}<br>
                Case: {st.session_state.case_type}<br>
                Status: {"<span style='color:#E05A68;'>Emergency case</span>" if st.session_state.emergent else "Elective case"}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with top_c2:
    st.markdown("#### Stability index")
    width_pct = int(round(gauge_idx * 100))
    st.markdown(
        f"""
        <div class="halo-card-soft">
            <div style="display:flex;justify-content:space-between;margin-bottom:4px;font-size:0.9rem;">
                <span>{gauge_label}</span>
                <span>Score: {risk_score}</span>
            </div>
            <div class="halo-gauge-outer">
                <div class="halo-gauge-inner" style="width:{width_pct}%;"></div>
            </div>
            <div style="margin-top:6px;">
                <span style="font-size:0.78rem;border-radius:999px;border:1px solid var(--halo-border);padding:2px 8px;">
                    {risk_band}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with top_c3:
    st.markdown("#### Run control")
    c_status1, c_status2 = st.columns(2)
    if c_status1.button("▶ Start", key="btn_start"):
        st.session_state.running = True
    if c_status2.button("■ Stop", key="btn_stop"):
        st.session_state.running = False

    st.slider(
        "Samples / second",
        1,
        10,
        int(st.session_state.get("sim_hz", 5)),
        key="sim_hz",
    )

    mode_text = f"Mode: {st.session_state.mode} · {'Running' if st.session_state.running else 'Stopped'}"
    st.markdown(
        f"<div class='halo-top-pill' style='margin-top:6px;'>{mode_text} · Samples: {len(df)}</div>",
        unsafe_allow_html=True,
    )

# ================== TABS ==================
tab_trends, tab_assistant, tab_dq, tab_audit = st.tabs(
    ["Trends & cockpit", "Assistant", "Data quality", "Audit & export"]
)

# ================== TAB 1: TRENDS & COCKPIT ==================
with tab_trends:
    st.subheader("Primary vitals (with 60s forecast)")

    # vital legend
    st.markdown(
        """
        <div class="halo-legend">
          <div class="halo-legend-item">
            <div class="halo-legend-swatch" style="background:#F59E0B;"></div>
            <span>Heart rate</span>
          </div>
          <div class="halo-legend-item">
            <div class="halo-legend-swatch" style="background:#EF4444;"></div>
            <span>MAP</span>
          </div>
          <div class="halo-legend-item">
            <div class="halo-legend-swatch" style="background:#3B82F6;"></div>
            <span>SpO₂</span>
          </div>
          <div class="halo-legend-item">
            <div class="halo-legend-swatch" style="background:#10B981;"></div>
            <span>EtCO₂</span>
          </div>
          <div class="halo-legend-item">
            <div class="halo-legend-swatch" style="background:#A855F7;"></div>
            <span>Respiratory rate</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not df.empty:
        mc1, mc2, mc3 = st.columns(3)
        sim_hz_current = int(st.session_state.get("sim_hz", 5))

        ch_hr = main_chart_with_forecast(df, "HR", "Heart Rate (bpm)", sim_hz_current)
        ch_map = main_chart_with_forecast(df, "MAP", "MAP (mmHg)", sim_hz_current)
        ch_spo2 = main_chart_with_forecast(df, "SpO2", "SpO₂ (%)", sim_hz_current)

        if ch_hr is not None:
            mc1.altair_chart(ch_hr, use_container_width=True)
        if ch_map is not None:
            mc2.altair_chart(ch_map, use_container_width=True)
        if ch_spo2 is not None:
            mc3.altair_chart(ch_spo2, use_container_width=True)
    else:
        st.info("Click Start (Live) or upload a CSV (Replay) to begin streaming vitals.")

    if not df.empty:
        st.subheader("Ventilation")
        sc1, sc2 = st.columns(2)
        ch_et = main_chart_with_forecast(df, "EtCO2", "EtCO₂ (mmHg)", int(st.session_state.sim_hz))
        ch_rr = main_chart_with_forecast(df, "RR", "Respiratory Rate (bpm)", int(st.session_state.sim_hz))
        if ch_et is not None:
            sc1.altair_chart(ch_et, use_container_width=True)
        if ch_rr is not None:
            sc2.altair_chart(ch_rr, use_container_width=True)

        sim_hz_current = int(st.session_state.get("sim_hz", 5))
        summary_md = live_summary_block(df)
        phys_narr = physiology_story(df, window_sec=90, sim_hz=sim_hz_current)
        st.markdown(
            f"""
            <div class="halo-summary-card">
                <div style="font-weight:800;color:var(--halo-blue);font-size:1.0rem;margin-bottom:4px;">
                    Trajectory & risk (last 60–90 s)
                </div>
                <div style="margin-bottom:4px;">{summary_md}</div>
                <div style="font-weight:700;color:var(--halo-text-main);margin-top:6px;margin-bottom:2px;">
                    Physiology narrative
                </div>
                <div>{phys_narr}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # current state model card (prediction + confidence)
        state = current_state_model(df, sim_hz_current)
        support_items = ""
        if state["supporting_signals"]:
            lis = "".join(f"<li>{s}</li>" for s in state["supporting_signals"])
            support_items = f"<ul style='margin-top:4px;margin-left:18px;'>{lis}</ul>"
        else:
            support_items = "<span style='color:var(--halo-text-muted);'>No strong directional signals in this window.</span>"

        st.markdown(
            f"""
            <div class="halo-summary-card" style="margin-top:6px;">
                <div style="font-weight:800;color:var(--halo-blue);font-size:1.0rem;margin-bottom:4px;">
                    Current state model
                </div>
                <div style="margin-bottom:4px;">
                    <b>{state['label']}</b>
                </div>
                <div style="font-size:0.9rem;margin-bottom:4px;">
                    Confidence: <b>{int(state['confidence']*100)}%</b> · Composite risk score: {state['risk_score']} ({state['risk_band']})
                </div>
                <div style="font-size:0.9rem;">
                    Key supporting signals:
                    {support_items}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # alarm panel + microcharts (more detailed)
        st.subheader("Alarm panel")

        alerts_trend: List[dict] = []
        hypox_t = tachy_t = hypot_t = hypercap_t = hypovent_t = False

        hypox_t = (not arts["SpO2"]) and persistent_low(
            df, "SpO2", eff["low_spo2"], st.session_state.win_spo2
        )
        tachy_t = (not arts["HR"]) and persistent_high(
            df, "HR", eff["tachy_hr"], st.session_state.win_hr
        )
        hypot_t = (not arts["MAP"]) and persistent_low(
            df, "MAP", eff["low_map"], st.session_state.win_map
        )
        hypercap_t = (not arts["EtCO2"]) and persistent_high(
            df, "EtCO2", eff["high_et"], st.session_state.win_resp
        )
        hypovent_t = (not arts["RR"]) and persistent_low(
            df, "RR", eff["low_rr"], st.session_state.win_resp
        )

        if hypox_t:
            alerts_trend.append(
                {
                    "label": "Low SpO₂",
                    "severity": "Warning",
                    "why": f"SpO₂ < {eff['low_spo2']}% for ≥{st.session_state.win_spo2}s (exit > {eff['exit_spo2']}%)",
                }
            )
        if hypot_t:
            alerts_trend.append(
                {
                    "label": "Low MAP",
                    "severity": "Warning",
                    "why": f"MAP < {eff['low_map']} mmHg for ≥{st.session_state.win_map}s (exit > {eff['exit_map']} mmHg)",
                }
            )
        if tachy_t:
            alerts_trend.append(
                {
                    "label": "Tachycardia",
                    "severity": "Advisory",
                    "why": f"HR > {eff['tachy_hr']} bpm for ≥{st.session_state.win_hr}s",
                }
            )
        if hypercap_t or hypovent_t:
            msg = []
            if hypercap_t:
                msg.append(f"EtCO₂ > {eff['high_et']} mmHg")
            if hypovent_t:
                msg.append(f"RR < {eff['low_rr']} bpm")
            if hypox_t:
                msg.append(f"SpO₂ < {eff['low_spo2']}%")
            sev = "Advisory" if not (hypercap_t and hypovent_t and hypox_t) else "Warning"
            alerts_trend.append(
                {
                    "label": "Respiratory concern",
                    "severity": sev,
                    "why": ", ".join(msg),
                }
            )

        if alerts_trend:
            for a in alerts_trend:
                st.session_state.audit.append(
                    {"t": int(df["Time"].iloc[-1]), "action": "alert_raise", **a}
                )
                severity = a["severity"]
                if severity == "Advisory":
                    cls = "halo-alarm-advisory"
                elif severity == "Warning":
                    cls = "halo-alarm-warning"
                else:
                    cls = "halo-alarm-critical"
                st.markdown(
                    f"""
                    <div class="halo-alarm-card {cls}">
                        <div style="font-weight:700;margin-bottom:2px;">{severity}: {a['label']}</div>
                        <div style="font-size:0.9rem;">Why: {a['why']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            wmini = last_n(df, 12)
            if wmini is not None and not wmini.empty:
                st.markdown(
                    "<div style='font-size:0.9rem;margin-top:6px;margin-bottom:4px;'>Micro-trends (last 12 seconds)</div>",
                    unsafe_allow_html=True,
                )
                mini_c = st.columns(4)
                charts = [
                    ("HR", "HR — last 12s"),
                    ("MAP", "MAP — last 12s"),
                    ("SpO2", "SpO₂ — last 12s"),
                    ("EtCO2", "EtCO₂ — last 12s"),
                ]
                for i, (name, title) in enumerate(charts):
                    ch = interactive_12s_chart(wmini, name, title)
                    if ch is not None:
                        with mini_c[i]:
                            st.altair_chart(ch, use_container_width=True)
        else:
            st.markdown(
                """
                <div class="halo-ok-banner">
                    <div style="font-weight:700;margin-bottom:2px;">No active alarms</div>
                    <div style="font-size:0.9rem;">All monitored signals are within current educational thresholds.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.caption(
            "Within limits = any suggested actions remain inside your defined safety rails for targets and ranges. "
            "This is an educational assistant and does not issue clinical orders."
        )

# ================== TAB 2: ASSISTANT ==================
with tab_assistant:
    st.subheader("Conversational assistant")

    colQ1, colQ2 = st.columns([3, 1])
    q_text = colQ1.text_input(
        "Ask about the current physiology (e.g., “Why is MAP falling?” or “Summarize the current situation.”)",
        key="halo_q_input",
        placeholder="Type your question here…",
    )

    voice_text = ""
    with colQ2:
        st.markdown(
            "<div style='font-weight:700;color:var(--halo-blue);margin-bottom:4px;font-size:.95rem;'>Voice dictation</div>",
            unsafe_allow_html=True,
        )
        if voice_available():
            voice_text = voice_widget(label="🎤 Dictate question")
            status = st.session_state.get("voice_status", "Idle")
            st.caption(f"Microphone status: {status}")
        else:
            st.caption("Voice capture not available (missing optional dependencies). Use text instead.")

    latest_transcript = st.session_state.get("last_voice_transcript", "")
    if latest_transcript:
        st.markdown(
            f"""
            <div class="halo-voice-card">
                <div style="font-weight:700;margin-bottom:2px;font-size:0.98rem;color:var(--halo-amber);">
                    Detected speech
                </div>
                <div style="font-size:0.95rem;">“{latest_transcript}”</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    final_q = voice_text if (voice_text and len(voice_text) > 0) else q_text
    do_answer = (
        st.button("Generate explanation", type="primary", key="answer_btn") or bool(voice_text)
    )

    if do_answer and final_q:
        raw = answer_query(final_q, df, int(st.session_state.get("sim_hz", 5)))
        halo_text = raw["text"]
        html_text = halo_text.replace("\n", "<br>")

        st.markdown(
            f"""
            <div class="halo-response-card">
                <div style="font-weight:800;color:var(--halo-blue);font-size:1.05rem;margin-bottom:4px;">
                    HALO explanation (training only)
                </div>
                <div style="font-size:0.94rem;">{html_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if "conversation_log" not in st.session_state:
            st.session_state.conversation_log = []
        st.session_state.conversation_log.append({"q": final_q, "a": halo_text})
        st.session_state.conversation_log = st.session_state.conversation_log[-3:]

    if st.session_state.conversation_log:
        st.markdown("<h4 style='margin-top:1rem;margin-bottom:0.4rem;'>Recent Q&A</h4>", unsafe_allow_html=True)
        for i, entry in enumerate(reversed(st.session_state.conversation_log), 1):
            st.markdown(
                f"""
                <div class="halo-card-soft" style="border-left:4px solid var(--halo-blue);margin-bottom:6px;">
                    <div style="font-size:0.9rem;"><b>Q{i}:</b> {entry['q']}</div>
                    <div style="font-size:0.9rem;margin-top:2px;"><b>A{i}:</b> {entry['a']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ================== TAB 3: DATA QUALITY ==================
with tab_dq:
    st.subheader("Data quality (last 60s)")

    if len(df) >= 10:
        q1, q2, q3, q4, q5 = st.columns(5)
        rates = {
            "HR": (
                artifact_rate(df["HR"]),
                dropout_rate(df["HR"]),
                flatline_rate(df["HR"]),
            ),
            "SpO₂": (
                artifact_rate(df["SpO2"]),
                dropout_rate(df["SpO2"]),
                flatline_rate(df["SpO2"]),
            ),
            "MAP": (
                artifact_rate(df["MAP"]),
                dropout_rate(df["MAP"]),
                flatline_rate(df["MAP"]),
            ),
            "EtCO₂": (
                artifact_rate(df["EtCO2"]),
                dropout_rate(df["EtCO2"]),
                flatline_rate(df["EtCO2"]),
            ),
            "RR": (
                artifact_rate(df["RR"]),
                dropout_rate(df["RR"]),
                flatline_rate(df["RR"]),
            ),
        }

        def dq_badge(name, a, d, f):
            if a < 0.05:
                color = "#2f6b4b"
            elif a < 0.15:
                color = "#8a6a2a"
            else:
                color = "#8c3746"
            return (
                "<div class='halo-dq-badge' style='border-left:4px solid "
                + f"{color};'>"
                f"{name}: <b style='color:{color}'>{int(a*100)}%</b> artifact | "
                f"<b>{int(d*100)}%</b> dropout | "
                f"<b>{int(f*100)}%</b> flatline"
                "</div>"
            )

        q1.markdown(dq_badge("HR", *rates["HR"]), unsafe_allow_html=True)
        q2.markdown(dq_badge("SpO₂", *rates["SpO₂"]), unsafe_allow_html=True)
        q3.markdown(dq_badge("MAP", *rates["MAP"]), unsafe_allow_html=True)
        q4.markdown(dq_badge("EtCO₂", *rates["EtCO₂"]), unsafe_allow_html=True)
        q5.markdown(dq_badge("RR", *rates["RR"]), unsafe_allow_html=True)
    else:
        st.write("Collecting enough data to estimate quality…")

# ================== TAB 4: AUDIT & EXPORT ==================
with tab_audit:
    st.subheader("Alarm audit trail")
    if len(st.session_state.audit) == 0:
        st.write("No alerts have been raised yet.")
    else:
        st.dataframe(
            pd.DataFrame(st.session_state.audit),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Data export")
    if df.empty:
        st.write("No data yet.")
    else:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download vitals (CSV)",
            data=csv,
            file_name="halo_vitals.csv",
            mime="text/csv",
        )
        st.caption("Export shows exactly what HALO observed over time in this session.")

# ================== SELF-REFRESH ==================
if st.session_state.running:
    time.sleep(1)
    st.rerun()
