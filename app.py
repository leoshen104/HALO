# HALO — Hypothesis & Alarm Logic Orchestrator
# v4.0 — Live Summary + Conversational Assistant (Text + Voice)
# Not for clinical use.

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
# ===================== TITLE / BANNER =====================
st.markdown(
    """
    <div style="
        display:flex;
        align-items:center;
        justify-content:space-between;
        margin-bottom:8px;
    ">
        <div style="
            font-size:0.9rem;
            color:var(--halo-text-muted);
            padding:6px 12px;
            border-radius:999px;
            border:1px solid var(--halo-border);
            background:var(--halo-surface-soft);
        ">
            ⚠ Demo build · Not for clinical use
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.title("HALO — Hypothesis & Alarm Logic Orchestrator")
st.caption(
    "Interpretive assistant for anesthesia monitoring (assistant, not decider)."
)

# ===================== HALO PRO DARK MEDICAL UI (NO GRADIENTS, REBALANCED LAYOUT) =====================
HALO_PROFESSIONAL_DARK = """
<style>

:root {
    /* CORE PALETTE — Neutral, modern, medical */
    --halo-bg: #0D0F12;                  /* Main deep background */
    --halo-surface: #16181D;             /* Cards / panels */
    --halo-surface-soft: #1C1F25;        /* Soft secondary cards */
    --halo-border: rgba(255,255,255,0.08);
    --halo-border-strong: rgba(255,255,255,0.14);
    --halo-text-main: #F7F7F7;
    --halo-text-soft: #C8CCD2;
    --halo-text-muted: #8B9097;

    /* Accent colors — subtle but visible */
    --halo-blue: #4DA8FF;
    --halo-blue-soft: rgba(77,168,255,0.16);
    --halo-teal: #59E3C2;
    --halo-purple: #B798FF;
    --halo-amber: #E6C87A;
    --halo-red: #E05A68;

    --halo-radius: 22px;
    --halo-pill: 999px;
    --halo-shadow: 0 6px 18px rgba(0,0,0,0.45);

    --halo-font: "Inter", -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
}

/* ========== GLOBAL RESET & MAIN LAYOUT ========== */

html, body, [data-testid="stAppViewContainer"], .stApp {
    background-color: var(--halo-bg) !important;
    color: var(--halo-text-main) !important;
    font-family: var(--halo-font) !important;
}

/* KEEP STREAMLIT TOP BAR (for sidebar toggle) BUT MAKE IT INVISIBLE / FLAT */
[data-testid="stHeader"] {
    background: transparent !important;
    box-shadow: none !important;
    color: var(--halo-text-soft) !important;
    height: 3rem !important;          /* small bar just for the toggle */
}

/* Make the native <header> transparent instead of hiding it */
header {
    background: transparent !important;
    box-shadow: none !important;
}


/* Main content container: full-width with consistent padding */
.block-container {
    width: 100% !important;
    max-width: 100% !important;
    margin: 0 !important;                    /* remove centering gutter */
    padding-top: 18px !important;
    padding-bottom: 32px !important;
    padding-left: 32px !important;
    padding-right: 32px !important;
    box-sizing: border-box !important;       /* ensure padding doesn't shrink charts */
}

/* Tighten vertical rhythm between sections */
section.main h1 {
    margin-bottom: 0.35rem !important;
}
section.main h2,
section.main h3 {
    margin-top: 1.4rem !important;
    margin-bottom: 0.55rem !important;
}

/* Make normal body text comfortably readable */
.stMarkdown, .stCaption, p, span {
    font-size: 0.94rem !important;
    color: var(--halo-text-soft) !important;
}

/* ========== SIDEBAR ========== */

[data-testid="stSidebar"] {
    background-color: #111317 !important;
    border-right: 1px solid var(--halo-border-strong);
    padding-top: 18px !important;
}

[data-testid="stSidebar"] * {
    color: var(--halo-text-soft) !important;
}

/* Sidebar headings */
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--halo-text-main) !important;
    font-weight: 600 !important;
}

/* Sidebar inputs */
[data-testid="stSidebar"] .stTextInput > div > input,
[data-testid="stSidebar"] .stNumberInput > div > input {
    background: var(--halo-surface-soft) !important;
    color: var(--halo-text-main) !important;
    border-radius: 14px !important;
    border: 1px solid var(--halo-border) !important;
}

/* Sidebar sliders */
[data-testid="stSidebar"] [data-testid="stSlider"] > div > div > div {
    background: var(--halo-blue-soft) !important;
}

/* ========== BUTTONS ========== */

.stButton > button {
    background: var(--halo-surface) !important;
    border: 1px solid var(--halo-blue) !important;
    border-radius: var(--halo-pill) !important;
    color: var(--halo-blue) !important;
    padding: 0.45rem 1.15rem !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.45);
    transition: all 0.15s ease-in-out;
}
.stButton > button:hover {
    background: var(--halo-blue-soft) !important;
    color: var(--halo-blue) !important;
    transform: translateY(-1px);
}

/* Start/Stop & scenario buttons sit in a cleaner grid */
[data-testid="stHorizontalBlock"] > div {
    gap: 0.65rem !important;
}

/* ========== GENERIC CARDS / PANELS ========== */

.halo-card,
.halo-card-soft,
.halo-response-card {
    background: var(--halo-surface) !important;
    border-radius: var(--halo-radius) !important;
    padding: 18px 20px !important;
    border: 1px solid var(--halo-border) !important;
    box-shadow: var(--halo-shadow);
}

.halo-card-soft {
    background: var(--halo-surface-soft) !important;
}

/* Voice transcript card */
.halo-voice-card {
    background: rgba(230,200,122,0.10) !important;
    border-left: 8px solid var(--halo-amber) !important;
    border-radius: 20px !important;
    padding: 14px 18px !important;
    color: var(--halo-text-main) !important;
}

/* Alarm panels */
.halo-alarm {
    border-radius: 22px !important;
    padding: 16px 20px !important;
    margin-top: 10px !important;
    margin-bottom: 10px !important;
    box-shadow: var(--halo-shadow);
}
.halo-alarm-advisory {
    background: var(--halo-blue-soft);
    border-left: 8px solid var(--halo-blue);
}
.halo-alarm-warning {
    background: rgba(230,200,122,0.16);
    border-left: 8px solid var(--halo-amber);
}
.halo-alarm-critical {
    background: rgba(224,90,104,0.20);
    border-left: 8px solid var(--halo-red);
}

/* Live summary panel */
.halo-summary-card {
    background: var(--halo-surface-soft) !important;
    border-left: 8px solid var(--halo-blue);
    border-radius: var(--halo-radius);
    padding: 16px 18px;
    margin-top: 12px;
}

/* Top Suggested Action + Data Quality (outer card look) */
.halo-tsa-border,
.halo-dq-border {
    background: var(--halo-surface) !important;
    border-radius: var(--halo-radius);
    border: 1px solid var(--halo-border-strong) !important;
    box-shadow: var(--halo-shadow);
    padding: 14px 16px !important;
}

/* Small status pills */
.halo-status-pill {
    display: inline-flex;
    align-items: center;
    padding: 6px 12px;
    border-radius: var(--halo-pill);
    border: 1px solid var(--halo-border);
    background: var(--halo-surface-soft);
    color: var(--halo-text-soft);
    font-size: 0.82rem;
    font-weight: 600;
}

/* ========== CHART WRAPPERS & HIERARCHY ========== */

/* Only wrap real charts (those containers that hold canvas or svg) */
[data-testid="stVerticalBlock"] .element-container:has(canvas),
[data-testid="stVerticalBlock"] .element-container:has(svg) {
    background: var(--halo-surface) !important;
    border-radius: var(--halo-radius) !important;
    border: 1px solid var(--halo-border) !important;
    padding: 10px 12px !important;
    margin-top: 8px !important;
    margin-bottom: 18px !important;
    box-shadow: var(--halo-shadow);
}

/* Remove inner white rectangle — let chart sit cleanly in the card */
canvas, svg {
    background: transparent !important;
}

/* Chart titles */
h2, h3, h4 {
    color: var(--halo-text-main) !important;
    letter-spacing: -0.01em !important;
}

/* Axes / ticks */
g text {
    fill: var(--halo-text-soft) !important;
    font-size: 0.83rem !important;
}

/* Line strokes for the main vital charts (thicker + round caps) */
svg .mark-line path {
    stroke-width: 3px !important;
    stroke-linecap: round !important;
}

/* 12-second mini charts */
.halo-mini-chart {
    background: var(--halo-surface-soft) !important;
    border-radius: 20px !important;
    border: 1px solid var(--halo-border) !important;
    padding: 12px 14px !important;
    margin-bottom: 14px !important;
}

/* ========== VOICE WIDGET ========== */

.audio-recorder,
.audio-recorder button {
    border-radius: var(--halo-pill) !important;
    background: var(--halo-surface-soft) !important;
    border: 1px solid var(--halo-blue) !important;
    color: var(--halo-blue) !important;
}

/* ========== MISC SMALL TWEAKS ========== */

/* Give a touch more vertical breathing room between stacked rows */
[data-testid="stVerticalBlock"] > div {
    margin-bottom: 0.4rem !important;
}

/* Ensure checkboxes + labels are readable */
.stCheckbox label {
    color: var(--halo-text-soft) !important;
}

/* Tooltips on charts */
div[role="tooltip"] {
    background: #111317 !important;
    border-radius: 10px !important;
    border: 1px solid var(--halo-border-strong) !important;
    color: var(--halo-text-main) !important;
}

</style>
"""

st.markdown(HALO_PROFESSIONAL_DARK, unsafe_allow_html=True)



# ================== PAGE / BRANDING ==================
st.set_page_config(page_title="HALO v4.0", page_icon="🛡️", layout="wide")
DEMO_MODE = False  # used only to toggle researcher controls in the sidebar


# ================== VOICE CORE (unchanged features) ==================
def voice_available() -> bool:
    try:
        import speech_recognition  # type: ignore
        from audio_recorder_streamlit import audio_recorder  # type: ignore
        return True
    except Exception:
        return False

def voice_widget(label: str = "🎤 Hold to speak") -> str:
    try:
        from audio_recorder_streamlit import audio_recorder  # type: ignore
        import speech_recognition as sr  # type: ignore

        st.markdown(
            "<div class='halo-card-soft' style='margin-bottom:6px;font-size:.9rem;'>🎤 Voice dictation — click the mic, speak in a full sentence, then pause briefly.</div>",
            unsafe_allow_html=True,
        )
        audio_bytes = audio_recorder(text=label, icon_size="3x")
        if not audio_bytes:
            return ""

        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 200
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.8

        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio, language="en-US")
        except sr.UnknownValueError:
            st.warning("I couldn’t understand that clearly — try speaking a bit closer or slower.")
            return ""
        except Exception as e:
            st.warning(f"Transcription error: {e}")
            return ""

        text = text.strip()
        st.session_state["last_voice_transcript"] = text
        return text
    except Exception as e:
        st.info(f"Voice capture not available (optional dependencies missing): {e}")
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
        story += "Overall, physiology appears dynamic, with evolving trends that merit ongoing attention."

    return story

# ================== Q&A HELPERS ==================
def _last_window(df: pd.DataFrame, seconds: int, sim_hz: int) -> pd.DataFrame:
    if df is None or df.empty or sim_hz is None or sim_hz <= 0:
        return df if df is not None else pd.DataFrame()
    n = max(1, min(len(df), int(seconds * sim_hz)))
    return df.iloc[-n:].copy()

def _slope(series: pd.Series) -> float:
    try:
        if series is None or len(series) < 2:
            return 0.0
        return float(series.iloc[-1] - series.iloc[0])
    except Exception:
        return 0.0

def _safe_corr(a: np.ndarray | List[float], b: np.ndarray | List[float]):
    try:
        a = np.asarray(a); b = np.asarray(b)
        if a.size < 3 or b.size < 3:
            return None
        c = float(np.corrcoef(a, b)[0, 1])
        return None if np.isnan(c) else c
    except Exception:
        return None

def _mini_line_fig(x, y, ylabel: str, title: str = ""):
    fig, ax = plt.subplots(figsize=(4.5, 2.0))
    ax.plot(x, y, linewidth=2)
    ax.set_ylabel(ylabel)
    ax.set_xticks([])
    ax.set_title(title)
    fig.tight_layout()
    return fig

# ================== CONVERSATIONAL ANSWER ==================
def answer_query(q: str, df: pd.DataFrame, sim_hz: int = 5):
    if df is None or df.empty:
        return {
            "text": ("**Short answer:** I don’t have enough data yet to answer that.\n\n"
                     "Start the live stream or upload a replay, then ask again."),
            "figs": [],
        }

    cols = _resolve_cols(df)
    if not cols["Time"]:
        return {
            "text": ("**Short answer:** I can't interpret this stream because there is no time index."),
            "figs": [],
        }

    look_secs = 90
    keep_cols = [
        c for c in
        [cols["Time"], cols["MAP"], cols["HR"], cols["SpO2"], cols["EtCO2"], cols["RR"]]
        if c
    ]
    w = _last_window(df[keep_cols], look_secs, max(int(sim_hz or 1), 1))
    if w is None or w.empty:
        return {
            "text": ("**Short answer:** There isn't enough recent data in this window to interpret."),
            "figs": [],
        }

    def S(name: str):
        c = cols.get(name)
        return w[c] if c and c in w.columns else None

    def last_safe(series: Optional[pd.Series]) -> Optional[float]:
        if series is None or series.empty:
            return None
        try:
            return float(series.iloc[-1])
        except Exception:
            return None

    def slope_safe(series: Optional[pd.Series]) -> float:
        try:
            return _slope(series) if series is not None else 0.0
        except Exception:
            return 0.0

    vals = {v: last_safe(S(v)) for v in ["MAP", "HR", "SpO2", "EtCO2", "RR"]}
    slopes = {v: slope_safe(S(v)) for v in ["MAP", "HR", "SpO2", "EtCO2", "RR"]}

    def trend_label(s: float) -> str:
        if s > 0.3:
            return "rising"
        if s < -0.3:
            return "falling"
        return "stable"

    def status_line(label: str, val: Optional[float], s: float, unit: str) -> str:
        if val is None:
            return f"{label}: —"
        t = trend_label(s)
        return f"{label}: {int(round(val))} {unit} ({t})"

    def classify_status(values: dict, trends: dict) -> tuple[str, str, list[str]]:
        score = 0
        issues: list[str] = []
        m = values.get("MAP")
        if m is not None:
            if m < 65:
                score += 2
                issues.append(f"MAP {m:.0f} mmHg is below a typical target of 65.")
            if trends["MAP"] < -0.3:
                score += 1
                issues.append("MAP is trending downward.")
        s = values.get("SpO2")
        if s is not None:
            if s < 92:
                score += 2
                issues.append(f"SpO₂ {s:.0f}% is below a usual safety floor (≈92%).")
            if trends["SpO2"] < -0.2:
                score += 1
                issues.append("SpO₂ is trending down.")
            if s > 100:
                issues.append(
                    f"SpO₂ reading {s:.0f}% is likely artifact; probe or calibration issue."
                )
        e = values.get("EtCO2")
        if e is not None:
            if e < 25 or e > 50:
                score += 1
                issues.append(
                    f"EtCO₂ {e:.0f} mmHg is outside a typical 25–50 mmHg range."
                )
            if abs(trends["EtCO2"]) > 0.3:
                issues.append("EtCO₂ is changing quickly.")
        h = values.get("HR")
        if h is not None:
            if h > 120 or h < 45:
                score += 1
                issues.append(
                    f"HR {h:.0f} bpm is outside usual 45–120 bpm range under anesthesia."
                )
        r = values.get("RR")
        if r is not None:
            if r < 8 or r > 28:
                score += 1
                issues.append(
                    f"RR {r:.0f} bpm is outside a typical 8–28 bpm window."
                )
        if score <= 1:
            return "Stable", "No major deviations from typical targets.", issues
        if score == 2:
            return (
                "Watch closely",
                "Borderline physiology with emerging risk; not fully stable.",
                issues,
            )
        return (
            "Unstable",
            "Parameters suggest physiologic instability that warrants intervention.",
            issues,
        )

    stability_label, stability_rationale, issues = classify_status(vals, slopes)

    s_spo2, s_et = S("SpO2"), S("EtCO2")
    corr = (
        _safe_corr(s_spo2.values, s_et.values)
        if s_spo2 is not None and s_et is not None
        else None
    )
    corr_txt = (
        f"SpO₂–EtCO₂ correlation ≈ {corr:+.2f}."
        if corr is not None
        else "SpO₂–EtCO₂ relationship not clearly defined in this window."
    )

    q_low = (q or "").lower().strip()
    is_stable_q = (
        "stable" in q_low
        and ("is the" in q_low or "is patient" in q_low or "is this" in q_low or "are they" in q_low)
    )

    interpretation = ""
    map_s, hr_s, spo2_s, etco2_s, rr_s = (
        slopes["MAP"],
        slopes["HR"],
        slopes["SpO2"],
        slopes["EtCO2"],
        slopes["RR"],
    )
    if map_s < -0.3 and hr_s > 0.3:
        interpretation = (
            "MAP is falling while HR is rising — pattern consistent with compensated "
            "hypotension or relative hypovolemia."
        )
    elif map_s < -0.3 and hr_s <= 0.1:
        interpretation = (
            "MAP is falling without a strong HR response — more compatible with vasodilation "
            "or anesthetic drug effect than acute bleeding."
        )
    elif hr_s > 0.3 and map_s > 0.2:
        interpretation = (
            "MAP and HR are rising together — sympathetic activation such as pain or light "
            "anesthesia is likely."
        )
    elif spo2_s < -0.2 and etco2_s > 0.2:
        interpretation = (
            "SpO₂ is trending down while EtCO₂ is rising — suggests hypoventilation or an "
            "airway/ventilation problem."
        )
    else:
        interpretation = (
            "No single catastrophic pattern dominates, but trends should be followed over time."
        )

    if is_stable_q:
        if stability_label == "Stable":
            short_answer = (
                "**Short answer:** Yes — within this ~90 second window, HALO would describe "
                "the patient as physiologically stable."
            )
        elif stability_label == "Watch closely":
            short_answer = (
                "**Short answer:** Not fully — vital signs are borderline and trending away "
                "from usual targets, so this should not be called fully stable."
            )
        else:
            short_answer = (
                "**Short answer:** No — HALO would consider this patient physiologically "
                "unstable in this window."
            )
    else:
        if any(w in q_low for w in ["why", "cause", "reason"]):
            short_answer = (
                "**Short answer:** Here’s the most likely physiologic explanation for what "
                "you're seeing."
            )
        elif any(w in q_low for w in ["what", "status", "trend", "happening"]):
            short_answer = (
                "**Short answer:** Here’s the current physiologic status based on recent data."
            )
        elif any(w in q_low for w in ["should", "do", "action", "intervene"]):
            short_answer = (
                "**Short answer:** Based on current trends, this patient warrants attention; "
                "see suggested actions below."
            )
        else:
            short_answer = (
                "**Short answer:** Here’s how HALO currently interprets the physiology."
            )

    quick_status_lines = [
        status_line("MAP", vals["MAP"], map_s, "mmHg"),
        status_line("HR", vals["HR"], hr_s, "bpm"),
        status_line("SpO₂", vals["SpO2"], spo2_s, "%"),
        status_line("EtCO₂", vals["EtCO2"], etco2_s, "mmHg"),
        status_line("RR", vals["RR"], rr_s, "bpm"),
    ]
    quick_status = "\n".join(f"- {ln}" for ln in quick_status_lines)

    try:
        tsa = top_suggested_action(df)
    except Exception:
        tsa = {}

    if tsa:
        action_block = (
            f"**HALO suggested next step (demo only):** {tsa['Suggested']}\n\n"
            f"- Rationale: {tsa['Rationale']}\n"
            f"- Within limits: {tsa['Within']}\n"
            f"- Confidence: {tsa['Confidence']}\n"
            f"- Anticipatory: {tsa['Anticipatory']}"
        )
    else:
        if "hypotension" in interpretation.lower() or (
            vals["MAP"] is not None and vals["MAP"] < 65
        ):
            fallback_action = (
                "Assess volume status, anesthetic depth, and consider vasopressor or fluid "
                "support as clinically appropriate."
            )
        elif "hypoventilation" in interpretation.lower():
            fallback_action = (
                "Check airway and ventilation — consider increasing minute ventilation or "
                "troubleshooting the circuit."
            )
        else:
            fallback_action = (
                "Continue close monitoring and reassess trends over the next 1–2 minutes."
            )
        action_block = f"**HALO suggested next step (demo only):** {fallback_action}"

    try:
        phys_story = physiology_story(df, window_sec=90, sim_hz=sim_hz)
    except Exception:
        phys_story = ""

    text_parts: list[str] = []
    text_parts.append(short_answer)
    text_parts.append("")
    text_parts.append(
        f"**Stability classification:** {stability_label} — {stability_rationale}"
    )
    text_parts.append("")
    text_parts.append("**Quick physiologic status (last ~60–90 s):**")
    text_parts.append(quick_status)
    text_parts.append("")
    text_parts.append(f"**Interpretation:** {interpretation}")
    text_parts.append(f"**Signal relationship:** {corr_txt}")
    text_parts.append("")
    text_parts.append(action_block)
    if phys_story:
        text_parts.append("")
        text_parts.append("**Physiology narrative (last ~90 s):**")
        text_parts.append(phys_story)

    final_text = "\n".join(text_parts)
    return {"text": final_text, "figs": []}

# ================== SESSION DEFAULTS ==================
def _setdefault(k, v):
    if k not in st.session_state:
        st.session_state[k] = v

_setdefault("history", pd.DataFrame(columns=["Time", "HR", "SpO2", "MAP", "EtCO2", "RR"]))
_setdefault("running", False)
_setdefault("mode", "Live")  # "Live" or "Replay"
_setdefault("replay_df", None)
_setdefault("replay_idx", 0)
_setdefault("replay_speed", 1)    # rows per tick
_setdefault("sim_hz", 5)          # samples per second
_setdefault("scenario_name", None)
_setdefault("scenario_end", 0.0)
_setdefault("events", [])
_setdefault("audit", [])
_setdefault("sim_val", {"HR": 80, "SpO2": 97, "MAP": 80, "EtCO2": 37, "RR": 12})
_setdefault("sim_time", 0)
_setdefault("conversation_log", [])

# Thresholds (base)
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

# Noise / artifact injection
_setdefault("enable_noise", True)
_setdefault("artifact_pct", 5)

# ================== TOP CONTROLS ==================
ctrl = st.container()
with ctrl:
    c1, c2, c3, c4 = st.columns([1, 1, 2, 2])
    if c1.button("▶ Start", key="btn_start"):
        st.session_state.running = True
    if c2.button("■ Stop", key="btn_stop"):
        st.session_state.running = False

    c3.slider(
        "Samples per second",
        1,
        10,
        st.session_state.get("sim_hz", 5),
        key="sim_hz",
    )

    status = "Running" if st.session_state.running else "Stopped"
    c4.markdown(
        f"<span class='halo-status-pill'>Mode: {st.session_state.mode} · "
        f"{status} · Samples: {len(st.session_state.history)}</span>",
        unsafe_allow_html=True,
    )

# ================== SIDEBAR (same features, new look) ==================
with st.sidebar:
    st.header("Mode")
    st.session_state.mode = st.radio(
        "Input source",
        ["Live", "Replay"],
        index=0 if st.session_state.mode == "Live" else 1,
        key="mode_radio",
    )

    if st.session_state.mode == "Replay":
        up = st.file_uploader(
            "Upload CSV (Time, HR, SpO2, MAP, optional EtCO2, RR)",
            type=["csv"],
            key="replay_csv",
        )
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
            "Replay speed (rows/tick)",
            1,
            20,
            st.session_state.get("replay_speed", 1),
            key="replay_speed",
        )

    show_details = (
        True
        if not DEMO_MODE
        else st.checkbox("Show researcher details", value=False, key="show_details")
    )

    if show_details:
        st.header("Thresholds")
        st.slider(
            "Low SpO₂ (%)",
            85,
            96,
            st.session_state.get("low_spo2", 92),
            key="low_spo2",
        )
        st.slider(
            "Tachycardia HR (bpm)",
            90,
            160,
            st.session_state.get("tachy_hr", 120),
            key="tachy_hr",
        )
        st.slider(
            "Low MAP (mmHg)",
            50,
            80,
            st.session_state.get("low_map", 65),
            key="low_map",
        )

        st.subheader("Respiratory")
        st.slider(
            "High EtCO₂ (mmHg)",
            40,
            60,
            st.session_state.get("high_et", 50),
            key="high_et",
        )
        st.slider(
            "Low EtCO₂ (mmHg)",
            20,
            40,
            st.session_state.get("low_et", 30),
            key="low_et",
        )
        st.slider(
            "Low RR (bpm)",
            4,
            20,
            st.session_state.get("low_rr", 8),
            key="low_rr",
        )
        st.slider(
            "High RR (bpm)",
            18,
            40,
            st.session_state.get("high_rr", 28),
            key="high_rr",
        )

        st.header("Persistence (sec)")
        st.slider(
            "SpO₂ persistence",
            5,
            30,
            st.session_state.get("win_spo2", 8),
            key="win_spo2",
        )
        st.slider(
            "HR persistence",
            5,
            30,
            st.session_state.get("win_hr", 8),
            key="win_hr",
        )
        st.slider(
            "MAP persistence",
            5,
            30,
            st.session_state.get("win_map", 10),
            key="win_map",
        )
        st.slider(
            "Resp persistence",
            5,
            30,
            st.session_state.get("win_resp", 12),
            key="win_resp",
        )

        st.header("Hysteresis & Cooldown")
        st.slider(
            "SpO₂ hysteresis (+%)",
            1,
            6,
            st.session_state.get("hys_spo2", 2),
            key="hys_spo2",
        )
        st.slider(
            "MAP hysteresis (+mmHg)",
            2,
            12,
            st.session_state.get("hys_map", 5),
            key="hys_map",
        )
        st.slider(
            "Alarm cooldown (s)",
            0,
            120,
            st.session_state.get("cooldown", 30),
            key="cooldown",
        )

        st.header("Noise / Artifacts (Live)")
        st.checkbox(
            "Inject artifact/noise",
            value=st.session_state.get("enable_noise", True),
            key="enable_noise",
        )
        st.slider(
            "Artifact chance (%)",
            0,
            20,
            st.session_state.get("artifact_pct", 5),
            key="artifact_pct",
        )

    st.divider()
    if st.button("Reset Data", key="reset_btn"):
        st.session_state.history = pd.DataFrame(
            columns=["Time", "HR", "SpO2", "MAP", "EtCO2", "RR"]
        )
        st.session_state.replay_idx = 0
        st.session_state.events = []
        st.session_state.audit = []
        st.session_state.sim_time = 0
        st.success("Buffers cleared.")

# ================== SCENARIOS & EVENTS (unchanged) ==================
sc1, sc2, sc3, sc4, sc5 = st.columns(5)
if sc1.button("Scenario: Bleed (60s)", key="sc_bleed"):
    st.session_state.scenario_name = "Bleed"
    st.session_state.scenario_end = time.time() + 60
if sc2.button("Scenario: Bronchospasm (60s)", key="sc_bronch"):
    st.session_state.scenario_name = "Bronchospasm"
    st.session_state.scenario_end = time.time() + 60
if sc3.button("Scenario: Vasodilation (60s)", key="sc_vaso"):
    st.session_state.scenario_name = "Vasodilation"
    st.session_state.scenario_end = time.time() + 60
if sc4.button("Scenario: Pain/Light (60s)", key="sc_pain"):
    st.session_state.scenario_name = "Pain/Light"
    st.session_state.scenario_end = time.time() + 60
if sc5.button("End Scenario", key="sc_end"):
    st.session_state.scenario_name = None
    st.session_state.scenario_end = 0.0

e1, e2, e3, e4, e5 = st.columns(5)
if e1.button("Mark: Incision", key="mark_incision"):
    st.session_state.events.append({"t": len(st.session_state.history), "name": "Incision"})
if e2.button("Mark: Position change", key="mark_pos"):
    st.session_state.events.append({"t": len(st.session_state.history), "name": "Position change"})
if e3.button("Mark: Fluids 250 mL", key="mark_fluid"):
    st.session_state.events.append({"t": len(st.session_state.history), "name": "Fluids 250 mL"})
if e4.button("Mark: Vasopressor", key="mark_press"):
    st.session_state.events.append({"t": len(st.session_state.history), "name": "Vasopressor"})
if e5.button("Clear Events", key="mark_clear"):
    st.session_state.events = []

# ================== DYNAMIC THRESHOLD ENGINE (unchanged) ==================
def _time_now_index() -> int:
    if st.session_state.history.empty:
        return 0
    return int(st.session_state.history["Time"].iloc[-1])

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

def _scenario_mods():
    name = st.session_state.scenario_name
    active = (name is not None) and (time.time() <= st.session_state.scenario_end)
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

# ================== SIMULATION / REPLAY (unchanged) ==================
def __step(val, target, sigma, lo, hi):
    v = val + random.gauss(0, sigma) + (target - val) * random.uniform(0.02, 0.08)
    return max(lo, min(hi, v))

def _scenario_targets(base):
    name = st.session_state.scenario_name
    active = (name is not None) and (time.time() <= st.session_state.scenario_end)
    if not active:
        return dict(
            HR=base["HR"],
            SpO2=base["SpO2"],
            MAP=base["MAP"],
            EtCO2=base["EtCO2"],
            RR=base["RR"],
        )
    targets = dict(
        HR=base["HR"],
        SpO2=base["SpO2"],
        MAP=base["MAP"],
        EtCO2=base["EtCO2"],
        RR=base["RR"],
    )
    if name == "Bleed":
        targets["MAP"] = base["MAP"] - 15
        targets["HR"] = base["HR"] + 15
        targets["SpO2"] = base["SpO2"] - 1
    elif name == "Bronchospasm":
        targets["SpO2"] = base["SpO2"] - 6
        targets["EtCO2"] = base["EtCO2"] + 8
        targets["RR"] = base["RR"] + 6
    elif name == "Vasodilation":
        targets["MAP"] = base["MAP"] - 12
        targets["HR"] = base["HR"] - 3
    elif name == "Pain/Light":
        targets["HR"] = base["HR"] + 12
        targets["MAP"] = base["MAP"] + 5
        targets["RR"] = base["RR"] + 4
    return targets

def _tick_live(sim_hz):
    for _ in range(sim_hz):
        base = st.session_state.sim_val
        targets = _scenario_targets(base)
        hr = __step(base["HR"], targets["HR"], 1.2, 30, 180)
        spo2 = __step(base["SpO2"], targets["SpO2"], 0.6, 70, 100)
        map_ = __step(base["MAP"], targets["MAP"], 1.5, 40, 120)
        et = __step(base["EtCO2"], targets["EtCO2"], 0.8, 20, 60)
        rr = __step(base["RR"], targets["RR"], 0.8, 6, 40)

        if st.session_state.enable_noise and random.random() < (
            st.session_state.artifact_pct / 100.0
        ):
            pick = random.random()
            if pick < 0.25:
                hr += random.choice([-12, 12])
            elif pick < 0.50:
                spo2 += random.choice([-8, 8])
            elif pick < 0.75:
                map_ += random.choice([-10, 10])
            else:
                et += random.choice([-6, 6])

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
    step = st.session_state.replay_speed
    end = min(i + step, len(df_r))
    chunk = df_r.iloc[i:end].copy()
    base = len(st.session_state.history)
    chunk["Time"] = np.arange(base, base + len(chunk))
    st.session_state.history = pd.concat(
        [st.session_state.history, chunk], ignore_index=True
    )
    st.session_state.replay_idx = end

if st.session_state.running:
    if st.session_state.mode == "Live":
        _tick_live(st.session_state.sim_hz)
    else:
        _tick_replay()

df = st.session_state.history
st.session_state["df"] = df

# ================== VITALS CHARTS ==================
c1, c2, c3, c4, c5 = st.columns(5)
if not df.empty:
    c1.subheader("HR (bpm)")
    c1.line_chart(df.set_index("Time")["HR"], use_container_width=True)
    c2.subheader("SpO₂ (%)")
    c2.line_chart(df.set_index("Time")["SpO2"], use_container_width=True)
    c3.subheader("MAP (mmHg)")
    c3.line_chart(df.set_index("Time")["MAP"], use_container_width=True)
    c4.subheader("EtCO₂ (mmHg)")
    c4.line_chart(df.set_index("Time")["EtCO2"], use_container_width=True)
    c5.subheader("RR (bpm)")
    c5.line_chart(df.set_index("Time")["RR"], use_container_width=True)
else:
    st.info("Click Start (Live) or upload a CSV (Replay).")

# ================== LIVE SITUATION SUMMARY (SHORT) ==================
def live_summary_block(df_: pd.DataFrame, sim_hz: int) -> str:
    window = df_.tail(min(len(df_), 90))

    def slope_of(col):
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
        else ("MAP rising — recovery/pressor response." if s_map > 0.3 else "MAP stable.")
    )
    parts.append(
        "HR increasing — possible pain/stress or compensation."
        if s_hr > 0.3
        else ("HR decreasing — drug/depth effect." if s_hr < -0.3 else "HR stable.")
    )
    parts.append(
        "SpO₂ trending down — check airway/oxygenation."
        if s_spo2 < -0.2
        else ("SpO₂ improving." if s_spo2 > 0.2 else "SpO₂ stable.")
    )
    parts.append(
        "EtCO₂ falling — hyperventilation or perfusion drop."
        if s_et < -0.3
        else ("EtCO₂ rising — hypoventilation/CO₂ retention." if s_et > 0.3 else "EtCO₂ stable.")
    )
    parts.append(
        "RR rising — compensatory hyperventilation."
        if s_rr > 0.3
        else ("RR decreasing — sedation/airway depression." if s_rr < -0.3 else "RR stable.")
    )
    return " · ".join(parts)

st.markdown(
    f"""
    <div class="halo-summary-card">
        <div style="font-weight:800;color:var(--halo-accent);font-size:1.1rem;margin-bottom:6px;">
            🧩 Live Situation Summary
        </div>
        <div style="margin:0;">{live_summary_block(df, int(st.session_state.get("sim_hz",5)))}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ================== ALARM HELPERS ==================
def last_n(df_: pd.DataFrame, seconds: int) -> pd.DataFrame:
    if df_ is None or df_.empty:
        return df_
    end = df_["Time"].iloc[-1]
    start = max(0, end - seconds + 1)
    return df_[df_["Time"].between(start, end)]

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

eff = effective_thresholds()

# ================== ALARM FUSION ==================
alerts: List[dict] = []
hypox = tachy = hypot = hypercap = hypovent = False

if not df.empty:
    arts = recent_artifact(df)
    hypox = (not arts["SpO2"]) and persistent_low(
        df, "SpO2", eff["low_spo2"], st.session_state.win_spo2
    )
    tachy = (not arts["HR"]) and persistent_high(
        df, "HR", eff["tachy_hr"], st.session_state.win_hr
    )
    hypot = (not arts["MAP"]) and persistent_low(
        df, "MAP", eff["low_map"], st.session_state.win_map
    )
    hypercap = (not arts["EtCO2"]) and persistent_high(
        df, "EtCO2", eff["high_et"], st.session_state.win_resp
    )
    hypovent = (not arts["RR"]) and persistent_low(
        df, "RR", eff["low_rr"], st.session_state.win_resp
    )

    if hypox:
        alerts.append(
            {
                "label": "Low SpO₂",
                "severity": "Warning",
                "why": f"SpO₂ < {eff['low_spo2']}% ≥{st.session_state.win_spo2}s (exit > {eff['exit_spo2']}%)",
            }
        )
    if hypot:
        alerts.append(
            {
                "label": "Low MAP",
                "severity": "Warning",
                "why": f"MAP < {eff['low_map']} ≥{st.session_state.win_map}s (exit > {eff['exit_map']})",
            }
        )
    if tachy:
        alerts.append(
            {
                "label": "Tachycardia",
                "severity": "Advisory",
                "why": f"HR > {eff['tachy_hr']} ≥{st.session_state.win_hr}s",
            }
        )
    if hypercap or hypovent:
        msg = []
        if hypercap:
            msg.append(f"EtCO₂ > {eff['high_et']}")
        if hypovent:
            msg.append(f"RR < {eff['low_rr']}")
        if hypox:
            msg.append(f"SpO₂ < {eff['low_spo2']}")
        sev = (
            "Advisory"
            if not (hypercap and hypovent and hypox)
            else "Warning"
        )
        alerts.append(
            {
                "label": "Respiratory Concern",
                "severity": sev,
                "why": ", ".join(msg),
            }
        )

st.caption(
    "**Within limits** = HALO’s recommendation stays inside your clinician-defined safety rails "
    "for drug dosages, infusion rates, and vital sign targets. HALO will never suggest exceeding those bounds."
)

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
            stats[k] = {
                "mean": float(np.mean(w[k])),
                "trend": slope_local(w[k]),
            }
    if not stats:
        return {}

    action = "Observe"
    rationale = "Signals stable."
    confidence = 0.5
    anticipatory = "No immediate deterioration expected."
    alts = ["Reassess in 30–60s", "Check sensor positions"]
    within = "✅ within limits"

    if "MAP" in stats and "HR" in stats:
        if stats["MAP"]["mean"] < 65 or stats["MAP"]["trend"] < -0.3:
            if stats["HR"]["trend"] > 0.2:
                action = "Evaluate volume; consider fluids if appropriate. Support MAP ≥ 65."
                rationale = (
                    "MAP low/falling with HR rising — pattern consistent with relative hypovolemia."
                )
                confidence = 0.7
                anticipatory = (
                    "MAP likely to remain below floor without intervention."
                )
                alts = [
                    "Vasopressor if volume adequate",
                    "Reduce anesthetic depth if appropriate",
                ]
            else:
                action = "Consider vasopressor support; review anesthetic depth."
                rationale = (
                    "MAP low/falling with limited HR response — vasodilation/drug effect likely."
                )
                confidence = 0.65
                anticipatory = "Sustained hypotension risk without treatment."
                alts = [
                    "Small fluid bolus if uncertain",
                    "Lower hypnotic dose if clinically safe",
                ]

    if "SpO2" in stats and "EtCO2" in stats:
        if stats["SpO2"]["trend"] < -0.2 and stats["EtCO2"]["trend"] > 0.2:
            action = "Address ventilation/airway; increase support."
            rationale = (
                "SpO₂ falling while EtCO₂ rising — hypoventilation/airway issue likely."
            )
            confidence = 0.75
            anticipatory = "Oxygenation may worsen if not treated."
            alts = [
                "Re-seat probe (rule out artifact)",
                "Increase minute ventilation",
                "Suction/check circuit",
            ]

    return {
        "Suggested": action,
        "Rationale": rationale,
        "Within": within,
        "Confidence": f"{int(round(confidence * 100))}%",
        "Anticipatory": anticipatory,
        "Alternatives": alts,
    }

def interactive_12s_chart(df_: pd.DataFrame, field: str, title: str) -> Optional[alt.Chart]:
    if df_ is None or df_.empty or field not in df_.columns:
        return None
    w = last_n(df_, 12)
    if w is None or w.empty:
        w = df_.iloc[-min(12, len(df_)):]
    chart_df = w[["Time", field]].copy()
    chart = (
        alt.Chart(chart_df)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "Time:Q",
                axis=alt.Axis(
                    title="(last 12s)",
                    labelColor="#555555",
                    titleColor="#555555",
                ),
            ),
            y=alt.Y(
                f"{field}:Q",
                axis=alt.Axis(
                    title=field,
                    labelColor="#555555",
                    titleColor="#555555",
                ),
            ),
            tooltip=["Time", field],
        )
        .properties(title=title, height=180, background="#ffffff")
        .configure_axis(gridColor="#e2ddd1")
    )
    return chart

st.subheader("Alarm Panel")

def alarm_banner(a):
    label, severity, why = a["label"], a["severity"], a["why"]
    palettes = {
        "Advisory": {
            "border": "var(--halo-accent)",
            "bg": "rgba(31,59,87,0.06)",
            "title": "var(--halo-accent)",
            "text": "var(--halo-text-soft)",
        },
        "Warning": {
            "border": "var(--halo-amber)",
            "bg": "rgba(138,106,42,0.07)",
            "title": "var(--halo-amber)",
            "text": "var(--halo-text-soft)",
        },
        "Critical": {
            "border": "var(--halo-red)",
            "bg": "rgba(140,55,70,0.07)",
            "title": "var(--halo-red)",
            "text": "var(--halo-text-soft)",
        },
    }
    p = palettes.get(
        severity,
        {
            "border": "#777777",
            "bg": "#f5f2ec",
            "title": "#333333",
            "text": "#555555",
        },
    )
    st.markdown(
        f"""
        <div class="halo-card" style="border-left:10px solid {p['border']}; background:{p['bg']};">
            <div class="halo-alarm-title" style="color:{p['title']};">{severity}: {label}</div>
            <div style="color:{p['text']};"><b>Why:</b> {why}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    wmini = last_n(df, 12)
    if wmini is None or wmini.empty:
        return
    cols = st.columns(4)
    charts = [
        ("HR", "HR — last 12s"),
        ("MAP", "MAP — last 12s"),
        ("SpO2", "SpO₂ — last 12s"),
        ("EtCO2", "EtCO₂ — last 12s"),
    ]
    for i, (colname, title) in enumerate(charts):
        ch = interactive_12s_chart(wmini, colname, title)
        if ch is not None:
            cols[i].altair_chart(ch, use_container_width=True)

if alerts:
    for a in alerts:
        st.session_state.audit.append(
            {"t": int(df["Time"].iloc[-1]), "action": "alert_raise", **a}
        )
        alarm_banner(a)
else:
    st.markdown(
        """
        <div class="halo-ok-banner">
            <div style="font-weight:800;font-size:1.05rem;margin-bottom:4px;">No active alarms</div>
            <div>System nominal within current thresholds.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Show Top Suggested Action card on main page
tsa = top_suggested_action(df) if not df.empty else {}
if tsa:
    alts_html = "".join(f"<li>{x}</li>" for x in tsa["Alternatives"])
    st.markdown(
        f"""
        <div class="halo-tsa">
            <div style="font-weight:800;color:var(--halo-accent);font-size:1.1rem;margin-bottom:6px;">
                Top Suggested Action
            </div>
            <div><b>Suggested:</b> {tsa["Suggested"]}</div>
            <div><b>Rationale:</b> {tsa["Rationale"]}</div>
            <div><b>{tsa["Within"]}</b></div>
            <div><b>Confidence:</b> {tsa["Confidence"]}</div>
            <div><b>Anticipatory:</b> {tsa["Anticipatory"]}</div>
            <div><b>Alternatives:</b>
                <ul style="margin:6px 0 0 18px;">{alts_html}</ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ================== DATA QUALITY ==================
st.subheader("Data Quality (last 60s)")

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

    def badge(name, a, d, f):
        if a < 0.05:
            color = "#2f6b4b"
        elif a < 0.15:
            color = "#8a6a2a"
        else:
            color = "#8c3746"
        return (
            "<div class='halo-dq-badge' style='border-left-color:"
            + f"{color};'>"
            f"{name}: <b style='color:{color}'>{int(a*100)}%</b> artifact | "
            f"<b>{int(d*100)}%</b> dropout | "
            f"<b>{int(f*100)}%</b> flatline"
            "</div>"
        )

    q1.markdown(badge("HR", *rates["HR"]), unsafe_allow_html=True)
    q2.markdown(badge("SpO₂", *rates["SpO₂"]), unsafe_allow_html=True)
    q3.markdown(badge("MAP", *rates["MAP"]), unsafe_allow_html=True)
    q4.markdown(badge("EtCO₂", *rates["EtCO₂"]), unsafe_allow_html=True)
    q5.markdown(badge("RR", *rates["RR"]), unsafe_allow_html=True)
else:
    st.write("Collecting data...")

# ================== LIVE TRAJECTORY & RISK (kept) ==================
if not df.empty:
    st.subheader("Live Trajectory & Risk Analysis")
    sim_hz_current = int(st.session_state.get("sim_hz", 5))
    try:
        summary_md = live_summary_block(df, sim_hz_current)
    except Exception:
        summary_md = "Unable to generate full trajectory summary for this window."
    try:
        phys_narrative = physiology_story(df, window_sec=90, sim_hz=sim_hz_current)
    except Exception:
        phys_narrative = "Unable to compute physiology narrative for the most recent window."

    st.markdown(
        f"""
        <div class="halo-summary-card">
            <div style="font-weight:800;color:var(--halo-accent);font-size:1.1rem;
                        margin-bottom:6px;">🧭 Trajectory & Risk</div>
            <div style="margin-bottom:8px;">{summary_md.replace('\\n','<br>')}</div>
            <div style="font-weight:700;color:var(--halo-accent);margin-top:6px;
                        margin-bottom:4px;">Physiology Narrative (last 90s)</div>
            <div>{phys_narrative}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ================== CONVERSATIONAL ASSISTANT (TEXT + VOICE) ==================
st.divider()
st.subheader("Conversational Assistant")

colQ1, colQ2 = st.columns([3, 1])
q_text = colQ1.text_input(
    "Ask HALO a question (e.g., “Why is MAP falling?” or “Summarize the current situation.”)",
    key="halo_q_input",
    placeholder="Type your question here…",
)

voice_text = ""
with colQ2:
    st.markdown(
        "<div style='font-weight:800;color:var(--halo-accent);margin-bottom:4px;"
        "font-size:.95rem;'>🎤 Voice Dictation</div>",
        unsafe_allow_html=True,
    )
    if voice_available():
        voice_text = voice_widget(label="🎤 Click to speak")
    else:
        st.caption(
            "Voice capture not available (missing optional dependencies). "
            "You can still type questions."
        )

latest_transcript = st.session_state.get("last_voice_transcript", "")
if latest_transcript:
    st.markdown(
        f"""
        <div class="halo-voice-card">
            <div style="font-weight:800;margin-bottom:4px;font-size:1.05rem;
                        color:var(--halo-amber);">Detected speech</div>
            <div style="font-weight:600;">“{latest_transcript}”</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

final_q = voice_text if (voice_text and len(voice_text) > 0) else q_text
do_answer = (
    st.button("Get HALO Answer", type="primary", key="answer_btn") or bool(voice_text)
)

if do_answer and final_q:
    resp = answer_query(final_q, df, int(st.session_state.get("sim_hz", 5)))
    if "conversation_log" not in st.session_state:
        st.session_state.conversation_log = []
    st.session_state.conversation_log.append({"q": final_q, "a": resp["text"]})
    st.session_state.conversation_log = st.session_state.conversation_log[-3:]
    st.markdown(
        f"""
        <div class="halo-card">
            <div style="font-weight:800;color:var(--halo-accent);font-size:1.2rem;
                        margin-bottom:6px;">HALO Response</div>
            <div>{resp['text'].replace('\n','<br>')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if st.session_state.conversation_log:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<h4 style='color:var(--halo-accent);font-weight:800;margin-top:1rem;"
        "margin-bottom:.5rem;'>Recent HALO Conversation</h4>",
        unsafe_allow_html=True,
    )
    for i, entry in enumerate(reversed(st.session_state.conversation_log), 1):
        st.markdown(
            f"""
            <div class="halo-card-soft" style="border-left:5px solid var(--halo-accent);">
                <b style="color:var(--halo-accent);">Q{i}:</b> {entry['q']}<br>
                <b style="color:var(--halo-accent-soft);">A{i}:</b> {entry['a']}
            </div>
            """,
            unsafe_allow_html=True,
        )

# ================== AUDIT & EXPORT ==================
st.subheader("Alarm Audit Trail")
if len(st.session_state.audit) == 0:
    st.write("No events yet.")
else:
    st.dataframe(
        pd.DataFrame(st.session_state.audit),
        use_container_width=True,
        hide_index=True,
    )

st.subheader("Data")
if df.empty:
    st.write("No data yet.")
else:
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download HALO vitals.csv",
        data=csv,
        file_name="halo_vitals.csv",
        mime="text/csv",
    )
    st.caption("Export shows exactly what HALO observed in real time.")

# ================== SELF-REFRESH ==================
if st.session_state.running:
    time.sleep(1)
    st.rerun()
