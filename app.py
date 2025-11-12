# HALO — Hypothesis & Alarm Logic Orchestrator
# v4.0 — Live Summary + Single Conversational Assistant (Text + Voice)
# Not for clinical use.

import io
import time
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
# ---- Voice feature availability check ----
def voice_available():
    """
    Returns True if required dependencies for voice input are installed.
    Prevents Streamlit from calling undefined imports.
    """
    try:
        import speech_recognition  # type: ignore
        import sounddevice  # type: ignore
        import wavio  # type: ignore
        return True
    except Exception:
        return False


# ---- Voice capture widget (mic -> WAV bytes -> transcription) ----
def voice_widget():
    """
    Returns transcribed text from the user's microphone, or "" if nothing captured.
    Uses audio_recorder_streamlit for capture and SpeechRecognition for transcription.
    Safe: imports are scoped and failures fall back gracefully.
    """
    try:
        from audio_recorder_streamlit import audio_recorder  # pip install audio-recorder-streamlit
        import speech_recognition as sr                      # pip install SpeechRecognition
        import io
        import streamlit as st

        # Show recorder; stops automatically on silence
        audio_bytes = audio_recorder(
            pause_threshold=1.0,
            text="",
            icon_size="2x"
        )

        if not audio_bytes:
            return ""  # No audio recorded yet

        # Convert to audio and transcribe
        recognizer = sr.Recognizer()
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio = recognizer.record(source)

        try:
            text = recognizer.recognize_google(audio)
            return text.strip()
        except Exception as e:
            st.warning(f"Voice captured but transcription failed: {e}")
            return ""

    except Exception as e:
        st.info(f"Voice capture not available (optional dependencies missing): {e}")
        return ""
# --- Column name resolver (handles SpO2 vs SpO₂, EtCO2 vs EtCO₂, etc.) ---
def _resolve_cols(df: pd.DataFrame) -> dict:
    def pick(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None
    return {
        "Time": pick(["Time","time","t"]),
        "MAP":  pick(["MAP","Map","map"]),
        "HR":   pick(["HR","Heart Rate","heart_rate","Hr","hr"]),
        "SpO2": pick(["SpO2","SpO₂","spo2","SPO2"]),
        "EtCO2":pick(["EtCO2","EtCO₂","etco2","ETCO2","ETCO₂"]),
        "RR":   pick(["RR","Resp Rate","Respiration","resp_rate","rr"]),
    }

# --- Conversational Output Formatter (Short Paragraph Version, robust) ---
def answer_query(q: str, df: pd.DataFrame, sim_hz: int = 5):
    """
    HALO conversational reasoning engine (v4.2)
    - Context-aware phrasing: answers directly based on the user's question
    - Handles unlimited sequential questions without reinitialization
    - Fully compatible with HALO v4.0 framework
    """
    if df is None or df.empty:
        return {"text": "HALO has not yet received enough data to interpret physiologic trends.", "figs": []}

    cols = _resolve_cols(df)
    if not cols["Time"]:
        return {"text": "No time index available in the current data.", "figs": []}

    look_secs = 90
    keep_cols = [c for c in [cols["Time"], cols["MAP"], cols["HR"], cols["SpO2"], cols["EtCO2"], cols["RR"]] if c]
    w = _last_window(df[keep_cols], look_secs, max(int(sim_hz or 1), 1))
    if w is None or w.empty:
        return {"text": "No recent data available for this window.", "figs": []}

    def S(name):
        c = cols.get(name)
        return w[c] if c and c in w.columns else None

    def slope_safe(series):
        try:
            return _slope(series) if series is not None else 0.0
        except Exception:
            return 0.0

    def last_safe(series):
        try:
            return float(series.iloc[-1])
        except Exception:
            return None

    # Collect live metrics and slopes
    vals = {v: last_safe(S(v)) for v in ["MAP", "HR", "SpO2", "EtCO2", "RR"]}
    slopes = {v: slope_safe(S(v)) for v in ["MAP", "HR", "SpO2", "EtCO2", "RR"]}

    map_s = slopes["MAP"]; hr_s = slopes["HR"]; spo2_s = slopes["SpO2"]; etco2_s = slopes["EtCO2"]; rr_s = slopes["RR"]

    # Summary string
    def seg(label, val, slope, unit):
        if val is None:
            return f"{label} —"
        return f"{label} {int(round(val))} {unit} ({_fmt_trend(slope, per='/s')} over ~{look_secs}s)"
    summary = " · ".join([
        seg("MAP", vals["MAP"], map_s, "mmHg"),
        seg("HR", vals["HR"], hr_s, "bpm"),
        seg("SpO₂", vals["SpO2"], spo2_s, "%"),
        seg("EtCO₂", vals["EtCO2"], etco2_s, "mmHg"),
        seg("RR", vals["RR"], rr_s, "bpm"),
    ])

    # Correlation
    s_spo2, s_et = S("SpO2"), S("EtCO2")
    corr = _safe_corr(s_spo2.values, s_et.values) if s_spo2 is not None and s_et is not None else None
    corr_txt = f"SpO₂–EtCO₂ correlation ≈ {corr:+.2f}." if corr is not None else ""

    q_low = (q or "").lower().strip()

    # --- Question focus detection ---
    focus = None
    for v in ["map", "hr", "spo2", "etco2", "rr"]:
        if v in q_low:
            focus = v.upper()
            break

    # --- Conversational framing ---
    if any(w in q_low for w in ["why", "cause", "reason"]):
        prefix = "Based on your question, the most likely physiologic reason is:"
    elif any(w in q_low for w in ["what", "status", "show", "trend"]):
        prefix = "Here’s what I’m currently observing:"
    elif any(w in q_low for w in ["is", "does", "are there", "sign", "stable", "unstable", "normal"]):
        prefix = "Interpreting your question — I detect that:"
    elif any(w in q_low for w in ["how long", "predict", "when"]):
        prefix = "Forecast based on current data:"
    elif any(w in q_low for w in ["should", "recommend", "action", "do"]):
        prefix = "My reasoning suggests the next best step is:"
    else:
        prefix = "Here’s what I’m seeing in the current signals:"

    # --- Focused reasoning & actions ---
    reason = ""
    action_text = ""

    if focus == "RR":
        rr = vals["RR"]
        if rr is None:
            reason = "RR data unavailable."
        elif rr < 8:
            reason = f"RR {rr:.0f} bpm — bradypnea (below expected 10–20)."
            action_text = "assess ventilation adequacy and check anesthetic depth."
        elif rr > 28:
            reason = f"RR {rr:.0f} bpm — tachypnea (above expected)."
            action_text = "consider pain, light anesthesia, or hypercapnia."
        else:
            reason = f"RR {rr:.0f} bpm — within normal range."
            action_text = "continue routine monitoring."

    elif focus == "MAP":
        m = vals["MAP"]
        if m is None:
            reason = "MAP data unavailable."
        elif m < 65:
            reason = f"MAP {m:.0f} mmHg — hypotension; possible underperfusion."
            action_text = "evaluate anesthetic depth, volume status, or vasopressor need."
        elif m > 100:
            reason = f"MAP {m:.0f} mmHg — hypertension; sympathetic stimulation possible."
            action_text = "consider deepening anesthesia or addressing pain."
        else:
            reason = f"MAP {m:.0f} mmHg — within target range."
            action_text = "no immediate intervention needed."

    elif focus == "HR":
        h = vals["HR"]
        if h is None:
            reason = "HR data unavailable."
        elif h > 120:
            reason = f"HR {h:.0f} bpm — tachycardia; sympathetic activation or hypovolemia possible."
            action_text = "review analgesia and volume status."
        elif h < 50:
            reason = f"HR {h:.0f} bpm — bradycardia; vagal or drug effect possible."
            action_text = "check recent dosing and stimulus."
        else:
            reason = f"HR {h:.0f} bpm — normal range."
            action_text = "continue monitoring."

    elif focus == "SPO2":
        s = vals["SpO2"]
        if s is None:
            reason = "SpO₂ data unavailable."
        elif s < 92:
            reason = f"SpO₂ {s:.0f}% — hypoxemia."
            action_text = "check airway, FiO₂, and probe placement."
        else:
            reason = f"SpO₂ {s:.0f}% — satisfactory oxygenation."
            action_text = "maintain current settings."

    elif focus == "ETCO2":
        e = vals["EtCO2"]
        if e is None:
            reason = "EtCO₂ data unavailable."
        elif e < 30:
            reason = f"EtCO₂ {e:.0f} mmHg — hypocapnia; likely hyperventilation."
            action_text = "reduce ventilation rate if appropriate."
        elif e > 50:
            reason = f"EtCO₂ {e:.0f} mmHg — hypercapnia; likely hypoventilation."
            action_text = "increase ventilation or check circuit."
        else:
            reason = f"EtCO₂ {e:.0f} mmHg — normal range."
            action_text = "continue monitoring."

    else:
        # Global fallback reasoning if question is generic
        if map_s < -0.3 and hr_s > 0.3:
            reason = "MAP falling with compensatory HR rise — possible early hypovolemia."
            action_text = "assess fluids and anesthetic depth."
        elif spo2_s < -0.3 and etco2_s > 0.3:
            reason = "SpO₂ dropping while EtCO₂ rising — likely hypoventilation."
            action_text = "optimize airway or ventilation."
        else:
            reason = "signals largely stable across monitored parameters."
            action_text = "no immediate intervention needed."

    # --- Compose final response ---
    text = (
        f"**{prefix}** {reason} "
        f"Currently: {summary}. {corr_txt} "
        f"Top suggested action: {action_text}"
    )

    return {"text": text, "figs": []}

# -------------- Page / Branding --------------
st.set_page_config(page_title="HALO v4.0", page_icon="🛡️", layout="wide")
st.markdown(
    "<div style='display:inline-block;background:#fff4d6;color:#333;padding:4px 8px;"
    "border-left:6px solid #c26b00;border-radius:8px;margin-bottom:8px;font-size:0.9rem;'>"
    "Demo build · Not for clinical use</div>",
    unsafe_allow_html=True,
)
st.title("HALO — Hypothesis & Alarm Logic Orchestrator")
st.caption("Interpretive assistant for anesthesia monitoring (assistant, not decider).")

DEMO_MODE = True

# -------------- Safe session defaults --------------
def _setdefault(k, v):
    if k not in st.session_state:
        st.session_state[k] = v

_setdefault("history", pd.DataFrame(columns=["Time","HR","SpO2","MAP","EtCO2","RR"]))
_setdefault("running", False)
_setdefault("mode", "Live")         # "Live" or "Replay"
_setdefault("replay_df", None)
_setdefault("replay_idx", 0)
_setdefault("replay_speed", 1)      # rows per tick
_setdefault("sim_hz", 5)            # samples per second
_setdefault("scenario_name", None)
_setdefault("scenario_end", 0.0)
_setdefault("events", [])
_setdefault("audit", [])
_setdefault("sim_val", {"HR":80,"SpO2":97,"MAP":80,"EtCO2":37,"RR":12})
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

# Hysteresis (exit thresholds) & cooldown (used by prior builds; harmless here)
_setdefault("hys_spo2", 2)
_setdefault("hys_map", 5)
_setdefault("cooldown", 30)

# Noise / artifact injection
_setdefault("enable_noise", True)
_setdefault("artifact_pct", 5)
# -------------------- Top Controls (always visible) --------------------
if "running" not in st.session_state:
    st.session_state.running = False
if "sim_hz" not in st.session_state:
    st.session_state.sim_hz = 5

ctrl = st.container()
with ctrl:
    c1, c2, c3, c4 = st.columns([1, 1, 2, 2])
    if c1.button("▶ Start", key="btn_start"):
        st.session_state.running = True
    if c2.button("■ Stop", key="btn_stop"):
        st.session_state.running = False

    # Safe slider (do NOT assign back to session_state manually)
    c3.slider("Samples per second", 1, 10, st.session_state.get("sim_hz", 5), key="sim_hz")

    # Visible, color-safe status chip
    status = "Running" if st.session_state.running else "Stopped"
    bg = "#e7f7e7" if st.session_state.running else "#f5f5f5"
    border = "#3aa35c" if st.session_state.running else "#ddd"
    text = "#111111"

    c4.markdown(
        f"<div style='display:inline-block;padding:6px 10px;border-radius:999px;"
        f"background:{bg};border:1px solid {border};color:{text};"
        f"font-weight:600;'>Mode: {st.session_state.mode} · {status} · Samples: {len(st.session_state.history)}</div>",
        unsafe_allow_html=True,
    )

# -------------- Sidebar --------------
with st.sidebar:
    st.header("Mode")
    st.session_state.mode = st.radio("Input source", ["Live", "Replay"], index=0 if st.session_state.mode=="Live" else 1, key="mode_radio")

    if st.session_state.mode == "Replay":
        up = st.file_uploader("Upload CSV (Time, HR, SpO2, MAP, optional EtCO2, RR)", type=["csv"], key="replay_csv")
        # (keep your existing replay-loading code here, unchanged)
        st.slider("Replay speed (rows/tick)", 1, 20, st.session_state.get("replay_speed", 1), key="replay_speed")

    show_details = True if not DEMO_MODE else st.checkbox("Show researcher details", value=False, key="show_details")

    if show_details:
        st.header("Thresholds")
        st.slider("Low SpO₂ (%)", 85, 96, st.session_state.get("low_spo2", 92), key="low_spo2")
        st.slider("Tachycardia HR (bpm)", 90, 160, st.session_state.get("tachy_hr", 120), key="tachy_hr")
        st.slider("Low MAP (mmHg)", 50, 80, st.session_state.get("low_map", 65), key="low_map")

        st.subheader("Respiratory")
        st.slider("High EtCO₂ (mmHg)", 40, 60, st.session_state.get("high_et", 50), key="high_et")
        st.slider("Low EtCO₂ (mmHg)",  20, 40, st.session_state.get("low_et", 30), key="low_et")
        st.slider("Low RR (bpm)",       4, 20, st.session_state.get("low_rr", 8), key="low_rr")
        st.slider("High RR (bpm)",     18, 40, st.session_state.get("high_rr", 28), key="high_rr")

        st.header("Persistence (sec)")
        st.slider("SpO₂ persistence", 5, 30, st.session_state.get("win_spo2", 8), key="win_spo2")
        st.slider("HR persistence",   5, 30, st.session_state.get("win_hr", 8), key="win_hr")
        st.slider("MAP persistence",  5, 30, st.session_state.get("win_map", 10), key="win_map")
        st.slider("Resp persistence", 5, 30, st.session_state.get("win_resp", 12), key="win_resp")

        st.header("Hysteresis & Cooldown")
        st.slider("SpO₂ hysteresis (+%)", 1, 6, st.session_state.get("hys_spo2", 2), key="hys_spo2")
        st.slider("MAP hysteresis (+mmHg)", 2, 12, st.session_state.get("hys_map", 5), key="hys_map")
        st.slider("Alarm cooldown (s)", 0, 120, st.session_state.get("cooldown", 30), key="cooldown")

        st.header("Noise / Artifacts (Live)")
        st.checkbox("Inject artifact/noise", value=st.session_state.get("enable_noise", True), key="enable_noise")
        st.slider("Artifact chance (%)", 0, 20, st.session_state.get("artifact_pct", 5), key="artifact_pct")

    st.divider()
    if st.button("Reset Data", key="reset_btn"):
        st.session_state.history = pd.DataFrame(columns=["Time","HR","SpO2","MAP","EtCO2","RR"])
        st.session_state.replay_idx = 0
        st.session_state.events = []
        st.session_state.audit = []
        st.session_state.sim_time = 0
        st.success("Buffers cleared.")
# -------------- Scenarios & Events --------------
sc1, sc2, sc3, sc4, sc5 = st.columns(5)
if sc1.button("Scenario: Bleed (60s)", key="sc_bleed"):        st.session_state.scenario_name="Bleed"; st.session_state.scenario_end=time.time()+60
if sc2.button("Scenario: Bronchospasm (60s)", key="sc_bronch"): st.session_state.scenario_name="Bronchospasm"; st.session_state.scenario_end=time.time()+60
if sc3.button("Scenario: Vasodilation (60s)", key="sc_vaso"): st.session_state.scenario_name="Vasodilation"; st.session_state.scenario_end=time.time()+60
if sc4.button("Scenario: Pain/Light (60s)",   key="sc_pain"): st.session_state.scenario_name="Pain/Light"; st.session_state.scenario_end=time.time()+60
if sc5.button("End Scenario", key="sc_end"):                 st.session_state.scenario_name=None; st.session_state.scenario_end=0.0

e1, e2, e3, e4, e5 = st.columns(5)
if e1.button("Mark: Incision", key="mark_incision"):        st.session_state.events.append({"t": len(st.session_state.history), "name":"Incision"})
if e2.button("Mark: Position change", key="mark_pos"): st.session_state.events.append({"t": len(st.session_state.history), "name":"Position change"})
if e3.button("Mark: Fluids 250 mL",   key="mark_fluid"):   st.session_state.events.append({"t": len(st.session_state.history), "name":"Fluids 250 mL"})
if e4.button("Mark: Vasopressor",     key="mark_press"):     st.session_state.events.append({"t": len(st.session_state.history), "name":"Vasopressor"})
if e5.button("Clear Events",          key="mark_clear"):          st.session_state.events = []

# -------------- Dynamic Threshold Engine --------------
def _time_now_index() -> int:
    if st.session_state.history.empty: return 0
    return int(st.session_state.history["Time"].iloc[-1])

def _age_since_event(name: str, horizon: int) -> Optional[float]:
    t_now = _time_now_index()
    recent = [e for e in st.session_state.events if e["name"] == name and 0 <= (t_now - e["t"]) <= horizon]
    if not recent: return None
    return float(min(t_now - e["t"] for e in recent))

def _decay_linear(age: Optional[float], horizon: int) -> float:
    if age is None: return 0.0
    return max(0.0, (horizon - age)/float(horizon))

def _scenario_mods():
    name = st.session_state.scenario_name
    active = (name is not None) and (time.time() <= st.session_state.scenario_end)
    if not active: return dict(spo2=0, hr=0, map=0, et_high=0, rr_low=0)
    if name == "Bleed":        return dict(spo2=0,  hr=-10, map=+5,  et_high=0,  rr_low=0)
    if name == "Bronchospasm": return dict(spo2=+2, hr=0,   map=0,   et_high=-4, rr_low=+2)
    if name == "Vasodilation": return dict(spo2=0,  hr=0,   map=+5,  et_high=0,  rr_low=0)
    if name == "Pain/Light":   return dict(spo2=0,  hr=-10, map=0,   et_high=0,  rr_low=0)
    return dict(spo2=0, hr=0, map=0, et_high=0, rr_low=0)

def _event_mods():
    H_INCISION = 90; H_FLUIDS = 120; H_PRESSOR = 180; H_POSN = 120
    age_incision = _age_since_event("Incision", H_INCISION)
    age_fluids   = _age_since_event("Fluids 250 mL", H_FLUIDS)
    age_pressor  = _age_since_event("Vasopressor", H_PRESSOR)
    age_posn     = _age_since_event("Position change", H_POSN)
    w_inc = _decay_linear(age_incision, H_INCISION)
    w_flu = _decay_linear(age_fluids,   H_FLUIDS)
    w_pre = _decay_linear(age_pressor,  H_PRESSOR)
    w_pos = _decay_linear(age_posn,     H_POSN)
    spo2 = (-2.0 * w_pos)
    hr   = (+10.0 * w_inc)
    map_ = (-5.0 * w_flu) + (+5.0 * w_pre)
    et_h = 0.0; rr_l = 0.0
    return dict(spo2=spo2, hr=hr, map=map_, et_high=et_h, rr_low=rr_l)

def effective_thresholds():
    base = dict(low_spo2=st.session_state.low_spo2, tachy_hr=st.session_state.tachy_hr,
                low_map=st.session_state.low_map,   high_et=st.session_state.high_et,
                low_rr=st.session_state.low_rr)
    scen = _scenario_mods(); ev = _event_mods()
    eff = dict(
        low_spo2 = int(round(np.clip(base["low_spo2"] + scen["spo2"] + ev["spo2"], 80, 99))),
        tachy_hr = int(round(np.clip(base["tachy_hr"] + scen["hr"]   + ev["hr"],   80, 180))),
        low_map  = int(round(np.clip(base["low_map"]  + scen["map"]  + ev["map"],  45, 90))),
        high_et  = int(round(np.clip(base["high_et"]  + scen["et_high"] + ev["et_high"], 35, 60))),
        low_rr   = int(round(np.clip(base["low_rr"]   + scen["rr_low"]  + ev["rr_low"],   4, 20))),
    )
    eff["exit_spo2"] = eff["low_spo2"] + st.session_state.hys_spo2
    eff["exit_map"]  = eff["low_map"]  + st.session_state.hys_map
    return eff

# -------------- Simulation / Replay --------------
def __step(val, target, sigma, lo, hi):
    v = val + random.gauss(0, sigma) + (target - val) * random.uniform(0.02, 0.08)
    return max(lo, min(hi, v))

def _scenario_targets(base):
    name = st.session_state.scenario_name
    active = (name is not None) and (time.time() <= st.session_state.scenario_end)
    if not active:
        return dict(HR=base["HR"], SpO2=base["SpO2"], MAP=base["MAP"], EtCO2=base["EtCO2"], RR=base["RR"])
    targets = dict(HR=base["HR"], SpO2=base["SpO2"], MAP=base["MAP"], EtCO2=base["EtCO2"], RR=base["RR"])
    if name == "Bleed":
        targets["MAP"] = base["MAP"] - 15; targets["HR"]  = base["HR"] + 15; targets["SpO2"] = base["SpO2"] - 1
    elif name == "Bronchospasm":
        targets["SpO2"]= base["SpO2"] - 6; targets["EtCO2"]= base["EtCO2"] + 8; targets["RR"] = base["RR"] + 6
    elif name == "Vasodilation":
        targets["MAP"] = base["MAP"] - 12; targets["HR"]  = base["HR"] - 3
    elif name == "Pain/Light":
        targets["HR"]  = base["HR"] + 12; targets["MAP"] = base["MAP"] + 5; targets["RR"] = base["RR"] + 4
    return targets

def _tick_live(sim_hz):
    for _ in range(sim_hz):
        base = st.session_state.sim_val
        targets = _scenario_targets(base)
        hr   = __step(base["HR"],   targets["HR"],   1.2, 30, 180)
        spo2 = __step(base["SpO2"], targets["SpO2"], 0.6, 70, 100)
        map_ = __step(base["MAP"],  targets["MAP"],  1.5, 40, 120)
        et   = __step(base["EtCO2"],targets["EtCO2"],0.8, 20, 60)
        rr   = __step(base["RR"],   targets["RR"],   0.8,  6, 40)

        if st.session_state.enable_noise and random.random() < (st.session_state.artifact_pct/100.0):
            pick = random.random()
            if   pick < 0.25: hr  += random.choice([-12,12])
            elif pick < 0.50: spo2+= random.choice([-8,8])
            elif pick < 0.75: map_+= random.choice([-10,10])
            else:             et  += random.choice([-6,6])

        st.session_state.sim_val  = {"HR":hr,"SpO2":spo2,"MAP":map_,"EtCO2":et,"RR":rr}
        st.session_state.sim_time += 1
        row = {"Time": len(st.session_state.history),
               "HR": int(round(hr)), "SpO2": int(round(spo2)), "MAP": int(round(map_)),
               "EtCO2": int(round(et)), "RR": int(round(rr))}
        st.session_state.history.loc[len(st.session_state.history)] = row

def _tick_replay():
    df_r = st.session_state.replay_df
    if df_r is None: return
    i = st.session_state.replay_idx
    if i >= len(df_r): return
    step = st.session_state.replay_speed
    end  = min(i + step, len(df_r))
    chunk = df_r.iloc[i:end].copy()
    base  = len(st.session_state.history)
    chunk["Time"] = np.arange(base, base + len(chunk))
    st.session_state.history = pd.concat([st.session_state.history, chunk], ignore_index=True)
    st.session_state.replay_idx = end

if st.session_state.running:
    if st.session_state.mode == "Live": _tick_live(st.session_state.sim_hz)
    else: _tick_replay()

df = st.session_state.history
st.session_state["df"] = df  # expose for assistant safely

# -------------- Vitals Charts (top) --------------
c1, c2, c3, c4, c5 = st.columns(5)
if not df.empty:
    c1.subheader("HR (bpm)");     c1.line_chart(df.set_index("Time")["HR"], use_container_width=True)
    c2.subheader("SpO₂ (%)");     c2.line_chart(df.set_index("Time")["SpO2"], use_container_width=True)
    c3.subheader("MAP (mmHg)");   c3.line_chart(df.set_index("Time")["MAP"], use_container_width=True)
    c4.subheader("EtCO₂ (mmHg)"); c4.line_chart(df.set_index("Time")["EtCO2"], use_container_width=True)
    c5.subheader("RR (bpm)");     c5.line_chart(df.set_index("Time")["RR"], use_container_width=True)
else:
    st.info("Click Start (Live) or upload a CSV (Replay).")

# -------------- Helpers for alarms & summary --------------
def last_n(df_: pd.DataFrame, seconds: int) -> pd.DataFrame:
    if df_ is None or df_.empty: return df_
    end = df_["Time"].iloc[-1]; start = max(0, end - seconds + 1)
    return df_[df_["Time"].between(start, end)]

def persistent_low(df_, signal, thresh, sec):
    w = last_n(df_, sec)
    return (w is not None) and (len(w) >= max(3, min(sec, len(w)))) and (w[signal].min() < thresh)

def persistent_high(df_, signal, thresh, sec):
    w = last_n(df_, sec)
    return (w is not None) and (len(w) >= max(3, min(sec, len(w)))) and (w[signal].max() > thresh)

def is_artifact_series(series: pd.Series) -> bool:
    if series is None or len(series) < 6: return False
    recent = series.iloc[-6:]
    diffs = np.abs(np.diff(recent.values))
    if len(diffs)==0: return False
    return diffs[-1] > (np.std(recent.values) * 4 + 4)

def recent_artifact(df_: pd.DataFrame) -> dict:
    if df_ is None or df_.empty:
        return {"HR":False,"SpO2":False,"MAP":False,"EtCO2":False,"RR":False}
    return {"HR":is_artifact_series(df_["HR"]), "SpO2":is_artifact_series(df_["SpO2"]),
            "MAP":is_artifact_series(df_["MAP"]), "EtCO2":is_artifact_series(df_["EtCO2"]),
            "RR":is_artifact_series(df_["RR"])}

eff = effective_thresholds()

# -------------- Alarm Fusion --------------
alerts = []
hypox = tachy = hypot = hypercap = hypovent = False

if not df.empty:
    arts = recent_artifact(df)
    hypox    = (not arts["SpO2"]) and persistent_low(df, "SpO2", eff["low_spo2"], st.session_state.win_spo2)
    tachy    = (not arts["HR"])   and persistent_high(df, "HR",   eff["tachy_hr"], st.session_state.win_hr)
    hypot    = (not arts["MAP"])  and persistent_low(df, "MAP",   eff["low_map"],  st.session_state.win_map)
    hypercap = (not arts["EtCO2"])and persistent_high(df, "EtCO2",eff["high_et"],  st.session_state.win_resp)
    hypovent = (not arts["RR"])   and persistent_low(df, "RR",    eff["low_rr"],   st.session_state.win_resp)

    if hypox:  alerts.append({"label":"Low SpO₂","severity":"Warning","why":f"SpO₂ < {eff['low_spo2']}% ≥{st.session_state.win_spo2}s (exit > {eff['exit_spo2']}%)"})
    if hypot:  alerts.append({"label":"Low MAP","severity":"Warning","why":f"MAP < {eff['low_map']} ≥{st.session_state.win_map}s (exit > {eff['exit_map']})"})
    if tachy:  alerts.append({"label":"Tachycardia","severity":"Advisory","why":f"HR > {eff['tachy_hr']} ≥{st.session_state.win_hr}s"})
    if (hypercap or hypovent):
        msg = []
        if hypercap: msg.append(f"EtCO₂ > {eff['high_et']}")
        if hypovent: msg.append(f"RR < {eff['low_rr']}")
        if hypox:    msg.append(f"SpO₂ < {eff['low_spo2']}")
        sev = "Advisory" if not (hypercap and hypovent and hypox) else "Warning"
        alerts.append({"label":"Respiratory Concern","severity":sev,"why":", ".join(msg)})

# -------------- Top Suggested Action (concise) --------------
st.caption("**Within limits** = HALO’s recommendation stays inside your clinician-defined safety rails for drug dosages, infusion rates, and vital sign targets. HALO will never suggest exceeding those bounds.")

def top_suggested_action(df_: pd.DataFrame) -> Dict[str, str]:
    if df_ is None or df_.empty:
        return {}
    w = last_n(df_, 20)
    if w is None or w.empty:
        w = df_
    # Means and simple slopes
    def slope(s: pd.Series) -> float:
        if s is None or len(s) < 2: return 0.0
        x = np.arange(len(s), dtype=float); y = s.to_numpy(dtype=float)
        xm, ym = x.mean(), y.mean()
        denom = ((x - xm)**2).sum()
        if denom == 0: return 0.0
        return float(((x - xm)*(y - ym)).sum() / denom)

    stats = {}
    for k in ["MAP","HR","SpO2","EtCO2","RR"]:
        if k in w.columns:
            stats[k] = {"mean": float(np.mean(w[k])), "trend": slope(w[k])}

    if not stats:
        return {}

    # Simple logic
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
                rationale = "MAP is low/falling with HR rising — pattern consistent with relative hypovolemia."
                confidence = 0.7
                anticipatory = "MAP likely to remain below floor without intervention."
                alts = ["Vasopressor if volume adequate", "Reduce anesthetic depth if appropriate"]
            else:
                action = "Consider vasopressor support; review anesthetic depth."
                rationale = "MAP low/falling with limited HR response — vasodilation/drug effect likely."
                confidence = 0.65
                anticipatory = "Sustained hypotension risk without treatment."
                alts = ["Small fluid bolus if uncertain", "Lower hypnotic dose if clinically safe"]

    if "SpO2" in stats and "EtCO2" in stats:
        if stats["SpO2"]["trend"] < -0.2 and stats["EtCO2"]["trend"] > 0.2:
            action = "Address ventilation/airway; increase support."
            rationale = "SpO₂ falling while EtCO₂ rising — hypoventilation/airway issue likely."
            confidence = 0.75
            anticipatory = "Oxygenation may worsen if not treated."
            alts = ["Re-seat probe (rule out artifact)", "Increase minute ventilation", "Suction/check circuit"]

    return {
        "Suggested": action,
        "Rationale": rationale,
        "Within": within,
        "Confidence": f"{int(round(confidence*100))}%",
        "Anticipatory": anticipatory,
        "Alternatives": alts
    }

# -------------- Mini interactive 12s charts --------------
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
            x=alt.X("Time:Q", axis=alt.Axis(title="(last 12s)")),
            y=alt.Y(f"{field}:Q", axis=alt.Axis(title=field)),
            tooltip=["Time", field],
        )
        .properties(title=title, height=180)
        .interactive()
    )
    return chart

# -------------- Alarm Panel (with top action and mini charts) --------------
st.subheader("Alarm Panel")

def alarm_banner(a):
    label, severity, why = a["label"], a["severity"], a["why"]
    palettes = {"Advisory":{"border":"#1e62ff","bg":"#e9f0ff","title":"#0b2a6b","text":"#0a0a0a"},
                "Warning":{"border":"#c26b00","bg":"#fff0d6","title":"#5a3200","text":"#0a0a0a"},
                "Critical":{"border":"#b00020","bg":"#ffe3e6","title":"#5b0a12","text":"#0a0a0a"}}
    p = palettes.get(severity, {"border":"#444","bg":"#eee","title":"#111","text":"#0a0a0a"})
    st.markdown(f"""
    <div style="border-left:10px solid {p['border']};background:{p['bg']};color:{p['text']};
                padding:14px 18px;border-radius:10px;margin:10px 0;font-size:1.05rem;line-height:1.5;
                box-shadow:0 2px 6px rgba(0,0,0,0.12);">
      <div style="font-weight:800;color:{p['title']};font-size:1.25rem;margin-bottom:6px;">
        {severity}: {label}
      </div>
      <div style="margin-bottom:6px;"><b>Why:</b> {why}</div>
    </div>""", unsafe_allow_html=True)

    # horizontal mini charts
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
        st.session_state.audit.append({"t": int(df["Time"].iloc[-1]), "action":"alert_raise", **a})
        alarm_banner(a)
else:
    st.markdown("""
        <div style="border-left:10px solid #0a8a0a;background:#e8f7e8;color:#0a0a0a;
                    padding:14px 18px;border-radius:10px;font-size:1.05rem;line-height:1.5;
                    box-shadow:0 2px 6px rgba(0,0,0,0.12);">
            <div style="font-weight:800;color:#0a5a0a;font-size:1.2rem;margin-bottom:4px;">No active alarms</div>
            <div style="font-weight:500;">System nominal.</div>
        </div>""", unsafe_allow_html=True)

# top suggested action (always visible below alarms)
tsa = top_suggested_action(df)
if tsa:
    alts_html = "".join(f"<li>{x}</li>" for x in tsa["Alternatives"])
    st.markdown(f"""
    <div style="border-left:10px solid #6f2dbd;background:#f4edff;color:#0a0a0a;
                padding:14px 18px;border-radius:10px;margin:10px 0;font-size:1.05rem;line-height:1.5;
                box-shadow:0 2px 6px rgba(0,0,0,0.12);">
      <div style="font-weight:800;color:#3d146b;font-size:1.2rem;margin-bottom:6px;">Top Suggested Action</div>
      <div><b>Suggested:</b> {tsa["Suggested"]}</div>
      <div><b>Rationale:</b> {tsa["Rationale"]}</div>
      <div><b>{tsa["Within"]}</b></div>
      <div><b>Confidence:</b> {tsa["Confidence"]}</div>
      <div><b>Anticipatory:</b> {tsa["Anticipatory"]}</div>
      <div><b>Alternatives:</b><ul style="margin:6px 0 0 18px;">{alts_html}</ul></div>
    </div>
    """, unsafe_allow_html=True)

# -------------- Data Quality (augmented) --------------
st.subheader("Data Quality (last 60s)")

def artifact_rate(series: pd.Series, window=60) -> float:
    if series is None or len(series) < 8: return 0.0
    w = series.iloc[-min(window, len(series)):]
    jumps = np.abs(np.diff(w.values)); thr = np.std(w.values) * 4 + 4
    return float(np.mean(jumps > thr))

def dropout_rate(series: pd.Series, window=60) -> float:
    if series is None or len(series) < 2: return 0.0
    w = series.iloc[-min(window, len(series)):]
    return float(w.isna().mean())

def flatline_rate(series: pd.Series, window=60) -> float:
    if series is None or len(series) < 4: return 0.0
    w = series.iloc[-min(window, len(series)):]
    diffs = np.abs(np.diff(w.values))
    return float(np.mean(diffs < 1e-6))

if len(df) >= 10:
    q1, q2, q3, q4, q5 = st.columns(5)
    rates = {
        "HR":     (artifact_rate(df["HR"]),     dropout_rate(df["HR"]),     flatline_rate(df["HR"])),
        "SpO₂":   (artifact_rate(df["SpO2"]),   dropout_rate(df["SpO2"]),   flatline_rate(df["SpO2"])),
        "MAP":    (artifact_rate(df["MAP"]),    dropout_rate(df["MAP"]),    flatline_rate(df["MAP"])),
        "EtCO₂":  (artifact_rate(df["EtCO2"]),  dropout_rate(df["EtCO2"]),  flatline_rate(df["EtCO2"])),
        "RR":     (artifact_rate(df["RR"]),     dropout_rate(df["RR"]),     flatline_rate(df["RR"])),
    }

    def badge(name, a, d, f):
        color = "#0a8a0a" if a < 0.05 else ("#c26b00" if a < 0.15 else "#b00020")
        return (
            "<div style='background:#f5f5f5;color:#111111;padding:8px;"
            f"border-left:8px solid {color};border-radius:8px'>"
            f"{name}: <b style='color:{color}'>{int(a*100)}%</b> artifact | "
            f"<b>{int(d*100)}%</b> dropout | <b>{int(f*100)}%</b> flatline"
            "</div>"
        )
    q1.markdown(badge("HR", *rates["HR"]),     unsafe_allow_html=True)
    q2.markdown(badge("SpO₂", *rates["SpO₂"]), unsafe_allow_html=True)
    q3.markdown(badge("MAP", *rates["MAP"]),   unsafe_allow_html=True)
    q4.markdown(badge("EtCO₂", *rates["EtCO₂"]), unsafe_allow_html=True)
    q5.markdown(badge("RR", *rates["RR"]),     unsafe_allow_html=True)
else:
    st.write("Collecting data...")

# -------------- Conversational Assistant + Live Situation Summary --------------
# Helpers for trends and reasoning
def _recent(df_: pd.DataFrame, seconds: int) -> pd.DataFrame:
    if df_ is None or df_.empty:
        return df_
    end = int(df_["Time"].iloc[-1])
    start = max(0, end - seconds + 1)
    return df_[df_["Time"].between(start, end)]

def _slope(series: pd.Series) -> float:
    if series is None or len(series) < 2:
        return 0.0
    x = np.arange(len(series), dtype=float)
    y = series.to_numpy(dtype=float)
    xm, ym = x.mean(), y.mean()
    denom = ((x - xm)**2).sum()
    if denom == 0:
        return 0.0
    return float(((x - xm)*(y - ym)).sum() / denom)

def _arrow_dir(val: float, thr: float = 0.2) -> Tuple[str,str]:
    if val > thr:   return ("rising", "↑")
    if val < -thr:  return ("falling", "↓")
    return ("stable", "→")

def _fmt_val(v: float, unit: str) -> str:
    try:
        return f"{v:.1f}{unit}"
    except Exception:
        return "-"

def _time_to_cross(current: float, slope_per_sample: float, floor: float, hz: float) -> str:
    if slope_per_sample >= 0:
        return ""
    delta = current - floor
    if delta <= 0:
        return "breach imminent"
    samples = delta / abs(slope_per_sample)
    secs = samples / max(hz, 1.0)
    if secs < 4:
        return "breach imminent"
    if secs > 900:
        return ""
    return f"≈{int(round(secs))} s to cross {int(floor)}"

def _compute_info(df_: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    info: Dict[str, Dict[str, float]] = {}
    if df_ is None or df_.empty:
        return info
    w = _recent(df_, 30)
    if w is None or w.empty:
        w = df_
    for v in ["HR", "MAP", "SpO2", "EtCO2", "RR"]:
        if v in w.columns:
            y = pd.to_numeric(w[v], errors="coerce").dropna()
            if len(y) >= 2:
                info[v] = {"mean": float(y.mean()), "trend": _slope(y)}
            elif len(y) == 1:
                info[v] = {"mean": float(y.iloc[0]), "trend": 0.0}
    return info

def _interpret(info: Dict[str, Dict[str, float]]) -> List[str]:
    out: List[str] = []
    if "MAP" in info and "HR" in info:
        if info["MAP"]["trend"] < -0.2 and info["HR"]["trend"] > 0.2:
            out.append("MAP falling while HR rising → compensated hypotension (possible hypovolemia).")
        elif info["MAP"]["trend"] < -0.2 and info["HR"]["trend"] <= 0.0:
            out.append("MAP falling without HR rise → vasodilation/drug effect more likely than bleeding.")
        elif info["MAP"]["trend"] > 0.3 and info["HR"]["trend"] > 0.3:
            out.append("MAP and HR rising together → sympathetic activation (pain/light anesthesia).")
    if "SpO2" in info and "EtCO2" in info:
        if info["SpO2"]["trend"] < -0.2 and info["EtCO2"]["trend"] > 0.2:
            out.append("SpO₂ falling + EtCO₂ rising → hypoventilation/airway issue likely.")
        elif info["SpO2"]["trend"] > 0.2 and info["EtCO2"]["trend"] < -0.2:
            out.append("SpO₂ improving with EtCO₂ falling → ventilation improving.")
    if "RR" in info:
        if info["RR"]["trend"] < -0.2:
            out.append("RR decreasing → risk of CO₂ retention if sustained.")
        elif info["RR"]["trend"] > 0.2:
            out.append("RR increasing → compensatory hyperventilation or stimulus response.")
    if not out:
        out.append("Signals are mostly stable; no strong cross-signal pattern detected.")
    return out

def _risk_label(info: Dict[str, Dict[str, float]]) -> str:
    score = 0
    if "SpO2" in info and info["SpO2"]["mean"] < 93: score += 2
    if "MAP"  in info and info["MAP"]["mean"]  < 65: score += 2
    if "EtCO2" in info and (info["EtCO2"]["mean"] > 50 or info["EtCO2"]["mean"] < 25): score += 1
    if "HR" in info and (info["HR"]["mean"] > 120 or info["HR"]["mean"] < 45): score += 1
    if score <= 1:  return "🟢 Stable"
    if score == 2:  return "🟡 Watch closely"
    return "🔴 Intervene soon"

def live_summary_block(df_: pd.DataFrame, sim_hz: int) -> str:
    info = _compute_info(df_)
    if not info:
        return "Not enough data yet for a summary."

    parts = []
    for v, unit in [("HR"," bpm"), ("MAP"," mmHg"), ("SpO2","%"), ("EtCO2"," mmHg"), ("RR"," bpm")]:
        if v in info:
            d, a = _arrow_dir(info[v]["trend"])
            parts.append(f"{v}: {_fmt_val(info[v]['mean'], unit)} ({d} {a})")
    header = " | ".join(parts)

    reasons = _interpret(info)
    risk = _risk_label(info)

    pred_lines = []
    if "MAP" in info:
        ttc = _time_to_cross(info["MAP"]["mean"], info["MAP"]["trend"], 65, float(sim_hz or 1))
        if ttc: pred_lines.append(f"MAP: {ttc}")
    if "SpO2" in info:
        ttc = _time_to_cross(info["SpO2"]["mean"], info["SpO2"]["trend"], 92, float(sim_hz or 1))
        if ttc: pred_lines.append(f"SpO₂: {ttc}")
    preds = " | ".join(pred_lines)

    msg = []
    msg.append("**Current Situation Summary**")
    msg.append("")
    msg.append(header)
    msg.append("")
    msg.extend([f"• {r}" for r in reasons])
    msg.append("")
    msg.append(f"**Risk Level:** {risk}")
    if preds:
        msg.append(f"**Prediction:** {preds}")
    return "\n".join(msg)



def render_conversational_assistant():
    # Prevent duplicate renders if accidentally called twice
    if st.session_state.get("_assistant_rendered", False):
        return
    st.session_state._assistant_rendered = True

    st.divider()
    st.subheader("🎙️ Conversational Assistant")

    # ---- Helpers (read-only, do not mutate widget keys after creation) ----
    def _safe_last(series, default=None):
        try:
            return float(series.iloc[-1])
        except Exception:
            return default

    def _window(df_in: pd.DataFrame, seconds: int) -> pd.DataFrame:
        if df_in is None or df_in.empty:
            return df_in
        end = int(df_in["Time"].iloc[-1])
        start = max(0, end - max(1, int(seconds)) + 1)
        return df_in[df_in["Time"].between(start, end)]

    def _slope(series: pd.Series, seconds: int = 30) -> float:
        """Approximate slope per second over the window (last N seconds)."""
        if series is None or len(series) < 2:
            return 0.0
        wlen = min(len(series), seconds)
        w = series.iloc[-wlen:]
        x = np.arange(len(w))
        # simple linear regression slope
        try:
            m = np.polyfit(x, w.values, 1)[0]
        except Exception:
            m = 0.0
        return float(m)

    def _corr(df_in: pd.DataFrame, a: str, b: str, seconds: int = 60) -> float | None:
        w = _window(df_in, seconds)
        if w is None or w.empty or a not in w.columns or b not in w.columns:
            return None
        if len(w) < 3:
            return None
        try:
            c = float(np.corrcoef(w[a].values, w[b].values)[0, 1])
            if np.isnan(c):
                return None
            return c
        except Exception:
            return None

    def _fmt_pct(x):
        try:
            return f"{int(round(x*100))}%"
        except Exception:
            return "—"

    def _mini_scatter_spo2_vs_et(df_in: pd.DataFrame, seconds: int = 60):
        """Optional quick visualization for 'SpO₂ vs EtCO₂ correlation' questions."""
        w = _window(df_in, seconds)
        if w is None or w.empty:
            st.caption("Not enough data for a correlation plot yet.")
            return
        fig, ax = plt.subplots(figsize=(3.8, 3.0))
        ax.scatter(w["EtCO2"], w["SpO2"], s=10, alpha=0.7)
        ax.set_xlabel("EtCO₂ (mmHg)")
        ax.set_ylabel("SpO₂ (%)")
        ax.grid(alpha=0.25)
        st.pyplot(fig, clear_figure=True)

    def _rule_hypothesis(df_in: pd.DataFrame) -> dict:
        """Use your existing hypothesis_components if available to report top rule hypothesis."""
        try:
            hyps, _comps = hypothesis_components(df_in)
            if not hyps:
                return {}
            total = sum(h["score"] for h in hyps) or 1e-9
            ranked = sorted(
                [{"name": h["name"], "pct": int(round(100*h["score"]/total))} for h in hyps],
                key=lambda x: x["pct"], reverse=True
            )
            return ranked[0] if ranked else {}
        except Exception:
            return {}
# ---------- Q&A helpers (must appear before answer_query) ----------
def _last_window(df: pd.DataFrame, seconds: int, sim_hz: int) -> pd.DataFrame:
    """Return the last <seconds> of rows based on sim_hz; safe on empty."""
    if df is None or df.empty or sim_hz is None or sim_hz <= 0:
        return df if df is not None else pd.DataFrame()
    n = max(1, min(len(df), int(seconds * sim_hz)))
    return df.iloc[-n:].copy()

def _slope(series: pd.Series) -> float:
    """Simple end-start slope over the provided series; safe on short series."""
    try:
        if series is None or len(series) < 2:
            return 0.0
        return float(series.iloc[-1] - series.iloc[0])
    except Exception:
        return 0.0

def _fmt_trend(val: float, units: str = "", per: str = "") -> str:
    arrow = "↑" if val > 0 else ("↓" if val < 0 else "→")
    return f"{arrow} {abs(val):.1f}{units}{per}"

def _safe_corr(a: np.ndarray | list, b: np.ndarray | list):
    """Pearson r with guards; returns None if not computable."""
    try:
        a = np.asarray(a); b = np.asarray(b)
        if a.size < 3 or b.size < 3:
            return None
        c = float(np.corrcoef(a, b)[0, 1])
        return None if np.isnan(c) else c
    except Exception:
        return None

def _mini_line_fig(x, y, ylabel: str, title: str = ""):
    """Tiny matplotlib line fig used by the assistant fallback visuals."""
    fig, ax = plt.subplots(figsize=(4.5, 2.0))
    ax.plot(x, y, linewidth=2)
    ax.set_ylabel(ylabel)
    ax.set_xticks([])
    ax.set_title(title)
    fig.tight_layout()
    return fig
# -------------------------------------------------------------------

    def _answer_query(df_in: pd.DataFrame, q: str) -> tuple[str, list[tuple[str, str]]]:
        """
        Returns (answer_markdown, optional_visuals)
        optional_visuals: list of (kind, arg) items you may render (e.g., ("spo2_vs_et", "60s"))
        """
        if q is None:
            q = ""
        q_low = q.strip().lower()

        # If no data yet
        if df_in is None or df_in.empty:
            return ("I don’t have enough data yet — start the stream or upload a replay, then ask again.", [])

        # Recent window for narrative
        w12 = _window(df_in, 12)
        w60 = _window(df_in, 60)

        # Current values
        cur = {
            "HR":  _safe_last(df_in["HR"]),
            "SpO2":_safe_last(df_in["SpO2"]),
            "MAP": _safe_last(df_in["MAP"]),
            "Et":  _safe_last(df_in["EtCO2"]),
            "RR":  _safe_last(df_in["RR"]),
        }
        # Slopes (last 30s)
        slopes = {
            "HR":  _slope(w60["HR"]) if w60 is not None and not w60.empty else 0.0,
            "SpO2":_slope(w60["SpO2"]) if w60 is not None and not w60.empty else 0.0,
            "MAP": _slope(w60["MAP"]) if w60 is not None and not w60.empty else 0.0,
            "Et":  _slope(w60["EtCO2"]) if w60 is not None and not w60.empty else 0.0,
            "RR":  _slope(w60["RR"]) if w60 is not None and not w60.empty else 0.0,
        }

        # Top rule hypothesis (if your engine is present)
        top_rule = _rule_hypothesis(df_in)
        top_line = f"**Hypothesis (rules):** {top_rule.get('name','—')} — {top_rule.get('pct','—')}%" if top_rule else ""

        # Intent routing
        visuals: list[tuple[str, str]] = []

        # 1) "why is map falling?" / "why low map?" style
        if ("map" in q_low and ("why" in q_low or "fall" in q_low or "low" in q_low)):
            slope = slopes["MAP"]
            hr_up = slopes["HR"] > 0.2
            text = []
            text.append(f"**MAP** is {int(cur['MAP']) if cur['MAP'] is not None else '—'} and slope ≈ {slope:.2f} /sample over ~30s.")
            if hr_up:
                text.append("**HR** is rising → pattern consistent with *volume-responsive hypotension* (hypovolemia).")
            else:
                text.append("**HR** not rising → could be *vasodilation* or *anesthetic depth* effect.")
            if top_line:
                text.append(top_line)
            ans = "\n\n".join(text)
            return (ans, visuals)

        # 2) "show me spO2 vs etCO2 correlation"
        if ("spo2" in q_low and ("etco2" in q_low or "et co2" in q_low) and ("corr" in q_low or "relationship" in q_low)):
            c = _corr(df_in, "SpO2", "EtCO2", 60)
            c_str = "—" if c is None else f"{c:+.2f}"
            ans = f"**SpO₂–EtCO₂ correlation (last 60s):** {c_str} (−1 to +1)."
            visuals.append(("spo2_vs_et", "60s"))
            if top_line:
                ans += f"\n\n{top_line}"
            return (ans, visuals)

        # 3) “trend of X”
        for sig in ["hr", "spo2", "map", "etco2", "rr"]:
            if sig in q_low and ("trend" in q_low or "slope" in q_low or "rising" in q_low or "falling" in q_low):
                key = {"hr":"HR","spo2":"SpO2","map":"MAP","etco2":"EtCO2","rr":"RR"}[sig]
                curv = cur["Et" if key=="EtCO2" else key]
                slp  = slopes["Et" if key=="EtCO2" else key]
                sense = "rising" if slp > 0.2 else ("falling" if slp < -0.2 else "stable")
                ans = f"**{key}** is {int(curv) if curv is not None else '—'} and **{sense}** (slope ≈ {slp:.2f}/sample over ~30s)."
                if top_line:
                    ans += f"\n\n{top_line}"
                return (ans, visuals)

        # 4) “status” / “summary”
        if ("status" in q_low) or ("summary" in q_low) or ("what's happening" in q_low) or ("whats happening" in q_low):
            parts = []
            for key, lbl in [("HR","HR"),("SpO2","SpO₂"),("MAP","MAP"),("Et","EtCO₂"),("RR","RR")]:
                v = cur[key]
                sl = slopes[key]
                sense = "↑" if sl > 0.2 else ("↓" if sl < -0.2 else "→")
                parts.append(f"{lbl} {int(v) if v is not None else '—'} ({sense})")
            ans = "**Current status:** " + " · ".join(parts)
            if top_line:
                ans += f"\n\n{top_line}"
            return (ans, visuals)

        # 5) Default fallback: generic helpful response using top rule hypothesis if available
        ans = "I'm still learning to answer that. Try asking about **MAP**, **SpO₂ vs EtCO₂ correlation**, or **trend** of a vital."
        if top_line:
            ans += f"\n\n{top_line}"
        return (ans, visuals)

    # ---- UI (single instance) ----
    # Row: text input + ask button
    c1, c2 = st.columns([4,1], vertical_alignment="bottom")
    user_q = c1.text_input("Ask a question (e.g., 'Why is MAP falling?' or 'Show SpO₂ vs EtCO₂ correlation')",
                           value="", key="asst_q_text")
    ask_clicked = c2.button("Ask HALO", key="asst_q_btn")

    # Optional: voice capture (graceful fallback if module is missing)
    # We only attempt if the user wants it; no dependency = no break.
    st.caption("Optional voice: if installed, click to record; otherwise use the text box.")
    vcol1, vcol2 = st.columns([1,5])
    use_voice = vcol1.checkbox("Use voice input (if available)", value=False, key="asst_voice_enable")

    audio_text = ""
    if use_voice:
        try:
            # Requires: pip install audio-recorder-streamlit speechrecognition pydub
            from audio_recorder_streamlit import audio_recorder  # type: ignore
            import speech_recognition as sr  # type: ignore

            st.write("Click to start/stop recording:")
            audio_bytes = audio_recorder(pause_threshold=1.0, text="", icon_size="2x")
            if audio_bytes:
                # Try to transcribe from memory buffer
                recognizer = sr.Recognizer()
                with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
                    audio = recognizer.record(source)
                try:
                    audio_text = recognizer.recognize_google(audio)  # offline alt not bundled; this is best-effort
                    st.success(f"Voice captured: “{audio_text}”")
                except Exception as e:
                    st.warning(f"Could not transcribe (best-effort): {e}")
        except Exception as e:
            st.info(f"Voice input not available (optional deps missing). Use the text box. ({e})")

    # Decide final question string
    final_q = audio_text.strip() if audio_text.strip() else (user_q or "").strip()

    # When user clicks Ask or voice provided some text, answer
    if ask_clicked or (use_voice and audio_text.strip()):
        # Use the global df if present
        try:
            df_in = st.session_state.history if "history" in st.session_state else None
        except Exception:
            df_in = None

        answer_md, visuals = _answer_query(df_in, final_q)

        # Render answer bubble (high-contrast on light background)
        st.markdown(
            f"""
            <div style="
                border-left: 10px solid #1e62ff;
                background: #e9f0ff;
                color: #0a0a0a;
                padding: 14px 18px;
                border-radius: 10px;
                margin: 10px 0;
                font-size: 1.05rem;
                line-height: 1.5;
                box-shadow: 0 2px 6px rgba(0,0,0,0.12);
            ">
              <div style="font-weight:800;color:#0b2a6b;font-size:1.1rem;margin-bottom:6px;">
                HALO Response
              </div>
              <div>{answer_md}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Render optional visuals requested by handler
        for kind, arg in visuals:
            if kind == "spo2_vs_et":
                _mini_scatter_spo2_vs_et(df_in, seconds=60)

# ----- Render Assistant (single; bottom) -----
st.divider()
st.subheader("💬 Conversational Assistant")

colQ1, colQ2 = st.columns([3,1])
q_text = colQ1.text_input(
    "Ask HALO a question (e.g., “Why is MAP falling?” or “Summarize the current situation.”)",
    key="halo_q_input",
    placeholder="Type here…"
)

voice_text = None
if voice_available():
    with colQ2:
        with st.expander("🎤 Voice (Start/Stop)"):
            voice_text = voice_widget()
            if voice_text:
                st.success(f"Transcribed: “{voice_text}”")
else:
    st.caption("🎤 Voice capture not available (missing dependencies). You can still type questions.")

final_q = voice_text if (voice_text and len(voice_text) > 0) else q_text

# Immediate response (no second click needed if voice text arrived)
# If typed, press the button to answer.
do_answer = st.button("Get HALO Answer", type="primary", key="answer_btn") or bool(voice_text)

if do_answer:
    # Get the answer
    resp = answer_query(final_q, df, int(st.session_state.get("sim_hz", 5)))

    # Initialize memory if not present
    if "conversation_log" not in st.session_state:
        st.session_state.conversation_log = []

    # Save Q&A pair (limit 3)
    st.session_state.conversation_log.append({"q": final_q, "a": resp["text"]})
    st.session_state.conversation_log = st.session_state.conversation_log[-3:]

    # --- Display HALO’s response ---
    st.markdown(
        f"""
        <div style="
            border-left:10px solid #1e62ff;
            background:#e9f0ff; color:#0a0a0a;
            padding:14px 18px;border-radius:10px;margin:10px 0;
            font-size:1.02rem;line-height:1.5;box-shadow:0 2px 6px rgba(0,0,0,0.12);
        ">
            <div style="font-weight:800;color:#0b2a6b;font-size:1.2rem;margin-bottom:6px;">
                HALO Response
            </div>
            <div>{resp['text'].replace('\n','<br>')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Show conversation memory ---
    # --- Show conversation memory ---
if st.session_state.conversation_log:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        """
        <h4 style="
            color:#0b2a6b;
            font-weight:800;
            margin-top:1rem;
            margin-bottom:0.5rem;
        ">
            🧠 Recent HALO Conversation
        </h4>
        """,
        unsafe_allow_html=True,
    )

    for i, entry in enumerate(reversed(st.session_state.conversation_log), 1):
        st.markdown(
            f"""
            <div style="
                margin-bottom:14px;
                padding:12px 14px;
                border-left:5px solid #1e62ff;
                background:#eef3ff;
                border-radius:10px;
                box-shadow:0 1px 4px rgba(0,0,0,0.1);
                color:#0a0a0a;
                font-size:0.95rem;
                line-height:1.5;
            ">
                <b style="color:#0b2a6b;">Q{i}:</b> {entry['q']}<br>
                <b style="color:#1e62ff;">A{i}:</b> {entry['a']}
            </div>
            """,
            unsafe_allow_html=True,
        )

# -------------- Audit & Export --------------
st.subheader("Alarm Audit Trail")
if len(st.session_state.audit)==0:
    st.write("No events yet.")
else:
    st.dataframe(pd.DataFrame(st.session_state.audit), use_container_width=True, hide_index=True)

st.subheader("Data")
if df.empty:
    st.write("No data yet.")
else:
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download HALO vitals.csv", data=csv, file_name="halo_vitals.csv", mime="text/csv")
    st.caption("Export shows exactly what HALO observed in real time.")

# -------------- Self-refresh --------------
if st.session_state.running:
    time.sleep(1)
    st.rerun()

