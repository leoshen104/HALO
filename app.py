# HALO — Hypothesis & Alarm Logic Orchestrator
# v3.3 — Dynamic thresholds from Scenarios & Event Marks (Cloud-ready)
# Not for clinical use.

import io
import json
import math
import time
import random
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -------------------- Page / Branding --------------------
st.set_page_config(page_title="HALO v3.3", page_icon="🛡️", layout="wide")
st.markdown(
    "<div style='display:inline-block;background:#fff4d6;color:#333;padding:4px 8px;"
    "border-left:6px solid #c26b00;border-radius:8px;margin-bottom:8px;font-size:0.9rem;'>"
    "Demo build · Not for clinical use</div>",
    unsafe_allow_html=True,
)
st.title("HALO — Hypothesis & Alarm Logic Orchestrator")
st.caption("Interpretive assistant for anesthesia alarm fusion (assistant, not decider).")

# -------------------- Demo Mode --------------------
DEMO_MODE = True

# -------------------- State Init --------------------
def _setdefault(k, v):
    if k not in st.session_state:
        st.session_state[k] = v

# Core data buffers / run state
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
_setdefault("muted_until", {})      # per alarm label
_setdefault("cooldown_until", {})   # per alarm label (after recovery)
_setdefault("sim_val", {"HR":80,"SpO2":97,"MAP":80,"EtCO2":37,"RR":12})
_setdefault("sim_time", 0)

# Thresholds (base researcher-tunable)
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

# Hysteresis (exit thresholds) & cooldown
_setdefault("hys_spo2", 2)  # exit if SpO2 > low_spo2 + hys_spo2
_setdefault("hys_map", 5)   # exit if MAP > low_map + hys_map
_setdefault("cooldown", 30) # seconds after recovery

# Noise / artifact injection
_setdefault("enable_noise", True)
_setdefault("artifact_pct", 5)  # % chance/sample

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("Mode")
    st.session_state.mode = st.radio("Input source", ["Live", "Replay"], index=0 if st.session_state.mode=="Live" else 1)

    if st.session_state.mode == "Replay":
        up = st.file_uploader("Upload CSV (Time, HR, SpO2, MAP, optional EtCO2, RR)", type=["csv"])
        if up is not None:
            try:
                df_r = pd.read_csv(io.StringIO(up.getvalue().decode("utf-8")), encoding_errors="ignore")
                cols = {c.lower(): c for c in df_r.columns}
                def pick(*names):
                    for n in names:
                        if n in cols: return cols[n]
                    return None
                c_time = pick("time","t","seconds","sec","index")
                c_hr   = pick("hr","heart","bpm")
                c_s2   = pick("spo2","sao2","o2","oxygen")
                c_map  = pick("map","mean arterial pressure","mean")
                c_et   = pick("etco2","etco₂","co2","et co2")
                c_rr   = pick("rr","resp","resp rate","respiration","respiratory rate")
                need = [c_time, c_hr, c_s2, c_map]
                if any(c is None for c in need):
                    st.error("CSV must include Time, HR, SpO2, MAP. Optional: EtCO2, RR.")
                else:
                    df_r = df_r.rename(columns={
                        c_time:"Time", c_hr:"HR", c_s2:"SpO2", c_map:"MAP",
                        **({c_et:"EtCO2"} if c_et else {}), **({c_rr:"RR"} if c_rr else {})
                    })
                    for c in ["Time","HR","SpO2","MAP","EtCO2","RR"]:
                        if c in df_r.columns:
                            df_r[c] = pd.to_numeric(df_r[c], errors="coerce")
                    df_r = df_r.dropna(subset=["Time","HR","SpO2","MAP"]).reset_index(drop=True)
                    if "EtCO2" not in df_r.columns: df_r["EtCO2"] = np.clip(37 + np.random.normal(0,2,len(df_r)), 20, 60)
                    if "RR" not in df_r.columns:    df_r["RR"]    = np.clip(12 + np.random.normal(0,2,len(df_r)), 6, 40)
                    df_r["Time"] = np.arange(len(df_r))
                    st.session_state.replay_df = df_r[["Time","HR","SpO2","MAP","EtCO2","RR"]]
                    st.session_state.replay_idx = 0
                    st.session_state.history = pd.DataFrame(columns=["Time","HR","SpO2","MAP","EtCO2","RR"])
                    st.success(f"Loaded {len(df_r)} rows for Replay.")
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")
        st.session_state.replay_speed = st.slider("Replay speed (rows/tick)", 1, 20, st.session_state.replay_speed)

    show_details = True if not DEMO_MODE else st.checkbox("Show researcher details", value=False)

    if show_details:
        st.header("Thresholds")
        st.session_state.low_spo2 = st.slider("Low SpO₂ (%)", 85, 96, st.session_state.low_spo2)
        st.session_state.tachy_hr = st.slider("Tachycardia HR (bpm)", 90, 160, st.session_state.tachy_hr)
        st.session_state.low_map  = st.slider("Low MAP (mmHg)", 50, 80, st.session_state.low_map)

        st.subheader("Respiratory")
        st.session_state.high_et  = st.slider("High EtCO₂ (mmHg)", 40, 60, st.session_state.high_et)
        st.session_state.low_et   = st.slider("Low EtCO₂ (mmHg)",  20, 40, st.session_state.low_et)
        st.session_state.low_rr   = st.slider("Low RR (bpm)",       4, 20, st.session_state.low_rr)
        st.session_state.high_rr  = st.slider("High RR (bpm)",     18, 40, st.session_state.high_rr)

        st.header("Persistence (sec)")
        st.session_state.win_spo2 = st.slider("SpO₂ persistence", 5, 30, st.session_state.win_spo2)
        st.session_state.win_hr   = st.slider("HR persistence",   5, 30, st.session_state.win_hr)
        st.session_state.win_map  = st.slider("MAP persistence",  5, 30, st.session_state.win_map)
        st.session_state.win_resp = st.slider("Resp persistence", 5, 30, st.session_state.win_resp)

        st.header("Hysteresis & Cooldown")
        st.session_state.hys_spo2 = st.slider("SpO₂ hysteresis (+%)", 1, 6, st.session_state.hys_spo2)
        st.session_state.hys_map  = st.slider("MAP hysteresis (+mmHg)", 2, 12, st.session_state.hys_map)
        st.session_state.cooldown = st.slider("Alarm cooldown (s)", 0, 120, st.session_state.cooldown)

        st.header("Noise / Artifacts (Live)")
        st.session_state.enable_noise = st.checkbox("Inject artifact/noise", value=st.session_state.enable_noise)
        st.session_state.artifact_pct = st.slider("Artifact chance (%)", 0, 20, st.session_state.artifact_pct)

        st.header("Config")
        cfg = {k: st.session_state[k] for k in [
            "low_spo2","tachy_hr","low_map","high_et","low_et","low_rr","high_rr",
            "win_spo2","win_hr","win_map","win_resp","hys_spo2","hys_map","cooldown",
            "enable_noise","artifact_pct","sim_hz"
        ]}
        st.download_button("Download HALO config.json", data=json.dumps(cfg, indent=2),
                           file_name="halo_config.json", mime="application/json")
        upcfg = st.file_uploader("Load config.json", type=["json"], key="cfg_uploader")
        if upcfg is not None:
            try:
                c = json.load(upcfg)
                for k,v in c.items(): st.session_state[k]=v
                st.success("Config applied.")
            except Exception as e:
                st.error(f"Config load failed: {e}")
    else:
        st.header("Preset")
        if st.button("Apply: False-Alarm ↓"):
            preset = dict(low_spo2=94, win_spo2=12, tachy_hr=120, win_hr=10, low_map=75, win_map=12,
                          hys_spo2=3, hys_map=6, cooldown=45, enable_noise=True, artifact_pct=8,
                          high_et=55, low_et=28, low_rr=8, high_rr=28, win_resp=12, sim_hz=5)
            for k,v in preset.items(): st.session_state[k]=v
            st.success("Preset applied.")

    st.divider()
    if st.button("Reset Data"):
        st.session_state.history = pd.DataFrame(columns=["Time","HR","SpO2","MAP","EtCO2","RR"])
        st.session_state.replay_idx = 0
        st.session_state.events = []
        st.session_state.audit = []
        st.session_state.sim_time = 0
        st.success("Buffers cleared.")

# -------------------- Top Controls --------------------
c1, c2, c3, c4 = st.columns([1,1,2,2])
if c1.button("Start"): st.session_state.running = True
if c2.button("Stop"):  st.session_state.running = False
st.session_state.sim_hz = c3.slider("Samples per second", 1, 10, st.session_state.sim_hz)
c4.write(f"Mode: **{st.session_state.mode}** · Running: **{st.session_state.running}** · Samples: **{len(st.session_state.history)}**")

# -------------------- Scenarios & Events --------------------
sc1, sc2, sc3, sc4, sc5 = st.columns(5)
if sc1.button("Scenario: Bleed (60s)"):        st.session_state.scenario_name="Bleed"; st.session_state.scenario_end=time.time()+60
if sc2.button("Scenario: Bronchospasm (60s)"): st.session_state.scenario_name="Bronchospasm"; st.session_state.scenario_end=time.time()+60
if sc3.button("Scenario: Vasodilation (60s)"): st.session_state.scenario_name="Vasodilation"; st.session_state.scenario_end=time.time()+60
if sc4.button("Scenario: Pain/Light (60s)"):   st.session_state.scenario_name="Pain/Light"; st.session_state.scenario_end=time.time()+60
if sc5.button("End Scenario"):                 st.session_state.scenario_name=None; st.session_state.scenario_end=0.0

e1, e2, e3, e4, e5 = st.columns(5)
if e1.button("Mark: Incision"):        st.session_state.events.append({"t": len(st.session_state.history), "name":"Incision"})
if e2.button("Mark: Position change"): st.session_state.events.append({"t": len(st.session_state.history), "name":"Position change"})
if e3.button("Mark: Fluids 250 mL"):   st.session_state.events.append({"t": len(st.session_state.history), "name":"Fluids 250 mL"})
if e4.button("Mark: Vasopressor"):     st.session_state.events.append({"t": len(st.session_state.history), "name":"Vasopressor"})
if e5.button("Clear Events"):          st.session_state.events = []

# -------------------- Dynamic Threshold Engine --------------------
def _time_now_index() -> int:
    if st.session_state.history.empty: return 0
    return int(st.session_state.history["Time"].iloc[-1])

def _age_since_event(name: str, horizon: int) -> float:
    """Return age in seconds (samples) since most recent named event within horizon; None if none."""
    t_now = _time_now_index()
    recent = [e for e in st.session_state.events if e["name"] == name and 0 <= (t_now - e["t"]) <= horizon]
    if not recent: return None
    age = min(t_now - e["t"] for e in recent)
    return float(age)

def _decay_linear(age: float, horizon: int) -> float:
    if age is None: return 0.0
    return max(0.0, (horizon - age)/float(horizon))

def _scenario_mods():
    """Scenario-based threshold deltas (more sensitive during relevant scenarios)."""
    name = st.session_state.scenario_name
    active = (name is not None) and (time.time() <= st.session_state.scenario_end)
    if not active: return dict(spo2=0, hr=0, map=0, et_high=0, rr_low=0)
    if name == "Bleed":
        return dict(spo2=0,  hr=-10, map=+5,  et_high=0,  rr_low=0)   # HR more sensitive; MAP more sensitive
    if name == "Bronchospasm":
        return dict(spo2=+2, hr=0,   map=0,   et_high=-4, rr_low=+2) # SpO2 & RR more sensitive; EtCO2 more sensitive
    if name == "Vasodilation":
        return dict(spo2=0,  hr=0,   map=+5,  et_high=0,  rr_low=0)   # MAP more sensitive
    if name == "Pain/Light":
        return dict(spo2=0,  hr=-10, map=0,   et_high=0,  rr_low=0)   # HR more sensitive
    return dict(spo2=0, hr=0, map=0, et_high=0, rr_low=0)

def _event_mods():
    """Event-based threshold deltas with time-decay."""
    t_now = _time_now_index()
    # Horizons (seconds ~ samples)
    H_INCISION = 90
    H_FLUIDS   = 120
    H_PRESSOR  = 180
    H_POSN     = 120

    age_incision = _age_since_event("Incision", H_INCISION)
    age_fluids   = _age_since_event("Fluids 250 mL", H_FLUIDS)
    age_pressor  = _age_since_event("Vasopressor", H_PRESSOR)
    age_posn     = _age_since_event("Position change", H_POSN)

    w_inc = _decay_linear(age_incision, H_INCISION)  # high right after incision
    w_flu = _decay_linear(age_fluids,   H_FLUIDS)
    w_pre = _decay_linear(age_pressor,  H_PRESSOR)
    w_pos = _decay_linear(age_posn,     H_POSN)

    # Sign convention:
    #  - For SpO2 low threshold: + makes more sensitive (alarm sooner), - relaxes (alarm later)
    #  - For HR tachy threshold: - makes more sensitive (lower bar), + relaxes (higher bar)
    #  - For MAP low threshold:  + makes more sensitive, - relaxes
    #  - For EtCO2 high:         - makes more sensitive
    #  - For RR low:             + makes more sensitive
    spo2 = (-2.0 * w_pos)           # position change → transient SpO2 dips likely; relax a bit
    hr   = (+10.0 * w_inc)          # incision → expected sympathetic HR rise; relax tachy to reduce false alarms
    map_ = (-5.0 * w_flu) + (+5.0 * w_pre)  # fluids → relax MAP; pressor given → be stricter on MAP
    et_h = (0.0)                    # leave EtCO2 event-neutral here
    rr_l = (0.0)
    return dict(spo2=spo2, hr=hr, map=map_, et_high=et_h, rr_low=rr_l)

def effective_thresholds():
    """Combine base thresholds + scenario deltas + event deltas."""
    base = dict(
        low_spo2 = st.session_state.low_spo2,
        tachy_hr = st.session_state.tachy_hr,
        low_map  = st.session_state.low_map,
        high_et  = st.session_state.high_et,
        low_rr   = st.session_state.low_rr
    )
    scen = _scenario_mods()
    ev   = _event_mods()

    eff = dict(
        low_spo2 = int(round(np.clip(base["low_spo2"] + scen["spo2"] + ev["spo2"], 80, 99))),
        tachy_hr = int(round(np.clip(base["tachy_hr"] + scen["hr"]   + ev["hr"],   80, 180))),
        low_map  = int(round(np.clip(base["low_map"]  + scen["map"]  + ev["map"],  45, 90))),
        high_et  = int(round(np.clip(base["high_et"]  + scen["et_high"] + ev["et_high"], 35, 60))),
        low_rr   = int(round(np.clip(base["low_rr"]   + scen["rr_low"]  + ev["rr_low"],   4, 20))),
    )
    # Exit thresholds with hysteresis based on *effective* entries
    eff["exit_spo2"] = eff["low_spo2"] + st.session_state.hys_spo2
    eff["exit_map"]  = eff["low_map"]  + st.session_state.hys_map
    return eff

# -------------------- Simulation / Replay --------------------
def _step(val, target, sigma, lo, hi):
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
        hr   = _step(base["HR"],   targets["HR"],   1.2, 30, 180)
        spo2 = _step(base["SpO2"], targets["SpO2"], 0.6, 70, 100)
        map_ = _step(base["MAP"],  targets["MAP"],  1.5, 40, 120)
        et   = _step(base["EtCO2"],targets["EtCO2"],0.8, 20, 60)
        rr   = _step(base["RR"],   targets["RR"],   0.8,  6, 40)

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

# Tick
if st.session_state.running:
    if st.session_state.mode == "Live": _tick_live(st.session_state.sim_hz)
    else: _tick_replay()

df = st.session_state.history

# -------------------- Charts --------------------
c1, c2, c3, c4, c5 = st.columns(5)
if not df.empty:
    c1.subheader("HR (bpm)");     c1.line_chart(df.set_index("Time")["HR"])
    c2.subheader("SpO₂ (%)");     c2.line_chart(df.set_index("Time")["SpO2"])
    c3.subheader("MAP (mmHg)");   c3.line_chart(df.set_index("Time")["MAP"])
    c4.subheader("EtCO₂ (mmHg)"); c4.line_chart(df.set_index("Time")["EtCO2"])
    c5.subheader("RR (bpm)");     c5.line_chart(df.set_index("Time")["RR"])
else:
    st.info("Click Start (Live) or upload a CSV (Replay).")

# -------------------- Helpers --------------------
def last_n(df: pd.DataFrame, seconds: int) -> pd.DataFrame:
    if df.empty: return df
    end = df["Time"].iloc[-1]; start = max(0, end - seconds + 1)
    return df[df["Time"].between(start, end)]

def persistent_low(df, signal, thresh, sec):
    w = last_n(df, sec)
    return (len(w) >= sec) and (w[signal].min() < thresh)

def persistent_high(df, signal, thresh, sec):
    w = last_n(df, sec)
    return (len(w) >= sec) and (w[signal].max() > thresh)

def is_artifact_series(series: pd.Series) -> bool:
    if len(series) < 6: return False
    recent = series.iloc[-6:]
    diffs = np.abs(np.diff(recent.values))
    if len(diffs)==0: return False
    return diffs[-1] > (np.std(recent.values) * 4 + 4)

def recent_artifact(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"HR":False,"SpO2":False,"MAP":False,"EtCO2":False,"RR":False}
    return {"HR":is_artifact_series(df["HR"]), "SpO2":is_artifact_series(df["SpO2"]),
            "MAP":is_artifact_series(df["MAP"]), "EtCO2":is_artifact_series(df["EtCO2"]),
            "RR":is_artifact_series(df["RR"])}

def in_cooldown(label: str) -> bool:
    return time.time() < st.session_state.cooldown_until.get(label, 0)
def is_muted(label: str) -> bool:
    return time.time() < st.session_state.muted_until.get(label, 0)
def countdown(label: str) -> int:
    return max(0, int(round(st.session_state.cooldown_until.get(label,0) - time.time())))

# -------------------- Alarm Fusion (uses EFFECTIVE thresholds) --------------------
alerts = []
hypox = tachy = hypot = hypercap = hypovent = False  # safe defaults

eff = effective_thresholds()  # <<< dynamic thresholds (scenario + events)

if not df.empty:
    arts = recent_artifact(df)

    # Persistent patterns vs effective thresholds
    hypox    = (not arts["SpO2"]) and persistent_low(df, "SpO2", eff["low_spo2"], st.session_state.win_spo2)
    tachy    = (not arts["HR"])   and persistent_high(df, "HR",   eff["tachy_hr"], st.session_state.win_hr)
    hypot    = (not arts["MAP"])  and persistent_low(df, "MAP",   eff["low_map"],  st.session_state.win_map)
    hypercap = (not arts["EtCO2"])and persistent_high(df, "EtCO2",eff["high_et"],  st.session_state.win_resp)
    hypovent = (not arts["RR"])   and persistent_low(df, "RR",    eff["low_rr"],   st.session_state.win_resp)

    # Compose alerts (corroboration elevates severity)
    if hypox and not is_muted("Low SpO₂") and not in_cooldown("Low SpO₂"):
        sev = "Warning"; why = f"SpO₂ < {eff['low_spo2']}% ≥{st.session_state.win_spo2}s (exit > {eff['exit_spo2']}%)"
        if tachy: sev = "Critical"; why += f" + HR > {eff['tachy_hr']} ≥{st.session_state.win_hr}s"
        alerts.append({"label":"Low SpO₂","severity":sev,"why":why})

    if hypot and not is_muted("Low MAP") and not in_cooldown("Low MAP"):
        sev = "Warning"; why = f"MAP < {eff['low_map']} ≥{st.session_state.win_map}s (exit > {eff['exit_map']})"
        if tachy: sev = "Critical"; why += f" + HR > {eff['tachy_hr']} ≥{st.session_state.win_hr}s"
        alerts.append({"label":"Low MAP","severity":sev,"why":why})

    if tachy and not is_muted("Tachycardia"):
        alerts.append({"label":"Tachycardia","severity":"Advisory","why":f"HR > {eff['tachy_hr']} ≥{st.session_state.win_hr}s"})

    if (hypercap or hypovent) and not is_muted("Respiratory Concern"):
        msg = []
        if hypercap: msg.append(f"EtCO₂ high (> {eff['high_et']})")
        if hypovent: msg.append(f"RR low (< {eff['low_rr']})")
        if hypox:    msg.append(f"SpO₂ low (< {eff['low_spo2']})")
        sev = "Advisory"
        if hypercap and hypovent and hypox: sev = "Warning"
        alerts.append({"label":"Respiratory Concern","severity":sev,"why":", ".join(msg)})

# -------------------- Hypothesis Engine (unchanged logic) --------------------
def _delta(series: pd.Series, window: int) -> float:
    if len(series) < 2: return 0.0
    w = series.iloc[-window:] if len(series) >= window else series
    return float(w.iloc[-1] - w.iloc[0])

def _z(x, lo, hi):
    if x <= lo: return 0.0
    if x >= hi: return 1.0
    return (x - lo) / float(hi - lo)

def _bounded(x): return max(0.0, min(1.0, float(x)))

def _event_weight(name: str, t_now: int, horizon=180, decay=0.4):
    rel = [e for e in st.session_state.events if e["name"] == name and 0 <= (t_now - e["t"]) <= horizon]
    if not rel: return 0.0
    ages = [t_now - e["t"] for e in rel]
    w = max(0.0, 1.0 - min(ages)/horizon)
    return w**(1-decay)

def hypothesis_components(df: pd.DataFrame):
    if df.empty: return [], {}, {}, {}
    hr = df["HR"]; s2 = df["SpO2"]; mp = df["MAP"]; et = df["EtCO2"]; rr = df["RR"]
    comps = {
        "MAP_down": _bounded(_z(80 - mp.iloc[-1], 0, 25) + _z(-_delta(mp,10), 3, 15)),
        "HR_up":    _bounded(_z(hr.iloc[-1], 90, 130) + _z(_delta(hr,10), 3, 15)),
        "SpO2_down":_bounded(_z(95 - s2.iloc[-1], 0, 10) + _z(-_delta(s2,10), 1, 6)),
        "Et_high":  _bounded(_z(et.iloc[-1], 40, 55)),
        "Et_low":   _bounded(_z(35 - et.iloc[-1], 0, 10)),
        "RR_low":   _bounded(_z(12 - rr.iloc[-1], 0, 6)),
        "RR_high":  _bounded(_z(rr.iloc[-1], 20, 35)),
    }
    arts = recent_artifact(df) if len(df) >= 6 else {"HR":False,"SpO2":False,"MAP":False,"EtCO2":False,"RR":False}
    w = {"MAP": 0.6 if arts["MAP"] else 1.0, "HR": 0.6 if arts["HR"] else 1.0,
         "SpO2":0.6 if arts["SpO2"] else 1.0, "EtCO2":0.6 if arts["EtCO2"] else 1.0, "RR":0.6 if arts["RR"] else 1.0}
    t_now = int(df["Time"].iloc[-1])
    pri = {"incision": _event_weight("Incision", t_now, 180, 0.4),
           "fluids":   _event_weight("Fluids 250 mL", t_now, 120, 0.4),
           "pressor":  _event_weight("Vasopressor", t_now, 180, 0.4),
           "position": _event_weight("Position change", t_now, 120, 0.4)}
    hypovolemia = _bounded(0.55*w["MAP"]*comps["MAP_down"] + 0.35*w["HR"]*comps["HR_up"] + 0.10*(1-comps["SpO2_down"]) + 0.25*pri["incision"] - 0.20*pri["fluids"])
    vasodilation= _bounded(0.60*w["MAP"]*comps["MAP_down"] + 0.25*(1-w["HR"]*comps["HR_up"]) + 0.15*(1 - comps["SpO2_down"]) - 0.25*pri["pressor"])
    respiratory = _bounded(0.45*w["SpO2"]*comps["SpO2_down"] + 0.25*w["EtCO2"]*comps["Et_high"] + 0.25*w["RR"]*max(comps["RR_low"], comps["RR_high"]) + 0.15*pri["position"])
    pain_light  = _bounded(0.65*w["HR"]*comps["HR_up"] + 0.20*(1 - w["SpO2"]*comps["SpO2_down"]) + 0.15*(max(0.0, _delta(hr,10))/15 if len(df)>=10 else 0.0))
    hyps = [
        {"name":"Hypovolemia",       "score":hypovolemia, "evidence":["MAP↓","HR↑","SpO₂↔","Incision","–Fluids"]},
        {"name":"Vasodilation",      "score":vasodilation,"evidence":["MAP↓","HR↔/↓","SpO₂↔","–Pressor"]},
        {"name":"Respiratory issue", "score":respiratory, "evidence":["SpO₂↓","EtCO₂↑/RR abnormal","Position"]},
        {"name":"Pain / light",      "score":pain_light,  "evidence":["HR↑","SpO₂↔","ΔHR(10s)"]},
    ]
    return hyps, comps, w, pri

def hypothesis_panel(df, show_bars=False):
    st.subheader("Hypothesis (assistant)")
    if df.empty:
        st.write("Insufficient data yet."); return
    hyps, comps, w, pri = hypothesis_components(df)
    if not hyps:
        st.write("Insufficient data yet."); return
    total = sum(h["score"] for h in hyps) or 1e-9
    for h in hyps: h["pct"] = int(round(100 * h["score"] / total))
    top = sorted(hyps, key=lambda x: x["pct"], reverse=True)
    st.markdown(f"**Likely:** {top[0]['name']} — **{top[0]['pct']}%**")
    others = ", ".join([f"{h['name']} {h['pct']}%" for h in top[1:]])
    if others: st.write("Other possibilities:", others)

    if show_bars:
        def bar(label, value):
            width = int(100 * max(0.0, min(1.0, float(value))))
            return f"""
            <div style="margin:2px 0;">
              <span style="display:inline-block;width:120px">{label}</span>
              <span style="display:inline-block;background:#e9f0ff;width:220px;border-radius:6px;vertical-align:middle;">
                <span style="display:inline-block;background:#1e62ff;height:10px;width:{width*2.2}px;border-radius:6px;"></span>
              </span>
              <span style="margin-left:6px;font-size:0.9rem">{width}%</span>
            </div>
            """
        with st.expander("Explain contributions (feature bars)"):
            cL, cR = st.columns(2)
            cL.markdown("**Core signals**", unsafe_allow_html=True)
            cL.markdown(bar("MAP↓", comps["MAP_down"]), unsafe_allow_html=True)
            cL.markdown(bar("HR↑",  comps["HR_up"]),    unsafe_allow_html=True)
            cL.markdown(bar("SpO₂↓",comps["SpO2_down"]),unsafe_allow_html=True)
            cR.markdown("**Respiratory**", unsafe_allow_html=True)
            cR.markdown(bar("EtCO₂↑",comps["Et_high"]), unsafe_allow_html=True)
            cR.markdown(bar("EtCO₂↓",comps["Et_low"]),  unsafe_allow_html=True)
            cR.markdown(bar("RR low", comps["RR_low"]),  unsafe_allow_html=True)
            cR.markdown(bar("RR high",comps["RR_high"]), unsafe_allow_html=True)

hypothesis_panel(df, show_bars=show_details)

# -------------------- Show Effective Thresholds (for transparency) --------------------
if show_details:
    st.subheader("Effective thresholds (with scenario & event modifiers)")
    eff_view = pd.DataFrame([{
        "SpO₂ low trigger": eff["low_spo2"],
        "SpO₂ exit": eff["exit_spo2"],
        "Tachy HR trigger": eff["tachy_hr"],
        "MAP low trigger": eff["low_map"],
        "MAP exit": eff["exit_map"],
        "EtCO₂ high trigger": eff["high_et"],
        "RR low trigger": eff["low_rr"],
        "Scenario": st.session_state.scenario_name or "None",
        "Recent events": ", ".join(f"{e['name']}@{e['t']}" for e in st.session_state.events[-4:]) or "—",
    }])
    st.dataframe(eff_view, use_container_width=True, hide_index=True)

# -------------------- Data Quality Panel --------------------
st.subheader("Data Quality (last 60s)")

def artifact_rate(series: pd.Series, window=60):
    if len(series) < 8: return 0.0
    w = series.iloc[-min(window, len(series)):]
    jumps = np.abs(np.diff(w.values)); thr = np.std(w.values) * 4 + 4
    return float(np.mean(jumps > thr))

if len(df) >= 10:
    q1, q2, q3, q4, q5 = st.columns(5)
    rates = {"HR":artifact_rate(df["HR"]), "SpO₂":artifact_rate(df["SpO2"]),
             "MAP":artifact_rate(df["MAP"]), "EtCO₂":artifact_rate(df["EtCO2"]), "RR":artifact_rate(df["RR"])}
    def badge(name, r):
        color = "#0a8a0a" if r < 0.05 else ("#c26b00" if r < 0.15 else "#b00020")
        return (
            "<div style='background:#f5f5f5;color:#111111;padding:8px;"
            f"border-left:8px solid {color};border-radius:8px'>"
            f"{name}: <b style='color:{color}'>{int(r*100)}%</b> artifact-like jumps</div>"
        )
    q1.markdown(badge("HR", rates["HR"]),   unsafe_allow_html=True)
    q2.markdown(badge("SpO₂", rates["SpO₂"]), unsafe_allow_html=True)
    q3.markdown(badge("MAP", rates["MAP"]), unsafe_allow_html=True)
    q4.markdown(badge("EtCO₂", rates["EtCO₂"]), unsafe_allow_html=True)
    q5.markdown(badge("RR", rates["RR"]),   unsafe_allow_html=True)
else:
    st.write("Collecting data...")

# -------------------- Alarm Panel with Acknowledge / Mute & Sparklines --------------------
st.subheader("Alarm Panel")

def sparkline_with_window(series, window, thresh=None, invert=False):
    if len(series) < 3: return
    fig, ax = plt.subplots(figsize=(3.5, 1.3))
    ax.plot(series.index, series.values)
    xmax = series.index.max()
    xmin = max(series.index.min(), xmax - window + 1)
    ax.axvspan(xmin, xmax, alpha=0.15, color="#1e62ff")
    if thresh is not None:
        ax.axhline(thresh, linestyle="--", alpha=0.6, color="#b00020" if not invert else "#0a0a0a")
    ax.set_yticklabels([]); ax.set_xticklabels([])
    ax.set_yticks([]); ax.set_xticks([])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    st.pyplot(fig, clear_figure=True)

def audit_log(action, meta):
    entry = {"t": int(df["Time"].iloc[-1]) if not df.empty else 0, "action": action}
    entry.update(meta or {})
    st.session_state.audit.append(entry)

def countdown(label: str) -> int:
    return max(0, int(round(st.session_state.cooldown_until.get(label,0) - time.time())))

def banner(a, idx):
    label, severity, why = a["label"], a["severity"], a["why"]
    palettes = {"Advisory":{"border":"#1e62ff","bg":"#e9f0ff","title":"#0b2a6b","text":"#0a0a0a"},
                "Warning":{"border":"#c26b00","bg":"#fff0d6","title":"#5a3200","text":"#0a0a0a"},
                "Critical":{"border":"#b00020","bg":"#ffe3e6","title":"#5b0a12","text":"#0a0a0a"}}
    p = palettes.get(severity, {"border":"#444","bg":"#eee","title":"#111","text":"#0a0a0a"})
    muted_remain = max(0, int(round(st.session_state.muted_until.get(label,0) - time.time())))
    chip = ""
    if muted_remain>0: chip = f"<span style='background:#333;color:#fff;padding:2px 6px;border-radius:10px;margin-left:6px;font-size:0.8rem;'>muted {muted_remain}s</span>"

    st.markdown(f"""
    <div style="border-left:10px solid {p['border']};background:{p['bg']};color:{p['text']};
                padding:14px 18px;border-radius:10px;margin:10px 0;font-size:1.05rem;line-height:1.5;
                box-shadow:0 2px 6px rgba(0,0,0,0.12);">
      <div style="font-weight:800;color:{p['title']};font-size:1.25rem;margin-bottom:6px;">
        {severity}: {label} {chip}
      </div>
      <div style="margin-bottom:6px;"><b>Why:</b> {why}</div>
    </div>""", unsafe_allow_html=True)

    # Sparklines for last 12s
    w = last_n(df, 12).set_index("Time")
    cols = st.columns(3)
    if label == "Low SpO₂" and not w.empty:
        cols[1].markdown("**SpO₂ last 12s**")
        sparkline_with_window(w["SpO2"], 12, thresh=eff["low_spo2"], invert=False)
        st.caption("Quick checks: sensor position/perfusion; airway/ventilation; FiO₂; hemodynamics.")
    elif label == "Low MAP" and not w.empty:
        cols[2].markdown("**MAP last 12s**")
        sparkline_with_window(w["MAP"], 12, thresh=eff["low_map"], invert=False)
        st.caption("Quick checks: bleeding/EBL, anesthetic depth/vasodilation, measurement accuracy.")
    elif label == "Tachycardia" and not w.empty:
        cols[0].markdown("**HR last 12s**")
        sparkline_with_window(w["HR"], 12, thresh=eff["tachy_hr"], invert=True)
        st.caption("Quick checks: stimulus/pain, volume status, artifact (ECG leads).")
    elif label == "Respiratory Concern" and not w.empty:
        cols[0].markdown("**EtCO₂ last 12s**"); sparkline_with_window(w["EtCO2"], 12, thresh=eff["high_et"], invert=False)
        cols[1].markdown("**RR last 12s**");    sparkline_with_window(w["RR"], 12, thresh=eff["low_rr"],  invert=True)
        st.caption("Quick checks: ventilation adequacy, circuit patency, auscultation if possible.")

    a1, a2, _ = st.columns([1,1,6])
    if a1.button("Acknowledge 60s", key=f"ack_{label}_{idx}"):
        st.session_state.muted_until[label] = time.time() + 60
        audit_log("acknowledge", {"label":label,"duration_s":60})
    if a2.button("Mute 5m", key=f"mute_{label}_{idx}"):
        st.session_state.muted_until[label] = time.time() + 300
        audit_log("mute", {"label":label,"duration_s":300})

if alerts:
    for a in alerts:
        audit_log("alert_raise", {"label":a["label"], "severity":a["severity"], "why":a["why"]})
    for i, a in enumerate(alerts):
        banner(a, i)
else:
    st.markdown("""
        <div style="border-left:10px solid #0a8a0a;background:#e8f7e8;color:#0a0a0a;
                    padding:14px 18px;border-radius:10px;font-size:1.05rem;line-height:1.5;
                    box-shadow:0 2px 6px rgba(0,0,0,0.12);">
            <div style="font-weight:800;color:#0a5a0a;font-size:1.2rem;margin-bottom:4px;">No active alarms</div>
            <div style="font-weight:500;">System nominal.</div>
        </div>""", unsafe_allow_html=True)

# -------------------- Audit & Export --------------------
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

# -------------------- Self-refresh --------------------
if st.session_state.running:
    time.sleep(1)
    st.rerun()
