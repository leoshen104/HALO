# app.py — Simulate HR + SpO2 and plot live (demo)
# Not for clinical use.

import time, random
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Vitals Stream Demo", layout="wide")
st.title("Vitals Streaming Demo")
st.caption("Simulating HR and SpO₂ updates.")

# keep data in memory across reruns
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["Time","HR","SpO2"])
if "running" not in st.session_state:
    st.session_state.running = False

col1, col2 = st.columns(2)
if col1.button("Start"):
    st.session_state.running = True
if col2.button("Stop"):
    st.session_state.running = False

# one new sample per second when running
if st.session_state.running:
    new_row = {
        "Time": len(st.session_state.history),
        "HR": random.randint(60, 95),
        "SpO2": random.randint(88, 99),
    }
    st.session_state.history.loc[len(st.session_state.history)] = new_row
    time.sleep(1)
   st.rerun()

df = st.session_state.history

left, right = st.columns(2)
if not df.empty:
    left.subheader("Heart Rate (bpm)")
    left.line_chart(df.set_index("Time")["HR"])
    right.subheader("SpO₂ (%)")
    right.line_chart(df.set_index("Time")["SpO2"])
else:
    st.info("Click Start to begin streaming.")

