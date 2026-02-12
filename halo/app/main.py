from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from halo.config.loader import load_config
from halo.utils.provenance import write_manifest
from halo.utils.seeding import set_deterministic_seed


st.set_page_config(page_title="HALO Research Scaffold", page_icon="🛡️", layout="wide")
st.title("HALO Research-Grade Scaffold")
st.caption("Non-autonomous research tool scaffold. Not for clinical use.")

cfg = load_config("configs/default.json")
seed = set_deterministic_seed(cfg.runtime.seed)
manifest_path = write_manifest(
    "artifacts/run_manifest.json",
    config=cfg.to_dict(),
    seed=seed,
)

st.success("Scaffold initialized deterministically.")
st.write("Seed:", seed)
st.write("Manifest:", str(manifest_path))
st.json(cfg.to_dict())
