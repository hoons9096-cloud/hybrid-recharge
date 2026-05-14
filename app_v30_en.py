"""
app_v30_en.py — English entry point for hybrid-recharge v30.

Exposes the same backend as app_v30.py but with English-localised UI for
the two paper-aligned tabs (Tab 1 single-well basic analysis and Tab 10
watershed-scale bias-aware framework). Other tabs (pumping, EnKF, etc.)
are reused from the original Korean ui module — they are auxiliary for
the present manuscript.

Run:
    streamlit run app_v30_en.py
"""
from __future__ import annotations

import os
import sys
import io
import tempfile
import hashlib
import warnings

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"streamlit")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"plotly")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
_PUMP_DIR = os.path.join(_THIS_DIR, "pump_preprocess")
if _PUMP_DIR not in sys.path:
    sys.path.insert(0, _PUMP_DIR)

from core_sim_v27 import core_sim_v27 as _run_sim
from cross_validation import split_sample_test, temporal_kfold_cv
from data_loader import load_timeseries_file
from uncertainty import bootstrap_uncertainty
from soil_db import (
    SOIL_NAMES_NUMBERED as SOIL_NAMES,
    CLAY_LIKE_SET, get_soil,
)
from scoring import score_dataframe, score_k_stress, score_sy_match
from bma import compute_bma, bma_summary_table
from sensitivity import (
    kalman_sensitivity_sweep, topsis_weight_sensitivity,
    objective_weight_sensitivity,
)
from core_sim_v27 import (
    get_kalman_uncertainty, propagate_kalman_recharge_uncertainty,
)

# UI modules
from ui import TabContext, has_v27_error
from ui import tab_pump, tab_compare, tab_crossval, tab_uncertainty
from ui import tab_sensitivity, tab_spatial, tab_field_report

# English variants
from ui_en import tab_base_en, tab_watershed_report_en, tab_pump_en


# ═════════════════════════════════════════════════════════
# Page config
# ═════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Hybrid Recharge AI Lab v30 — English",
    layout="wide",
    page_icon="💧",
)

# CSS (carry over from Korean version)
st.markdown("""
<style>
.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"]{
    background-color:#FAFBFD!important;color:#1B2A4A!important}
section[data-testid="stSidebar"],section[data-testid="stSidebar"]>div{
    background-color:#F0F2F6!important;color:#1B2A4A!important}
.stApp p,.stApp span,.stApp div,.stApp label,.stApp li,.stApp td,.stApp th,
[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] li,[data-testid="stMarkdownContainer"] code{
    color:#1B2A4A!important}
h1,h2,h3,h4{color:#1B2A4A!important;font-weight:800!important}
div[data-testid="stMetric"]{background:linear-gradient(135deg,#FFF 0%,#F8F9FC 100%)!important;
    border:1px solid #E2E6ED!important;border-radius:12px;padding:16px 20px;
    box-shadow:0 2px 8px rgba(0,0,0,.04)}
div[data-testid="stMetric"] label{color:#4A5A75!important;font-size:.8rem!important;
    font-weight:600!important;letter-spacing:.5px;text-transform:uppercase}
div[data-testid="stMetric"] div[data-testid="stMetricValue"]{
    color:#0F1A2E!important;font-size:1.6rem!important;font-weight:700!important}
.stButton>button{border-radius:8px;font-weight:600;
    background:#FFF!important;color:#1B2A4A!important;border:1.5px solid #CBD5E1!important}
.stButton>button[kind="primary"]{background:#DC2626!important;color:#FFF!important;border:none!important}
div[data-testid="stTabs"]>div:first-child{overflow-x:auto!important;scrollbar-width:thin}
div[data-testid="stTabs"] button[role="tab"]{flex:0 0 auto!important;white-space:nowrap!important}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════
# Session state
# ═════════════════════════════════════════════════════════
_DEFAULTS = {
    "uploaded_tmp_path": None, "uploaded_file_sig": None,
    "uploaded_name": None, "result_v27": None,
    "scan_data": None, "best_soil": None,
    "best_soil_conf": "MEDIUM", "best_soil_tentative": False,
    "monte_carlo": None, "pump_result": None,
    "pump_detection": None, "enkf_result": None,
    "enkf_object": None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ═════════════════════════════════════════════════════════
# Title
# ═════════════════════════════════════════════════════════
st.title("💧 Hybrid Recharge AI Lab v30 — Bias-Aware WTF Framework")
st.caption("Pump pre-processing + Augmented Kalman filter + "
           "Soil-weighted upscaling + Hierarchical Bayesian + "
           "Learned bias correction")

# ═════════════════════════════════════════════════════════
# Workflow guide
# ═════════════════════════════════════════════════════════
with st.expander("📖 How to use — workflow by scenario", expanded=False):
    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown("""
        ##### 🅰️ Single-well analysis
        **Goal**: estimate recharge for one well

        1. **Sidebar** → upload data file (.txt/.csv: date, level, rain)
        2. **Tab 1 (Basic)** → "Run analysis"
        3. *(opt)* **Tab 2** pumping correction
        4. **Tab 3 (Soil compare)**
        5. **Tab 4–6** validation
        6. **Tab 8 (Well report)** → HTML output
        """)
    with g2:
        st.markdown("""
        ##### 🅱️ Watershed recharge (multi-well)
        **Goal**: bias-aware area-weighted recharge

        1. Add well coordinates in `wells_registry`
        2. Prepare digital soil map `.shp`
        3. **Tab 10** → select watershed → run
        4. **Compare** Lumped / Soil-weighted / Bias-corrected (α)
        5. Multi-proxy envelope check
        6. Download CSV

        ⚠️ Sidebar upload *not required* — `.txt` files matched automatically
        """)
    with g3:
        st.markdown("""
        ##### 🅲 Synthetic benchmark (paper)
        **Goal**: controlled experiment with known truth

        1. **Tab 7 (EnKF synthetic)** S1–S5
        2. **Tab 9 (Field report)** integrated

        Lumped / Soil-weighted / Bias-corrected / Hierarchical / EnKF —
        five-method comparison with 5–49% bias quantification
        """)
    st.markdown("---")
    st.caption(
        "**Tab grouping**: Tabs 1–6 = single-well analysis · "
        "Tabs 7, 9 = synthetic benchmark · "
        "Tabs 8, 10 = integrated reports (8 = well, 10 = watershed)"
    )


# ═════════════════════════════════════════════════════════
# Sidebar — quick guide + data upload
# ═════════════════════════════════════════════════════════
with st.sidebar.expander("ℹ️ Quick guide", expanded=False):
    st.markdown("""
**Starting points by scenario**

- **Single well** → upload file → Tab 1
- **Watershed (multi)** → Tab 10 (no upload needed)
- **Synthetic benchmark** → Tab 7

**Required input format**
- 3 columns: `date \\t level(m) \\t rain(mm)`
- Daily, no missing values preferred

**Tab 10 prerequisites**
- Wells registered in `wells_registry.json`
- `.shp` soil map at default path
- Per-well `.txt` files in project root
    """)

st.sidebar.header("📁 Data & AI")
api_key = st.sidebar.text_input("OpenAI API key (for AI commentary)",
                                 type="password")
uploaded_file = st.sidebar.file_uploader(
    "Data file (CSV: date, level, rain)", type=["csv", "txt", "dat"],
)

file_path_to_send = "DEMO"
if uploaded_file is not None:
    try:
        file_bytes = uploaded_file.getvalue()
        file_sig = (uploaded_file.name, len(file_bytes),
                    hashlib.md5(file_bytes).hexdigest())
        if st.session_state["uploaded_file_sig"] != file_sig:
            old = st.session_state.get("uploaded_tmp_path")
            if old and os.path.exists(old):
                try: os.remove(old)
                except OSError: pass
            suffix = os.path.splitext(uploaded_file.name)[1] or ".dat"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_bytes)
                st.session_state["uploaded_tmp_path"] = tmp.name
            st.session_state["uploaded_file_sig"] = file_sig
            st.session_state["uploaded_name"] = uploaded_file.name
            st.session_state["pump_result"] = None
            st.session_state["pump_detection"] = None
            st.session_state["result_v27"] = None
        file_path_to_send = st.session_state["uploaded_tmp_path"]
        st.sidebar.success(f"Loaded: {st.session_state['uploaded_name']}")
    except Exception as _e:
        st.sidebar.error(f"File error: {_e}")

st.sidebar.markdown("---")

# ═════════════════════════════════════════════════════════
# Sidebar — basic parameters (English labels)
# ═════════════════════════════════════════════════════════
with st.sidebar.expander("🔧 Kalman parameters", expanded=True):
    k_val = st.slider("Decay constant (k)", -0.50, -0.0001, -0.015,
                       step=0.0001, format="%.4f")
    z_val = st.slider("Damping (z)", 0.1, 10.0, 3.0, step=0.1)
    lag_val = st.slider("Lag (days)", 0, 30, 0, step=1)
    sn_idx = st.slider("Soil index (1=Sand, 12=Loam)",
                        1, 12, 12, step=1)
    q_val = st.slider("Process noise Q", 1e-6, 1.0, 0.005,
                       step=1e-4, format="%.4f")
    r_val = st.slider("Observation noise R", 1e-6, 1.0, 0.10,
                       step=1e-4, format="%.4f")
    rc_val = st.slider("Recharge coefficient", 1e-5, 0.05, 0.005,
                        step=1e-4, format="%.4f")
    sens_val = st.slider("Sensitivity scale", 0.1, 5.0, 1.0, step=0.1)
    ignore_pump = st.slider("Ignore pumping", 0.0, 1.0, 0.0, step=0.1)
    auto_optimize = st.checkbox("Auto-optimize k, z, lag", value=True)
    show_pure = st.checkbox("Show Pure WTF (no Kalman)", value=True)

st.sidebar.markdown("---")

# Pump pre-processing parameters
with st.sidebar.expander("🔧 Pumping pre-processing", expanded=False):
    detect_methods = st.multiselect(
        "Detection methods",
        ["sigma", "rolling_baseline", "rolling_baseline_robust", "isolation_forest"],
        default=["sigma", "rolling_baseline"],
    )
    sigma_drop = st.slider("σ-drop threshold", 1.0, 5.0, 2.5, step=0.1)
    buffer_days = st.slider("Event buffer (days)", 1, 7, 2, step=1)
    correction_strategy = st.selectbox("Correction strategy",
                                        ["auto", "interpolation", "exponential"])

st.sidebar.markdown("---")

# Run button
if st.sidebar.button("🚀 Run basic analysis (Step 1)", type="primary"):
    with st.spinner("Running v27 simulation…"):
        result = _run_sim(
            file_path_to_send, k_val, z_val, lag_val, sn_idx,
            q_val, r_val, rc_val, ignore_pump, sens_val, auto_optimize,
        )
        st.session_state["result_v27"] = result
        if has_v27_error(result):
            st.sidebar.error(f"Failed: {result.get('error', '?')}")
        else:
            st.sidebar.success("✅ Basic analysis complete")


# ═════════════════════════════════════════════════════════
# Build TabContext
# ═════════════════════════════════════════════════════════
_ctx = TabContext(
    sn_idx=sn_idx, k_val=k_val, z_val=z_val, lag_val=lag_val,
    q_val=q_val, r_val=r_val, rc_val=rc_val,
    sens_val_for_send=sens_val, ignore_pump=ignore_pump,
    auto_optimize=auto_optimize, show_pure=show_pure,
    file_path_to_send=file_path_to_send, api_key=api_key,
    detect_methods=detect_methods, sigma_drop=sigma_drop,
    buffer_days=buffer_days, correction_strategy=correction_strategy,
)


# ═════════════════════════════════════════════════════════
# Tabs
# ═════════════════════════════════════════════════════════
(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10) = st.tabs([
    "1. Basic analysis",
    "2. Pumping correction",
    "3. Soil compare",
    "4. Cross-validation",
    "5. Uncertainty",
    "6. Sensitivity",
    "7. EnKF (synthetic)",
    "8. 📋 Well report",
    "9. Field (synthetic)",
    "10. 🗾 Watershed recharge",
])

tab_base_en.render(t1, _ctx)            # English Tab 1 (full)
tab_pump_en.render(t2, _ctx)            # English Tab 2 (full)
# Tabs 3–7, 9 — Korean UI retained (auxiliary, not paper-critical)
with t3:
    st.caption("ℹ️ This auxiliary tab uses the Korean UI. "
               "Paper-critical functionality is exposed in Tabs 1, 2, 8, 10.")
tab_compare.render(t3, _ctx)
with t4:
    st.caption("ℹ️ Korean UI (auxiliary).")
tab_crossval.render(t4, _ctx)
with t5:
    st.caption("ℹ️ Korean UI (auxiliary).")
tab_uncertainty.render(t5, _ctx)
with t6:
    st.caption("ℹ️ Korean UI (auxiliary).")
tab_sensitivity.render(t6, _ctx)
with t7:
    st.caption("ℹ️ Korean UI (auxiliary).")
tab_spatial.render(t7, _ctx)
# Tab 8 well report — Korean variant retained; HTML output supports
# both Korean and English depending on AI prompt language
with t8:
    st.caption("ℹ️ Tab 8 currently uses the Korean UI. "
               "The HTML report can be generated in English by switching "
               "the AI commentary language inside the report.")
from ui import tab_well_report
tab_well_report.render(t8, _ctx)
with t9:
    st.caption("ℹ️ Korean UI (auxiliary, synthetic field report).")
tab_field_report.render(t9, _ctx)

try:
    tab_watershed_report_en.render(t10, _ctx)   # English Tab 10
except Exception as e:
    with t10:
        st.error(f"⚠️ Tab 10 render failed: {e}")
        import traceback
        st.code(traceback.format_exc())
