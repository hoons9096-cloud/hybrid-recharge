"""
ui/ — hybrid-recharge v30 Streamlit UI modules.

Shared constants, helpers, and TabContext dataclass used by all tab renderers.
Each tab module exposes a ``render(tab, ctx)`` function.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from soil_db import (
    SOIL_NAMES_NUMBERED as SOIL_NAMES,
    CLAY_LIKE_SET,
    get_soil,
)
from scoring import score_dataframe, score_k_stress, score_sy_match

# ═══════════════════════════════════════════════════════
# Plotly colour palette & base layout
# ═══════════════════════════════════════════════════════
C = {
    "observed":  "#94A3B8",
    "kalman":    "#2563EB",
    "corrected": "#DC2626",
    "rain":      "#93C5FD",
    "pump_bg":   "rgba(239,68,68,0.10)",
    "pump_line": "rgba(239,68,68,0.4)",
    "rech_raw":  "#10B981",
    "rech_corr": "#8B5CF6",
    "grid":      "#E2E6ED",
}

LAYOUT_BASE = dict(
    font=dict(family="Pretendard, -apple-system, sans-serif", size=12, color="#1B2A4A"),
    paper_bgcolor="#FAFBFD",
    plot_bgcolor="#FFFFFF",
    margin=dict(l=60, r=30, t=60, b=50),
)


# ═══════════════════════════════════════════════════════
# TabContext — shared state passed to every tab renderer
# ═══════════════════════════════════════════════════════
@dataclass
class TabContext:
    """Bundles sidebar parameters and runtime state for tab renderers."""
    # Sidebar params
    sn_idx: int = 1
    k_val: float = -0.015
    z_val: float = 3.0
    lag_val: int = 0
    q_val: float = 0.005
    r_val: float = 0.10
    rc_val: float = 0.005
    sens_val_for_send: float = 1.0
    ignore_pump: float = 0.0
    auto_optimize: bool = True
    show_pure: bool = True
    file_path_to_send: str = "DEMO"
    api_key: str = ""
    # Pump preprocess sidebar params
    detect_methods: list = field(default_factory=lambda: ["sigma", "rolling_baseline"])
    sigma_drop: float = 2.5
    buffer_days: int = 2
    correction_strategy: str = "auto"
    # Sensitivity sidebar params
    n_sweep: int = 9
    topsis_delta: float = 0.20
    # EnKF sidebar params (populated only when Tab 7 is used)
    enkf_kwargs: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════
# Shared helper functions
# ═══════════════════════════════════════════════════════

def build_hybrid_radar(df_scan: pd.DataFrame):
    """Build a radar chart for the top / bottom / mid soil scan rows."""
    n = len(df_scan)
    idx_set = [0]
    if n > 1:
        idx_set.append(1)
    if n > 4:
        idx_set.append(n // 2)
    if n > 2:
        idx_set.append(n - 1)
    seen, display_idx = set(), []
    for x in idx_set:
        if x not in seen:
            seen.add(x)
            display_idx.append(x)
    display_rows = df_scan.iloc[display_idx]

    palette = ["#DC2626", "#3B82F6", "#059669", "#F59E0B", "#8B5CF6"]
    cats = ["k-Stress", "Sy Match", "Math Fit", "Rain Resp", "Cleanliness"]
    fig = go.Figure()

    for i, (_, row) in enumerate(display_rows.iterrows()):
        sn = int(row["Index"])
        s_stress = (
            score_k_stress(float(row["OptK"]), sn)
            if "OptK" in row and pd.notna(row.get("OptK"))
            else float(row.get("StressScore", 50))
        )
        soil = get_soil(sn)
        sy_eff_v = float(row.get("SyEff", soil.sy_lit))
        s_sy = score_sy_match(sy_eff_v, sn)
        s_fit = float(row.get("FitScore", 50))
        s_resp = float(row.get("RespScore", 50))
        s_clean = float(row.get("CleanScore", 50))
        vals = [s_stress, s_sy, s_fit, s_resp, s_clean, s_stress]
        theta = cats + [cats[0]]
        h = palette[i].lstrip("#")
        r_c, g_c, b_c = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=theta,
            name=f"{row['Soil']} ({row['HybridScore']:.0f})",
            fill="toself", fillcolor=f"rgba({r_c},{g_c},{b_c},0.08)",
            line=dict(color=palette[i], width=2.5), marker=dict(size=5),
        ))

    fig.update_layout(
        **LAYOUT_BASE, height=440,
        title="<b>Soil Fitness Radar</b>  (1위·2위·중간·최하위)",
        polar=dict(
            bgcolor="#FFF",
            radialaxis=dict(range=[0, 100], tickfont=dict(size=10), gridcolor="#DDD"),
            angularaxis=dict(tickfont=dict(size=11), gridcolor="#DDD"),
        ),
        legend=dict(yanchor="top", y=1.0, xanchor="left", x=1.05, font=dict(size=11)),
    )
    return fig


def shade_pump_plotly(fig, days, pump_mask, color=None, row=None, col=None):
    """Shade pumping intervals as Plotly vrects."""
    if color is None:
        color = C["pump_bg"]
    in_pump = False
    s = 0
    for i, m in enumerate(pump_mask):
        if m and not in_pump:
            in_pump = True
            s = i
        if in_pump and (not m or i == len(pump_mask) - 1):
            e = i if not m else i + 1
            fig.add_vrect(x0=s, x1=e, fillcolor=color, line_width=0,
                          layer="below", row=row, col=col)
            in_pump = False


def has_v27_error(result):
    return result is None or "error" in result


def coord_input(
    *,
    default_lat: float = 36.35,
    default_lon: float = 127.37,
    key_prefix: str = "coord",
    show_other_form: bool = True,
):
    """좌표 입력 위젯 (WGS84 ↔ Korean TM 토글).

    Returns
    -------
    (lat: float, lon: float)
        항상 WGS84 (lat, lon) 으로 정규화된 값.

    UI
    --
    1. 라디오: '위경도 (WGS84)' / 'Korean TM (X, Y)'
    2. 두 개 number_input — 선택한 단위에 맞춰 표시
    3. 다른 형식의 좌표를 보조표시 (검증용)
    """
    from coord_utils import (
        tm_to_wgs84, wgs84_to_tm, looks_like_tm, looks_like_wgs84,
    )

    mode = st.radio(
        "좌표 형식",
        options=["위경도 (WGS84)", "Korean TM (X, Y)"],
        key=f"{key_prefix}_mode", horizontal=True,
    )

    c1, c2 = st.columns(2)
    if mode.startswith("위경도"):
        with c1:
            lat = st.number_input(
                "위도 (WGS84)", min_value=33.0, max_value=39.0,
                value=float(default_lat), step=0.0001, format="%.4f",
                key=f"{key_prefix}_lat",
            )
        with c2:
            lon = st.number_input(
                "경도 (WGS84)", min_value=124.0, max_value=132.0,
                value=float(default_lon), step=0.0001, format="%.4f",
                key=f"{key_prefix}_lon",
            )
        if show_other_form and looks_like_wgs84(lat, lon):
            try:
                x, y = wgs84_to_tm(lat, lon)
                st.caption(f"= TM (X, Y) ≈ ({x:,.0f} m, {y:,.0f} m)  "
                            f"[EPSG:5186 Korean TM Central]")
            except Exception:
                pass
        return float(lat), float(lon)

    # ── TM mode ──
    # default_lat/lon → 기본 X, Y 추정
    try:
        default_x, default_y = wgs84_to_tm(default_lat, default_lon)
    except Exception:
        default_x, default_y = 250000.0, 450000.0
    with c1:
        x = st.number_input(
            "X (Easting, m)", min_value=50_000.0, max_value=500_000.0,
            value=float(default_x), step=1.0, format="%.1f",
            key=f"{key_prefix}_tm_x",
            help="EPSG:5186 Korean TM Central — 한국 토양도 좌표계",
        )
    with c2:
        y = st.number_input(
            "Y (Northing, m)", min_value=200_000.0, max_value=800_000.0,
            value=float(default_y), step=1.0, format="%.1f",
            key=f"{key_prefix}_tm_y",
        )

    if not looks_like_tm(x, y):
        st.warning("⚠️ 입력값이 한국 TM 범위를 벗어납니다. 좌표계를 확인하세요.")
    try:
        lat, lon = tm_to_wgs84(x, y)
    except Exception as e:
        st.error(f"TM → WGS84 변환 실패: {e}")
        lat, lon = float(default_lat), float(default_lon)

    if show_other_form:
        st.caption(f"= WGS84 (lat, lon) ≈ ({lat:.5f}, {lon:.5f})")
    return float(lat), float(lon)
