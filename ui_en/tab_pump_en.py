"""Tab 2 (English) — Pumping pre-processing results."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from ui import C, LAYOUT_BASE, TabContext, shade_pump_plotly


def render(tab, ctx: TabContext):
    with tab:
        pr = st.session_state.get("pump_result")
        if pr is None:
            st.info("Sidebar → run **'Step 2: Pumping pre-processing + reanalysis'** first.")
            return

        st.markdown("### 🔧 Pumping pre-processing results")

        if not pr.get("pump_detected", True):
            st.warning("⚠️ **No pumping segments detected.** Corrected level = raw level")

        # v27 recharge comparison
        v27o = pr.get("v27_orig"); v27c = pr.get("v27_corr")
        if v27o and v27c:
            st.markdown("#### 📊 v27 WTF recharge comparison")
            vc1, vc2, vc3, vc4 = st.columns(4)
            vc1.metric("Raw recharge", f"{v27o['rech_rate']:.2f}%")
            delta_rr = v27c['rech_rate'] - v27o['rech_rate']
            vc2.metric("Corrected recharge", f"{v27c['rech_rate']:.2f}%",
                        delta=f"{delta_rr:+.2f}pp")
            vc3.metric("Raw RMSE", f"{v27o['rmse']:.4f} m")
            delta_rmse = v27c['rmse'] - v27o['rmse']
            vc4.metric("Corrected RMSE", f"{v27c['rmse']:.4f} m",
                        delta=f"{delta_rmse:+.4f}", delta_color="inverse")
            st.caption("ℹ️ Same v27 WTF algorithm applied to raw and corrected water levels.")

        # Detection metrics
        st.markdown("#### 🔍 Pumping detection")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pump days", f"{int(pr['pump_mask'].sum())}",
                  delta=f"{pr['pump_fraction']*100:.1f}% of record")
        c2.metric("Events", f"{pr['n_events']}")
        c3.metric("Correction", pr['correction_strategy'])
        c4.metric("Kalman RMSE improvement",
                  f"{pr['raw']['rmse'] - pr['corrected']['rmse']:+.4f} m",
                  delta_color="inverse")

        _render_wl_comparison(pr)
        _render_method_masks_en(pr)
        _render_kalman_comparison_en(pr)
        _render_recharge_comparison_en(pr)


def _render_wl_comparison(pr):
    st.markdown("#### Water level: raw vs corrected")
    days = np.arange(len(pr["raw_wl"]))
    pm = pr["pump_mask"]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(x=days, y=pr["raw_wl"], mode="lines",
        name="Raw WL", line=dict(color=C["observed"], width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=days, y=pr["corrected_wl"], mode="lines",
        name="Corrected WL", line=dict(color=C["corrected"], width=1.5)),
        row=1, col=1)
    shade_pump_plotly(fig, days, pm, row=1, col=1)
    fig.add_trace(go.Scatter(x=days, y=pr["confidence"], fill="tozeroy",
        name="Detection confidence",
        fillcolor="rgba(251,191,36,0.3)",
        line=dict(color="rgba(251,191,36,0.8)", width=1)), row=2, col=1)
    fig.add_hline(y=0.5, line_dash="dot", line_color="gray", row=2, col=1)

    layout = {**LAYOUT_BASE, "margin": dict(l=60, r=30, t=60, b=100)}
    fig.update_layout(**layout, height=560,
                       title="<b>Water level correction overview</b>",
                       hovermode="x unified",
                       legend=dict(orientation="h", y=-0.18, x=0.5,
                                    xanchor="center"))
    fig.update_yaxes(title="GW level (m)", row=1, col=1, gridcolor=C["grid"])
    fig.update_yaxes(title="Confidence", range=[0, 1], row=2, col=1,
                      gridcolor=C["grid"])
    fig.update_xaxes(title="Time (days)", row=2, col=1, gridcolor=C["grid"],
                      tickfont=dict(size=11), nticks=12)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    st.plotly_chart(fig, use_container_width=True, theme=None)


def _render_method_masks_en(pr):
    methods = pr.get("method_masks", {})
    if not methods:
        return
    st.markdown("#### Detection masks per method")
    days = np.arange(len(pr["raw_wl"]))
    fig = go.Figure()
    colors_pal = ["#DC2626", "#0891B2", "#7C3AED", "#10B981", "#F59E0B"]
    for i, (method_name, mask) in enumerate(methods.items()):
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            continue
        fig.add_trace(go.Scatter(
            x=days[mask], y=[i] * mask.sum(), mode="markers",
            name=method_name,
            marker=dict(color=colors_pal[i % len(colors_pal)], size=4),
        ))
    fig.update_layout(
        height=180 + 30 * len(methods),
        title="Per-method pumping flags",
        xaxis_title="Day", yaxis=dict(showticklabels=False),
        legend=dict(orientation="h"),
        margin=dict(l=20, r=20, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)


def _render_kalman_comparison_en(pr):
    st.markdown("#### Kalman fit: raw vs corrected")
    raw, cor = pr["raw"], pr["corrected"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Raw RMSE", f"{raw['rmse']:.4f} m")
    c2.metric("Corrected RMSE", f"{cor['rmse']:.4f} m",
              delta=f"{cor['rmse'] - raw['rmse']:+.4f}", delta_color="inverse")
    c3.metric("Δ recharge",
              f"{cor['rech_rate'] - raw['rech_rate']:+.2f}pp",
              help="Corrected − Raw")


def _render_recharge_comparison_en(pr):
    st.markdown("#### Recharge — raw vs corrected")
    raw_r = pr["raw"]["rech_rate"]
    cor_r = pr["corrected"]["rech_rate"]
    fig = go.Figure(go.Bar(
        x=["Raw", "Corrected"], y=[raw_r, cor_r],
        marker_color=["#9CA3AF", "#DC2626"],
        text=[f"{raw_r:.2f}%", f"{cor_r:.2f}%"], textposition="outside",
    ))
    fig.update_layout(height=320,
                       yaxis_title="Recharge ratio (% of P)",
                       margin=dict(l=40, r=20, t=30, b=40))
    st.plotly_chart(fig, use_container_width=True, theme=None)
