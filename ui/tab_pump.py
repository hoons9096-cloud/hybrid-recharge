"""Tab 2 — 펌핑 전처리 결과."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from ui import C, LAYOUT_BASE, TabContext, shade_pump_plotly


def render(tab, ctx: TabContext):
    """Render Tab 2 inside the given Streamlit tab container."""
    with tab:
        pr = st.session_state.get("pump_result")
        if pr is None:
            st.info("사이드바 → '2단계: 펌핑 전처리 + 재분석' 을 실행하세요.")
            return

        st.markdown("### 🔧 펌핑 전처리 결과")

        # ── 펌핑 미탐지 경고 ──
        if not pr.get("pump_detected", True):
            st.warning("⚠️ **펌핑 구간이 탐지되지 않았습니다.** 보정 수위 = 원본 수위")

        # ── v27 함양율 비교 (핵심: 같은 알고리즘으로 비교) ──
        v27o = pr.get("v27_orig")
        v27c = pr.get("v27_corr")
        if v27o and v27c:
            st.markdown("#### 📊 v27 WTF 함양율 비교")
            vc1, vc2, vc3, vc4 = st.columns(4)
            vc1.metric("원본 함양율", f"{v27o['rech_rate']:.2f}%")
            delta_rr = v27c['rech_rate'] - v27o['rech_rate']
            vc2.metric("보정 함양율", f"{v27c['rech_rate']:.2f}%",
                       delta=f"{delta_rr:+.2f}%p")
            vc3.metric("원본 RMSE", f"{v27o['rmse']:.4f} m")
            delta_rmse = v27c['rmse'] - v27o['rmse']
            vc4.metric("보정 RMSE", f"{v27c['rmse']:.4f} m",
                       delta=f"{delta_rmse:+.4f}", delta_color="inverse")
            st.caption("ℹ️ 동일한 v27 WTF 알고리즘으로 원본/보정 수위를 분석한 결과입니다.")

        # ── 펌핑 탐지 지표 ──
        st.markdown("#### 🔍 펌핑 탐지")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pump Days", f"{int(pr['pump_mask'].sum())}",
                  delta=f"{pr['pump_fraction']*100:.1f}% of record")
        c2.metric("Events", f"{pr['n_events']}")
        c3.metric("Correction", pr['correction_strategy'])
        c4.metric("Kalman RMSE 개선",
                  f"{pr['raw']['rmse'] - pr['corrected']['rmse']:+.4f} m",
                  delta_color="inverse")

        # ── 수위 비교 차트 ──
        _render_wl_comparison(pr)

        # ── 탐지 방법별 마스크 ──
        _render_method_masks(pr)

        # ── Kalman 시뮬레이션 비교 ──
        _render_kalman_comparison(pr)

        # ── 함양 비교 ──
        _render_recharge_comparison(pr)

        # ── 💾 펌프보정 결과 직접 저장 (well_results_store) ──
        _render_pump_save_button(pr, ctx)


def _render_pump_save_button(pr, ctx: TabContext):
    """펌프보정된 함양율을 well_results_store 에 직접 저장.

    Tab 1 까지 이동할 필요 없이, 펌핑 전처리 결과 화면에서 바로 저장.
    `recharge_ratio_corrected` 값을 우선 사용 + `pump_corrected=True` 플래그.
    """
    import os
    st.markdown("---")
    st.markdown("#### 💾 펌프보정 함양율 저장")

    v27c = pr.get("v27_corr")
    v27o = pr.get("v27_orig")
    if not v27c or not v27o:
        st.info("v27 비교 결과가 없습니다.")
        return

    rr_corr = float(v27c["rech_rate"])
    rr_orig = float(v27o["rech_rate"])
    st.caption(
        f"보정 함양율 **{rr_corr:.2f}%** (원본 {rr_orig:.2f}%, "
        f"Δ {rr_corr-rr_orig:+.2f}%p) 를 저장합니다. "
        "Tab 10 (유역 함양율) 의 Cached 모드에서 즉시 사용됩니다."
    )

    try:
        from wells_registry import WELLS
        registered = sorted(WELLS.keys())
    except Exception:
        registered = []

    upl = st.session_state.get("uploaded_name", "") or ""
    base = os.path.splitext(upl)[0] if upl else ""
    default_idx = registered.index(base) if base in registered else 0

    col1, col2 = st.columns([3, 1])
    with col1:
        if registered:
            well_name = st.selectbox(
                "관정명 (wells_registry 등록명)", options=registered,
                index=default_idx, key="pump_save_well",
            )
        else:
            st.error("등록된 관정이 없습니다. Tab 1 에서 먼저 등록하세요.")
            return
    with col2:
        st.write("")
        st.write("")
        if st.button("💾 펌프보정 저장",
                      type="primary", use_container_width=True,
                      key="pump_save_btn"):
            try:
                from well_results_store import (
                    from_result_v27 as _from_v27, save as _save,
                )
                # 원본 result_v27 가 있으면 그 위에 보정값을 얹고 저장
                base_result = st.session_state.get("result_v27") or {}
                merged = dict(base_result)
                merged["recharge_ratio_corrected"] = rr_corr
                merged["recharge_ratio"] = rr_orig
                merged["rmse"] = float(v27c.get("rmse", merged.get("rmse", 0)))
                merged["cc"]   = float(v27c.get("cc",   merged.get("cc", 0)))
                if "opt_k" in v27c:
                    merged["opt_k"] = float(v27c["opt_k"])
                if "opt_z" in v27c:
                    merged["opt_z"] = float(v27c["opt_z"])

                # 메타 (registry 에서 가져옴)
                aquifer = hydro_type = soil_code = None
                lat = lon = None
                if well_name in WELLS:
                    info = WELLS[well_name]
                    aquifer = info.aquifer
                    lat, lon = info.lat, info.lon
                    try:
                        from shp_soil_mapper import query_point
                        sq = query_point(well_name, info.lat, info.lon)
                        hydro_type = sq.hydro_type
                        soil_code = sq.soil_code
                    except Exception:
                        pass

                stored = _from_v27(
                    well_name=well_name,
                    result_v27=merged,
                    file_path=ctx.file_path_to_send or "",
                    sn_idx=int(ctx.sn_idx),
                    pump_corrected=True,
                    aquifer=aquifer, hydro_type=hydro_type,
                    soil_code=soil_code, lat=lat, lon=lon,
                )
                path = _save(stored)
                st.success(
                    f"✅ 저장 완료: `{path}`  \n"
                    f"recharge_ratio_pct = {stored.recharge_ratio_pct:.2f}%, "
                    f"pump_corrected = True"
                )
            except Exception as e:
                st.error(f"저장 실패: {type(e).__name__}: {e}")
                import traceback
                with st.expander("Traceback"):
                    st.code(traceback.format_exc())


def _render_wl_comparison(pr):
    """Water level: raw vs corrected."""
    st.markdown("#### 수위: 원본 vs 보정")
    days = np.arange(len(pr["raw_wl"]))
    pm = pr["pump_mask"]

    fig_wl = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           row_heights=[0.7, 0.3], vertical_spacing=0.08)

    fig_wl.add_trace(go.Scatter(x=days, y=pr["raw_wl"], mode="lines",
        name="Raw WL", line=dict(color=C["observed"], width=1)), row=1, col=1)
    fig_wl.add_trace(go.Scatter(x=days, y=pr["corrected_wl"], mode="lines",
        name="Corrected WL", line=dict(color=C["corrected"], width=1.5)), row=1, col=1)
    shade_pump_plotly(fig_wl, days, pm, row=1, col=1)

    fig_wl.add_trace(go.Scatter(x=days, y=pr["confidence"],
        fill="tozeroy", name="Detection Confidence",
        fillcolor="rgba(251,191,36,0.3)",
        line=dict(color="rgba(251,191,36,0.8)", width=1)), row=2, col=1)
    fig_wl.add_hline(y=0.5, line_dash="dot", line_color="gray", row=2, col=1)

    _layout_wl = {**LAYOUT_BASE, "margin": dict(l=60, r=30, t=60, b=100)}
    fig_wl.update_layout(**_layout_wl, height=560,
        title="<b>Water Level Correction Overview</b>",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"))
    fig_wl.update_yaxes(title="GW Level (m)", row=1, col=1, gridcolor=C["grid"])
    fig_wl.update_yaxes(title="Confidence", range=[0, 1], row=2, col=1, gridcolor=C["grid"])
    fig_wl.update_xaxes(title="Time (days)", row=2, col=1, gridcolor=C["grid"],
                         tickfont=dict(size=11), nticks=12)
    fig_wl.update_xaxes(showticklabels=False, row=1, col=1)
    st.plotly_chart(fig_wl, use_container_width=True, theme=None)


def _render_method_masks(pr):
    """Per-method detection masks."""
    with st.expander("🔍 탐지 방법별 상세", expanded=False):
        mm = pr.get("method_masks", {})
        if mm:
            days = np.arange(len(pr["raw_wl"]))
            n_m = len(mm)
            fig_det = make_subplots(rows=n_m, cols=1, shared_xaxes=True,
                                     vertical_spacing=0.05,
                                     subplot_titles=list(mm.keys()))
            colors_m = ["#DC2626", "#2563EB", "#10B981"]
            for i, (mname, mask) in enumerate(mm.items()):
                fig_det.add_trace(go.Scatter(
                    x=days, y=mask.astype(float),
                    fill="tozeroy", name=mname,
                    fillcolor=f"rgba({','.join(str(int(c, 16)) for c in [colors_m[i%3][1:3], colors_m[i%3][3:5], colors_m[i%3][5:7]])},0.5)",
                    line=dict(width=0),
                ), row=i + 1, col=1)
            fig_det.update_layout(**LAYOUT_BASE, height=200 * n_m, showlegend=False)
            st.plotly_chart(fig_det, use_container_width=True, theme=None)


def _render_kalman_comparison(pr):
    """Kalman simulation: before vs after preprocessing."""
    st.markdown("#### Kalman 수위 재현: 전처리 전 vs 후")
    days = np.arange(len(pr["raw_wl"]))
    pm = pr["pump_mask"]

    fig_kal = make_subplots(specs=[[{"secondary_y": True}]])

    rain_plot = pr["rainfall"]
    fig_kal.add_trace(go.Bar(x=days, y=rain_plot, name="Rain (mm)",
        marker=dict(color=C["rain"], opacity=0.3, line=dict(width=0))),
        secondary_y=True)
    fig_kal.add_trace(go.Scatter(x=days, y=pr["raw_wl"], mode="markers",
        name="Observed", marker=dict(color=C["observed"], size=3, opacity=0.4)),
        secondary_y=False)
    fig_kal.add_trace(go.Scatter(x=days, y=pr["raw"]["h_sim"], mode="lines",
        name=f"Kalman (raw)  RMSE={pr['raw']['rmse']:.4f}",
        line=dict(color=C["kalman"], width=2)), secondary_y=False)
    fig_kal.add_trace(go.Scatter(x=days, y=pr["corrected"]["h_sim"], mode="lines",
        name=f"Kalman (corrected)  RMSE={pr['corrected']['rmse']:.4f}",
        line=dict(color=C["corrected"], width=2, dash="dash")), secondary_y=False)
    shade_pump_plotly(fig_kal, days, pm)

    _layout_kal = {**LAYOUT_BASE, "margin": dict(l=60, r=30, t=60, b=110)}
    fig_kal.update_layout(**_layout_kal, height=520,
        title="<b>Kalman Simulation — Before vs After Pump Preprocessing</b>",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.24, x=0.5, xanchor="center",
                    bgcolor="rgba(255,255,255,0.9)", bordercolor="#CCC", borderwidth=1))
    fig_kal.update_xaxes(title="Time (days)", gridcolor=C["grid"],
                          tickfont=dict(size=11), nticks=12)
    fig_kal.update_yaxes(title="GW Level (m)", secondary_y=False, gridcolor=C["grid"])
    r_max = float(np.nanmax(rain_plot)) if len(rain_plot) > 0 else 10
    fig_kal.update_yaxes(title="Rain (mm)", range=[r_max * 3, 0], secondary_y=True)
    st.plotly_chart(fig_kal, use_container_width=True, theme=None)


def _render_recharge_comparison(pr):
    """Daily recharge bar charts."""
    st.markdown("#### 일 함양량 비교")
    days = np.arange(len(pr["raw_wl"]))
    fig_rech = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              vertical_spacing=0.08,
                              subplot_titles=["Recharge (no preproc)", "Recharge (corrected)"])

    rech0 = np.nan_to_num(pr["raw"]["rech_total"], 0)
    rech1 = np.nan_to_num(pr["corrected"]["rech_total"], 0)

    fig_rech.add_trace(go.Bar(x=days, y=rech0, name="No preproc",
        marker=dict(color=C["rech_raw"], opacity=0.8)), row=1, col=1)
    fig_rech.add_trace(go.Bar(x=days, y=rech1, name="Corrected",
        marker=dict(color=C["rech_corr"], opacity=0.8)), row=2, col=1)

    _layout_rech = {**LAYOUT_BASE, "margin": dict(l=60, r=30, t=60, b=100)}
    fig_rech.update_layout(**_layout_rech, height=450,
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"))
    fig_rech.update_yaxes(title="mm/day", row=1, col=1, gridcolor=C["grid"])
    fig_rech.update_yaxes(title="mm/day", row=2, col=1, gridcolor=C["grid"])
    fig_rech.update_xaxes(showticklabels=False, row=1, col=1)
    fig_rech.update_xaxes(title="Time (days)", row=2, col=1, gridcolor=C["grid"],
                           tickfont=dict(size=11), nticks=12)
    st.plotly_chart(fig_rech, use_container_width=True, theme=None)
