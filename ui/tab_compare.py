"""Tab 3 — 비교 분석 (전처리 효과)."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui import C, LAYOUT_BASE, SOIL_NAMES, TabContext


def render(tab, ctx: TabContext):
    """Render Tab 3 inside the given Streamlit tab container."""
    with tab:
        pr = st.session_state.get("pump_result")
        if pr is None:
            st.info("2단계 펌핑 전처리를 먼저 실행하면 비교 분석이 표시됩니다.")
            return

        st.markdown("### ⚖️ 전처리 효과 비교 (v27 WTF 기준)")

        pump_detected_tab3 = pr.get("pump_detected", True)
        v27o = pr.get("v27_orig")
        v27c = pr.get("v27_corr")

        if not pump_detected_tab3:
            st.warning("⚠️ **펌핑 구간이 탐지되지 않았습니다.** 보정 수위 = 원본 수위이므로 함양율 차이가 없습니다.")

        # ── v27 비교 (핵심) ──
        if v27o and v27c:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### v27 WTF — 원본 수위")
                m1, m2, m3 = st.columns(3)
                m1.metric("RMSE", f"{v27o['rmse']:.4f} m")
                m2.metric("CC", f"{v27o['cc']:.4f}")
                m3.metric("Recharge", f"{v27o['rech_rate']:.2f}%")

            with col2:
                st.markdown("##### v27 WTF — 보정 수위")
                m1, m2, m3 = st.columns(3)
                m1.metric("RMSE", f"{v27c['rmse']:.4f} m",
                          delta=f"{v27c['rmse'] - v27o['rmse']:+.4f}", delta_color="inverse")
                m2.metric("CC", f"{v27c['cc']:.4f}",
                          delta=f"{v27c['cc'] - v27o['cc']:+.4f}")
                delta_rr = v27c['rech_rate'] - v27o['rech_rate']
                m3.metric("Recharge", f"{v27c['rech_rate']:.2f}%",
                          delta=f"{delta_rr:+.2f}%p")

            st.caption("ℹ️ 동일한 v27 WTF 알고리즘으로 원본/보정 수위를 분석 — 차이는 순수하게 펌핑 전처리 효과입니다.")
        else:
            st.info("v27 비교 결과가 없습니다. ③ 펌핑 전처리를 다시 실행하세요.")

        # ── AugKalman 참고 (접을 수 있게) ──
        raw = pr["raw"]
        cor = pr["corrected"]
        with st.expander("🔬 참고: Augmented Kalman 비교 (다른 알고리즘)", expanded=False):
            st.caption("2-상태 확장 칼만 필터 결과입니다. v27 함양율과 직접 비교하지 마세요.")
            ac1, ac2 = st.columns(2)
            with ac1:
                st.metric("AugKalman 원본", f"{raw['rech_rate']:.1f}%")
            with ac2:
                st.metric("AugKalman 보정", f"{cor['rech_rate']:.1f}%",
                          delta=f"{cor['rech_rate'] - raw['rech_rate']:+.1f}%p")

        st.markdown("---")

        # ── 토양 점수 비교 ──
        _render_soil_score_comparison(raw, cor)

        # ── 누적 함양 비교 (v27 기준) ──
        _render_cumulative_recharge(v27o, v27c)

        # ── 상세 비교 테이블 ──
        _render_detail_table(v27o, v27c, raw, cor)

        # ── CSV 내보내기 ──
        _render_export(pr)


def _render_soil_score_comparison(raw, cor):
    st.markdown("#### 토양별 복합 점수 비교")
    fig_soil = go.Figure()
    x_soil = list(range(12))
    fig_soil.add_trace(go.Bar(
        x=x_soil, y=raw["soil_scores"], name="No preproc",
        marker=dict(color=C["kalman"], opacity=0.7), width=0.35,
        offset=-0.18,
    ))
    fig_soil.add_trace(go.Bar(
        x=x_soil, y=cor["soil_scores"], name="Corrected",
        marker=dict(color=C["corrected"], opacity=0.7), width=0.35,
        offset=0.18,
    ))
    _layout_soil = {**LAYOUT_BASE, "margin": dict(l=60, r=30, t=60, b=110)}
    fig_soil.update_layout(**_layout_soil, height=430,
        title="<b>Soil Identification Score (12 Standard Soils)</b>",
        xaxis=dict(
            tickmode="array",
            tickvals=x_soil,
            ticktext=[s.split(". ")[1] if ". " in s else s for s in SOIL_NAMES],
            tickangle=40,
            tickfont=dict(size=10),
        ),
        yaxis=dict(title="Composite Score", gridcolor=C["grid"]),
        legend=dict(orientation="h", y=-0.28, x=0.5, xanchor="center"),
        barmode="group",
    )
    st.plotly_chart(fig_soil, use_container_width=True, theme=None)


def _render_cumulative_recharge(v27o, v27c):
    st.markdown("#### 누적 함양량 비교")
    if v27o and v27c and v27o.get("rech") and v27c.get("rech"):
        rech_v27_orig = np.array(v27o["rech"], dtype=float)
        rech_v27_corr = np.array(v27c["rech"], dtype=float)
        cum_orig = np.nancumsum(rech_v27_orig) * 1000.0
        cum_corr = np.nancumsum(rech_v27_corr) * 1000.0
        days_cum = np.arange(len(cum_orig))

        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(x=days_cum, y=cum_orig, mode="lines",
            name=f"v27 원본 ({v27o['rech_rate']:.2f}%)",
            line=dict(color=C["rech_raw"], width=2)))
        fig_cum.add_trace(go.Scatter(x=days_cum, y=cum_corr, mode="lines",
            name=f"v27 보정 ({v27c['rech_rate']:.2f}%)",
            line=dict(color=C["rech_corr"], width=2)))
        _layout_cum = {**LAYOUT_BASE, "margin": dict(l=60, r=30, t=60, b=100)}
        fig_cum.update_layout(**_layout_cum, height=400,
            title="<b>Cumulative Recharge (v27 WTF)</b>",
            xaxis=dict(title="Time (days)", gridcolor=C["grid"]),
            yaxis=dict(title="Cumulative Recharge (mm)", gridcolor=C["grid"]),
            legend=dict(orientation="h", y=-0.24, x=0.5, xanchor="center"))
        st.plotly_chart(fig_cum, use_container_width=True, theme=None)
    else:
        st.info("v27 누적 함양 데이터가 없습니다.")


def _render_detail_table(v27o, v27c, raw, cor):
    with st.expander("📋 상세 비교 테이블", expanded=True):
        if v27o and v27c:
            comp_df = pd.DataFrame({
                "지표": ["RMSE (m)", "CC", "함양율 (%)", "k 값", "z 값"],
                "v27 원본": [
                    f"{v27o['rmse']:.4f}", f"{v27o['cc']:.4f}",
                    f"{v27o['rech_rate']:.2f}", f"{v27o['opt_k']:.4f}", f"{v27o['opt_z']:.2f}",
                ],
                "v27 보정": [
                    f"{v27c['rmse']:.4f}", f"{v27c['cc']:.4f}",
                    f"{v27c['rech_rate']:.2f}", f"{v27c['opt_k']:.4f}", f"{v27c['opt_z']:.2f}",
                ],
            })
            st.dataframe(comp_df, hide_index=True, use_container_width=True)
        st.caption("참고: Augmented Kalman (별도 알고리즘)")
        comp_df_ak = pd.DataFrame({
            "지표": ["RMSE (m)", "NSE", "함양율 (%)", "토양"],
            "AugKalman 원본": [
                f"{raw['rmse']:.4f}", f"{raw['nse']:.3f}",
                f"{raw['rech_rate']:.1f}", raw["soil"],
            ],
            "AugKalman 보정": [
                f"{cor['rmse']:.4f}", f"{cor['nse']:.3f}",
                f"{cor['rech_rate']:.1f}", cor["soil"],
            ],
        })
        st.dataframe(comp_df_ak, hide_index=True, use_container_width=True)


def _render_export(pr):
    with st.expander("📥 결과 내보내기", expanded=False):
        df_export = pd.DataFrame({
            "day": np.arange(len(pr["raw_wl"])),
            "raw_wl": pr["raw_wl"],
            "corrected_wl": pr["corrected_wl"],
            "pump_mask": pr["pump_mask"].astype(int),
            "rainfall_mm": pr["rainfall"],
            "rech_raw": pr["raw"]["rech_total"],
            "rech_corrected": pr["corrected"]["rech_total"],
            "h_sim_raw": pr["raw"]["h_sim"],
            "h_sim_corrected": pr["corrected"]["h_sim"],
        })
        csv_bytes = df_export.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="⬇️ 전체 결과 CSV 다운로드",
            data=csv_bytes,
            file_name="wtf_v30_pump_result.csv",
            mime="text/csv",
        )
