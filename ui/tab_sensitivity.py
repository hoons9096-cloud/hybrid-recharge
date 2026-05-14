"""Tab 6 — 민감도 분석 (Sensitivity Analysis)."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from ui import LAYOUT_BASE, TabContext


def render(tab, ctx: TabContext):
    """Render Tab 6 inside the given Streamlit tab container."""
    with tab:
        result = st.session_state.get("result_v27")
        if result is None:
            st.info("① 기본 분석을 먼저 실행하세요.")
            return

        st.markdown("### 🔬 민감도 분석")
        st.caption(
            "Kalman 하이퍼파라미터와 TOPSIS 가중치의 민감도를 체계적으로 평가합니다. "
            "SCI 논문 투고 시 필수 요소입니다 (Saltelli et al., 2004)."
        )

        sa_col1, sa_col2 = st.columns(2)
        with sa_col1:
            n_sweep = st.slider("Sweep 단계 수", 5, 15, 9, key="sa_n_sweep")
        with sa_col2:
            topsis_delta = st.slider("TOPSIS 가중치 변동폭 (%)", 10, 40, 20, key="sa_delta") / 100.0

        run_sa = st.button("▶ 민감도 분석 실행", key="btn_sa")

        if run_sa:
            _execute_sensitivity(result, ctx, n_sweep, topsis_delta)

        # ── Display results ──
        _display_kalman_sensitivity()
        _display_objective_weight_sensitivity()
        _display_topsis_sensitivity(topsis_delta)


def _execute_sensitivity(result, ctx: TabContext, n_sweep, topsis_delta):
    from core_sim_v27 import run_logic_v27, calc_error, optimize_parameters
    from core_sim_config import DEFAULT_Q_NOISE, DEFAULT_R_NOISE
    from sensitivity import kalman_sensitivity_sweep, topsis_weight_sensitivity, objective_weight_sensitivity

    po_arr = np.array(result["po_shifted"])
    ho_arr = np.array(result["ho"])
    pm_arr = np.array(result["pump_mask"], dtype=bool)
    opt_k = result["opt_k"]
    opt_z = result["opt_z"]
    sn = int(ctx.sn_idx)

    with st.spinner("Kalman 하이퍼파라미터 sweep 중..."):
        ksa = kalman_sensitivity_sweep(
            run_func=run_logic_v27,
            k=opt_k, z_unsat=opt_z, sn=sn,
            po_in=po_arr, ho_in=ho_arr,
            q_base=DEFAULT_Q_NOISE, r_base=DEFAULT_R_NOISE,
            r_c=ctx.rc_val,
            pump_mask=pm_arr,
            n_steps=n_sweep,
        )
        st.session_state["kalman_sensitivity"] = ksa

    with st.spinner("목적함수 가중치 민감도 분석 중..."):
        owsa = objective_weight_sensitivity(
            k=opt_k, z=opt_z, sn=sn,
            po_shifted=po_arr, ho=ho_arr,
            rc=ctx.rc_val,
            pump_mask=pm_arr,
            calc_error_func=calc_error,
            run_func=run_logic_v27,
            optimize_func=optimize_parameters,
            q_val=DEFAULT_Q_NOISE, r_val=DEFAULT_R_NOISE,
        )
        st.session_state["obj_weight_sensitivity"] = owsa

    scan_df = st.session_state.get("soil_scan_df")
    if scan_df is not None:
        with st.spinner("TOPSIS 가중치 민감도 분석 중..."):
            tsa = topsis_weight_sensitivity(scan_df, delta=topsis_delta)
            st.session_state["topsis_sensitivity"] = tsa

    st.success("민감도 분석 완료!")


def _display_kalman_sensitivity():
    ksa = st.session_state.get("kalman_sensitivity")
    if ksa is None:
        return

    st.markdown("#### 1. Kalman 하이퍼파라미터 민감도")

    e1, e2, e3 = st.columns(3)
    e1.metric("ρ 탄성도", f"{ksa.sensitivity_rho:.3f}")
    e2.metric("Q/R 탄성도", f"{ksa.sensitivity_qr:.3f}")
    e3.metric("α 탄성도", f"{ksa.sensitivity_alpha:.3f}")
    st.caption("탄성도 = |(ΔR/R) / (Δp/p)|: 1.0 이상이면 함양율이 해당 파라미터에 민감합니다.")

    # Tornado diagram
    st.markdown("##### 토네이도 다이어그램")
    tornado = ksa.tornado_data
    fig_t = go.Figure()
    base = ksa.baseline_recharge
    for i, (name, r_lo, r_hi, _, _) in enumerate(tornado):
        fig_t.add_trace(go.Bar(
            y=[name], x=[r_lo - base], base=[base],
            orientation="h", marker_color="steelblue",
            name=f"{name} (하한)", showlegend=False,
            text=[f"{r_lo:.1f}%"], textposition="inside",
        ))
        fig_t.add_trace(go.Bar(
            y=[name], x=[r_hi - base], base=[base],
            orientation="h", marker_color="coral",
            name=f"{name} (상한)", showlegend=False,
            text=[f"{r_hi:.1f}%"], textposition="inside",
        ))
    fig_t.add_vline(x=base, line_dash="dash", line_color="black",
                    annotation_text=f"기준: {base:.1f}%")
    fig_t.update_layout(
        title="Kalman 파라미터 토네이도 (함양율 %)",
        xaxis_title="함양율 (%)",
        barmode="overlay",
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_t, use_container_width=True)

    # Line plots
    with st.expander("파라미터별 sweep 곡선"):
        fig_sw = make_subplots(rows=1, cols=3,
            subplot_titles=["ρ (persistence)", "Q/R ratio", "Blend α"])

        rho_x = [p for p, _ in ksa.rho_sweep]
        rho_y = [r for _, r in ksa.rho_sweep]
        fig_sw.add_trace(go.Scatter(x=rho_x, y=rho_y, mode="lines+markers",
            name="ρ"), row=1, col=1)
        fig_sw.add_vline(x=ksa.baseline_rho, line_dash="dot",
            line_color="red", row=1, col=1)

        qr_x = [p for p, _ in ksa.qr_ratio_sweep]
        qr_y = [r for _, r in ksa.qr_ratio_sweep]
        fig_sw.add_trace(go.Scatter(x=qr_x, y=qr_y, mode="lines+markers",
            name="Q/R"), row=1, col=2)
        fig_sw.add_vline(x=ksa.baseline_qr, line_dash="dot",
            line_color="red", row=1, col=2)

        al_x = [p for p, _ in ksa.alpha_sweep]
        al_y = [r for _, r in ksa.alpha_sweep]
        fig_sw.add_trace(go.Scatter(x=al_x, y=al_y, mode="lines+markers",
            name="α"), row=1, col=3)
        fig_sw.add_vline(x=ksa.baseline_alpha, line_dash="dot",
            line_color="red", row=1, col=3)

        fig_sw.update_yaxes(title_text="함양율 (%)", row=1, col=1)
        fig_sw.update_layout(height=300, showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_sw, use_container_width=True)


def _display_objective_weight_sensitivity():
    owsa = st.session_state.get("obj_weight_sensitivity")
    if owsa is None:
        return

    st.markdown("#### 2. 목적함수 가중치 민감도")
    st.caption(
        "calc_error의 3개 가중치(w_fit, w_resp, w_rech)를 ±50% 변동시켜 "
        "재최적화 후 함양율 변화를 측정합니다. 가중치 선택의 주관성을 "
        "정량적으로 평가합니다 (Saltelli et al., 2004)."
    )

    ow1, ow2 = st.columns(2)
    ow1.metric("기준 함양율", f"{owsa.baseline_recharge:.1f}%")
    # Combine all three weight sweeps to find max deviation
    _all_rech = ([r for _, r in owsa.w_fit_sweep]
                 + [r for _, r in owsa.w_resp_sweep]
                 + [r for _, r in owsa.w_rech_sweep])
    rech_range = max(
        abs(max(_all_rech) - owsa.baseline_recharge),
        abs(min(_all_rech) - owsa.baseline_recharge),
    ) if _all_rech else 0.0
    ow2.metric("최대 변동폭", f"±{rech_range:.1f}%p")

    # ── Tornado chart (or "insensitive" notice) ──
    base_ow = owsa.baseline_recharge
    _max_spread = max(abs(r_hi - r_lo)
                      for _, r_lo, r_hi, _, _ in owsa.tornado_data) \
                  if owsa.tornado_data else 0.0

    if _max_spread < 0.05:
        # All bars would be invisible → show explicit result instead
        st.success(
            f"✅ **가중치 비민감(Weight-Insensitive):** "
            f"w_fit, w_resp, w_rech를 각각 ±50% 변동해도 "
            f"함양율이 {base_ow:.1f}%에서 변하지 않았습니다.\n\n"
            f"이는 최적 파라미터가 fitting·강우응답·함양범위 기준을 "
            f"**동시에 만족**하여 가중치 선택의 주관성이 결과에 "
            f"영향을 미치지 않음을 의미합니다 (Saltelli et al., 2004)."
        )
        # Compact table showing the sweep extremes for transparency
        _rows = []
        for name, r_lo, r_hi, w_lo, w_hi in owsa.tornado_data:
            _rows.append({
                "가중치": name,
                "탐색 범위": f"{w_lo:.3f} – {w_hi:.3f}",
                "함양율 (low)": f"{r_lo:.2f}%",
                "함양율 (high)": f"{r_hi:.2f}%",
                "변동폭": f"{abs(r_hi - r_lo):.2f}%p",
            })
        if _rows:
            import pandas as _pd
            st.dataframe(_pd.DataFrame(_rows), hide_index=True,
                         use_container_width=True)
    else:
        fig_ow = go.Figure()
        for name, r_lo, r_hi, _, _ in owsa.tornado_data:
            fig_ow.add_trace(go.Bar(
                y=[name], x=[r_lo - base_ow], base=[base_ow],
                orientation="h", marker_color="teal",
                showlegend=False,
                text=[f"{r_lo:.1f}%"], textposition="inside",
            ))
            fig_ow.add_trace(go.Bar(
                y=[name], x=[r_hi - base_ow], base=[base_ow],
                orientation="h", marker_color="salmon",
                showlegend=False,
                text=[f"{r_hi:.1f}%"], textposition="inside",
            ))
        fig_ow.add_vline(x=base_ow, line_dash="dash", line_color="black",
                         annotation_text=f"기준: {base_ow:.1f}%")
        fig_ow.update_layout(
            title="목적함수 가중치 토네이도 (함양율 %)",
            xaxis_title="함양율 (%)",
            barmode="overlay",
            height=250,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_ow, use_container_width=True)

    # ── Diagnostic: objective component decomposition at baseline ──
    _nrmse = getattr(owsa, 'baseline_nrmse', 0.0)
    _resp = getattr(owsa, 'baseline_resp_mismatch', 0.0)
    _rechv = getattr(owsa, 'baseline_rech_violation', 0.0)
    with st.expander("🔍 목적함수 성분 진단 (baseline)", expanded=False):
        d1, d2, d3 = st.columns(3)
        d1.metric("NRMSE", f"{_nrmse:.4f}",
                  help="최적 파라미터에서의 정규화 RMSE")
        d2.metric("Rain-Response 불일치", f"{_resp:.4f}",
                  help="관측-모델 강우응답 차이 (0=일치)")
        d3.metric("함양범위 위반", f"{_rechv:.4f}",
                  help="문헌 함양범위 이탈 정도 (0=범위 내)")
        if _resp < 0.01 and _rechv < 0.01:
            st.info(
                "ℹ️ resp_mismatch ≈ 0, rech_violation ≈ 0 → 이 두 penalty가 "
                "최적점에서 이미 비활성이므로 가중치를 변경해도 최적 파라미터가 변하지 않습니다. "
                "이는 **모델이 모든 목적함수 기준을 동시에 만족**한다는 의미입니다."
            )
        elif _resp < 0.01:
            st.info(
                "ℹ️ resp_mismatch ≈ 0 → 강우응답 일치가 이미 양호하여 "
                "w_resp 가중치 변동이 최적해에 영향을 주지 않습니다."
            )
        elif _rechv < 0.01:
            st.info(
                "ℹ️ rech_violation ≈ 0 → 함양율이 문헌 범위 내에 있어 "
                "w_rech 가중치 변동이 최적해에 영향을 주지 않습니다."
            )

    # Elasticity summary
    st.caption(
        f"탄성도: w_fit={owsa.sensitivity_w_fit:.3f}  "
        f"w_resp={owsa.sensitivity_w_resp:.3f}  "
        f"w_rech={owsa.sensitivity_w_rech:.3f}"
    )


def _display_topsis_sensitivity(topsis_delta):
    tsa = st.session_state.get("topsis_sensitivity")
    if tsa is None:
        return

    st.markdown("#### 3. TOPSIS 가중치 민감도")

    st1, st2 = st.columns(2)
    st1.metric("추천 토양 안정성", f"{tsa.stability_ratio*100:.0f}%",
               help="가중치 변동 시 동일 추천 토양 유지 비율")
    st2.metric("기준 추천 토양", tsa.original_best_soil)

    if tsa.stability_ratio >= 0.9:
        st.success("토양 추천이 가중치 변동에 매우 안정적입니다 (≥90%).")
    elif tsa.stability_ratio >= 0.7:
        st.warning("토양 추천이 일부 가중치 조합에서 변동합니다.")
    else:
        st.error("토양 추천이 가중치에 민감합니다. 결과 해석 시 주의 필요.")

    pert_data = []
    for name, w_lo, w_hi, soil_lo, soil_hi in tsa.perturbation_results:
        changed = "⚠️" if (soil_lo != tsa.original_best_soil or
                            soil_hi != tsa.original_best_soil) else "✅"
        pert_data.append({
            "기준": name,
            f"가중치 -{topsis_delta*100:.0f}%": f"{w_lo:.3f}",
            f"추천 토양 (↓)": soil_lo,
            f"가중치 +{topsis_delta*100:.0f}%": f"{w_hi:.3f}",
            f"추천 토양 (↑)": soil_hi,
            "안정": changed,
        })
    st.dataframe(pd.DataFrame(pert_data), use_container_width=True, hide_index=True)

    st.markdown("##### TOPSIS 점수 토네이도")
    fig_tt = go.Figure()
    for name, t_lo, t_hi in tsa.tornado_topsis:
        fig_tt.add_trace(go.Bar(
            y=[name], x=[t_hi - t_lo], base=[t_lo],
            orientation="h", marker_color="mediumpurple",
            text=[f"{t_lo:.1f}~{t_hi:.1f}"], textposition="inside",
            showlegend=False,
        ))
    fig_tt.update_layout(
        title="TOPSIS 점수 변동 범위 (추천 토양)",
        xaxis_title="TOPSIS 점수",
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_tt, use_container_width=True)
