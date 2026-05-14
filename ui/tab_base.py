"""Tab 1 — 기본 분석 결과 (v27)."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from ui import (
    C, LAYOUT_BASE, SOIL_NAMES, TabContext,
    build_hybrid_radar, shade_pump_plotly, has_v27_error,
)
from scoring import score_dataframe
from bma import compute_bma, bma_summary_table


def render(tab, ctx: TabContext):
    """Render Tab 1 inside the given Streamlit tab container."""
    with tab:
        result = st.session_state.get("result_v27")
        if result is None:
            st.info("사이드바 → '1단계: 기본 분석' 을 실행하세요.")
            return

        st.markdown("### 📊 분석 결과 (v27)")

        pump_idx_now = float(result.get("pump_contam_idx", 0))
        sy_eff = float(result.get("Sy_eff", 0))

        # 함양율: v27 원본과 전처리 보정값 분리 표시
        rr_v27 = float(result["recharge_ratio"])
        rr_corr = result.get("recharge_ratio_corrected")
        has_pump_corr = rr_corr is not None

        # ── Row 1: Primary metrics ──
        if has_pump_corr:
            cA, cB, cC, cC2, cD = st.columns(5)
        else:
            cA, cB, cC, cD = st.columns(4)
        cA.metric("RMSE", f"{float(result['rmse']):.4f} m")
        cB.metric("CC", f"{float(result['cc']):.4f}")
        cC.metric("Recharge (v27 WTF)", f"{rr_v27:.2f}%")
        if has_pump_corr:
            delta_rr = float(rr_corr) - rr_v27
            cC2.metric("Recharge (보정후)", f"{float(rr_corr):.2f}%",
                        delta=f"{delta_rr:+.2f}%p")
        cD.metric("Pump Contam. Idx", f"{pump_idx_now:.2f}")

        # ── Row 2: Standard hydrological metrics (Moriasi et al. 2007) ──
        nse_val = result.get("nse")
        kge_val = result.get("kge")
        pbias_val = result.get("pbias")
        if nse_val is not None:
            cE, cF, cG, cH = st.columns(4)
            nse_f = float(nse_val)
            nse_color = "🟢" if nse_f >= 0.75 else ("🟡" if nse_f >= 0.50 else "🔴")
            cE.metric(f"NSE {nse_color}", f"{nse_f:.3f}")
            kge_f = float(kge_val) if kge_val is not None else 0.0
            kge_color = "🟢" if kge_f >= 0.75 else ("🟡" if kge_f >= 0.50 else "🔴")
            cF.metric(f"KGE {kge_color}", f"{kge_f:.3f}")
            pb_f = float(pbias_val) if pbias_val is not None else 0.0
            pb_color = "🟢" if abs(pb_f) < 10 else ("🟡" if abs(pb_f) < 25 else "🔴")
            cG.metric(f"PBIAS {pb_color}", f"{pb_f:+.1f}%")
            cH.metric("Sy (유효)", f"{sy_eff:.4f}")

        if has_pump_corr:
            st.caption("ℹ️ **v27 WTF**: 원본 수위 | **보정후**: 펌핑 제거 수위로 동일 v27 재분석")

        if pump_idx_now >= 0.45:
            st.error("🚫 펌핑 오염도 높음 → ③ 펌핑 전처리를 권장합니다.")
        elif pump_idx_now >= 0.25:
            st.warning("⚠️ 펌핑 의심 → ③ 펌핑 전처리를 시도해 보세요.")

        # 경계 도달 경고 표시
        bw = result.get("boundary_warnings", [])
        if bw:
            for w_msg in bw:
                st.warning(f"⚠️ 최적화 경고: {w_msg}")
        else:
            st.success("✅ 펌핑 오염도 낮음")

        # ── 토양 추천 결과 (② 스캔 실행 후) ──
        _render_scan_results()

        # 메인 차트
        _render_main_chart(result, ctx, rr_v27)

        # ── Bayesian Sy posterior (Phase 1) ──
        _render_bayesian_sy(result, ctx)

        # ── 유역 분석용 저장 (Tab 10 에서 로딩) ──
        _render_save_for_watershed(result, ctx, has_pump_corr)

        # ── AI 수문학적 소견 ──
        _render_ai_opinion(result, ctx, rr_v27, rr_corr, has_pump_corr, sy_eff)


def _render_scan_results():
    """Show scan / BMA results if available."""
    scan_df = st.session_state.get("scan_data")
    if scan_df is not None:
        with st.expander("🛡️ 토양 추천 결과 (Hybrid Scan)", expanded=True):
            best_row = scan_df.iloc[0]
            conf = st.session_state.get("best_soil_conf", "MEDIUM")
            tentative = st.session_state.get("best_soil_tentative", False)

            col_a, col_b = st.columns([1, 2])
            with col_a:
                label = best_row["Soil"] + (" (tentative)" if tentative else "")
                st.success(f"🏆 최적 토양: **{label}**")
                st.metric("TOPSIS Score", f"{best_row.get('TopsisScore', best_row['HybridScore']):.1f}")
                st.write(f"신뢰도: **{conf}**")
                st.write(f"펌핑지수: `{best_row.get('PumpIdx', 0):.2f}`")
                st.caption("※ 함양율은 상단 메트릭 참조 (v27 WTF / 보정후 분리 표시)")

            with col_b:
                show_cols = ["Soil", "TopsisScore", "StressScore", "SyScore",
                             "SlopeErr", "PumpIdx", "EvalN", "RecoFlag"]
                available_cols = [c for c in show_cols if c in scan_df.columns]
                disp = scan_df[available_cols].head(5).copy()
                col_map = {
                    "Soil": "토양", "TopsisScore": "TOPSIS", "StressScore": "k적합도",
                    "SyScore": "Sy적합도", "SlopeErr": "감수편차",
                    "PumpIdx": "펌핑지수", "EvalN": "유효샘플", "RecoFlag": "판정",
                }
                disp.columns = [col_map.get(c, c) for c in available_cols]

                def _hl_topsis(row):
                    v = row.get("TOPSIS", 0)
                    bg = "#D1FAE5" if v >= 70 else ("#FEF3C7" if v >= 45 else "#FEE2E2")
                    return [f"background-color:{bg}" if c == "TOPSIS" else "" for c in row.index]

                def _hl_pump(row):
                    v = row.get("펌핑지수", 0)
                    bg = "#FEE2E2" if v >= 0.45 else ("#FEF3C7" if v >= 0.25 else "")
                    return [f"background-color:{bg}" if c == "펌핑지수" else "" for c in row.index]

                styled = (disp.style
                          .apply(_hl_topsis, axis=1)
                          .apply(_hl_pump, axis=1)
                          .format({"TOPSIS": "{:.1f}", "k적합도": "{:.1f}",
                                   "Sy적합도": "{:.1f}", "감수편차": "{:.4f}",
                                   "펌핑지수": "{:.2f}"}))
                st.dataframe(styled, hide_index=True, use_container_width=True)

            st.plotly_chart(build_hybrid_radar(scan_df),
                            use_container_width=True, theme=None)

            # ── BMA (Bayesian Model Averaging) 토양 사후확률 ──
            bma_res = st.session_state.get("bma_result")
            if bma_res is not None:
                st.markdown("---")
                st.markdown("#### 📊 토양 사후확률 — Bayesian Model Averaging")
                bma_c1, bma_c2 = st.columns([1, 2])
                with bma_c1:
                    st.metric("추천 토양 확률", f"{bma_res.dominant_prob * 100:.1f}%")
                    st.metric("유효 모델 수", f"{bma_res.n_effective_models:.1f} / 12")
                    st.write(f"확신도: **{bma_res.confidence_label}**")
                    if bma_res.n_effective_models >= 4:
                        st.info("💡 유효 모델 수가 높음 → 토양 특정이 어려운 데이터")
                with bma_c2:
                    bma_df = bma_summary_table(bma_res)
                    bma_top = bma_df.head(6)
                    fig_bma = go.Figure(go.Bar(
                        x=bma_top["사후확률(%)"].values,
                        y=bma_top["토양"].values,
                        orientation="h",
                        marker_color=[
                            "#10B981" if v >= 30 else ("#F59E0B" if v >= 15 else "#9CA3AF")
                            for v in bma_top["사후확률(%)"].values
                        ],
                        text=[f"{v:.1f}%" for v in bma_top["사후확률(%)"].values],
                        textposition="outside",
                    ))
                    fig_bma.update_layout(
                        title="토양별 사후확률 P(Mk|D)",
                        xaxis_title="사후확률 (%)",
                        yaxis=dict(autorange="reversed"),
                        height=280, margin=dict(l=10, r=10, t=40, b=30),
                    )
                    st.plotly_chart(fig_bma, use_container_width=True, theme=None)
                st.caption(
                    "BMA는 각 토양이 관측 데이터를 얼마나 잘 설명하는지를 "
                    "확률로 산출합니다 (Hoeting et al. 1999). "
                    "함양율은 추천된 토양의 시뮬레이션 결과를 사용합니다."
                )
    else:
        st.info(
            "ℹ️ 사이드바 → ② Hybrid 토양 정밀 진단을 실행하면 토양 추천이 표시됩니다.\n\n"
            "💡 **파라미터 민감도 분석**: Auto-Optimize를 끄고 k, z 슬라이더를 조정하면 "
            "함양율·그래프 변화를 직접 관찰할 수 있습니다."
        )


def _render_main_chart(result, ctx: TabContext, rr_v27: float):
    """Water level time-series chart."""
    days = np.arange(len(result["ho"]))
    ho = np.array(result["ho"], dtype=float)
    hs_kf = np.array(result["hs_kf"], dtype=float)
    po = np.array(result.get("po_shifted", result["po"]), dtype=float)
    pump_mask_v27 = np.array(result.get("pump_mask", [0] * len(ho))).astype(bool)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if pump_mask_v27.any():
        shade_pump_plotly(fig, days, pump_mask_v27)

    rain_mm = po * 1000
    fig.add_trace(go.Bar(x=days, y=rain_mm, name="Rain (mm)",
        marker=dict(color=C["rain"], opacity=0.35, line=dict(width=0))),
        secondary_y=True)
    fig.add_trace(go.Scatter(x=days, y=ho, mode="markers", name="Observed",
        marker=dict(color=C["observed"], size=4, opacity=0.55)), secondary_y=False)
    fig.add_trace(go.Scatter(x=days, y=hs_kf, mode="lines", name="Kalman",
        line=dict(color=C["kalman"], width=2.5)), secondary_y=False)

    if ctx.show_pure and "hs_pure" in result:
        hs_pure = np.array(result["hs_pure"], dtype=float)
        fig.add_trace(go.Scatter(x=days, y=hs_pure, mode="lines", name="Pure WTF",
            line=dict(color="#F59E0B", width=2.0, dash="dash")), secondary_y=False)

    opt_k = float(result.get("opt_k", ctx.k_val))
    opt_z = float(result.get("opt_z", ctx.z_val))
    opt_lag = int(result.get("opt_lag", ctx.lag_val))
    opt_rho_v = float(result.get("opt_rho", 0.85))
    opt_alpha_v = float(result.get("opt_alpha", 0.4))
    mode_label = "Auto-Opt" if ctx.auto_optimize else "Manual"
    soil_label = (SOIL_NAMES[int(ctx.sn_idx) - 1].split(". ")[1]
                  if ". " in SOIL_NAMES[int(ctx.sn_idx) - 1]
                  else SOIL_NAMES[int(ctx.sn_idx) - 1])

    _layout_fig = {**LAYOUT_BASE, "margin": dict(l=60, r=30, t=80, b=100)}
    fig.update_layout(
        **_layout_fig, height=520,
        title=(f"<b>Groundwater Level — v27 [{mode_label}]</b>"
               f"<br><span style='font-size:12px;color:#555'>"
               f"k={opt_k:.4f}  z={opt_z:.1f}m  lag={opt_lag}d  "
               f"ρ={opt_rho_v:.2f}  α={opt_alpha_v:.2f}  "
               f"Soil={soil_label}  Rech={rr_v27:.2f}%</span>"),
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center",
                    bgcolor="rgba(255,255,255,0.9)", bordercolor="#CCC", borderwidth=1),
    )
    fig.update_xaxes(title="Time (days)", gridcolor=C["grid"],
                     tickfont=dict(size=11), nticks=12)
    fig.update_yaxes(title="GW Level (m)", secondary_y=False, gridcolor=C["grid"])
    rain_max = float(np.nanmax(rain_mm)) if len(rain_mm) > 0 else 10
    fig.update_yaxes(title="Rain (mm)", range=[rain_max * 3.5, 0], secondary_y=True)
    st.plotly_chart(fig, use_container_width=True, theme=None)


def _render_bayesian_sy(result, ctx: TabContext):
    """Sy + 함양율의 Bayesian posterior (Phase 1).

    HSG 와 대수층 타입 prior + 관측 Sy_eff likelihood + (옵션) 양수시험 데이터.
    """
    st.markdown("---")
    st.markdown("#### 🎲 Bayesian Sy / 함양율 후행분포 (Phase 1)")
    st.caption(
        "표면 토양 HSG 와 대수층 타입을 prior 로 두고, "
        "WTF 추정 Sy 를 likelihood 로 결합한 Bayesian 후행분포. "
        "양수시험 Sy 값이 있으면 strong likelihood 로 추가 가능."
    )

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        hsg_choice = st.selectbox(
            "표면 HSG",
            options=["A", "B", "C", "D"],
            index=0,
            key="bayes_hsg",
            help="표면 토양 Hydrologic Soil Group (정밀토양도에서 자동 매핑하거나 수동 선택)",
        )
    with col2:
        aq_choice = st.selectbox(
            "대수층 타입",
            options=["alluvial", "bedrock"],
            index=0,
            key="bayes_aq",
            help="alluvial=충적층, bedrock=암반/풍화대",
        )
    with col3:
        pump_sy_input = st.text_input(
            "양수시험 Sy (선택)",
            value="",
            key="bayes_pump_sy",
            help="실측 양수시험으로 산출된 Sy (예: 0.18). 입력 시 strong likelihood 추가.",
        )
        try:
            pump_sy = float(pump_sy_input) if pump_sy_input.strip() else None
            if pump_sy is not None and not (0.001 < pump_sy < 0.5):
                st.error("Sy 는 0.001~0.5 범위여야 합니다.")
                pump_sy = None
        except ValueError:
            pump_sy = None
            if pump_sy_input.strip():
                st.error("숫자로 입력하세요.")

    if not st.button("🎲 Bayesian 추정 실행", key="run_bayes"):
        st.info("위 버튼을 눌러 Sy + 함양율의 후행분포를 계산하세요.")
        return

    try:
        from bayes_sy import from_result_v27 as bayes_from_v27
        with st.spinner("Importance sampling 중..."):
            br = bayes_from_v27(
                result, hsg=hsg_choice, aquifer=aq_choice,
                pump_test_sy=pump_sy, n_samples=10000,
            )

        # 결과 표시
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Sy posterior**")
            st.metric(
                "Sy (posterior 평균)",
                f"{br.sy_post_mean:.3f}",
                delta=f"{br.sy_post_mean - br.sy_prior_mean:+.3f} vs prior",
            )
            st.caption(
                f"95% CI: [{br.sy_post_lo95:.3f}, {br.sy_post_hi95:.3f}]  "
                f"·  prior μ={br.sy_prior_mean:.3f} ± σ={br.sy_prior_sd:.3f}  "
                f"·  sn={br.sn_used}"
            )
        with col_b:
            st.markdown("**함양율 posterior**")
            if np.isfinite(br.rech_pct_post_mean):
                st.metric(
                    "Recharge (posterior 평균)",
                    f"{br.rech_pct_post_mean:.2f} %",
                )
                st.caption(
                    f"95% CI: [{br.rech_pct_post_lo95:.2f}, {br.rech_pct_post_hi95:.2f}] %  "
                    f"·  σ={br.rech_pct_post_sd:.2f}"
                )
            else:
                st.caption("입력 데이터로 함양율 분포 계산 불가 (cum_rise 또는 P 부족)")

        # 진단
        st.caption(
            f"📊 Effective Sample Size: **{br.n_eff:.0f}** / {br.n_samples} "
            f"({'✅ 수렴' if br.converged else '⚠️ ESS<100, 결과 신뢰성 낮음'})"
        )

        # 시각화 (간단한 히스토그램)
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.rcParams['font.family'] = ['AppleGothic', 'sans-serif']
            from scipy.stats import truncnorm

            fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
            # Prior PDF
            x = np.linspace(0.01, 0.45, 200)
            a = (0.01 - br.sy_prior_mean) / br.sy_prior_sd
            b = (0.45 - br.sy_prior_mean) / br.sy_prior_sd
            prior_pdf = truncnorm.pdf(x, a, b, loc=br.sy_prior_mean, scale=br.sy_prior_sd)

            ax = axes[0]
            ax.plot(x, prior_pdf, label="Prior", color="#6c757d", linestyle="--")
            ax.axvline(br.sy_post_mean, color="#dc2626", linewidth=2, label=f"Posterior μ={br.sy_post_mean:.3f}")
            ax.axvspan(br.sy_post_lo95, br.sy_post_hi95, color="#dc2626", alpha=0.15, label="95% CI")
            ax.set_xlabel("Sy"); ax.set_ylabel("density")
            ax.set_title("Sy posterior vs prior")
            ax.legend(fontsize=8); ax.grid(alpha=0.3)

            ax2 = axes[1]
            if np.isfinite(br.rech_pct_post_mean):
                ax2.axvline(br.rech_pct_post_mean, color="#0891b2", linewidth=2,
                           label=f"Posterior μ={br.rech_pct_post_mean:.1f}%")
                ax2.axvspan(br.rech_pct_post_lo95, br.rech_pct_post_hi95,
                            color="#0891b2", alpha=0.15, label="95% CI")
                ax2.set_xlabel("Recharge (% of P)"); ax2.set_ylabel("")
                ax2.set_title("함양율 posterior")
                ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
            else:
                ax2.text(0.5, 0.5, "함양율 분포 N/A",
                         transform=ax2.transAxes, ha="center", va="center")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception as e:
            st.caption(f"히스토그램 렌더 실패: {e}")

        # 세션에 저장 (이후 watershed 분석에서 사용 가능)
        st.session_state["bayes_sy_result"] = br

    except Exception as e:
        st.error(f"Bayesian 분석 실패: {e}")
        import traceback
        st.code(traceback.format_exc())


def _render_soil_source_picker(ctx: TabContext) -> int | None:
    """저장 시 토양 인덱스 출처를 사용자가 고를 수 있게 한다.

    세 가지 후보:
      1) 슬라이더 값 (ctx.sn_idx)        — 사용자 수동 설정
      2) 추천 (best_soil)                — 토양 스캔 알고리즘 점수 1위
      3) 토양도 (HSG + aquifer → sn_idx) — shp_soil_mapper + HSG_AQUIFER_TO_SN

    값이 누락된 옵션은 라벨에 "(미가용)" 표시되고 disabled 처리.
    선택값(int 1–12) 반환. 모두 미가용이면 None → 호출부에서 ctx.sn_idx fallback.
    """
    st.markdown("##### 🌱 저장할 토양 선택")
    st.caption("세 출처를 비교하고 어느 값을 유역 분석에 쓸지 고르세요. "
                "기본값은 가장 신뢰도 높은 출처(추천 → 토양도 → 슬라이더 순).")

    # 후보 1: 슬라이더 값
    slider_sn = int(ctx.sn_idx)
    slider_label = SOIL_NAMES[slider_sn - 1] if 1 <= slider_sn <= 12 else "?"

    # 후보 2: 추천 (스캔 결과)
    rec_sn = st.session_state.get("best_soil")
    rec_conf = st.session_state.get("best_soil_conf", "")
    rec_tent = st.session_state.get("best_soil_tentative", False)
    rec_label = (SOIL_NAMES[int(rec_sn) - 1]
                  if rec_sn and 1 <= int(rec_sn) <= 12 else None)

    # 후보 3: 토양도 (HSG + aquifer)
    shp_sn = None
    shp_label = None
    shp_hsg = None
    try:
        choice_well = st.session_state.get("save_well_choice", "")
        from wells_registry import WELLS
        from shp_soil_mapper import query_point
        from bayes_sy import HSG_AQUIFER_TO_SN
        if choice_well and choice_well in WELLS:
            info = WELLS[choice_well]
            sq = query_point(choice_well, info.lat, info.lon)
            shp_hsg = sq.hydro_type
            shp_sn = HSG_AQUIFER_TO_SN.get((shp_hsg, info.aquifer))
            if shp_sn:
                shp_label = SOIL_NAMES[shp_sn - 1]
    except Exception:
        pass

    # 옵션 라벨 구성
    options: list[tuple[str, int | None]] = []
    options.append((
        f"📊 슬라이더: #{slider_sn} {slider_label}",
        slider_sn,
    ))
    if rec_sn:
        marker = "🥇" if rec_conf == "HIGH" else "🥈"
        tent_note = " (잠정)" if rec_tent else ""
        options.append((
            f"{marker} 추천 (스캔): #{rec_sn} {rec_label}  "
            f"[{rec_conf}{tent_note}]",
            int(rec_sn),
        ))
    else:
        options.append(("🥇 추천: (스캔 미실행)", None))
    if shp_sn:
        options.append((
            f"🗺️ 토양도: HSG-{shp_hsg} → #{shp_sn} {shp_label}",
            int(shp_sn),
        ))
    else:
        options.append(("🗺️ 토양도: (좌표/.shp 미가용)", None))

    # 디폴트 우선순위: 추천 → 토양도 → 슬라이더
    default_idx = 0
    if rec_sn:
        default_idx = 1
    elif shp_sn:
        default_idx = 2

    # 사용 가능 옵션만 노출
    labels = [o[0] for o in options]
    pick = st.radio("토양 출처", options=labels, index=default_idx,
                      key="save_soil_source", horizontal=False)
    sel = options[labels.index(pick)][1]
    if sel is None:
        st.warning("선택한 출처는 사용 불가합니다. 슬라이더 값으로 fallback.")
        sel = slider_sn
    if sel != slider_sn:
        st.info(f"💡 슬라이더(#{slider_sn})와 다른 값(#{sel})을 저장합니다.")
    return int(sel)


def _render_save_for_watershed(result, ctx: TabContext, has_pump_corr: bool):
    """현재 분석 결과를 well_results/{well_name}.json 에 저장.

    Tab 10 (유역 함양율) 의 'Cached' 모드가 이 파일들을 읽어 면적 가중 집계.
    """
    import os
    st.markdown("---")
    st.markdown("#### 💾 유역 분석용 저장")
    st.caption(
        "이 결과를 저장하면 **Tab 10 (유역 함양율)** 에서 재계산 없이 사용합니다. "
        "관정별로 정성껏 튜닝한 sn_idx · k · 펌핑보정이 그대로 반영됩니다."
    )

    try:
        from wells_registry import WELLS
        registered_wells = list(WELLS.keys())
    except Exception:
        registered_wells = []

    # 파일명 추정 (업로드된 파일명 → 관정명 후보)
    upl = st.session_state.get("uploaded_name", "") or ""
    base = os.path.splitext(upl)[0] if upl else ""
    default_name = base if base in registered_wells else (
        registered_wells[0] if registered_wells else base or "관정1"
    )

    # ── 저장할 토양 선택 (3개 출처 비교) ──────────────────
    save_sn_idx = _render_soil_source_picker(ctx)

    col1, col2 = st.columns([2, 1])
    with col1:
        if registered_wells:
            options = registered_wells + ["(새 관정 등록)"]
            try:
                idx = options.index(default_name)
            except ValueError:
                idx = 0
            choice = st.selectbox(
                "관정명 (wells_registry 등록명)",
                options=options, index=idx,
                key="save_well_choice",
                help="목록에 없는 관정은 '(새 관정 등록)' 선택 → 좌표 입력 후 자동 등록",
            )
        else:
            choice = "(새 관정 등록)"
            st.info("등록된 관정이 없습니다. 새 관정으로 등록합니다.")

        # 새 관정 등록 시 좌표 입력 폼
        is_new = (choice == "(새 관정 등록)")
        if is_new:
            from ui import coord_input
            new_name = st.text_input("새 관정명", value=default_name, key="new_well_name")
            new_lat, new_lon = coord_input(
                default_lat=36.35, default_lon=127.37,
                key_prefix="new_well",
            )
            sub2, sub3 = st.columns(2)
            with sub2:
                new_ws = st.text_input("유역명", placeholder="예: 갑천", key="new_well_ws")
                new_aq = st.selectbox(
                    "대수층", options=["alluvial", "bedrock"], index=1,
                    key="new_well_aq",
                )
            with sub3:
                new_kma = st.number_input(
                    "ASOS ID", min_value=100, max_value=300,
                    value=133, step=1, key="new_well_kma",
                    help="대전=133, 추풍령=135, 대구=143",
                )
            well_name = new_name
        else:
            well_name = choice
            # 기존 관정 선택 시 → 삭제 옵션 노출
            with st.expander(f"🗑️ '{well_name}' 삭제", expanded=False):
                st.caption(
                    "이 관정을 레지스트리(wells_registry) + 분석 결과 파일에서 "
                    "동시에 제거합니다. 유역 함양율 탭에서도 즉시 사라집니다."
                )
                confirm_del = st.text_input(
                    f"확인을 위해 관정명 '{well_name}'을(를) 그대로 입력",
                    key=f"tab1_del_confirm_{well_name}",
                )
                if st.button("🗑️ 삭제 실행",
                              type="primary",
                              key=f"tab1_del_btn_{well_name}"):
                    if confirm_del.strip() != well_name:
                        st.error("관정명이 일치하지 않습니다.")
                    else:
                        try:
                            import wells_registry as wr
                            from well_results_store import delete as _wrs_del
                        except Exception as e:
                            st.error(f"모듈 로드 실패: {e}")
                            return
                        n_reg = 1 if wr.remove_well(well_name) else 0
                        try:
                            n_store = 1 if _wrs_del(well_name) else 0
                        except Exception:
                            n_store = 0
                        if n_reg or n_store:
                            st.success(
                                f"🗑️ 삭제됨: {well_name}  "
                                f"(registry={n_reg}, 결과 파일={n_store})"
                            )
                            st.rerun()
                        else:
                            st.warning(
                                f"⚠️ {well_name} 가 어디에도 없습니다."
                            )
    with col2:
        if st.button("💾 저장", type="primary", use_container_width=True):
            try:
                from well_results_store import (
                    from_result_v27, save, DEFAULT_DIR,
                )
                import wells_registry as wr

                # 새 관정이면 먼저 등록
                if is_new:
                    if not well_name.strip():
                        st.error("관정명을 입력하세요.")
                        return
                    if not new_ws.strip():
                        st.error("유역명을 입력하세요.")
                        return
                    try:
                        wr.add_well(
                            well_name.strip(), new_lat, new_lon, new_ws.strip(),
                            aquifer=new_aq, nearest_kma=int(new_kma),
                            overwrite=True,
                        )
                    except Exception as e:
                        st.error(f"관정 등록 실패: {e}")
                        return

                # wells_registry 정보 자동 추가 (새로 등록한 경우 포함)
                aquifer = hydro_type = soil_code = None
                lat = lon = None
                if well_name in wr.WELLS:
                    info = wr.WELLS[well_name]
                    aquifer = info.aquifer
                    lat, lon = info.lat, info.lon
                    try:
                        from shp_soil_mapper import query_point
                        sq = query_point(well_name, info.lat, info.lon)
                        hydro_type = sq.hydro_type
                        soil_code = sq.soil_code
                    except Exception:
                        pass

                # 사용자가 위에서 고른 토양 인덱스를 우선 사용 (없으면 ctx 슬라이더값)
                sn_to_save = save_sn_idx if save_sn_idx is not None else int(ctx.sn_idx)
                stored = from_result_v27(
                    well_name=well_name,
                    result_v27=result,
                    file_path=st.session_state.get("uploaded_tmp_path", ""),
                    sn_idx=int(sn_to_save),
                    soil_name=SOIL_NAMES[int(sn_to_save) - 1] if 1 <= int(sn_to_save) <= 12 else None,
                    pump_corrected=has_pump_corr,
                    aquifer=aquifer,
                    hydro_type=hydro_type,
                    soil_code=soil_code,
                    lat=lat, lon=lon,
                )
                # Bayesian 결과가 세션에 있으면 같이 저장
                br = st.session_state.get("bayes_sy_result")
                if br is not None:
                    stored.bayes_sy_post_mean = br.sy_post_mean
                    stored.bayes_sy_post_sd = br.sy_post_sd
                    stored.bayes_sy_post_lo95 = br.sy_post_lo95
                    stored.bayes_sy_post_hi95 = br.sy_post_hi95
                    stored.bayes_rech_pct_post_mean = (
                        br.rech_pct_post_mean
                        if np.isfinite(br.rech_pct_post_mean) else None
                    )
                    stored.bayes_rech_pct_post_lo95 = (
                        br.rech_pct_post_lo95
                        if np.isfinite(br.rech_pct_post_lo95) else None
                    )
                    stored.bayes_rech_pct_post_hi95 = (
                        br.rech_pct_post_hi95
                        if np.isfinite(br.rech_pct_post_hi95) else None
                    )
                    stored.bayes_n_eff = br.n_eff
                    stored.pump_test_sy = br.pump_test_sy
                path = save(stored)
                if is_new:
                    st.success(f"✅ 신규 관정 등록 + 저장 완료: `{path}`")
                else:
                    st.success(f"✅ 저장됨: `{path}`")
                st.caption(
                    f"함양율 {stored.recharge_ratio_pct:.2f}% · "
                    f"sn={stored.sn_idx} ({stored.soil_name}) · "
                    f"P={stored.P_annual_mm:.0f} mm/yr"
                )
            except Exception as e:
                st.error(f"저장 실패: {e}")

    # 현재 저장된 목록 (간단히)
    try:
        from well_results_store import list_stored
        saved = list_stored()
        if saved:
            with st.expander(f"📂 현재 저장된 관정 ({len(saved)}개)", expanded=False):
                for s in saved:
                    st.write(
                        f"- **{s.well_name}** — {s.recharge_ratio_pct:.2f}% "
                        f"(sn={s.sn_idx}, {s.analyzed_at})"
                    )
    except Exception:
        pass


def _render_ai_opinion(result, ctx: TabContext, rr_v27, rr_corr, has_pump_corr, sy_eff):
    """AI hydrogeological opinion section."""
    st.markdown("---")
    if st.button("🧠 AI 수문학적 소견 요청"):
        if not ctx.api_key:
            st.warning("⚠️ 사이드바에 OpenAI API Key를 입력하세요.")
        else:
            with st.spinner("AI 소견서 작성 중..."):
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=ctx.api_key)
                    soil_name = SOIL_NAMES[int(ctx.sn_idx) - 1]
                    pr_now = st.session_state.get("pump_result")
                    pump_txt = ""
                    if pr_now:
                        pump_txt = (
                            f"\n[펌핑 전처리 결과]\n"
                            f"- 펌핑 오염 비율: {pr_now['pump_fraction']*100:.1f}%\n"
                            f"- 탐지 이벤트: {pr_now['n_events']}건\n"
                            f"- 전처리 후 RMSE: {pr_now['corrected']['rmse']:.4f} m\n"
                            f"- 전처리 후 함양율: {pr_now['corrected']['rech_rate']:.2f}%\n"
                            f"- 전처리 전 함양율: {pr_now['raw']['rech_rate']:.2f}%"
                        )
                    prompt = f"""
[분석 데이터]
- 토양: {soil_name}
- RMSE={float(result['rmse']):.4f}, CC={float(result['cc']):.4f}
- v27 WTF 함양율={rr_v27:.2f}%{f'  보정후 함양율={float(rr_corr):.2f}%' if has_pump_corr else ''}  Sy_eff={sy_eff:.4f}
- 펌핑지수={float(result.get('pump_contam_idx',0)):.2f}
- 이벤트={int(result.get('pump_event_count',0))}건  최대런={int(result.get('pump_max_run',0))}일
{pump_txt}

1. 감수 곡선 형태·기울기 근거로 토양 분류 타당성 평가
2. 산출 함양율이 해당 토양 수리 특성 이론 범위 내인지 검증
3. 관측 데이터에서 양수(Pumping)·외부 교란 징후 진단
4. 펌핑 전처리 전/후 함양율 차이의 수문학적 해석
"""
                    resp = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "당신은 수문지질학 법의학 전문가입니다."},
                            {"role": "user", "content": prompt},
                        ],
                    )
                    st.info(resp.choices[0].message.content)
                except ImportError:
                    st.error("openai 패키지가 필요합니다: pip install openai")
                except Exception as _e:
                    st.error(f"AI 오류: {_e}")
