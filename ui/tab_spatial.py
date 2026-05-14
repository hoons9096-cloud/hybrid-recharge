"""Tab 7 — 공간 함양 분석 (EnKF)."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui import C, LAYOUT_BASE, TabContext


def render(tab, ctx: TabContext):
    """Render Tab 7 inside the given Streamlit tab container."""
    with tab:
        st.markdown("### 🗺️ 공간 함양 분석 (Ensemble Kalman Filter)")
        st.caption(
            "관정 1개의 hybrid-recharge 결과를 앵커로 사용하여 "
            "주변 격자점의 함양률을 앙상블 Kalman 필터로 공간 추정합니다."
        )

        result_v27 = st.session_state.get("result_v27")
        if result_v27 is None or "error" in result_v27:
            st.info("① 기본 분석을 먼저 실행하세요. (사이드바 → ▶ ① 기본 분석)")
            return

        # ── EnKF import (지연 로딩) ──
        try:
            from enkf_spatial import SpatialEnKF, SpatialPoint, EnKFConfig
            from core_sim_v27 import load_core_data
            ENKF_AVAILABLE = True
        except ImportError as _e:
            ENKF_AVAILABLE = False
            st.error(f"enkf_spatial.py 로드 실패: {_e}")

        if not ENKF_AVAILABLE:
            return

        # ── 설정 영역 ──
        col_cfg, col_map = st.columns([1, 2])

        with col_cfg:
            n_ens, loc_radius, obs_noise, perturb_z, well_x, well_y, grid_coords, run_enkf = \
                _render_config(ctx)

        # ── EnKF 실행 ──
        if run_enkf:
            _execute_enkf(ctx, n_ens, loc_radius, obs_noise, perturb_z,
                          well_x, well_y, grid_coords)

        # ── 결과 표시 ──
        enkf_result = st.session_state.get("enkf_result")
        with col_map:
            if enkf_result is None:
                _render_preview(well_x, well_y, grid_coords, loc_radius)
            else:
                _render_results(enkf_result, loc_radius)

        # ── 추가 차트 ──
        if enkf_result is not None:
            _render_extra_charts(enkf_result)


def _render_config(ctx: TabContext):
    """Render EnKF configuration sidebar (inside tab column)."""
    st.markdown("#### ⚙️ EnKF 설정")

    n_ens = st.slider("앙상블 크기", 30, 300, 100, step=10,
                       help="클수록 정확하나 느림. 100 권장.")
    loc_radius = st.slider("국소화 반경 (km)", 1.0, 30.0, 8.0, step=0.5,
                            help="관정 관측이 격자에 영향 미치는 거리. 표준유역 반경의 60~80%.")
    obs_noise = st.slider("관측 불확실성 (mm/yr)", 5, 50, 20, step=5,
                           help="관정 함양률 추정 오차. 클수록 격자가 사전분포에 머뭄. 기본 20mm/yr.")
    perturb_z = st.slider("z_unsat 불확실성 (m)", 0.5, 3.0, 1.0, step=0.5,
                           help="불포화대 두께 섭동. 클수록 앙상블 폭 넓음.")

    st.markdown("---")
    st.markdown("#### 📍 관정 위치")
    well_x = st.number_input("관정 X (km)", value=0.0, step=0.1,
                              help="UTM-K 기준 km, 또는 상대 좌표")
    well_y = st.number_input("관정 Y (km)", value=0.0, step=0.1)

    st.markdown("#### 📐 격자점 설정")
    st.caption("함양률을 추정할 위치를 입력하세요 (관정 기준 상대 좌표 km)")

    n_grid = st.number_input("격자점 수", 1, 12, 4, step=1)
    grid_coords = []
    for gi in range(int(n_grid)):
        gc1, gc2 = st.columns(2)
        gx = gc1.number_input(f"G{gi+1:02d} X", value=float((gi % 3 - 1) * 3),
                               key=f"gx_{gi}", step=0.5)
        gy = gc2.number_input(f"G{gi+1:02d} Y", value=float((gi // 3 + 1) * 3),
                               key=f"gy_{gi}", step=0.5)
        grid_coords.append((gx, gy))

    run_enkf = st.button("🚀 EnKF 공간 분석 실행",
                          type="primary",
                          help="관정 최적화 후 앙상블 실행 (30초~3분)")

    return n_ens, loc_radius, obs_noise, perturb_z, well_x, well_y, grid_coords, run_enkf


def _execute_enkf(ctx, n_ens, loc_radius, obs_noise, perturb_z,
                  well_x, well_y, grid_coords):
    from enkf_spatial import SpatialEnKF, SpatialPoint, EnKFConfig
    from core_sim_v27 import load_core_data

    with st.spinner("EnKF 실행 중... (앙상블 크기에 따라 1~3분 소요)"):
        try:
            well_name = st.session_state.get(
                "uploaded_name", "관정").replace(".txt", "").replace(".csv", "")
            points = [
                SpatialPoint(x=well_x, y=well_y, name=well_name, is_well=True)
            ]
            for gi, (gx, gy) in enumerate(grid_coords):
                points.append(
                    SpatialPoint(x=gx, y=gy, name=f"G_{gi+1:02d}", is_well=False)
                )

            cfg = EnKFConfig(
                n_ensemble=n_ens,
                localization_radius=loc_radius,
                perturb_k_std=0.002,
                perturb_z_std=perturb_z,
                obs_noise_mm=float(obs_noise),
                random_seed=42,
            )

            enkf = SpatialEnKF(points, cfg)

            fpath = ctx.file_path_to_send
            cdata = load_core_data(fpath)
            ho_m = cdata.ho.ravel()
            po_m = cdata.po.ravel()

            enkf.add_well(well_name, ho_m, po_m, sn=ctx.sn_idx)

            po_raw = enkf._wells[well_name]["po"]
            for gi in range(len(grid_coords)):
                enkf.add_grid(f"G_{gi+1:02d}", po_raw)

            enkf_result = enkf.run()
            st.session_state["enkf_result"] = enkf_result
            st.success("✅ EnKF 분석 완료!")

        except Exception as _e:
            st.error(f"EnKF 실행 오류: {_e}")
            import traceback
            st.code(traceback.format_exc())


def _render_preview(well_x, well_y, grid_coords, loc_radius):
    """Show spatial point preview before running EnKF."""
    st.info("왼쪽에서 격자점을 설정하고 'EnKF 공간 분석 실행'을 누르세요.")
    st.markdown("#### 📍 설정된 공간점 미리보기")

    fig_prev = go.Figure()
    fig_prev.add_trace(go.Scatter(
        x=[well_x], y=[well_y], mode="markers+text",
        marker=dict(size=18, color="#2563EB", symbol="star"),
        text=[f"관정<br>(0, 0)"], textposition="top center",
        name="관정",
    ))
    if grid_coords:
        gxs = [c[0] for c in grid_coords]
        gys = [c[1] for c in grid_coords]
        glabels = [f"G_{i+1:02d}<br>({x:.1f},{y:.1f})"
                   for i, (x, y) in enumerate(grid_coords)]
        fig_prev.add_trace(go.Scatter(
            x=gxs, y=gys, mode="markers+text",
            marker=dict(size=12, color="#DC2626", symbol="square"),
            text=glabels, textposition="top center",
            name="격자점",
        ))
    theta_c = np.linspace(0, 2 * np.pi, 100)
    fig_prev.add_trace(go.Scatter(
        x=well_x + loc_radius * np.cos(theta_c),
        y=well_y + loc_radius * np.sin(theta_c),
        mode="lines",
        line=dict(color="green", dash="dash", width=1.5),
        name=f"국소화 반경 ({loc_radius:.1f}km)",
    ))
    fig_prev.update_layout(
        height=380,
        xaxis_title="X (km)", yaxis_title="Y (km)",
        xaxis=dict(scaleanchor="y"),
        legend=dict(x=0, y=1),
        margin=dict(l=40, r=20, t=40, b=40),
        title="공간점 배치 미리보기",
        **{k: v for k, v in LAYOUT_BASE.items() if k != "margin"},
    )
    st.plotly_chart(fig_prev, use_container_width=True)


def _render_results(enkf_result, loc_radius):
    """Display EnKF results: table, validation, spatial map."""
    st.markdown("#### 📊 공간 함양 추정 결과")

    rows = []
    for i, pt in enumerate(enkf_result.points):
        rows.append({
            "구분": "관정 ★" if pt.is_well else "격자 ◆",
            "이름": pt.name,
            "X (km)": pt.x,
            "Y (km)": pt.y,
            "함양량 (mm/yr)": f"{enkf_result.ann_rech_mm[i]:.0f}",
            "함양률 (%)": f"{enkf_result.ann_rech_pct[i]:.1f}",
            "불확실성 ±σ": f"{enkf_result.ann_rech_std[i]:.0f}",
            "95% CI 하한": f"{enkf_result.ann_rech_ci_lo[i]:.0f}",
            "95% CI 상한": f"{enkf_result.ann_rech_ci_hi[i]:.0f}",
        })
    df_enkf = pd.DataFrame(rows)
    st.dataframe(df_enkf, use_container_width=True, hide_index=True)

    # 관정 검증 메트릭
    well_name_disp = [p.name for p in enkf_result.points if p.is_well]
    if well_name_disp:
        wn = well_name_disp[0]
        wp = enkf_result.well_params.get(wn, {})
        ref = wp.get("ref_mm")
        i_w = next(j for j, p in enumerate(enkf_result.points) if p.name == wn)
        ev = enkf_result.ann_rech_mm[i_w]
        lo = enkf_result.ann_rech_ci_lo[i_w]
        hi = enkf_result.ann_rech_ci_hi[i_w]

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("EnKF 함양량", f"{ev:.0f} mm/yr")
        if ref:
            mc2.metric("run_logic 기준값", f"{ref:.0f} mm/yr", delta=f"{ev-ref:+.0f}")
            ok = "✅ CI 포함" if lo <= ref <= hi else "⚠️ CI 미포함"
            mc3.metric("검증 결과", ok)
        mc4.metric("95% CI", f"[{lo:.0f}, {hi:.0f}]")

    # ── 공간 분포 지도 ──
    _render_spatial_map(enkf_result, loc_radius)


def _render_spatial_map(enkf_result, loc_radius):
    st.markdown("#### 🗺️ 공간 함양률 분포")
    fig_map = go.Figure()

    grid_pts = [(i, p) for i, p in enumerate(enkf_result.points) if not p.is_well]
    if grid_pts:
        gidxs, gpts = zip(*grid_pts)
        fig_map.add_trace(go.Scatter(
            x=[p.x for p in gpts],
            y=[p.y for p in gpts],
            mode="markers+text",
            marker=dict(
                size=20,
                color=[enkf_result.ann_rech_mm[i] for i in gidxs],
                colorscale="Blues",
                showscale=True,
                colorbar=dict(title="mm/yr", x=1.02),
                symbol="square",
                line=dict(color="black", width=1),
            ),
            text=[f"◆{p.name}<br>{enkf_result.ann_rech_mm[i]:.0f}"
                  f"±{enkf_result.ann_rech_std[i]:.0f}"
                  for i, p in zip(gidxs, gpts)],
            textposition="top center",
            name="격자점",
            hovertemplate=(
                "<b>%{text}</b><br>"
                "R: %{marker.color:.0f} mm/yr<extra></extra>"
            ),
        ))

    well_pts = [(i, p) for i, p in enumerate(enkf_result.points) if p.is_well]
    if well_pts:
        widxs, wpts = zip(*well_pts)
        fig_map.add_trace(go.Scatter(
            x=[p.x for p in wpts],
            y=[p.y for p in wpts],
            mode="markers+text",
            marker=dict(
                size=22, color="#DC2626",
                symbol="star",
                line=dict(color="black", width=1.5),
            ),
            text=[f"★{p.name}<br>{enkf_result.ann_rech_mm[i]:.0f}mm/yr"
                  for i, p in zip(widxs, wpts)],
            textposition="top center",
            name="관정 (앵커)",
        ))

    if well_pts:
        wp0 = well_pts[0][1]
        theta_c = np.linspace(0, 2 * np.pi, 100)
        fig_map.add_trace(go.Scatter(
            x=wp0.x + loc_radius * np.cos(theta_c),
            y=wp0.y + loc_radius * np.sin(theta_c),
            mode="lines",
            line=dict(color="green", dash="dash", width=1.5),
            name=f"국소화 반경 ({loc_radius:.1f}km)",
        ))

    fig_map.update_layout(
        height=420,
        xaxis_title="X (km)", yaxis_title="Y (km)",
        xaxis=dict(scaleanchor="y"),
        legend=dict(x=0, y=1, bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(l=40, r=80, t=40, b=40),
        title="공간 함양률 분포 (관정 앵커 기반)",
        **{k: v for k, v in LAYOUT_BASE.items() if k != "margin"},
    )
    st.plotly_chart(fig_map, use_container_width=True)


def _render_extra_charts(enkf_result):
    """Prior vs posterior and ensemble histograms."""
    st.markdown("---")
    tab_prior, tab_hist2 = st.tabs(["📊 사전 vs 사후 분포", "📈 앙상블 히스토그램"])

    with tab_prior:
        st.caption(
            "사전(Prior): 파라미터 불확실성만 반영. "
            "사후(Posterior): 관정 함양량 관측 동화 후."
        )
        n_pts_r = len(enkf_result.points)
        cols_pr = st.columns(min(n_pts_r, 4))
        for i, pt in enumerate(enkf_result.points):
            col_p = cols_pr[i % 4]
            prior_m = float(enkf_result.prior_mean[i]) if len(enkf_result.prior_mean) > i else 0
            prior_s = float(enkf_result.prior_std[i]) if len(enkf_result.prior_std) > i else 0
            post_m = enkf_result.ann_rech_mm[i]
            post_s = enkf_result.ann_rech_std[i]
            lo = enkf_result.ann_rech_ci_lo[i]
            hi = enkf_result.ann_rech_ci_hi[i]

            fig_pr = go.Figure()
            fig_pr.add_trace(go.Bar(
                x=["사전(Prior)", "사후(Posterior)"],
                y=[prior_m, post_m],
                error_y=dict(type="data", array=[prior_s * 1.96, post_s * 1.96],
                             visible=True),
                marker_color=["#94A3B8", "#2563EB" if pt.is_well else "#DC2626"],
                name=pt.name,
            ))
            if pt.is_well and pt.name in enkf_result.well_params:
                ref = enkf_result.well_params[pt.name].get("ref_mm")
                if ref:
                    fig_pr.add_hline(
                        y=ref, line_color="red",
                        line_dash="dot", line_width=2,
                        annotation_text=f"Ref={ref:.0f}",
                        annotation_font_color="red",
                    )
            sym = "★" if pt.is_well else "◆"
            fig_pr.update_layout(
                height=260,
                title=f"{sym}{pt.name}  95%CI=[{lo:.0f},{hi:.0f}]",
                yaxis_title="함양량 (mm/yr)",
                showlegend=False,
                margin=dict(l=30, r=10, t=50, b=30),
                **{k: v for k, v in LAYOUT_BASE.items() if k != "margin"},
            )
            col_p.plotly_chart(fig_pr, use_container_width=True)

    with tab_hist2:
        n_pts_r = len(enkf_result.points)
        cols_h2 = st.columns(min(n_pts_r, 4))
        for i, pt in enumerate(enkf_result.points):
            col_h2 = cols_h2[i % 4]
            ens_vals = enkf_result.ann_rech_ens[i]
            color = "#2563EB" if pt.is_well else "#DC2626"
            fig_h2 = go.Figure()
            fig_h2.add_trace(go.Histogram(
                x=ens_vals, nbinsx=30,
                marker_color=color, opacity=0.75,
            ))
            fig_h2.add_vline(
                x=enkf_result.ann_rech_mm[i],
                line_color="black", line_width=2,
                annotation_text=f"평균<br>{enkf_result.ann_rech_mm[i]:.0f}",
            )
            fig_h2.add_vline(x=enkf_result.ann_rech_ci_lo[i],
                line_color="gray", line_dash="dash", line_width=1)
            fig_h2.add_vline(x=enkf_result.ann_rech_ci_hi[i],
                line_color="gray", line_dash="dash", line_width=1)
            if pt.is_well and pt.name in enkf_result.well_params:
                ref = enkf_result.well_params[pt.name].get("ref_mm")
                if ref:
                    fig_h2.add_vline(
                        x=ref, line_color="red",
                        line_dash="dot", line_width=2,
                        annotation_text=f"Ref<br>{ref:.0f}",
                        annotation_font_color="red",
                    )
            sym = "★" if pt.is_well else "◆"
            fig_h2.update_layout(
                height=260, title=f"{sym}{pt.name}",
                xaxis_title="mm/yr", showlegend=False,
                margin=dict(l=30, r=10, t=50, b=40),
                **{k: v for k, v in LAYOUT_BASE.items() if k != "margin"},
            )
            col_h2.plotly_chart(fig_h2, use_container_width=True)

    # ── 해석 가이드 ──
    st.markdown("---")
    with st.expander("💡 결과 해석 가이드"):
        st.markdown("""
**관정 (★)** — `run_logic_v27` 최적화 함양률이 관측값. Ref가 95% CI 안에 있으면 ✓

**격자점 (◆)** — 파라미터(k, z) 섭동 앙상블 + 관정 R 관측 동화 결과
- 관정에 가까울수록 관정 R에 수렴, 불확실성 작음
- 국소화 반경 밖은 사전분포로 회귀 (관정과 독립)
- 사전≈사후이면 격자가 국소화 반경 밖에 있다는 의미 → 반경을 늘리거나 격자를 관정에 더 가깝게

**국소화 반경 설정**
- 작게: 관정 영향 범위 좁음 → 격자가 빠르게 사전분포로 회귀
- 크게: 관정 영향 넓음 → 먼 격자도 관정 R에 당겨짐
- 권장: 표준유역 반경의 50~80%

**obs_noise_mm (관측 불확실성)**
- 크게: 관정 R을 덜 신뢰 → 격자가 사전분포에 더 머뭄
- 작게: 관정 R을 강하게 신뢰 → 격자가 관정에 강하게 수렴
        """)
