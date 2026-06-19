"""Tab 10 — 유역 단위 함양율 리포트 (Lumped vs Soil-weighted).

다중 관정 .txt + 정밀토양도 .shp 를 결합해 유역 단위 면적 가중 함양율을 계산.
논문에서 제시할 핵심 비교 — 단순 평균(Lumped) 대비 토양 면적 가중(Soil-weighted)
방법의 차이를 정량적으로 보여준다.
"""
from __future__ import annotations

import os
import traceback
from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st

# TabContext import 시도 — 실패해도 진행
try:
    from ui import TabContext
except ImportError as e:
    st.warning(f"⚠️ TabContext import 실패: {e}")
    TabContext = type('TabContext', (), {})


def render(tab, ctx):
    with tab:
        st.markdown("### 🗾 유역 함양율 리포트 (Lumped vs Soil-weighted)")
        st.caption(
            "정밀토양도(.shp) 의 HSG 면적 분포로 다중 관정 함양율을 면적 가중 평균합니다.  "
            "Lumped(기존, 단순 평균) 와 Soil-weighted(제안 방법) 를 동시 비교."
        )

        # 의존성 점검
        try:
            import wells_registry as wr
            from wells_registry import WELLS, WATERSHEDS
            from watershed_aggregator import estimate_watershed
            from shp_soil_mapper import SHP_PATH_DEFAULT
        except Exception as e:
            st.error(f"모듈 임포트 실패: {e}")
            return

        # ── 관정/유역 관리 (CRUD) ──
        _render_registry_manager(wr)
        # 관리 후 최신 상태 다시 가져오기
        WELLS = wr.WELLS
        WATERSHEDS = wr.WATERSHEDS
        if not WATERSHEDS:
            st.error("등록된 관정이 없습니다. 위 '관정/유역 관리' 에서 추가하세요.")
            return

        if not os.path.exists(SHP_PATH_DEFAULT):
            st.warning(
                f"토양도 .shp 가 없습니다: `{SHP_PATH_DEFAULT}`\n\n"
                "이 탭은 정밀토양도 파일이 있어야 작동합니다."
            )
            return

        # ── 분석 모드 ──
        try:
            from well_results_store import list_stored
            stored_list = list_stored()
        except Exception:
            stored_list = []
        stored_names = {s.well_name for s in stored_list}

        mode_col1, mode_col2 = st.columns([2, 3])
        with mode_col1:
            mode = st.radio(
                "분석 모드",
                options=["⚡ 저장된 결과 사용 (권장)", "🔄 실시간 재실행"],
                index=0 if stored_list else 1,
                help=(
                    "저장된 결과: Tab 1 에서 정성껏 튜닝하고 💾 저장한 결과를 로딩.\n"
                    "실시간 재실행: 기본 파라미터로 .txt 재계산 (빠른 미리보기용)."
                ),
            )
            use_cached = mode.startswith("⚡")
        with mode_col2:
            if stored_list:
                st.success(
                    f"📂 현재 저장된 관정: {len(stored_list)}개 — "
                    + ", ".join(sorted(stored_names))
                )
            else:
                st.info("ℹ️ 저장된 관정 결과 없음 — Tab 1 에서 분석 후 💾 저장하세요.")

        # ── 유역 선택 ──
        col1, col2 = st.columns([2, 1])
        with col1:
            ws_choice = st.selectbox(
                "유역 선택",
                options=list(WATERSHEDS.keys()),
                index=0,
                help="wells_registry.WATERSHEDS 에 정의된 유역",
            )
        with col2:
            buffer_km = st.number_input(
                "관정 버퍼 (km)",
                min_value=0.5, max_value=10.0, value=2.0, step=0.5,
                help="유역 폴리곤이 없을 때 관정 주위 buffer 영역으로 토양 분포 산출",
            )

        # 유역에 속한 관정
        well_names = WATERSHEDS[ws_choice]
        st.markdown(f"**유역 `{ws_choice}` 관정**: " + ", ".join(well_names))

        cwd = os.getcwd()
        file_paths: Dict[str, str] = {}

        if use_cached:
            # 캐시 모드 — 저장된 관정 확인
            cached_in_ws = [n for n in well_names if n in stored_names]
            missing_cached = [n for n in well_names if n not in stored_names]
            if missing_cached:
                st.warning(
                    f"⚠️ 저장된 결과 없는 관정: {missing_cached} — "
                    "Tab 1 에서 해당 관정 분석 후 💾 저장하세요. 진행은 가능 (누락 관정은 fallback 평균)."
                )
            if not cached_in_ws:
                st.error("이 유역의 저장된 관정이 하나도 없습니다.")
                return
        else:
            # 실시간 모드 — .txt 매칭
            missing = []
            for n in well_names:
                for cand in [f"{n}.txt", os.path.join(cwd, f"{n}.txt")]:
                    if os.path.exists(cand):
                        file_paths[n] = cand
                        break
                else:
                    missing.append(n)
            if missing:
                st.warning(f".txt 파일 누락: {missing} — 해당 관정은 분석에서 제외됩니다.")
            if not file_paths:
                st.error("사용 가능한 .txt 파일이 없습니다.")
                return

        # ── 분석 결과 캐시 키 (rerun 사이 유지) ───────────────
        # 버튼 클릭 → estimate_watershed → session_state[result_key] 저장.
        # 이후 rerun (Hierarchical 버튼 등) 에서는 캐시를 그대로 표시 →
        # 탭이 "리셋"되지 않음.
        result_key = f"ws_result_{ws_choice}_{use_cached}_{buffer_km}"
        cached_r = st.session_state.get(result_key)

        bcol1, bcol2 = st.columns([3, 1])
        do_run = bcol1.button(
            "🚀 유역 함양율 분석 실행",
            type="primary" if cached_r is None else "secondary",
            key=f"run_ws_{ws_choice}",
        )
        if cached_r is not None:
            if bcol2.button("🔄 다시 실행", key=f"clear_ws_{ws_choice}"):
                st.session_state.pop(result_key, None)
                st.rerun()

        if do_run:
            spinner_msg = (
                f"{ws_choice} 유역 — 저장된 결과 로딩..."
                if use_cached else f"{ws_choice} 유역 분석 중 (실시간 재실행)..."
            )
            with st.spinner(spinner_msg):
                try:
                    r = estimate_watershed(
                        ws_choice,
                        file_paths=file_paths,
                        run_fao=False,
                        buffer_km=buffer_km,
                        use_cached=use_cached,
                    )
                except Exception as e:
                    st.error(f"분석 실패: {e}")
                    st.code(traceback.format_exc())
                    return
            st.session_state[result_key] = r
        elif cached_r is not None:
            r = cached_r
        else:
            st.info("위 버튼을 눌러 분석을 시작하세요. "
                     "결과는 세션 동안 자동 캐시되어 다른 버튼(Hierarchical 등) "
                     "클릭 시에도 유지됩니다.")
            return

        # ── 결과 표시 ──
        st.success(f"✅ {len(r.wells)}개 관정 분석 완료")

        # 유역 토양 분포
        st.markdown("#### 1. 유역 토양 분포 (HSG 면적 비율)")
        col_a, col_b = st.columns([1, 1])
        with col_a:
            fr_df = pd.DataFrame({
                "HSG": list(r.profile.hsg_fractions.keys()),
                "면적 비율": [f"{v*100:.1f}%" for v in r.profile.hsg_fractions.values()],
            })
            st.dataframe(fr_df, hide_index=True, use_container_width=True)
            st.metric("유역 총면적 (버퍼 union)", f"{r.profile.total_area_km2:.1f} km²")
            st.metric("Dominant HSG", r.profile.dominant_hsg)
        with col_b:
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(4, 4))
                hsgs = list(r.profile.hsg_fractions.keys())
                vals = [r.profile.hsg_fractions[h]*100 for h in hsgs]
                colors = {"A": "#fde725", "B": "#5ec962",
                          "C": "#21918c", "D": "#3b528b"}
                ax.pie(vals, labels=[f"HSG {h}\n{v:.1f}%" for h, v in zip(hsgs, vals)],
                       colors=[colors.get(h, "gray") for h in hsgs],
                       autopct=None, startangle=90)
                ax.set_title(f"{ws_choice} HSG 분포")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            except Exception as e:
                st.caption(f"파이차트 렌더 실패: {e}")

        # 관정별 결과 (Bayesian 결과 있으면 같이 표시)
        st.markdown("#### 2. 관정별 함양율")
        bayes_pcts = []
        well_rows = []
        for w in r.wells:
            row = {
                "관정": w.well.name,
                "대수층": w.well.aquifer,
                "HSG": w.soil.hydro_type,
                "sn": w.sn_used,
                "WTF (%)": f"{w.wtf_pct:.2f}" if w.wtf_pct is not None else "-",
                "WTF (mm/yr)": f"{w.wtf_mm:.0f}" if w.wtf_mm is not None else "-",
            }
            # cached 모드에서 stored 객체 추가 컬럼
            if use_cached:
                try:
                    from well_results_store import load as load_stored
                    s = load_stored(w.well.name)
                except Exception:
                    s = None
                if s and s.bayes_rech_pct_post_mean is not None:
                    row["Bayes (%) [95% CI]"] = (
                        f"{s.bayes_rech_pct_post_mean:.2f} "
                        f"[{s.bayes_rech_pct_post_lo95:.2f}, "
                        f"{s.bayes_rech_pct_post_hi95:.2f}]"
                    )
                    bayes_pcts.append({
                        "well": w.well.name,
                        "hsg": w.soil.hydro_type,
                        "rech_pct": s.bayes_rech_pct_post_mean,
                        "lo95": s.bayes_rech_pct_post_lo95,
                        "hi95": s.bayes_rech_pct_post_hi95,
                    })
                else:
                    row["Bayes (%) [95% CI]"] = "—"
            well_rows.append(row)
        st.dataframe(pd.DataFrame(well_rows), hide_index=True, use_container_width=True)
        if use_cached and not bayes_pcts:
            st.caption("ℹ️ Bayesian 결과가 저장된 관정 없음 — Tab 1 에서 🎲 Bayesian 추정 후 💾 저장하세요.")

        # 유역 종합 — Lumped vs Soil-weighted
        st.markdown("#### 3. 유역 종합 함양율 — Lumped vs Soil-weighted")
        col_l, col_s, col_d = st.columns(3)
        with col_l:
            st.metric(
                "Lumped (기존)",
                f"{r.lumped_wtf_pct:.2f} %" if r.lumped_wtf_pct is not None else "-",
                help="모든 관정의 단순 산술평균. 토양 변이 무시.",
            )
        with col_s:
            st.metric(
                "Soil-weighted (제안)",
                f"{r.soil_weighted_wtf_pct:.2f} %" if r.soil_weighted_wtf_pct is not None else "-",
                help="HSG 면적 비율로 가중 평균. 관측 관정이 없는 HSG 는 "
                     "전체 평균이 아니라 'Sy 단위당 함양 × 해당 HSG 문헌 Sy' 로 추정.",
            )
        with col_d:
            if r.lumped_wtf_pct is not None and r.soil_weighted_wtf_pct is not None:
                delta = r.soil_weighted_wtf_pct - r.lumped_wtf_pct
                st.metric("Δ (제안 - 기존)", f"{delta:+.2f} %p",
                          help="토양 변이성이 함양 추정에 미치는 영향 크기.")

        if r.P_annual_mm:
            st.caption(
                f"📌 평균 연강수량 (관정 데이터 환산): **{r.P_annual_mm:.0f} mm/yr** — "
                "데이터 기간이 짧을 수 있으니 외삽치임을 유의."
            )

        # ── Phase 5: Bias-Aware WTF (학습 기반 보정 + α conservatism) ──
        _render_bias_aware_section(r)

        # ── Hierarchical Bayesian (Phase 3) ──
        # 항상 표시. 사용 가능 여부는 섹션 내부에서 진단/안내.
        if len(r.wells) >= 1:
            _render_hierarchical_bayesian(
                ws_choice, [w.well.name for w in r.wells], r.profile,
                use_cached=use_cached,
            )

        # Bayesian 유역 함양율 (있는 관정만 평균/면적가중)
        if use_cached and bayes_pcts:
            st.markdown("#### 3-B. 🎲 Bayesian 유역 함양율 (Phase 1)")
            # Lumped Bayesian = 단순 평균
            lumped_bayes = float(np.mean([b["rech_pct"] for b in bayes_pcts]))
            lumped_lo = float(np.mean([b["lo95"] for b in bayes_pcts]))
            lumped_hi = float(np.mean([b["hi95"] for b in bayes_pcts]))

            # Soil-weighted Bayesian = HSG 면적 가중
            # 미관측 HSG 는 grand mean 대신 Sy 비율 스케일링 (watershed_aggregator 와 일치)
            from shp_soil_mapper import HSG_TO_SY
            _ratios = [(b["rech_pct"] / HSG_TO_SY[b["hsg"]],
                        b["lo95"] / HSG_TO_SY[b["hsg"]],
                        b["hi95"] / HSG_TO_SY[b["hsg"]])
                       for b in bayes_pcts if HSG_TO_SY.get(b["hsg"], 0) > 0]
            r_per_sy = (
                tuple(float(np.mean([x[i] for x in _ratios])) for i in range(3))
                if _ratios else None
            )
            sw_bayes = 0.0
            sw_lo = 0.0
            sw_hi = 0.0
            wsum = 0.0
            for hsg, frac in r.profile.hsg_fractions.items():
                hsg_bayes = [b for b in bayes_pcts if b["hsg"] == hsg]
                if hsg_bayes:
                    rep_mean = float(np.mean([b["rech_pct"] for b in hsg_bayes]))
                    rep_lo = float(np.mean([b["lo95"] for b in hsg_bayes]))
                    rep_hi = float(np.mean([b["hi95"] for b in hsg_bayes]))
                elif r_per_sy is not None and HSG_TO_SY.get(hsg, 0) > 0:
                    sy_h = HSG_TO_SY[hsg]
                    rep_mean = r_per_sy[0] * sy_h
                    rep_lo = r_per_sy[1] * sy_h
                    rep_hi = r_per_sy[2] * sy_h
                else:
                    rep_mean = lumped_bayes
                    rep_lo = lumped_lo
                    rep_hi = lumped_hi
                sw_bayes += frac * rep_mean
                sw_lo += frac * rep_lo
                sw_hi += frac * rep_hi
                wsum += frac
            if wsum > 0:
                sw_bayes /= wsum; sw_lo /= wsum; sw_hi /= wsum

            cb1, cb2 = st.columns(2)
            with cb1:
                st.metric(
                    "Lumped (Bayesian 평균)",
                    f"{lumped_bayes:.2f} %",
                )
                st.caption(f"95% CI band: [{lumped_lo:.2f}, {lumped_hi:.2f}]")
            with cb2:
                st.metric(
                    "Soil-weighted (Bayesian)",
                    f"{sw_bayes:.2f} %",
                    delta=f"{sw_bayes - lumped_bayes:+.2f} %p vs Lumped",
                )
                st.caption(f"95% CI band: [{sw_lo:.2f}, {sw_hi:.2f}]")
            st.caption(
                "✏️ Bayesian 추정은 점추정 대신 95% 신뢰구간을 제공합니다. "
                "관정 수가 늘어나면 CI 가 좁아지며, 양수시험 Sy 가 추가되면 더욱 좁아집니다."
            )

        # 메모
        if r.method_notes:
            with st.expander("⚠️ 메모 / 경고"):
                for m in r.method_notes:
                    st.write("- " + m)

        # 다운로드
        st.markdown("#### 4. 결과 내보내기")
        rows = []
        for w in r.wells:
            rows.append({
                "watershed": ws_choice,
                "well": w.well.name,
                "aquifer": w.well.aquifer,
                "HSG": w.soil.hydro_type,
                "soil_code": w.soil.soil_code,
                "sn_idx": w.sn_used,
                "wtf_pct": w.wtf_pct,
                "wtf_mm": w.wtf_mm,
                "P_annual_mm": w.P_annual_mm,
            })
        rows.append({
            "watershed": ws_choice, "well": "<LUMPED>",
            "wtf_pct": r.lumped_wtf_pct,
        })
        rows.append({
            "watershed": ws_choice, "well": "<SOIL_WEIGHTED>",
            "wtf_pct": r.soil_weighted_wtf_pct,
        })
        csv = pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 CSV 다운로드",
            data=csv,
            file_name=f"watershed_{ws_choice}_recharge.csv",
            mime="text/csv",
        )


def _render_bias_aware_section(r):
    """Tab 10 — Phase 5: bias correction + α conservatism + multi-proxy + Fig 13."""
    import os, json
    st.markdown("#### 4. 🎯 Bias-Aware WTF (Phase 5 — 학습 기반 보정)")
    st.caption(
        "WTF 의 구조적 bias 를 cascade-truth 학습 회귀로 보정. "
        "α 슬라이더로 보수성 조절 (0=원본 WTF, 0.3=권장, 1=full)."
    )

    # bias_model.json 로드
    bm_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "bias_model.json",
    )
    if not os.path.exists(bm_path):
        st.warning(
            "⚠️ `bias_model.json` 없음 — `python -m bias_correction --n_rep 8 --output bias_model` "
            "을 먼저 실행하세요."
        )
        return
    with open(bm_path) as f:
        bm = json.load(f)
    coefs = np.array(bm["coefs"])

    # α 슬라이더
    col_a, col_b = st.columns([2, 1])
    with col_a:
        alpha = st.slider(
            "Conservatism α",
            min_value=0.0, max_value=1.0, value=0.3, step=0.05,
            help="0=no correction, 0.3=권장 default (한국 lit. 범위), 1=full cascade-strength",
        )
    with col_b:
        ET_over_P = st.number_input(
            "ET/P (지역 climate)",
            min_value=0.1, max_value=1.0, value=0.5, step=0.05,
            help="FAO-56 기반 한국 monsoon 평균 ≈ 0.5",
        )

    # β̂ 계산 (HSG 면적 가중)
    try:
        from soil_db import SOIL_DB
        from watershed_aggregator import HSG_TO_SN_ALLUVIAL
    except Exception as e:
        st.error(f"모듈 import 실패: {e}")
        return

    obs_sigma = 0.02
    betas, weights = [], []
    for hsg, frac in r.profile.hsg_fractions.items():
        if frac < 0.01:
            continue
        sn = HSG_TO_SN_ALLUVIAL.get(hsg, 12)
        sr = SOIL_DB[sn]
        x = np.array([1.0, sr.sy_lit, np.log(sr.tau), ET_over_P, obs_sigma])
        betas.append(float(np.dot(coefs, x)))
        weights.append(frac)
    if not betas:
        st.warning("HSG 분포 없음 — 보정 불가")
        return
    weights = np.array(weights) / sum(weights)
    beta_hat = float(np.sum(np.array(betas) * weights))

    # 보정 결과
    sw_pct = r.soil_weighted_wtf_pct or 0.0
    bc_results = {}
    for a_label, a_val in [("α=0 (원본)", 0.0), ("α=0.3 (권장)", 0.3),
                            ("α=0.5", 0.5), ("α=1.0 (full)", 1.0),
                            (f"α={alpha:.2f} (선택)", alpha)]:
        denom = max(1.0 + a_val * beta_hat, 0.05)
        bc_results[a_label] = sw_pct / denom

    # 지표 표시
    st.markdown("**선택된 α 값에서:**")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Soil-weighted (α=0)", f"{sw_pct:.2f} %")
    with c2:
        delta = bc_results[f"α={alpha:.2f} (선택)"] - sw_pct
        st.metric(
            f"Bias-corrected (α={alpha:.2f})",
            f"{bc_results[f'α={alpha:.2f} (선택)']:.2f} %",
            delta=f"{delta:+.2f} %p vs 원본",
        )
    with c3:
        st.metric("β̂ (correction factor)",
                  f"{beta_hat:+.3f}",
                  help="음수 = WTF 가 truth 대비 under-predict → 보정으로 증가")

    # 모든 α 결과 표
    df_alpha = pd.DataFrame({
        "α": list(bc_results.keys()),
        "함양율 (%)": [f"{v:.2f}" for v in bc_results.values()],
    })
    st.dataframe(df_alpha, hide_index=True, use_container_width=False)

    # Multi-proxy envelope (Phase 5) — 한국 climate normal 기준
    st.markdown("---")
    st.markdown("**🛡️ Multi-proxy consistency check (synthetic-independent)**")
    try:
        from evaluation.proxy_validation import proxy_envelope
        env = proxy_envelope(P_annual_mm=1100.0)
    except Exception as e:
        st.caption(f"⚠️ proxy envelope 모듈 로드 실패: {e}")
        return

    # Plotly 인터랙티브 decision chart (Fig 13 equivalent)
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        # Background bands
        fig.add_hrect(y0=env.envelope_lo, y1=env.envelope_hi,
                      fillcolor="#10B981", opacity=0.10, line_width=0,
                      annotation_text="Multi-proxy envelope",
                      annotation_position="top right")
        fig.add_hrect(y0=env.molit_lo, y1=env.molit_hi,
                      fillcolor="#F59E0B", opacity=0.18, line_width=0)
        fig.add_hrect(y0=env.cmb_lo, y1=env.cmb_hi,
                      fillcolor="#7C3AED", opacity=0.13, line_width=0)
        fig.add_hrect(y0=env.bfi_lo, y1=env.bfi_hi,
                      fillcolor="#0891B2", opacity=0.08, line_width=0)

        # α curve
        alphas_grid = np.linspace(0, 1, 51)
        rech_curve = [sw_pct / max(1.0 + a * beta_hat, 0.05) for a in alphas_grid]
        fig.add_trace(go.Scatter(
            x=alphas_grid, y=rech_curve,
            mode="lines", name=f"{r.watershed} 함양율",
            line=dict(color="#DC2626", width=3),
        ))
        # 마커: α = 0, 0.3, 1.0
        for a_mark, sym in zip([0.0, 0.3, 1.0], ["circle", "square", "star"]):
            R_mark = sw_pct / max(1.0 + a_mark * beta_hat, 0.05)
            fig.add_trace(go.Scatter(
                x=[a_mark], y=[R_mark],
                mode="markers+text",
                marker=dict(symbol=sym, size=14, color="#DC2626",
                            line=dict(color="black", width=1)),
                text=[f"  {R_mark:.1f}%"],
                textposition="middle right",
                showlegend=False,
            ))
        # 사용자 선택 α (highlight)
        R_user = sw_pct / max(1.0 + alpha * beta_hat, 0.05)
        fig.add_trace(go.Scatter(
            x=[alpha], y=[R_user],
            mode="markers",
            marker=dict(symbol="diamond", size=18, color="#F59E0B",
                        line=dict(color="black", width=2)),
            name=f"선택 α={alpha:.2f}",
        ))
        fig.update_layout(
            title=f"α-spectrum decision chart — {r.watershed}",
            xaxis_title="Conservatism α",
            yaxis_title="Recharge ratio (% of P)",
            xaxis=dict(range=[-0.05, 1.05]),
            yaxis=dict(range=[0, max(50, max(rech_curve) * 1.2)]),
            height=420, margin=dict(l=50, r=20, t=50, b=50),
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True, theme=None)

        st.caption(
            f"**Envelope check**: Multi-proxy 12–45% (BFI {env.bfi_lo:.0f}–{env.bfi_hi:.0f}%, "
            f"CMB {env.cmb_lo:.0f}–{env.cmb_hi:.0f}%, MOLIT {env.molit_lo:.0f}–{env.molit_hi:.0f}%). "
            "선택한 α 가 envelope 안에 있으면 ✅ 운영적 방어 가능."
        )

        # Envelope 안/밖 판정
        if env.in_envelope(R_user):
            st.success(f"✅ α={alpha:.2f} 결과 ({R_user:.2f}%) 가 multi-proxy envelope 안에 있습니다.")
        else:
            st.warning(f"⚠️ α={alpha:.2f} 결과 ({R_user:.2f}%) 가 envelope 밖 — α 를 낮추세요.")
    except Exception as e:
        st.caption(f"plotly 차트 렌더 실패: {e}")


def _render_hierarchical_bayesian(ws_choice, well_names, profile,
                                    use_cached: bool = True):
    """Tab 10 — Phase 3 hierarchical Bayesian section.

    Parameters
    ----------
    use_cached : bool
        분석 모드 정보. False (실시간 재실행 모드) 면 fit_from_stored 가
        well_results/*.json 을 못 읽으므로 안내 후 종료.
    """
    st.markdown("#### 3-C. 🔮 Hierarchical Bayesian (Phase 3)")
    st.caption(
        "유역-HSG-관정 3-level 계층 모델 (emcee MCMC). "
        "각 관정 Sy 의 posterior + 유역 함양율 95% CI 산출. "
        "양수시험 데이터가 있는 관정은 더 좁은 CI 를 가짐."
    )

    # ── 모드 진단 — 실시간 모드면 안내 후 종료 ────────────
    # Hierarchical 은 저장된 well_results/*.json 의 Sy_eff, P_annual 등을
    # 사용합니다. 실시간 모드에서는 저장이 안 되므로 fit_from_stored 가
    # 동작 불가.
    if not use_cached:
        st.info(
            "ℹ️ **실시간 재실행 모드**에서는 Hierarchical 분석을 사용할 수 없습니다.\n\n"
            "이유: Hierarchical 모델은 저장된 관정 결과 "
            "(`well_results/*.json` — Sy_eff, P_annual, hydro_type 등) 를 "
            "입력으로 사용합니다.\n\n"
            "**사용 방법:**\n"
            "1. Tab 1 에서 각 관정 분석 후 💾 **저장** 버튼 클릭\n"
            "2. 위 분석 모드를 **'⚡ 저장된 결과 사용 (권장)'** 로 변경\n"
            "3. 다시 이 섹션으로 와서 적합 실행"
        )
        # 어떤 관정이 이미 저장되어 있는지 알려주기 (재실행 안 해도 도움)
        try:
            from well_results_store import load as _load
            stored_avail = []
            for n in well_names:
                if _load(n) is not None:
                    stored_avail.append(n)
            if stored_avail:
                st.caption(
                    f"💡 이 유역에 이미 저장된 관정: "
                    f"{', '.join(stored_avail)} ({len(stored_avail)}/{len(well_names)})"
                )
            else:
                st.caption(
                    f"💡 이 유역의 관정 중 저장된 것 없음 (총 {len(well_names)}개)."
                )
        except Exception:
            pass
        return

    # 관측 HSG 커버리지 사전 진단
    try:
        from well_results_store import load
        from wells_registry import WELLS as _WELLS
        observed_hsgs = set()
        for n in well_names:
            s = load(n)
            if s and s.hydro_type:
                observed_hsgs.add(s.hydro_type)
        all_hsgs_in_basin = set(profile.hsg_fractions.keys())
        unobserved = all_hsgs_in_basin - observed_hsgs
        if unobserved:
            obs_frac = sum(profile.hsg_fractions.get(h, 0) for h in observed_hsgs) * 100
            unobs_frac = 100 - obs_frac
            st.warning(
                f"⚠️ **관측 안 된 HSG: {sorted(unobserved)} "
                f"({unobs_frac:.0f}% 면적)** — "
                "이 HSG 들은 관정이 없어서 *literature prior* 로만 추정됩니다. "
                f"관측 신뢰 면적 = {obs_frac:.0f}% ({sorted(observed_hsgs)})."
            )
            st.caption(
                "💡 Hierarchical 모델은 관측되지 않은 HSG 의 평균을 prior 분포 "
                "(HSG_AQUIFER_TO_SN → SOIL_DB.sy_lit) 로 자동 추정합니다. "
                "관측 면적이 작을수록 posterior CI 가 넓어지는 게 정상이며, "
                "관측 데이터가 더 추가되면 prior dependency 는 자동으로 줄어듭니다."
            )
    except Exception:
        pass

    # ── 결과 캐시 (rerun 사이 보존) ────────────────────────
    cache_key = f"h_bayes_result_{ws_choice}"
    cached = st.session_state.get(cache_key)

    bcol_run, bcol_clear = st.columns([3, 1])
    do_run = bcol_run.button(
        "🔮 Hierarchical Bayesian 적합 실행 (수십 초)",
        key=f"run_h_bayes_{ws_choice}",
        type="primary" if cached is None else "secondary",
    )
    if cached is not None:
        if bcol_clear.button("🔄 결과 지우기",
                              key=f"clear_h_bayes_{ws_choice}"):
            st.session_state.pop(cache_key, None)
            st.rerun()

    if do_run:
        try:
            from bayes_hierarchical import fit_from_stored
            with st.spinner("emcee MCMC 샘플링... (수십 초)"):
                res = fit_from_stored(
                    ws_choice, well_names,
                    hsg_fractions=profile.hsg_fractions,
                    n_walkers=24, n_steps=2000, burn_in=500,
                )
            st.session_state[cache_key] = res
            cached = res
        except Exception as e:
            st.error(f"Hierarchical 분석 실패: {type(e).__name__}: {e}")
            import traceback
            with st.expander("Traceback (디버깅용)"):
                st.code(traceback.format_exc())
            return

    if cached is None:
        st.info("위 버튼을 눌러 hierarchical posterior 를 산출하세요. "
                 "결과는 세션 동안 자동 캐시됩니다.")
        return

    res = cached
    st.success(f"✅ 결과 표시 중 — 다른 탭/스크롤해도 유지됩니다 "
                f"(다시 돌리려면 '🔄 결과 지우기' 클릭).")

    # 진단
    st.caption(
        f"📊 N samples={res.n_samples}, "
        f"acceptance={res.mean_acceptance_rate:.2f}, "
        f"converged={'✅' if res.converged else '⚠️'}"
    )

    # 유역 평균 Sy
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "μ_watershed (Sy)",
            f"{res.mu_watershed_mean:.3f}",
        )
        st.caption(f"95% CI: [{res.mu_watershed_lo95:.3f}, {res.mu_watershed_hi95:.3f}]")
    with col2:
        if np.isfinite(res.rech_pct_watershed_mean):
            st.metric(
                "유역 함양율 (Hierarchical)",
                f"{res.rech_pct_watershed_mean:.2f} %",
            )
            st.caption(
                f"95% CI: [{res.rech_pct_watershed_lo95:.2f}, "
                f"{res.rech_pct_watershed_hi95:.2f}] %"
            )

    # HSG-level posterior — 관측 vs prior-only 표시
    st.markdown("**HSG 별 posterior μ**")
    observed_hsgs = set(res.hsgs)
    rows = []
    for h, (m, lo, hi) in res.mu_hsg_summary.items():
        is_observed = h in observed_hsgs
        rows.append({
            "HSG": h,
            "출처": "📊 관측" if is_observed else "📚 prior 추정",
            "Sy 평균": f"{m:.3f}",
            "95% CI": f"[{lo:.3f}, {hi:.3f}]",
            "CI 폭": f"{hi - lo:.3f}",
            "면적비율": f"{profile.hsg_fractions.get(h, 0)*100:.1f}%",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(
        "📊 = 그 HSG 에 관정이 있어 데이터로 학습됨 · "
        "📚 = 관정 없음, prior 분포만으로 추정 (CI 폭이 ↑)."
    )

    # 관정별 posterior
    st.markdown("**관정별 posterior Sy**")
    rows = []
    for i, n in enumerate(res.well_names):
        rows.append({
            "관정": n,
            "HSG": res.hsgs[i],
            "대수층": res.aquifers[i],
            "Sy posterior": f"{res.sy_well_mean[i]:.3f}",
            "95% CI": f"[{res.sy_well_lo95[i]:.3f}, {res.sy_well_hi95[i]:.3f}]",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _render_registry_manager(wr):
    """관정/유역 등록·편집 UI."""
    with st.expander("🛠️ 관정 / 유역 관리 (추가·편집·삭제)", expanded=False):
        WELLS = wr.WELLS
        WATERSHEDS = wr.WATERSHEDS

        # 현재 등록 상태
        if WELLS:
            rows = []
            for name, w in WELLS.items():
                rows.append({
                    "관정명": name, "유역": w.watershed,
                    "lat": w.lat, "lon": w.lon,
                    "대수층": w.aquifer, "ASOS": w.nearest_kma,
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            st.info("등록된 관정이 없습니다.")

        tab_add, tab_edit, tab_ws = st.tabs([
            "➕ 새 관정 추가", "✏️ 편집 / 삭제", "🗂️ 유역명 변경",
        ])

        # ── 추가 ──
        with tab_add:
            from ui import coord_input
            # 좌표는 form 밖 — 라디오 토글이 즉시 반영되도록
            new_lat, new_lon = coord_input(
                default_lat=36.35, default_lon=127.37,
                key_prefix="ws_add",
            )
            with st.form("add_well_form", clear_on_submit=True):
                c1, c2 = st.columns(2)
                with c1:
                    new_name = st.text_input("관정명", placeholder="예: 대전유성")
                    new_ws = st.text_input(
                        "유역",
                        placeholder="예: 갑천 (기존 또는 새 이름)",
                        help="기존 유역명을 입력하면 그 유역에 추가됨",
                    )
                with c2:
                    new_aq = st.selectbox(
                        "대수층 타입",
                        options=["alluvial", "bedrock"],
                        index=1,
                        help="alluvial = 충적층, bedrock = 암반(풍화) 대수층",
                    )
                    new_kma = st.number_input(
                        "ASOS 기상관측소 ID",
                        min_value=100, max_value=300, value=133, step=1,
                        help="대전=133, 추풍령=135, 대구=143, 서울=108 등",
                    )
                ok = st.form_submit_button("➕ 등록", type="primary")
                if ok:
                    try:
                        info = wr.add_well(
                            new_name, new_lat, new_lon, new_ws,
                            aquifer=new_aq, nearest_kma=int(new_kma),
                        )
                        st.success(f"✅ 등록됨: {info.name} ({info.watershed})")
                        st.rerun()
                    except Exception as e:
                        st.error(f"실패: {e}")

        # ── 편집 / 삭제 ──
        with tab_edit:
            if not WELLS:
                st.info("등록된 관정 없음")
            else:
                tgt = st.selectbox("관정 선택", options=list(WELLS.keys()),
                                   key="edit_well_target")
                w = WELLS[tgt]
                # 좌표는 form 밖에 두어야 라디오 토글이 즉시 반영됨
                from ui import coord_input
                e_lat, e_lon = coord_input(
                    default_lat=float(w.lat), default_lon=float(w.lon),
                    key_prefix=f"edit_{tgt}",
                )
                with st.form(f"edit_well_{tgt}"):
                    c1, c2 = st.columns(2)
                    with c1:
                        e_ws = st.text_input("유역", value=w.watershed)
                        e_aq = st.selectbox(
                            "대수층",
                            options=["alluvial", "bedrock"],
                            index=0 if w.aquifer == "alluvial" else 1,
                        )
                    with c2:
                        e_kma = st.number_input(
                            "ASOS ID", value=int(w.nearest_kma),
                            step=1,
                        )
                    bsave, bdel = st.columns(2)
                    do_save = bsave.form_submit_button("💾 변경 저장")
                    do_del = bdel.form_submit_button("🗑️ 삭제", type="primary")
                    if do_save:
                        try:
                            wr.update_well(
                                tgt,
                                lat=e_lat, lon=e_lon,
                                watershed=e_ws,
                                aquifer=e_aq,
                                nearest_kma=int(e_kma),
                            )
                            st.success(f"✅ 변경됨: {tgt}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"실패: {e}")
                    if do_del:
                        ok_reg = wr.remove_well(tgt)
                        # 저장된 분석 결과 파일도 같이 제거
                        ok_store = False
                        try:
                            from well_results_store import delete as _wrs_del
                            ok_store = _wrs_del(tgt)
                        except Exception:
                            ok_store = False
                        if ok_reg or ok_store:
                            msg_parts = []
                            if ok_reg:
                                msg_parts.append("registry")
                            if ok_store:
                                msg_parts.append("결과 파일")
                            st.success(
                                f"🗑️ 삭제됨: {tgt} "
                                f"({' + '.join(msg_parts) if msg_parts else '레지스트리'})"
                            )
                            st.rerun()
                        else:
                            st.warning(f"⚠️ {tgt} 가 어디에도 없습니다.")

        # ── 유역명 변경 / 유역 삭제 ──
        with tab_ws:
            if not WATERSHEDS:
                st.info("등록된 유역 없음")
            else:
                # (1) 이름 변경
                with st.form("rename_ws"):
                    old = st.selectbox("기존 유역명",
                                        options=list(WATERSHEDS.keys()),
                                        key="rename_ws_old")
                    new = st.text_input("새 유역명")
                    ok2 = st.form_submit_button("🔄 일괄 변경")
                    if ok2 and new.strip():
                        n = wr.rename_watershed(old, new.strip())
                        st.success(f"✅ {n}개 관정의 유역명 변경: "
                                    f"{old} → {new}")
                        st.rerun()

                st.markdown("---")
                # (2) 유역 통째 삭제 (소속 관정 + 결과 파일 함께)
                with st.form("delete_ws"):
                    st.markdown("##### 🗑️ 유역 통째 삭제")
                    st.caption("선택한 유역에 속한 모든 관정과 분석 결과를 한 번에 삭제합니다.")
                    del_ws = st.selectbox(
                        "삭제할 유역",
                        options=list(WATERSHEDS.keys()),
                        key="delete_ws_target",
                    )
                    members = WATERSHEDS.get(del_ws, [])
                    st.write(f"속한 관정 ({len(members)}개): "
                              f"{', '.join(members) if members else '(없음)'}")
                    confirm = st.text_input(
                        f"확인을 위해 유역명 '{del_ws}'을(를) 그대로 입력하세요",
                        key="delete_ws_confirm",
                    )
                    do_del_ws = st.form_submit_button(
                        "🗑️ 유역 + 소속 관정 모두 삭제", type="primary",
                    )
                    if do_del_ws:
                        if confirm.strip() != del_ws:
                            st.error("유역명이 일치하지 않습니다. 안전을 위해 정확히 입력해 주세요.")
                        else:
                            try:
                                from well_results_store import delete as _wrs_del
                            except Exception:
                                _wrs_del = None
                            n_reg = n_store = 0
                            for w_name in list(members):
                                if wr.remove_well(w_name):
                                    n_reg += 1
                                if _wrs_del is not None:
                                    try:
                                        if _wrs_del(w_name):
                                            n_store += 1
                                    except Exception:
                                        pass
                            st.success(
                                f"🗑️ '{del_ws}' 삭제됨: "
                                f"관정 {n_reg}개 (registry), "
                                f"결과 파일 {n_store}개"
                            )
                            st.rerun()
