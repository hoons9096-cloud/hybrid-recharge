"""Tab 8 — 관측정 리포트 (현재 분석 중인 단일 관측정).

사이드바에서 로드한 자료에 대해 Tab 1~6에서 실행한 분석 결과를
session_state에서 직접 가져와 단일 HTML 리포트로 묶는다.

§3 Method comparison 추가됨 — hybrid-recharge + SCS-CN + FAO-56 일별 SWB 의
3-방법 비교 (Choi & Ahn 1998 표준에서 영감, FAO-56으로 현대화).
SCS-CN/FAO-56은 사이드 패널에서 옵션 켜면 동일 강수 시계열에 대해 자동
실행.

Tab 9 (Field 리포트, 합성 시나리오)와 명확히 구분.
"""
from __future__ import annotations

import os

import numpy as np
import streamlit as st

from ui import TabContext, has_v27_error


# ══════════════════════════════════════════════════════════════════════
# 토지이용 ↔ Kc preset 매핑 (UI 표시용)
# ══════════════════════════════════════════════════════════════════════
LAND_USE_TO_KC = {
    "혼합농경지":     "혼합농경지",
    "논":             "논",
    "밭(직선경작)":   "밭(직선경작)",
    "산림(good)":     "산림(활엽수림)",
    "산림(fair)":     "산림(활엽수림)",
    "초지/공한지":    "초지/공한지",
    "주거(저밀도)":   "초지/공한지",
    "주거(고밀도)":   "초지/공한지",
}


def render(tab, ctx: TabContext):
    """Render Tab 8 inside the given Streamlit tab container."""
    with tab:
        st.markdown("### 📋 관측정 리포트 (현재 분석 데이터)")
        st.caption(
            "Tab 1~6에서 실행한 분석 + (선택) SCS-CN / FAO-56 추가 방법을 묶어 "
            "단일 HTML 리포트로 생성.  Choi & Ahn 1998 표준의 3-방법 수렴 검증."
        )

        result_v27 = st.session_state.get("result_v27")
        if result_v27 is None or has_v27_error(result_v27):
            st.info("① **기본 분석을 먼저 실행하세요** (사이드바 → ▶ ① 기본 분석).")
            return

        if hasattr(result_v27, "to_dict"):
            result_v27 = result_v27.to_dict()

        # 가용 보조 분석
        uc_result = st.session_state.get("uc_result")
        bma_result = st.session_state.get("bma_result")
        kalman_sens = st.session_state.get("kalman_sensitivity")
        pump_result = st.session_state.get("pump_result")

        # ── 가용 분석 상태 (8개) ──
        st.markdown("#### 🔎 포함될 섹션")
        cols = st.columns(8)
        cols[0].markdown("**Core**\n\n✓")
        cols[1].markdown("**Plausibility**\n\n✓")
        cols[2].markdown("**3-Method**\n\n옵션↓")
        cols[3].markdown("**시계열**\n\n✓")
        cols[4].markdown("**Bootstrap**\n\n" + ("✓" if uc_result else "⊘ Tab 5"))
        cols[5].markdown("**BMA**\n\n" + ("✓" if bma_result else "⊘ Tab 1"))
        cols[6].markdown("**민감도**\n\n" + ("✓" if kalman_sens else "⊘ Tab 6"))
        cols[7].markdown("**펌핑**\n\n" + ("✓" if pump_result else "⊘ Tab 2"))

        st.markdown("---")

        # ── 사이트 / 토양 ──
        col_in1, col_in2 = st.columns(2)
        with col_in1:
            uploaded = st.session_state.get("uploaded_name", "")
            default_site = uploaded.replace(".txt", "").replace(".csv", "") \
                if uploaded else "Untitled-Well"
            site_name = st.text_input("사이트 이름 (리포트 헤더)",
                                      value=default_site)
        with col_in2:
            from soil_db import SOIL_NAMES_NUMBERED as SOIL_NAMES
            sn = int(ctx.sn_idx)
            soil_label = (
                f"{SOIL_NAMES[sn-1]} (sn={sn})"
                if 1 <= sn <= len(SOIL_NAMES) else f"sn={sn}"
            )
            st.text_input("토양 모델 (자동)", value=soil_label, disabled=True)

        # ── §3-Method 옵션 (확장 가능 패널) ──
        st.markdown("#### ⚙️ §3 Method comparison 옵션")
        st.caption(
            "체크하면 같은 강수 시계열에 대해 SCS-CN / FAO-56을 추가 실행하고 "
            "3-방법 수렴을 §3에 표시합니다.  WTF만 보고 싶으면 모두 끄세요."
        )

        run_scs, run_fao56, lu, station_id, runoff_frac, api_key = \
            _render_method_options(ctx)

        st.markdown("---")

        # ── 빌드 버튼 ──
        col_btn1, col_btn2 = st.columns([1, 2])
        with col_btn1:
            build_btn = st.button(
                "▶ 리포트 빌드",
                type="primary",
                use_container_width=True,
                key="btn_well_report_build",
            )
        with col_btn2:
            clear_btn = st.button("🗑 초기화", use_container_width=True,
                                  key="btn_well_report_clear")
        if clear_btn:
            for k in ("well_report_html", "well_report_meta",
                      "well_scs_result", "well_fao56_result",
                      "well_kma_data"):
                st.session_state.pop(k, None)
            st.rerun()

        if build_btn:
            with st.spinner("리포트 생성 중..."):
                _build_and_cache(
                    result_v27=result_v27, ctx=ctx,
                    site_name=site_name, soil_label=soil_label,
                    uc_result=uc_result, bma_result=bma_result,
                    kalman_sens=kalman_sens, pump_result=pump_result,
                    run_scs=run_scs, run_fao56=run_fao56,
                    land_use=lu, station_id=station_id,
                    runoff_frac=runoff_frac, api_key=api_key,
                )

        # ── 결과 표시 ──
        html = st.session_state.get("well_report_html")
        if html is None:
            st.info("위 '▶ 리포트 빌드'를 눌러 리포트를 생성하세요.")
            return

        _render_results(html, result_v27, uc_result, bma_result, kalman_sens)


# ══════════════════════════════════════════════════════════════════════
# 옵션 패널
# ══════════════════════════════════════════════════════════════════════
def _render_method_options(ctx: TabContext):
    """SCS-CN / FAO-56 옵션 입력 패널.  Returns 모든 입력값 tuple."""
    from kma_adapter import KMA_STATIONS, list_stations

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**SCS-CN (개량)**")
        run_scs = st.checkbox("SCS-CN 추가 실행", value=True, key="opt_run_scs")
        from scs_cn import LAND_USE_CN
        lu_options = list(LAND_USE_CN.keys())
        lu = st.selectbox(
            "관정 주변 토지이용 (~500m)",
            options=lu_options,
            index=lu_options.index("혼합농경지"),
            help="WTF는 점-기반이므로 SCS-CN도 관정 주변값으로 일치시킴",
        )

    with col_b:
        st.markdown("**FAO-56 일별 SWB**")
        run_fao56 = st.checkbox("FAO-56 추가 실행 (기온 자료 필요)",
                                value=True, key="opt_run_fao56")
        stations = list_stations()
        station_labels = [f"{s['stn_id']} {s['name']}" for s in stations]
        default_idx = next(
            (i for i, s in enumerate(stations) if s["stn_id"] == 135), 0,
        )
        sel = st.selectbox(
            "기상관측소 (KMA ASOS)",
            options=range(len(stations)),
            format_func=lambda i: station_labels[i],
            index=default_idx,
            help="관정과 가장 가까운 ASOS — 김천=추풍령(135), 대전=대전(133)",
        )
        station_id = stations[sel]["stn_id"]

    runoff_frac = st.slider(
        "FAO-56 표면유출 비율 (P 중)",
        min_value=0.0, max_value=0.50, value=0.20, step=0.05,
        help="한국 몬순 일반 0.15–0.30 (지형·토양에 따라).  0이면 SCS-CN과 동치",
    )

    api_key = os.environ.get("KMA_API_KEY", "")
    if not api_key and run_fao56:
        api_key = st.text_input(
            "KMA APIHub 인증키 (환경변수 KMA_API_KEY 미설정 시)",
            type="password",
            help="https://apihub.kma.go.kr 에서 발급 + ASOS 일자료 활용신청 필요",
        )
    return run_scs, run_fao56, lu, station_id, runoff_frac, api_key


# ══════════════════════════════════════════════════════════════════════
# 빌드 파이프라인
# ══════════════════════════════════════════════════════════════════════
def _build_and_cache(
    *, result_v27, ctx, site_name, soil_label,
    uc_result, bma_result, kalman_sens, pump_result,
    run_scs, run_fao56, land_use, station_id, runoff_frac, api_key,
):
    from evaluation.well_report import build_well_html_report

    # WTF 강수 시계열 추출 (m/day → mm/day)
    po_shifted = np.asarray(result_v27.get("po_shifted", []), dtype=float)
    if po_shifted.size == 0:
        po_shifted = np.asarray(result_v27.get("po", []), dtype=float)
    P_mm = po_shifted * 1000.0  # mm/day
    n_days = len(P_mm)

    if n_days == 0:
        st.error("강수 시계열이 비어 있습니다 — Tab 1을 먼저 실행하세요.")
        return

    # 데이터 시작 DOY — KMA 가져올 때와 동일 가정
    from datetime import date, timedelta
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=n_days - 1)
    start_doy = start.timetuple().tm_yday

    # SCS-CN 실행
    scs_result = None
    if run_scs:
        scs_result = _run_scs_cn(P_mm, ctx.sn_idx, land_use, start_doy)

    # FAO-56 실행 — KMA에서 기온 자료 fetch 필요
    fao56_result = None
    if run_fao56:
        fao56_result = _run_fao56(
            P_mm=P_mm, n_days=n_days,
            sn_idx=ctx.sn_idx, land_use=land_use,
            station_id=station_id, runoff_frac=runoff_frac,
            api_key=api_key,
        )

    # 리포트 빌드
    html = build_well_html_report(
        result_v27=result_v27,
        site_name=site_name,
        soil_label=soil_label,
        uc_result=uc_result,
        bma_result=bma_result,
        kalman_sens=kalman_sens,
        pump_result=pump_result,
        scs_result=scs_result,
        fao56_result=fao56_result,
    )
    st.session_state["well_report_html"] = html
    st.session_state["well_report_meta"] = {
        "site_name": site_name, "soil_label": soil_label,
        "land_use": land_use, "station_id": station_id,
    }
    st.session_state["well_scs_result"] = scs_result
    st.session_state["well_fao56_result"] = fao56_result


def _run_scs_cn(P_mm: np.ndarray, sn_idx: int, land_use: str, start_doy: int = 1):
    """SCS-CN 실행 — 결과 또는 None 반환."""
    try:
        from scs_cn import (
            estimate_recharge_scs_cn, derive_cn_from_soil_db,
            soil_group_from_texture,
        )
        from soil_db import SOIL_DB

        soil = SOIL_DB[int(sn_idx)]
        group = soil_group_from_texture(soil.texture_group)
        cn, _ = derive_cn_from_soil_db(int(sn_idx), land_use=land_use)

        result = estimate_recharge_scs_cn(
            P_daily_mm=P_mm,
            CN=cn,
            soil_hydro_group=group,
            land_use=land_use,
            texture_group=soil.texture_group,
            start_doy=start_doy,
        )
        st.success(
            f"✓ SCS-CN 완료 — CN = {cn:.0f} (group {group}, {land_use}), "
            f"함양율 = {result.recharge_ratio_pct:.2f}% "
            f"(±{result.cn_uncertainty_band_pct/2:.2f}%p)"
        )
        return result
    except Exception as e:
        st.warning(f"SCS-CN 실패: {e}")
        return None


def _run_fao56(
    *, P_mm, n_days, sn_idx, land_use, station_id, runoff_frac, api_key,
):
    """FAO-56 실행 — KMA에서 기온 fetch + SWB.  결과 또는 None."""
    try:
        from kma_adapter import (
            fetch_kma_daily, fetch_mock_korean_climate, KMA_STATIONS,
        )
        from fao56_swb import estimate_recharge_fao56
        from soil_db import SOIL_DB

        # 강수 기간 — 현재는 입력 P_mm 길이만큼 mock으로 매핑
        # (실 데이터 통합 시 날짜 컬럼 사용 필요)
        # 임시: 마지막 날짜 = 오늘 기준 n_days 전부터
        from datetime import date, timedelta
        end = date.today() - timedelta(days=1)
        start = end - timedelta(days=n_days - 1)
        start_doy = start.timetuple().tm_yday

        # KMA fetch (auth_key 우선) — 실패 시 mock으로 fallback
        kma = None
        if api_key:
            try:
                kma = fetch_kma_daily(
                    stn_id=station_id,
                    start_date=start.isoformat(),
                    end_date=end.isoformat(),
                    auth_key=api_key,
                )
                st.success(f"✓ KMA fetch — {kma.stn_name} ({kma.n_days}일, "
                           f"결측 보간 T:{kma.n_missing_T} P:{kma.n_missing_P})")
            except Exception as e:
                st.warning(f"KMA API 실패 → mock 데이터로 fallback: {e}")
        if kma is None:
            kma = fetch_mock_korean_climate(
                stn_id=station_id,
                start_date=start.isoformat(),
                end_date=end.isoformat(),
            )
            st.info(f"ℹ️ Mock 기상자료 사용 — {kma.stn_name} ({kma.n_days}일).  "
                    f"실 API 활용신청 후 자동 전환")

        # 기온 길이가 P_mm과 다르면 짧은 쪽으로 정렬
        n = min(n_days, kma.n_days)
        Tm = kma.Tmean_C[:n]; Tx = kma.Tmax_C[:n]; Tn = kma.Tmin_C[:n]
        P_use = P_mm[:n]

        soil = SOIL_DB[int(sn_idx)]
        # FAO-56 작물계수 매핑
        kc_lu = LAND_USE_TO_KC.get(land_use, "혼합농경지")

        result = estimate_recharge_fao56(
            P_daily_mm=P_use,
            Tmean_C=Tm, Tmax_C=Tx, Tmin_C=Tn,
            lat_deg=kma.lat_deg,
            texture_group=soil.texture_group,
            land_use=kc_lu,
            runoff_fraction=float(runoff_frac),
            start_doy=start_doy,    # 실제 시작 일자 (Ra 및 Kc 곡선에 필수)
        )
        st.success(
            f"✓ FAO-56 완료 — ETo {result.ETo_annual_mm:.0f} mm/yr, "
            f"ETa {result.ETa_annual_mm:.0f} mm/yr, "
            f"함양율 = {result.recharge_ratio_pct:.2f}%"
        )
        st.session_state["well_kma_data"] = kma
        return result
    except Exception as e:
        st.warning(f"FAO-56 실패: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════
# 결과 렌더링
# ══════════════════════════════════════════════════════════════════════
def _render_results(html, result_v27, uc_result, bma_result, kalman_sens):
    """리포트 다운로드 + 인라인 figure 미리보기."""
    meta = st.session_state["well_report_meta"]
    scs_result = st.session_state.get("well_scs_result")
    fao56_result = st.session_state.get("well_fao56_result")
    size_kb = len(html.encode("utf-8")) / 1024

    st.markdown("---")

    # 헤드라인 — ET-반영 수렴 + SCS-CN 보조 분리 표시
    if scs_result is not None or fao56_result is not None:
        from evaluation.well_report import (
            _collect_method_estimates, _convergence_verdict, _split_by_category,
        )
        po = np.asarray(result_v27.get("po_shifted", []), dtype=float)
        n = len(po)
        P_annual_mm = (
            float(np.nansum(po)) * 1000.0 / max(n / 365.25, 1.0)
        ) if n > 0 else 0.0
        estimates = _collect_method_estimates(
            result_v27, scs_result, fao56_result, P_annual_mm=P_annual_mm,
        )
        conv = _convergence_verdict(estimates)
        et_aware, infil = _split_by_category(estimates)

        # ── 권장 보고값 (있으면) ──
        st.markdown("#### 🎯 권장 보고값")
        rec_pct = conv.get("recommended_pct", 0.0)
        rec_mm = rec_pct / 100.0 * P_annual_mm
        if conv["verdict"] == "converged":
            primary_names = ", ".join(conv["primary_names"])
            st.success(
                f"### **{rec_pct:.1f}%** ({rec_mm:.0f} mm/yr)\n\n"
                f"**ET-반영 수렴**: {primary_names} median, "
                f"범위 [{conv['min_pct']:.1f}, {conv['max_pct']:.1f}]%, "
                f"상대 폭 {conv['rel_spread']*100:.1f}%"
            )
        elif conv["verdict"] == "single":
            st.warning(
                f"### **{rec_pct:.1f}%** ({rec_mm:.0f} mm/yr)\n\n"
                f"**단일 ET-반영 추정**: {conv['primary_names'][0]}.  "
                f"두 번째 ET-반영 방법(FAO-56) 추가 시 수렴 검증 가능."
            )
        elif conv["verdict"] == "diverged":
            st.error(
                f"### ⚠ ET-반영 방법 발산\n\n"
                f"median {rec_pct:.1f}%, 범위 "
                f"[{conv['min_pct']:.1f}, {conv['max_pct']:.1f}]%.  "
                f"가정(Sy, runoff_fraction, land_use) 차이 추적 필요."
            )

        # ── ET-반영 카드들 (Primary) ──
        st.markdown("##### Primary — ET-반영 (실제 함양)")
        if et_aware:
            cols = st.columns(len(et_aware))
            for i, e in enumerate(et_aware):
                cols[i].metric(
                    e["name"],
                    f"{e['rech_pct']:.1f}%",
                    delta=f"{e['rech_mm']:.0f} mm/yr",
                    delta_color="off",
                )

        # ── SCS-CN 보조 카드 (Supplementary) ──
        if infil:
            st.markdown("##### Supplementary — 침투 상한 (ET 미반영)")
            scs_cols = st.columns(2)
            scs = infil[0]
            scs_cols[0].metric(
                f"{scs['name']} (참고)",
                f"{scs['rech_pct']:.1f}%",
                delta=f"{scs['rech_mm']:.0f} mm/yr",
                delta_color="off",
            )
            scs_cols[1].info(
                "ⓘ SCS-CN은 ET 미반영 → *침투량*을 보고합니다. "
                "Primary 추정보다 항상 큼 (Choi & Ahn 1998 baseline)."
            )

    st.markdown("#### 💾 다운로드")
    d1, d2 = st.columns([1, 2])
    with d1:
        fname = f"well_report_{meta['site_name']}.html".replace(" ", "_")
        st.download_button(
            label="⬇ HTML 다운로드",
            data=html.encode("utf-8"),
            file_name=fname,
            mime="text/html",
            type="primary",
            use_container_width=True,
        )
    with d2:
        st.caption(
            f"파일 크기: ~{size_kb:.0f} KB · "
            f"브라우저에서 열고 `Cmd+P → PDF로 저장`."
        )

    # ── 인라인 figure 미리보기 ──
    with st.expander("🖼 Figure 미리보기", expanded=False):
        from evaluation.well_report import (
            plot_well_time_series, plot_well_recharge_cumulative,
            plot_uncertainty_histogram, plot_bma_posterior,
            plot_sensitivity_tornado, plot_method_comparison,
            _collect_method_estimates,
        )

        if scs_result is not None or fao56_result is not None:
            po = np.asarray(result_v27.get("po_shifted", []), dtype=float)
            n = len(po)
            P_annual_mm = (
                float(np.nansum(po)) * 1000.0 / max(n / 365.25, 1.0)
            ) if n > 0 else 0.0
            estimates = _collect_method_estimates(
                result_v27, scs_result, fao56_result, P_annual_mm=P_annual_mm,
            )
            fig = plot_method_comparison(estimates, P_annual_mm)
            if fig is not None:
                st.markdown("**Figure — Method comparison**")
                st.pyplot(fig, use_container_width=True)

        st.markdown("**Figure 1 — Time series**")
        st.pyplot(plot_well_time_series(result_v27), use_container_width=True)

        st.markdown("**Figure 2 — Cumulative water balance**")
        st.pyplot(plot_well_recharge_cumulative(result_v27),
                  use_container_width=True)

        if uc_result is not None:
            fig = plot_uncertainty_histogram(uc_result)
            if fig is not None:
                st.markdown("**Bootstrap distribution**")
                st.pyplot(fig, use_container_width=True)

        if bma_result is not None:
            fig = plot_bma_posterior(bma_result)
            if fig is not None:
                st.markdown("**BMA posterior**")
                st.pyplot(fig, use_container_width=True)

        if kalman_sens is not None:
            fig = plot_sensitivity_tornado(kalman_sens)
            if fig is not None:
                st.markdown("**Sensitivity tornado**")
                st.pyplot(fig, use_container_width=True)
