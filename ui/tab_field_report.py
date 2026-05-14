"""Tab 8 — Field-mode 통합 리포트.

합성 시나리오(S1~S5)에서 선택한 방법(Lumped / Soil-weighted / EnKF)로
field-mode 일관성 리포트를 생성한다.  정답(true recharge)을 사용하지 않는
4가지 지표(spread, plausibility, soil-class coherence, well consistency)와
이를 시각화한 그림 5개를 단일 HTML로 묶어 다운로드 가능하게 제공한다.

향후 김천/대덕 등 실측 자료가 들어오면 동일 인터페이스로 확장 가능하도록
설계되었다 (observations dict + domain 객체만 맞으면 됨).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import streamlit as st

from ui import TabContext


# 시나리오 설명 (UI 표시용)
_SCENARIO_DESC = {
    "S1": "균질 / 높은 관측정 / 낮은 노이즈",
    "S2": "약한 불균질 / 높은 관측정 / 낮은 노이즈",
    "S3": "강한 불균질 / 높은 관측정 / 낮은 노이즈",
    "S4": "강한 불균질 / 낮은 관측정 / 낮은 노이즈",
    "S5": "강한 불균질 / 높은 관측정 / 높은 노이즈",
}

_METHOD_KEYS = {
    "Lumped": "lumped",
    "Soil-weighted": "soil",
    "EnKF": "enkf",
}


def render(tab, ctx: TabContext):
    """Render Tab 8 inside the given Streamlit tab container."""
    with tab:
        st.markdown("### 📑 Field-mode 통합 리포트")
        st.caption(
            "정답(true recharge)을 모르는 가정에서 추정 결과의 *내적 일관성*을 "
            "4가지 지표로 평가하고 단일 HTML 리포트로 묶습니다. "
            "현재는 합성 시나리오(S1~S5) 기준이며, 실측 데이터 어댑터는 추후 추가 예정."
        )

        # ── 설정 영역 ──
        col_cfg, col_run = st.columns([2, 1])

        with col_cfg:
            scenario = st.selectbox(
                "시나리오 선택",
                options=list(_SCENARIO_DESC.keys()),
                format_func=lambda s: f"{s} — {_SCENARIO_DESC[s]}",
                index=2,  # S3 기본
            )
            method_labels = st.multiselect(
                "비교할 방법",
                options=list(_METHOD_KEYS.keys()),
                default=["Lumped", "Soil-weighted"],
                help="EnKF는 계산 시간이 더 걸립니다 (앙상블 기반).",
            )
            site_name = st.text_input(
                "사이트 이름 (리포트 헤더 표시용)",
                value=f"Synthetic-{scenario}",
            )

        with col_run:
            st.markdown("####  ")  # 여백 맞추기
            run = st.button(
                "▶ 리포트 생성",
                type="primary",
                use_container_width=True,
                disabled=(len(method_labels) == 0),
            )
            if len(method_labels) == 0:
                st.caption("⚠ 방법을 1개 이상 선택하세요.")
            clear = st.button(
                "🗑 결과 초기화",
                use_container_width=True,
            )
            if clear:
                for k in ("field_report_html", "field_report_results",
                          "field_report_meta"):
                    st.session_state.pop(k, None)
                st.rerun()

        if run:
            with st.spinner("합성 데이터 생성 + 함양 추정 + 리포트 빌드 중..."):
                _run_pipeline(scenario, method_labels, site_name)

        # ── 결과 표시 ──
        if "field_report_html" not in st.session_state:
            st.info("위 설정 후 '▶ 리포트 생성'을 눌러주세요.")
            return

        meta = st.session_state["field_report_meta"]
        results = st.session_state["field_report_results"]
        html = st.session_state["field_report_html"]

        st.markdown("---")
        _render_summary(meta, results)
        st.markdown("---")
        _render_inline_figures(meta, results)
        st.markdown("---")
        _render_download(html, meta)


# ══════════════════════════════════════════════════════════════════════
# 파이프라인 실행
# ══════════════════════════════════════════════════════════════════════
def _run_pipeline(scenario: str, method_labels: list, site_name: str):
    """합성 시나리오 → 방법 실행 → field_report HTML 빌드 → session_state 저장."""
    from synthetic.generate_domain import generate_domain, DomainConfig
    from synthetic.generate_data import generate_data
    from evaluation.field_report import build_html_report

    # 1. 도메인 + 데이터
    cfg_factory = getattr(DomainConfig, scenario)
    domain = generate_domain(cfg_factory())
    data = generate_data(domain)
    observations = {
        "P": data.P, "ET": data.ET,
        "ho_obs": data.ho_obs, "well_soil_types": data.well_soil_types,
    }
    P_annual_mm = float(np.sum(data.P)) * 1000.0 / (data.n_days / 365.0)

    # 2. 선택된 방법 실행
    method_results: Dict[str, np.ndarray] = {}
    for label in method_labels:
        key = _METHOD_KEYS[label]
        if key == "lumped":
            from methods.wtf_lumped import estimate_recharge
            method_results[label] = estimate_recharge(domain, observations)
        elif key == "soil":
            from methods.wtf_soil_weighted import estimate_recharge
            method_results[label] = estimate_recharge(domain, observations)
        elif key == "enkf":
            from methods.wtf_enkf_spatial import estimate_recharge
            method_results[label] = estimate_recharge(domain, observations)

    # 3. HTML 리포트 빌드
    html = build_html_report(
        method_results=method_results,
        observations=observations,
        domain=domain,
        P_annual_mm=P_annual_mm,
        site_name=site_name,
    )

    # 4. session_state 캐시 (figure 함수에서도 재사용)
    st.session_state["field_report_html"] = html
    st.session_state["field_report_results"] = method_results
    st.session_state["field_report_meta"] = {
        "scenario": scenario,
        "site_name": site_name,
        "P_annual_mm": P_annual_mm,
        "n_days": data.n_days,
        "n_wells": data.n_wells,
        "domain": domain,
        "observations": observations,
        "true_recharge_annual": data.true_recharge_annual,  # 참고용 (UI에만)
    }


# ══════════════════════════════════════════════════════════════════════
# 요약 표시
# ══════════════════════════════════════════════════════════════════════
def _render_summary(meta, results):
    """상단 요약 메트릭."""
    from evaluation.field_metrics import (
        between_method_spread,
        plausibility_check,
    )

    st.markdown("#### 📊 요약")

    n_methods = len(results)
    P = meta["P_annual_mm"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("시나리오", meta["scenario"])
    col2.metric("연 강수량", f"{P:.0f} mm/yr")
    col3.metric("관측정 수", meta["n_wells"])
    col4.metric("관측 일수", meta["n_days"])

    # 방법별 평균 R + R/P
    st.markdown("##### 방법별 평균 함양량")
    cols = st.columns(max(n_methods, 1))
    for i, (name, R) in enumerate(results.items()):
        rep = plausibility_check(R, P, method_name=name)
        with cols[i]:
            st.metric(
                name,
                f"{rep.mean_R:.1f} mm/yr",
                delta=f"R/P = {rep.R_over_P*100:.1f}%",
                delta_color="off",
            )
            if rep.flags:
                st.caption(f"⚠ {len(rep.flags)} flag")
            else:
                st.caption("✓ plausibility OK")

    # 방법 간 spread
    if n_methods >= 2:
        sp = between_method_spread(results)
        st.markdown("##### Epistemic uncertainty (방법 선택만의 불확실성)")
        c1, c2, c3 = st.columns(3)
        c1.metric("도메인 평균 R", f"{sp.domain_mean:.1f} mm/yr")
        c2.metric("셀별 std (방법 간)", f"{sp.domain_mean_std:.1f} mm/yr")
        c3.metric("평균 CV", f"{sp.domain_mean_cv:.3f}")

    # 참고: 합성 시나리오라 true 알고 있음
    true_R = meta.get("true_recharge_annual")
    if true_R is not None:
        with st.expander("🔍 참고: 합성 데이터 정답(true recharge)", expanded=False):
            st.caption(
                "현재는 합성 시나리오이므로 진짜 함양량을 알고 있습니다. "
                "실측에서는 이 값이 없으며, 위 4지표만으로 결과를 방어하게 됩니다."
            )
            st.metric("True mean R", f"{float(np.mean(true_R)):.1f} mm/yr")


# ══════════════════════════════════════════════════════════════════════
# 인라인 figure
# ══════════════════════════════════════════════════════════════════════
def _render_inline_figures(meta, results):
    """리포트의 figure 5개를 streamlit 안에서 직접 표시."""
    from evaluation.field_report import (
        plot_method_comparison_maps,
        plot_method_spread_map,
        plot_recharge_histograms,
        plot_soil_class_boxplots,
        plot_well_consistency_scatter,
    )
    from evaluation.field_metrics import between_method_spread

    domain = meta["domain"]
    obs = meta["observations"]
    P = meta["P_annual_mm"]

    st.markdown("#### 🖼 Figures")

    # 1. comparison maps
    with st.expander("Figure 1 — 방법별 함양량 맵", expanded=True):
        st.pyplot(plot_method_comparison_maps(results, domain),
                  use_container_width=True)

    # 2. spread map (>=2 methods)
    if len(results) >= 2:
        with st.expander("Figure 2 — 방법 간 spread (epistemic uncertainty)",
                         expanded=True):
            sp = between_method_spread(results)
            st.pyplot(plot_method_spread_map(sp, domain),
                      use_container_width=True)
    else:
        st.info("Figure 2 (spread)는 방법 2개 이상에서 활성화됩니다.")

    # 3. histograms
    with st.expander("Figure 3 — 분포 히스토그램 + plausibility 참조선",
                     expanded=False):
        st.pyplot(plot_recharge_histograms(results, P),
                  use_container_width=True)

    # 4. boxplots
    with st.expander("Figure 4 — 토양 클래스별 박스플롯", expanded=False):
        st.pyplot(plot_soil_class_boxplots(results, domain.soil_map),
                  use_container_width=True)

    # 5. well scatter
    with st.expander("Figure 5 — 관측정별 self-consistency", expanded=False):
        st.pyplot(plot_well_consistency_scatter(results, obs, domain),
                  use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# 다운로드
# ══════════════════════════════════════════════════════════════════════
def _render_download(html: str, meta):
    """HTML 다운로드 버튼."""
    st.markdown("#### 💾 리포트 다운로드")
    fname = f"field_report_{meta['scenario']}_{meta['site_name']}.html"
    fname = fname.replace(" ", "_")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.download_button(
            label="⬇ HTML 다운로드",
            data=html.encode("utf-8"),
            file_name=fname,
            mime="text/html",
            type="primary",
            use_container_width=True,
        )
    with col2:
        st.caption(
            f"파일 크기: ~{len(html.encode('utf-8'))/1024:.0f} KB · "
            f"브라우저에서 열고 `Cmd+P → PDF로 저장`하면 PDF로 변환됩니다."
        )

    with st.expander("📄 텍스트 요약 미리보기", expanded=False):
        from evaluation.field_metrics import field_summary
        text = field_summary(
            st.session_state["field_report_results"],
            meta["observations"],
            meta["domain"],
            meta["P_annual_mm"],
        )
        st.code(text, language="text")
