"""Tab 5 — 불확실성 분석 (Bootstrap Uncertainty)."""

import numpy as np
import streamlit as st

from ui import TabContext, has_v27_error
from uncertainty import bootstrap_uncertainty
from core_sim_v27 import get_kalman_uncertainty, propagate_kalman_recharge_uncertainty


def render(tab, ctx: TabContext):
    """Render Tab 5 inside the given Streamlit tab container."""
    with tab:
        result_v27 = st.session_state.get("result_v27")
        if result_v27 is None or has_v27_error(result_v27):
            st.info("① 기본 분석을 먼저 실행하세요. 불확실성 분석은 최적 파라미터를 기반으로 합니다.")
            return
        if ctx.file_path_to_send == "DEMO":
            st.warning("불확실성 분석은 실측 데이터 파일이 필요합니다. 데이터를 업로드하세요.")
            return

        st.markdown("### 📐 블록 부트스트랩 불확실성 분석")
        st.caption(
            "Efron & Tibshirani (1993) 블록 부트스트랩으로 함양율 신뢰구간을 추정합니다. "
            "블록 길이는 토양별 tau (자기상관 시간규모)를 사용합니다."
        )

        uc_col1, uc_col2 = st.columns(2)
        with uc_col1:
            n_boot = st.number_input("부트스트랩 반복 수", value=200, min_value=20,
                                     max_value=2000, step=50,
                                     help="BCa CI 안정성: B≥200 실무, B≥1000 논문 수준 (Efron & Tibshirani, 1993).")
        with uc_col2:
            ci_level = st.selectbox("신뢰수준", [0.90, 0.95, 0.99], index=1)

        uc_soil = int(ctx.sn_idx)
        run_uc = st.button("▶ 불확실성 분석 실행", key="btn_uc")

        if run_uc:
            with st.spinner(f"블록 부트스트랩 (n={n_boot}) 실행 중... (1~3분 소요)"):
                try:
                    _opt_k = result_v27.get("opt_k")
                    _opt_z = result_v27.get("opt_z")
                    _opt_lag = result_v27.get("opt_lag")
                    _opt_rho = result_v27.get("opt_rho")
                    _opt_alpha = result_v27.get("opt_alpha")
                    uc_result = bootstrap_uncertainty(
                        file_path=ctx.file_path_to_send,
                        soil_num=uc_soil,
                        k_init=float(_opt_k if _opt_k is not None else ctx.k_val),
                        z_init=float(_opt_z if _opt_z is not None else ctx.z_val),
                        q_val=float(ctx.q_val), r_val=float(ctx.r_val),
                        rc_val=float(ctx.rc_val), sens_val=float(ctx.sens_val_for_send),
                        n_bootstrap=int(n_boot),
                        confidence=float(ci_level),
                        opt_k=float(_opt_k) if _opt_k is not None else None,
                        opt_z=float(_opt_z) if _opt_z is not None else None,
                        opt_lag=int(_opt_lag) if _opt_lag is not None else None,
                        opt_rho=float(_opt_rho) if _opt_rho is not None else None,
                        opt_alpha=float(_opt_alpha) if _opt_alpha is not None else None,
                    )
                    st.session_state["uc_result"] = uc_result
                except Exception as e:
                    st.error(f"불확실성 분석 실패: {e}")

        uc_result = st.session_state.get("uc_result")
        if uc_result is not None:
            _render_bootstrap_results(uc_result, ci_level)

            # ── Kalman covariance-based recharge uncertainty ──
            _render_kalman_uncertainty(result_v27)


def _render_bootstrap_results(uc_result, ci_level):
    st.markdown("#### 함양율 신뢰구간")
    _baseline = getattr(uc_result, 'rech_baseline', 0.0)
    if _baseline > 0:
        u0, u1, u2, u3 = st.columns(4)
        u0.metric("기본 함양율", f"{_baseline:.2f}%")
        u1.metric("부트스트랩 평균", f"{uc_result.rech_mean:.2f}%",
                  delta=f"{uc_result.rech_mean - _baseline:+.2f}%p")
    else:
        u1, u2, u3 = st.columns(3)
        u1.metric("평균 함양율", f"{uc_result.rech_mean:.2f}%")
    u2.metric(f"{ci_level*100:.0f}% CI 하한", f"{uc_result.rech_ci_lower:.2f}%")
    u3.metric(f"{ci_level*100:.0f}% CI 상한", f"{uc_result.rech_ci_upper:.2f}%")

    _bias_pct = getattr(uc_result, 'bootstrap_bias_pct', 0.0)
    if _bias_pct > 20.0:
        st.warning(
            f"⚠️ 부트스트랩 바이어스 {_bias_pct:.1f}% 감지. "
            f"기본 함양율({_baseline:.2f}%)과 부트스트랩 평균({uc_result.rech_mean:.2f}%)의 "
            f"괴리가 큽니다. 부트스트랩 반복수(B) 증가 또는 매개변수 경계 조건 확인을 권장합니다."
        )
    elif _bias_pct > 10.0:
        st.info(
            f"ℹ️ 부트스트랩 바이어스 {_bias_pct:.1f}%. "
            f"기본({_baseline:.2f}%) vs 부트스트랩({uc_result.rech_mean:.2f}%). "
            f"비선형 매개변수 변환에 의한 정상적 편향일 수 있습니다."
        )

    st.markdown("#### 파라미터 불확실성")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("RMSE", f"{uc_result.rmse_mean:.4f} ± {uc_result.rmse_std:.4f}")
    p2.metric("k", f"{uc_result.k_mean:.4f} ± {uc_result.k_std:.4f}")
    p3.metric("z", f"{uc_result.z_mean:.2f} ± {uc_result.z_std:.2f}")
    p4.metric("Sy", f"{uc_result.sy_mean:.4f} ± {uc_result.sy_std:.4f}")

    ci_width = uc_result.rech_ci_upper - uc_result.rech_ci_lower
    if ci_width < 5.0:
        st.success(f"신뢰구간 폭 {ci_width:.1f}%p → 함양율 추정이 안정적입니다.")
    elif ci_width < 15.0:
        st.warning(f"신뢰구간 폭 {ci_width:.1f}%p → 중간 수준 불확실성. 데이터 기간 연장 권장.")
    else:
        st.error(f"신뢰구간 폭 {ci_width:.1f}%p → 높은 불확실성. 데이터 품질·양 검토 필요.")


def _render_kalman_uncertainty(result_v27):
    kf_extras = get_kalman_uncertainty()
    if not kf_extras:
        return

    st.markdown("#### Kalman 필터 기반 함양 불확실성")
    st.caption(
        "RTS 평활기 공분산 P(h)에서 전파된 이벤트별 함양 불확실성입니다. "
        "부트스트랩(매개변수 불확실성)과 독립적인 관측 조건부 불확실성을 제공합니다."
    )
    rech_arr = np.array(result_v27["rech"])
    sy_val = result_v27.get("Sy_eff", 0.05)
    rech_sigma = propagate_kalman_recharge_uncertainty(
        rech_arr, sy_val, kf_extras["P_h_var"],
    )
    event_idx = np.where(rech_arr > 0)[0]
    if len(event_idx) > 0:
        total_rain = max(float(np.sum(result_v27["po"])), 1e-9)
        rech_pct = rech_arr[event_idx] / total_rain * 100
        sigma_pct = rech_sigma[event_idx] / total_rain * 100
        avg_sigma = float(np.mean(sigma_pct))
        max_sigma = float(np.max(sigma_pct))
        c1, c2 = st.columns(2)
        c1.metric("평균 이벤트 σ(R)", f"±{avg_sigma:.2f}%p")
        c2.metric("최대 이벤트 σ(R)", f"±{max_sigma:.2f}%p")
