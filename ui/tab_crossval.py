"""Tab 4 — 교차검증 (Cross-Validation)."""

import pandas as pd
import streamlit as st

from ui import TabContext, has_v27_error
from cross_validation import split_sample_test, temporal_kfold_cv


def render(tab, ctx: TabContext):
    """Render Tab 4 inside the given Streamlit tab container."""
    with tab:
        result_v27 = st.session_state.get("result_v27")
        if result_v27 is None or has_v27_error(result_v27):
            st.info("① 기본 분석을 먼저 실행하세요. 교차검증은 기본 분석 파라미터를 사용합니다.")
            return
        if ctx.file_path_to_send == "DEMO":
            st.warning("교차검증은 실측 데이터 파일이 필요합니다. 데이터를 업로드하세요.")
            return

        st.markdown("### 🔄 시간적 교차검증")
        st.caption(
            "모델 과적합 여부를 진단합니다. "
            "**Split-sample** (Klemeš, 1986): 전반/후반 분할 검증. "
            "**Temporal k-fold**: 연속 시간 블록으로 k-fold CV."
        )

        cv_col1, cv_col2 = st.columns(2)
        with cv_col1:
            cv_method = st.selectbox("검증 방식", ["Split-sample (Klemeš 1986)", "Temporal 3-fold"])
        with cv_col2:
            cv_soil = int(ctx.sn_idx)

        run_cv = st.button("▶ 교차검증 실행", key="btn_cv")

        if run_cv:
            with st.spinner("교차검증 실행 중... (데이터 길이에 따라 1~3분 소요)"):
                try:
                    if "Split" in cv_method:
                        cv_report = split_sample_test(
                            ctx.file_path_to_send,
                            soil_num=cv_soil,
                            k_init=float(result_v27.get("opt_k", ctx.k_val)),
                            z_init=float(result_v27.get("opt_z", ctx.z_val)),
                            q_val=float(ctx.q_val), r_val=float(ctx.r_val),
                            rc_val=float(ctx.rc_val), sens_val=float(ctx.sens_val_for_send),
                        )
                    else:
                        cv_report = temporal_kfold_cv(
                            ctx.file_path_to_send,
                            soil_num=cv_soil,
                            k_init=float(result_v27.get("opt_k", ctx.k_val)),
                            z_init=float(result_v27.get("opt_z", ctx.z_val)),
                            q_val=float(ctx.q_val), r_val=float(ctx.r_val),
                            rc_val=float(ctx.rc_val), sens_val=float(ctx.sens_val_for_send),
                            n_folds=3,
                        )
                    st.session_state["cv_report"] = cv_report
                except Exception as e:
                    st.error(f"교차검증 실패: {e}")

        cv_report = st.session_state.get("cv_report")
        if cv_report is not None:
            gr = cv_report.generalisation_ratio
            gr_color = "🟢" if gr < 1.2 else ("🟡" if gr < 1.5 else "🔴")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric(f"일반화 비율 {gr_color}", f"{gr:.2f}",
                      help="val_RMSE / cal_RMSE. 1.0=완벽, >1.5=과적합 의심")
            m2.metric("Cal RMSE (mean)", f"{cv_report.cal_rmse_mean:.4f} m")
            m3.metric("Val RMSE (mean)", f"{cv_report.val_rmse_mean:.4f} m")
            m4.metric("Val CC (mean)", f"{cv_report.val_cc_mean:.4f}")

            st.markdown("#### Fold 상세")
            fold_data = []
            for f in cv_report.folds:
                fold_data.append({
                    "Fold": f.fold_id,
                    "Cal RMSE": f"{f.cal_rmse:.4f}",
                    "Val RMSE": f"{f.val_rmse:.4f}",
                    "Cal CC": f"{f.cal_cc:.4f}",
                    "Val CC": f"{f.val_cc:.4f}",
                    "Cal Rech%": f"{f.cal_rech:.2f}",
                    "Val Rech%": f"{f.val_rech:.2f}",
                    "opt_k": f"{f.opt_k:.4f}",
                })
            st.dataframe(pd.DataFrame(fold_data), hide_index=True, use_container_width=True)

            if gr < 1.2:
                st.success("일반화 비율 < 1.2 → 모델이 과적합 없이 안정적입니다.")
            elif gr < 1.5:
                st.warning("일반화 비율 1.2~1.5 → 경미한 과적합 가능성. 파라미터 범위 확인 권장.")
            else:
                st.error("일반화 비율 > 1.5 → 과적합 의심. 데이터 품질 및 파라미터 검토 필요.")
