"""
app_v30.py — Hybrid Recharge AI Lab v30: 펌핑 전처리 통합 버전
==========================================================
기존 app.py(v27) 기능을 유지하면서 펌핑 전처리 모듈을 통합.

실행:
    streamlit run app_v30.py
"""

import os
import sys
import io
import tempfile
import hashlib
import warnings

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Suppress only known-harmless Streamlit and Plotly deprecation warnings.
# Do NOT use blanket warnings.filterwarnings("ignore") — it hides real bugs.
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"streamlit")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"plotly")

# ── 경로 설정 ────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
_PUMP_DIR = os.path.join(_THIS_DIR, "pump_preprocess")
if _PUMP_DIR not in sys.path:
    sys.path.insert(0, _PUMP_DIR)

from core_sim_v27 import core_sim_v27 as _run_sim  # noqa
from cross_validation import split_sample_test, temporal_kfold_cv
from data_loader import load_timeseries_file
from uncertainty import bootstrap_uncertainty

# ── pump_preprocess 모듈 import ──────────────────────────
try:
    from pump_preprocess.preprocess.detector import PumpingDetector
    from pump_preprocess.preprocess.corrector import WaterLevelCorrector
    from pump_preprocess.kalman.wtf_kalman import AugmentedKalmanWTF
    PUMP_MODULE_AVAILABLE = True
except ImportError as e:
    PUMP_MODULE_AVAILABLE = False
    _PUMP_IMPORT_ERR = str(e)

from soil_db import (
    SOIL_NAMES_NUMBERED as SOIL_NAMES,
    CLAY_LIKE_SET,
    get_soil,
)
from scoring import score_dataframe, score_k_stress, score_sy_match
from bma import compute_bma, bma_summary_table
from sensitivity import kalman_sensitivity_sweep, topsis_weight_sensitivity, objective_weight_sensitivity
from core_sim_v27 import get_kalman_uncertainty, propagate_kalman_recharge_uncertainty

# ── UI modules ───────────────────────────────────────────
from ui import TabContext, has_v27_error
from ui import tab_base, tab_pump, tab_compare
from ui import tab_crossval, tab_uncertainty, tab_sensitivity, tab_spatial
from ui import tab_field_report, tab_well_report, tab_watershed_report

MSG_UPLOAD_FIRST = "데이터 파일을 먼저 업로드하세요."
MSG_BASE_ANALYSIS_DONE = "기본 분석 완료!"
MSG_SCAN_FAILED = "스캔 실패: 유효한 결과 없음"


# ═════════════════════════════════════════════════════════
# 페이지 설정
# ═════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Hybrid Recharge AI Lab v30 (Pump Preprocess)",
    layout="wide",
    page_icon="💧",
)

# ── CSS (v27 계승 + 확장) ────────────────────────────────
st.markdown("""
<style>
.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"]{
    background-color:#FAFBFD!important;color:#1B2A4A!important}
section[data-testid="stSidebar"],section[data-testid="stSidebar"]>div{
    background-color:#F0F2F6!important;color:#1B2A4A!important}
.stApp p,.stApp span,.stApp div,.stApp label,.stApp li,.stApp td,.stApp th,
[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] li,[data-testid="stMarkdownContainer"] code{
    color:#1B2A4A!important}
h1,h2,h3,h4{color:#1B2A4A!important;font-weight:800!important}
div[data-testid="stMetric"]{background:linear-gradient(135deg,#FFF 0%,#F8F9FC 100%)!important;
    border:1px solid #E2E6ED!important;border-radius:12px;padding:16px 20px;
    box-shadow:0 2px 8px rgba(0,0,0,.04);transition:transform .2s,box-shadow .2s}
div[data-testid="stMetric"]:hover{transform:translateY(-2px);box-shadow:0 4px 16px rgba(0,0,0,.08)}
div[data-testid="stMetric"] label{color:#4A5A75!important;font-size:.8rem!important;
    font-weight:600!important;letter-spacing:.5px;text-transform:uppercase}
div[data-testid="stMetric"] div[data-testid="stMetricValue"]{
    color:#0F1A2E!important;font-size:1.6rem!important;font-weight:700!important}
.stButton>button{border-radius:8px;font-weight:600;transition:all .2s;
    background:#FFF!important;color:#1B2A4A!important;border:1.5px solid #CBD5E1!important}
.stButton>button:hover{transform:translateY(-1px);box-shadow:0 4px 12px rgba(0,0,0,.12);
    background:#F0F2F6!important}
.stButton>button[kind="primary"]{background:#DC2626!important;color:#FFF!important;border:none!important}
hr{border-color:#E2E6ED!important}
.pump-good{background:#D1FAE5;padding:4px 10px;border-radius:6px;display:inline-block}
.pump-warn{background:#FEF3C7;padding:4px 10px;border-radius:6px;display:inline-block}
.pump-bad{background:#FEE2E2;padding:4px 10px;border-radius:6px;display:inline-block}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════
# 세션 상태
# ═════════════════════════════════════════════════════════
_DEFAULTS = {
    "uploaded_tmp_path": None,
    "uploaded_file_sig": None,
    "uploaded_name": None,
    # v27 기본 결과
    "result_v27": None,
    "scan_data": None,
    "best_soil": None,
    "best_soil_conf": "MEDIUM",
    "best_soil_tentative": False,
    "monte_carlo": None,
    # pump preprocess 결과
    "pump_result": None,
    "pump_detection": None,
    # EnKF 공간 분석 결과
    "enkf_result": None,
    "enkf_object": None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ═════════════════════════════════════════════════════════
# 타이틀
# ═════════════════════════════════════════════════════════
st.title("💧 Hybrid Recharge AI Lab v30")
st.caption("Pump Pre-processing + Augmented Kalman Filter 통합 분석 플랫폼")

# ═════════════════════════════════════════════════════════
# 사용법 가이드 — 워크플로우 시나리오
# ═════════════════════════════════════════════════════════
with st.expander("📖 사용법 — 시나리오별 탭 사용 순서 (처음이라면 펼쳐 보세요)", expanded=False):
    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown("""
        ##### 🅰️ 단일 관정 분석
        **목표**: 관정 1개의 함양율 추정

        1. **사이드바** → 데이터 파일 업로드 (.txt/.csv: 날짜·수위·강우)
        2. **Tab 1 (기본분석)** → "분석 실행" 버튼
        3. *(선택)* **Tab 2** 펌핑 영향 보정
        4. **Tab 3 (토양비교)** 토양종류별 결과
        5. **Tab 4~6** 검증 (CV / 불확실성 / 민감도)
        6. **Tab 8 (관정 리포트)** → HTML 출력
        """)
    with g2:
        st.markdown("""
        ##### 🅱️ 유역 함양율 (다중 관정)
        **목표**: 유역 단위 면적 가중 함양율

        1. `wells_registry.py` 에 관정 좌표/유역 등록
        2. 정밀토양도 `.shp` 준비
        3. **Tab 10 (유역 함양율)** → 유역 선택 → 실행
        4. **Lumped vs Soil-weighted** 비교 확인
        5. CSV 다운로드

        ⚠️ 사이드바 파일 업로드 *불필요* — `.txt` 자동 매칭
        """)
    with g3:
        st.markdown("""
        ##### 🅲 합성 벤치마크 (논문용)
        **목표**: True recharge 정답 보유 시나리오 비교

        1. **Tab 7 (EnKF 합성)** 시나리오 S1~S5 실행
        2. **Tab 9 (Field 리포트)** 합성 결과 통합

        Lumped / Soil-weighted / EnKF 3-방법 비교 — 토양 불균질성 효과 정량화

        실측 데이터 부재 시 방법론 검증용
        """)
    st.markdown("---")
    st.caption(
        "**탭 분류**: Tab 1~6 = 단일 관정 분석 / Tab 7,9 = 합성 벤치마크 / "
        "Tab 8,10 = 통합 리포트 (8=관정, 10=유역)"
    )

# ═════════════════════════════════════════════════════════
# 사이드바 — 데이터 업로드
# ═════════════════════════════════════════════════════════
with st.sidebar.expander("ℹ️ 빠른 가이드", expanded=False):
    st.markdown("""
**시나리오별 출발점**

- **단일 관정** → 파일 업로드 → Tab 1
- **유역 (다중)** → Tab 10 (업로드 불필요)
- **합성 벤치마크** → Tab 7

**필수 입력 파일 형식**
- 컬럼 3개: `날짜 \\t 수위(m) \\t 강우(mm)`
- 일 단위, 결측 없음 권장

**Tab 10 사용 조건**
- `wells_registry.py` 에 관정 등록
- `.shp` 토양도 (기본 경로 사용)
- 유역 내 .txt 파일이 프로젝트 루트에 존재
    """)

st.sidebar.header("📁 데이터 & AI")
api_key = st.sidebar.text_input("OpenAI API Key (AI 소견용)", type="password")

# ── 프로젝트 폴더 .txt 파일 드롭다운 ──────────────────────
_proj_dir = os.path.dirname(os.path.abspath(__file__))
_local_txts = sorted([
    f for f in os.listdir(_proj_dir)
    if f.endswith(".txt") and f not in ("requirements.txt", "requirements-lock.txt")
])
_local_choice = None
if _local_txts:
    _options = ["(업로드 파일 사용)"] + _local_txts
    _sel = st.sidebar.selectbox("📂 저장된 관측정 파일", _options, key="local_file_sel")
    if _sel != "(업로드 파일 사용)":
        _local_choice = os.path.join(_proj_dir, _sel)

uploaded_file = st.sidebar.file_uploader("데이터 파일 업로드 (CSV/TXT)", type=["csv", "txt", "dat"])

file_path_to_send = "DEMO"
if _local_choice:
    file_path_to_send = _local_choice
    st.sidebar.success(f"로드됨: {os.path.basename(_local_choice)}")
elif uploaded_file is not None:
    try:
        file_bytes = uploaded_file.getvalue()
        file_sig = (uploaded_file.name, len(file_bytes), hashlib.md5(file_bytes).hexdigest())
        if st.session_state["uploaded_file_sig"] != file_sig:
            old = st.session_state.get("uploaded_tmp_path")
            if old and os.path.exists(old):
                try: os.remove(old)
                except OSError: pass
            suffix = os.path.splitext(uploaded_file.name)[1] or ".dat"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_bytes)
                st.session_state["uploaded_tmp_path"] = tmp.name
            st.session_state["uploaded_file_sig"] = file_sig
            st.session_state["uploaded_name"] = uploaded_file.name
            # 새 파일 → 이전 결과 초기화
            st.session_state["pump_result"] = None
            st.session_state["pump_detection"] = None
            st.session_state["result_v27"] = None
        file_path_to_send = st.session_state["uploaded_tmp_path"]
        st.sidebar.success(f"로드됨: {st.session_state['uploaded_name']}")
    except Exception as _e:
        st.sidebar.error(f"파일 오류: {_e}")

st.sidebar.markdown("---")

# ═════════════════════════════════════════════════════════
# 사이드바 — 기본 파라미터 (v27 호환)
# ═════════════════════════════════════════════════════════
with st.sidebar.expander("🔧 Kalman 파라미터", expanded=True):
    k_val = st.slider("Decay Constant (k)", -0.50, -0.0001, -0.015, step=0.0001, format="%.4f")
    z_val = st.slider("Unsat. Depth z (m)", 0.5, 30.0, 3.0, step=0.1)
    lag_val = st.slider("Lag Time (day)", 0, 15, 0)
    sn_idx = st.selectbox("Soil Type", range(1, 13), index=1,
                          format_func=lambda x: SOIL_NAMES[x - 1])
    auto_optimize = st.checkbox("Auto-Optimize (k, z 자동 최적화)", value=True,
                                help="체크 해제 시 슬라이더 k, z 값을 직접 사용하여 함양율·그래프 변화를 관찰할 수 있습니다")
    if auto_optimize:
        st.caption("🔒 Auto-Opt ON → k, z 슬라이더는 초기값만 제공 (최적화 결과가 우선)")
    else:
        st.caption("🔓 Manual Mode → k, z 슬라이더 값 그대로 적용됨")

with st.sidebar.expander("⚙️ 고급 (Kalman Q/R)", expanded=False):
    q_val = st.number_input("Process Noise Q", value=0.005, format="%.4f")
    r_val = st.number_input("Measure Noise R", value=0.1, format="%.2f")

with st.sidebar.expander("🚫 이상치 전처리 (v27)", expanded=False):
    sens_val = st.slider("Filter Intensity", 0.0, 10.0, 0.0, step=0.1)
    rc_val = st.slider("Ignore Rain < X (m)", 0.0, 0.05, 0.005, step=0.001, format="%.3f")
    show_pure = st.checkbox("Show Pure WTF", value=True, help="순수 WTF 수위 시계열 오버레이")

ignore_pump = 1.0 if sens_val > 0.001 else 0.0
sens_val_for_send = sens_val if sens_val > 0 else 1.0

st.sidebar.markdown("---")

# ═════════════════════════════════════════════════════════
# 사이드바 — 펌핑 전처리 설정
# ═════════════════════════════════════════════════════════
st.sidebar.subheader("🔧 펌핑 전처리 (v30)")
with st.sidebar.expander("탐지 설정", expanded=True):
    detect_methods = st.multiselect(
        "탐지 방법",
        ["sigma", "rolling_baseline", "fourier"],
        default=["sigma", "rolling_baseline"],
        help="sigma: 급강하 탐지 | rolling_baseline: 기준선 편차 탐지 | fourier: 주기성 필터"
    )
    sigma_drop = st.slider("Sigma 임계값", 1.5, 4.0, 2.5, step=0.1,
                           help="급강하 판별 승수 (클수록 보수적)")
    buffer_days = st.slider("Buffer (days)", 0, 5, 2,
                            help="탐지 구간 전후 여유 일수")

with st.sidebar.expander("보정 설정", expanded=False):
    correction_strategy = st.selectbox(
        "보정 전략",
        ["auto", "recession_fill", "spline_fill", "baseline_shift"],
        help="auto: 구간 길이에 따라 자동 선택"
    )

st.sidebar.markdown("---")
st.sidebar.subheader("🚀 실행")
st.sidebar.caption("순서: ① 기본분석 → ② 토양 스캔 → ③ 펌핑전처리")


# ═════════════════════════════════════════════════════════
# 실행 버튼들
# ═════════════════════════════════════════════════════════

# ── 1. 기본 분석 (v27) ──
run_base = st.sidebar.button("▶ ① 기본 분석 (v27 Auto-Fit)")

# ── 1b. Hybrid Soil Scan ──
run_scan = st.sidebar.button("🔍 ② Hybrid 토양 정밀 진단 (12종)")

# ── 2. 펌핑 전처리 + 재분석 ──
run_pump = st.sidebar.button("🔧 ③ 펌핑 전처리 + 재분석",
                              disabled=not PUMP_MODULE_AVAILABLE)

if not PUMP_MODULE_AVAILABLE:
    st.sidebar.caption(f"⚠️ pump_preprocess 모듈 로드 실패: {_PUMP_IMPORT_ERR}")


# ═════════════════════════════════════════════════════════
# 유틸 함수
# ═════════════════════════════════════════════════════════

def load_data_as_arrays(fpath):
    """Load and validate a time series file for the Streamlit workflow."""
    data = load_timeseries_file(
        fpath,
        interpolate_water_level=True,
        rainfall_unit="mm",
        require_dates=False,
    )
    return data.dates, data.water_level, data.rainfall_mm


def validate_input_file(fpath):
    """Fail fast on malformed input before running longer analyses."""
    load_timeseries_file(
        fpath,
        interpolate_water_level=False,
        rainfall_unit="mm",
        require_dates=False,
    )
    return True


def _normalize_v27_result(result):
    """Apply app-level defaults to a successful v27 result payload."""
    if result is None or "error" in result:
        return result

    normalized = dict(result)
    ho = list(normalized.get("ho", []))
    zeros = [0] * len(ho)

    defaults = {
        "sigma_ho": 0.1,
        "stress": np.nan,
        "recharge_ratio": np.nan,
        "flash_diff": 0.0,
        "rec_slope_err": 0.0,
        "pump_contam_idx": 0.0,
        "pump_event_count": 0,
        "pump_max_run": 0,
        "rain_resp_obs": np.nan,
        "rain_resp_sim": np.nan,
        "eval_n": 0,
        "hs_kf": zeros,
        "hs_pure": zeros,
        "po": zeros,
        "po_shifted": zeros,
        "rech": zeros,
        "pump_mask": zeros,
    }
    for key, value in defaults.items():
        normalized.setdefault(key, value)
    return normalized


def _has_v27_error(result):
    return result is None or "error" in result


def _build_scan_row(soil_index, result, default_k):
    """Convert a normalized v27 result into one scan table row."""
    ho_arr = np.array(result["ho"], dtype=float)
    hs_pure_arr = np.array(result["hs_pure"], dtype=float)
    pm_arr = np.array(result.get("pump_mask", [0] * len(ho_arr))).astype(bool)
    valid_pure = ~np.isnan(ho_arr) & ~np.isnan(hs_pure_arr) & ~pm_arr
    if valid_pure.sum() >= 5:
        pure_rmse = float(np.sqrt(np.nanmean((ho_arr[valid_pure] - hs_pure_arr[valid_pure]) ** 2)))
    else:
        pure_rmse = float(result.get("rmse", np.nan))

    return {
        "Soil": SOIL_NAMES[soil_index - 1].split(".")[1].strip(),
        "Index": soil_index,
        "RMSE": result.get("rmse", np.nan),
        "PureRMSE": pure_rmse,
        "SigmaHo": result.get("sigma_ho", 0.1),
        "Stress": result.get("stress", np.nan),
        "Recharge": result.get("recharge_ratio", np.nan),
        "FlashDiff": result.get("flash_diff", 0.0),
        "SlopeErr": result.get("rec_slope_err", 0.0),
        "PumpIdx": result.get("pump_contam_idx", 0.0),
        "PumpEvents": result.get("pump_event_count", 0),
        "PumpRun": result.get("pump_max_run", 0),
        "RainRespObs": result.get("rain_resp_obs", np.nan),
        "RainRespSim": result.get("rain_resp_sim", np.nan),
        "EvalN": result.get("eval_n", 0),
        "OptK": result.get("opt_k", default_k),
        "SyEff": result.get("Sy_eff", np.nan),
    }


# ═════════════════════════════════════════════════════════
# ① 기본 분석 실행
# ═════════════════════════════════════════════════════════
def _v27_param_key():
    if auto_optimize:
        return (file_path_to_send, "AUTO", float(lag_val),
                float(sn_idx), float(q_val), float(r_val), float(rc_val),
                float(ignore_pump), float(sens_val_for_send), True)
    else:
        return (file_path_to_send, float(k_val), float(z_val), float(lag_val),
                float(sn_idx), float(q_val), float(r_val), float(rc_val),
                float(ignore_pump), float(sens_val_for_send), False)

def _run_v27_analysis():
    validate_input_file(file_path_to_send)
    do_opt = 1.0 if auto_optimize else 0.0
    result = _run_sim(
        file_path_to_send,
        float(k_val), float(z_val), float(lag_val),
        float(sn_idx),
        float(q_val), float(r_val), float(rc_val),
        float(ignore_pump), float(sens_val_for_send),
        do_opt,
    )
    result = _normalize_v27_result(result)
    if _has_v27_error(result):
        st.error(f"시뮬레이션 오류: {result['error']}")
    else:
        st.session_state["result_v27"] = result
        st.session_state["_v27_param_key"] = _v27_param_key()
    return result

if run_base:
    if file_path_to_send == "DEMO":
        st.sidebar.warning(MSG_UPLOAD_FIRST)
    else:
        with st.spinner("v27 기본 분석 실행 중..."):
            result = _run_v27_analysis()
            if not _has_v27_error(result):
                st.sidebar.success(MSG_BASE_ANALYSIS_DONE)

elif (st.session_state.get("result_v27") is not None
      and file_path_to_send != "DEMO"
      and st.session_state.get("_v27_param_key") != _v27_param_key()):
    with st.spinner("파라미터 변경 감지 → v27 재분석 중..."):
        _run_v27_analysis()


# ═════════════════════════════════════════════════════════
# ② Hybrid 토양 스캔
# ═════════════════════════════════════════════════════════
if run_scan:
    if file_path_to_send == "DEMO":
        st.sidebar.warning(MSG_UPLOAD_FIRST)
    else:
        validate_input_file(file_path_to_send)
        scan_results = []
        prog = st.sidebar.progress(0)
        status = st.sidebar.empty()

        for i in range(1, 13):
            status.text(f"스캔 중: {SOIL_NAMES[i-1]} ({i}/12) — 개별 최적화...")
            try:
                res = _run_sim(
                    file_path_to_send,
                    float(k_val), float(z_val), float(lag_val),
                    float(i), float(q_val), float(r_val),
                    float(rc_val), float(ignore_pump), float(sens_val_for_send),
                    1.0,
                )
                res = _normalize_v27_result(res)
                if not (_has_v27_error(res) or "rmse" not in res):
                    scan_results.append(_build_scan_row(i, res, k_val))
            except Exception as _e:
                st.sidebar.caption(f"⚠️ {SOIL_NAMES[i-1]} 실패: {_e}")
            prog.progress(i / 12)

        prog.empty()
        status.empty()

        if not scan_results:
            st.sidebar.error(MSG_SCAN_FAILED)
        else:
            df = score_dataframe(pd.DataFrame(scan_results))
            best = df.iloc[0]
            gap = float(best["TopsisScore"] - df.iloc[1]["TopsisScore"]) if len(df) > 1 else 99.0
            pump_b = float(best["PumpIdx"])
            reco_conf = "LOW" if pump_b >= 0.45 else ("MEDIUM" if pump_b >= 0.25 or gap < 5 else "HIGH")
            tentative = int(best["Index"]) in CLAY_LIKE_SET and pump_b >= 0.20

            try:
                bma_result = compute_bma(df)
                st.session_state["bma_result"] = bma_result
                if bma_result.dominant_prob >= 0.40:
                    best_idx = bma_result.dominant_soil
                    best = df[df["Index"] == best_idx].iloc[0]
            except Exception as _bma_err:
                st.sidebar.caption(f"BMA 계산 참고: {_bma_err}")

            st.session_state.update({
                "scan_data": df,
                "best_soil": int(best["Index"]),
                "best_soil_conf": reco_conf,
                "best_soil_tentative": tentative,
            })

            label = SOIL_NAMES[int(best["Index"])-1] + (" (tentative)" if tentative else "")
            st.sidebar.success(f"최적 토양: {label}\n신뢰도: {reco_conf}")
            if pump_b >= 0.45:
                st.sidebar.error("🚫 펌핑 오염도 높음 → 전처리 후 재진단 권장")
            elif pump_b >= 0.25:
                st.sidebar.warning("⚠️ 펌핑 의심 → ③ 전처리 실행 권장")
            st.rerun()


# ═════════════════════════════════════════════════════════
# ③ 펌핑 전처리 + 재분석
# ═════════════════════════════════════════════════════════
if run_pump:
    if file_path_to_send == "DEMO":
        st.sidebar.warning(MSG_UPLOAD_FIRST)
    elif not PUMP_MODULE_AVAILABLE:
        st.sidebar.error("펌핑 전처리 모듈을 로드할 수 없습니다.")
    else:
        with st.spinner("펌핑 전처리 + Augmented Kalman 실행 중..."):
            try:
                dates, wl, rain = load_data_as_arrays(file_path_to_send)
                n = len(wl)

                detector = PumpingDetector(
                    methods=detect_methods if detect_methods else ["sigma"],
                    sigma_drop=sigma_drop,
                    buffer_days=buffer_days,
                )
                det_result = detector.detect(dates, wl, rain)

                corrector = WaterLevelCorrector(strategy=correction_strategy)
                cor_result = corrector.correct(dates, wl, det_result.pump_mask, rain)

                kalman_raw = AugmentedKalmanWTF(
                    soil_num=int(sn_idx),
                    auto_optimize=True,
                    exclude_pump_from_kalman=True,
                )
                rain_mm = rain if rain.max() > 1.0 else rain * 1000.0
                result_raw = kalman_raw.run(wl, rain_mm, det_result.pump_mask)

                kalman_corr = AugmentedKalmanWTF(
                    soil_num=int(sn_idx),
                    auto_optimize=True,
                    exclude_pump_from_kalman=False,
                )
                result_corr = kalman_corr.run(
                    cor_result.corrected_wl, rain_mm, None
                )

                pump_frac = float(det_result.pump_mask.sum()) / n
                pump_detected = pump_frac >= 0.005

                st.session_state["pump_result"] = {
                    "dates": dates,
                    "raw_wl": wl,
                    "corrected_wl": cor_result.corrected_wl,
                    "rainfall": rain_mm,
                    "pump_mask": det_result.pump_mask,
                    "confidence": det_result.confidence,
                    "method_masks": det_result.method_masks,
                    "pump_fraction": pump_frac,
                    "pump_detected": pump_detected,
                    "n_events": len(det_result.drop_events),
                    "correction_strategy": cor_result.strategy_used,
                    "raw": {
                        "h_sim": result_raw.h_sim,
                        "rmse": result_raw.rmse,
                        "nse": result_raw.nse,
                        "cc": result_raw.cc,
                        "rech_rate": result_raw.rech_rate_pct,
                        "rech_total": result_raw.rech_total,
                        "soil": result_raw.best_soil_name,
                        "k": result_raw.k_val,
                        "soil_scores": result_raw.soil_scores,
                    },
                    "corrected": {
                        "h_sim": result_corr.h_sim,
                        "rmse": result_corr.rmse,
                        "nse": result_corr.nse,
                        "cc": result_corr.cc,
                        "rech_rate": result_corr.rech_rate_pct,
                        "rech_total": result_corr.rech_total,
                        "soil": result_corr.best_soil_name,
                        "k": result_corr.k_val,
                        "soil_scores": result_corr.soil_scores,
                    },
                }

                corrected_wl = cor_result.corrected_wl
                tmp_df = pd.DataFrame({
                    "date": dates,
                    "wl": corrected_wl,
                    "rain": rain,
                })
                tmp_path = os.path.join(
                    tempfile.gettempdir(),
                    f"wtf_corrected_{hashlib.md5(file_path_to_send.encode()).hexdigest()[:8]}.csv"
                )
                tmp_df.to_csv(tmp_path, index=False, header=False)

                result_v27_corr = _run_sim(
                    tmp_path,
                    float(k_val), float(z_val), float(lag_val),
                    float(sn_idx),
                    float(q_val), float(r_val), float(rc_val),
                    0.0, float(sens_val_for_send), 1.0,
                )
                result_v27_corr = _normalize_v27_result(result_v27_corr)

                result_v27_orig = _run_sim(
                    file_path_to_send,
                    float(k_val), float(z_val), float(lag_val),
                    float(sn_idx),
                    float(q_val), float(r_val), float(rc_val),
                    float(ignore_pump), float(sens_val_for_send), 1.0,
                )
                result_v27_orig = _normalize_v27_result(result_v27_orig)

                rr_v27_orig = float(result_v27_orig.get("recharge_ratio", 0))
                rr_v27_corr = float(result_v27_corr.get("recharge_ratio", 0)) if not _has_v27_error(result_v27_corr) else rr_v27_orig

                st.session_state["pump_result"]["v27_orig"] = {
                    "rech_rate": rr_v27_orig,
                    "rmse": float(result_v27_orig.get("rmse", 0)),
                    "cc": float(result_v27_orig.get("cc", 0)),
                    "opt_k": float(result_v27_orig.get("opt_k", 0)),
                    "opt_z": float(result_v27_orig.get("opt_z", 0)),
                    "rech": result_v27_orig.get("rech", []),
                    "hs_kf": result_v27_orig.get("hs_kf", []),
                }
                st.session_state["pump_result"]["v27_corr"] = {
                    "rech_rate": rr_v27_corr,
                    "rmse": float(result_v27_corr.get("rmse", 0)),
                    "cc": float(result_v27_corr.get("cc", 0)),
                    "opt_k": float(result_v27_corr.get("opt_k", 0)),
                    "opt_z": float(result_v27_corr.get("opt_z", 0)),
                    "rech": result_v27_corr.get("rech", []),
                    "hs_kf": result_v27_corr.get("hs_kf", []),
                }

                existing_v27 = st.session_state.get("result_v27")
                if existing_v27 is not None:
                    existing_v27["recharge_ratio_corrected"] = rr_v27_corr
                    st.session_state["result_v27"] = existing_v27

                if not pump_detected:
                    st.sidebar.warning(
                        f"⚠️ 펌핑 구간 미탐지 (비율: {pump_frac*100:.2f}%)\n"
                        f"v27 함양율: {rr_v27_orig:.2f}% → {rr_v27_corr:.2f}% (변화 없음)"
                    )
                else:
                    delta = rr_v27_corr - rr_v27_orig
                    st.sidebar.success(
                        f"펌핑 전처리 완료! (펌핑 {pump_frac*100:.1f}%)\n"
                        f"v27 함양율: {rr_v27_orig:.2f}% → {rr_v27_corr:.2f}% ({delta:+.2f}%p)"
                    )

            except Exception as e:
                st.error(f"펌핑 전처리 오류: {e}")
                import traceback
                st.code(traceback.format_exc())


# ═════════════════════════════════════════════════════════
# Build TabContext for UI modules
# ═════════════════════════════════════════════════════════
_ctx = TabContext(
    sn_idx=sn_idx,
    k_val=k_val,
    z_val=z_val,
    lag_val=lag_val,
    q_val=q_val,
    r_val=r_val,
    rc_val=rc_val,
    sens_val_for_send=sens_val_for_send,
    ignore_pump=ignore_pump,
    auto_optimize=auto_optimize,
    show_pure=show_pure,
    file_path_to_send=file_path_to_send,
    api_key=api_key,
    detect_methods=detect_methods,
    sigma_drop=sigma_drop,
    buffer_days=buffer_days,
    correction_strategy=correction_strategy,
)


# ═════════════════════════════════════════════════════════
# 결과 표시 — Tab Rendering (delegated to ui/ modules)
# ═════════════════════════════════════════════════════════
# 탭 가로 스크롤 + 화살표 버튼 활성화 (탭 개수 많을 때)
st.markdown("""
<style>
div[data-testid="stTabs"] > div:first-child {
    overflow-x: auto !important;
    scrollbar-width: thin;
}
div[data-testid="stTabs"] > div:first-child::-webkit-scrollbar {
    height: 8px;
}
div[data-testid="stTabs"] > div:first-child::-webkit-scrollbar-thumb {
    background: #888;
    border-radius: 4px;
}
div[data-testid="stTabs"] button[role="tab"] {
    flex: 0 0 auto !important;
    white-space: nowrap !important;
}
</style>
""", unsafe_allow_html=True)

(
    main_tab1, main_tab2, main_tab3, main_tab4, main_tab5,
    main_tab6, main_tab7, main_tab8, main_tab9, main_tab10,
) = st.tabs([
    "1. 기본분석",
    "2. 펌핑보정",
    "3. 토양비교",
    "4. 교차검증",
    "5. 불확실성",
    "6. 민감도",
    "7. EnKF (합성)",
    "8. 📋 관정 리포트",
    "9. Field (합성)",
    "10. 🗾 유역 함양율",
])

tab_base.render(main_tab1, _ctx)
tab_pump.render(main_tab2, _ctx)
tab_compare.render(main_tab3, _ctx)
tab_crossval.render(main_tab4, _ctx)
tab_uncertainty.render(main_tab5, _ctx)
tab_sensitivity.render(main_tab6, _ctx)
tab_spatial.render(main_tab7, _ctx)
tab_well_report.render(main_tab8, _ctx)
tab_field_report.render(main_tab9, _ctx)
# 임시 주석: Tab 10 에러 진단용
try:
    tab_watershed_report.render(main_tab10, _ctx)
except Exception as e:
    with main_tab10:
        import streamlit as st
        st.error(f"⚠️ Tab 10 렌더 실패: {e}")
        import traceback
        st.code(traceback.format_exc())
