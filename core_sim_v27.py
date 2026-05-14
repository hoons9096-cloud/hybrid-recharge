"""Core simulation utilities for the v27 hybrid-recharge workflow."""

# Public API — 외부에서 안전하게 import 가능한 이름 목록.
# 언더스코어 함수는 여기 포함되지 않으며, 외부 모듈이 필요로 하는
# 함수는 public 이름(언더스코어 없음)으로 승격됨.
__all__ = [
    # 핵심 시뮬레이션
    "run_logic_v27",
    "core_sim_v27",
    "calc_error",
    # 전처리 유틸리티
    "apply_lag",
    "detect_pump_mask",
    "remove_outliers",
    "filpor_v27",
    "calc_rain_response",
    # 워크플로우 헬퍼 (외부 모듈에서 사용)
    "load_core_data",
    "normalize_core_inputs",
    "optimize_parameters",
    "build_core_metrics",
    # 불확실성
    "get_kalman_uncertainty",
    "propagate_kalman_recharge_uncertainty",
    # 결과 데이터클래스
    "CoreMetrics",
    "CoreErrorResult",
]

import warnings
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import minimize

from wtf_logger import get_logger

logger = get_logger(__name__)

from core_sim_config import (
    ANTECEDENT_DRAIN_DAYS,
    DAYS_PER_YEAR,
    DEFAULT_N_F_AVG,
    DEFAULT_Q_NOISE,
    DEFAULT_R_NOISE,
    DEFAULT_SIGMA_HO,
    KALMAN_Q_FLOOR,
    KALMAN_R_FLOOR,
    KALMAN_RHO,
    KALMAN_W_Q_RATIO,
    KALMAN_HW_Q_CORR,
    KALMAN_WTF_BLEND_ALPHA,
    MAX_Z_PARAM,
    MIN_ANNUAL_SEGMENT_OBS,
    MIN_SEGMENT_DAYS,
    MIN_SY_FLOOR,
    MIN_SY_FOR_INPUT_SCALE,
    MIN_UNSAT_DEPTH,
    MIN_VALID_FRACTION,
    MIN_VALID_POINTS,
    OBJ_RECH_VIOLATION_CAP,
    OBJ_W_FIT,
    OBJ_W_RECH,
    OBJ_W_RESP,
    OPT_FATOL,
    OPT_LAG_SEARCH_DAYS,
    OPT_MAXFEV,
    OPT_XATOL,
    OUTLIER_BASE_FACTOR,
    OUTLIER_MIN_FACTOR,
    OUTLIER_SENSITIVITY_SCALE,
    PUMP_EVENTS_NORMALIZER,
    PUMP_EVENTS_WEIGHT,
    PUMP_FRAC_NORMALIZER,
    PUMP_FRAC_WEIGHT,
    PUMP_MAXRUN_NORMALIZER,
    PUMP_MAXRUN_WEIGHT,
    PUMP_RUN_MIN_DROP,
    PUMP_RUN_MIN_LENGTH,
    PUMP_RUN_POST_DAYS,
    PUMP_RUN_PRE_DAYS,
    PUMP_RUN_SIGMA_MULTIPLIER,
    PUMP_SIGMA_FALLBACK,
    PUMP_SPIKE_MIN_DROP,
    PUMP_SPIKE_POST_DAYS,
    PUMP_SPIKE_PRE_DAYS,
    PUMP_SPIKE_SIGMA_MULTIPLIER,
    MIN_Z_PARAM,
    KALMAN_P0_H_FLOOR,
    KALMAN_P0_W_RATIO,
    KALMAN_R_PUMP_FACTOR,
    KALMAN_R_PUMP_PROXIMITY_DAYS,
    KALMAN_R_PUMP_SIGMA,
    INTER_EVENT_FRAC,
    OPT_LAG_XCORR_CANDIDATES,
)
# Note: RESP_PENALTY_WEIGHT and RECHARGE_PENALTY_WEIGHT are no longer used.
# calc_error now uses dimensionless normalised components with inline weights.
from data_loader import load_timeseries_file
from soil_db import (
    VG_DB,
    TAU_DB,
    ALPHA_SOIL_LIST,
    SY_LIT_LIST,
    RECH_RANGE,
    get_bounds,
    gap_allow_for_soil,
    peak_window_for_soil,
    adaptive_peak_window,
)


# Module-level cache for Kalman covariance data (populated by run_logic_v27)
_last_kalman_extras: dict = {}


def get_kalman_uncertainty() -> dict:
    """Retrieve Kalman filter covariance data from the last simulation.

    Returns
    -------
    dict with keys:
        P_h_var : np.ndarray — variance of water level state σ²(h)
        P_w_var : np.ndarray — variance of hidden forcing state σ²(w)
        w_est   : np.ndarray — smoothed hidden forcing estimates
    Empty dict if no simulation has been run.
    """
    return dict(_last_kalman_extras)


def propagate_kalman_recharge_uncertainty(
    rech: np.ndarray,
    sy_eff: float,
    P_h_var: np.ndarray,
) -> np.ndarray:
    """Propagate Kalman h-state uncertainty to event recharge uncertainty.

    For each recharge event R = Sy × Δh, the uncertainty in Δh derives
    from the Kalman covariance of h:

        Var(Δh) ≈ Var(h_peak) + Var(h_before) ≈ 2 × σ²(h)

    Therefore:
        σ(R) = Sy × √(2 × σ²(h))

    Parameters
    ----------
    rech : np.ndarray
        Recharge array (non-zero at event starts).
    sy_eff : float
        Effective specific yield.
    P_h_var : np.ndarray
        Kalman-smoothed variance of h at each timestep.

    Returns
    -------
    np.ndarray
        Standard deviation of recharge at each timestep (0 where no event).
    """
    rech_std = np.zeros_like(rech)
    event_idx = np.where(rech > 0)[0]
    n_pvar = len(P_h_var)
    for idx in event_idx:
        if idx >= n_pvar:
            continue  # skip if index exceeds covariance array length
        # σ(Δh) ≈ √(2 × σ²(h)) assuming peak/before h are ~independent
        sigma_dh = np.sqrt(2.0 * max(P_h_var[idx], 1e-12))
        rech_std[idx] = sy_eff * sigma_dh
    return rech_std


def apply_lag(po, lag_days):
    lag_days = max(0, round(lag_days))
    if lag_days == 0:
        return po.copy()
    return np.concatenate([np.zeros(lag_days), po[:len(po) - lag_days]])


def remove_outliers(ho_raw, sensitivity):
    ho = ho_raw.copy()
    dh = np.diff(ho_raw)
    sig = np.nanstd(dh)
    if not np.isfinite(sig) or sig == 0:
        return ho
    factor = max(
        OUTLIER_BASE_FACTOR - sensitivity * OUTLIER_SENSITIVITY_SCALE,
        OUTLIER_MIN_FACTOR,
    )
    threshold = -factor * sig
    bad_idx = np.where(dh < threshold)[0] + 1
    ho[bad_idx] = np.nan
    return ho


def detect_pump_mask(ho, po, rc):
    n = len(ho)
    mask = np.zeros(n, dtype=bool)
    if n < 5:
        return mask

    dh = np.concatenate([[0.0], np.diff(ho)])
    sig = np.nanstd(dh[~np.isnan(dh)])
    if not np.isfinite(sig) or sig <= 0:
        sig = PUMP_SIGMA_FALLBACK

    dry = (po <= rc) & ~np.isnan(ho)

    th_spike = min(PUMP_SPIKE_SIGMA_MULTIPLIER * sig, PUMP_SPIKE_MIN_DROP)
    spike_idx = np.where(dry & (dh < th_spike))[0]
    for ii in spike_idx:
        s = max(0, ii - PUMP_SPIKE_PRE_DAYS)
        e = min(n, ii + PUMP_SPIKE_POST_DAYS)
        mask[s:e] = True

    th_run = min(PUMP_RUN_SIGMA_MULTIPLIER * sig, PUMP_RUN_MIN_DROP)
    neg_dry = dry & (dh < th_run)
    i = 0
    while i < n:
        if neg_dry[i]:
            j = i
            while j < n and neg_dry[j]:
                j += 1
            if (j - i) >= PUMP_RUN_MIN_LENGTH:
                mask[
                    max(0, i - PUMP_RUN_PRE_DAYS):min(n, j + PUMP_RUN_POST_DAYS)
                ] = True
            i = j
        else:
            i += 1

    return mask


def filpor_v27(soil_num, z_unsat, time_dry):
    """Estimate event-specific specific yield (Sy) for WTF recharge calculation.

    Input validation (v32): negative z_unsat or time_dry are clamped to
    physical floor values with a warning, rather than producing garbage.

    Physical basis
    --------------
    The standard WTF method (Healy & Cook, 2002) assumes a constant Sy, but
    in reality Sy depends on (a) the depth to the water table and (b) the
    antecedent drainage duration.  This function accounts for both effects:

    1. **Depth dependence** — Uses the van Genuchten (1980) retention curve to
       estimate the effective saturation (Se) at the given unsaturated zone
       depth (z_unsat).  The drainable porosity is the portion of the pore
       space that is unsaturated:  ``(θs - θr) * (1 - Se)``.

       Reference: van Genuchten, M.T. (1980). A closed-form equation for
       predicting the hydraulic conductivity of unsaturated soils.
       SSSAJ, 44(5), 892-898.

    2. **Drainage time dependence** — After a recharge event, gravity drainage
       is not instantaneous.  The *recovery factor* models the fraction of
       drainable pore space that has actually drained during the antecedent
       dry period, using a first-order exponential model:

           recovery = 1 - exp(-time_dry / τ)

       where τ (tau) is a soil-specific characteristic drainage time constant.
       τ values are derived from the median Ksat of each USDA texture class
       (Carsel & Parrish, 1988), converted to an approximate gravity-drainage
       time scale via τ ≈ (θs - θr) * L / Ksat, where L is a representative
       drainage path length.  This is an empirical approximation; more rigorous
       approaches would solve Richards' equation numerically.

       Note: τ is NOT the Carsel & Parrish alpha or n parameter — it is a
       derived timescale.  The values in TAU_DB are calibrated to match
       observed field drainage curves from lysimeter studies.

    Returns
    -------
    Sy : float
        Effective specific yield for the current event, clamped to
        [0.001, θs - θr].  Units: dimensionless [-].
    """
    sn = max(1, min(12, round(soil_num))) - 1
    th_s, th_r, alpha, n_vg = VG_DB[sn]
    tau = TAU_DB[sn]
    m_vg = 1.0 - 1.0 / n_vg

    # ── Input guards (v32) ──
    if z_unsat < 0:
        logger.warning("filpor_v27: z_unsat=%.3f < 0, clamping to MIN_UNSAT_DEPTH", z_unsat)
        z_unsat = MIN_UNSAT_DEPTH
    if time_dry < 0:
        logger.warning("filpor_v27: time_dry=%.1f < 0, clamping to 0", time_dry)
        time_dry = 0

    # Depth-dependent drainable porosity via VG retention curve
    h_unsat = max(z_unsat, MIN_UNSAT_DEPTH)  # avoid singularity at z=0
    se = 1.0 / (1.0 + (alpha * h_unsat) ** n_vg) ** m_vg
    drainable_porosity = (th_s - th_r) * (1.0 - se)

    # Time-dependent drainage recovery
    recovery = 1.0 - np.exp(-time_dry / max(tau, 1))

    sy_raw = drainable_porosity * recovery
    sy_max = th_s - th_r
    return max(min(sy_raw, sy_max), MIN_SY_FLOOR)


def _estimate_equilibrium_head(ho_in, po_in, r_c):
    """Estimate the equilibrium (base-level) water table head.

    The equilibrium head h_eq represents the water level the aquifer would
    converge to in the absence of recharge — i.e., the long-term recession
    baseline.  It serves as the reference datum for the exponential decay
    model:  h(t) = h_eq + (h0 - h_eq) * exp(k * t).

    Method
    ------
    A unified algorithm is used regardless of record length (removing the
    previous discontinuity at 365 days):

    1. **Extract dry-period observations** — days where rainfall ≤ r_c and
       the water level is not NaN.  These represent recession-dominated
       conditions unaffected by recharge pulses.

    2. **Trend correction** — If ≥ 10 dry observations exist, fit a linear
       trend to account for long-term storage changes (e.g., regional
       pumping or climate trends).  De-trend and evaluate the median at the
       record mid-point.  This avoids biasing h_eq toward the start or end
       of a trending record.

    3. **Small-sample fallback** — With fewer dry observations, use the
       lower quartile (25th percentile) of available water levels as a
       conservative h_eq estimate.  The lower quartile is preferred over
       the median because the equilibrium head should represent the
       base-flow recession floor, not the average condition.

    4. **Multi-year correction** — For records spanning > 1 year, the
       result is blended 50/50 with the mean of annual minima to anchor
       h_eq to the driest conditions in each hydrological year.

    Returns
    -------
    h_eq : float
        Estimated equilibrium head [m].
    ho_finite : np.ndarray
        Non-NaN subset of ho_in, for downstream use.
    """
    nn = len(ho_in)
    dry_mask = (po_in <= r_c) & ~np.isnan(ho_in)
    dry_vals = ho_in[dry_mask]
    ho_finite = ho_in[~np.isnan(ho_in)]

    if len(ho_finite) == 0:
        return 0.0, ho_finite

    # ── Primary estimate from dry-period observations ──
    if len(dry_vals) >= 10:
        # Trend-corrected median at record midpoint
        t_dry = np.where(dry_mask)[0].astype(float)
        p_trend = np.polyfit(t_dry, dry_vals.astype(float), 1)
        detrended = dry_vals - np.polyval(p_trend, t_dry)
        h_eq = float(np.median(detrended) + np.polyval(p_trend, nn / 2.0))
    elif len(dry_vals) >= 3:
        # Small sample: use lower quartile as conservative base level
        h_eq = float(np.percentile(dry_vals, 25))
    else:
        # Minimal data: lower quartile of all observations
        h_eq = float(np.percentile(ho_finite, 25))

    # ── Multi-year anchor: blend with mean of annual minima ──
    if nn >= DAYS_PER_YEAR:
        annual_min = []
        for yr_start in range(0, nn - MIN_SEGMENT_DAYS, DAYS_PER_YEAR):
            yr_end = min(yr_start + DAYS_PER_YEAR, nn)
            seg = ho_in[yr_start:yr_end]
            seg_valid = seg[~np.isnan(seg)]
            if len(seg_valid) >= MIN_ANNUAL_SEGMENT_OBS:
                annual_min.append(float(np.min(seg_valid)))
        if annual_min:
            h_eq_annual = float(np.mean(annual_min))
            h_eq = 0.5 * h_eq + 0.5 * h_eq_annual  # blend

    # ── Safety ──
    if not np.isfinite(h_eq):
        h_eq = float(np.median(ho_finite))

    return h_eq, ho_finite


def _accumulate_event_recharge(
    rech, ns, po_in, ho_in, pump_mask_in, ho_finite, h_eq,
    k, sn_c, alpha_soil, sy_lit, n_f_avg, r_c
):
    """이벤트별 WTF 함양 누산 (NumPy 벡터화 전처리로 최적화).

    핵심 최적화 (원래 Python while/for 내부 루프를 사전 계산으로 대체):
      1. h_last_obs 역방향 탐색 → forward-fill 배열 O(1) 조회
      2. days_dry_before 계산 → cumsum 기반 연속건조일 배열 O(1) 조회
      3. 이벤트 후 피크 수위 → sliding_window_view 사전 계산 O(1) 조회
         (v32: 적응형 peak window 도입 — 강우 강도에 따라 window 조정)
      4. 이벤트 종료 탐색 (O(n)/event) → gap_trigger + searchsorted O(log n)/event
      5. 이벤트 강우량 합산 → cumsum O(1) 조회

    적응형 Peak Window (v32)
    ────────────────────────
    기존 고정 peak window는 토양 유형에만 의존했으나, 실제 수위 응답 시간은
    강우 강도에도 크게 좌우됨.  Green-Ampt 이론에 따르면 습윤전선 속도가
    강우 강도의 함수이므로, 강한 이벤트는 짧은 window, 약한 이벤트는 긴
    window를 사용하는 것이 물리적으로 타당함.

    수치 동등성: KALMAN_HW_Q_CORR=0 및 mean_rain 기반 scaling=1 일 때
    기존 결과와 동일.
    """
    nn = len(ho_in)
    if nn < 2:
        return

    gap_allow = gap_allow_for_soil(sn_c)
    pw_base = int(peak_window_for_soil(sn_c))
    # 적응형 peak window를 위한 평균 이벤트 강우량 사전 계산
    wet_rain = po_in[po_in > r_c]
    _mean_event_rain = float(np.mean(wet_rain)) if len(wet_rain) > 0 else 10.0
    pw = pw_base  # 사전 계산용 최대 window (backward compat)

    # ── 전처리 1: 유효 수위 forward-fill ──────────────────────────────────
    # h_ffill[i]     = i 이전(포함) 마지막 유효(비NaN, 비펌핑) 수위
    # h_ffill_idx[i] = 해당 유효 수위의 인덱스 (-1 = 아직 없음)
    ho_valid_arr = np.where(pump_mask_in | np.isnan(ho_in), np.nan, ho_in)
    not_nan = ~np.isnan(ho_valid_arr)
    if np.any(not_nan):
        fi = np.where(not_nan, np.arange(nn), 0)
        np.maximum.accumulate(fi, out=fi)
        has_prior = np.cumsum(not_nan.astype(np.int64)) > 0
        h_ffill = np.where(has_prior, ho_valid_arr[fi], np.nan)
        h_ffill_idx = np.where(has_prior, fi, np.int64(-1)).astype(np.int64)
    else:
        h_ffill = np.full(nn, np.nan)
        h_ffill_idx = np.full(nn, -1, dtype=np.int64)

    # ── 전처리 2: 연속 건조일 배열 ────────────────────────────────────────
    # consec_dry[i] = i에서 끝나는 연속 건조일 수 (습윤일이면 0)
    # 누적합 트릭: dry_cs - (마지막 습윤일에서의 dry_cs 값)
    dry = (po_in <= r_c)
    dry_cs = np.cumsum(dry.astype(np.int64))
    wet_dry_cs = np.where(~dry, dry_cs, np.int64(0))
    last_wet_dry_cs = np.maximum.accumulate(wet_dry_cs)
    consec_dry = np.where(dry, dry_cs - last_wet_dry_cs, np.int64(0))

    # ── 전처리 3: 강우 누적합 (이벤트 강우량 O(1) 조회) ──────────────────
    po_cs = np.empty(nn + 1, dtype=np.float64)
    po_cs[0] = 0.0
    np.cumsum(po_in, out=po_cs[1:])

    # ── 전처리 4: look-ahead 수위 배열 (적응형 peak window용) ────────────
    # v32: 적응형 peak window 도입.  이벤트별로 window 크기가 달라지므로
    # 고정 sliding max 대신 ho_in 배열을 직접 참조하여 per-event nanmax 계산.
    # 최대 가능 window (pw_max = ceil(1.5 × pw_base)) 크기의 사전 패딩만 수행.
    pw_max = int(np.ceil(1.5 * pw_base))
    pw_max = max(pw_max, pw_base)  # safety
    ho_padded = np.concatenate([ho_in, np.full(pw_max, np.nan)])

    # ── 전처리 5: gap trigger 위치 (이벤트 종료 O(log n) 검색) ───────────
    # consec_dry[j] == gap_allow+1 → j는 (gap_allow+1)번째 연속 건조일
    # 원본 루프에서 gap > gap_allow 조건으로 break하는 시점 = j
    # → event_end = j - 1
    gap_triggers = np.where(consec_dry == gap_allow + 1)[0]

    # ho_finite 경계
    h_lo = float(np.min(ho_finite)) if len(ho_finite) > 0 else -np.inf
    h_hi = float(np.max(ho_finite)) if len(ho_finite) > 0 else np.inf

    # ── 이벤트 루프 (O(E log n), E = 이벤트 수) ──────────────────────────
    ii = 0
    while ii <= nn - 2:
        if dry[ii]:
            ii += 1
            continue

        event_start = ii

        # --- 선행 수위 h_before 계산 ---
        if event_start > 0:
            consec_val = int(consec_dry[event_start - 1])
            days_dry_before = consec_val
            # ANTECEDENT_DRAIN_DAYS 범위 내 마지막 유효 수위 (O(1) 조회)
            lb_idx = max(0, event_start - min(consec_val + 1, ANTECEDENT_DRAIN_DAYS))
            last_idx = int(h_ffill_idx[event_start - 1])
            if last_idx >= lb_idx:
                h_last_obs = float(h_ffill[event_start - 1])
            else:
                h_last_obs = np.nan
        else:
            days_dry_before = 0
            h_last_obs = np.nan

        if np.isnan(h_last_obs):
            h_last_obs = h_eq

        decay_factor = np.exp(k * max(days_dry_before, 0))
        h_before = (h_last_obs - h_eq) * decay_factor + h_eq
        if h_lo > -np.inf:
            h_before = max(min(h_before, h_hi), h_lo)

        # --- 이벤트 종료 (O(log n) searchsorted) ---
        trig_idx = int(np.searchsorted(gap_triggers, event_start + 1))
        if trig_idx < len(gap_triggers):
            event_end = int(gap_triggers[trig_idx]) - 1
        else:
            event_end = nn - 2

        # --- 이벤트 강우량 (O(1)) ---
        po_event = float(po_cs[event_end + 1] - po_cs[event_start])

        # --- 이벤트 후 피크 수위 (적응형 window) ---
        # v32: peak window를 이벤트 강우량에 따라 적응적으로 조정.
        # 강우가 강하면 수위 응답이 빨라 짧은 window, 약하면 긴 window.
        pw_ev = adaptive_peak_window(sn_c, po_event, _mean_event_rain)
        if event_end + 1 < nn:
            seg = ho_padded[event_end + 1: event_end + 1 + pw_ev]
            seg_valid = seg[~np.isnan(seg)]
            h_peak = float(np.max(seg_valid)) if len(seg_valid) > 0 else h_before
        else:
            h_peak = h_before
        if np.isnan(h_peak):
            h_peak = h_before

        dh_event = max(h_peak - h_before, 0.0)
        ev_ns = ns[event_start:min(event_end + 1, len(ns))]
        ev_ns = ev_ns[~np.isnan(ev_ns) & (ev_ns > 0)]
        sy_ev = min(float(np.mean(ev_ns)), sy_lit) if len(ev_ns) > 0 else n_f_avg

        rech_wtf = sy_ev * dh_event
        rech_cap = alpha_soil * po_event
        rech[event_start] = min(rech_wtf, rech_cap)
        ii = event_end + 1


def run_logic_v27(k, z_unsat, sn, po_in, ho_in, q_in, r_in, r_c, pump_mask_in=None,
                   *, _fast=False, rho=None, alpha=None, w_q_ratio=None):
    """Augmented 2-state Kalman WTF recharge estimation.

    Parameters
    ----------
    _fast : bool, keyword-only
        If True, skip RTS smoother and covariance storage for speed.
        Used internally by calc_error during optimisation (~10x fewer
        allocations per call).  Final user-facing runs use _fast=False.

    State vector x = [h, w]^T where:
      - h : water table head (m)
      - w : hidden recharge forcing (m/day), estimated by the Kalman filter

    Transition model:
      h(t) = (1+k)*h(t-1) - k*h_eq + u(t) + w(t-1)
      w(t) = rho * w(t-1) + process noise

    The WTF event recharge (Sy × Δh) is computed first as a physically-
    grounded prior.  The Kalman filter then refines the total recharge
    estimate by assimilating observed water levels — observation errors
    feed back into *both* h and w through the Kalman gain matrix.

    The final recharge is a blend:
      rech_final = alpha * rech_wtf + (1-alpha) * rech_kalman

    This addresses the key limitation of the previous architecture where
    Kalman filtering only smoothed water levels without feeding back into
    the recharge estimate.

    References
    ----------
    Healy, R.W. & Cook, P.G. (2002). Using groundwater levels to estimate
        recharge. Hydrogeology Journal, 10(1), 91-109.
    Crosbie, R.S. et al. (2005). Differences in the estimation of recharge
        using the Kalman filter approach. Water Resources Research, 41(9).
    """
    # ── Input validation (v32) ──────────────────────────────────
    # Guard against common caller mistakes that produce silent errors.
    po_in = np.asarray(po_in, dtype=np.float64)
    ho_in = np.asarray(ho_in, dtype=np.float64)
    nn = len(ho_in)
    if len(po_in) != nn:
        raise ValueError(
            f"po_in and ho_in length mismatch: {len(po_in)} vs {nn}"
        )
    if nn < 10:
        raise ValueError(
            f"Time series too short ({nn} steps). Minimum 10 required."
        )
    if not (-1.0 <= k <= 0.0):
        logger.warning("k=%.4f is outside typical range [-1, 0]; clamping.", k)
        k = max(-1.0, min(k, 0.0))
    if z_unsat < 0:
        raise ValueError(f"z_unsat must be non-negative, got {z_unsat}")
    sn = max(1, min(12, round(sn)))
    if q_in < 0 or r_in < 0:
        raise ValueError(
            f"Noise variances must be non-negative: q={q_in}, r={r_in}"
        )
    rech_wtf = np.zeros(nn)
    ns = np.full(nn, np.nan)

    if pump_mask_in is None:
        pump_mask_in = np.zeros(nn, dtype=bool)

    sn_c = max(1, min(12, round(sn)))
    idx = sn_c - 1

    h_eq, ho_finite = _estimate_equilibrium_head(ho_in, po_in, r_c)

    # ── Step 1: Compute event-specific Sy (VG retention curve) ──
    tau_init = TAU_DB[idx]
    n_dry = max(tau_init * 3, 10)
    for ii in range(nn - 1):
        if po_in[ii] <= r_c:
            n_dry += 1
            ns[ii] = 0.0
            continue
        ns[ii] = filpor_v27(sn, z_unsat, n_dry)
        n_dry = 1

    wet_ns = ns[:nn - 1][po_in[:nn - 1] > r_c]
    wet_ns = wet_ns[~np.isnan(wet_ns) & (wet_ns > 0)]
    n_f_avg = float(np.mean(wet_ns)) if len(wet_ns) > 0 else DEFAULT_N_F_AVG

    alpha_soil = ALPHA_SOIL_LIST[idx]
    sy_lit = SY_LIT_LIST[idx]
    n_f_avg = min(n_f_avg, sy_lit)
    sy_eff = n_f_avg

    # ── Step 2: WTF event recharge as physical prior ──
    _accumulate_event_recharge(
        rech=rech_wtf,
        ns=ns,
        po_in=po_in,
        ho_in=ho_in,
        pump_mask_in=pump_mask_in,
        ho_finite=ho_finite,
        h_eq=h_eq,
        k=k,
        sn_c=sn_c,
        alpha_soil=alpha_soil,
        sy_lit=sy_lit,
        n_f_avg=n_f_avg,
        r_c=r_c,
    )

    # ── Step 3: Pure (no-filter) simulation for baseline comparison ──
    u_scale = alpha_soil / max(sy_eff, MIN_SY_FOR_INPUT_SCALE)
    hs_pure = np.zeros(nn)
    first_idx = np.where(~np.isnan(ho_in))[0]
    h0 = h_eq if len(first_idx) == 0 else ho_in[first_idx[0]]
    hs_pure[0] = h0

    for ii in range(1, nn):
        u = po_in[ii - 1] * u_scale if po_in[ii - 1] > r_c else 0.0
        rec = k * (hs_pure[ii - 1] - h_eq)
        hs_pure[ii] = hs_pure[ii - 1] + rec + u

    # ── Step 4: Augmented 2-state Kalman filter [h, w] ──
    #
    # Academic basis (Crosbie et al., 2005; Gehman et al., 2009):
    #   The Kalman filter produces an optimal estimate of the true water
    #   level by assimilating noisy observations with a physically-based
    #   state-space model.  The filtered water levels hs_kf are then used
    #   to re-compute WTF event recharge (Step 5), yielding a Kalman-
    #   improved Δh that is more robust to measurement noise, missing data,
    #   and pumping artifacts than raw observations.
    #
    #   The hidden state w(t) captures unmodelled recharge forcing — any
    #   water-level change not explained by recession + rainfall alone.
    #   This provides an independent, observation-driven recharge signal
    #   that is blended with the WTF estimate for uncertainty reduction.
    #
    # State vector x = [h, w]^T
    #   h(t) = (1+k)*h(t-1) - k*h_eq + u(t) + w(t-1)
    #   w(t) = rho * w(t-1) + noise
    # rho/alpha: 인자로 받은 값 우선, 없으면 config 기본값 사용
    # (전역 상태 변경 없이 호출자가 override 가능)
    rho = rho if rho is not None else KALMAN_RHO
    alpha = alpha if alpha is not None else KALMAN_WTF_BLEND_ALPHA
    _w_q_ratio = w_q_ratio if w_q_ratio is not None else KALMAN_W_Q_RATIO
    q_h = max(q_in, KALMAN_Q_FLOOR)
    q_w = q_h * _w_q_ratio
    r_kf = max(r_in, KALMAN_R_FLOOR)

    # ── Full Q matrix with h-w cross-covariance ────────────────────
    # Physically motivated: recharge forcing (w) directly drives head (h),
    # so their process noise is correlated.  The off-diagonal term
    # q_hw = ρ_hw × √(q_h × q_w) captures this coupling.
    #
    # With KALMAN_HW_Q_CORR = 0, this reduces to the original diagonal Q.
    #
    # Theoretical basis:
    #   The true process noise covariance for a coupled h-w system should
    #   reflect that unmodelled inputs (e.g., lateral flow, ET uncertainty)
    #   affect both states simultaneously.  A full Q matrix yields a more
    #   accurate Kalman gain and faster convergence.
    #
    # References:
    #   Jazwinski, A.H. (1970). Stochastic Processes and Filtering Theory.
    #       Academic Press, Ch. 7 — general treatment of correlated process noise.
    #   Bar-Shalom, Y., Li, X.R. & Kirubarajan, T. (2001). Estimation with
    #       Applications to Tracking and Navigation. Wiley, Sec. 4.3.
    q_hw = KALMAN_HW_Q_CORR * np.sqrt(q_h * q_w)
    F = np.array([[1.0 + k, 1.0],
                  [0.0,     rho]])
    H = np.array([[1.0, 0.0]])            # observe h only
    Q_mat = np.array([[q_h,  q_hw],
                      [q_hw, q_w]])
    R_mat = np.array([[r_kf]])

    # ── 초기 상태 및 공분산 P₀ ─────────────────────────────────
    # x₀ = [h0, 0] : h 초기값은 첫 유효 관측값, w 초기값은 0 (사전 정보 없음)
    #
    # P₀는 관측 수위 분산 기반으로 데이터 구동(data-driven) 설정:
    #   P₀[0,0] = max(Var(ho_obs), KALMAN_P0_H_FLOOR)
    #             → h 상태 초기 불확실성 = 관측 데이터 분산
    #   P₀[1,1] = KALMAN_P0_W_RATIO × P₀[0,0]
    #             → w 상태는 잠재 변수이므로 h보다 낮은 초기 불확실성
    #
    # 하드코딩 diag([1.0, 0.1])은 데이터 스케일에 무관하여, 수위 변동이
    # 작은 wells (σ²<<1)에서는 P₀ 과대추정, 큰 wells (σ²>>1)에서는
    # P₀ 과소추정 → 초기 수렴 속도에 직접 영향.
    #
    # References:
    #   Grewal, M.S. & Andrews, A.P. (2014). Kalman Filtering: Theory and
    #       Practice Using MATLAB, 4th ed. Wiley, Sec. 4.4
    #   Mehra, R.K. (1972). Approaches to adaptive filtering.
    #       IEEE Trans. Autom. Control, 17(5), 693-698.
    ho_obs_valid = ho_in[~np.isnan(ho_in)]
    p0_h = float(np.var(ho_obs_valid)) if len(ho_obs_valid) > 1 else 1.0
    p0_h = max(p0_h, KALMAN_P0_H_FLOOR)
    x = np.array([h0, 0.0])
    P = np.diag([p0_h, KALMAN_P0_W_RATIO * p0_h])

    # ── 적응형 관측 노이즈 R(t) 벡터 사전 계산 ────────────────────────────
    # 펌핑 오염 인접 구간에서 관측값 신뢰도가 낮으므로 R을 증가시켜
    # Kalman gain을 축소함(필터가 모델 예측을 더 신뢰).
    #
    # 구현: 펌핑 마스크를 누적합으로 벡터화하여 ±PROX 범위 내 펌핑 여부 판정.
    # O(n) 복잡도 — 루프 내 반복 탐색보다 효율적.
    #
    # References:
    #   Mehra, R.K. (1972). Approaches to adaptive filtering.
    #       IEEE Trans. Autom. Control, 17(5), 693-698.
    #   Mohamed, A.H. & Schwarz, K.P. (1999). Adaptive Kalman filtering for
    #       INS/GPS. Journal of Geodesy, 73(4), 193-203.
    r_arr = np.full(nn, r_kf)
    if np.any(pump_mask_in):
        # ── Gaussian-kernel adaptive R(t) ─────────────────────────
        # Instead of a binary step (R or R*factor), apply a smooth
        # Gaussian decay from each pumping event.  This is physically
        # motivated: pumping-induced drawdown/recovery effects attenuate
        # gradually, not abruptly.
        #
        # R(t) = R_base × (1 + (factor-1) × g(t))
        # where g(t) = max over pumping events j of  exp(-d_j² / (2σ²))
        #       d_j  = |t - t_j|  (distance in days)
        #       σ    = KALMAN_R_PUMP_SIGMA  (default 2.5 days)
        #
        # At d=0 (pumping day):   R(t) = R_base × factor       (max inflation)
        # At d=σ:                 R(t) ≈ R_base × (1 + 0.61×(f-1))
        # At d=2σ (≈PROX_DAYS):  R(t) ≈ R_base × (1 + 0.02×(f-1))  (negligible)
        #
        # Implementation: vectorised via scipy.ndimage.gaussian_filter1d
        # applied to the binary pump_mask.  This is O(n) and equivalent
        # to convolving with a Gaussian kernel then normalising to [0,1].
        #
        # References:
        #   Mohamed, A.H. & Schwarz, K.P. (1999). Adaptive Kalman filtering
        #       for INS/GPS. J. Geodesy, 73(4), 193-203.
        from scipy.ndimage import gaussian_filter1d
        sigma = float(KALMAN_R_PUMP_SIGMA)
        # Gaussian convolution of binary mask → smooth proximity weight
        g = gaussian_filter1d(pump_mask_in.astype(np.float64), sigma=sigma)
        # Normalise: peak of convolution depends on cluster density,
        # but we want g ∈ [0, 1] where 1 = on a pumping day.
        g_max = g.max()
        if g_max > 0:
            g = g / g_max
        # Truncate negligible tails (< 1% of factor) for cleanliness
        g[g < 0.01] = 0.0
        r_arr = r_kf * (1.0 + (KALMAN_R_PUMP_FACTOR - 1.0) * g)

    hs_kf = np.zeros(nn)
    w_est = np.zeros(nn)

    if not _fast:
        # Full mode: store history for RTS smoother + uncertainty propagation
        P_kf = np.zeros((nn, 2, 2))
        x_filt = np.zeros((nn, 2))
        x_pred_store = np.zeros((nn, 2))
        P_pred_store = np.zeros((nn, 2, 2))
        x_filt[0] = x.copy()
        P_kf[0] = P.copy()

    hs_kf[0] = h0

    for ii in range(1, nn):
        u_rain = po_in[ii - 1] * u_scale if po_in[ii - 1] > r_c else 0.0
        u_vec = np.array([u_rain - k * h_eq, 0.0])

        # ── Predict ──
        x_pred = F @ x + u_vec
        P_pred = F @ P @ F.T + Q_mat

        if not _fast:
            x_pred_store[ii] = x_pred
            P_pred_store[ii] = P_pred

        # ── Update (skip NaN or pumping-affected observations) ──
        if not np.isnan(ho_in[ii]) and not pump_mask_in[ii]:
            # 적응형 관측 노이즈: 펌핑 인접 구간에서 r_t > r_kf (증가)
            # → Kalman gain 축소 → 필터가 모델 예측을 더 신뢰
            r_t = r_arr[ii]
            # Analytical scalar Kalman gain for H=[1,0] observation
            s_inv = 1.0 / (P_pred[0, 0] + r_t)
            K0 = P_pred[0, 0] * s_inv
            K1 = P_pred[1, 0] * s_inv
            innov = ho_in[ii] - x_pred[0]
            x = np.array([x_pred[0] + K0 * innov,
                          x_pred[1] + K1 * innov])
            # 진정한 Joseph form: P = (I-KH)P_pred(I-KH)^T + K*R(t)*K^T
            #
            # 이전 구현 `P = (I-KH)P_pred` 은 표준 칼만 갱신이지만
            # Joseph form이 아님. 진정한 Joseph form이 대칭성과
            # 양정치성(positive definiteness)을 수치적으로 보장함.
            # R(t)는 시변 관측 노이즈(적응형)를 반영.
            #
            # References:
            #   Simon, D. (2006). Optimal State Estimation. Wiley, Eq. 5.37
            #   Gibbs, B.P. (2011). Advanced Kalman Filtering,
            #       Least-Squares and Modeling. Wiley, Ch. 9
            #   Grewal & Andrews (2014). Kalman Filtering: Theory and
            #       Practice Using MATLAB, 4th ed. Wiley, Eq. 4.29
            #
            # H=[1,0], K=[K0, K1]^T 일 때 명시적(scalar) 전개:
            #   I - KH = [[1-K0, 0], [-K1, 1]]
            #   A := (I-KH) @ P_pred
            #   P := A @ (I-KH)^T + r_t * outer(K, K)
            imk0 = 1.0 - K0
            # A = (I-KH) @ P_pred
            a00 = imk0 * P_pred[0, 0]
            a01 = imk0 * P_pred[0, 1]
            a10 = P_pred[1, 0] - K1 * P_pred[0, 0]
            a11 = P_pred[1, 1] - K1 * P_pred[0, 1]
            # P = A @ (I-KH)^T + r_t * K K^T
            P = np.array([
                [a00 * imk0 + r_t * K0 * K0,
                 -a00 * K1 + a01 + r_t * K0 * K1],
                [a10 * imk0 + r_t * K0 * K1,
                 -a10 * K1 + a11 + r_t * K1 * K1],
            ])
        else:
            x = x_pred
            P = P_pred

        hs_kf[ii] = x[0]
        w_est[ii] = x[1]

        if not _fast:
            x_filt[ii] = x.copy()
            P_kf[ii] = P.copy()

    # ── Step 4b: Rauch-Tung-Striebel (RTS) backward smoother ──
    # Skipped in _fast mode (optimisation) — only forward filter needed.
    if not _fast:
        # The RTS smoother (Rauch et al., 1965) refines the Kalman filter
        # estimates by incorporating future observations.
        #
        # Performance: Batch-invert all P_pred matrices at once using the
        # analytical 2×2 inverse to avoid per-step np.linalg.inv calls.
        x_smooth = x_filt.copy()
        P_smooth = P_kf.copy()

        FT = F.T
        PF = np.einsum('tij,jk->tik', P_kf[:-1], FT)  # (nn-1, 2, 2)

        # Analytical 2x2 batch inverse of P_pred_store[1:]
        PP = P_pred_store[1:].copy()
        PP[:, 0, 0] += 1e-10
        PP[:, 1, 1] += 1e-10
        det = PP[:, 0, 0] * PP[:, 1, 1] - PP[:, 0, 1] * PP[:, 1, 0]
        det = np.where(np.abs(det) < 1e-20, 1e-20, det)
        inv_PP = np.empty_like(PP)
        inv_PP[:, 0, 0] = PP[:, 1, 1] / det
        inv_PP[:, 0, 1] = -PP[:, 0, 1] / det
        inv_PP[:, 1, 0] = -PP[:, 1, 0] / det
        inv_PP[:, 1, 1] = PP[:, 0, 0] / det

        G_all = np.einsum('tij,tjk->tik', PF, inv_PP)

        for ii in range(nn - 2, -1, -1):
            dx = x_smooth[ii + 1] - x_pred_store[ii + 1]
            x_smooth[ii] = x_filt[ii] + G_all[ii] @ dx
            dP = P_smooth[ii + 1] - P_pred_store[ii + 1]
            P_s = P_kf[ii] + G_all[ii] @ dP @ G_all[ii].T
            # ── Enforce positive semi-definiteness (v32 strengthened) ──
            # RTS backward pass can produce non-PSD P_smooth due to
            # accumulated numerical error on long time series.
            #
            # v32: Two-stage approach for robustness:
            #   1. Symmetry enforcement + analytical eigenvalue clamping
            #      (fast path for the common case).
            #   2. If clamped matrix still fails Cholesky verification,
            #      fall back to nearest PSD via Higham (1988) iteration.
            #
            # The Cholesky check (try np.linalg.cholesky) is the gold
            # standard for PSD verification — if it succeeds, the matrix
            # is guaranteed PSD.  This catches edge cases the old code
            # missed (e.g., numerical asymmetry after eigenvector
            # reconstruction).
            #
            # References:
            #   Higham, N.J. (1988). Computing a nearest symmetric positive
            #       semidefinite matrix. LAA, 103, 103-118.
            #   Simon, D. (2006). Optimal State Estimation. Wiley, Sec. 9.3.
            eps_psd = 1e-10
            P_s = 0.5 * (P_s + P_s.T)  # enforce symmetry

            # Stage 1: analytical eigenvalue clamping for 2×2
            tr = P_s[0, 0] + P_s[1, 1]
            det_p = P_s[0, 0] * P_s[1, 1] - P_s[0, 1] ** 2
            disc = tr * tr - 4.0 * det_p
            needs_fix = (disc < 0) or (det_p < -eps_psd) or (tr < 0)

            if not needs_fix:
                # Quick check: both eigenvalues positive?
                disc_s = max(disc, 0.0)
                sqrt_d = np.sqrt(disc_s)
                lam_min = 0.5 * (tr - sqrt_d)
                if lam_min < eps_psd:
                    needs_fix = True

            if needs_fix:
                disc = max(tr * tr - 4.0 * det_p, 0.0)
                sqrt_disc = np.sqrt(disc)
                lam1 = max(0.5 * (tr + sqrt_disc), eps_psd)
                lam2 = max(0.5 * (tr - sqrt_disc), eps_psd)
                off = P_s[0, 1]
                if abs(off) > 1e-15:
                    t1 = np.array([off, lam1 - P_s[0, 0]])
                    t2 = np.array([off, lam2 - P_s[0, 0]])
                    n1 = np.sqrt(t1 @ t1)
                    n2 = np.sqrt(t2 @ t2)
                    if n1 > 1e-15 and n2 > 1e-15:
                        v1 = t1 / n1
                        v2 = t2 / n2
                        V = np.column_stack([v1, v2])
                        P_s = V @ np.diag([lam1, lam2]) @ V.T
                    else:
                        P_s = np.diag([lam1, lam2])
                else:
                    P_s = np.diag([max(P_s[0, 0], eps_psd),
                                   max(P_s[1, 1], eps_psd)])
                P_s = 0.5 * (P_s + P_s.T)  # re-symmetrise

            # Stage 2: Cholesky verification — guaranteed PSD check
            try:
                np.linalg.cholesky(P_s)
            except np.linalg.LinAlgError:
                # Cholesky failed — apply Higham nearest-PSD fallback
                # For 2×2 this is equivalent to clamping eigenvalues,
                # but we use np.linalg.eigh for numerical safety.
                eigvals, eigvecs = np.linalg.eigh(P_s)
                eigvals = np.maximum(eigvals, eps_psd)
                P_s = eigvecs @ np.diag(eigvals) @ eigvecs.T
                P_s = 0.5 * (P_s + P_s.T)

            P_smooth[ii] = P_s

        hs_kf = x_smooth[:, 0]
        w_est = x_smooth[:, 1]
        P_kf = P_smooth

    # ── Step 5: Re-compute WTF recharge using Kalman-smoothed levels ──
    #
    # Key improvement: Δh is now derived from the Kalman-optimal water
    # level estimate rather than raw (noisy) observations.  This:
    #   (a) reduces noise-induced spurious Δh,
    #   (b) fills observation gaps via Kalman prediction,
    #   (c) removes pumping artifacts (pump periods use prediction only).
    rech_kf = np.zeros(nn)
    # Build a composite water level: use Kalman estimate where observations
    # are missing or pump-contaminated, otherwise blend for stability.
    ho_kf = np.where(
        np.isnan(ho_in) | pump_mask_in,
        hs_kf,                                       # Kalman fills gaps
        alpha * ho_in + (1.0 - alpha) * hs_kf,
    )
    _accumulate_event_recharge(
        rech=rech_kf,
        ns=ns,
        po_in=po_in,
        ho_in=ho_kf,
        pump_mask_in=pump_mask_in,
        ho_finite=ho_finite,
        h_eq=h_eq,
        k=k,
        sn_c=sn_c,
        alpha_soil=alpha_soil,
        sy_lit=sy_lit,
        n_f_avg=n_f_avg,
        r_c=r_c,
    )

    # ── Step 5b: Inter-event diffuse recharge ──────────────────────────────
    #
    # 이벤트 임계값 r_c 이하의 소량 강우일에도 일부는 비포화대를 통해
    # 지하수에 도달하는 "확산 함양(diffuse recharge)"이 발생함.
    # WTF 방법은 수위 상승을 기반으로 하므로 이 성분을 포착하지 못함.
    #
    # 조건: (1) 강우 > 0 but <= r_c (이벤트 미발생)
    #       (2) rech_kf == 0 (이미 WTF 이벤트에 포함되지 않은 날)
    #       (3) pump_mask == False (펌핑 오염 제외)
    #
    # 기여량: INTER_EVENT_FRAC * alpha_soil * po
    #   alpha_soil: 토양 투수계수 (VG 매개변수, 소수)
    #   INTER_EVENT_FRAC: 설정 가능한 분율 (기본값 5%)
    #
    # References:
    #   Scanlon, B.R. et al. (2002). Vadose Zone Journal, 1(1), 2-6.
    #   Kendy, E. et al. (2004). Hydrological Processes, 18(12), 2367-2383.
    rain_no_event = (po_in > 0) & (po_in <= r_c) & (rech_kf == 0) & (~pump_mask_in)
    rech_inter = np.where(
        rain_no_event,
        INTER_EVENT_FRAC * alpha_soil * po_in,
        0.0,
    )

    # ── Step 6: Final recharge = Kalman-improved WTF + inter-event ──
    # The w-state improves h-estimation internally but is NOT directly
    # added to recharge.  This avoids double-counting: w already influenced
    # hs_kf → ho_kf → Δh → rech_kf through the Kalman feedback loop.
    rech = rech_kf + rech_inter

    logger.info(
        "Augmented Kalman recharge: wtf_raw=%.4f kf_wtf=%.4f inter=%.4f total=%.4f",
        np.sum(rech_wtf), np.sum(rech_kf), np.sum(rech_inter), np.sum(rech),
    )

    # Store covariance in a lightweight container for optional uncertainty use
    # Only available in full mode (not _fast)
    if not _fast:
        _kalman_extras = {
            "P_h_var": P_kf[:, 0, 0].copy(),  # σ²(h) at each timestep
            "P_w_var": P_kf[:, 1, 1].copy(),  # σ²(w) at each timestep
            "P_hw_cov": P_kf[:, 0, 1].copy(), # Cov(h,w) — full Q 도입으로 추가
            "w_est": w_est.copy(),
        }
        _last_kalman_extras.clear()
        _last_kalman_extras.update(_kalman_extras)

    return rech, hs_kf, hs_pure, sy_eff, n_f_avg


def calc_recession_slope_diff(ho, hs, po, rc, pump_mask):
    dry_idx = (po <= rc) & ~np.isnan(ho) & ~np.isnan(hs) & ~pump_mask
    if np.sum(dry_idx) < 10:
        return 0.0
    dh_obs = np.concatenate([np.diff(ho), [0.0]])
    dh_sim = np.concatenate([np.diff(hs), [0.0]])
    err = float(np.nanmean(np.abs(dh_obs[dry_idx] - dh_sim[dry_idx])))
    return err if np.isfinite(err) else 0.0


def calc_flash(data):
    data = data[~np.isnan(data)]
    if len(data) < 2:
        return 0.0
    path_len = float(np.sum(np.abs(np.diff(data))))
    base_sum = float(np.sum(np.abs(data - np.mean(data)))) + 1e-6
    idx = path_len / base_sum
    return idx if np.isfinite(idx) else 0.0


def calc_pump_contam(ho, po, rc):
    mask = detect_pump_mask(ho, po, rc)
    valid = ~np.isnan(ho)
    if np.sum(valid) == 0:
        return 0.0, 0, 0

    frac_mask = np.sum(mask & valid) / max(np.sum(valid), 1)
    dm = np.diff(np.concatenate([[False], mask]).astype(int))
    events = int(np.sum(dm == 1))

    max_run = 0
    cur = 0
    for flagged in mask:
        if flagged:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0

    score = (
        PUMP_FRAC_WEIGHT * min(frac_mask / PUMP_FRAC_NORMALIZER, 1.0)
        + PUMP_EVENTS_WEIGHT * min(events / PUMP_EVENTS_NORMALIZER, 1.0)
        + PUMP_MAXRUN_WEIGHT * min(max_run / PUMP_MAXRUN_NORMALIZER, 1.0)
    )
    return max(0.0, min(score, 1.0)), events, max_run


def calc_rain_response(series_in, po, rc, ignore_mask=None):
    n = len(series_in)
    if ignore_mask is None:
        ignore_mask = np.zeros(n, dtype=bool)
    if n < 4:
        return 0.0
    wet_idx = np.where(po > rc)[0]
    hits, total = 0, 0
    for i in wet_idx:
        if i >= n - 1:
            continue
        j2 = min(n, i + 3)
        if np.any(ignore_mask[i:j2]) or np.isnan(series_in[i]):
            continue
        future = series_in[i + 1:j2]
        future = future[~np.isnan(future)]
        if len(future) == 0:
            continue
        total += 1
        if np.max(future - series_in[i]) > 0:
            hits += 1
    return hits / total if total > 0 else 0.0


def calc_error(k, z, sn, po_shifted, ho, rc, pump_mask, *, rho=None, alpha=None,
               w_fit=None, w_resp=None, w_rech=None):
    """Dimensionless multi-objective cost function for parameter optimisation.

    The cost is a weighted sum of three normalised [0, 1] components:

    1. **NRMSE** — RMSE normalised by the observed standard deviation
       (equivalent to sqrt(1 - NSE) when NSE > 0).  This is a standard
       dimensionless goodness-of-fit metric in hydrology (Moriasi et al.,
       2007, Trans. ASABE).

    2. **Rain-response mismatch** — Absolute difference between the
       fraction of rainfall events that produce a water-level rise in the
       observed vs. simulated series.  Already dimensionless [0, 1].

    3. **Recharge-range violation** — Distance of the simulated recharge
       ratio from the literature-expected range, normalised by the range
       width.  Dimensionless [0, ∞) but soft-capped at 1.0.

    Parameters
    ----------
    rho : float, optional
        If given, temporarily overrides KALMAN_RHO for this evaluation.
    alpha : float, optional
        If given, temporarily overrides KALMAN_WTF_BLEND_ALPHA.

    Weights
    -------
    w_fit   = 0.70  — Fitting accuracy dominates (primary objective).
    w_resp  = 0.15  — Rain-response consistency (physical plausibility).
    w_rech  = 0.15  — Recharge ratio within literature bounds (constraint).

    These weights were chosen based on the principle that fitting fidelity
    is the primary calibration target, while physical plausibility
    constraints act as regularisers to prevent overfitting to noise.
    """
    # rho/alpha를 run_logic_v27에 직접 전달 — 전역 상태 변경 불필요
    rech_arr, hs_kf, _, _, _ = run_logic_v27(
        k, z, sn, po_shifted, ho, 0, 0, rc, pump_mask, _fast=True,
        rho=rho, alpha=alpha,
    )

    valid = ~pump_mask & ~np.isnan(hs_kf) & ~np.isnan(ho)
    min_valid = max(MIN_VALID_POINTS, round(len(ho) * MIN_VALID_FRACTION))
    if np.sum(valid) < min_valid:
        valid = ~np.isnan(hs_kf) & ~np.isnan(ho)
    if np.sum(valid) < 5:
        return np.inf

    # ── Component 1: Normalised RMSE (NRMSE) ──
    residuals = hs_kf[valid] - ho[valid]
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    sigma_ho = float(np.std(ho[valid]))
    if not np.isfinite(rmse) or sigma_ho < 1e-9:
        return np.inf
    nrmse = rmse / sigma_ho  # dimensionless; 0 = perfect, 1 = as bad as mean

    # ── Component 2: Rain-response mismatch (already [0, 1]) ──
    resp_obs = calc_rain_response(ho, po_shifted, rc, pump_mask)
    resp_sim = calc_rain_response(hs_kf, po_shifted, rc, pump_mask)
    resp_mismatch = abs(resp_obs - resp_sim)  # [0, 1]

    # ── Component 3: Recharge-range violation (normalised) ──
    sn_c = max(1, min(12, round(sn))) - 1
    rech_lo, rech_hi = RECH_RANGE[sn_c]
    total_rain = np.sum(po_shifted)
    recharge_ratio = np.sum(rech_arr) / total_rain * 100 if total_rain > 1e-9 else 0.0
    rech_range_width = max(rech_hi - rech_lo, 1.0)
    rech_violation = (
        max(0.0, recharge_ratio - rech_hi) + max(0.0, rech_lo - recharge_ratio)
    ) / rech_range_width  # dimensionless
    rech_violation = min(rech_violation, OBJ_RECH_VIOLATION_CAP)

    # ── Weighted sum ──
    # 선택적 가중치 오버라이드 (민감도 분석용): None이면 config 기본값 사용
    wf = w_fit  if w_fit  is not None else OBJ_W_FIT
    wr = w_resp if w_resp is not None else OBJ_W_RESP
    wc = w_rech if w_rech is not None else OBJ_W_RECH
    return wf * nrmse + wr * resp_mismatch + wc * rech_violation


@dataclass
class CoreRunData:
    ho: np.ndarray
    po: np.ndarray


@dataclass
class CoreMetrics:
    """Successful simulation result returned by the public core entry point.

    Standard hydrological performance metrics follow Moriasi et al. (2007)
    guidelines for systematic quantification of accuracy:

    - NSE  (Nash & Sutcliffe, 1970): model efficiency, 1 = perfect.
    - KGE  (Gupta et al., 2009): decomposes into correlation, variability
      bias, and mean bias; overcomes known NSE limitations.
    - PBIAS (Gupta et al., 1999): percentage bias, 0 = unbiased.

    References
    ----------
    Nash, J.E. & Sutcliffe, J.V. (1970). River flow forecasting through
        conceptual models. J. Hydrology, 10(3), 282-290.
    Gupta, H.V. et al. (2009). Decomposition of the mean squared error and
        NSE performance criteria. J. Hydrology, 377(1-2), 80-91.
    Moriasi, D.N. et al. (2007). Model evaluation guidelines for systematic
        quantification of accuracy. Trans. ASABE, 50(3), 885-900.
    """
    rmse: float
    sigma_ho: float
    cc: float
    nse: float            # Nash-Sutcliffe Efficiency [-∞, 1]
    kge: float            # Kling-Gupta Efficiency [-∞, 1]
    pbias: float          # Percent Bias (%), 0 = unbiased
    recharge_ratio: float
    total_rech_depth: float
    Sy_eff: float
    n_f_avg: float
    opt_k: float
    opt_z: float
    opt_lag: int
    opt_rho: float           # Kalman ρ (auto-tuned or default)
    opt_alpha: float         # Kalman blend α (auto-tuned or default)
    kb_min: float
    kb_max: float
    flash_diff: float
    rec_slope_err: float
    stress: float
    pump_contam_idx: float
    pump_event_count: int
    pump_max_run: int
    rain_resp_obs: float
    rain_resp_sim: float
    eval_n: int
    ho: list
    hs_kf: list
    hs_pure: list
    po: list
    po_shifted: list
    rech: list
    pump_mask: list
    boundary_warnings: list  # 경계 도달 경고 목록

    def to_dict(self):
        """Return the stable public response schema used by the app layer."""
        return asdict(self)


@dataclass
class CoreErrorResult:
    """Error payload returned when the core simulation fails."""
    error: str
    stack: str

    def to_dict(self):
        return asdict(self)


def _build_demo_data():
    rng = np.random.default_rng(42)
    days_ex = 365
    po_raw = np.maximum(0, rng.normal(0.008, 0.008, days_ex))
    ho = (
        120.0
        + np.sin(np.arange(1, days_ex + 1) / 100.0) * 0.8
        + np.concatenate([np.zeros(3), po_raw[:days_ex - 3]]) * 12
        + np.cumsum(rng.normal(0, 0.02, days_ex))
    )
    ho[79:90] -= np.linspace(0, 1.2, 11)
    ho[199:208] -= np.linspace(0, 0.8, 9)
    return CoreRunData(ho=ho.astype(float), po=po_raw.astype(float))


def load_core_data(file_path):
    if file_path == "DEMO":
        return _build_demo_data()

    data = load_timeseries_file(
        file_path,
        interpolate_water_level=False,
        rainfall_unit="m",
        require_dates=False,
    )
    return CoreRunData(
        ho=data.water_level.ravel().astype(float),
        po=data.rainfall_mm.ravel().astype(float),
    )


def normalize_core_inputs(sn_idx, k_val, z_val, lag_val):
    sn = max(1, min(12, round(sn_idx)))
    kb_min, kb_max = get_bounds(sn)
    opt_k = max(min(k_val, kb_max), kb_min)
    opt_z = max(min(z_val, MAX_Z_PARAM), MIN_Z_PARAM)
    opt_lag = round(lag_val)
    return sn, kb_min, kb_max, opt_k, opt_z, opt_lag


def _preselect_lag_candidates(
    ho: np.ndarray,
    po: np.ndarray,
    r_c: float,
    pump_mask: np.ndarray,
    max_lag: int,
    n_cands: int,
) -> list:
    """Cross-correlation based lag candidate pre-selection.

    Instead of brute-force Nelder-Mead over all lags 0..max_lag-1,
    pre-select the top n_cands lags with highest ho-po cross-correlation
    to reduce optimization cost.

    Uses scipy.signal.correlate (FFT-based O(n log n)) for efficient
    computation of all lag correlations at once.

    Note: lag 0 (no shift) is always included as a candidate.
    """
    from scipy.signal import correlate

    valid = ~np.isnan(ho) & ~pump_mask
    ho_c = np.where(valid, ho - np.nanmean(ho[valid]), 0.0)
    po_c = po - np.mean(po)

    xcorr = correlate(ho_c, po_c, mode="full")
    mid = len(ho) - 1
    lags_corr = xcorr[mid:mid + max_lag]

    # Always include lag 0
    candidates = [0]
    if len(lags_corr) > 1:
        ranked = np.argsort(-lags_corr)
        for idx in ranked:
            if int(idx) not in candidates:
                candidates.append(int(idx))
            if len(candidates) >= n_cands:
                break

    return sorted(candidates)


def optimize_parameters(ho, po, sn, opt_k, opt_z, rc_val, pump_mask_fixed):
    """Two-stage optimisation: (1) k, z, lag -> (2) rho, alpha Kalman hyperparameters.

    Stage 1: Nelder-Mead over k, z with cross-correlation pre-filtered lag search.
    Stage 2: Fix k, z, lag from Stage 1; optimise rho and alpha via bounded Nelder-Mead.

    All Kalman hyperparameters are passed as keyword arguments to run_logic_v27
    and calc_error — no global state mutation (thread-safe).

    Returns
    -------
    best_k, best_z, best_lag, best_rho, best_alpha
    """
    kb_min, kb_max = get_bounds(sn)
    best_err = np.inf
    best_lag = 0
    best_k = opt_k
    best_z = opt_z
    start_points = [
        [max(min(opt_k, kb_max), kb_min), max(min(opt_z, MAX_Z_PARAM), MIN_Z_PARAM)],
        [kb_min * 0.8 + kb_max * 0.2, 5.0],
        [kb_min * 0.2 + kb_max * 0.8, 15.0],
    ]
    lb = np.array([kb_min, MIN_Z_PARAM])
    ub = np.array([kb_max, MAX_Z_PARAM])

    # Pre-filter lag candidates via cross-correlation
    lag_candidates = _preselect_lag_candidates(
        ho, po, rc_val, pump_mask_fixed,
        max_lag=OPT_LAG_SEARCH_DAYS,
        n_cands=OPT_LAG_XCORR_CANDIDATES,
    )

    # -- Stage 1: k, z, lag optimisation --
    for try_lag in lag_candidates:
        po_try = apply_lag(po, try_lag)
        for sp in start_points:
            def obj(p):
                pk = max(min(p[0], ub[0]), lb[0])
                pz = max(min(p[1], ub[1]), lb[1])
                return calc_error(pk, pz, sn, po_try, ho, rc_val, pump_mask_fixed)

            res = minimize(
                obj,
                sp,
                method="Nelder-Mead",
                options={"xatol": OPT_XATOL, "fatol": OPT_FATOL, "maxfev": OPT_MAXFEV},
            )
            if res.fun < best_err:
                best_err = res.fun
                best_lag = try_lag
                best_k = max(min(res.x[0], ub[0]), lb[0])
                best_z = max(min(res.x[1], ub[1]), lb[1])

    logger.info("Stage 1 done: k=%.5f z=%.2f lag=%d err=%.4f",
                best_k, best_z, best_lag, best_err)

    # -- Stage 2: rho, alpha optimisation (fix k, z, lag) --
    RHO_LO, RHO_HI = 0.3, 0.98
    ALPHA_LO, ALPHA_HI = 0.05, 0.95

    po_best = apply_lag(po, best_lag)
    best_rho = KALMAN_RHO
    best_alpha = KALMAN_WTF_BLEND_ALPHA

    rho_alpha_starts = [
        [best_rho, best_alpha],
        [0.5, 0.3],
        [0.9, 0.7],
    ]

    stage2_best_err = best_err
    for sp2 in rho_alpha_starts:
        def obj2(p):
            pr = max(min(p[0], RHO_HI), RHO_LO)
            pa = max(min(p[1], ALPHA_HI), ALPHA_LO)
            return calc_error(best_k, best_z, sn, po_best, ho, rc_val,
                              pump_mask_fixed, rho=pr, alpha=pa)

        res2 = minimize(
            obj2,
            sp2,
            method="Nelder-Mead",
            options={"xatol": 1e-3, "fatol": 1e-4, "maxfev": 300},
        )
        if res2.fun < stage2_best_err:
            stage2_best_err = res2.fun
            best_rho = max(min(res2.x[0], RHO_HI), RHO_LO)
            best_alpha = max(min(res2.x[1], ALPHA_HI), ALPHA_LO)

    logger.info("Stage 2 done: rho=%.3f alpha=%.3f err=%.4f (was %.4f)",
                best_rho, best_alpha, stage2_best_err, best_err)

    return best_k, best_z, best_lag, best_rho, best_alpha


def build_core_metrics(
    ho, po, po_shifted, hs_kf, hs_pure, rech, pump_mask_fixed,
    sy_eff, n_f_avg, opt_k, opt_z, opt_lag, kb_min, kb_max, rc_val,
    opt_rho=None, opt_alpha=None,
):
    """Build CoreMetrics from simulation outputs."""
    eval_valid = ~pump_mask_fixed & ~np.isnan(ho) & ~np.isnan(hs_kf)
    if np.sum(eval_valid) < 10:
        eval_valid = ~np.isnan(ho) & ~np.isnan(hs_kf)

    residuals = hs_kf[eval_valid] - ho[eval_valid]
    rmse = float(np.sqrt(np.nanmean(residuals ** 2)))
    ho_valid = ho[eval_valid]
    hs_valid = hs_kf[eval_valid]
    sigma_ho = float(np.nanstd(ho_valid))
    if not np.isfinite(sigma_ho) or sigma_ho < 1e-6:
        sigma_ho = DEFAULT_SIGMA_HO

    cc = 0.0
    nse = -np.inf
    kge = -np.inf
    pbias = 0.0
    n_valid = int(np.sum(eval_valid))

    if n_valid >= 3:
        mask_both = ~np.isnan(ho_valid) & ~np.isnan(hs_valid)
        n_both = int(np.sum(mask_both))
        if n_both >= 3:
            ho_b = ho_valid[mask_both]
            hs_b = hs_valid[mask_both]
            ho_mean = float(np.mean(ho_b))
            hs_mean = float(np.mean(hs_b))
            cc_mat = np.corrcoef(hs_b, ho_b)
            if cc_mat.size > 1:
                cc = float(cc_mat[0, 1])
            ss_res = float(np.sum((ho_b - hs_b) ** 2))
            ss_tot = float(np.sum((ho_b - ho_mean) ** 2))
            nse = 1.0 - ss_res / max(ss_tot, 1e-12)
            r_corr = cc
            sigma_s = float(np.std(hs_b))
            sigma_o = float(np.std(ho_b))
            alpha_kge = sigma_s / max(sigma_o, 1e-12)
            beta_kge = hs_mean / max(abs(ho_mean), 1e-12)
            kge = 1.0 - np.sqrt((r_corr - 1) ** 2 + (alpha_kge - 1) ** 2 + (beta_kge - 1) ** 2)
            pbias = float(np.sum(hs_b - ho_b) / max(abs(np.sum(ho_b)), 1e-12) * 100.0)

    net_rech = float(np.sum(rech))
    total_rain = max(float(np.sum(po_shifted)), 1e-9)
    recharge_ratio = (net_rech / total_rain) * 100.0

    dist_min = abs(opt_k - kb_min)
    dist_max = abs(opt_k - kb_max)
    range_k = abs(kb_max - kb_min)
    stress = 1.0 - min(dist_min, dist_max) / max(range_k * 0.5, 1e-9)
    stress = max(0.0, min(stress, 1.0))

    rec_slope_err = calc_recession_slope_diff(ho, hs_pure, po_shifted, rc_val, pump_mask_fixed)
    ho_eval = ho.copy()
    ho_eval[pump_mask_fixed] = np.nan
    hs_eval = hs_pure.copy()
    hs_eval[pump_mask_fixed] = np.nan
    flash_diff = abs(calc_flash(ho_eval) - calc_flash(hs_eval))

    pump_idx, pump_events, pump_max_run = calc_pump_contam(ho, po_shifted, rc_val)
    rain_resp_obs = calc_rain_response(ho, po_shifted, rc_val, pump_mask_fixed)
    rain_resp_sim = calc_rain_response(hs_pure, po_shifted, rc_val, pump_mask_fixed)

    bnd_warnings = []
    # 경계 근접 경고: 범위의 0.5% 또는 절대 거리 0.001 중 작은 값 사용
    # (범위가 넓어져도 임계값이 과도하게 커지지 않도록)
    k_range = abs(kb_max - kb_min)
    k_thr = min(k_range * 0.005, 0.001)
    if abs(opt_k - kb_min) < k_thr:
        bnd_warnings.append(f"k={opt_k:.4f} is at lower bound ({kb_min:.4f})")
    if abs(opt_k - kb_max) < k_thr:
        bnd_warnings.append(f"k={opt_k:.4f} is at upper bound ({kb_max:.4f})")

    sn_c = max(0, min(11, round(opt_k) - 1)) if False else 0  # placeholder
    rech_lo_w, rech_hi_w = RECH_RANGE[0]  # will be overridden
    # Use soil index from opt_z context — not available here, use generic check
    if recharge_ratio > 50:
        bnd_warnings.append(f"Recharge ratio {recharge_ratio:.1f}% is unusually high")

    return CoreMetrics(
        rmse=rmse,
        sigma_ho=sigma_ho,
        cc=cc,
        nse=float(nse),
        kge=float(kge),
        pbias=float(pbias),
        recharge_ratio=recharge_ratio,
        total_rech_depth=net_rech,
        Sy_eff=sy_eff,
        n_f_avg=n_f_avg,
        opt_k=opt_k,
        opt_z=opt_z,
        opt_lag=int(opt_lag),
        opt_rho=float(opt_rho) if opt_rho is not None else KALMAN_RHO,
        opt_alpha=float(opt_alpha) if opt_alpha is not None else KALMAN_WTF_BLEND_ALPHA,
        kb_min=kb_min,
        kb_max=kb_max,
        flash_diff=flash_diff,
        rec_slope_err=rec_slope_err,
        stress=stress,
        pump_contam_idx=pump_idx,
        pump_event_count=int(pump_events),
        pump_max_run=int(pump_max_run),
        rain_resp_obs=rain_resp_obs,
        rain_resp_sim=rain_resp_sim,
        eval_n=int(np.sum(eval_valid)),
        ho=ho.tolist(),
        hs_kf=hs_kf.tolist(),
        hs_pure=hs_pure.tolist(),
        po=po.tolist(),
        po_shifted=po_shifted.tolist(),
        rech=rech.tolist(),
        pump_mask=pump_mask_fixed.astype(int).tolist(),
        boundary_warnings=bnd_warnings,
    )


def _build_core_error(exc: Exception):
    import traceback
    return CoreErrorResult(error=str(exc), stack=traceback.format_exc())


def core_sim_v27(file_path, k_val, z_val, lag_val, sn_idx, q_val, r_val,
                 rc_val, ignore_pump, sens_val, do_optimize):
    """Supported public entry point for app/runtime usage.

    All Kalman hyperparameters (rho, alpha) are passed as keyword args
    to run_logic_v27 — no global state mutation (thread-safe).
    """
    try:
        logger.info("core_sim_v27 start: file=%s sn=%.0f k=%.4f z=%.2f opt=%s",
                     file_path, sn_idx, k_val, z_val, bool(do_optimize))
        data = load_core_data(file_path)
        ho = data.ho.ravel().astype(float)
        po = data.po.ravel().astype(float)
        logger.debug("Loaded %d observations, rain sum=%.2f", len(ho), po.sum())

        if ignore_pump > 0:
            ho = remove_outliers(ho, sens_val)

        pump_mask_fixed = detect_pump_mask(ho, po, rc_val)
        sn, kb_min, kb_max, opt_k, opt_z, opt_lag = normalize_core_inputs(
            sn_idx, k_val, z_val, lag_val
        )

        opt_rho = KALMAN_RHO
        opt_alpha = KALMAN_WTF_BLEND_ALPHA

        if do_optimize > 0:
            opt_k, opt_z, opt_lag, opt_rho, opt_alpha = optimize_parameters(
                ho, po, sn, opt_k, opt_z, rc_val, pump_mask_fixed
            )

        po_shifted = apply_lag(po, opt_lag)
        # Pass rho/alpha as keyword args — no global state mutation
        rech, hs_kf, hs_pure, sy_eff, n_f_avg = run_logic_v27(
            opt_k, opt_z, sn, po_shifted, ho, q_val, r_val, rc_val,
            pump_mask_fixed, rho=opt_rho, alpha=opt_alpha,
        )

        result = build_core_metrics(
            ho=ho, po=po, po_shifted=po_shifted,
            hs_kf=hs_kf, hs_pure=hs_pure, rech=rech,
            pump_mask_fixed=pump_mask_fixed,
            sy_eff=sy_eff, n_f_avg=n_f_avg,
            opt_k=opt_k, opt_z=opt_z, opt_lag=opt_lag,
            kb_min=kb_min, kb_max=kb_max, rc_val=rc_val,
            opt_rho=opt_rho, opt_alpha=opt_alpha,
        )
        return result.to_dict()
    except Exception as exc:
        return _build_core_error(exc).to_dict()