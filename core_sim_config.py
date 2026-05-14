"""Configuration constants for the v27 hybrid-recharge core simulation."""

OUTLIER_MIN_FACTOR = 0.5
OUTLIER_BASE_FACTOR = 5.0
OUTLIER_SENSITIVITY_SCALE = 0.45

PUMP_SIGMA_FALLBACK = 0.005
PUMP_SPIKE_SIGMA_MULTIPLIER = -2.5
PUMP_SPIKE_MIN_DROP = -0.20
PUMP_RUN_SIGMA_MULTIPLIER = -0.7
PUMP_RUN_MIN_DROP = -0.05
PUMP_RUN_MIN_LENGTH = 4
PUMP_SPIKE_PRE_DAYS = 1
PUMP_SPIKE_POST_DAYS = 3
PUMP_RUN_PRE_DAYS = 1
PUMP_RUN_POST_DAYS = 2

DEFAULT_N_F_AVG = 0.05
MIN_SY_FOR_INPUT_SCALE = 0.05

RESP_PENALTY_WEIGHT = 0.10
RECHARGE_PENALTY_WEIGHT = 0.005

OPT_LAG_SEARCH_DAYS = 16
OPT_LAG_XCORR_CANDIDATES = 5  # cross-correlation 기반 lag 사전 필터링 후보 수
OPT_XATOL = 1e-5
OPT_FATOL = 1e-5
OPT_MAXFEV = 2000

# ── Kalman filter noise defaults ──────────────────────────────────
# Canonical Q/R defaults for the scalar Kalman filter in run_logic_v27.
# Q (process noise): controls how much the model trusts its own prediction.
#   Larger Q → more weight on observations, faster response.
# R (measurement noise): controls how much the model trusts observations.
#   Larger R → more smoothing, less sensitivity to noisy readings.
#
# These values (Q=0.005, R=0.10) are tuned for the scalar 1-state WTF
# filter operating on daily water-level data in metres.  The AugmentedKalmanWTF
# (2-state filter in pump_preprocess/) uses smaller defaults because its Q
# matrix is 2×2 and the recharge-forcing state requires tighter constraints.
#
# All modules should import from here rather than hardcoding defaults.
DEFAULT_Q_NOISE = 0.005   # process noise variance (m²/day)
DEFAULT_R_NOISE = 0.10    # measurement noise variance (m²)
KALMAN_Q_FLOOR = 1e-6     # absolute minimum Q (numerical safety)
KALMAN_R_FLOOR = 1e-4     # absolute minimum R (numerical safety)

# ── Augmented 2-state Kalman [h, w] parameters ──────────
# rho: persistence of the hidden recharge forcing state w(t).
#   w(t) = rho * w(t-1) + process noise
#   rho ∈ [0,1]; higher → smoother recharge time-series.
KALMAN_RHO = 0.85
# Ratio of process noise for w-state relative to h-state.
# Smaller → Kalman trusts WTF prior more; larger → more Kalman correction.
KALMAN_W_Q_RATIO = 0.3

# Cross-covariance between h and w process noise.
# Physically: recharge forcing (w) directly drives head changes (h),
# so their process noise is correlated.  A positive value means that
# unmodelled recharge increases both h and w simultaneously.
#
# q_hw = KALMAN_HW_Q_CORR * sqrt(q_h * q_w)
#
# Crosbie et al. (2005) and Gehman et al. (2009) use diagonal Q for
# simplicity, but allowing off-diagonal terms captures the physical
# coupling and generally improves filter convergence.
#
# Default 0.3 is conservative — typical range [0.1, 0.6].
# Set to 0.0 to recover the original diagonal-Q behaviour.
KALMAN_HW_Q_CORR = 0.3
# Weight for blending WTF event recharge with Kalman recharge estimate.
#   final_rech = alpha * rech_wtf + (1-alpha) * rech_kalman
KALMAN_WTF_BLEND_ALPHA = 0.4

# ── 초기 공분산 행렬 P₀ 설정 ─────────────────────────────
# 기존 하드코딩 P = diag([1.0, 0.1]) 대신 데이터 분산 기반으로 자동 설정.
#
# P₀[0,0] = max(σ²(ho), KALMAN_P0_H_FLOOR)
#   수위 관측값의 분산 → h 상태 초기 불확실성
# P₀[1,1] = KALMAN_P0_W_RATIO * P₀[0,0]
#   w 상태는 h보다 초기 불확실성이 낮음 (잠재 변수 특성)
#
# References:
#   Grewal, M.S. & Andrews, A.P. (2014). Kalman Filtering: Theory and
#       Practice Using MATLAB, 4th ed. Wiley, Sec. 4.4
#   Mehra, R.K. (1972). Approaches to adaptive filtering. IEEE TAC, 17(5).
KALMAN_P0_H_FLOOR = 0.01  # P₀[0,0] 하한값 (m²) — 극단적으로 분산이 작은 경우 대비
KALMAN_P0_W_RATIO = 0.1   # P₀[1,1] = P0_W_RATIO × P₀[0,0]

# ── 적응형 관측 노이즈 R(t) ───────────────────────────────
# 펌핑 오염 인접 구간에서 관측 신뢰도가 낮으므로 R을 증가시켜
# Kalman gain을 축소하고 필터가 관측보다 모델 예측을 더 신뢰하도록 함.
#
# KALMAN_R_PUMP_FACTOR: 펌핑 인접 구간에서 R을 몇 배 증가시킬지.
#   4.0 = 표준편차 2배 증가 (분산 4배). 관측 신뢰도가 절반으로 떨어진다고 가정.
# KALMAN_R_PUMP_PROXIMITY_DAYS: 펌핑 이벤트로부터 ±며칠 이내를 인접 구간으로 볼지.
#   5일 = pumping 이벤트 전후 5일간 R 증가 적용.
#
# References:
#   Mehra, R.K. (1972). Approaches to adaptive filtering.
#       IEEE Trans. Autom. Control, 17(5), 693-698.
#   Mohamed, A.H. & Schwarz, K.P. (1999). Adaptive Kalman filtering for
#       INS/GPS. Journal of Geodesy, 73(4), 193-203.
KALMAN_R_PUMP_FACTOR = 4.0         # R 증가 배율 최대값 (펌핑 이벤트 당일)
KALMAN_R_PUMP_PROXIMITY_DAYS = 5   # 펌핑 이벤트 전후 인접 구간 (일)
# σ for Gaussian decay: ~95% of the inflation is within ±PROXIMITY_DAYS.
# σ = PROXIMITY_DAYS / 2 ensures exp(-d²/(2σ²)) ≈ 0.02 at d = PROXIMITY_DAYS.
KALMAN_R_PUMP_SIGMA = 2.5          # Gaussian kernel σ (days)
INTER_EVENT_FRAC = 0.05            # 이벤트 간 최소 간격 비율

# ── Physics / numerical safety floors ─────────────────────
MIN_UNSAT_DEPTH = 0.01       # m — singularity avoidance for VG h_unsat
MIN_SY_FLOOR = 0.001         # minimum Sy clamp (prevents division by zero)
DEFAULT_SIGMA_HO = 0.1       # fallback observed WL std dev (m)

# ── Equilibrium head estimation ───────────────────────────
DAYS_PER_YEAR = 365
MIN_SEGMENT_DAYS = 180       # half-year minimum for annual minimum extraction
MIN_ANNUAL_SEGMENT_OBS = 30  # minimum observations per segment for annual min

# ── Event recharge accumulation ───────────────────────────
ANTECEDENT_DRAIN_DAYS = 121  # look-back window for pre-event recession

# ── Pumping contamination sub-weights (calc_pump_contam) ──
PUMP_FRAC_WEIGHT = 0.55
PUMP_FRAC_NORMALIZER = 0.35
PUMP_EVENTS_WEIGHT = 0.25
PUMP_EVENTS_NORMALIZER = 8.0
PUMP_MAXRUN_WEIGHT = 0.20
PUMP_MAXRUN_NORMALIZER = 12.0

# ── calc_error objective function weights ─────────────────
OBJ_W_FIT = 0.70             # NRMSE weight
OBJ_W_RESP = 0.15            # rain-response mismatch weight
OBJ_W_RECH = 0.15            # recharge-range violation weight
OBJ_RECH_VIOLATION_CAP = 2.0 # soft cap on recharge violation term
MIN_VALID_POINTS = 15        # absolute minimum for evaluation
MIN_VALID_FRACTION = 0.15    # minimum valid fraction of record

# ── Parameter bounds ──────────────────────────────────────
MAX_Z_PARAM = 30.0           # upper bound for z_unsat (m)
MIN_Z_PARAM = 0.1            # lower bound for z_unsat (m)