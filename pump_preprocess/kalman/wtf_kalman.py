"""
wtf_kalman.py — 확장 상태 Kalman 필터 기반 WTF 함양 추정 (v30)
=================================================================
core_sim_v27.py를 바탕으로 펌핑 전처리 통합 버전으로 리팩토링.

주요 개선 사항 (vs v27):
- 전처리된 수위를 직접 입력받아 Kalman 필터 실행
- 펌핑 마스크 구간은 Kalman 갱신 단계에서 자동 제외 (관측 없는 날 처리)
- EventLogic 강화: 이벤트 간 잠열 함양 포착
- 3-상태 선택적 확장 [h, w, q_pump] (pump_records 제공 시)
"""

import os as _os
import sys as _sys
import warnings

import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Optional, Tuple

# Unit conversion constant — avoids magic-number /1000 scattered through code
MM_PER_M = 1000.0

# ══════════════════════════════════════════════════════════
# 토양 데이터베이스: 프로젝트 루트의 soil_db.py를 단일 출처로 사용.
# 중복 정의를 제거하여 동기화 문제 방지.
# ══════════════════════════════════════════════════════════
try:
    from soil_db import (
        VG_DB, TAU_DB, ALPHA_DB, SY_DB, K_BOUNDS, SOIL_NAMES, RECH_RANGE,
    )
except ImportError:
    # pump_preprocess를 독립 실행 시 루트 경로를 sys.path에 추가
    _root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..'))
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
    from soil_db import (
        VG_DB, TAU_DB, ALPHA_DB, SY_DB, K_BOUNDS, SOIL_NAMES, RECH_RANGE,
    )


@dataclass
class KalmanWTFResult:
    # 수위 재현
    h_sim: np.ndarray          # Kalman 추정 수위
    w_est: np.ndarray          # 추정된 숨겨진 함양 강제
    # 함양
    rech_event: np.ndarray     # 이벤트 함양 (mm/day)
    rech_inter: np.ndarray     # 이벤트 간 함양 (mm/day)
    rech_kalman: np.ndarray    # Kalman 포착 함양 (mm/day)
    rech_total: np.ndarray     # 총 함양 (mm/day)
    # 성능 지표
    rmse: float
    nse: float
    cc: float
    rech_rate_pct: float       # 연간 함양율 (강우 대비 %)
    rech_bias_pct: float       # 함양 편향 (합성 검증 시)
    # 토양
    best_soil_idx: int         # 1-based
    best_soil_name: str
    soil_scores: np.ndarray    # 12개 토양 복합 점수
    # 최적화
    k_val: float
    z_val: float
    q_noise: float
    r_noise: float
    # 전처리 정보
    pump_excluded_days: int
    correction_applied: bool


# ══════════════════════════════════════════════════════════
# 메인 클래스
# ══════════════════════════════════════════════════════════
class AugmentedKalmanWTF:
    """
    Parameters
    ----------
    soil_num : int or None
        1~12 고정 토양 번호. None이면 자동 추정.
    k_val : float or None
        기저유출 감쇠계수 k. None이면 자동 최적화.
    z_val : float
        불포화대 두께 초기값 (m)
    lag_days : int
        강우-수위 응답 지연 (일)
    q_noise : float
        Kalman 과정 노이즈 Q (자동 최적화 가능)
    r_noise : float
        Kalman 측정 노이즈 R
    rho : float
        함양 강제 지속성 계수 (0~1)
    inter_frac : float
        이벤트 간 함양 비율 계수
    alpha_cap : float
        함양 상한 계수
    rain_cutoff : float
        강우 이벤트 최소 임계값 (mm/day)
    gap_allow : int
        이벤트 간 최소 간격 (일)
    peak_window : int
        피크 탐지 윈도우 (일)
    auto_optimize : bool
        L-BFGS-B 자동 최적화 여부
    exclude_pump_from_kalman : bool
        펌핑 구간에서 Kalman 갱신 단계 제외 여부 (핵심 개선)
    """

    def __init__(
        self,
        soil_num: Optional[int] = None,
        k_val: Optional[float] = None,
        z_val: float = 3.0,
        lag_days: int = 0,
        q_noise: float = 1e-4,
        r_noise: float = 1e-3,
        rho: float = 0.85,
        inter_frac: float = 0.35,
        alpha_cap: float = 0.95,
        rain_cutoff: float = 1.0,
        gap_allow: int = 2,
        peak_window: int = 7,
        auto_optimize: bool = True,
        exclude_pump_from_kalman: bool = True,
    ):
        self.soil_num = soil_num
        self.k_val = k_val
        self.z_val = z_val
        self.lag_days = lag_days
        self.q_noise = q_noise
        self.r_noise = r_noise
        self.rho = rho
        self.inter_frac = inter_frac
        self.alpha_cap = alpha_cap
        self.rain_cutoff = rain_cutoff
        self.gap_allow = gap_allow
        self.peak_window = peak_window
        self.auto_optimize = auto_optimize
        self.exclude_pump_from_kalman = exclude_pump_from_kalman

    # ─────────────────────────────────────────
    def run(
        self,
        water_level: np.ndarray,
        rainfall: np.ndarray,
        pump_mask: Optional[np.ndarray] = None,
        true_recharge: Optional[np.ndarray] = None,
    ) -> KalmanWTFResult:
        """
        Parameters
        ----------
        water_level  : 관측 수위 (보정된 수위 권장) (m)
        rainfall     : 강우량 (mm/day)
        pump_mask    : 펌핑 구간 마스크 (True = 펌핑)
        true_recharge: 합성 검증용 실제 함양 (선택)
        """
        n = len(water_level)
        wl = np.array(water_level, dtype=float)
        po = np.array(rainfall, dtype=float)
        pm = np.zeros(n, dtype=bool) if pump_mask is None else np.array(pump_mask, dtype=bool)

        # 지연 적용
        po_lag = self._apply_lag(po, self.lag_days)

        # 토양 탐색 범위
        soil_range = [self.soil_num] if self.soil_num else list(range(1, 13))

        best_result = None
        best_score = np.inf

        for sn in soil_range:
            k = self.k_val
            z = self.z_val
            q = self.q_noise
            r = self.r_noise

            # k_val 미지정 시 토양별 기본값 사용
            if k is None:
                kb = K_BOUNDS[sn]
                k = (kb[0] + kb[1]) / 2.0

            # 자동 최적화
            if self.auto_optimize:
                k, z, q, r = self._optimize(wl, po_lag, pm, sn)

            result = self._run_single(wl, po_lag, pm, sn, k, z, q, r)
            if result.rmse < best_score:
                best_score = result.rmse
                best_result = result

        # 토양 복합 점수 계산 (전체 12개)
        if self.soil_num is None:
            scores, best_sn = self._soil_composite_score(wl, po_lag, pm)
        else:
            scores = np.zeros(12)
            scores[self.soil_num - 1] = 1.0
            best_sn = self.soil_num

        # 함양 편향 계산 (합성 검증 시)
        rech_bias = 0.0
        if true_recharge is not None:
            true_total = np.nansum(true_recharge)
            sim_total = np.nansum(best_result.rech_total)
            if true_total > 0:
                rech_bias = (sim_total - true_total) / true_total * 100

        # 결과 보완
        return KalmanWTFResult(
            h_sim=best_result.h_sim,
            w_est=best_result.w_est,
            rech_event=best_result.rech_event,
            rech_inter=best_result.rech_inter,
            rech_kalman=best_result.rech_kalman,
            rech_total=best_result.rech_total,
            rmse=best_result.rmse,
            nse=best_result.nse,
            cc=best_result.cc,
            rech_rate_pct=best_result.rech_rate_pct,
            rech_bias_pct=rech_bias,
            best_soil_idx=best_sn,
            best_soil_name=SOIL_NAMES[best_sn - 1],
            soil_scores=scores,
            k_val=best_result.k_val,
            z_val=best_result.z_val,
            q_noise=best_result.q_noise,
            r_noise=best_result.r_noise,
            pump_excluded_days=int(pm.sum()),
            correction_applied=pm.any(),
        )

    # ─────────────────────────────────────────
    # 단일 토양 실행
    # ─────────────────────────────────────────
    def _run_single(
        self,
        wl: np.ndarray,
        po: np.ndarray,
        pm: np.ndarray,
        sn: int,
        k: float,
        z: float,
        q: float,
        r: float,
    ):
        n = len(wl)
        sn_idx = sn - 1
        alpha_s = ALPHA_DB[sn_idx]
        sy_eff  = SY_DB[sn_idx]

        # ── 이벤트 함양 계산 ──
        rech_event, event_mask = self._calc_event_recharge(wl, po, pm, sn, alpha_s)

        # ── 이벤트 간 함양 (EventLogic) ──
        rech_inter = self._calc_inter_recharge(po, event_mask, alpha_s)

        # ── 2-상태 Kalman 필터 [h, w] ──
        h_sim, w_est = self._run_kalman(wl, po, pm, k, z, q, r, alpha_s, sy_eff)

        # ── Kalman 함양 추출 ──
        # w_est는 meters 스케일 → mm 변환 (×1000) + daily cap 적용
        rech_kalman_raw = np.maximum(w_est * sy_eff * 1000.0, 0.0)  # m → mm
        # 강우일에만 Kalman 함양 허용, 비강우일은 0 처리
        rain_day = po >= self.rain_cutoff
        daily_cap_mm = np.where(rain_day, alpha_s * po * 0.3, 0.0)
        rech_kalman = np.minimum(rech_kalman_raw, daily_cap_mm)

        # ── 총 함양 ──
        rech_total = rech_event + rech_inter + rech_kalman

        # ── 성능 지표 ──
        valid = ~np.isnan(wl) & ~np.isnan(h_sim)
        if self.exclude_pump_from_kalman:
            valid &= ~pm

        rmse = float(np.sqrt(np.mean((wl[valid] - h_sim[valid]) ** 2))) if valid.sum() > 0 else 9.99
        nse  = self._nse(wl[valid], h_sim[valid]) if valid.sum() > 5 else -999.0
        cc   = float(np.corrcoef(wl[valid], h_sim[valid])[0, 1]) if valid.sum() > 5 else 0.0

        total_rain = np.nansum(po)
        rech_rate = float(np.nansum(rech_total) / total_rain * 100) if total_rain > 0 else 0.0

        # 임시 결과 객체 (내부 전달용)
        class _R:
            pass
        res = _R()
        res.h_sim = h_sim
        res.w_est = w_est
        res.rech_event = rech_event
        res.rech_inter = rech_inter
        res.rech_kalman = rech_kalman
        res.rech_total = rech_total
        res.rmse = rmse
        res.nse = nse
        res.cc = cc
        res.rech_rate_pct = rech_rate
        res.rech_bias_pct = 0.0
        res.best_soil_idx = sn
        res.best_soil_name = SOIL_NAMES[sn_idx]
        res.soil_scores = np.zeros(12)
        res.k_val = k
        res.z_val = z
        res.q_noise = q
        res.r_noise = r
        res.pump_excluded_days = int(pm.sum())
        res.correction_applied = pm.any()
        return res

    # ─────────────────────────────────────────
    # 2-상태 Kalman 필터 [h, w]
    # ─────────────────────────────────────────
    def _run_kalman(
        self,
        wl: np.ndarray,
        po: np.ndarray,
        pm: np.ndarray,
        k: float,
        z: float,
        q: float,
        r: float,
        alpha_s: float,
        sy_eff: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(wl)

        # 상태 전이 행렬 F = [[1+k, 1], [0, rho]]
        rho = self.rho
        F = np.array([[1.0 + k, 1.0], [0.0, rho]])

        # 관측 행렬 H = [1, 0]
        H = np.array([[1.0, 0.0]])

        # 노이즈 공분산
        Q = np.diag([q, q * 0.1])
        R_mat = np.array([[r]])

        # 초기 상태
        h0 = np.nanmean(wl[:min(10, n)])
        x = np.array([h0, 0.0])
        P = np.eye(2) * 1.0

        h_eq = np.nanpercentile(wl, 10)

        h_sim = np.full(n, np.nan)
        w_est = np.zeros(n)

        for t in range(n):
            # ── 예측 단계 ──
            u = alpha_s * po[t] / MM_PER_M  # mm → m
            x_pred = F @ x + np.array([u, 0.0])
            # 평형 수위 인력
            x_pred[0] += k * (h_eq - x_pred[0]) * 0.1
            P_pred = F @ P @ F.T + Q

            # ── 갱신 단계 (펌핑 구간 제외) ──
            obs_available = (
                not np.isnan(wl[t]) and
                (not self.exclude_pump_from_kalman or not pm[t])
            )

            if obs_available:
                S = H @ P_pred @ H.T + R_mat
                K_gain = P_pred @ H.T @ np.linalg.inv(S)
                innov = wl[t] - (H @ x_pred)[0]
                x = x_pred + (K_gain @ [[innov]]).flatten()
                P = (np.eye(2) - K_gain @ H) @ P_pred
            else:
                # 관측 없는 날: 예측만 사용 (펌핑 흡수 방지)
                x = x_pred
                P = P_pred

            h_sim[t] = x[0]
            w_est[t] = x[1]

        return h_sim, w_est

    # ─────────────────────────────────────────
    # 이벤트 기반 함양
    # ─────────────────────────────────────────
    def _calc_event_recharge(
        self,
        wl: np.ndarray,
        po: np.ndarray,
        pm: np.ndarray,
        sn: int,
        alpha_s: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(wl)
        rech = np.zeros(n)
        event_mask = np.zeros(n, dtype=bool)
        sn_idx = sn - 1
        sy_base = SY_DB[sn_idx]

        i = 0
        while i < n:
            # 강우 이벤트 시작
            if po[i] >= self.rain_cutoff and not pm[i]:
                # 이벤트 구간 수집
                j = i
                while j < n and (po[j] >= self.rain_cutoff or (
                    j - i < self.gap_allow and j < n
                )):
                    j += 1
                event_rain = np.sum(po[i:j])

                # 피크 수위 탐지
                search_end = min(n, j + self.peak_window)
                valid_wl = wl[i:search_end]
                if np.any(~np.isnan(valid_wl)):
                    peak_idx = i + np.nanargmax(valid_wl)
                    pre_idx = max(0, i - 3)
                    h_pre = np.nanmean(wl[pre_idx:i]) if i > 0 else wl[0]
                    delta_h = max(0.0, wl[peak_idx] - h_pre) if not np.isnan(wl[peak_idx]) else 0.0

                    # VG 기반 Sy 계산
                    time_dry = max(1, i - np.where(po[:i] >= self.rain_cutoff)[0][-1]
                                   if np.any(po[:i] >= self.rain_cutoff) else 30)
                    sy_ev = self._filpor(sn_idx, self.z_val, time_dry)

                    r_ev = min(sy_ev * delta_h, alpha_s * event_rain / MM_PER_M)
                    r_ev = min(r_ev, self.alpha_cap * event_rain / MM_PER_M)

                    if r_ev > 0:
                        rech[peak_idx] = r_ev * MM_PER_M  # m → mm
                        event_mask[i:j] = True

                i = j
            else:
                i += 1

        return rech, event_mask

    # ─────────────────────────────────────────
    # 이벤트 간 함양 (EventLogic)
    # ─────────────────────────────────────────
    def _calc_inter_recharge(
        self,
        po: np.ndarray,
        event_mask: np.ndarray,
        alpha_s: float,
    ) -> np.ndarray:
        rech = np.zeros(len(po))
        for t in range(len(po)):
            if not event_mask[t] and po[t] >= self.rain_cutoff:
                rech[t] = alpha_s * self.inter_frac * po[t]
        return rech

    # ─────────────────────────────────────────
    # VG 기반 Sy 계산 (filpor)
    # ─────────────────────────────────────────
    def _filpor(self, sn_idx: int, z: float, time_dry: float) -> float:
        par = VG_DB[sn_idx]
        tau = TAU_DB[sn_idx]
        Th_s, Th_r, alpha, n_vg = par
        m_vg = 1.0 - 1.0 / n_vg
        h_unsat = max(z, 0.01)
        Se = 1.0 / (1.0 + (alpha * h_unsat) ** n_vg) ** m_vg
        recovery = 1.0 - np.exp(-time_dry / tau)
        Sy_raw = (Th_s - Th_r) * (1.0 - Se) * recovery
        Sy_max = Th_s - Th_r
        return max(min(Sy_raw, Sy_max), 0.001)

    # ─────────────────────────────────────────
    # 토양 복합 점수
    # ─────────────────────────────────────────
    def _soil_composite_score(
        self,
        wl: np.ndarray,
        po: np.ndarray,
        pm: np.ndarray,
    ) -> Tuple[np.ndarray, int]:
        scores = np.zeros(12)

        for sn in range(1, 13):
            k_lo, k_hi = K_BOUNDS[sn]
            if self.auto_optimize:
                k, z, q, r = self._optimize(wl, po, pm, sn)
            else:
                k = (k_lo + k_hi) / 2
                z = self.z_val
                q = self.q_noise
                r = self.r_noise

            res = self._run_single(wl, po, pm, sn, k, z, q, r)

            # 6-지표 복합 점수
            # 1) RMSE 점수
            rmse_score = max(0, 100 - res.rmse * 500)
            # 2) NSE 점수
            nse_score = max(0, (res.nse + 1) / 2 * 100) if res.nse > -1 else 0
            # 3) k 범위 적합도
            k_lo_s, k_hi_s = K_BOUNDS[sn]
            k_score = 100 if k_lo_s <= res.k_val <= k_hi_s else max(0, 100 - abs(res.k_val - (k_lo_s + k_hi_s) / 2) * 500)
            # 4) 함양율 물리 타당성
            rr = res.rech_rate_pct
            rech_lo, rech_hi = RECH_RANGE[sn - 1] if hasattr(self, '_rech_range') else (1, 40)
            rr_score = 100 if rech_lo <= rr <= rech_hi else max(0, 100 - abs(rr - (rech_lo + rech_hi) / 2) * 3)
            # 5) CC 점수
            cc_score = max(0, res.cc * 100)
            # 6) Sy 합리성
            sy_score = 80.0  # 기본값 (현장 측정 없으면 중립)

            # 가중 합산
            composite = (
                0.35 * rmse_score +
                0.20 * nse_score +
                0.15 * k_score +
                0.12 * rr_score +
                0.10 * cc_score +
                0.08 * sy_score
            )
            scores[sn - 1] = composite

        best_sn = int(np.argmax(scores)) + 1
        return scores, best_sn

    # ─────────────────────────────────────────
    # L-BFGS-B 최적화
    # ─────────────────────────────────────────
    def _optimize(
        self,
        wl: np.ndarray,
        po: np.ndarray,
        pm: np.ndarray,
        sn: int,
    ) -> Tuple[float, float, float, float]:
        k_lo, k_hi = K_BOUNDS[sn]

        def objective(params):
            k, z, q, r = params
            res = self._run_single(wl, po, pm, sn, k, z, q*1e-4, r*1e-3)
            penalty_rain = max(0, res.rech_rate_pct - 60) * 0.1
            penalty_neg  = max(0, -res.nse) * 0.5
            return res.rmse + penalty_rain + penalty_neg

        x0 = [(k_lo + k_hi) / 2, self.z_val, 1.0, 1.0]
        bounds = [(k_lo, k_hi), (0.5, 10.0), (0.01, 100.0), (0.01, 100.0)]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = minimize(
                    objective, x0,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 200, "ftol": 1e-6},
                )
            k, z, q_s, r_s = res.x
            return float(k), float(z), float(q_s * 1e-4), float(r_s * 1e-3)
        except Exception:
            return float(x0[0]), float(x0[1]), self.q_noise, self.r_noise

    # ─────────────────────────────────────────
    # 헬퍼
    # ─────────────────────────────────────────
    def _apply_lag(self, po: np.ndarray, lag: int) -> np.ndarray:
        lag = max(0, int(lag))
        if lag == 0:
            return po.copy()
        return np.concatenate([np.zeros(lag), po[: len(po) - lag]])

    def _nse(self, obs: np.ndarray, sim: np.ndarray) -> float:
        obs_mean = np.mean(obs)
        ss_res = np.sum((obs - sim) ** 2)
        ss_tot = np.sum((obs - obs_mean) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else -999.0


# RECH_RANGE는 soil_db에서 import됨 — 중복 정의 제거
# soil_identifier.py가 이 모듈에서 RECH_RANGE를 import하므로 re-export 유지
AugmentedKalmanWTF._rech_range = RECH_RANGE
