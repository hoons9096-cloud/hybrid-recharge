"""
wtf_lumped.py -- Lumped WTF method (baseline).

Uses a single Sy value (mean of all well-location Sy values) for the
entire domain. Computes WTF recharge at each well, averages them, and
applies the single mean value uniformly to every grid cell.

This is the simplest baseline that ignores all spatial variability.
"""
from __future__ import annotations

import sys
import os

import numpy as np

# soil_db 임포트 경로 설정
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from soil_db import SOIL_DB


# ──────────────────────────────────────────────────────────
# WTF 단일 관정 함양 계산
# ──────────────────────────────────────────────────────────
def _estimate_recession_k(ho: np.ndarray, P: np.ndarray,
                          min_dry_days: int = 5) -> float:
    """Estimate recession constant k from dry periods.

    건기(P < 0.001 m/day 연속 구간)에서 수위 감소 패턴을 분석하여
    지수 감쇠 상수 k를 추정한다.

    모델: h(t) = h(t-1) * (1 + k) - k * h_eq
    건기에서 dh/dt ~ k * (h - h_eq) 이므로
    k = median(dh / (h - h_eq)) (건기 일자에 대해)

    Parameters
    ----------
    ho : (n_days,) water level [m]
    P  : (n_days,) precipitation [m/day]
    min_dry_days : int
        최소 건기 연속 일수

    Returns
    -------
    k : float
        Recession constant (음수, 일반적으로 -0.3 ~ -0.001)
    """
    n = len(ho)
    # 건기 식별: 연속 무강수일 구간
    is_dry = P < 0.001
    dry_dh = []

    i = 0
    while i < n:
        # 건기 시작 찾기
        if not is_dry[i]:
            i += 1
            continue
        j = i
        while j < n and is_dry[j]:
            j += 1
        # i~j-1 구간이 건기
        if j - i >= min_dry_days:
            seg = ho[i:j]
            for t in range(1, len(seg)):
                if seg[t-1] > 0:
                    dry_dh.append((seg[t] - seg[t-1]) / seg[t-1])
        i = j

    if len(dry_dh) < 3:
        # 건기 데이터 부족 시 기본값 사용
        return -0.01

    k_est = float(np.median(dry_dh))
    # k는 반드시 음수 (감쇠)
    k_est = min(k_est, -0.0005)
    k_est = max(k_est, -0.3)
    return k_est


def _wtf_single_well(ho: np.ndarray, P: np.ndarray,
                     Sy: float, tau: int) -> float:
    """Compute annual recharge at one well using WTF method.

    WTF (Water Table Fluctuation) 방법으로 단일 관정의 연간 함양량을 계산.
    강우 이벤트 후 수위 상승(dh)에 Sy를 곱하여 이벤트별 함양량을 구하고
    연간 합계를 반환한다.

    Parameters
    ----------
    ho : (n_days,) water level [m]
    P  : (n_days,) precipitation [m/day]
    Sy : specific yield [-]
    tau : drainage time constant [days] -- 피크 탐색 윈도우로 사용

    Returns
    -------
    annual_recharge : float
        Annual recharge [mm/yr]
    """
    n = len(ho)
    if n < 10:
        return 0.0

    # 1. recession constant 추정
    k = _estimate_recession_k(ho, P)

    # 2. 평형 수위 추정 (장기 건기 최저 수위)
    h_eq = float(np.percentile(ho, 10))

    # 3. 강우 이벤트 탐지 및 함양 계산
    rain_threshold = 0.001  # m/day
    event_recharges = []  # 각 이벤트 함양량 [m]

    i = 0
    while i < n:
        if P[i] <= rain_threshold:
            i += 1
            continue

        # 이벤트 시작: 강우일
        event_start = i

        # 이벤트 직전 수위 (recession 시작점)
        h_pre = ho[max(0, event_start - 1)]

        # 피크 탐색 윈도우: tau 일 이내
        search_end = min(n, event_start + tau)
        if search_end <= event_start:
            i += 1
            continue

        # 피크 수위 찾기
        window = ho[event_start:search_end]
        peak_idx_local = int(np.argmax(window))
        h_peak = window[peak_idx_local]
        peak_day = event_start + peak_idx_local

        # recession 투영: h_pre에서 k 적용하여 피크 시점의 예상 수위
        n_days_elapsed = peak_day - event_start + 1
        h_projected = h_pre
        for _ in range(n_days_elapsed):
            h_projected = h_projected * (1 + k) - k * h_eq

        # 수위 상승량
        dh = h_peak - h_projected
        if dh > 0:
            rech_event = Sy * dh  # [m]
            event_recharges.append(rech_event)

        # 다음 이벤트: 피크 이후부터 탐색
        i = peak_day + 1

    # 4. 연간 함양량으로 변환 [mm/yr]
    total_rech_m = sum(event_recharges)
    n_years = max(n / 365.25, 1.0)
    annual_rech_mm = total_rech_m / n_years * 1000.0

    return annual_rech_mm


# ──────────────────────────────────────────────────────────
# 메인 인터페이스
# ──────────────────────────────────────────────────────────
def estimate_recharge(domain, observations: dict) -> np.ndarray:
    """Estimate recharge using the Lumped WTF method.

    Uses a SINGLE Sy value (mean of all well-location Sy) for the entire
    domain. Computes WTF recharge at each well, then assigns the mean
    recharge uniformly to every grid cell.

    Parameters
    ----------
    domain : SyntheticDomain
        From synthetic/generate_domain.py. Has soil_map (ny,nx), Sy_map,
        well_rows, well_cols, etc.
    observations : dict
        'P': (n_days,) precipitation [m/day]
        'ET': (n_days,) evapotranspiration [m/day]
        'ho_obs': (n_wells, n_days) observed water levels [m]
        'well_soil_types': (n_wells,) soil type index at each well

    Returns
    -------
    recharge_map : np.ndarray, shape (ny, nx)
        Estimated annual recharge [mm/yr] for each grid cell
    """
    P = observations['P']
    ho_obs = observations['ho_obs']
    well_soil_types = observations['well_soil_types']
    n_wells = ho_obs.shape[0]

    # 단일 Sy: 모든 관정 위치 Sy의 평균
    well_Sy_values = np.array([
        SOIL_DB[int(st)].sy_lit for st in well_soil_types
    ])
    Sy_lumped = float(np.mean(well_Sy_values))

    # tau도 평균 사용
    well_tau_values = np.array([
        SOIL_DB[int(st)].tau for st in well_soil_types
    ])
    tau_lumped = int(np.mean(well_tau_values))

    # 각 관정에서 WTF 함양 계산 (동일한 Sy_lumped 사용)
    well_recharges = []
    for w in range(n_wells):
        ho_w = ho_obs[w]
        rech_w = _wtf_single_well(ho_w, P, Sy_lumped, tau_lumped)
        well_recharges.append(rech_w)

    # 전체 도메인: 관정 평균값을 균일하게 적용
    mean_recharge = float(np.mean(well_recharges))
    ny, nx = domain.soil_map.shape
    recharge_map = np.full((ny, nx), mean_recharge)

    return recharge_map


# ──────────────────────────────────────────────────────────
# 테스트
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # S3 시나리오로 테스트
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from synthetic.generate_domain import generate_domain, DomainConfig

    print("=== Lumped WTF Method Test (S3) ===\n")

    # 도메인 생성
    domain = generate_domain(DomainConfig.S3())
    print(domain.summary())
    print()

    # 합성 관측 데이터 생성 (간단한 테스트용)
    rng = np.random.default_rng(42)
    n_days = 730  # 2년
    n_wells = domain.n_wells

    # 강수: 간헐적 강우 패턴
    P = np.where(rng.random(n_days) < 0.15,
                 rng.exponential(0.005, n_days), 0.0)

    # ET: 계절 변동
    day_of_year = np.arange(n_days) % 365
    ET = 0.003 * (1 + 0.5 * np.sin(2 * np.pi * day_of_year / 365))

    # 관측 수위: 각 관정별로 간단한 합성 수위 생성
    ho_obs = np.zeros((n_wells, n_days))
    well_soil_types = np.array([
        int(domain.soil_map[domain.well_rows[w], domain.well_cols[w]])
        for w in range(n_wells)
    ])

    for w in range(n_wells):
        soil_idx = well_soil_types[w]
        Sy_true = SOIL_DB[soil_idx].sy_lit
        # 기본 수위 + 강수 반응 + 노이즈
        h_base = 5.0
        h = np.full(n_days, h_base)
        for t in range(1, n_days):
            # recession
            h[t] = h[t-1] - 0.005 * (h[t-1] - 3.0)
            # 강수 반응: dh = P * alpha / Sy
            if P[t] > 0.001:
                h[t] += P[t] * 0.2 / Sy_true
        # 관측 노이즈
        h += rng.normal(0, 0.01, n_days)
        ho_obs[w] = h

    observations = {
        'P': P,
        'ET': ET,
        'ho_obs': ho_obs,
        'well_soil_types': well_soil_types,
    }

    # 함양 추정
    rech_map = estimate_recharge(domain, observations)

    print(f"Recharge map shape: {rech_map.shape}")
    print(f"Uniform recharge: {rech_map[0, 0]:.1f} mm/yr")
    print(f"  (모든 셀 동일: min={rech_map.min():.1f}, "
          f"max={rech_map.max():.1f} mm/yr)")
    print(f"Total annual P: {P.sum() / (n_days/365.25) * 1000:.0f} mm/yr")
    print(f"Recharge ratio: "
          f"{rech_map[0,0] / (P.sum()/(n_days/365.25)*1000) * 100:.1f}%")
