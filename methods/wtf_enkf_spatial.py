"""
wtf_enkf_spatial.py -- EnKF R-state method (advanced method).

Simplified EnKF for the synthetic benchmark. State vector = R (annual
recharge rate) for each grid cell. Uses well WTF estimates as observations
with Gaspari-Cohn localization.

This is a streamlined version for the method comparison paper, not a
production EnKF implementation.
"""
from __future__ import annotations

import sys
import os

import numpy as np

# soil_db 임포트 경로 설정
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from soil_db import SOIL_DB


# ──────────────────────────────────────────────────────────
# Gaspari-Cohn 국소화 함수
# ──────────────────────────────────────────────────────────
def _gaspari_cohn(d: float, r: float) -> float:
    """Gaspari-Cohn 5th-order polynomial localization.

    Gaspari & Cohn (1999) 국소화 함수.
    d=0일 때 1, d>=r일 때 0을 반환.

    Parameters
    ----------
    d : float
        두 점 사이 거리
    r : float
        국소화 반경 (cutoff distance)

    Returns
    -------
    weight : float
        국소화 가중치 [0, 1]
    """
    if r <= 0 or d >= r:
        return 0.0
    c = r / 2.0
    z = d / c
    if z < 1.0:
        return (1.0 - (5.0/4)*z**2 + (5.0/3)*z**3
                + (5.0/8)*z**4 - (1.0/2)*z**5)
    return (4.0 - 5*z + (5.0/3)*z**2 + (5.0/8)*z**3
            - (1.0/2)*z**4 + (1.0/12)*z**5 - 2.0/(3*z))


# ──────────────────────────────────────────────────────────
# WTF 단일 관정 함양 계산
# ──────────────────────────────────────────────────────────
def _estimate_recession_k(ho: np.ndarray, P: np.ndarray,
                          min_dry_days: int = 5) -> float:
    """Estimate recession constant k from dry periods.

    건기(P < 0.001 m/day 연속 구간)에서 수위 감소 패턴을 분석하여
    지수 감쇠 상수 k를 추정한다.

    Parameters
    ----------
    ho : (n_days,) water level [m]
    P  : (n_days,) precipitation [m/day]
    min_dry_days : int
        최소 건기 연속 일수

    Returns
    -------
    k : float
        Recession constant (음수, -0.3 ~ -0.001)
    """
    n = len(ho)
    is_dry = P < 0.001
    dry_dh = []

    i = 0
    while i < n:
        if not is_dry[i]:
            i += 1
            continue
        j = i
        while j < n and is_dry[j]:
            j += 1
        if j - i >= min_dry_days:
            seg = ho[i:j]
            for t in range(1, len(seg)):
                if seg[t-1] > 0:
                    dry_dh.append((seg[t] - seg[t-1]) / seg[t-1])
        i = j

    if len(dry_dh) < 3:
        return -0.01

    k_est = float(np.median(dry_dh))
    k_est = min(k_est, -0.0005)
    k_est = max(k_est, -0.3)
    return k_est


def _wtf_single_well(ho: np.ndarray, P: np.ndarray,
                     Sy: float, tau: int) -> float:
    """Compute annual recharge at one well using WTF method.

    WTF (Water Table Fluctuation) 방법으로 단일 관정의 연간 함양량을 계산.

    Parameters
    ----------
    ho : (n_days,) water level [m]
    P  : (n_days,) precipitation [m/day]
    Sy : specific yield [-]
    tau : drainage time constant [days]

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

    # 2. 평형 수위 추정
    h_eq = float(np.percentile(ho, 10))

    # 3. 강우 이벤트 탐지 및 함양 계산
    rain_threshold = 0.001
    event_recharges = []

    i = 0
    while i < n:
        if P[i] <= rain_threshold:
            i += 1
            continue

        event_start = i
        h_pre = ho[max(0, event_start - 1)]

        # 피크 탐색 윈도우
        search_end = min(n, event_start + tau)
        if search_end <= event_start:
            i += 1
            continue

        window = ho[event_start:search_end]
        peak_idx_local = int(np.argmax(window))
        h_peak = window[peak_idx_local]
        peak_day = event_start + peak_idx_local

        # recession 투영
        n_days_elapsed = peak_day - event_start + 1
        h_projected = h_pre
        for _ in range(n_days_elapsed):
            h_projected = h_projected * (1 + k) - k * h_eq

        dh = h_peak - h_projected
        if dh > 0:
            event_recharges.append(Sy * dh)

        i = peak_day + 1

    # 4. 연간 함양량 [mm/yr]
    total_rech_m = sum(event_recharges)
    n_years = max(n / 365.25, 1.0)
    return total_rech_m / n_years * 1000.0


# ──────────────────────────────────────────────────────────
# 앙상블 사전분포 생성
# ──────────────────────────────────────────────────────────
def _generate_prior_ensemble(
    ho: np.ndarray, P: np.ndarray,
    Sy_base: float, tau: int,
    n_ensemble: int, rng: np.random.Generator,
) -> np.ndarray:
    """Generate ensemble of recharge estimates by perturbing Sy and k.

    Sy를 +-20%, recession k를 +-0.002 섭동하여 앙상블 생성.
    각 앙상블 멤버마다 WTF 함양을 재계산한다.

    Parameters
    ----------
    ho : (n_days,) water level [m]
    P  : (n_days,) precipitation [m/day]
    Sy_base : float
        Base specific yield [-]
    tau : int
        Drainage time constant [days]
    n_ensemble : int
        Ensemble size
    rng : np.random.Generator
        Random number generator

    Returns
    -------
    R_ens : (n_ensemble,) recharge estimates [mm/yr]
    """
    # Sy 섭동: +-20%
    Sy_ens = np.clip(
        Sy_base * (1 + rng.normal(0, 0.20, n_ensemble)),
        0.01, 0.50,
    )

    R_ens = np.zeros(n_ensemble)
    for m in range(n_ensemble):
        R_ens[m] = _wtf_single_well(ho, P, Sy_ens[m], tau)

    return R_ens


# ──────────────────────────────────────────────────────────
# 메인 인터페이스
# ──────────────────────────────────────────────────────────
def estimate_recharge(domain, observations: dict,
                      n_ensemble: int = 100,
                      loc_radius_m: float = 3000.0,
                      obs_noise_mm: float = 20.0,
                      random_seed: int = 42) -> np.ndarray:
    """Estimate recharge using simplified EnKF R-state method.

    Steps:
      1. Compute recharge at each well using WTF (per-well Sy)
      2. Generate ensemble prior for each grid cell (by soil type)
      3. EnKF update: use well R as observations, Gaspari-Cohn localization
         (radius = 3 km), sequential assimilation
      4. Return posterior mean as recharge map

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
    n_ensemble : int
        Ensemble size (default 100)
    loc_radius_m : float
        Localization radius [m] (default 3000 = 3 km)
    obs_noise_mm : float
        Observation noise for well recharge [mm/yr] (default 20)
    random_seed : int
        Random seed for reproducibility (default 42)

    Returns
    -------
    recharge_map : np.ndarray, shape (ny, nx)
        Estimated annual recharge [mm/yr] for each grid cell
    """
    P = observations['P']
    ho_obs = observations['ho_obs']
    well_soil_types = observations['well_soil_types']
    n_wells = ho_obs.shape[0]
    ny, nx = domain.soil_map.shape

    rng = np.random.default_rng(random_seed)
    N = n_ensemble

    # ── Step 1: 관정 WTF 함양률 (관측값으로 사용) ──
    well_R = np.zeros(n_wells)
    for w in range(n_wells):
        soil_idx = int(well_soil_types[w])
        rec = SOIL_DB[soil_idx]
        well_R[w] = _wtf_single_well(ho_obs[w], P, rec.sy_lit, rec.tau)

    # 관정 좌표 [m]
    well_x = domain.x_centers[domain.well_cols]
    well_y = domain.y_centers[domain.well_rows]

    # ── Step 2: 토양 유형별 앙상블 사전분포 생성 ──
    # 효율을 위해 토양 유형별로 한 번만 앙상블 생성
    # 각 토양의 대표 관정 수위를 사용하여 WTF 계산
    unique_soils = np.unique(domain.soil_map)

    # 토양별 대표 관정 찾기 (해당 토양에 위치한 관정 중 첫 번째)
    soil_repr_well = {}
    for si in unique_soils:
        si = int(si)
        matching = [w for w in range(n_wells) if int(well_soil_types[w]) == si]
        if matching:
            soil_repr_well[si] = matching[0]
        else:
            # 해당 토양에 관정이 없으면 가장 가까운 관정 사용
            soil_repr_well[si] = 0

    # 토양별 앙상블 사전분포
    soil_prior_ens = {}
    for si in unique_soils:
        si = int(si)
        w_repr = soil_repr_well[si]
        rec = SOIL_DB[si]
        soil_prior_ens[si] = _generate_prior_ensemble(
            ho_obs[w_repr], P, rec.sy_lit, rec.tau, N, rng,
        )

    # 전체 격자 앙상블: (ny*nx, N)
    n_pts = ny * nx
    R_ens = np.zeros((n_pts, N))
    for idx in range(n_pts):
        row, col = divmod(idx, nx)
        si = int(domain.soil_map[row, col])
        R_ens[idx] = soil_prior_ens[si].copy()

    # ── Step 3: EnKF 업데이트 (sequential assimilation) ──
    R_post = R_ens.copy()

    for w in range(n_wells):
        R_obs = well_R[w]
        wx, wy = well_x[w], well_y[w]

        # 관정에 해당하는 격자 인덱스
        w_grid_idx = domain.well_rows[w] * nx + domain.well_cols[w]

        # 관정 앙상블의 공분산
        X_w = R_post[w_grid_idx]
        X_w_mean = X_w.mean()
        X_w_anom = X_w - X_w_mean
        PHT_ww = float(np.dot(X_w_anom, X_w_anom) / (N - 1))
        S = PHT_ww + obs_noise_mm**2

        if S < 1e-10:
            continue  # 공분산 붕괴 방지

        # 혁신 (stochastic EnKF: 관측에 노이즈 추가)
        innov = (R_obs + rng.normal(0, obs_noise_mm, N)) - X_w

        # 국소화 반경 내 격자점만 업데이트
        for idx in range(n_pts):
            row, col = divmod(idx, nx)
            cx = domain.x_centers[col]
            cy = domain.y_centers[row]

            dist = np.sqrt((cx - wx)**2 + (cy - wy)**2)
            L = _gaspari_cohn(dist, loc_radius_m)

            if L < 1e-6:
                continue  # 국소화 반경 밖

            # 교차 공분산
            X_i_anom = R_post[idx] - R_post[idx].mean()
            PHT_iw = float(np.dot(X_i_anom, X_w_anom) / (N - 1))

            # Kalman gain (국소화 적용)
            K = L * PHT_iw / S

            # 앙상블 업데이트
            R_post[idx] = R_post[idx] + K * innov

    # ── Step 4: 사후분포 평균 -> 함양 맵 ──
    R_mean = R_post.mean(axis=1)
    R_mean = np.maximum(R_mean, 0.0)  # 음수 방지
    recharge_map = R_mean.reshape(ny, nx)

    return recharge_map


# ──────────────────────────────────────────────────────────
# 테스트
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from synthetic.generate_domain import generate_domain, DomainConfig

    print("=== EnKF R-state Method Test (S3) ===\n")

    # 도메인 생성
    domain = generate_domain(DomainConfig.S3())
    print(domain.summary())
    print()

    # 합성 관측 데이터 생성 (테스트용)
    rng = np.random.default_rng(42)
    n_days = 730  # 2년
    n_wells = domain.n_wells
    ny, nx = domain.soil_map.shape

    # 강수: 간헐적 강우 패턴
    P = np.where(rng.random(n_days) < 0.15,
                 rng.exponential(0.005, n_days), 0.0)

    # ET: 계절 변동
    day_of_year = np.arange(n_days) % 365
    ET = 0.003 * (1 + 0.5 * np.sin(2 * np.pi * day_of_year / 365))

    # 관측 수위: 토양별 Sy에 따라 다른 수위 반응 생성
    ho_obs = np.zeros((n_wells, n_days))
    well_soil_types = np.array([
        int(domain.soil_map[domain.well_rows[w], domain.well_cols[w]])
        for w in range(n_wells)
    ])

    for w in range(n_wells):
        soil_idx = well_soil_types[w]
        Sy_true = SOIL_DB[soil_idx].sy_lit
        h_base = 5.0
        h = np.full(n_days, h_base)
        for t in range(1, n_days):
            # recession
            h[t] = h[t-1] - 0.005 * (h[t-1] - 3.0)
            # 강수 반응
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
    print("Running EnKF (n_ensemble=100, loc_radius=3km)...")
    rech_map = estimate_recharge(domain, observations,
                                 n_ensemble=100, random_seed=42)

    print(f"\nRecharge map shape: {rech_map.shape}")
    print(f"Recharge range: [{rech_map.min():.1f}, {rech_map.max():.1f}] mm/yr")
    print(f"Recharge mean:  {rech_map.mean():.1f} mm/yr")
    print(f"Recharge std:   {rech_map.std():.1f} mm/yr")

    # 토양 유형별 함양 확인
    unique_soils = np.unique(domain.soil_map)
    print("\n  Soil-type breakdown:")
    for st in unique_soils:
        mask_st = domain.soil_map == st
        r_mean = rech_map[mask_st].mean()
        r_std = rech_map[mask_st].std()
        frac = mask_st.sum() / (ny * nx)
        print(f"    {SOIL_DB[int(st)].name:<20s} "
              f"(Sy={SOIL_DB[int(st)].sy_lit:.2f}): "
              f"R={r_mean:.1f} +/- {r_std:.1f} mm/yr  "
              f"({frac*100:.1f}% area)")

    # 관정 근처 vs 먼 곳 비교 (공간 변동 확인)
    print("\n  Spatial variation (well proximity):")
    for w in range(min(5, n_wells)):
        wr, wc = domain.well_rows[w], domain.well_cols[w]
        r_at_well = rech_map[wr, wc]
        # 10셀 (1km) 떨어진 곳
        far_r = min(wr + 10, ny - 1)
        far_c = min(wc + 10, nx - 1)
        r_far = rech_map[far_r, far_c]
        print(f"    Well {w}: at_well={r_at_well:.1f}, "
              f"1km_away={r_far:.1f} mm/yr")
