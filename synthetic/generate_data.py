"""
generate_data.py -- 합성 시계열 데이터 생성기

SyntheticDomain을 입력받아 강수, 증발산, 참 함양, 지하수위 시계열을 생성한다.
모든 데이터는 논문의 controlled experiment 용도로, 참값(true recharge)이
알려진 상태에서 방법론 비교를 가능하게 한다.

생성 항목:
    1. 강수량 (P) -- 일 단위, 730일 (2년), 몬순 기후 패턴
    2. 증발산 (ET) -- 일 단위, 사인파 계절 변동
    3. 참 함양 (true recharge) -- 격자별, 토양 alpha_recharge 기반
    4. 지하수위 (ho) -- 참 함양에서 전진 시뮬레이션
    5. 관측 수위 -- 참 수위 + 가우시안 노이즈

Usage:
    from synthetic.generate_domain import generate_domain, DomainConfig
    from synthetic.generate_data import generate_data

    domain = generate_domain(DomainConfig.S3())
    data = generate_data(domain)
"""
from __future__ import annotations

import sys
import os
from dataclasses import dataclass

import numpy as np

# soil_db 임포트 경로 설정 (generate_domain.py와 동일)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from soil_db import SOIL_DB

from synthetic.generate_domain import SyntheticDomain, DomainConfig, generate_domain


# ──────────────────────────────────────────────────────────
# 결과 데이터 클래스
# ──────────────────────────────────────────────────────────
@dataclass
class SyntheticData:
    """Generated synthetic time-series data.

    All units are in meters and days unless otherwise noted.
    """
    # 기상 입력
    P: np.ndarray               # (n_days,) 강수량 [m/day]
    ET: np.ndarray              # (n_days,) 증발산 [m/day]

    # 참 함양
    true_recharge_map: np.ndarray   # (ny, nx, n_days) 격자별 참 함양 [m/day]
    true_recharge_annual: np.ndarray  # (ny, nx) 연평균 함양 [mm/yr]

    # 지하수위 (관측정 위치만)
    ho_obs: np.ndarray          # (n_wells, n_days) 관측 수위 [m]
    ho_true: np.ndarray         # (n_wells, n_days) 참 수위 [m]

    # 관측정별 토양 유형
    well_soil_types: np.ndarray  # (n_wells,) 토양 인덱스 (1-based)

    @property
    def n_days(self) -> int:
        return len(self.P)

    @property
    def n_wells(self) -> int:
        return self.ho_obs.shape[0]

    def summary(self) -> str:
        """Print data summary."""
        n_years = self.n_days / 365.0
        total_P_mm = float(np.sum(self.P)) * 1000.0
        annual_P_mm = total_P_mm / n_years
        total_ET_mm = float(np.sum(self.ET)) * 1000.0

        # 전체 도메인 평균 연 함양
        mean_annual_rech = float(np.mean(self.true_recharge_annual))

        # 관측정별 수위 범위
        h_min = float(np.min(self.ho_true))
        h_max = float(np.max(self.ho_true))

        # 관측 노이즈 실측 표준편차
        noise_actual = float(np.std(self.ho_obs - self.ho_true))

        lines = [
            f"--- Synthetic Data Summary ---",
            f"  Duration       : {self.n_days} days ({n_years:.1f} years)",
            f"  Total P        : {total_P_mm:.1f} mm",
            f"  Annual P       : {annual_P_mm:.0f} mm/yr",
            f"  Total ET       : {total_ET_mm:.1f} mm",
            f"  Wet days       : {int(np.sum(self.P > 0))} / {self.n_days}"
            f" ({np.sum(self.P > 0) / self.n_days * 100:.1f}%)",
            f"  Mean annual R  : {mean_annual_rech:.1f} mm/yr"
            f" (R/P = {mean_annual_rech / annual_P_mm * 100:.1f}%)",
            f"  Recharge range : [{np.min(self.true_recharge_annual):.1f},"
            f" {np.max(self.true_recharge_annual):.1f}] mm/yr",
            f"  Wells          : {self.n_wells}",
            f"  h range        : [{h_min:.2f}, {h_max:.2f}] m",
            f"  Obs noise (std): {noise_actual:.4f} m",
        ]
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────
# 강수량 생성
# ──────────────────────────────────────────────────────────
def _generate_precipitation(
    n_days: int,
    rng: np.random.Generator,
    wet_fraction: float = 0.25,
    mean_intensity_mm: float = 15.0,
) -> np.ndarray:
    """Generate daily precipitation with monsoon-climate seasonality.

    한국 기후 기준 연 강수량 ~1000-1200 mm/yr을 목표로 설계.
    여름 몬순에 집중(확률·강도 모두 계절 변동).

    Parameters
    ----------
    n_days : int
        Number of simulation days.
    rng : numpy Generator
        Random number generator.
    wet_fraction : float
        Overall *mean* fraction of wet days (~25%).
    mean_intensity_mm : float
        Annual mean event intensity on wet days [mm].

    Returns
    -------
    np.ndarray
        (n_days,) precipitation [m/day].
    """
    P = np.zeros(n_days)
    day_of_year = np.arange(n_days) % 365

    # 계절별 습윤 확률: wet_fraction을 중심으로 ±50% 변동
    # 여름 피크 ~DOY 200 (7월 중순), 겨울 최소 ~DOY 15 (1월)
    seasonal_mod = np.sin(2.0 * np.pi * (day_of_year - 80) / 365.0)
    seasonal_prob = wet_fraction * (1.0 + 0.5 * seasonal_mod)
    seasonal_prob = np.clip(seasonal_prob, 0.05, 0.60)

    # 습윤/건조일 결정
    is_wet = rng.random(n_days) < seasonal_prob

    # 강수 강도: 계절에 따라 평균 강도 변동
    # 여름 몬순: 강도 1.5배, 겨울: 0.5배
    seasonal_intensity = mean_intensity_mm * (1.0 + 0.5 * seasonal_mod)
    seasonal_intensity = np.clip(seasonal_intensity, 5.0, 40.0)

    # 지수분포 샘플링 (습윤일만)
    intensities_mm = rng.exponential(
        scale=seasonal_intensity[is_wet],
    )
    # 최소 0.5mm, 최대 150mm (극한 강수 제한)
    intensities_mm = np.clip(intensities_mm, 0.5, 150.0)

    P[is_wet] = intensities_mm / 1000.0  # mm -> m

    return P


# ──────────────────────────────────────────────────────────
# 증발산 생성
# ──────────────────────────────────────────────────────────
def _generate_et(
    n_days: int,
    rng: np.random.Generator,
    et_max_mm: float = 5.0,
    et_min_mm: float = 1.0,
) -> np.ndarray:
    """Generate daily evapotranspiration with sinusoidal seasonal pattern.

    Parameters
    ----------
    n_days : int
        Number of simulation days.
    rng : numpy Generator
        Random number generator.
    et_max_mm : float
        Maximum daily ET in summer [mm/day].
    et_min_mm : float
        Minimum daily ET in winter [mm/day].

    Returns
    -------
    np.ndarray
        (n_days,) evapotranspiration [m/day].
    """
    day_of_year = np.arange(n_days) % 365

    # 사인파 계절 패턴: 여름 최대, 겨울 최소
    # 피크: DOY ~200 (7월 중순)
    et_mean = (et_max_mm + et_min_mm) / 2.0
    et_amp = (et_max_mm - et_min_mm) / 2.0
    ET_mm = et_mean + et_amp * np.sin(2.0 * np.pi * (day_of_year - 80) / 365.0)

    # 일별 랜덤 변동 (CV ~10%)
    noise = rng.normal(1.0, 0.10, n_days)
    noise = np.clip(noise, 0.5, 1.5)
    ET_mm = ET_mm * noise
    ET_mm = np.clip(ET_mm, 0.3, et_max_mm * 1.5)

    return ET_mm / 1000.0  # mm -> m


# ──────────────────────────────────────────────────────────
# 참 함양 계산
# ──────────────────────────────────────────────────────────
def _compute_true_recharge_cascade(
    P: np.ndarray,
    ET: np.ndarray,
    domain: SyntheticDomain,
    n_layers: int = 5,
    L_total_m: float = 2.0,
    init_theta_frac: float = 0.5,
) -> np.ndarray:
    """Phase 2: Multi-layer cascade vadose model 로 true recharge 산출.

    각 (ny, nx) 셀의 soil_map[i,j] 에 따라 cascade simulator 실행.
    효율: unique soil 별로 한 번만 시뮬레이션 후 broadcast.

    Returns
    -------
    np.ndarray : (ny, nx, n_days) true recharge [m/day]
    """
    from .vadose_cascade import build_layers_from_sn, simulate_cascade

    ny, nx = domain.config.ny, domain.config.nx
    n_days = len(P)
    P_mm = P * 1000.0   # m → mm
    ET_mm = ET * 1000.0

    # Unique soil types in domain
    unique_sn = sorted(set(int(s) for s in domain.soil_map.flatten()))
    rech_per_soil = {}    # sn → (n_days,) [m/day]
    for sn in unique_sn:
        layers = build_layers_from_sn(
            sn, L_total_m=L_total_m, n_layers=n_layers, root_decay=1.5,
        )
        r = simulate_cascade(P_mm, ET_mm, layers,
                             init_theta_frac=init_theta_frac)
        rech_per_soil[sn] = r.recharge / 1000.0   # mm → m

    # Broadcast: (ny, nx, n_days)
    Rech = np.zeros((ny, nx, n_days))
    for sn, rech_t in rech_per_soil.items():
        mask = (domain.soil_map == sn)
        Rech[mask] = rech_t
    return Rech


def _compute_true_recharge(
    P: np.ndarray,
    ET: np.ndarray,
    domain: SyntheticDomain,
    rng: np.random.Generator,
    net_p_threshold_mm: float = 5.0,
) -> np.ndarray:
    """Compute true recharge for each grid cell.

    Recharge model:
        net_P(t)     = max(P(t) - ET(t), 0)
        Rech(i,j,t)  = alpha_recharge(i,j) * net_P(t)   if net_P > threshold
                      = 0                                 otherwise

    ET를 초과하는 순 강수에서만 함양이 발생한다.
    이는 물리적으로 더 현실적이며, 뚜렷한 이벤트 위주의 함양 패턴을
    생성하여 WTF 방법과의 일관성을 높인다.

    토양별 alpha_recharge에 약간의 공간 변동 추가 (CV ~5%)

    Parameters
    ----------
    P : np.ndarray
        (n_days,) precipitation [m/day].
    ET : np.ndarray
        (n_days,) evapotranspiration [m/day].
    domain : SyntheticDomain
        Domain with alpha_map.
    rng : numpy Generator
        Random number generator.
    net_p_threshold_mm : float
        Minimum net precipitation (P-ET) for recharge [mm/day].

    Returns
    -------
    np.ndarray
        (ny, nx, n_days) true recharge [m/day].
    """
    ny, nx = domain.config.ny, domain.config.nx
    n_days = len(P)
    net_p_threshold = net_p_threshold_mm / 1000.0  # mm -> m

    # 순 강수: P - ET (음수 → 0)
    net_P = np.maximum(P - ET, 0.0)  # (n_days,)

    # alpha 맵에 셀 단위 미세 변동 추가 (재현성 위해 rng 사용)
    alpha_noise = rng.normal(1.0, 0.05, (ny, nx))
    alpha_noise = np.clip(alpha_noise, 0.80, 1.20)
    alpha_local = domain.alpha_map * alpha_noise
    alpha_local = np.clip(alpha_local, 0.0, 1.0)

    # 함양 계산: 순 강수가 임계값 이상일 때만
    rech_mask = net_P > net_p_threshold  # (n_days,)

    # 벡터화: (ny, nx, 1) * (1, 1, n_days)
    Rech = alpha_local[:, :, np.newaxis] * net_P[np.newaxis, np.newaxis, :]
    # 임계값 미만은 0
    Rech[:, :, ~rech_mask] = 0.0

    return Rech


# ──────────────────────────────────────────────────────────
# 지하수위 전진 시뮬레이션
# ──────────────────────────────────────────────────────────
# 대수층 recession 상수 (토양 texture group 기반)
# 비포화대 tau와 다름 — 대수층 배수는 투수량계수·저류계수·경계조건 의존
#
# 반감기:
#   coarse: ln(2)/0.03 ≈ 23일 (빠른 지하수 배수)
#   medium: ln(2)/0.015 ≈ 46일
#   fine:   ln(2)/0.005 ≈ 139일 (느린 점토질 대수층)
#
# Healy & Cook (2002) 보고 범위 -0.001 ~ -0.1과 일치.
_AQUIFER_K = {
    "coarse": -0.03,
    "medium": -0.015,
    "fine": -0.005,
}


def _simulate_groundwater_levels(
    Rech: np.ndarray,
    domain: SyntheticDomain,
    h_eq: float = 10.0,
    h_init: float = 10.0,
) -> np.ndarray:
    """Forward-simulate groundwater levels at well locations.

    Water table dynamics:
        dh(t) = Rech(t) / Sy
        h(t) = h(t-1) + k_aq * (h(t-1) - h_eq) + dh(t)

    k_aq는 대수층 recession 상수로, 비포화대 tau와 구분된다.
    토양 texture group에 따라 대수층 특성을 대리한다.

    Parameters
    ----------
    Rech : np.ndarray
        (ny, nx, n_days) true recharge [m/day].
    domain : SyntheticDomain
        Domain with Sy_map, tau_map, well locations.
    h_eq : float
        Equilibrium water level [m] (long-term base level).
    h_init : float
        Initial water level [m].

    Returns
    -------
    np.ndarray
        (n_wells, n_days) true groundwater levels at wells [m].
    """
    n_days = Rech.shape[2]
    n_wells = domain.n_wells

    ho_true = np.zeros((n_wells, n_days))

    for w in range(n_wells):
        row = domain.well_rows[w]
        col = domain.well_cols[w]

        Sy = domain.Sy_map[row, col]
        soil_idx = int(domain.soil_map[row, col])
        texture_group = SOIL_DB[soil_idx].texture_group

        # 대수층 recession 상수 (texture group 기반)
        k_aq = _AQUIFER_K[texture_group]

        # 해당 셀의 함양 시계열
        rech_well = Rech[row, col, :]  # (n_days,)

        # 전진 시뮬레이션
        h = np.zeros(n_days)
        h[0] = h_init

        for t in range(1, n_days):
            # 함양에 의한 수위 상승
            dh = rech_well[t] / Sy
            # recession: 평형 수위로 지수 감쇠
            recession = k_aq * (h[t - 1] - h_eq)
            h[t] = h[t - 1] + recession + dh

        ho_true[w, :] = h

    return ho_true


# ──────────────────────────────────────────────────────────
# 메인 생성 함수
# ──────────────────────────────────────────────────────────
def generate_data(
    domain: SyntheticDomain,
    n_days: int = 730,
    recharge_model: str = "alpha",   # "alpha" | "cascade"
) -> SyntheticData:
    """Generate synthetic time-series data for a given domain.

    Parameters
    ----------
    domain : SyntheticDomain
        Domain generated by generate_domain().
    n_days : int
        Number of simulation days (default: 730 = 2 years).

    Returns
    -------
    SyntheticData
        Complete synthetic dataset with precipitation, ET, recharge,
        and groundwater levels.
    """
    rng = np.random.default_rng(domain.config.random_seed)

    # 1. 강수량 생성
    P = _generate_precipitation(n_days, rng)

    # 2. 증발산 생성
    ET = _generate_et(n_days, rng)

    # 3. 참 함양 계산
    if recharge_model == "cascade":
        # Phase 2 — multi-layer cascade vadose 모델 (FAO-56/SWAT 계열)
        true_recharge_map = _compute_true_recharge_cascade(P, ET, domain)
    else:
        # 기본 — alpha_recharge × (P-ET) 단순 모델
        true_recharge_map = _compute_true_recharge(P, ET, domain, rng)

    # 4. 연평균 함양 [mm/yr]
    n_years = n_days / 365.0
    true_recharge_annual = (
        np.sum(true_recharge_map, axis=2) * 1000.0 / n_years
    )  # (ny, nx) [mm/yr]

    # 5. 지하수위 전진 시뮬레이션 (관측정 위치)
    ho_true = _simulate_groundwater_levels(true_recharge_map, domain)

    # 6. 관측 노이즈 추가
    obs_noise = rng.normal(
        0.0,
        domain.config.obs_noise_std,
        ho_true.shape,
    )
    ho_obs = ho_true + obs_noise

    # 7. 관측정별 토양 유형
    well_soil_types = np.array([
        int(domain.soil_map[domain.well_rows[w], domain.well_cols[w]])
        for w in range(domain.n_wells)
    ])

    return SyntheticData(
        P=P,
        ET=ET,
        true_recharge_map=true_recharge_map,
        true_recharge_annual=true_recharge_annual,
        ho_obs=ho_obs,
        ho_true=ho_true,
        well_soil_types=well_soil_types,
    )


# ──────────────────────────────────────────────────────────
# CLI 실행
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Generating S3 domain ===")
    domain = generate_domain(DomainConfig.S3())
    print(domain.summary())
    print()

    print("=== Generating synthetic data ===")
    data = generate_data(domain)
    print(data.summary())
    print()

    # 토양 유형별 함양 통계
    print("=== Recharge by soil type ===")
    for si in sorted(domain.soil_fractions.keys()):
        rec = SOIL_DB[si]
        mask = domain.soil_map == si
        rech_soil = data.true_recharge_annual[mask]
        if len(rech_soil) > 0:
            print(
                f"  {rec.name:<20s}: "
                f"mean={np.mean(rech_soil):6.1f} mm/yr, "
                f"std={np.std(rech_soil):5.1f}, "
                f"alpha={rec.alpha_recharge:.2f}"
            )

    # 관측정별 수위 변동 범위
    print()
    print("=== Well water level ranges ===")
    for w in range(min(5, data.n_wells)):
        si = data.well_soil_types[w]
        rec = SOIL_DB[si]
        h_range = np.max(data.ho_true[w]) - np.min(data.ho_true[w])
        print(
            f"  Well {w:2d} ({rec.name:<15s}, Sy={rec.sy_lit:.2f}): "
            f"dh={h_range:.3f} m"
        )
