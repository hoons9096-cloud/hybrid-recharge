"""
scs_cn.py -- Improved SCS Curve Number method for groundwater recharge estimation.

Estimates daily infiltration from precipitation using the SCS Curve Number
method with modern enhancements that go beyond the basic 1972 USDA formulation.

Improvements over Choi & Ahn (1998) which uses the unmodified SCS-CN:
1. Continuous AMC (Antecedent Moisture Condition) adjustment day-by-day
   based on previous 5-day rainfall — vs single average CN.
2. CN uncertainty propagation (CN ± delta) — vs deterministic point estimate.
3. Daily resolution with seasonal AMC thresholds — vs annual lump.
4. Per-cell CN derivation from soil_db texture group + land use lookup.

This module is the *second methodologically independent* recharge estimator
to complement the hybrid-recharge pipeline.  WTF measures "what got stored in the
aquifer"; SCS-CN measures "what infiltrated past the surface".  Convergence
between the two ⇒ defensible single-site recharge estimate.

References
----------
USDA-SCS (1972). National Engineering Handbook, Section 4 Hydrology.
    USDA Soil Conservation Service.
USDA-NRCS (2004). Part 630 Hydrology National Engineering Handbook,
    Chapter 10 — Estimation of Direct Runoff from Storm Rainfall.
Mishra, S.K. & Singh, V.P. (2003). Soil Conservation Service Curve Number
    (SCS-CN) Methodology.  Springer.  (AMC continuous formulation)
Chow, V.T., Maidment, D.R., Mays, L.W. (1988). Applied Hydrology.
    McGraw-Hill, Ch. 5.
Choi, B.S. & Ahn, J.G. (1998). A Study on the Estimation of Regional
    Groundwater Recharge Ratio.  대한지하수환경학회지, 5(2), 57–65.
    (Korean baseline reference; this module supersedes its CN treatment.)
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# soil_db 경로
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from soil_db import SOIL_DB


# ══════════════════════════════════════════════════════════════════════
# CN lookup tables (AMC II, 정상 토양수분 조건)
# ══════════════════════════════════════════════════════════════════════
# 한국 농촌 환경에 맞춰 USDA-NRCS (2004) 표 9-1·9-3 발췌 + 적용
#
# 토양 수문그룹 (Hydrologic Soil Group):
#   A: 매우 낮은 유출 (모래·양질사토)
#   B: 낮은 유출 (사양토·양토)
#   C: 보통 유출 (양질식토·식양토)
#   D: 높은 유출 (식토)
#
# 한국 토양도 SOIL_DB의 texture_group → 수문그룹 매핑
# 대부분의 한국 충적층은 A–B, 잔적풍화토는 B–C, 점토층은 D
TEXTURE_TO_HYDRO_GROUP: Dict[str, str] = {
    "coarse": "A",
    "medium": "B",
    "fine":   "C",  # 보수적 (Group D는 사용자 지정 가능)
}


# Land-use × hydrologic-soil-group → CN (AMC II)
# 한국 농촌 일반 지표면 위주 프리셋
LAND_USE_CN: Dict[str, Dict[str, int]] = {
    # 한국 사이트 일반 분류
    "혼합농경지":      {"A": 65, "B": 75, "C": 82, "D": 86},
    "논":             {"A": 60, "B": 70, "C": 78, "D": 85},
    "밭(직선경작)":   {"A": 67, "B": 78, "C": 85, "D": 89},
    "초지/공한지":    {"A": 49, "B": 69, "C": 79, "D": 84},
    # 산림
    "산림(good)":     {"A": 30, "B": 55, "C": 70, "D": 77},
    "산림(fair)":     {"A": 36, "B": 60, "C": 73, "D": 79},
    # 도시
    "주거(저밀도)":   {"A": 51, "B": 68, "C": 79, "D": 84},
    "주거(고밀도)":   {"A": 77, "B": 85, "C": 90, "D": 92},
}


# AMC 분류 임계값 — 직전 5일 누적강수량 (mm)
# USDA-NRCS 2004 Table 10-1 (성장기 vs 비성장기)
AMC_THRESHOLDS = {
    "growing": {"I_max": 36.0, "III_min": 53.0},  # 성장기
    "dormant": {"I_max": 13.0, "III_min": 28.0},  # 비성장기
}


# ══════════════════════════════════════════════════════════════════════
# 결과 dataclass
# ══════════════════════════════════════════════════════════════════════
@dataclass
class SCSCNResult:
    """Improved SCS-CN recharge estimation result."""
    # 핵심 추정값
    recharge_ratio_pct: float           # ΣF / ΣP × 100
    R_annual_mm: float                  # 연 침투량 (mm/yr)
    P_annual_mm: float                  # 연 강수량 (mm/yr)
    n_days: int
    n_wet_days: int                     # P > 0.1 mm
    n_runoff_days: int                  # Q > 0
    total_runoff_mm: float

    # CN 입력
    CN_input: float                     # 사용자 / 자동 도출 CN (AMC II)
    soil_hydro_group: str               # A/B/C/D
    land_use: str
    texture_group: Optional[str] = None  # soil_db에서 도출 시

    # 불확실성 (CN ±delta sensitivity)
    delta_cn: float = 5.0
    recharge_cn_low: float = 0.0        # CN-delta에서 함양율
    recharge_cn_high: float = 0.0       # CN+delta에서 함양율

    # AMC 분포
    n_amc_I: int = 0                    # 건조
    n_amc_II: int = 0                   # 보통
    n_amc_III: int = 0                  # 습윤

    # 일별 시계열 (선택 — 그림용)
    daily_runoff_mm: List[float] = field(default_factory=list)
    daily_infiltration_mm: List[float] = field(default_factory=list)
    daily_amc: List[str] = field(default_factory=list)

    @property
    def runoff_ratio_pct(self) -> float:
        if self.P_annual_mm <= 0:
            return 0.0
        return self.total_runoff_mm / (self.P_annual_mm * self.n_days / 365.25) \
               * 100.0

    @property
    def cn_uncertainty_band_pct(self) -> float:
        """CN ±delta로 인한 함양율 변동 폭 (%p)."""
        return self.recharge_cn_high - self.recharge_cn_low


# ══════════════════════════════════════════════════════════════════════
# 헬퍼 — CN 변환 (AMC I/II/III)
# ══════════════════════════════════════════════════════════════════════
def cn_to_amc_i(cn_ii: float) -> float:
    """Convert CN_II (normal) to CN_I (dry).  Hawkins et al. 1985."""
    if cn_ii <= 0 or cn_ii >= 100:
        return cn_ii
    return 4.2 * cn_ii / (10.0 - 0.058 * cn_ii)


def cn_to_amc_iii(cn_ii: float) -> float:
    """Convert CN_II (normal) to CN_III (wet).  Hawkins et al. 1985."""
    if cn_ii <= 0 or cn_ii >= 100:
        return cn_ii
    return 23.0 * cn_ii / (10.0 + 0.13 * cn_ii)


def classify_amc(
    prev_5day_rain_mm: float,
    is_growing_season: bool = True,
) -> str:
    """Classify AMC from previous 5-day cumulative precipitation.

    Returns
    -------
    "I", "II", or "III"
    """
    th = AMC_THRESHOLDS["growing" if is_growing_season else "dormant"]
    if prev_5day_rain_mm < th["I_max"]:
        return "I"
    elif prev_5day_rain_mm > th["III_min"]:
        return "III"
    else:
        return "II"


# ══════════════════════════════════════════════════════════════════════
# CN 도출 (토양 + 토지이용)
# ══════════════════════════════════════════════════════════════════════
def soil_group_from_texture(texture_group: str) -> str:
    """SOIL_DB texture_group → SCS hydrologic group."""
    return TEXTURE_TO_HYDRO_GROUP.get(texture_group, "B")


def derive_cn(soil_hydro_group: str, land_use: str) -> float:
    """Look up CN_II for (soil_group, land_use)."""
    if land_use not in LAND_USE_CN:
        raise ValueError(
            f"Unknown land_use '{land_use}'. "
            f"Available: {list(LAND_USE_CN.keys())}"
        )
    if soil_hydro_group not in LAND_USE_CN[land_use]:
        raise ValueError(
            f"Unknown soil group '{soil_hydro_group}'. Use A/B/C/D."
        )
    return float(LAND_USE_CN[land_use][soil_hydro_group])


def derive_cn_from_soil_db(
    sn_idx: int,
    land_use: str = "혼합농경지",
) -> Tuple[float, str]:
    """Derive CN from soil_db sn_idx + land_use string.

    Returns
    -------
    (CN, soil_group_letter)
    """
    soil = SOIL_DB[int(sn_idx)]
    group = soil_group_from_texture(soil.texture_group)
    cn = derive_cn(group, land_use)
    return cn, group


# ══════════════════════════════════════════════════════════════════════
# 핵심 알고리즘 — 일별 SCS-CN
# ══════════════════════════════════════════════════════════════════════
def _runoff_q(P_mm: float, S_mm: float, ia_ratio: float = 0.2) -> Tuple[float, float]:
    """Single-day SCS runoff & infiltration.

    Q = (P - Ia)² / (P - Ia + S)   if P > Ia
        0                          otherwise

    Returns (Q_mm, F_mm) where F = P - Q.
    """
    Ia = ia_ratio * S_mm
    if P_mm <= Ia or P_mm <= 0:
        return 0.0, max(P_mm, 0.0)
    excess = P_mm - Ia
    Q = excess * excess / (excess + S_mm)
    F = P_mm - Q
    return float(Q), float(F)


def _doy_is_growing(day_idx: int, start_doy: int = 1) -> bool:
    """4월~10월 (DOY 91~304)을 성장기로 간주."""
    doy = ((day_idx + start_doy - 1) % 365) + 1
    return 91 <= doy <= 304


def estimate_recharge_scs_cn(
    P_daily_mm: np.ndarray,
    CN: float,
    apply_amc_correction: bool = True,
    delta_cn_uncertainty: float = 5.0,
    soil_hydro_group: str = "B",
    land_use: str = "혼합농경지",
    texture_group: Optional[str] = None,
    start_doy: int = 1,
    return_daily: bool = True,
) -> SCSCNResult:
    """Run improved SCS-CN over a daily precipitation series.

    Parameters
    ----------
    P_daily_mm : (n,) np.ndarray
        Daily precipitation [mm/day].  Must be non-negative.
    CN : float
        Curve Number at AMC II (normal moisture).  10 < CN < 100.
    apply_amc_correction : bool
        If True (default), reclassify CN per day using prev-5-day rainfall
        and adjust to CN_I or CN_III as needed.  This is the modernization
        over Choi & Ahn 1998.
    delta_cn_uncertainty : float
        CN ± delta sensitivity range for uncertainty band.  Default 5.
    soil_hydro_group, land_use, texture_group : str
        Metadata only — passed through to result for reporting.
    start_doy : int
        Day-of-year for index 0 of P_daily_mm.  Default 1 (Jan 1).
        Used for AMC season classification.
    return_daily : bool
        If True, populate daily_* arrays in result.  Set False for memory.

    Returns
    -------
    SCSCNResult
    """
    P = np.asarray(P_daily_mm, dtype=float)
    if P.ndim != 1:
        raise ValueError(f"P_daily_mm must be 1-D, got shape {P.shape}")
    if not (10 < CN < 100):
        raise ValueError(f"CN must be in (10, 100), got {CN}")
    if np.any(P < 0):
        raise ValueError("P_daily_mm contains negative values")

    n = len(P)

    # 사전 변환: CN_I, CN_III
    cn_i = cn_to_amc_i(CN)
    cn_iii = cn_to_amc_iii(CN)
    s_ii = 25400.0 / CN - 254.0
    s_i = 25400.0 / cn_i - 254.0
    s_iii = 25400.0 / cn_iii - 254.0

    Q_daily = np.zeros(n)
    F_daily = np.zeros(n)
    amc_daily: List[str] = []

    n_amc = {"I": 0, "II": 0, "III": 0}

    for t in range(n):
        # AMC 결정
        if apply_amc_correction:
            i0 = max(0, t - 5)
            prev5 = float(np.sum(P[i0:t]))  # t 이전 5일
            growing = _doy_is_growing(t, start_doy)
            amc = classify_amc(prev5, is_growing_season=growing)
        else:
            amc = "II"

        n_amc[amc] += 1
        amc_daily.append(amc)

        # 해당 AMC의 S
        if amc == "I":
            S = s_i
        elif amc == "III":
            S = s_iii
        else:
            S = s_ii

        Q, F = _runoff_q(float(P[t]), S)
        Q_daily[t] = Q
        F_daily[t] = F

    sum_P = float(np.sum(P))
    sum_F = float(np.sum(F_daily))
    sum_Q = float(np.sum(Q_daily))
    rech_pct = (sum_F / sum_P * 100.0) if sum_P > 0 else 0.0
    n_years = max(n / 365.25, 1.0)
    R_annual = sum_F / n_years
    P_annual = sum_P / n_years
    n_wet = int(np.sum(P > 0.1))
    n_runoff = int(np.sum(Q_daily > 0))

    # 불확실성 — CN ± delta
    if delta_cn_uncertainty > 0:
        cn_lo = max(10.1, CN - delta_cn_uncertainty)
        cn_hi = min(99.9, CN + delta_cn_uncertainty)
        rech_lo = _quick_recharge_pct(P, cn_lo, apply_amc_correction, start_doy)
        rech_hi = _quick_recharge_pct(P, cn_hi, apply_amc_correction, start_doy)
        # CN이 클수록 유출↑ → 함양↓ 이므로 lo/hi 정렬
        rech_cn_low = min(rech_lo, rech_hi)
        rech_cn_high = max(rech_lo, rech_hi)
    else:
        rech_cn_low = rech_pct
        rech_cn_high = rech_pct

    return SCSCNResult(
        recharge_ratio_pct=rech_pct,
        R_annual_mm=R_annual,
        P_annual_mm=P_annual,
        n_days=n,
        n_wet_days=n_wet,
        n_runoff_days=n_runoff,
        total_runoff_mm=sum_Q,
        CN_input=float(CN),
        soil_hydro_group=soil_hydro_group,
        land_use=land_use,
        texture_group=texture_group,
        delta_cn=float(delta_cn_uncertainty),
        recharge_cn_low=rech_cn_low,
        recharge_cn_high=rech_cn_high,
        n_amc_I=n_amc["I"],
        n_amc_II=n_amc["II"],
        n_amc_III=n_amc["III"],
        daily_runoff_mm=Q_daily.tolist() if return_daily else [],
        daily_infiltration_mm=F_daily.tolist() if return_daily else [],
        daily_amc=amc_daily if return_daily else [],
    )


def _quick_recharge_pct(
    P: np.ndarray, CN: float,
    apply_amc: bool, start_doy: int,
) -> float:
    """불확실성 계산용 경량 함양율 (%) — 시계열 저장 안 함."""
    cn_i = cn_to_amc_i(CN)
    cn_iii = cn_to_amc_iii(CN)
    s_ii = 25400.0 / CN - 254.0
    s_i = 25400.0 / cn_i - 254.0
    s_iii = 25400.0 / cn_iii - 254.0

    n = len(P)
    sum_F = 0.0
    for t in range(n):
        if apply_amc:
            i0 = max(0, t - 5)
            prev5 = float(np.sum(P[i0:t]))
            growing = _doy_is_growing(t, start_doy)
            amc = classify_amc(prev5, is_growing_season=growing)
        else:
            amc = "II"
        S = s_i if amc == "I" else (s_iii if amc == "III" else s_ii)
        _, F = _runoff_q(float(P[t]), S)
        sum_F += F

    sum_P = float(np.sum(P))
    return (sum_F / sum_P * 100.0) if sum_P > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════
# 자체 시연
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=== scs_cn.py self-test ===\n")

    # 한국 몬순 기후 모사 — 2년 일별 강수, 연 ~1200 mm 목표
    rng = np.random.default_rng(42)
    n = 730
    doy = np.arange(n) % 365
    # 여름 몬순 집중: 습윤일 확률 + 강도 모두 계절 변동
    wet_prob = 0.18 + 0.20 * np.sin(2 * np.pi * (doy - 80) / 365)
    wet_prob = np.clip(wet_prob, 0.05, 0.55)
    is_wet = rng.random(n) < wet_prob
    intensity_scale = 8.0 + 25.0 * np.clip(
        np.sin(2 * np.pi * (doy - 80) / 365), 0.0, 1.0
    )
    intensity = np.where(is_wet, rng.exponential(intensity_scale), 0.0)
    intensity = np.clip(intensity, 0, 200)

    # 혼합농경지, 토양 그룹 B (sn=6 가정 = Loam)
    cn, group = derive_cn_from_soil_db(sn_idx=6, land_use="혼합농경지")
    print(f"CN derived: sn=6 (Loam) + 혼합농경지 + Group {group} → CN = {cn}")

    result = estimate_recharge_scs_cn(
        P_daily_mm=intensity,
        CN=cn,
        soil_hydro_group=group,
        land_use="혼합농경지",
        texture_group="medium",
    )

    print(f"\n--- Result ---")
    print(f"  Annual P:        {result.P_annual_mm:.0f} mm/yr")
    print(f"  Annual recharge: {result.R_annual_mm:.0f} mm/yr")
    print(f"  Recharge ratio:  {result.recharge_ratio_pct:.2f}%")
    print(f"  Runoff total:    {result.total_runoff_mm:.0f} mm "
          f"({result.runoff_ratio_pct:.1f}%)")
    print(f"  Wet days:        {result.n_wet_days}/{result.n_days}")
    print(f"  Runoff events:   {result.n_runoff_days}")
    print(f"\n--- AMC distribution ---")
    print(f"  AMC I (dry):     {result.n_amc_I}")
    print(f"  AMC II (normal): {result.n_amc_II}")
    print(f"  AMC III (wet):   {result.n_amc_III}")
    print(f"\n--- Uncertainty (CN ±{result.delta_cn:.0f}) ---")
    print(f"  Recharge band:   [{result.recharge_cn_low:.2f}, "
          f"{result.recharge_cn_high:.2f}]% "
          f"(±{result.cn_uncertainty_band_pct/2:.2f}%p)")
