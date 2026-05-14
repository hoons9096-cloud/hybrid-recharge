"""
fao56_swb.py -- FAO-56 daily soil-water-balance recharge estimator.

Computes daily reference ET (ET₀) via Hargreaves (1985), applies a crop
coefficient to obtain crop ET (ETc), runs a single-bucket soil-water
balance, and reports deep percolation as the recharge estimate.

This is the methodologically-independent companion to the hybrid-recharge
pipeline.  WTF measures storage change in the aquifer; FAO-56 SWB measures
the surface→subsurface pathway.  Convergence between the two ⇒ defensible
single-site recharge estimate.

Why FAO-56 (1998) rather than SCS-CN (1972) as the second method
-----------------------------------------------------------------
* SCS-CN reports *infiltration* (= P − Q), not *recharge*.  ET losses are
  not subtracted, so SCS-CN systematically over-estimates true recharge.
* FAO-56 reports *deep percolation past the root zone* — what actually
  reaches the saturated zone.  This is the physically meaningful
  recharge.
* International standard for irrigation/water-balance work
  (Allen et al. 1998, FAO Irrigation and Drainage Paper 56;
  ASCE-EWRI 2005 Standardised Reference Evapotranspiration Equation).

Method outline
--------------
For each day t:
    ETo(t)   = Hargreaves(Tmean, Tmax, Tmin, latitude, DOY)
    Kc(t)    = crop coefficient (preset growth-stage curve)
    ETc(t)   = ETo(t) * Kc(t)
    AW_max   = field_capacity − wilting_point  (root zone, mm)
    AW(t)    = AW(t-1) + (P(t) − Q(t)) − ETa(t) + ΔAW
       where Q(t) is runoff (SCS-CN style threshold) and
       ETa(t) = ETc(t) reduced if AW < stress threshold (FAO-56 §43).
    If AW(t) > AW_max:
         deep_perc(t) = AW(t) - AW_max
         AW(t)        = AW_max
    else: deep_perc(t) = 0

    Annual recharge ratio = Σ deep_perc / Σ P × 100 %

References
----------
Allen, R.G., Pereira, L.S., Raes, D., Smith, M. (1998). Crop
    evapotranspiration — Guidelines for computing crop water requirements.
    FAO Irrigation and Drainage Paper 56.  Rome.
Hargreaves, G.H. & Samani, Z.A. (1985). Reference crop evapotranspiration
    from temperature.  Applied Engineering in Agriculture, 1(2), 96–99.
ASCE-EWRI (2005). The ASCE Standardised Reference Evapotranspiration
    Equation.  ASCE-EWRI Task Committee on Standardization.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ══════════════════════════════════════════════════════════════════════
# 작물계수 프리셋 (FAO-56 표 12·17에서 발췌·한국 적용)
# ══════════════════════════════════════════════════════════════════════
# 형식: (Kc_ini, Kc_mid, Kc_end)
# 각 단계 일수 (kc_stages): (L_ini, L_dev, L_mid, L_late) — 일
KC_PRESETS: Dict[str, Dict] = {
    "혼합농경지": {
        "Kc": (0.50, 1.00, 0.85),
        "stages": (30, 30, 80, 30),
        "growing_doy_start": 105,  # 4월 중순
    },
    "논": {
        "Kc": (1.05, 1.20, 0.90),
        "stages": (30, 30, 60, 30),  # 이앙~출수~수확
        "growing_doy_start": 135,    # 5월 중순 이앙
    },
    "밭(직선경작)": {
        "Kc": (0.40, 1.15, 0.70),
        "stages": (25, 35, 50, 30),
        "growing_doy_start": 105,
    },
    "산림(활엽수림)": {
        "Kc": (1.00, 1.10, 0.90),
        "stages": (60, 30, 120, 30),
        "growing_doy_start": 90,
    },
    "초지/공한지": {
        "Kc": (0.40, 0.85, 0.85),
        "stages": (10, 20, 200, 30),  # 거의 연중 일정
        "growing_doy_start": 90,
    },
}


def kc_curve(land_use: str, n_days: int, start_doy: int = 1) -> np.ndarray:
    """Build daily Kc curve from preset stage profile.

    FAO-56 §44 sketch:
        Kc(t) = Kc_ini for [0, L_ini)
        linear ramp to Kc_mid over L_dev
        Kc_mid for [L_ini+L_dev, L_ini+L_dev+L_mid)
        linear ramp to Kc_end over L_late
        Kc_ini outside the growing season

    Parameters
    ----------
    land_use : str
        Key into KC_PRESETS.
    n_days : int
        Output array length.
    start_doy : int
        DOY of index 0 (default 1).

    Returns
    -------
    Kc : (n_days,) np.ndarray
    """
    if land_use not in KC_PRESETS:
        raise ValueError(
            f"Unknown land_use '{land_use}'. Available: {list(KC_PRESETS)}"
        )
    p = KC_PRESETS[land_use]
    kc_ini, kc_mid, kc_end = p["Kc"]
    L_ini, L_dev, L_mid, L_late = p["stages"]
    g_start = p["growing_doy_start"]

    Kc = np.full(n_days, kc_ini, dtype=float)
    for t in range(n_days):
        doy = ((t + start_doy - 1) % 365) + 1
        days_into_season = (doy - g_start) % 365
        if days_into_season < 0 or days_into_season >= (L_ini + L_dev + L_mid + L_late):
            Kc[t] = kc_ini  # 비성장기
            continue
        if days_into_season < L_ini:
            Kc[t] = kc_ini
        elif days_into_season < L_ini + L_dev:
            frac = (days_into_season - L_ini) / max(L_dev, 1)
            Kc[t] = kc_ini + (kc_mid - kc_ini) * frac
        elif days_into_season < L_ini + L_dev + L_mid:
            Kc[t] = kc_mid
        else:
            frac = (days_into_season - L_ini - L_dev - L_mid) / max(L_late, 1)
            Kc[t] = kc_mid + (kc_end - kc_mid) * frac
    return Kc


# ══════════════════════════════════════════════════════════════════════
# Hargreaves Reference ET (FAO-56 §B.5)
# ══════════════════════════════════════════════════════════════════════
def extraterrestrial_radiation_mj(doy: np.ndarray, lat_deg: float) -> np.ndarray:
    """Daily extraterrestrial radiation Ra (MJ/m²/day) — FAO-56 eq. 21.

    Parameters
    ----------
    doy : (n,) array of day-of-year (1–366)
    lat_deg : float, latitude in decimal degrees (positive N)

    Returns
    -------
    Ra : (n,) MJ/m²/day
    """
    phi = np.deg2rad(lat_deg)
    Gsc = 0.0820  # MJ/m²/min, solar constant
    dr = 1.0 + 0.033 * np.cos(2 * np.pi * doy / 365.0)
    delta = 0.409 * np.sin(2 * np.pi * doy / 365.0 - 1.39)
    cos_arg = np.clip(-np.tan(phi) * np.tan(delta), -1.0, 1.0)
    omega_s = np.arccos(cos_arg)
    Ra = (24 * 60 / np.pi) * Gsc * dr * (
        omega_s * np.sin(phi) * np.sin(delta)
        + np.cos(phi) * np.cos(delta) * np.sin(omega_s)
    )
    return Ra  # MJ/m²/day


def hargreaves_eto(
    Tmean_C: np.ndarray,
    Tmax_C: np.ndarray,
    Tmin_C: np.ndarray,
    lat_deg: float,
    start_doy: int = 1,
) -> np.ndarray:
    """Hargreaves & Samani (1985) reference evapotranspiration.

    ET₀ = 0.0023 × (T_mean + 17.8) × (T_max − T_min)^0.5 × Ra / λ

    where Ra is converted to mm/day equivalent (Ra_MJ / 2.45).

    Parameters
    ----------
    Tmean_C, Tmax_C, Tmin_C : (n,) np.ndarray, °C
    lat_deg : float
    start_doy : int

    Returns
    -------
    ET0 : (n,) np.ndarray, mm/day (non-negative)
    """
    n = len(Tmean_C)
    if not (len(Tmax_C) == len(Tmin_C) == n):
        raise ValueError("Tmean/Tmax/Tmin lengths must match")
    doy = ((np.arange(n) + start_doy - 1) % 365) + 1
    Ra = extraterrestrial_radiation_mj(doy, lat_deg)  # MJ/m²/day
    Ra_mm = Ra / 2.45  # mm/day (latent heat 2.45 MJ/kg)
    delta_T = np.clip(Tmax_C - Tmin_C, 0.0, None)
    eto = 0.0023 * (Tmean_C + 17.8) * np.sqrt(delta_T) * Ra_mm
    return np.clip(eto, 0.0, None)


# ══════════════════════════════════════════════════════════════════════
# 토양 수분 매개변수 — soil_db texture_group → 가용수분 [mm]
# ══════════════════════════════════════════════════════════════════════
# FAO-56 표 19 단순화 — 근권 1m 가정, 보유능 (FC − WP)
ROOT_DEPTH_DEFAULT_M = 1.0


def soil_water_capacity_mm(texture_group: str, root_depth_m: float = 1.0) -> float:
    """Plant-available water (PAW) in root zone [mm].

    FAO-56 표 19에서 발췌:
      coarse (sand/loamy sand)  : ~70 mm/m
      medium (loam, silt loam)  : ~140 mm/m
      fine   (clay loam, clay)  : ~190 mm/m
    """
    paw_per_m = {
        "coarse": 70.0,
        "medium": 140.0,
        "fine":   190.0,
    }.get(texture_group, 130.0)
    return paw_per_m * root_depth_m


# ══════════════════════════════════════════════════════════════════════
# 결과 dataclass
# ══════════════════════════════════════════════════════════════════════
@dataclass
class FAO56Result:
    """FAO-56 daily soil-water-balance recharge result."""
    # 핵심
    recharge_ratio_pct: float          # Σ deep_perc / Σ P × 100
    R_annual_mm: float                 # mm/yr
    P_annual_mm: float                 # mm/yr
    ETa_annual_mm: float               # actual ET annual mm/yr
    ETo_annual_mm: float               # reference ET annual mm/yr
    runoff_annual_mm: float            # mm/yr

    # 입력 메타
    n_days: int
    lat_deg: float
    land_use: str
    texture_group: Optional[str] = None
    AW_max_mm: float = 0.0             # 가용수분 최대치 (mm)

    # 시계열 (선택)
    daily_eto_mm: List[float] = field(default_factory=list)
    daily_etc_mm: List[float] = field(default_factory=list)
    daily_eta_mm: List[float] = field(default_factory=list)
    daily_runoff_mm: List[float] = field(default_factory=list)
    daily_deep_perc_mm: List[float] = field(default_factory=list)
    daily_aw_mm: List[float] = field(default_factory=list)
    daily_kc: List[float] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════
# 핵심 알고리즘 — 일별 SWB
# ══════════════════════════════════════════════════════════════════════
def estimate_recharge_fao56(
    P_daily_mm: np.ndarray,
    Tmean_C: np.ndarray,
    Tmax_C: np.ndarray,
    Tmin_C: np.ndarray,
    lat_deg: float,
    texture_group: str = "medium",
    land_use: str = "혼합농경지",
    root_depth_m: float = 1.0,
    runoff_fraction: float = 0.0,
    aw_init_frac: float = 0.5,
    stress_threshold_frac: float = 0.5,
    start_doy: int = 1,
    return_daily: bool = True,
) -> FAO56Result:
    """Run FAO-56 daily soil-water balance.

    Parameters
    ----------
    P_daily_mm : (n,) precipitation [mm/day]
    Tmean_C, Tmax_C, Tmin_C : (n,) daily temperature stats [°C]
    lat_deg : float, site latitude (decimal degrees, +N)
    texture_group : "coarse"/"medium"/"fine" — sets root-zone PAW
    land_use : key into KC_PRESETS
    root_depth_m : effective root depth (default 1.0 m)
    runoff_fraction : fraction of daily P that becomes surface runoff before
        infiltrating.  Default 0 (all P enters root zone — minimal runoff
        method).  Can be coupled with SCS-CN externally.
    aw_init_frac : initial soil water as fraction of AW_max (default 0.5)
    stress_threshold_frac : AW level below which ETa < ETc (FAO-56 §43,
        default 0.5 = 50% of TAW depleted before stress).
    start_doy : DOY of index 0
    return_daily : populate daily_* arrays

    Returns
    -------
    FAO56Result
    """
    P = np.asarray(P_daily_mm, dtype=float)
    Tm = np.asarray(Tmean_C, dtype=float)
    Tx = np.asarray(Tmax_C, dtype=float)
    Tn = np.asarray(Tmin_C, dtype=float)
    n = len(P)

    if not (len(Tm) == len(Tx) == len(Tn) == n):
        raise ValueError("P/Tmean/Tmax/Tmin lengths must match")
    if np.any(P < 0):
        raise ValueError("Precipitation has negative values")
    if not (-90 <= lat_deg <= 90):
        raise ValueError(f"lat_deg out of range: {lat_deg}")
    if not (0.0 <= runoff_fraction < 1.0):
        raise ValueError("runoff_fraction must be in [0, 1)")

    # ETo, Kc, ETc
    eto = hargreaves_eto(Tm, Tx, Tn, lat_deg, start_doy=start_doy)
    Kc = kc_curve(land_use, n_days=n, start_doy=start_doy)
    etc = eto * Kc

    AW_max = soil_water_capacity_mm(texture_group, root_depth_m)
    AW = aw_init_frac * AW_max
    stress_thr = stress_threshold_frac * AW_max

    eta_arr = np.zeros(n)
    deep_perc_arr = np.zeros(n)
    runoff_arr = np.zeros(n)
    aw_arr = np.zeros(n)

    for t in range(n):
        # 표면 유출 (단순 비례)
        Q = runoff_fraction * P[t]
        infil = P[t] - Q
        runoff_arr[t] = Q

        # 가용수분 갱신 (강수 추가)
        AW = AW + infil

        # 실제 ET — 토양수분 stress 적용 (FAO-56 §43)
        # AW > stress_thr: ETa = ETc (수분 충분)
        # AW < stress_thr: ETa = ETc * (AW / stress_thr)  선형 감쇠
        if AW >= stress_thr:
            ETa = etc[t]
        else:
            ETa = etc[t] * max(AW, 0.0) / max(stress_thr, 1e-6)
        ETa = min(ETa, max(AW, 0.0))  # AW 부족 시 ETa는 AW 한도
        AW = AW - ETa
        eta_arr[t] = ETa

        # Deep percolation
        if AW > AW_max:
            DP = AW - AW_max
            AW = AW_max
        else:
            DP = 0.0
        deep_perc_arr[t] = DP
        aw_arr[t] = AW

    sum_P = float(np.sum(P))
    sum_DP = float(np.sum(deep_perc_arr))
    sum_ETa = float(np.sum(eta_arr))
    sum_ETo = float(np.sum(eto))
    sum_Q = float(np.sum(runoff_arr))
    n_years = max(n / 365.25, 1.0)

    rech_pct = (sum_DP / sum_P * 100.0) if sum_P > 0 else 0.0

    return FAO56Result(
        recharge_ratio_pct=rech_pct,
        R_annual_mm=sum_DP / n_years,
        P_annual_mm=sum_P / n_years,
        ETa_annual_mm=sum_ETa / n_years,
        ETo_annual_mm=sum_ETo / n_years,
        runoff_annual_mm=sum_Q / n_years,
        n_days=n,
        lat_deg=lat_deg,
        land_use=land_use,
        texture_group=texture_group,
        AW_max_mm=AW_max,
        daily_eto_mm=eto.tolist() if return_daily else [],
        daily_etc_mm=etc.tolist() if return_daily else [],
        daily_eta_mm=eta_arr.tolist() if return_daily else [],
        daily_runoff_mm=runoff_arr.tolist() if return_daily else [],
        daily_deep_perc_mm=deep_perc_arr.tolist() if return_daily else [],
        daily_aw_mm=aw_arr.tolist() if return_daily else [],
        daily_kc=Kc.tolist() if return_daily else [],
    )


# ══════════════════════════════════════════════════════════════════════
# 자체 시연 (mock 한국 기온 + 강수)
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=== fao56_swb.py self-test (한국 몬순 + 합성 기온) ===\n")

    rng = np.random.default_rng(42)
    n = 730
    doy = np.arange(n) % 365

    # 강수 (한국 몬순)
    wet_prob = np.clip(0.18 + 0.20 * np.sin(2*np.pi*(doy-80)/365), 0.05, 0.55)
    is_wet = rng.random(n) < wet_prob
    intensity_scale = 8.0 + 25.0 * np.clip(
        np.sin(2*np.pi*(doy-80)/365), 0.0, 1.0
    )
    P = np.where(is_wet, rng.exponential(intensity_scale), 0.0)
    P = np.clip(P, 0, 200)

    # 기온 (대전 평균: 연 12.5°C, 일진폭 8°C, 계절진폭 ±15°C)
    Tmean = 12.5 + 15.0 * np.sin(2*np.pi*(doy-110)/365) + rng.normal(0, 2, n)
    Tmax = Tmean + 4.0 + rng.normal(0, 1, n)
    Tmin = Tmean - 4.0 + rng.normal(0, 1, n)

    result = estimate_recharge_fao56(
        P_daily_mm=P,
        Tmean_C=Tmean, Tmax_C=Tmax, Tmin_C=Tmin,
        lat_deg=36.37,                    # 대전 ASOS
        texture_group="medium",
        land_use="혼합농경지",
    )

    print(f"--- Annual water budget ---")
    print(f"  P:           {result.P_annual_mm:7.1f} mm/yr")
    print(f"  ET₀ (Hargreaves): {result.ETo_annual_mm:7.1f} mm/yr")
    print(f"  ETa (actual):     {result.ETa_annual_mm:7.1f} mm/yr")
    print(f"  Runoff:           {result.runoff_annual_mm:7.1f} mm/yr")
    print(f"  Deep perc (R):    {result.R_annual_mm:7.1f} mm/yr")
    print(f"  Recharge ratio:   {result.recharge_ratio_pct:.2f}% of P")
    print(f"  Closure check:    P − ETa − Q − R = "
          f"{result.P_annual_mm - result.ETa_annual_mm - result.runoff_annual_mm - result.R_annual_mm:+.1f} mm/yr "
          f"(저류 변화)")
    print(f"\n--- Soil ---")
    print(f"  AW_max:           {result.AW_max_mm:.1f} mm")
