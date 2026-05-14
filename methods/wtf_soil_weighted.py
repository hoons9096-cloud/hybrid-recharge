"""
wtf_soil_weighted.py -- Soil-weighted WTF method (proposed method).

Uses the CORRECT Sy for each well's soil type from soil_db.
For grid cells without wells, maps recharge by soil type:
  - Same soil as a well -> uses that well's recharge rate
  - No well with that soil -> scales from nearest-Sy well using Sy ratio

This is the proposed method that preserves spatial variation through
soil-type awareness while keeping implementation simple and practical.
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
def _estimate_recession_k(ho, P, min_dry_days=5):
    """Estimate recession constant k from dry periods."""
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


def _wtf_single_well(ho, P, Sy, tau):
    """Compute annual recharge at one well using WTF method.

    Parameters
    ----------
    ho : (n_days,) water level [m]
    P  : (n_days,) precipitation [m/day]
    Sy : specific yield [-]
    tau : drainage time constant [days]

    Returns
    -------
    annual_recharge : float, [mm/yr]
    """
    n = len(ho)
    if n < 10:
        return 0.0

    k = _estimate_recession_k(ho, P)
    h_eq = float(np.percentile(ho, 10))

    rain_threshold = 0.001
    event_recharges = []

    i = 0
    while i < n:
        if P[i] <= rain_threshold:
            i += 1
            continue

        event_start = i
        h_pre = ho[max(0, event_start - 1)]

        search_end = min(n, event_start + tau)
        if search_end <= event_start:
            i += 1
            continue

        window = ho[event_start:search_end]
        peak_idx_local = int(np.argmax(window))
        h_peak = window[peak_idx_local]
        peak_day = event_start + peak_idx_local

        n_days_elapsed = peak_day - event_start + 1
        h_projected = h_pre
        for _ in range(n_days_elapsed):
            h_projected = h_projected * (1 + k) - k * h_eq

        dh = h_peak - h_projected
        if dh > 0:
            event_recharges.append(Sy * dh)

        i = peak_day + 1

    total_rech_m = sum(event_recharges)
    n_years = max(n / 365.25, 1.0)
    return total_rech_m / n_years * 1000.0


# ──────────────────────────────────────────────────────────
# 메인 인터페이스
# ──────────────────────────────────────────────────────────
def estimate_recharge(domain, observations):
    """Estimate recharge using the Soil-weighted WTF method.

    Uses per-well Sy from soil_db, then maps recharge to all grid cells
    by soil type. Grid cells with the same soil type as a well get that
    well's recharge rate. Cells with unobserved soil types are scaled
    using Sy ratios.

    Parameters
    ----------
    domain : SyntheticDomain
    observations : dict
        'P', 'ET', 'ho_obs', 'well_soil_types'

    Returns
    -------
    recharge_map : np.ndarray, shape (ny, nx) [mm/yr]
    """
    P = observations['P']
    ho_obs = observations['ho_obs']
    well_soil_types = observations['well_soil_types']
    n_wells = ho_obs.shape[0]
    ny, nx = domain.soil_map.shape

    # 각 관정에서 해당 토양의 Sy, tau를 사용하여 WTF 계산
    well_recharges = {}  # {soil_index: [recharge values]}

    for w in range(n_wells):
        soil_idx = int(well_soil_types[w])
        rec = SOIL_DB[soil_idx]
        Sy = rec.sy_lit
        tau = rec.tau

        rech_w = _wtf_single_well(ho_obs[w], P, Sy, tau)

        if soil_idx not in well_recharges:
            well_recharges[soil_idx] = []
        well_recharges[soil_idx].append(rech_w)

    # 토양 유형별 평균 함양률
    soil_mean_rech = {}
    for si, vals in well_recharges.items():
        soil_mean_rech[si] = float(np.mean(vals))

    # 관정이 없는 토양 유형의 함양률 추정: Sy 비율로 스케일링
    observed_soils = list(soil_mean_rech.keys())
    if observed_soils:
        total_wells = sum(len(well_recharges[si]) for si in observed_soils)
        weighted_r_per_sy = sum(
            soil_mean_rech[si] / SOIL_DB[si].sy_lit * len(well_recharges[si])
            for si in observed_soils
        ) / total_wells
    else:
        weighted_r_per_sy = 100.0

    # 격자 셀별 함양률 할당
    recharge_map = np.zeros((ny, nx))

    unique_soils = np.unique(domain.soil_map)
    for si in unique_soils:
        si = int(si)
        mask = domain.soil_map == si

        if si in soil_mean_rech:
            recharge_map[mask] = soil_mean_rech[si]
        else:
            Sy_cell = SOIL_DB[si].sy_lit
            recharge_map[mask] = weighted_r_per_sy * Sy_cell

    return recharge_map


if __name__ == "__main__":
    from synthetic.generate_domain import generate_domain, DomainConfig
    from synthetic.generate_data import generate_data

    print("=== Soil-weighted WTF Method Test (S3) ===\n")

    domain = generate_domain(DomainConfig.S3())
    data = generate_data(domain)

    observations = {
        'P': data.P,
        'ET': data.ET,
        'ho_obs': data.ho_obs,
        'well_soil_types': data.well_soil_types,
    }

    rech_map = estimate_recharge(domain, observations)

    print(f"Recharge map shape: {rech_map.shape}")
    print(f"Mean recharge: {rech_map.mean():.1f} mm/yr")
    print(f"Range: [{rech_map.min():.1f}, {rech_map.max():.1f}] mm/yr")
    print(f"Std: {rech_map.std():.1f} mm/yr")
