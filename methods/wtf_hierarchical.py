"""wtf_hierarchical.py — Hierarchical Bayesian WTF method (Phase 3).

emcee MCMC 로 관정별 Sy 의 hierarchical posterior 를 추정한 뒤,
posterior 평균 Sy 를 사용해 토양형별 함양율 격자 맵 산출.

기존 wtf_soil_weighted 와 동일 인터페이스: estimate_recharge(domain, obs).
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from soil_db import SOIL_DB
from methods.wtf_soil_weighted import _wtf_single_well, _estimate_recession_k


# ---------------------------------------------------------------------------
# USDA texture index → HSG 추정 (대략적 매핑, hierarchical prior 용)
# ---------------------------------------------------------------------------
_USDA_TO_HSG = {
    1: "A",   # Sand
    2: "A",   # Loamy Sand
    3: "B",   # Sandy Loam
    4: "C",   # Silt Loam
    5: "C",   # Silt
    6: "D",   # Clay
    7: "D",   # Silty Clay
    8: "C",   # Sandy Clay
    9: "D",   # Silty Clay Loam
    10: "C",  # Clay Loam
    11: "B",  # Sandy Clay Loam
    12: "B",  # Loam
}


def estimate_recharge(
    domain, observations,
    n_walkers: int = 32,
    n_steps: int = 12000,   # 수렴 기본값 (R̂<1.05); bayes_hierarchical 참조
    burn_in: int = 3000,
):
    """Hierarchical Bayesian WTF.

    1. 각 관정에서 WTF 함양율 + Sy 추정 (단순 inversion: Sy_eff = R / Σdh)
    2. emcee 로 hierarchical posterior 적합
    3. posterior 평균 Sy 로 격자 맵 재계산
    """
    from bayes_hierarchical import (
        WellObservation, fit_hierarchical, HAS_EMCEE,
    )

    P = observations["P"]
    ho_obs = observations["ho_obs"]
    well_soil_types = observations["well_soil_types"]
    n_wells = ho_obs.shape[0]
    ny, nx = domain.soil_map.shape

    if not HAS_EMCEE:
        # fallback to soil_weighted
        from methods.wtf_soil_weighted import estimate_recharge as fb
        return fb(domain, observations)

    # ── Step 1: 관정별 WTF 함양 + Sy_eff 추출 ──
    well_obs: List[WellObservation] = []
    well_dh_total: List[float] = []
    for w in range(n_wells):
        soil_idx = int(well_soil_types[w])
        rec = SOIL_DB[soil_idx]
        Sy_lit = rec.sy_lit
        tau = rec.tau

        rech_w_mm = _wtf_single_well(ho_obs[w], P, Sy_lit, tau)   # mm/yr
        # cumulative dh from rises (positive only)
        dh = np.diff(ho_obs[w])
        cum_rise = float(np.nansum(dh[dh > 0]))
        well_dh_total.append(cum_rise)
        n_yr = max(len(P) / 365.25, 1.0)
        P_total = float(np.sum(P)) / n_yr     # m/yr

        hsg = _USDA_TO_HSG.get(soil_idx, "B")
        well_obs.append(WellObservation(
            name=f"w{w}",
            hsg=hsg,
            aquifer="alluvial",
            sy_eff_obs=Sy_lit,           # prior 중심
            cumulative_dh_m=cum_rise,
            P_total_m=P_total,
            soil_area_frac=1.0,
        ))

    # ── Step 2: hierarchical fitting ──
    try:
        result = fit_hierarchical(
            well_obs,
            n_walkers=n_walkers, n_steps=n_steps,
            burn_in=burn_in, seed=0,
        )
    except Exception:
        # fallback
        from methods.wtf_soil_weighted import estimate_recharge as fb
        return fb(domain, observations)

    # ── Step 3: posterior 평균 Sy_well 로 함양 재계산 ──
    posterior_sy = result.sy_well_mean    # (n_wells,)

    # 각 관정의 함양률 (posterior Sy 기반): R = Sy_post × Σdh / n_yr × 1000
    n_yr = max(len(P) / 365.25, 1.0)
    well_rech = np.zeros(n_wells)
    for w in range(n_wells):
        well_rech[w] = posterior_sy[w] * well_dh_total[w] / n_yr * 1000.0  # mm/yr

    # 토양 유형별 평균
    soil_mean_rech: Dict[int, float] = {}
    for w in range(n_wells):
        si = int(well_soil_types[w])
        soil_mean_rech.setdefault(si, []).append(well_rech[w])
    soil_mean_rech = {si: float(np.mean(v)) for si, v in soil_mean_rech.items()}

    # 관측 토양 외 셀 — Sy 비율 스케일링
    observed_soils = list(soil_mean_rech.keys())
    if observed_soils:
        weighted = sum(
            soil_mean_rech[si] / SOIL_DB[si].sy_lit
            for si in observed_soils
        ) / len(observed_soils)
    else:
        weighted = 100.0

    rech_map = np.zeros((ny, nx))
    for si in np.unique(domain.soil_map):
        si = int(si)
        mask = (domain.soil_map == si)
        if si in soil_mean_rech:
            rech_map[mask] = soil_mean_rech[si]
        else:
            rech_map[mask] = weighted * SOIL_DB[si].sy_lit
    return rech_map
