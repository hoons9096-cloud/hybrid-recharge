"""uzf_kinematic.py — 1D kinematic-wave unsaturated flow solver.

Niswonger et al. (2006) MODFLOW UZF1 Package 스타일 kinematic-wave
근사. 중력우세 (gravity-dominant) 가정 하에서 ∂θ/∂t = -∂K/∂z.

van Genuchten-Mualem 토양 함수 사용:
    Se(h) = (1 + (α|h|)^n)^(-m),  m = 1 - 1/n
    K(θ) = Ks · Se^0.5 · (1 - (1-Se^(1/m))^m)^2

수치 모형:
    - Cell-centered finite volume, Z grid (dz uniform)
    - Top BC: prescribed flux (P-ET); excess → runoff
    - Bottom BC: free drainage q_bot = K(θ_bot)
    - Sub-daily explicit time stepping with CFL adaptive dt
    - ET partitioning by root_frac (exponential decay)

장점 vs 풀 Richards:
    - 수치적으로 매우 안정 (explicit + adaptive dt)
    - 비포화 영역에서 dominant 물리 캡처
    - HYDRUS-1D / MODFLOW UZF 와 같은 클래스의 근사
    - Tridiagonal 풀이 불필요

용도:
    cascade truth (synthetic/vadose_cascade.py) 의 외부검증 reference 로 사용.

References
----------
Niswonger, R.G., Prudic, D.E., & Regan, R.S. (2006).
   Documentation of the Unsaturated-Zone Flow (UZF1) Package for
   modeling unsaturated flow between the land surface and the water
   table with MODFLOW-2005.  USGS Techniques and Methods 6-A19.
van Genuchten, M.Th. (1980). A closed-form equation for predicting
   the hydraulic conductivity of unsaturated soils.  SSSAJ 44, 892-898.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# van Genuchten constitutive relations (θ-based form for stability)
# ---------------------------------------------------------------------------
def theta_to_Se(theta: np.ndarray, theta_s: float, theta_r: float) -> np.ndarray:
    """Effective saturation Se ∈ [0, 1]."""
    return np.clip((theta - theta_r) / max(theta_s - theta_r, 1e-9), 1e-6, 1.0 - 1e-6)


def K_unsat(theta: np.ndarray, theta_s: float, theta_r: float,
            n_vg: float, Ks: float) -> np.ndarray:
    """Mualem-van Genuchten K(θ) [m/day]."""
    m = 1.0 - 1.0 / n_vg
    Se = theta_to_Se(theta, theta_s, theta_r)
    # Mualem: K_r = Se^L · [1 - (1 - Se^(1/m))^m]^2  (L=0.5)
    inner = 1.0 - np.power(np.clip(1.0 - np.power(Se, 1.0 / m), 1e-9, 1.0), m)
    return Ks * np.sqrt(Se) * inner ** 2


def Se_to_theta(Se: np.ndarray, theta_s: float, theta_r: float) -> np.ndarray:
    return theta_r + Se * (theta_s - theta_r)


def theta_at_h(h_m: float, theta_s: float, theta_r: float,
               alpha: float, n_vg: float) -> float:
    """Single value θ(h) for setting initial conditions."""
    if h_m >= 0:
        return theta_s
    m = 1.0 - 1.0 / n_vg
    Se = (1.0 + (alpha * abs(h_m)) ** n_vg) ** (-m)
    return theta_r + Se * (theta_s - theta_r)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class UZFResult:
    z: np.ndarray                 # cell centers [m]
    theta: np.ndarray             # (Nt+1, Nz)
    flux_bottom: np.ndarray       # (Nt,) daily percolation [m/day]
    flux_top: np.ndarray          # (Nt,) daily applied infiltration [m/day]
    runoff: np.ndarray            # (Nt,) daily ponding overflow [m/day]
    ET_actual: np.ndarray         # (Nt,) daily actual ET [m/day]
    storage: np.ndarray           # (Nt+1,) total root-zone storage [m]
    mass_balance_err: np.ndarray  # (Nt,) cumulative MB error [m]

    @property
    def annual_recharge_mm(self) -> float:
        n_yr = max(len(self.flux_bottom) / 365.25, 1.0)
        return float(np.sum(self.flux_bottom) * 1000.0 / n_yr)


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------
def solve_uzf_kinematic(
    P_daily_m: np.ndarray,
    ET_daily_m: np.ndarray,
    L: float = 3.0,
    Nz: int = 30,
    soil: Optional[dict] = None,
    init_theta_frac: float = 0.6,   # init θ between θ_wp and θ_fc
    n_subdt_per_day: int = 24,      # base sub-step (adaptive may refine)
    cfl_safety: float = 0.4,        # CFL = (K * dt) / dz < cfl_safety
    root_decay: float = 1.5,
    return_profile: bool = False,
) -> UZFResult:
    """1D kinematic-wave UZF (Niswonger 2006 style).

    Parameters
    ----------
    P_daily_m, ET_daily_m : (Nt,) precipitation/ET [m/day]
    L : domain depth [m]
    Nz : number of cells
    soil : dict with theta_s, theta_r, alpha (1/m), n (van Genuchten), Ks (m/day)
    init_theta_frac : initial θ as fraction between θ_wp and θ_fc
    n_subdt_per_day : base time-stepping (adaptive CFL further refines)
    cfl_safety : Courant safety factor
    root_decay : exponential root-density decay (higher = more shallow)
    return_profile : if True, store full θ(z, t) profile

    Returns
    -------
    UZFResult
    """
    if soil is None:
        soil = dict(theta_s=0.43, theta_r=0.078, alpha=3.6, n_vg=1.56, Ks=0.25)
    theta_s = soil["theta_s"]
    theta_r = soil["theta_r"]
    alpha = soil["alpha"]
    n_vg = soil.get("n_vg", soil.get("n", 1.56))
    Ks = soil["Ks"]

    P = np.asarray(P_daily_m, dtype=float)
    ET = np.asarray(ET_daily_m, dtype=float)
    Nt = len(P)
    dz = L / Nz
    z = (np.arange(Nz) + 0.5) * dz

    # Field capacity and wilting point
    theta_fc = theta_at_h(-3.3, theta_s, theta_r, alpha, n_vg)
    theta_wp = theta_at_h(-150.0, theta_s, theta_r, alpha, n_vg)
    if theta_fc <= theta_wp + 0.01:
        theta_fc = max(theta_wp + 0.05, 0.20)

    # Initial θ — between θ_wp and θ_fc
    theta_init = theta_wp + init_theta_frac * (theta_fc - theta_wp)
    theta = np.full(Nz, theta_init)

    # Root density (exponential decay with depth)
    root_weights = np.exp(-root_decay * z / max(z[-1], 0.1))
    root_weights /= root_weights.sum()

    # Output buffers
    flux_bot = np.zeros(Nt)
    flux_top = np.zeros(Nt)
    runoff_d = np.zeros(Nt)
    ET_act = np.zeros(Nt)
    storage_d = np.zeros(Nt + 1)
    mb_err_d = np.zeros(Nt)
    storage_d[0] = float(np.sum(theta) * dz)

    if return_profile:
        theta_profile = np.zeros((Nt + 1, Nz))
        theta_profile[0] = theta.copy()

    base_dt = 1.0 / n_subdt_per_day
    cum_storage = storage_d[0]

    for day in range(Nt):
        P_d = P[day]
        ET_d = ET[day]

        # Adaptive sub-stepping based on CFL
        # K_max over current θ → adaptive dt
        elapsed = 0.0
        in_flux_day = 0.0
        out_flux_day = 0.0
        runoff_day = 0.0
        ET_day = 0.0
        sub_step = base_dt

        while elapsed < 1.0 - 1e-9:
            # CFL-bounded dt
            K_arr = K_unsat(theta, theta_s, theta_r, n_vg, Ks)
            K_max = float(K_arr.max() + 1e-12)
            dt_cfl = cfl_safety * dz / K_max
            dt = min(base_dt, dt_cfl, 1.0 - elapsed)

            # 1) Top: applied flux this sub-step = (P - ET_pot) × dt (m)
            #    but split between infiltration and ET partitioning
            # We use: P enters cell 0 fully (capped by saturation deficit),
            #   ET removed by root density × stress factor
            p_in = P_d * dt   # m of water trying to infiltrate
            ET_demand_layers = ET_d * dt * root_weights   # m of ET demand per cell

            # 2) Cell 0 saturation check
            theta_max = theta_s - 1e-6
            cap_0 = (theta_max - theta[0]) * dz   # max water cell 0 can take [m]
            if p_in <= cap_0:
                theta[0] += p_in / dz
                actual_in_sub = p_in
                runoff_sub = 0.0
            else:
                theta[0] = theta_max
                runoff_sub = p_in - cap_0
                actual_in_sub = cap_0
            in_flux_day += actual_in_sub   # accumulate infiltrated volume [m]
            runoff_day += runoff_sub

            # 3) ET removal — water-stress modulated
            ET_actual_sub = 0.0
            for i in range(Nz):
                # FAO-56 stress: full ET when θ > θ_fc·0.5 + θ_wp·0.5,
                # linearly reduced to 0 at θ_wp
                stress_thresh = 0.5 * (theta_fc + theta_wp)
                if theta[i] >= stress_thresh:
                    ks_stress = 1.0
                elif theta[i] <= theta_wp:
                    ks_stress = 0.0
                else:
                    ks_stress = (theta[i] - theta_wp) / max(stress_thresh - theta_wp, 1e-9)
                ET_take = ET_demand_layers[i] * ks_stress
                ET_take = min(ET_take, max(0, (theta[i] - theta_wp) * dz))
                theta[i] -= ET_take / dz
                ET_actual_sub += ET_take
            ET_day += ET_actual_sub

            # 4) Gravity drainage between cells (kinematic wave: q = K(θ))
            #    Compute K at each cell, drain top-down
            K_arr = K_unsat(theta, theta_s, theta_r, n_vg, Ks)
            # face flux (downward): q_face = K_upper (kinematic upstream weighting)
            # cell i loses K(θ_i)·dt water; cell i+1 gains K(θ_i)·dt
            # Bottom cell loses to drainage
            theta_new = theta.copy()
            # Process top-down
            for i in range(Nz):
                q_out = K_arr[i] * dt   # downward drainage [m]
                # Limit by available water above field capacity (drainage only when wet)
                # Actually kinematic wave allows drainage even at θ < θ_fc but slowly.
                # Just limit by available water
                avail = max(0.0, (theta_new[i] - theta_r) * dz)
                q_out = min(q_out, avail)
                theta_new[i] -= q_out / dz
                if i < Nz - 1:
                    cap_next = max(0, (theta_max - theta_new[i + 1]) * dz)
                    if q_out > cap_next:
                        # Backup: water can't fit, push back to current cell
                        # (saturated zone moving up)
                        theta_new[i] += (q_out - cap_next) / dz
                        q_out = cap_next
                    theta_new[i + 1] += q_out / dz
                else:
                    # Bottom cell — q_out leaves system as recharge
                    out_flux_day += q_out
            theta = theta_new

            elapsed += dt

        # Daily accumulators
        flux_top[day] = in_flux_day      # m/day equivalent
        flux_bot[day] = out_flux_day
        runoff_d[day] = runoff_day
        ET_act[day] = ET_day

        new_storage = float(np.sum(theta) * dz)
        storage_d[day + 1] = new_storage
        # Mass balance: P = infiltration + runoff (surface partition)
        #   infiltration = drainage_bottom + ETa + ΔStorage
        # → MB error = P_in - runoff - drainage - ETa - ΔStorage
        P_in = P[day]
        mb = P_in - runoff_day - out_flux_day - ET_day - (new_storage - cum_storage)
        mb_err_d[day] = mb
        cum_storage = new_storage

        if return_profile:
            theta_profile[day + 1] = theta.copy()

    return UZFResult(
        z=z,
        theta=theta_profile if return_profile else np.array([]),
        flux_bottom=flux_bot,
        flux_top=flux_top,
        runoff=runoff_d,
        ET_actual=ET_act,
        storage=storage_d,
        mass_balance_err=mb_err_d,
    )


# ---------------------------------------------------------------------------
# soil_db.py 인덱스 → UZF 입력 dict
# ---------------------------------------------------------------------------
def soil_params_from_sn(sn_idx: int) -> dict:
    """soil_db.py SOIL_DB[sn_idx] → UZF 입력.

    Ks: Carsel-Parrish 1988 Table 3 median (cm/h → m/day).
    """
    from soil_db import SOIL_DB
    s = SOIL_DB[sn_idx]
    KS_USDA_M_PER_DAY = {
        1: 7.13, 2: 3.50, 3: 1.06, 4: 0.27, 5: 0.16,
        6: 0.05, 7: 0.07, 8: 0.13, 9: 0.07, 10: 0.06,
        11: 0.18, 12: 0.25,
    }
    return dict(
        theta_s=s.theta_s, theta_r=s.theta_r,
        alpha=s.alpha_vg, n_vg=s.n_vg,
        Ks=KS_USDA_M_PER_DAY.get(sn_idx, 0.25),
    )
