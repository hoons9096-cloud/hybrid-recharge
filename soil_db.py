"""
soil_db.py — Single source of truth for soil hydraulic properties.

All soil parameters originate from the Carsel & Parrish (1988) compilation
of van Genuchten parameters for 12 USDA textural classes.

References
----------
Carsel, R.F. & Parrish, R.S. (1988). Developing joint probability
    distributions of soil water retention characteristics.
    Water Resources Research, 24(5), 755-769.

Healy, R.W. & Cook, P.G. (2002). Using groundwater levels to estimate
    recharge. Hydrogeology Journal, 10(1), 91-109.
    (Recharge range and Sy literature values)

Usage
-----
Every module that needs soil constants MUST import from this file:

    from soil_db import SOIL_DB, get_soil, get_bounds
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────
# Soil dataclass
# ──────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SoilRecord:
    """Immutable record for one USDA textural class."""

    index: int           # 1-based soil number
    name: str            # e.g. "Sand", "Loamy Sand"

    # van Genuchten parameters (Carsel & Parrish 1988) — MEAN values
    theta_s: float       # saturated water content  [-]
    theta_r: float       # residual water content   [-]
    alpha_vg: float      # VG alpha parameter       [1/m]
    n_vg: float          # VG n parameter           [-]

    # Standard deviations from Carsel & Parrish (1988) Table 6
    # These capture within-class variability of VG parameters.
    theta_s_sd: float    # std dev of θs  [-]
    theta_r_sd: float    # std dev of θr  [-]
    alpha_vg_sd: float   # std dev of α   [1/m]
    n_vg_sd: float       # std dev of n   [-]

    # Derived / literature properties
    #
    # tau: Characteristic gravity-drainage time constant [days].
    #   Physically represents the e-folding time for vadose-zone drainage
    #   after a recharge event:  recovery = 1 - exp(-t / τ).
    #
    #   Derivation:  τ ≈ (θs - θr) × L / Ksat
    #     where L = representative drainage path length (≈1 m),
    #     Ksat = median saturated hydraulic conductivity for the
    #     USDA texture class (Carsel & Parrish, 1988, Table 3).
    #
    #   This is an empirical approximation to the solution of Richards'
    #   equation for gravity-dominated vertical drainage.  More rigorous
    #   approaches (e.g., numerical Richards' equation with VG parameters)
    #   would yield depth-dependent τ, but the constant approximation is
    #   standard in WTF applications.
    #
    #   Values were cross-checked against field lysimeter drainage curves:
    #     Sand ≈ 2 days    (Sophocleous, 1991)
    #     Loam ≈ 7 days    (Kendy et al., 2004)
    #     Clay ≈ 35 days   (Scanlon et al., 2002)
    #
    #   References:
    #     Carsel, R.F. & Parrish, R.S. (1988). WRR, 24(5), 755-769.
    #     Sophocleous, M.A. (1991). J. Hydrol., 124, 1-29.
    #     Kendy, E. et al. (2004). Hydrol. Process., 18(12), 2367-2383.
    #     Scanlon, B.R. et al. (2002). Vadose Zone J., 1(1), 2-6.
    tau: int             # characteristic drainage time scale [days]
    alpha_recharge: float  # max recharge fraction of rainfall [-]
    sy_lit: float        # literature specific yield [-]

    # Optimiser parameter bounds
    k_bounds: Tuple[float, float]  # (k_min, k_max) for decay constant

    # Expected recharge ratio range (% of total rainfall)
    # Based on Healy & Cook (2002) and regional calibration studies
    rech_range: Tuple[float, float]  # (min_%, max_%)

    # Soil texture classification helpers
    texture_group: str   # "coarse", "medium", "fine"
    response_speed: str  # "very_fast", "fast", "moderate", "slow"

    @property
    def m_vg(self) -> float:
        """van Genuchten m = 1 - 1/n."""
        return 1.0 - 1.0 / self.n_vg

    @property
    def is_clay_like(self) -> bool:
        """Clay-dominated textures requiring extra pump contamination scrutiny."""
        return self.index in {6, 7, 8, 9, 10, 11}


# ──────────────────────────────────────────────────────────
# Master database — THE canonical source
# ──────────────────────────────────────────────────────────
_SOIL_RECORDS: list[SoilRecord] = [
    # Standard deviations sourced from Carsel & Parrish (1988), Table 6.
    # Where exact SD not available, conservative estimates based on
    # published coefficient of variation for the textural class.
    #
    # k upper bound (closest to zero) is uniformly set to -0.001 for
    # all 12 soil types.  Healy & Cook (2002, Hydrogeol. J.) report
    # WTF recession constants k in the range -0.001 to -0.1 across
    # diverse aquifer types.  The recession rate reflects not just soil
    # permeability but the entire aquifer system (thickness, storage,
    # boundary conditions), so soil texture alone cannot constrain it.
    # A universal -0.001 floor prevents degenerate zero-recession
    # while allowing the optimiser to find very slow drainage where
    # the data support it.
    #
    # 1  Sand
    SoilRecord(
        index=1, name="Sand",
        theta_s=0.43, theta_r=0.045, alpha_vg=14.50, n_vg=2.68,
        theta_s_sd=0.06, theta_r_sd=0.010, alpha_vg_sd=2.50, n_vg_sd=0.29,
        tau=2, alpha_recharge=0.50, sy_lit=0.33,
        k_bounds=(-0.50, -0.001),
        rech_range=(10.0, 38.0),
        texture_group="coarse", response_speed="very_fast",
    ),
    # 2  Loamy Sand
    SoilRecord(
        index=2, name="Loamy Sand",
        theta_s=0.41, theta_r=0.057, alpha_vg=12.40, n_vg=2.28,
        theta_s_sd=0.09, theta_r_sd=0.011, alpha_vg_sd=4.30, n_vg_sd=0.27,
        tau=3, alpha_recharge=0.40, sy_lit=0.28,
        k_bounds=(-0.50, -0.001),
        rech_range=(8.0, 32.0),
        texture_group="coarse", response_speed="very_fast",
    ),
    # 3  Sandy Loam
    SoilRecord(
        index=3, name="Sandy Loam",
        theta_s=0.41, theta_r=0.065, alpha_vg=7.50, n_vg=1.89,
        theta_s_sd=0.09, theta_r_sd=0.015, alpha_vg_sd=3.70, n_vg_sd=0.15,
        tau=5, alpha_recharge=0.32, sy_lit=0.20,
        k_bounds=(-0.30, -0.001),
        rech_range=(5.0, 25.0),
        texture_group="coarse", response_speed="fast",
    ),
    # 4  Silt Loam
    SoilRecord(
        index=4, name="Silt Loam",
        theta_s=0.45, theta_r=0.067, alpha_vg=2.00, n_vg=1.41,
        theta_s_sd=0.08, theta_r_sd=0.015, alpha_vg_sd=2.70, n_vg_sd=0.12,
        tau=12, alpha_recharge=0.22, sy_lit=0.14,
        k_bounds=(-0.08, -0.001),
        rech_range=(3.0, 18.0),
        texture_group="medium", response_speed="fast",
    ),
    # 5  Silt
    SoilRecord(
        index=5, name="Silt",
        theta_s=0.46, theta_r=0.034, alpha_vg=1.60, n_vg=1.37,
        theta_s_sd=0.11, theta_r_sd=0.010, alpha_vg_sd=1.20, n_vg_sd=0.05,
        tau=15, alpha_recharge=0.20, sy_lit=0.12,
        k_bounds=(-0.08, -0.001),
        rech_range=(2.0, 15.0),
        texture_group="medium", response_speed="fast",
    ),
    # 6  Clay
    SoilRecord(
        index=6, name="Clay",
        theta_s=0.38, theta_r=0.068, alpha_vg=0.80, n_vg=1.09,
        theta_s_sd=0.09, theta_r_sd=0.020, alpha_vg_sd=1.20, n_vg_sd=0.03,
        tau=35, alpha_recharge=0.06, sy_lit=0.05,
        k_bounds=(-0.03, -0.001),
        rech_range=(1.0, 8.0),
        texture_group="fine", response_speed="slow",
    ),
    # 7  Silty Clay
    SoilRecord(
        index=7, name="Silty Clay",
        theta_s=0.36, theta_r=0.070, alpha_vg=0.50, n_vg=1.09,
        theta_s_sd=0.07, theta_r_sd=0.015, alpha_vg_sd=0.27, n_vg_sd=0.03,
        tau=40, alpha_recharge=0.06, sy_lit=0.05,
        k_bounds=(-0.03, -0.001),
        rech_range=(1.0, 8.0),
        texture_group="fine", response_speed="slow",
    ),
    # 8  Sandy Clay
    SoilRecord(
        index=8, name="Sandy Clay",
        theta_s=0.38, theta_r=0.100, alpha_vg=2.70, n_vg=1.23,
        theta_s_sd=0.05, theta_r_sd=0.020, alpha_vg_sd=1.70, n_vg_sd=0.06,
        tau=22, alpha_recharge=0.08, sy_lit=0.06,
        k_bounds=(-0.05, -0.001),
        rech_range=(2.0, 10.0),
        texture_group="fine", response_speed="moderate",
    ),
    # 9  Silty Clay Loam
    SoilRecord(
        index=9, name="Silty Clay Loam",
        theta_s=0.43, theta_r=0.089, alpha_vg=1.00, n_vg=1.23,
        theta_s_sd=0.07, theta_r_sd=0.019, alpha_vg_sd=1.50, n_vg_sd=0.06,
        tau=28, alpha_recharge=0.10, sy_lit=0.07,
        k_bounds=(-0.03, -0.001),
        rech_range=(2.0, 12.0),
        texture_group="fine", response_speed="moderate",
    ),
    # 10 Clay Loam
    SoilRecord(
        index=10, name="Clay Loam",
        theta_s=0.41, theta_r=0.095, alpha_vg=1.90, n_vg=1.31,
        theta_s_sd=0.09, theta_r_sd=0.018, alpha_vg_sd=1.40, n_vg_sd=0.09,
        tau=22, alpha_recharge=0.12, sy_lit=0.08,
        k_bounds=(-0.05, -0.001),
        rech_range=(2.0, 12.0),
        texture_group="fine", response_speed="moderate",
    ),
    # 11 Sandy Clay Loam
    SoilRecord(
        index=11, name="Sandy Clay Loam",
        theta_s=0.39, theta_r=0.100, alpha_vg=5.90, n_vg=1.48,
        theta_s_sd=0.07, theta_r_sd=0.020, alpha_vg_sd=3.50, n_vg_sd=0.12,
        tau=10, alpha_recharge=0.14, sy_lit=0.10,
        k_bounds=(-0.30, -0.001),
        rech_range=(4.0, 18.0),
        texture_group="medium", response_speed="moderate",
    ),
    # 12 Loam
    SoilRecord(
        index=12, name="Loam",
        theta_s=0.43, theta_r=0.078, alpha_vg=3.60, n_vg=1.56,
        theta_s_sd=0.10, theta_r_sd=0.015, alpha_vg_sd=2.00, n_vg_sd=0.11,
        tau=7, alpha_recharge=0.18, sy_lit=0.13,
        k_bounds=(-0.15, -0.001),
        rech_range=(4.0, 22.0),
        texture_group="medium", response_speed="fast",
    ),
]


# ──────────────────────────────────────────────────────────
# Public lookup interface
# ──────────────────────────────────────────────────────────
SOIL_DB: Dict[int, SoilRecord] = {s.index: s for s in _SOIL_RECORDS}

# Convenience arrays for backward compatibility with existing code
SOIL_NAMES = [s.name for s in _SOIL_RECORDS]
SOIL_NAMES_NUMBERED = [f"{s.index}. {s.name}" for s in _SOIL_RECORDS]
CLAY_LIKE_SET = frozenset(s.index for s in _SOIL_RECORDS if s.is_clay_like)

# NumPy arrays — drop-in replacements for the old scattered arrays
VG_DB = np.array([[s.theta_s, s.theta_r, s.alpha_vg, s.n_vg] for s in _SOIL_RECORDS])
TAU_DB = np.array([s.tau for s in _SOIL_RECORDS])
ALPHA_SOIL_LIST = np.array([s.alpha_recharge for s in _SOIL_RECORDS])
SY_LIT_LIST = np.array([s.sy_lit for s in _SOIL_RECORDS])
RECH_RANGE = np.array([list(s.rech_range) for s in _SOIL_RECORDS])
K_BOUNDS = {s.index: s.k_bounds for s in _SOIL_RECORDS}
SY_DB = SY_LIT_LIST   # alias used by pump_preprocess
ALPHA_DB = ALPHA_SOIL_LIST  # alias used by pump_preprocess
SOIL_RECH_RANGE = {s.index: s.rech_range for s in _SOIL_RECORDS}


# Standard deviation arrays for Monte Carlo / BMA parametric uncertainty
VG_SD = np.array([[s.theta_s_sd, s.theta_r_sd, s.alpha_vg_sd, s.n_vg_sd]
                   for s in _SOIL_RECORDS])


def sample_vg_params(
    soil_num: int,
    n_samples: int = 100,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample VG parameters from Carsel & Parrish (1988) distributions.

    Draws from truncated normal distributions to respect physical
    constraints (θs > θr > 0, α > 0, n > 1).

    Parameters
    ----------
    soil_num : int
        1-based soil index.
    n_samples : int
        Number of Monte Carlo samples.
    rng : numpy Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    np.ndarray, shape (n_samples, 4)
        Columns: [θs, θr, α, n] for each sample.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    sn = max(1, min(12, round(soil_num))) - 1
    mean = VG_DB[sn]
    sd = VG_SD[sn]

    samples = np.zeros((n_samples, 4))
    for i in range(4):
        raw = rng.normal(mean[i], sd[i], size=n_samples)
        # Physical constraints
        if i == 0:    # θs: (0.05, 0.80)
            raw = np.clip(raw, 0.05, 0.80)
        elif i == 1:  # θr: (0.001, θs - 0.01)
            raw = np.clip(raw, 0.001, mean[0] - 0.01)
        elif i == 2:  # α: (0.01, ∞)
            raw = np.clip(raw, 0.01, None)
        elif i == 3:  # n: (1.01, ∞)
            raw = np.clip(raw, 1.01, None)
        samples[:, i] = raw

    # Ensure θs > θr for all samples
    bad = samples[:, 0] <= samples[:, 1]
    samples[bad, 1] = samples[bad, 0] - 0.01

    return samples


def get_soil(soil_num: int) -> SoilRecord:
    """Retrieve a soil record by 1-based index, clamped to [1, 12]."""
    sn = max(1, min(12, round(soil_num)))
    return SOIL_DB[sn]


def get_bounds(soil_num: int) -> Tuple[float, float]:
    """Return (k_min, k_max) for a soil class."""
    return get_soil(soil_num).k_bounds


def gap_allow_for_soil(soil_num: int) -> int:
    """Event gap tolerance (days) based on texture group."""
    group = get_soil(soil_num).texture_group
    return {"coarse": 1, "medium": 2, "fine": 3}[group]


def peak_window_for_soil(soil_num: int) -> int:
    """Post-event peak search window (days) based on response speed.

    Returns the *base* window.  For event-adaptive windows, use
    ``adaptive_peak_window()``.
    """
    speed = get_soil(soil_num).response_speed
    return {"very_fast": 4, "fast": 7, "moderate": 10, "slow": 14}[speed]


def adaptive_peak_window(soil_num: int, event_rain_mm: float,
                          mean_rain_mm: float = 10.0) -> int:
    """Event-adaptive post-event peak search window.

    The standard WTF method uses a fixed peak window, but physically the
    water-table response time depends on both soil type and recharge
    impulse magnitude.  Intense events saturate the vadose zone faster,
    producing an earlier peak; weak events may take longer to percolate.

    Algorithm
    ---------
    1. Start from the soil-based base window ``w_base``.
    2. Compute an intensity ratio ``r = event_rain / mean_rain``.
    3. Scale: ``w = w_base × clamp(1/√r, 0.5, 1.5)``
       - Heavy rain (r > 1) → shorter window (faster response)
       - Light rain (r < 1) → longer window (slower percolation)
       - Scaling bounded to [50%, 150%] of base to prevent extremes.
    4. Clamp final result to [2, 21] days.

    This approach is consistent with Green-Ampt infiltration theory
    where wetting front velocity scales with rainfall intensity.

    References
    ----------
    Green, W.H. & Ampt, G.A. (1911). Studies on soil physics.
        J. Agric. Sci., 4(1), 1-24.
    Healy, R.W. & Cook, P.G. (2002). Using groundwater levels to estimate
        recharge. Hydrogeology Journal, 10(1), 91-109.

    Parameters
    ----------
    soil_num : int
        1-based USDA texture class index.
    event_rain_mm : float
        Total rainfall for this event (mm).
    mean_rain_mm : float
        Long-term mean event rainfall (mm).  Default 10.0 mm.

    Returns
    -------
    int
        Adaptive peak search window in days.
    """
    import numpy as np

    w_base = peak_window_for_soil(soil_num)
    if mean_rain_mm <= 0 or event_rain_mm <= 0:
        return w_base

    ratio = event_rain_mm / mean_rain_mm
    # Inverse square-root scaling: heavier rain → shorter window
    scale = 1.0 / np.sqrt(max(ratio, 0.01))
    scale = max(0.5, min(scale, 1.5))
    w_adaptive = int(round(w_base * scale))
    return max(2, min(w_adaptive, 21))
