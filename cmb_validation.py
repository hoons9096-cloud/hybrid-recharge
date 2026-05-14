"""
cmb_validation.py — Chloride Mass Balance (CMB) independent recharge validation.

Provides an independent, hydrochemistry-based recharge estimate to
cross-validate hybrid-recharge results.  The CMB method assumes chloride
is a conservative tracer with no subsurface sources or sinks:

    R_cmb = (Cl_p × P) / Cl_gw          [simple CMB]
    R_cmb = ((Cl_p + D) × P) / Cl_gw    [with dry deposition]

where
    R_cmb : recharge (mm/yr)
    Cl_p  : chloride concentration in precipitation (mg/L)
    D     : dry deposition rate (mg/m²/yr), converted to equivalent
            concentration by dividing by P
    P     : annual precipitation (mm/yr)
    Cl_gw : chloride concentration in groundwater (mg/L)

The comparison with WTF-derived recharge provides method-independent
validation — WTF is physics-based (water level fluctuation) while CMB
is geochemistry-based (mass conservation of a conservative tracer).

Assumptions & Limitations
-------------------------
- Cl⁻ is conservative: no dissolution of halite, no anthropogenic
  inputs (road salt, fertiliser, septic effluent).
- Steady-state: long-term average Cl_p and P are representative.
- Well-mixed: Cl_gw represents the recharge-weighted mean, not
  a single snapshot.
- Piston flow or well-mixed aquifer (no preferential flow bypassing
  the chloride signal).

If any assumption is violated, the CMB estimate should be flagged
and interpreted with caution.  The module reports assumption-check
diagnostics alongside the recharge estimate.

References
----------
Allison, G.B. & Hughes, M.W. (1978). The use of environmental chloride
    and tritium to estimate total recharge to an unconfined aquifer.
    Australian Journal of Soil Research, 16(2), 181-195.

Scanlon, B.R., Healy, R.W. & Cook, P.G. (2002). Choosing appropriate
    techniques for quantifying groundwater recharge.
    Hydrogeology Journal, 10(1), 18-39.

Eriksson, E. & Khunakasem, V. (1969). Chloride concentration in
    groundwater, recharge rate and rate of deposition of chloride
    in the Israel Coastal Plain. Journal of Hydrology, 7(2), 178-197.

Wood, W.W. & Sanford, W.E. (1995). Chemical and isotopic methods for
    quantifying ground-water recharge in a regional, semiarid
    environment. Ground Water, 33(3), 458-468.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Sequence

import numpy as np


# ──────────────────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────────────────
@dataclass
class CMBResult:
    """Result of Chloride Mass Balance recharge estimation."""

    # ── Input echo ──
    cl_precip_mg_l: float       # Cl⁻ in precipitation (mg/L)
    cl_gw_mg_l: float           # Cl⁻ in groundwater (mg/L)
    precip_mm_yr: float         # Annual precipitation (mm/yr)
    dry_deposition_mg_m2_yr: float  # Dry Cl⁻ deposition (mg/m²/yr)

    # ── CMB recharge ──
    recharge_mm_yr: float       # R_cmb (mm/yr)
    recharge_ratio_pct: float   # R_cmb / P × 100 (%)

    # ── Comparison with WTF ──
    wtf_recharge_mm_yr: Optional[float] = None
    wtf_recharge_ratio_pct: Optional[float] = None
    ratio_cmb_to_wtf: Optional[float] = None  # R_cmb / R_wtf
    relative_error_pct: Optional[float] = None  # (CMB - WTF) / WTF × 100

    # ── Diagnostics ──
    cl_ratio: float = 0.0       # Cl_gw / Cl_p (enrichment factor)
    assumption_warnings: list = None

    def __post_init__(self):
        if self.assumption_warnings is None:
            self.assumption_warnings = []

    def to_dict(self):
        return asdict(self)

    def summary(self) -> str:
        """Human-readable summary string."""
        lines = [
            "═══ Chloride Mass Balance (CMB) Validation ═══",
            f"  Cl_precip     = {self.cl_precip_mg_l:.2f} mg/L",
            f"  Cl_gw         = {self.cl_gw_mg_l:.2f} mg/L",
            f"  Precipitation = {self.precip_mm_yr:.1f} mm/yr",
            f"  Dry deposition= {self.dry_deposition_mg_m2_yr:.1f} mg/m²/yr",
            f"  Cl enrichment = {self.cl_ratio:.1f}×",
            "",
            f"  CMB recharge  = {self.recharge_mm_yr:.1f} mm/yr "
            f"({self.recharge_ratio_pct:.1f}% of P)",
        ]
        if self.wtf_recharge_mm_yr is not None:
            lines += [
                "",
                "── Comparison with hybrid-recharge ──",
                f"  WTF recharge  = {self.wtf_recharge_mm_yr:.1f} mm/yr "
                f"({self.wtf_recharge_ratio_pct:.1f}% of P)",
                f"  CMB / WTF     = {self.ratio_cmb_to_wtf:.2f}",
                f"  Relative diff = {self.relative_error_pct:+.1f}%",
            ]
            # Interpretation
            abs_err = abs(self.relative_error_pct)
            if abs_err < 20:
                lines.append("  → Good agreement (< 20% difference)")
            elif abs_err < 50:
                lines.append("  → Moderate agreement (20-50% difference)")
            else:
                lines.append("  → Poor agreement (> 50% difference)")
                lines.append("    Check CMB assumptions or WTF calibration.")

        if self.assumption_warnings:
            lines += ["", "⚠ Assumption warnings:"]
            for w in self.assumption_warnings:
                lines.append(f"  - {w}")

        return "\n".join(lines)


# ──────────────────────────────────────────────────────────
# Core CMB computation
# ──────────────────────────────────────────────────────────
def cmb_recharge(
    cl_precip_mg_l: float,
    cl_gw_mg_l: float,
    precip_mm_yr: float,
    dry_deposition_mg_m2_yr: float = 0.0,
    wtf_recharge_mm_yr: Optional[float] = None,
    wtf_precip_mm_yr: Optional[float] = None,
) -> CMBResult:
    """Compute CMB recharge and compare with WTF estimate.

    Parameters
    ----------
    cl_precip_mg_l : float
        Mean chloride concentration in precipitation (mg/L).
        Typical range: 0.5 - 5.0 mg/L (inland), 5 - 20 mg/L (coastal).
    cl_gw_mg_l : float
        Mean chloride concentration in groundwater (mg/L).
        Must be > cl_precip_mg_l for CMB to be valid.
    precip_mm_yr : float
        Mean annual precipitation (mm/yr).
    dry_deposition_mg_m2_yr : float, optional
        Atmospheric dry deposition of Cl⁻ (mg/m²/yr). Default 0.
        Include if available; typical values 100-500 mg/m²/yr
        in semi-arid regions.
    wtf_recharge_mm_yr : float, optional
        hybrid-recharge recharge estimate (mm/yr) for comparison.
    wtf_precip_mm_yr : float, optional
        Total precipitation used in WTF analysis (mm/yr).
        If None, uses precip_mm_yr for ratio calculation.

    Returns
    -------
    CMBResult
        Recharge estimate with diagnostics and WTF comparison.

    Examples
    --------
    >>> r = cmb_recharge(cl_precip_mg_l=1.5, cl_gw_mg_l=15.0,
    ...                  precip_mm_yr=1200, wtf_recharge_mm_yr=130)
    >>> print(f"CMB recharge: {r.recharge_mm_yr:.0f} mm/yr")
    CMB recharge: 120 mm/yr
    """
    warnings = []

    # ── Input validation ──
    if cl_gw_mg_l <= 0:
        raise ValueError(f"cl_gw_mg_l must be > 0, got {cl_gw_mg_l}")
    if cl_precip_mg_l < 0:
        raise ValueError(f"cl_precip_mg_l must be >= 0, got {cl_precip_mg_l}")
    if precip_mm_yr <= 0:
        raise ValueError(f"precip_mm_yr must be > 0, got {precip_mm_yr}")

    # ── Assumption checks ──
    cl_ratio = cl_gw_mg_l / max(cl_precip_mg_l, 1e-6)

    if cl_gw_mg_l <= cl_precip_mg_l:
        warnings.append(
            f"Cl_gw ({cl_gw_mg_l:.1f}) ≤ Cl_p ({cl_precip_mg_l:.1f}): "
            "CMB requires evaporative enrichment (Cl_gw > Cl_p). "
            "Possible anthropogenic dilution or sampling error."
        )

    if cl_gw_mg_l > 250:
        warnings.append(
            f"Cl_gw = {cl_gw_mg_l:.0f} mg/L is very high. "
            "Possible halite dissolution, seawater intrusion, or "
            "anthropogenic contamination. CMB may overestimate recharge."
        )

    if cl_ratio < 2:
        warnings.append(
            f"Low enrichment factor ({cl_ratio:.1f}×). "
            "CMB estimate is highly sensitive to Cl_p measurement error "
            "when enrichment is small."
        )

    if cl_ratio > 100:
        warnings.append(
            f"Very high enrichment factor ({cl_ratio:.0f}×). "
            "Implies very low recharge fraction. Verify Cl_gw is not "
            "affected by subsurface Cl sources."
        )

    # ── CMB calculation ──
    # Convert dry deposition to equivalent concentration contribution:
    #   D (mg/m²/yr) / P (mm/yr) = D / P (mg/L)
    #   since 1 mm/yr over 1 m² = 1 L/yr
    cl_total = cl_precip_mg_l + dry_deposition_mg_m2_yr / max(precip_mm_yr, 1e-9)

    r_cmb = cl_total * precip_mm_yr / cl_gw_mg_l  # mm/yr
    r_ratio = r_cmb / precip_mm_yr * 100.0         # %

    # ── Physical plausibility check ──
    if r_ratio > 80:
        warnings.append(
            f"CMB recharge ratio = {r_ratio:.0f}% of precipitation. "
            "Physically implausible (> 80%). Check input values."
        )
    if r_ratio < 0.1:
        warnings.append(
            f"CMB recharge ratio = {r_ratio:.2f}% of precipitation. "
            "Extremely low — verify Cl_gw is not contaminated."
        )

    # ── WTF comparison ──
    wtf_ratio = None
    ratio_cmb_wtf = None
    rel_err = None

    if wtf_recharge_mm_yr is not None:
        wtf_p = wtf_precip_mm_yr if wtf_precip_mm_yr else precip_mm_yr
        wtf_ratio = wtf_recharge_mm_yr / max(wtf_p, 1e-9) * 100.0

        if abs(wtf_recharge_mm_yr) > 1e-9:
            ratio_cmb_wtf = r_cmb / wtf_recharge_mm_yr
            rel_err = (r_cmb - wtf_recharge_mm_yr) / wtf_recharge_mm_yr * 100.0

    return CMBResult(
        cl_precip_mg_l=cl_precip_mg_l,
        cl_gw_mg_l=cl_gw_mg_l,
        precip_mm_yr=precip_mm_yr,
        dry_deposition_mg_m2_yr=dry_deposition_mg_m2_yr,
        recharge_mm_yr=r_cmb,
        recharge_ratio_pct=r_ratio,
        wtf_recharge_mm_yr=wtf_recharge_mm_yr,
        wtf_recharge_ratio_pct=wtf_ratio,
        ratio_cmb_to_wtf=ratio_cmb_wtf,
        relative_error_pct=rel_err,
        cl_ratio=cl_ratio,
        assumption_warnings=warnings,
    )


# ──────────────────────────────────────────────────────────
# Time-series CMB (seasonal Cl variation)
# ──────────────────────────────────────────────────────────
def cmb_timeseries(
    cl_precip_series: Sequence[float],
    cl_gw_series: Sequence[float],
    precip_series: Sequence[float],
    dry_deposition_mg_m2_yr: float = 0.0,
) -> dict:
    """Compute CMB recharge for each time step in a series.

    Useful when Cl_gw varies seasonally (e.g., quarterly sampling).

    Parameters
    ----------
    cl_precip_series : sequence of float
        Cl⁻ in precipitation per period (mg/L).
    cl_gw_series : sequence of float
        Cl⁻ in groundwater per period (mg/L).
    precip_series : sequence of float
        Precipitation per period (mm).
    dry_deposition_mg_m2_yr : float
        Annual dry deposition, distributed proportionally.

    Returns
    -------
    dict with keys:
        recharge_series : np.ndarray (mm per period)
        recharge_total_mm : float
        precip_total_mm : float
        recharge_ratio_pct : float
        mean_cl_ratio : float
    """
    cl_p = np.asarray(cl_precip_series, dtype=float)
    cl_gw = np.asarray(cl_gw_series, dtype=float)
    precip = np.asarray(precip_series, dtype=float)

    n = len(cl_p)
    if len(cl_gw) != n or len(precip) != n:
        raise ValueError("All input series must have the same length.")

    # Distribute annual dry deposition proportionally to precipitation
    p_total = np.sum(precip)
    if p_total > 0 and dry_deposition_mg_m2_yr > 0:
        # Convert annual mg/m² to mg/L equivalent per period
        frac = precip / max(p_total, 1e-9)
        dry_per_period = dry_deposition_mg_m2_yr * frac / np.maximum(precip, 1e-9)
    else:
        dry_per_period = np.zeros(n)

    cl_total = cl_p + dry_per_period

    # CMB per period
    valid = cl_gw > 0
    recharge = np.zeros(n)
    recharge[valid] = cl_total[valid] * precip[valid] / cl_gw[valid]

    r_total = float(np.sum(recharge))
    p_total_f = float(np.sum(precip))

    # Mean enrichment factor
    valid_ratio = (cl_gw > 0) & (cl_p > 0)
    mean_ratio = float(np.mean(cl_gw[valid_ratio] / cl_p[valid_ratio])) if np.any(valid_ratio) else np.nan

    return {
        "recharge_series": recharge,
        "recharge_total_mm": r_total,
        "precip_total_mm": p_total_f,
        "recharge_ratio_pct": r_total / max(p_total_f, 1e-9) * 100.0,
        "mean_cl_ratio": mean_ratio,
    }


# ──────────────────────────────────────────────────────────
# Multi-well comparison
# ──────────────────────────────────────────────────────────
def cmb_multi_well(
    cl_precip_mg_l: float,
    cl_gw_values: Sequence[float],
    precip_mm_yr: float,
    wtf_recharge_values: Optional[Sequence[float]] = None,
    well_names: Optional[Sequence[str]] = None,
    dry_deposition_mg_m2_yr: float = 0.0,
) -> list[CMBResult]:
    """Run CMB for multiple wells and return comparison list.

    Parameters
    ----------
    cl_precip_mg_l : float
        Regional Cl⁻ in precipitation (mg/L).
    cl_gw_values : sequence of float
        Cl⁻ in groundwater for each well (mg/L).
    precip_mm_yr : float
        Regional mean annual precipitation (mm/yr).
    wtf_recharge_values : sequence of float, optional
        WTF recharge estimates per well (mm/yr).
    well_names : sequence of str, optional
        Well identifiers for reporting.
    dry_deposition_mg_m2_yr : float
        Dry deposition (mg/m²/yr).

    Returns
    -------
    list of CMBResult
    """
    n = len(cl_gw_values)
    wtf_vals = wtf_recharge_values if wtf_recharge_values is not None else [None] * n

    results = []
    for i in range(n):
        r = cmb_recharge(
            cl_precip_mg_l=cl_precip_mg_l,
            cl_gw_mg_l=cl_gw_values[i],
            precip_mm_yr=precip_mm_yr,
            dry_deposition_mg_m2_yr=dry_deposition_mg_m2_yr,
            wtf_recharge_mm_yr=wtf_vals[i],
        )
        results.append(r)

    return results
