"""run_cascade_uzf_yeongcheon.py

Cascade truth vs UZF kinematic-wave external validation using the
**actual Yeongcheon climate** (P = 956 mm/yr, ETo = 943 mm/yr)
rather than the high-P synthetic forcing used in the original comparison.

This script:
1. Generates synthetic daily P/ETo with Korean monsoon seasonality.
2. Runs cascade (vadose_cascade.py) and UZF (uzf_kinematic.py) for
   5 USDA soil types over a 3-year simulation (1st year = spin-up).
3. Reports annual recharge, bias, and daily r for years 2-3.
4. Saves results to cascade_vs_uzf_yc.{csv,json,log}.

Reference climate (from evaluation/eto_yeongcheon.json):
    P_annual = 956.2 mm/yr
    ETo_annual = 943.2 mm/yr
    ETa_annual = 620.0 mm/yr  (FAO-56 SWB actual ET)
    FAO-56 recharge ≈ 16% of P
"""
from __future__ import annotations

import sys
import os
import json
import numpy as np
import pandas as pd

# Ensure project root is on path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "synthetic"))

from vadose_cascade import build_layers_from_sn, simulate_cascade as run_cascade
from uzf_kinematic import solve_uzf_kinematic, soil_params_from_sn

# ---------------------------------------------------------------------------
# Synthetic Yeongcheon climate generator
# Korean monsoon: ~70% of P in June-September (doy 152-273)
# ETo: peaks in July-August, low in Dec-Feb
# ---------------------------------------------------------------------------
RNG = np.random.default_rng(42)

def make_yeongcheon_climate(n_years: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic daily P and ETo matching Yeongcheon annual statistics.

    Annual totals:
        P   = 956 mm/yr  (Korean monsoon pattern)
        ETo = 943 mm/yr  (summer-peaked)

    Returns
    -------
    P_d, ETo_d : (Nt,) arrays in mm/day
    """
    P_annual = 956.2       # mm/yr
    ETo_annual = 943.2     # mm/yr
    Nt = 365 * n_years
    doy = np.tile(np.arange(1, 366), n_years)[:Nt]

    # Monsoon seasonality: fraction of annual P per day-of-year
    # 70% in doy 152-273 (Jun-Sep), rest spread over other months
    monsoon = (doy >= 152) & (doy <= 273)
    w = np.where(monsoon, 0.70 / 122, 0.30 / 243)
    w = w / w.sum() * Nt / 365  # normalise per year

    # Daily P: stochastic gamma with mean = P_annual * w
    mean_daily = P_annual * w  # mm/day
    # Gamma shape=0.5 → skewed (many dry days, few large events)
    shape = 0.5
    P_raw = RNG.gamma(shape, mean_daily / shape)
    # Scale to exactly match target annual
    for yr in range(n_years):
        sl = slice(yr * 365, (yr + 1) * 365)
        P_raw[sl] *= P_annual / P_raw[sl].sum()
    P_d = np.clip(P_raw, 0.0, None)

    # ETo: sinusoidal seasonal curve
    ETo_daily_mean = ETo_annual / 365
    # Peak in summer (doy 185 ≈ early July), amplitude 60%
    amplitude = 0.60
    ETo_d = ETo_daily_mean * (1.0 + amplitude * np.sin(2 * np.pi * (doy - 60) / 365))
    ETo_d = np.clip(ETo_d, 0.1, None)
    # Scale to match target annual
    for yr in range(n_years):
        sl = slice(yr * 365, (yr + 1) * 365)
        ETo_d[sl] *= ETo_annual / ETo_d[sl].sum()

    return P_d, ETo_d


# ---------------------------------------------------------------------------
# Main comparison loop
# ---------------------------------------------------------------------------
SOIL_NUMS = [2, 3, 4, 6, 12]   # Loamy Sand, Sandy Loam, Silt Loam, Clay, Loam
N_YEARS   = 3
SPIN_UP   = 1    # First year discarded as spin-up

LOG_LINES: list[str] = []

def log(msg: str = "") -> None:
    print(msg)
    LOG_LINES.append(msg)


def main() -> None:
    P_d, ETo_d = make_yeongcheon_climate(N_YEARS)
    # Convert mm → m for UZF (which expects m/day)
    P_m   = P_d / 1000.0
    ETo_m = ETo_d / 1000.0
    # Analysis window (exclude spin-up)
    idx_start = SPIN_UP * 365

    P_annual_mm  = P_d.mean() * 365
    ETo_annual_mm = ETo_d.mean() * 365

    log(f"Yeongcheon climate: P = {P_annual_mm:.0f} mm/yr, ETo = {ETo_annual_mm:.0f} mm/yr  "
        f"(T={N_YEARS} yr, spin-up={SPIN_UP} yr)")
    log()

    rows: list[dict] = []
    for sn in SOIL_NUMS:
        from soil_db import SOIL_DB
        s = SOIL_DB[sn]

        # ---- Cascade ---------------------------------------------------
        layers = build_layers_from_sn(sn, L_total_m=2.0, n_layers=5)
        cr = run_cascade(P_d, ETo_d, layers)
        # Annual recharge from analysis window
        casc_rech_mm = cr.recharge[idx_start:].sum() / (N_YEARS - SPIN_UP)

        # ---- UZF -------------------------------------------------------
        soil_p = soil_params_from_sn(sn)
        uzf = solve_uzf_kinematic(
            P_m, ETo_m,
            L=3.0, Nz=30, soil=soil_p,
            init_theta_frac=0.6,
            n_subdt_per_day=24,
            cfl_safety=0.4,
            root_decay=1.5,
        )
        uzf_rech_mm_d = uzf.flux_bottom[idx_start:] * 1000.0  # m→mm/day
        uzf_rech_mm   = uzf_rech_mm_d.sum() / (N_YEARS - SPIN_UP)
        uzf_runoff_mm = uzf.runoff[idx_start:].sum() * 1000.0 / (N_YEARS - SPIN_UP)
        uzf_eta_mm    = uzf.ET_actual[idx_start:].sum() * 1000.0 / (N_YEARS - SPIN_UP)

        # MB error (should be small after fix)
        uzf_mb_err_cum = float(np.sum(np.abs(uzf.mass_balance_err[idx_start:])) * 1000.0)

        # Daily correlation (on analysis window, exclude zero-recharge days)
        casc_d = cr.recharge[idx_start:]          # mm/day
        uzf_d  = uzf_rech_mm_d                    # mm/day
        r = float(np.corrcoef(casc_d, uzf_d)[0, 1]) if casc_d.std() > 0 and uzf_d.std() > 0 else 0.0

        # Bias: (cascade - UZF) / UZF × 100 %
        if uzf_rech_mm > 0.01:
            bias_pct = (casc_rech_mm - uzf_rech_mm) / uzf_rech_mm * 100.0
        else:
            bias_pct = float('nan') if casc_rech_mm < 0.01 else float('inf')

        rmse_d = float(np.sqrt(np.mean((casc_d - uzf_d) ** 2)))

        log(f"=== sn={sn} ({s.name}) ===")
        log(f"  cascade: {casc_rech_mm:7.1f} mm/yr  ({casc_rech_mm/P_annual_mm*100:5.1f}% of P)")
        log(f"  UZF:     {uzf_rech_mm:7.1f} mm/yr  ({uzf_rech_mm/P_annual_mm*100:5.1f}% of P)")
        log(f"  UZF runoff: {uzf_runoff_mm:5.1f} mm/yr")
        log(f"  UZF ETa:    {uzf_eta_mm:6.1f} mm/yr")
        log(f"  UZF |MB| err (analysis): {uzf_mb_err_cum:.3f} mm cumulative")
        if np.isfinite(bias_pct):
            log(f"  → daily RMSE: {rmse_d:.3f} mm/d, total bias: {bias_pct:+.1f}%, daily r: {r:+.3f}")
        else:
            log(f"  → UZF recharge near zero — bias undefined (UZF routes mostly to runoff/ET)")
        log()

        rows.append(dict(
            sn=sn, soil=s.name,
            cascade_mm=casc_rech_mm, uzf_mm=uzf_rech_mm,
            cascade_pct=casc_rech_mm / P_annual_mm * 100,
            uzf_pct=uzf_rech_mm / P_annual_mm * 100,
            RMSE_mm_d=rmse_d, bias_pct=bias_pct, r=r,
            uzf_runoff_mm=uzf_runoff_mm, uzf_eta_mm=uzf_eta_mm,
            uzf_mb_err_mm=uzf_mb_err_cum,
        ))

    df = pd.DataFrame(rows)
    log("=== SUMMARY ===")
    log(df[["sn", "soil", "cascade_mm", "uzf_mm",
            "cascade_pct", "uzf_pct", "RMSE_mm_d", "bias_pct", "r"]].to_string(index=False))

    finite_bias = df["bias_pct"].replace([float("inf"), float("-inf")], float("nan")).dropna()
    median_bias = float(np.median(np.abs(finite_bias)))
    mean_r = float(df["r"].mean())
    log(f"\nMean |bias| (finite):  {float(np.nanmean(np.abs(finite_bias))):.1f}%")
    log(f"Median |bias| (finite):{median_bias:.1f}%")
    log(f"Mean daily r: {mean_r:.3f}")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    df.to_csv(os.path.join(out_dir, "cascade_vs_uzf_yc.csv"), index=False)
    with open(os.path.join(out_dir, "cascade_vs_uzf_yc.json"), "w") as f:
        json.dump(rows, f, indent=2)
    with open(os.path.join(out_dir, "cascade_vs_uzf_yc.log"), "w") as f:
        f.write("\n".join(LOG_LINES))
    log(f"\n✓ saved → cascade_vs_uzf_yc.{{csv,json,log}}")


if __name__ == "__main__":
    main()
