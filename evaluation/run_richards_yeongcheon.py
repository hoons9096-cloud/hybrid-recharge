"""run_richards_yeongcheon.py — HYDRUS-class Richards 1D verification at
Yeongcheon climate.

Same governing equations as HYDRUS-1D:
  · Mixed-form Richards equation (Celia et al. 1990)
  · van Genuchten-Mualem constitutive model

Climate (matches run_cascade_uzf_yeongcheon.py):
  P_annual   = 956 mm/yr  (Korean monsoon pattern)
  ETo_annual = 943 mm/yr

For each USDA soil (5 textures, n=12 = Loam = Yeongcheon dominant), runs
Richards 1D over a 3-year simulation (1st year = spin-up) and reports
annual recharge alongside the cascade truth and UZF kinematic-wave
estimate already published in cascade_vs_uzf_yc.csv.

Output:
  evaluation/richards_yeongcheon.{csv,json,log}
"""
from __future__ import annotations

import json
import os
import sys
from typing import List

import numpy as np
import pandas as pd

# Project root on path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "synthetic"))

from richards_1d import solve_richards_1d_v2 as solve_richards_1d
from uzf_kinematic import soil_params_from_sn

# Re-use the exact same climate generator as the UZF run for apples-to-apples
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_cascade_uzf_yeongcheon import make_yeongcheon_climate


# ---------------------------------------------------------------------------
# Soils to evaluate (matches cascade/UZF Yeongcheon runs)
# ---------------------------------------------------------------------------
SOIL_NUMS = [2, 3, 4, 6, 12]   # Loamy Sand, Sandy Loam, Silt Loam, Clay, Loam
N_YEARS   = 3
SPIN_UP   = 1
DOMAIN_L  = 3.0     # m
NZ        = 60
N_SUBDT   = 48      # sub-time steps per day

LOG_LINES: List[str] = []


def log(msg: str = "") -> None:
    print(msg)
    LOG_LINES.append(msg)


def richards_soil_dict(sn: int) -> dict:
    """Convert UZF soil dict (n_vg key) → Richards dict (n key)."""
    sp = soil_params_from_sn(sn)
    return dict(
        theta_s=sp["theta_s"], theta_r=sp["theta_r"],
        alpha=sp["alpha"], n=sp["n_vg"], Ks=sp["Ks"],
    )


def main() -> None:
    P_d, ETo_d = make_yeongcheon_climate(N_YEARS)
    P_m   = P_d   / 1000.0   # m/day
    ETo_m = ETo_d / 1000.0
    idx_start = SPIN_UP * 365

    P_annual_mm   = P_d.mean()   * 365
    ETo_annual_mm = ETo_d.mean() * 365

    log(f"HYDRUS-class Richards 1D verification @ Yeongcheon climate")
    log(f"  P   = {P_annual_mm:.0f} mm/yr,  ETo = {ETo_annual_mm:.0f} mm/yr")
    log(f"  Domain L = {DOMAIN_L:.1f} m, Nz = {NZ}, sub-dt = 1/{N_SUBDT} day")
    log(f"  Total {N_YEARS} yr  (spin-up = {SPIN_UP} yr)")
    log()

    # Reference values from existing cascade_vs_uzf_yc.csv for context
    uzf_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "cascade_vs_uzf_yc.csv")
    if os.path.exists(uzf_csv):
        df_ref = pd.read_csv(uzf_csv).set_index("sn")
    else:
        df_ref = None

    rows: List[dict] = []
    for sn in SOIL_NUMS:
        from soil_db import SOIL_DB
        s = SOIL_DB[sn]
        soil = richards_soil_dict(sn)

        log(f"=== sn={sn} ({s.name}) ===")
        log(f"  vG: θs={soil['theta_s']:.3f}, θr={soil['theta_r']:.3f}, "
            f"α={soil['alpha']:.2f} 1/m, n={soil['n']:.2f}, "
            f"Ks={soil['Ks']:.3f} m/day")

        try:
            res = solve_richards_1d(
                P_daily_m=P_m, ETp_daily_m=ETo_m,
                L=DOMAIN_L, Nz=NZ, soil=soil,
                h_init_m=-1.0, n_subdt=N_SUBDT,
                picard_max=50, picard_tol=1e-5,
                root_depth_m=1.0, root_decay=1.5,
                verbose=True,
            )
            rich_d = res.flux_bottom[idx_start:] * 1000.0   # mm/day
            rich_mm = rich_d.sum() / (N_YEARS - SPIN_UP)
            mb_err = float(np.sum(np.abs(res.mass_balance_err[idx_start:])) * 1000.0)
            ru_mm = float(np.sum(res.runoff[idx_start:]) * 1000.0 / (N_YEARS - SPIN_UP))
            et_mm = float(np.sum(res.ET_actual[idx_start:]) * 1000.0 / (N_YEARS - SPIN_UP))
            # Convergence flag: MB > 50 mm/yr OR rech > P → solver failed
            converged = (mb_err < 50.0 * (N_YEARS - SPIN_UP)) and (rich_mm < P_annual_mm * 1.1)
            if not converged:
                log(f"  ⚠ SOLVER DIVERGED — MB|err={mb_err:.0f}mm, rech={rich_mm:.0f}mm/yr")
                log(f"     (steep vG curve + high Ks → Picard instability; documented as limitation)")
                rich_mm = float("nan"); rich_pct_local = float("nan")
            log(f"  Richards runoff : {ru_mm:7.1f} mm/yr")
            log(f"  Richards ETa    : {et_mm:7.1f} mm/yr")
        except Exception as e:
            log(f"  ⚠ solver failed: {e}")
            rich_mm = float("nan"); mb_err = float("nan")

        rich_pct = rich_mm / P_annual_mm * 100 if np.isfinite(rich_mm) else float("nan")
        log(f"  Richards: {rich_mm:7.1f} mm/yr  ({rich_pct:5.2f}% of P)")

        row = dict(
            sn=sn, soil=s.name,
            richards_mm=rich_mm,
            richards_pct=rich_pct,
            richards_mb_err_mm=mb_err,
        )

        if df_ref is not None and sn in df_ref.index:
            r = df_ref.loc[sn]
            row.update(
                cascade_mm=float(r["cascade_mm"]),
                cascade_pct=float(r["cascade_pct"]),
                uzf_mm=float(r["uzf_mm"]),
                uzf_pct=float(r["uzf_pct"]),
            )
            log(f"  cascade : {row['cascade_mm']:7.1f} mm/yr  "
                f"({row['cascade_pct']:5.2f}% of P)")
            log(f"  UZF     : {row['uzf_mm']:7.1f} mm/yr  "
                f"({row['uzf_pct']:5.2f}% of P)")
            if np.isfinite(rich_mm) and r["uzf_mm"] > 0.01:
                ratio_cu = row["cascade_mm"] / max(rich_mm, 0.01)
                log(f"  cascade / Richards = {ratio_cu:.2f}×")

        log(f"  |MB| err (analysis): {mb_err:.3f} mm cumulative")
        log()
        rows.append(row)

    df = pd.DataFrame(rows)
    log("=== SUMMARY ===")
    cols = ["sn", "soil", "cascade_mm", "uzf_mm", "richards_mm",
            "cascade_pct", "uzf_pct", "richards_pct"]
    show = [c for c in cols if c in df.columns]
    log(df[show].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # Loam-focused summary (Yeongcheon dominant texture)
    loam = df[df.sn == 12].iloc[0] if (df.sn == 12).any() else None
    if loam is not None:
        log()
        log("--- Loam (Yeongcheon dominant texture, HSG-D alluvium) ---")
        if "cascade_pct" in df.columns:
            log(f"  Cascade truth     : {loam['cascade_pct']:5.2f}% of P")
            log(f"  UZF kinematic     : {loam['uzf_pct']:5.2f}% of P")
        log(f"  Richards 1D       : {loam['richards_pct']:5.2f}% of P  ◀ HYDRUS-class")
        log(f"  WTF field (YC-012):  8.60% of P")
        log(f"  α=0.3 corrected   : 10.30–10.36% of P (two watersheds)")
        log(f"  α=1.0 corrected   : 13.58–13.88% of P (two watersheds)")
        log(f"  FAO-56 SWB        : 16.0% of P")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    df.to_csv(os.path.join(out_dir, "richards_yeongcheon.csv"), index=False)
    with open(os.path.join(out_dir, "richards_yeongcheon.json"), "w") as f:
        json.dump(rows, f, indent=2)
    with open(os.path.join(out_dir, "richards_yeongcheon.log"), "w") as f:
        f.write("\n".join(LOG_LINES))
    log()
    log(f"✓ saved → richards_yeongcheon.{{csv,json,log}}")


if __name__ == "__main__":
    main()
