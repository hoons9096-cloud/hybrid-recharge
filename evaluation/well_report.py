"""
well_report.py -- Single-well field-mode integrated report.

Generates an HTML report summarising a single observation well's hybrid-recharge
analysis using whatever session_state outputs are available:
  - Core estimate          (always required)
  - Plausibility check     (always derivable)
  - Time-series diagnostics
  - Bootstrap CI           (optional, if uncertainty.bootstrap_uncertainty was run)
  - Bayesian Model Averaging across soil types (optional)
  - Kalman hyperparameter sensitivity (optional)
  - Pump preprocessing effect (optional)

Companion to evaluation/field_report.py, which targets the *spatial / multi-
method* synthetic benchmark.  This module serves the live single-well analysis
running in app_v30 tabs 1–6.

Usage
-----
    from evaluation.well_report import build_well_html_report
    html = build_well_html_report(
        result_v27=result_v27,
        site_name="김천지좌-1",
        soil_label="Loam (sn=6)",
        uc_result=uc,            # optional
        bma_result=bma,          # optional
        kalman_sens=ksa,         # optional
        pump_result=pr,          # optional
        output_path="report.html",
    )

References
----------
Healy, R.W. (2010). Estimating Groundwater Recharge.
Moriasi, D.N. et al. (2007). Model evaluation guidelines.  Trans. ASABE 50(3).
Efron, B. & Tibshirani, R.J. (1993). An Introduction to the Bootstrap.
"""
from __future__ import annotations

import base64
import io
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from html import escape
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt

# 프로젝트 루트
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ──────────────────────────────────────────────────────────
# 공통 스타일 / 헬퍼
# ──────────────────────────────────────────────────────────
_FONT_TITLE = 13
_FONT_LABEL = 11
_FONT_TICK = 9
_DPI = 150


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _safe(d: Any, key: str, default=None):
    """dict 또는 dataclass에서 안전하게 값 추출."""
    if isinstance(d, dict):
        return d.get(key, default)
    return getattr(d, key, default)


def _np(values) -> np.ndarray:
    """list/array을 numpy로."""
    if values is None:
        return np.array([])
    return np.asarray(values, dtype=float)


# ══════════════════════════════════════════════════════════════════════
# 1. Plausibility check (single value)
# ══════════════════════════════════════════════════════════════════════
@dataclass
class WellPlausibility:
    """Physical plausibility check on a single-well recharge estimate."""
    recharge_ratio_pct: float          # % of P
    R_annual_mm: float                 # mm/yr
    P_annual_mm: float                 # mm/yr
    R_over_P: float                    # 0–1
    sy_eff: float
    opt_k: float
    n_obs_days: int
    flags: List[str] = field(default_factory=list)

    @property
    def pass_basic(self) -> bool:
        return self.recharge_ratio_pct >= 0 and self.R_annual_mm <= self.P_annual_mm


def well_plausibility_check(
    result_v27: Dict,
    P_annual_mm: Optional[float] = None,
    typical_low_frac: float = 0.02,
    typical_high_frac: float = 0.50,
) -> WellPlausibility:
    """Apply physical bounds to a single-well hybrid-recharge result.

    Parameters
    ----------
    result_v27 : dict
        Output of core_sim_v27 (CoreMetrics.to_dict()).
    P_annual_mm : float, optional
        Annual precipitation (mm/yr).  If None, derived from po_shifted.
    typical_low_frac : float
        R/P below this is flagged.
    typical_high_frac : float
        R/P above this is flagged.

    Returns
    -------
    WellPlausibility
    """
    po_shifted = _np(_safe(result_v27, "po_shifted"))
    n_days = int(len(po_shifted))
    if n_days == 0:
        po_shifted = _np(_safe(result_v27, "po", []))
        n_days = int(len(po_shifted))

    if P_annual_mm is None and n_days > 0:
        # po is in m/day
        n_years = max(n_days / 365.25, 1.0)
        P_annual_mm = float(np.nansum(po_shifted)) * 1000.0 / n_years
    elif P_annual_mm is None:
        P_annual_mm = 0.0

    rech_pct = float(_safe(result_v27, "recharge_ratio", 0.0))
    R_annual_mm = P_annual_mm * rech_pct / 100.0
    sy_eff = float(_safe(result_v27, "Sy_eff", 0.0))
    opt_k = float(_safe(result_v27, "opt_k", 0.0))

    flags: List[str] = []
    if rech_pct < 0:
        flags.append(f"Recharge {rech_pct:.1f}% 이 음수 (물리적으로 불가능)")
    if R_annual_mm > P_annual_mm and P_annual_mm > 0:
        flags.append(f"R = {R_annual_mm:.0f} > P = {P_annual_mm:.0f} mm/yr (불가능)")
    if rech_pct > typical_high_frac * 100:
        flags.append(
            f"R/P = {rech_pct:.1f}% > {typical_high_frac*100:.0f}% (이례적으로 높음, "
            f"파라미터/입력 점검 필요)"
        )
    if 0 <= rech_pct < typical_low_frac * 100:
        flags.append(
            f"R/P = {rech_pct:.1f}% < {typical_low_frac*100:.0f}% (이례적으로 낮음)"
        )
    bnd = _safe(result_v27, "boundary_warnings", []) or []
    for w in bnd:
        flags.append(f"경계 경고: {w}")
    if sy_eff < 0.001 or sy_eff > 0.45:
        flags.append(f"Sy_eff = {sy_eff:.3f} 이 일반적 범위(0.001–0.45) 밖")

    return WellPlausibility(
        recharge_ratio_pct=rech_pct,
        R_annual_mm=R_annual_mm,
        P_annual_mm=P_annual_mm,
        R_over_P=R_annual_mm / P_annual_mm if P_annual_mm > 0 else float("nan"),
        sy_eff=sy_eff,
        opt_k=opt_k,
        n_obs_days=n_days,
        flags=flags,
    )


# ══════════════════════════════════════════════════════════════════════
# 2. Time-series figures
# ══════════════════════════════════════════════════════════════════════
def plot_well_time_series(result_v27: Dict):
    """Three-panel time series: water levels, precipitation, recharge events.

    Returns
    -------
    matplotlib.figure.Figure
    """
    ho = _np(_safe(result_v27, "ho"))
    hs_kf = _np(_safe(result_v27, "hs_kf"))
    hs_pure = _np(_safe(result_v27, "hs_pure"))
    po = _np(_safe(result_v27, "po_shifted"))
    rech = _np(_safe(result_v27, "rech"))
    pump_mask = _np(_safe(result_v27, "pump_mask"))

    n = max(len(ho), len(po), len(rech))
    days = np.arange(n)

    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.3, 1.3]})

    # Panel 1: water levels
    ax = axes[0]
    if len(ho) > 0:
        ax.plot(days[:len(ho)], ho, color="#94A3B8", lw=1.2,
                label="Observed h_obs", zorder=2)
    if len(hs_kf) > 0:
        ax.plot(days[:len(hs_kf)], hs_kf, color="#2563EB", lw=1.4,
                label="Kalman h_sim", zorder=3)
    if len(hs_pure) > 0 and not np.array_equal(hs_pure, hs_kf):
        ax.plot(days[:len(hs_pure)], hs_pure, color="#10B981", lw=1.0,
                ls="--", alpha=0.7, label="WTF-only (pure)", zorder=2)
    # 펌핑 구간 음영
    if len(pump_mask) > 0 and pump_mask.max() > 0:
        in_pump = False
        s = 0
        pm = pump_mask.astype(int)
        for i in range(len(pm)):
            if pm[i] and not in_pump:
                in_pump = True
                s = i
            if in_pump and (not pm[i] or i == len(pm) - 1):
                e = i if not pm[i] else i + 1
                ax.axvspan(s, e, color="#EF4444", alpha=0.10, zorder=1)
                in_pump = False
    ax.set_ylabel("Water level (m)", fontsize=_FONT_LABEL)
    ax.set_title("Time-series diagnostics", fontsize=_FONT_TITLE)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    ax.tick_params(axis="both", labelsize=_FONT_TICK)

    # Panel 2: precipitation
    ax = axes[1]
    if len(po) > 0:
        ax.bar(days[:len(po)], po * 1000.0, color="#93C5FD",
               edgecolor="none", width=1.0)
    ax.set_ylabel("P (mm/day)", fontsize=_FONT_LABEL)
    ax.grid(alpha=0.3)
    ax.tick_params(axis="both", labelsize=_FONT_TICK)

    # Panel 3: recharge events
    ax = axes[2]
    if len(rech) > 0:
        ax.bar(days[:len(rech)], rech * 1000.0, color="#10B981",
               edgecolor="none", width=1.0)
    ax.set_ylabel("Rech (mm/day)", fontsize=_FONT_LABEL)
    ax.set_xlabel("Day index", fontsize=_FONT_LABEL)
    ax.grid(alpha=0.3)
    ax.tick_params(axis="both", labelsize=_FONT_TICK)

    fig.tight_layout()
    return fig


def plot_well_recharge_cumulative(result_v27: Dict):
    """Cumulative recharge & cumulative precipitation (proportional check).

    Returns
    -------
    matplotlib.figure.Figure
    """
    rech = _np(_safe(result_v27, "rech"))
    po = _np(_safe(result_v27, "po_shifted"))
    n = max(len(rech), len(po))
    days = np.arange(n)

    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    if len(po) > 0:
        cum_p = np.cumsum(po) * 1000.0
        ax1.plot(days[:len(cum_p)], cum_p, color="#93C5FD",
                 lw=1.6, label="Cumulative P (mm)")
    if len(rech) > 0:
        cum_r = np.cumsum(rech) * 1000.0
        ax1.plot(days[:len(cum_r)], cum_r, color="#10B981",
                 lw=1.8, label="Cumulative recharge (mm)")
    ax1.set_xlabel("Day index", fontsize=_FONT_LABEL)
    ax1.set_ylabel("Cumulative depth (mm)", fontsize=_FONT_LABEL)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper left", fontsize=10)
    ax1.tick_params(axis="both", labelsize=_FONT_TICK)
    ax1.set_title("Cumulative water balance", fontsize=_FONT_TITLE)

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════
# 3. Bootstrap uncertainty histogram
# ══════════════════════════════════════════════════════════════════════
def plot_uncertainty_histogram(uc_result):
    """Histogram of bootstrap recharge samples with CI band overlay.

    Returns
    -------
    matplotlib.figure.Figure or None if no samples available.
    """
    samples = _safe(uc_result, "rech_samples", None)
    if samples is None or len(samples) == 0:
        return None
    samples = np.asarray(samples, dtype=float)

    rech_mean = float(_safe(uc_result, "rech_mean", 0.0))
    ci_lo = float(_safe(uc_result, "rech_ci_lower", 0.0))
    ci_hi = float(_safe(uc_result, "rech_ci_upper", 0.0))
    ci_level = float(_safe(uc_result, "confidence_level", 0.95))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(samples, bins=30, color="#4C72B0", alpha=0.6, edgecolor="white")
    ax.axvline(rech_mean, color="#1D4D8A", lw=2.0,
               label=f"Mean = {rech_mean:.2f}%")
    ax.axvline(ci_lo, color="#C44E52", lw=1.5, ls="--",
               label=f"{ci_level*100:.0f}% CI = [{ci_lo:.2f}, {ci_hi:.2f}]")
    ax.axvline(ci_hi, color="#C44E52", lw=1.5, ls="--")
    ax.axvspan(ci_lo, ci_hi, color="#C44E52", alpha=0.08)
    ax.set_xlabel("Recharge ratio (% of P)", fontsize=_FONT_LABEL)
    ax.set_ylabel("Bootstrap sample count", fontsize=_FONT_LABEL)
    ax.set_title(f"Bootstrap distribution (n = {len(samples)})",
                 fontsize=_FONT_TITLE)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.tick_params(axis="both", labelsize=_FONT_TICK)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════
# 4. Sensitivity tornado
# ══════════════════════════════════════════════════════════════════════
def plot_sensitivity_tornado(ks_result):
    """Tornado chart of recharge sensitivity to ρ, Q/R, α.

    Expects KalmanSensitivityResult-like object with attribute `tornado_data`:
        list of (param_name, rech_at_low, rech_at_high, low_val, high_val).
    """
    td = _safe(ks_result, "tornado_data", None)
    baseline = float(_safe(ks_result, "baseline_recharge", 0.0))
    if not td:
        return None

    # 효과 크기 기준 정렬
    items = sorted(td, key=lambda x: abs(x[2] - x[1]), reverse=True)

    fig, ax = plt.subplots(figsize=(8, 0.7 * len(items) + 1.5))
    y = np.arange(len(items))
    for i, (pname, lo, hi, lo_val, hi_val) in enumerate(items):
        # 좌(lo) / 우(hi) 막대를 baseline 기준으로
        left = lo - baseline
        right = hi - baseline
        ax.barh(i, left, color="#C44E52", height=0.55, edgecolor="white",
                label="low" if i == 0 else None)
        ax.barh(i, right, color="#4C72B0", height=0.55, edgecolor="white",
                label="high" if i == 0 else None)
        # 끝값 라벨
        ax.text(left, i, f"  {lo:.2f}% ({pname}={lo_val:.3g})",
                va="center", ha="right" if left < 0 else "left", fontsize=8)
        ax.text(right, i, f"  {hi:.2f}% ({pname}={hi_val:.3g})",
                va="center", ha="left" if right > 0 else "right", fontsize=8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([t[0] for t in items], fontsize=_FONT_LABEL)
    ax.set_xlabel(f"Δ Recharge ratio (%) vs baseline {baseline:.2f}%",
                  fontsize=_FONT_LABEL)
    ax.set_title("Kalman parameter sensitivity (tornado)",
                 fontsize=_FONT_TITLE)
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(axis="both", labelsize=_FONT_TICK)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════
# 5. BMA across soil candidates
# ══════════════════════════════════════════════════════════════════════
def plot_bma_posterior(bma_result):
    """Bar chart of soil-type posterior probabilities.

    Expects BMAResult with `posterior` (length 12) and `soil_names`.
    """
    posterior = _safe(bma_result, "posterior", None)
    soil_names = _safe(bma_result, "soil_names", None)
    if posterior is None or soil_names is None:
        return None
    posterior = np.asarray(posterior)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(posterior))
    colors = ["#4C72B0" if p == posterior.max() else "#94A3B8"
              for p in posterior]
    ax.bar(x, posterior, color=colors, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(soil_names, fontsize=8, rotation=40, ha="right")
    ax.set_ylabel("Posterior probability  P(M | data)", fontsize=_FONT_LABEL)
    ax.set_title("Bayesian Model Averaging across soil candidates",
                 fontsize=_FONT_TITLE)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="y", labelsize=_FONT_TICK)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════
# 6. Three-method comparison (WTF / SCS-CN / FAO-56)
# ══════════════════════════════════════════════════════════════════════
def _collect_method_estimates(
    result_v27: Dict,
    scs_result: Any = None,
    fao56_result: Any = None,
    P_annual_mm: Optional[float] = None,
) -> List[Dict]:
    """Collect available method estimates into a uniform list of dicts.

    Each entry: {
        "name": str,
        "rech_pct": float,         # % of P
        "rech_mm": float,          # mm/yr
        "lo_pct": float | None,    # CI / band lower (% of P)
        "hi_pct": float | None,
        "method_type": str,        # description for caption
        "category": str,           # "et_aware" (true recharge) or
                                   # "infiltration_only" (upper bound)
    }
    """
    out: List[Dict] = []

    # WTF (always present) — 대수층 측, ET 자연 반영
    rech_pct = float(_safe(result_v27, "recharge_ratio", 0.0))
    if P_annual_mm is None or P_annual_mm <= 0:
        po_shifted = _np(_safe(result_v27, "po_shifted"))
        n_days = len(po_shifted)
        if n_days > 0:
            n_yr = max(n_days / 365.25, 1.0)
            P_annual_mm = float(np.nansum(po_shifted)) * 1000.0 / n_yr
        else:
            P_annual_mm = 0.0
    R_mm = P_annual_mm * rech_pct / 100.0
    out.append({
        "name": "hybrid-recharge",
        "rech_pct": rech_pct,
        "rech_mm": R_mm,
        "lo_pct": None,
        "hi_pct": None,
        "method_type": "Aquifer storage (water level fluctuation)",
        "category": "et_aware",
    })

    # SCS-CN — 침투만 (ET 미반영) → 다른 카테고리
    if scs_result is not None:
        rech_pct = float(_safe(scs_result, "recharge_ratio_pct", 0.0))
        out.append({
            "name": "SCS-CN (improved)",
            "rech_pct": rech_pct,
            "rech_mm": float(_safe(scs_result, "R_annual_mm", 0.0)),
            "lo_pct": float(_safe(scs_result, "recharge_cn_low", rech_pct)),
            "hi_pct": float(_safe(scs_result, "recharge_cn_high", rech_pct)),
            "method_type": "Surface infiltration (no ET, upper bound)",
            "category": "infiltration_only",
        })

    # FAO-56 — 심부 percolation, ET 명시 차감
    if fao56_result is not None:
        rech_pct = float(_safe(fao56_result, "recharge_ratio_pct", 0.0))
        out.append({
            "name": "FAO-56 SWB",
            "rech_pct": rech_pct,
            "rech_mm": float(_safe(fao56_result, "R_annual_mm", 0.0)),
            "lo_pct": None,
            "hi_pct": None,
            "method_type": "Deep percolation past root zone (with ET)",
            "category": "et_aware",
        })

    return out


def _split_by_category(estimates: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Split estimates into (et_aware, infiltration_only) lists."""
    et_aware = [e for e in estimates if e.get("category") == "et_aware"]
    infil = [e for e in estimates if e.get("category") == "infiltration_only"]
    return et_aware, infil


def _convergence_verdict(estimates: List[Dict]) -> Dict:
    """Compute convergence metrics across ET-aware methods only.

    Critical fix: SCS-CN measures *infiltration* (no ET), which is a
    physically different quantity than recharge.  Lumping it with WTF and
    FAO-56 produces misleading divergence verdicts.  Therefore we judge
    convergence only over ET-aware methods (true recharge), and report
    SCS-CN separately as an "infiltration ceiling".
    """
    et_aware, infil = _split_by_category(estimates)
    n_primary = len(et_aware)

    base = {
        "n_methods": len(estimates),
        "n_primary": n_primary,
        "n_supplementary": len(infil),
        "primary_names": [e["name"] for e in et_aware],
        "supplementary_names": [e["name"] for e in infil],
    }

    if n_primary == 0:
        return {**base, "verdict": "n/a", "verdict_class": "flag-warn",
                "verdict_text": "ET-반영 추정 없음", "spread_pct": 0,
                "rel_spread": 0, "min_pct": 0, "max_pct": 0,
                "median_pct": 0, "recommended_pct": 0}

    if n_primary == 1:
        only = et_aware[0]
        return {
            **base, "verdict": "single", "verdict_class": "flag-warn",
            "verdict_text": f"ⓘ 단일 ET-반영 방법 ({only['name']})",
            "spread_pct": 0, "rel_spread": 0,
            "min_pct": only["rech_pct"], "max_pct": only["rech_pct"],
            "median_pct": only["rech_pct"],
            "recommended_pct": only["rech_pct"],
        }

    vals = np.array([e["rech_pct"] for e in et_aware])
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    med = float(np.median(vals))
    spread = hi - lo
    rel = spread / max(med, 1e-6)

    if rel < 0.20:
        verdict = "converged"
        verdict_class = "flag-pass"
        verdict_text = (
            f"✓ ET-반영 방법 수렴 ({', '.join(e['name'] for e in et_aware)}, "
            f"상대 폭 {rel*100:.1f}% &lt; 20%)"
        )
    elif rel < 0.40:
        verdict = "moderate"
        verdict_class = "flag-warn"
        verdict_text = f"△ 보통 (상대 폭 {rel*100:.1f}%)"
    else:
        verdict = "diverged"
        verdict_class = "flag-fail"
        verdict_text = (
            f"✗ 발산 (상대 폭 {rel*100:.1f}% ≥ 40%) — "
            f"가정 차이 추적 필요"
        )

    return {
        **base,
        "verdict": verdict,
        "verdict_class": verdict_class,
        "verdict_text": verdict_text,
        "spread_pct": spread,
        "rel_spread": rel,
        "min_pct": lo,
        "max_pct": hi,
        "median_pct": med,
        "recommended_pct": med,   # 권장 보고값 = ET-반영 median
    }


def plot_method_comparison(estimates: List[Dict], P_annual_mm: float):
    """Bar chart of method estimates with uncertainty bands.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not estimates:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # 좌: % of P
    names = [e["name"] for e in estimates]
    pcts = [e["rech_pct"] for e in estimates]
    colors = ["#4C72B0", "#C44E52", "#55A868", "#8172B2", "#CCB974"]
    bar_colors = [colors[i % len(colors)] for i in range(len(estimates))]

    ax1.barh(names, pcts, color=bar_colors, edgecolor="white", height=0.5)
    # CI / band 오버레이
    for i, e in enumerate(estimates):
        if e["lo_pct"] is not None and e["hi_pct"] is not None:
            ax1.errorbar(
                x=e["rech_pct"], y=i,
                xerr=[[e["rech_pct"]-e["lo_pct"]], [e["hi_pct"]-e["rech_pct"]]],
                fmt="o", color="black", capsize=4, lw=1.0, markersize=3,
            )
        # 값 라벨
        ax1.text(e["rech_pct"], i, f"  {e['rech_pct']:.1f}%",
                 va="center", fontsize=9)
    ax1.set_xlabel("Recharge ratio (% of P)", fontsize=_FONT_LABEL)
    ax1.set_title("(a) % of annual precipitation", fontsize=_FONT_TITLE)
    ax1.grid(axis="x", alpha=0.3)
    ax1.tick_params(axis="both", labelsize=_FONT_TICK)

    # 우: mm/yr
    mms = [e["rech_mm"] for e in estimates]
    ax2.barh(names, mms, color=bar_colors, edgecolor="white", height=0.5)
    for i, e in enumerate(estimates):
        ax2.text(e["rech_mm"], i, f"  {e['rech_mm']:.0f} mm",
                 va="center", fontsize=9)
    ax2.set_xlabel("Annual recharge (mm/yr)", fontsize=_FONT_LABEL)
    ax2.set_title(f"(b) absolute (P = {P_annual_mm:.0f} mm/yr)",
                  fontsize=_FONT_TITLE)
    ax2.grid(axis="x", alpha=0.3)
    ax2.tick_params(axis="both", labelsize=_FONT_TICK)

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════
# HTML 리포트 빌더
# ══════════════════════════════════════════════════════════════════════
_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue",
                 "Apple SD Gothic Neo", sans-serif;
    max-width: 1100px; margin: 32px auto; padding: 0 24px;
    color: #222; line-height: 1.55;
  }}
  h1 {{ border-bottom: 2px solid #2c4a6a; padding-bottom: 8px; color: #2c4a6a; }}
  h2 {{ color: #2c4a6a; margin-top: 36px;
        border-left: 4px solid #2c4a6a; padding-left: 10px; }}
  h3 {{ color: #444; }}
  .meta {{ color: #666; font-size: 0.9em; margin-bottom: 24px; }}
  .meta table {{ border-collapse: collapse; }}
  .meta td {{ padding: 2px 12px 2px 0; }}
  .figure {{ margin: 16px 0 28px; }}
  .figure img {{ max-width: 100%; height: auto;
                 border: 1px solid #d0d7de; border-radius: 4px; }}
  .figure .caption {{ font-size: 0.9em; color: #555;
                      margin-top: 6px; font-style: italic; }}
  .flag-pass {{ color: #1a7f37; font-weight: 600; }}
  .flag-warn {{ color: #b35900; font-weight: 600; }}
  .flag-fail {{ color: #b00020; font-weight: 600; }}
  table.metrics {{ border-collapse: collapse; margin: 12px 0; }}
  table.metrics th, table.metrics td {{
    border: 1px solid #d0d7de; padding: 6px 12px; text-align: right;
  }}
  table.metrics th {{ background: #f6f8fa; }}
  table.metrics td:first-child, table.metrics th:first-child {{ text-align: left; }}
  ul.flags li {{ color: #b35900; margin: 4px 0; }}
  .skipped {{ color: #999; font-style: italic; padding: 8px 12px;
              background: #f6f8fa; border-radius: 4px; }}
  .disclaimer {{
    background: #fff7e6; border: 1px solid #f0c674; border-radius: 6px;
    padding: 12px 16px; margin: 24px 0; font-size: 0.9em;
  }}
  @media print {{
    body {{ max-width: none; margin: 0; }}
    h2 {{ page-break-before: avoid; }}
    .figure {{ page-break-inside: avoid; }}
  }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="meta">
  <table>
    <tr><td><b>Site</b></td><td>{site_name}</td></tr>
    <tr><td><b>Generated</b></td><td>{timestamp}</td></tr>
    <tr><td><b>Soil model</b></td><td>{soil_label}</td></tr>
    <tr><td><b>Observation period</b></td><td>{n_days} days</td></tr>
    <tr><td><b>Annual precipitation</b></td><td>{P_annual:.0f} mm/yr</td></tr>
  </table>
</div>

<div class="disclaimer">
  <b>Notes on interpretation.</b> This report bundles internal-consistency
  checks for a single observation well.  Recharge in real watersheds has no
  ground truth — values reflect defensibility under stated assumptions
  (parameters, soil model, preprocessing), not validation against truth.
</div>

<h2>1. Core estimate</h2>
{core_table}

<h2>2. Plausibility check</h2>
{plausibility_section}

<h2>3. Method comparison (WTF / SCS-CN / FAO-56)</h2>
<p style="font-size:0.9em;color:#555;">
방법론적으로 독립인 추정값 간 수렴 정도가 결과의 신뢰성을 판단하는
핵심 지표입니다.  Choi &amp; Ahn (1998) 표준에서 영감을 받아 SCS-CN을 두 번째
방법으로, FAO-56 (Allen et al. 1998) 일별 토양수분 물수지를 세 번째
방법으로 사용합니다.
</p>
{method_comparison_section}

<h2>4. Time-series diagnostics</h2>
<div class="figure">
  <img src="data:image/png;base64,{img_timeseries}" alt="time series">
  <div class="caption">
    Figure 1. Top — observed (gray), Kalman-filtered (blue), pure WTF (green
    dashed) water levels with pump-flagged intervals shaded red.  Middle —
    daily precipitation.  Bottom — daily WTF event recharge.
  </div>
</div>
<div class="figure">
  <img src="data:image/png;base64,{img_cumulative}" alt="cumulative">
  <div class="caption">
    Figure 2. Cumulative precipitation vs cumulative recharge — rough
    proportional check.
  </div>
</div>

<h2>5. Bootstrap uncertainty</h2>
{uncertainty_section}

<h2>6. Soil-type Bayesian Model Averaging</h2>
{bma_section}

<h2>7. Kalman parameter sensitivity</h2>
{sensitivity_section}

<h2>8. Pump preprocessing effect</h2>
{pump_section}

<div class="meta" style="margin-top:48px;">
  Report produced by hybrid-recharge well_report.py.  Definitions: see
  evaluation/well_report.py and uncertainty.py / bma.py / sensitivity.py.
</div>
</body>
</html>
"""


def _format_core_table(result_v27: Dict) -> str:
    rows = []

    def add(label, value):
        rows.append(f"<tr><td>{escape(label)}</td><td>{value}</td></tr>")

    add("RMSE",       f"{float(_safe(result_v27,'rmse',0)):.4f} m")
    add("CC",         f"{float(_safe(result_v27,'cc',0)):.4f}")
    nse = _safe(result_v27, "nse")
    if nse is not None:
        add("NSE",    f"{float(nse):.3f}")
    kge = _safe(result_v27, "kge")
    if kge is not None:
        add("KGE",    f"{float(kge):.3f}")
    pb = _safe(result_v27, "pbias")
    if pb is not None:
        add("PBIAS",  f"{float(pb):+.1f}%")
    add("Recharge ratio", f"{float(_safe(result_v27,'recharge_ratio',0)):.2f}% of P")
    add("Sy_eff",      f"{float(_safe(result_v27,'Sy_eff',0)):.4f}")
    add("opt_k",       f"{float(_safe(result_v27,'opt_k',0)):.4f}")
    add("opt_z",       f"{float(_safe(result_v27,'opt_z',0)):.2f} m")
    add("opt_lag",     f"{int(_safe(result_v27,'opt_lag',0))} day(s)")
    add("opt_rho",     f"{float(_safe(result_v27,'opt_rho',0)):.3f}")
    add("opt_alpha",   f"{float(_safe(result_v27,'opt_alpha',0)):.3f}")
    add("Pump contam.", f"{float(_safe(result_v27,'pump_contam_idx',0)):.3f}")
    return ('<table class="metrics"><tr><th>Quantity</th><th>Value</th></tr>'
            + "".join(rows) + "</table>")


def _format_plausibility_section(plaus: WellPlausibility) -> str:
    if plaus.pass_basic and not plaus.flags:
        status = '<span class="flag-pass">✓ PASS — 모든 기본 검사 통과</span>'
    elif plaus.pass_basic:
        status = '<span class="flag-warn">△ WARN</span>'
    else:
        status = '<span class="flag-fail">✗ FAIL</span>'

    parts = [f"<p><b>Status:</b> {status}</p>"]
    parts.append(
        '<table class="metrics">'
        '<tr><th>Quantity</th><th>Value</th></tr>'
        f'<tr><td>Recharge ratio</td><td>{plaus.recharge_ratio_pct:.2f}%</td></tr>'
        f'<tr><td>Annual recharge</td><td>{plaus.R_annual_mm:.0f} mm/yr</td></tr>'
        f'<tr><td>Annual precipitation</td><td>{plaus.P_annual_mm:.0f} mm/yr</td></tr>'
        f'<tr><td>R / P</td><td>{plaus.R_over_P*100:.1f}%</td></tr>'
        f'<tr><td>Sy_eff</td><td>{plaus.sy_eff:.4f}</td></tr>'
        f'<tr><td>opt_k</td><td>{plaus.opt_k:.4f}</td></tr>'
        f'<tr><td>Observation days</td><td>{plaus.n_obs_days}</td></tr>'
        '</table>'
    )
    if plaus.flags:
        parts.append("<ul class='flags'>")
        for f in plaus.flags:
            parts.append(f"<li>⚠ {escape(f)}</li>")
        parts.append("</ul>")
    return "\n".join(parts)


def _format_uncertainty_section(uc_result, img_b64: str) -> str:
    if uc_result is None:
        return '<p class="skipped">Bootstrap CI 분석이 실행되지 않았습니다 (Tab 5에서 실행 가능).</p>'
    n_boot = int(_safe(uc_result, "n_bootstrap", 0))
    ci_lvl = float(_safe(uc_result, "confidence_level", 0.95))
    rech_mean = float(_safe(uc_result, "rech_mean", 0.0))
    rech_lo = float(_safe(uc_result, "rech_ci_lower", 0.0))
    rech_hi = float(_safe(uc_result, "rech_ci_upper", 0.0))
    rmse_lo = float(_safe(uc_result, "rmse_ci_lower", 0.0))
    rmse_hi = float(_safe(uc_result, "rmse_ci_upper", 0.0))
    sy_lo = float(_safe(uc_result, "sy_ci_lower", 0.0))
    sy_hi = float(_safe(uc_result, "sy_ci_upper", 0.0))

    img_html = ""
    if img_b64:
        img_html = (
            f'<div class="figure">'
            f'<img src="data:image/png;base64,{img_b64}" alt="bootstrap hist">'
            f'<div class="caption">Figure 3. Bootstrap distribution of recharge ratio.</div>'
            f'</div>'
        )

    return f"""{img_html}
<table class="metrics">
  <tr><th>Quantity</th><th>Mean</th><th>{ci_lvl*100:.0f}% CI</th></tr>
  <tr><td>Recharge ratio</td><td>{rech_mean:.2f}%</td>
      <td>[{rech_lo:.2f}, {rech_hi:.2f}]</td></tr>
  <tr><td>RMSE</td><td>{float(_safe(uc_result,'rmse_mean',0)):.4f} m</td>
      <td>[{rmse_lo:.4f}, {rmse_hi:.4f}]</td></tr>
  <tr><td>Sy_eff</td><td>{float(_safe(uc_result,'sy_mean',0)):.4f}</td>
      <td>[{sy_lo:.4f}, {sy_hi:.4f}]</td></tr>
</table>
<p style="font-size:0.9em;color:#555;">
Block bootstrap (n = {n_boot}), BCa CI (Efron 1987).
</p>
"""


def _format_bma_section(bma_result, img_b64: str) -> str:
    if bma_result is None:
        return '<p class="skipped">BMA 분석이 실행되지 않았습니다 (Tab 1 토양 스캔 필요).</p>'
    rech_mean = float(_safe(bma_result, "recharge_mean", 0.0))
    rech_lo = float(_safe(bma_result, "recharge_ci_lo", 0.0))
    rech_hi = float(_safe(bma_result, "recharge_ci_hi", 0.0))
    dom_soil = int(_safe(bma_result, "dominant_soil", 0))
    dom_prob = float(_safe(bma_result, "dominant_prob", 0.0))
    conf_label = str(_safe(bma_result, "confidence_label", ""))
    n_eff = float(_safe(bma_result, "n_effective_models", 0))

    img_html = ""
    if img_b64:
        img_html = (
            f'<div class="figure">'
            f'<img src="data:image/png;base64,{img_b64}" alt="BMA posterior">'
            f'<div class="caption">Figure 4. Posterior probability across 12 soil candidates.</div>'
            f'</div>'
        )

    return f"""{img_html}
<table class="metrics">
  <tr><th>Quantity</th><th>Value</th></tr>
  <tr><td>BMA-weighted recharge</td><td>{rech_mean:.2f}%</td></tr>
  <tr><td>90% credible interval</td>
      <td>[{rech_lo:.2f}, {rech_hi:.2f}]%</td></tr>
  <tr><td>Dominant soil (sn)</td>
      <td>{dom_soil} (P = {dom_prob:.2f}, confidence: {escape(conf_label)})</td></tr>
  <tr><td>Effective # of models</td><td>{n_eff:.2f}</td></tr>
</table>
"""


def _format_sensitivity_section(ks_result, img_b64: str) -> str:
    if ks_result is None:
        return '<p class="skipped">Kalman 파라미터 민감도 분석이 실행되지 않았습니다 (Tab 6에서 실행).</p>'

    s_rho = float(_safe(ks_result, "sensitivity_rho", 0.0))
    s_qr = float(_safe(ks_result, "sensitivity_qr", 0.0))
    s_alpha = float(_safe(ks_result, "sensitivity_alpha", 0.0))

    img_html = ""
    if img_b64:
        img_html = (
            f'<div class="figure">'
            f'<img src="data:image/png;base64,{img_b64}" alt="tornado">'
            f'<div class="caption">Figure 5. Tornado chart — recharge change vs ρ, Q/R, α.</div>'
            f'</div>'
        )

    return f"""{img_html}
<table class="metrics">
  <tr><th>Parameter</th><th>Sensitivity index (elasticity)</th></tr>
  <tr><td>ρ (persistence)</td><td>{s_rho:+.3f}</td></tr>
  <tr><td>Q/R ratio</td><td>{s_qr:+.3f}</td></tr>
  <tr><td>α (blend)</td><td>{s_alpha:+.3f}</td></tr>
</table>
<p style="font-size:0.9em;color:#555;">
Sensitivity index = (ΔR/R) / (Δp/p) — proportional change in recharge per
proportional change in parameter.
</p>
"""


def _format_method_comparison_section(
    estimates: List[Dict],
    convergence: Dict,
    img_b64: str,
    P_annual_mm: float,
) -> str:
    """3-method comparison HTML section.

    Layout:
      §3.1 Primary recommendation (ET-aware methods median)
      §3.2 Method comparison figure + table (all methods)
      §3.3 Supplementary: SCS-CN as infiltration ceiling
    """
    if len(estimates) < 2:
        return (
            '<p class="skipped">방법 비교는 SCS-CN 또는 FAO-56 결과가 '
            '추가로 있어야 활성화됩니다 (현재 WTF만 가용).</p>'
        )

    et_aware, infil = _split_by_category(estimates)

    # ── §3.1 권장 보고값 박스 ──
    rec_pct = convergence.get("recommended_pct", 0.0)
    rec_mm = rec_pct / 100.0 * P_annual_mm
    if convergence["verdict"] == "converged":
        rec_band = (
            f"[{convergence['min_pct']:.1f}, {convergence['max_pct']:.1f}]%"
        )
        rec_text = (
            f'<div style="background:#e8f5ee;border:1px solid #95d3ad;'
            f'border-radius:6px;padding:14px 18px;margin:10px 0;">'
            f'<p style="margin:0;font-size:1.05em;">'
            f'<b>📌 권장 보고값:</b> '
            f'<b style="font-size:1.2em;color:#1a7f37;">{rec_pct:.1f}%</b> '
            f'({rec_mm:.0f} mm/yr) — {rec_band} 범위</p>'
            f'<p style="margin:6px 0 0 0;font-size:0.9em;color:#555;">'
            f'ET-반영 방법 ({", ".join(convergence["primary_names"])}) median.'
            f'</p></div>'
        )
    elif convergence["verdict"] == "single":
        rec_text = (
            f'<div style="background:#fff7e6;border:1px solid #f0c674;'
            f'border-radius:6px;padding:14px 18px;margin:10px 0;">'
            f'<p style="margin:0;font-size:1.05em;">'
            f'<b>📌 단일 추정값:</b> '
            f'<b style="font-size:1.2em;color:#b35900;">{rec_pct:.1f}%</b> '
            f'({rec_mm:.0f} mm/yr) — {convergence["primary_names"][0]}</p>'
            f'<p style="margin:6px 0 0 0;font-size:0.9em;color:#555;">'
            f'두 번째 ET-반영 방법(FAO-56)을 추가하면 수렴 검증 가능.'
            f'</p></div>'
        )
    elif convergence["verdict"] == "diverged":
        rec_text = (
            f'<div style="background:#fde8e8;border:1px solid #e09090;'
            f'border-radius:6px;padding:14px 18px;margin:10px 0;">'
            f'<p style="margin:0;font-size:1.05em;">'
            f'<b>⚠ ET-반영 방법 발산</b> — '
            f'단일 보고값 권장 어려움.  median {rec_pct:.1f}%, 범위 '
            f'{convergence["min_pct"]:.1f}–{convergence["max_pct"]:.1f}%</p>'
            f'<p style="margin:6px 0 0 0;font-size:0.9em;color:#555;">'
            f'가정 차이 추적: Sy, runoff_fraction, land_use 등 점검.'
            f'</p></div>'
        )
    else:
        rec_text = (
            f'<div style="background:#fff7e6;border:1px solid #f0c674;'
            f'border-radius:6px;padding:14px 18px;margin:10px 0;">'
            f'<p style="margin:0;">{rec_pct:.1f}% (median '
            f'{convergence["min_pct"]:.1f}–{convergence["max_pct"]:.1f}%)</p>'
            f'</div>'
        )

    # ── §3.2 figure ──
    img_html = ""
    if img_b64:
        img_html = (
            f'<div class="figure">'
            f'<img src="data:image/png;base64,{img_b64}" alt="method comparison">'
            f'<div class="caption">'
            f'Figure 3. (a) Recharge ratio (% of P) by method.  '
            f'Error bars: SCS-CN CN ±5 sensitivity band.  '
            f'(b) Same in mm/yr.'
            f'</div></div>'
        )

    # ── §3.2 표 — 카테고리별 그룹화 ──
    def fmt_row(e: Dict) -> str:
        ci = (f"[{e['lo_pct']:.1f}, {e['hi_pct']:.1f}]"
              if e['lo_pct'] is not None else "&mdash;")
        return (
            f"<tr><td>{escape(e['name'])}</td>"
            f"<td>{e['rech_pct']:.2f}%</td>"
            f"<td>{e['rech_mm']:.0f}</td>"
            f"<td>{ci}</td>"
            f"<td><i>{escape(e['method_type'])}</i></td></tr>"
        )

    table_rows = []
    if et_aware:
        table_rows.append(
            '<tr style="background:#e8f5ee;"><td colspan="5"><b>'
            'ET-반영 (true recharge) — 핵심 비교</b></td></tr>'
        )
        table_rows.extend(fmt_row(e) for e in et_aware)
    if infil:
        table_rows.append(
            '<tr style="background:#fff5e6;"><td colspan="5"><b>'
            '표면-침투만 (ET 미반영, 상한 추정)</b></td></tr>'
        )
        table_rows.extend(fmt_row(e) for e in infil)

    table = (
        '<table class="metrics">'
        '<tr><th>Method</th><th>% of P</th><th>mm/yr</th>'
        '<th>Range</th><th>Physical basis</th></tr>'
        + "".join(table_rows) + "</table>"
    )

    # ── §3.3 verdict ──
    verdict_html = (
        f'<p style="margin-top:14px;"><b>핵심 수렴 판정:</b> '
        f'<span class="{convergence["verdict_class"]}">'
        f'{convergence["verdict_text"]}</span></p>'
    )

    # ── §3.4 SCS-CN 침투 상한 별도 안내 ──
    supp_html = ""
    if infil:
        scs = infil[0]
        supp_html = (
            f'<div style="background:#fff5e6;border-left:4px solid #f0c674;'
            f'padding:10px 16px;margin-top:12px;font-size:0.9em;">'
            f'<b>ⓘ SCS-CN ({scs["rech_pct"]:.1f}%) — 침투 상한</b><br/>'
            f'SCS-CN은 ET를 차감하지 않아 *침투량*을 보고합니다. 따라서 '
            f'WTF/FAO-56(실제 함양)과 같은 카테고리가 아니며, 수렴 판정에서 '
            f'제외됩니다.  Choi &amp; Ahn (1998) baseline 방법으로 보조 참고용.'
            f'</div>'
        )

    interpretation = (
        '<details style="margin-top:12px;">'
        '<summary style="cursor:pointer; color:#2c4a6a;"><b>물리적 해석</b></summary>'
        '<ul style="font-size:0.9em;">'
        '<li><b>hybrid-recharge</b> — 대수층 측 (수위 변동으로부터 저류 변화 측정).  ET 자연 반영.</li>'
        '<li><b>FAO-56 SWB</b> — 표면 측 (Penman-Monteith ET + 일별 토양수분 → 심부 percolation).  '
        '국제 표준 (Allen et al. 1998).</li>'
        '<li><b>SCS-CN</b> — 표면 측 침투만 (강수 - 유출).  ET 미반영 → 함양 *상한*.</li>'
        '</ul>'
        '<p style="font-size:0.9em;">예상되는 순서: '
        '<b>SCS-CN ≥ FAO-56 ≈ WTF</b>.  FAO-56과 WTF의 수렴(둘 다 ET 반영)이 '
        '가장 결정적인 신뢰 지표입니다.</p>'
        '</details>'
    )

    return f"{rec_text}\n{img_html}\n{table}\n{verdict_html}\n{supp_html}\n{interpretation}"


def _format_pump_section(pump_result) -> str:
    if pump_result is None:
        return '<p class="skipped">펌핑 전처리가 실행되지 않았습니다 (Tab 2).</p>'
    v27o = _safe(pump_result, "v27_orig")
    v27c = _safe(pump_result, "v27_corr")
    if not v27o or not v27c:
        return '<p class="skipped">펌핑 전처리는 실행되었으나 v27 비교가 누락되었습니다.</p>'
    rmse_o = float(_safe(v27o, "rmse", 0.0))
    rmse_c = float(_safe(v27c, "rmse", 0.0))
    rech_o = float(_safe(v27o, "rech_rate", 0.0))
    rech_c = float(_safe(v27c, "rech_rate", 0.0))
    return f"""
<table class="metrics">
  <tr><th>Quantity</th><th>원본 수위</th><th>전처리 후</th><th>Δ</th></tr>
  <tr><td>RMSE (m)</td><td>{rmse_o:.4f}</td><td>{rmse_c:.4f}</td>
      <td>{rmse_c - rmse_o:+.4f}</td></tr>
  <tr><td>Recharge (%)</td><td>{rech_o:.2f}</td><td>{rech_c:.2f}</td>
      <td>{rech_c - rech_o:+.2f}</td></tr>
</table>
<p style="font-size:0.9em;color:#555;">
펌핑 구간 보정으로 인한 함양율 변화는 펌핑 영향의 정량적 추정치입니다.
</p>
"""


def build_well_html_report(
    result_v27: Dict,
    site_name: str = "Untitled well",
    P_annual_mm: Optional[float] = None,
    soil_label: str = "",
    uc_result: Any = None,
    bma_result: Any = None,
    kalman_sens: Any = None,
    pump_result: Any = None,
    scs_result: Any = None,
    fao56_result: Any = None,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
) -> str:
    """Build a self-contained HTML report for a single observation well.

    Required: ``result_v27`` (dict from CoreMetrics.to_dict()).
    Optional: ``uc_result`` / ``bma_result`` / ``kalman_sens`` / ``pump_result`` /
    ``scs_result`` / ``fao56_result`` are gracefully skipped if None.

    Returns
    -------
    str
        HTML content (also written to ``output_path`` if given).
    """
    if result_v27 is None:
        raise ValueError("result_v27 is required")
    if title is None:
        title = f"Well-mode Recharge Report — {site_name}"

    # Plausibility
    plaus = well_plausibility_check(result_v27, P_annual_mm=P_annual_mm)

    # Always-on figures
    fig_ts = plot_well_time_series(result_v27)
    img_ts = _fig_to_base64(fig_ts)

    fig_cum = plot_well_recharge_cumulative(result_v27)
    img_cum = _fig_to_base64(fig_cum)

    # Method comparison (3-method)
    estimates = _collect_method_estimates(
        result_v27, scs_result, fao56_result,
        P_annual_mm=plaus.P_annual_mm,
    )
    convergence = _convergence_verdict(estimates)
    img_mc = ""
    if len(estimates) >= 2:
        fig_mc = plot_method_comparison(estimates, plaus.P_annual_mm)
        if fig_mc is not None:
            img_mc = _fig_to_base64(fig_mc)
    method_comparison_html = _format_method_comparison_section(
        estimates, convergence, img_mc, plaus.P_annual_mm,
    )

    # Optional figures
    img_uc = ""
    if uc_result is not None:
        fig_uc = plot_uncertainty_histogram(uc_result)
        if fig_uc is not None:
            img_uc = _fig_to_base64(fig_uc)

    img_bma = ""
    if bma_result is not None:
        fig_bma = plot_bma_posterior(bma_result)
        if fig_bma is not None:
            img_bma = _fig_to_base64(fig_bma)

    img_sens = ""
    if kalman_sens is not None:
        fig_sens = plot_sensitivity_tornado(kalman_sens)
        if fig_sens is not None:
            img_sens = _fig_to_base64(fig_sens)

    html = _HTML_TEMPLATE.format(
        title=escape(title),
        site_name=escape(site_name),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
        soil_label=escape(soil_label) if soil_label else "&mdash;",
        n_days=plaus.n_obs_days,
        P_annual=plaus.P_annual_mm,
        core_table=_format_core_table(result_v27),
        plausibility_section=_format_plausibility_section(plaus),
        method_comparison_section=method_comparison_html,
        img_timeseries=img_ts,
        img_cumulative=img_cum,
        uncertainty_section=_format_uncertainty_section(uc_result, img_uc),
        bma_section=_format_bma_section(bma_result, img_bma),
        sensitivity_section=_format_sensitivity_section(kalman_sens, img_sens),
        pump_section=_format_pump_section(pump_result),
    )

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

    return html


# ══════════════════════════════════════════════════════════════════════
# 자체 시연 (DEMO 데이터로)
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from core_sim_v27 import core_sim_v27

    print("=== well_report.py demo (DEMO data, sn=6) ===\n")
    result = core_sim_v27(
        file_path="DEMO", k_val=-0.015, z_val=3.0, lag_val=0,
        sn_idx=6, q_val=0.005, r_val=0.10, rc_val=0.005,
        ignore_pump=0.0, sens_val=1.0, do_optimize=True,
    )
    if isinstance(result, dict) and "error" in result:
        print(f"core_sim_v27 failed: {result['error']}")
        sys.exit(1)
    if hasattr(result, "to_dict"):
        result = result.to_dict()

    out = "/tmp/well_report_DEMO.html"
    html = build_well_html_report(
        result_v27=result,
        site_name="DEMO Well",
        soil_label="Loam (sn=6)",
        output_path=out,
    )
    print(f"Report written: {out}")
    print(f"Size: {len(html.encode('utf-8'))/1024:.0f} KB")
    print(f"Open with: open {out}")
