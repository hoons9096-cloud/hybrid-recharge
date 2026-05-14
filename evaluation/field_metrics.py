"""
field_metrics.py -- Field-mode evaluation metrics (no ground truth required).

For real-world deployment, true recharge is unknown. These metrics provide
defensibility through internal consistency checks rather than truth-based
error.  All four are complementary; a recharge estimate that passes all of
them is internally coherent — not "validated", but defensible.

Categories
----------
1. Between-method spread
   When several estimators are applied to the same data, their disagreement
   is a proxy for epistemic uncertainty.  Cells where Lumped/Soil-weighted/
   EnKF agree are robust; cells where they diverge flag structural sensitivity
   to method choice.

2. Physical plausibility
   Recharge cannot exceed precipitation, must be non-negative, and typically
   sits in 5–30 % of annual rainfall in humid temperate climates (Healy 2010).
   Cells outside these bounds indicate parameter or input issues.

3. Soil-class coherence
   Cells sharing a soil texture should yield similar recharge unless other
   drivers (slope, depth) dominate.  High within-class variance relative to
   between-class variance suggests noise, not signal.

4. Well-level consistency
   At each observation well, the cell-mean estimated recharge should be of
   similar magnitude to the recharge implied directly by the well's positive
   water-level fluctuations during wet events (a coarse data-driven proxy).
   Large gaps suggest the spatial mapping introduced bias.

References
----------
Healy, R.W. (2010). Estimating Groundwater Recharge. Cambridge University
    Press.  (Ch. 1 — typical R/P ratios by climate)
Scanlon, B.R., Healy, R.W., Cook, P.G. (2002). Choosing appropriate techniques
    for quantifying groundwater recharge. Hydrogeology Journal, 10(1), 18–39.
Beven, K. (2006). A manifesto for the equifinality thesis. Journal of
    Hydrology, 320(1–2), 18–36.  (Multi-model spread as uncertainty proxy)
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# soil_db 경로 설정
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from soil_db import SOIL_DB


# ══════════════════════════════════════════════════════════════════════
# 1. Between-method spread
# ══════════════════════════════════════════════════════════════════════
@dataclass
class MethodSpread:
    """Per-cell spread across estimation methods (epistemic uncertainty proxy)."""
    method_names: List[str]
    mean_map: np.ndarray              # (ny, nx) [mm/yr]
    std_map: np.ndarray               # (ny, nx) [mm/yr]
    range_map: np.ndarray             # (ny, nx) max - min [mm/yr]
    cv_map: np.ndarray                # (ny, nx) std / mean [-]
    domain_mean: float                # spatial mean of mean_map
    domain_mean_std: float            # spatial mean of std_map
    domain_mean_cv: float             # spatial mean of cv_map
    method_domain_means: Dict[str, float]   # {method: domain_mean_recharge}


def between_method_spread(method_results: Dict[str, np.ndarray]) -> MethodSpread:
    """Compute per-cell statistics across multiple recharge estimates.

    Parameters
    ----------
    method_results : dict
        {method_name: recharge_map (ny, nx) [mm/yr]}.  At least 2 methods
        required for meaningful spread.

    Returns
    -------
    MethodSpread
        Per-cell mean, std, range, CV maps and domain summaries.
    """
    if len(method_results) < 2:
        raise ValueError(
            f"Need >=2 methods for spread analysis, got {len(method_results)}"
        )

    names = list(method_results.keys())
    stack = np.stack([method_results[n] for n in names], axis=0)  # (M, ny, nx)

    # 셰입 검증
    shapes = {n: m.shape for n, m in method_results.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"All recharge maps must have the same shape, got {shapes}")

    mean_map = np.mean(stack, axis=0)
    std_map = np.std(stack, axis=0, ddof=1) if stack.shape[0] > 1 else np.zeros_like(mean_map)
    range_map = np.max(stack, axis=0) - np.min(stack, axis=0)

    # CV: 0으로 나누기 방지 (mean ~ 0인 셀은 NaN 처리)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv_map = np.where(np.abs(mean_map) > 1e-6, std_map / np.abs(mean_map), np.nan)

    return MethodSpread(
        method_names=names,
        mean_map=mean_map,
        std_map=std_map,
        range_map=range_map,
        cv_map=cv_map,
        domain_mean=float(np.mean(mean_map)),
        domain_mean_std=float(np.mean(std_map)),
        domain_mean_cv=float(np.nanmean(cv_map)),
        method_domain_means={n: float(np.mean(method_results[n])) for n in names},
    )


# ══════════════════════════════════════════════════════════════════════
# 2. Physical plausibility
# ══════════════════════════════════════════════════════════════════════
@dataclass
class PlausibilityReport:
    """Physical plausibility checks on a recharge estimate."""
    method_name: str
    n_cells: int
    P_annual_mm: float
    mean_R: float                     # [mm/yr]
    median_R: float
    min_R: float
    max_R: float
    R_over_P: float                   # mean(R)/P_annual
    n_negative: int                   # R < 0
    n_above_precip: int               # R > P_annual
    n_above_half_precip: int          # R > 0.5*P_annual (rare in humid temperate)
    n_below_typical: int              # R < 0.02*P_annual (very dry recharge)
    flags: List[str] = field(default_factory=list)

    @property
    def pass_basic(self) -> bool:
        """모든 셀이 비음수이고 강수 미만인가."""
        return self.n_negative == 0 and self.n_above_precip == 0


def plausibility_check(
    recharge_map: np.ndarray,
    P_annual_mm: float,
    method_name: str = "",
    typical_low_frac: float = 0.02,
    typical_high_frac: float = 0.50,
) -> PlausibilityReport:
    """Check a recharge map against physical bounds.

    Parameters
    ----------
    recharge_map : (ny, nx) np.ndarray
        Annual recharge [mm/yr].
    P_annual_mm : float
        Annual precipitation [mm/yr].  Must be > 0.
    method_name : str, optional
        Identifier for the method being checked.
    typical_low_frac : float
        R/P below this is flagged as suspiciously low (default 2%).
    typical_high_frac : float
        R/P above this is flagged as suspiciously high (default 50%).

    Returns
    -------
    PlausibilityReport
    """
    if P_annual_mm <= 0:
        raise ValueError(f"P_annual_mm must be positive, got {P_annual_mm}")

    R = recharge_map
    n_cells = R.size

    n_neg = int(np.sum(R < 0))
    n_above_p = int(np.sum(R > P_annual_mm))
    n_above_half = int(np.sum(R > typical_high_frac * P_annual_mm))
    n_below_typ = int(np.sum(R < typical_low_frac * P_annual_mm))

    mean_R = float(np.mean(R))
    r_over_p = mean_R / P_annual_mm

    flags: List[str] = []
    if n_neg > 0:
        flags.append(f"{n_neg} 셀이 음수 함양 (물리적으로 불가능)")
    if n_above_p > 0:
        flags.append(f"{n_above_p} 셀이 R > P (물리적으로 불가능)")
    if n_above_half > 0 and n_above_p == 0:
        flags.append(f"{n_above_half} 셀이 R > {typical_high_frac*100:.0f}% * P (이례적으로 높음)")
    if n_below_typ > 0:
        flags.append(f"{n_below_typ} 셀이 R < {typical_low_frac*100:.0f}% * P (이례적으로 낮음)")
    if r_over_p > typical_high_frac:
        flags.append(f"전체 평균 R/P = {r_over_p*100:.1f}% (전형적 5–30% 초과)")

    return PlausibilityReport(
        method_name=method_name,
        n_cells=n_cells,
        P_annual_mm=P_annual_mm,
        mean_R=mean_R,
        median_R=float(np.median(R)),
        min_R=float(np.min(R)),
        max_R=float(np.max(R)),
        R_over_P=r_over_p,
        n_negative=n_neg,
        n_above_precip=n_above_p,
        n_above_half_precip=n_above_half,
        n_below_typical=n_below_typ,
        flags=flags,
    )


# ══════════════════════════════════════════════════════════════════════
# 3. Soil-class coherence
# ══════════════════════════════════════════════════════════════════════
@dataclass
class SoilClassCoherence:
    """Within- vs between-soil-class variance decomposition."""
    method_name: str
    n_classes: int
    class_means: Dict[int, float]            # {soil_idx: mean R}
    class_stds: Dict[int, float]             # {soil_idx: std R}
    class_counts: Dict[int, int]             # {soil_idx: n_cells}
    within_class_variance: float             # average within-class variance
    between_class_variance: float            # variance of class means
    coherence_ratio: float                   # between / (within + between), 0-1


def soil_class_coherence(
    recharge_map: np.ndarray,
    soil_map: np.ndarray,
    method_name: str = "",
) -> SoilClassCoherence:
    """Decompose recharge variance into within- and between-soil-class.

    A high coherence_ratio (close to 1) means most variation is explained by
    soil type, suggesting the method respects soil heterogeneity.  A low ratio
    (close to 0) suggests within-class noise dominates — either the data are
    very noisy, or the method is not soil-aware.

    Parameters
    ----------
    recharge_map : (ny, nx) np.ndarray
    soil_map     : (ny, nx) np.ndarray  (integer soil indices)
    method_name  : str

    Returns
    -------
    SoilClassCoherence
    """
    if recharge_map.shape != soil_map.shape:
        raise ValueError(
            f"shape mismatch: recharge {recharge_map.shape} vs soil {soil_map.shape}"
        )

    unique_soils = np.unique(soil_map)
    class_means: Dict[int, float] = {}
    class_stds: Dict[int, float] = {}
    class_counts: Dict[int, int] = {}
    within_vars = []
    weights = []

    for si in unique_soils:
        si = int(si)
        mask = soil_map == si
        n = int(mask.sum())
        if n == 0:
            continue
        vals = recharge_map[mask]
        class_means[si] = float(np.mean(vals))
        class_stds[si] = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        class_counts[si] = n
        if n > 1:
            within_vars.append(np.var(vals, ddof=1))
            weights.append(n)

    if not within_vars:
        within_var = 0.0
    else:
        within_var = float(np.average(within_vars, weights=weights))

    means_arr = np.array(list(class_means.values()))
    between_var = float(np.var(means_arr, ddof=1)) if len(means_arr) > 1 else 0.0

    total = within_var + between_var
    coherence = (between_var / total) if total > 1e-12 else 0.0

    return SoilClassCoherence(
        method_name=method_name,
        n_classes=len(class_means),
        class_means=class_means,
        class_stds=class_stds,
        class_counts=class_counts,
        within_class_variance=within_var,
        between_class_variance=between_var,
        coherence_ratio=coherence,
    )


# ══════════════════════════════════════════════════════════════════════
# 4. Well-level consistency
# ══════════════════════════════════════════════════════════════════════
@dataclass
class WellConsistencyRecord:
    """One well's consistency between estimated recharge and its observed dh."""
    well_idx: int
    soil_type_idx: int
    Sy_at_well: float
    estimated_R: float                # [mm/yr] from cell containing well
    obs_implied_R: float              # [mm/yr] from raw water level dh × Sy
    relative_diff: float              # (est - obs) / obs (NaN if obs ≈ 0)
    n_wet_events: int


@dataclass
class WellConsistencySummary:
    """Aggregated well-level consistency across all wells."""
    method_name: str
    records: List[WellConsistencyRecord]
    n_wells: int
    median_relative_diff: float
    fraction_within_20pct: float      # |rel_diff| < 0.20
    fraction_within_50pct: float


def _observed_implied_recharge(
    ho_obs: np.ndarray,
    P: np.ndarray,
    Sy: float,
    rain_threshold_m: float = 0.001,
) -> Tuple[float, int]:
    """Approximate annual recharge from observed dh during wet events.

    For each rainy day, take the maximum positive dh in the following 5 days
    and accumulate Sy * dh.  This is a coarse, transparent proxy — the same
    physical idea as WTF but without recession projection — meant only as a
    sanity check, not as a method itself.

    Parameters
    ----------
    ho_obs : (n_days,) [m]
    P      : (n_days,) [m/day]
    Sy     : float
    rain_threshold_m : float
        Min daily P to count as a wet event [m/day].

    Returns
    -------
    annual_R_mm : float
    n_events   : int
    """
    n = len(ho_obs)
    if n < 10:
        return 0.0, 0

    total_R_m = 0.0
    n_events = 0
    i = 0
    while i < n:
        if P[i] <= rain_threshold_m:
            i += 1
            continue
        # 5일 윈도우 내 최대 양의 dh
        end = min(n, i + 6)
        window = ho_obs[i:end]
        if len(window) < 2:
            i += 1
            continue
        h_pre = ho_obs[max(0, i - 1)]
        dh = float(np.max(window) - h_pre)
        if dh > 0:
            total_R_m += Sy * dh
            n_events += 1
        # 다음 이벤트 탐색: 윈도우 종료 이후
        i = end

    n_years = max(n / 365.25, 1.0)
    return total_R_m / n_years * 1000.0, n_events


def well_consistency(
    recharge_map: np.ndarray,
    observations: Dict,
    domain,
    method_name: str = "",
) -> WellConsistencySummary:
    """Compare cell-level estimated recharge with each well's observed-dh proxy.

    Uses a deliberately simple "max-dh in 5-day window × Sy" estimator on the
    raw water levels (no recession correction) as an *independent* check.
    This is biased high (ignores recession) but is transparent and serves as
    a coarse sanity check.

    Parameters
    ----------
    recharge_map : (ny, nx) np.ndarray [mm/yr]
    observations : dict
        Must contain 'P', 'ho_obs', 'well_soil_types'.
    domain : SyntheticDomain (or compatible)
        Must expose well_rows, well_cols.
    method_name : str

    Returns
    -------
    WellConsistencySummary
    """
    P = np.asarray(observations["P"])
    ho_obs = np.asarray(observations["ho_obs"])
    well_soils = np.asarray(observations["well_soil_types"])
    well_rows = np.asarray(domain.well_rows)
    well_cols = np.asarray(domain.well_cols)
    n_wells = ho_obs.shape[0]

    records: List[WellConsistencyRecord] = []
    rel_diffs: List[float] = []

    for w in range(n_wells):
        soil_idx = int(well_soils[w])
        Sy = SOIL_DB[soil_idx].sy_lit
        est_R = float(recharge_map[int(well_rows[w]), int(well_cols[w])])
        obs_R, n_evt = _observed_implied_recharge(ho_obs[w], P, Sy)

        if obs_R > 1e-3:
            rel = (est_R - obs_R) / obs_R
        else:
            rel = float("nan")
        rel_diffs.append(rel)

        records.append(WellConsistencyRecord(
            well_idx=w,
            soil_type_idx=soil_idx,
            Sy_at_well=Sy,
            estimated_R=est_R,
            obs_implied_R=obs_R,
            relative_diff=rel,
            n_wet_events=n_evt,
        ))

    rel_arr = np.array(rel_diffs, dtype=float)
    finite = rel_arr[np.isfinite(rel_arr)]
    if finite.size == 0:
        median_rel = float("nan")
        frac_20 = 0.0
        frac_50 = 0.0
    else:
        median_rel = float(np.median(finite))
        frac_20 = float(np.mean(np.abs(finite) < 0.20))
        frac_50 = float(np.mean(np.abs(finite) < 0.50))

    return WellConsistencySummary(
        method_name=method_name,
        records=records,
        n_wells=n_wells,
        median_relative_diff=median_rel,
        fraction_within_20pct=frac_20,
        fraction_within_50pct=frac_50,
    )


# ══════════════════════════════════════════════════════════════════════
# 통합 요약 리포트
# ══════════════════════════════════════════════════════════════════════
def field_summary(
    method_results: Dict[str, np.ndarray],
    observations: Dict,
    domain,
    P_annual_mm: float,
) -> str:
    """Generate a human-readable text report combining all field-mode metrics.

    Parameters
    ----------
    method_results : dict
        {method_name: recharge_map}.
    observations   : dict
        Must contain 'P', 'ho_obs', 'well_soil_types'.
    domain         : SyntheticDomain
    P_annual_mm    : float
        Annual precipitation [mm/yr].

    Returns
    -------
    str
        Formatted multi-section text report.
    """
    lines = []
    sep = "═" * 72
    sub = "─" * 72

    lines.append(sep)
    lines.append(" Field-mode Consistency Report")
    lines.append(sep)
    lines.append(f"  Annual precipitation : {P_annual_mm:8.1f} mm/yr")
    lines.append(f"  Methods compared     : {', '.join(method_results.keys())}")
    lines.append(f"  Domain shape         : {next(iter(method_results.values())).shape}")
    lines.append("")

    # 1. 방법 간 spread
    lines.append(" [1] Between-method spread (epistemic uncertainty proxy)")
    lines.append(sub)
    if len(method_results) >= 2:
        sp = between_method_spread(method_results)
        lines.append(f"  Domain mean R       : {sp.domain_mean:8.1f} mm/yr")
        lines.append(f"  Mean spread (std)   : {sp.domain_mean_std:8.1f} mm/yr")
        lines.append(f"  Mean CV across cells: {sp.domain_mean_cv:8.3f}")
        lines.append(f"  Per-method domain means:")
        for name, m in sp.method_domain_means.items():
            lines.append(f"    {name:<28s}: {m:8.1f} mm/yr")
    else:
        lines.append("  (need >=2 methods for spread analysis)")
    lines.append("")

    # 2. 각 방법별 plausibility
    lines.append(" [2] Physical plausibility per method")
    lines.append(sub)
    for name, R in method_results.items():
        rep = plausibility_check(R, P_annual_mm, method_name=name)
        lines.append(f"  · {name}")
        lines.append(
            f"      mean R = {rep.mean_R:7.1f} mm/yr "
            f"(R/P = {rep.R_over_P*100:5.1f}%), "
            f"range = [{rep.min_R:6.1f}, {rep.max_R:6.1f}]"
        )
        if rep.flags:
            for f in rep.flags:
                lines.append(f"      ⚠ {f}")
        else:
            lines.append("      ✓ 모든 기본 검사 통과")
    lines.append("")

    # 3. 토양유형 내부 일관성
    lines.append(" [3] Soil-class coherence (between-class / total variance)")
    lines.append(sub)
    for name, R in method_results.items():
        coh = soil_class_coherence(R, domain.soil_map, method_name=name)
        lines.append(
            f"  · {name:<28s}: coherence = {coh.coherence_ratio:5.3f} "
            f"({coh.n_classes} classes, "
            f"between/within = {coh.between_class_variance:.1f}/"
            f"{coh.within_class_variance:.1f})"
        )
    lines.append("")

    # 4. 관측정별 self-consistency
    lines.append(" [4] Well-level consistency (estimated vs observed-dh proxy)")
    lines.append(sub)
    for name, R in method_results.items():
        wc = well_consistency(R, observations, domain, method_name=name)
        lines.append(
            f"  · {name:<28s}: median Δ = {wc.median_relative_diff*100:+6.1f}%, "
            f"|Δ|<20% in {wc.fraction_within_20pct*100:5.1f}% of wells, "
            f"|Δ|<50% in {wc.fraction_within_50pct*100:5.1f}%"
        )
    lines.append("")
    lines.append(sep)

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# 자체 테스트
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=== field_metrics.py self-test ===\n")
    rng = np.random.default_rng(42)
    ny, nx = 50, 50

    # 가상의 3가지 방법 결과
    R_lumped = np.full((ny, nx), 100.0)
    R_soil = 100.0 + 30.0 * rng.standard_normal((ny, nx))
    R_enkf = 100.0 + 20.0 * rng.standard_normal((ny, nx))

    method_results = {
        "Lumped": R_lumped,
        "Soil-weighted": R_soil,
        "EnKF": R_enkf,
    }

    # 1. spread
    sp = between_method_spread(method_results)
    print(f"[1] spread: mean R = {sp.domain_mean:.1f}, mean std = {sp.domain_mean_std:.1f}")

    # 2. plausibility
    rep = plausibility_check(R_soil, P_annual_mm=1200.0, method_name="Soil-weighted")
    print(f"[2] plausibility: R/P = {rep.R_over_P*100:.1f}%, flags = {len(rep.flags)}")

    # 3. soil coherence (가짜 토양 맵)
    soil_map = rng.integers(1, 5, (ny, nx))
    coh = soil_class_coherence(R_soil, soil_map, method_name="Soil-weighted")
    print(f"[3] coherence ratio = {coh.coherence_ratio:.3f}")

    print("\nself-test passed")
