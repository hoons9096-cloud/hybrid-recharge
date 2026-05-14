"""
field_report.py -- Field-mode integrated HTML report generator.

Combines field_metrics outputs with diagnostic figures into a single
self-contained HTML file (PNGs embedded as base64).  Designed for
practitioner deliverables: open in any browser, "Print → Save as PDF"
to share with stakeholders.

Unlike figures.py (which compares estimates against truth), all figures
here are field-mode: they work with observations alone and visualise
internal consistency rather than truth-based error.

Figures
-------
1. Method comparison maps   -- side-by-side recharge maps (no truth panel)
2. Spread / uncertainty map -- per-cell std across methods
3. Recharge histograms      -- with R/P plausibility bounds
4. Soil-class boxplots      -- recharge distribution by soil texture
5. Well consistency scatter -- estimated vs observed-dh-implied recharge

Usage
-----
    from evaluation.field_report import build_html_report
    out = build_html_report(
        method_results, observations, domain,
        P_annual_mm=1200.0,
        site_name="Daedeok-1",
        output_path="report.html",
    )
"""
from __future__ import annotations

import base64
import io
import os
import sys
from datetime import datetime
from html import escape
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt

# 프로젝트 루트 경로
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soil_db import SOIL_DB
from evaluation.field_metrics import (
    between_method_spread,
    plausibility_check,
    soil_class_coherence,
    well_consistency,
    field_summary,
    MethodSpread,
)


# ──────────────────────────────────────────────────────────
# 공통 스타일
# ──────────────────────────────────────────────────────────
_FONT_TITLE = 13
_FONT_LABEL = 11
_FONT_TICK = 9
_CMAP_RECHARGE = "YlGnBu"
_CMAP_SPREAD = "magma"
_DPI = 150


def _fig_to_base64(fig) -> str:
    """matplotlib Figure → base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# ══════════════════════════════════════════════════════════════════════
# 1. Method comparison maps (no truth)
# ══════════════════════════════════════════════════════════════════════
def plot_method_comparison_maps(
    method_results: Dict[str, np.ndarray],
    domain,
):
    """Side-by-side recharge maps for each method, shared color scale.

    Unlike figures.plot_recharge_comparison(), this does NOT include a
    "true recharge" panel — it is for field deployment where truth is
    unknown.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = len(method_results)
    if n == 0:
        raise ValueError("method_results is empty")

    all_maps = list(method_results.values())
    vmin = min(float(m.min()) for m in all_maps)
    vmax = max(float(m.max()) for m in all_maps)
    if vmax - vmin < 1e-6:
        vmax = vmin + 1.0  # 색상 범위 보호

    cfg = domain.config
    extent = [0, cfg.nx * cfg.dx / 1000, 0, cfg.ny * cfg.dy / 1000]
    wx = domain.x_centers[domain.well_cols] / 1000
    wy = domain.y_centers[domain.well_rows] / 1000

    fig, axes = plt.subplots(1, n, figsize=(4.0 * n + 1.0, 4.2))
    if n == 1:
        axes = [axes]

    labels = "abcdefghij"
    for i, (name, R) in enumerate(method_results.items()):
        ax = axes[i]
        im = ax.imshow(R, origin="lower", extent=extent,
                       cmap=_CMAP_RECHARGE, vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        ax.scatter(wx, wy, c="red", marker="^", s=22, edgecolors="k",
                   linewidths=0.4, zorder=5)
        char = labels[i] if i < len(labels) else ""
        ax.set_title(f"({char}) {name}", fontsize=_FONT_TITLE)
        ax.set_xlabel("X (km)", fontsize=_FONT_LABEL)
        if i == 0:
            ax.set_ylabel("Y (km)", fontsize=_FONT_LABEL)
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis="both", labelsize=_FONT_TICK)

    cbar = fig.colorbar(im, ax=axes, shrink=0.85, aspect=30, pad=0.02)
    cbar.set_label("Recharge (mm/yr)", fontsize=_FONT_LABEL)
    cbar.ax.tick_params(labelsize=_FONT_TICK)

    return fig


# ══════════════════════════════════════════════════════════════════════
# 2. Spread / uncertainty map
# ══════════════════════════════════════════════════════════════════════
def plot_method_spread_map(spread: MethodSpread, domain):
    """Per-cell standard deviation across methods (epistemic uncertainty).

    Returns
    -------
    matplotlib.figure.Figure
    """
    cfg = domain.config
    extent = [0, cfg.nx * cfg.dx / 1000, 0, cfg.ny * cfg.dy / 1000]
    wx = domain.x_centers[domain.well_cols] / 1000
    wy = domain.y_centers[domain.well_rows] / 1000

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # (a) mean across methods
    ax = axes[0]
    im0 = ax.imshow(spread.mean_map, origin="lower", extent=extent,
                    cmap=_CMAP_RECHARGE, interpolation="nearest")
    ax.scatter(wx, wy, c="red", marker="^", s=22, edgecolors="k", linewidths=0.4)
    ax.set_title("(a) Mean across methods", fontsize=_FONT_TITLE)
    ax.set_xlabel("X (km)", fontsize=_FONT_LABEL)
    ax.set_ylabel("Y (km)", fontsize=_FONT_LABEL)
    ax.tick_params(axis="both", labelsize=_FONT_TICK)
    cb0 = fig.colorbar(im0, ax=ax, shrink=0.85)
    cb0.set_label("Mean R (mm/yr)", fontsize=_FONT_LABEL)
    cb0.ax.tick_params(labelsize=_FONT_TICK)

    # (b) std across methods (uncertainty)
    ax = axes[1]
    im1 = ax.imshow(spread.std_map, origin="lower", extent=extent,
                    cmap=_CMAP_SPREAD, interpolation="nearest")
    ax.scatter(wx, wy, c="cyan", marker="^", s=22, edgecolors="k", linewidths=0.4)
    ax.set_title("(b) Method-to-method spread (std)", fontsize=_FONT_TITLE)
    ax.set_xlabel("X (km)", fontsize=_FONT_LABEL)
    ax.set_yticklabels([])
    ax.tick_params(axis="both", labelsize=_FONT_TICK)
    cb1 = fig.colorbar(im1, ax=ax, shrink=0.85)
    cb1.set_label("Std (mm/yr)", fontsize=_FONT_LABEL)
    cb1.ax.tick_params(labelsize=_FONT_TICK)

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════
# 3. Recharge histograms with plausibility bounds
# ══════════════════════════════════════════════════════════════════════
def plot_recharge_histograms(
    method_results: Dict[str, np.ndarray],
    P_annual_mm: float,
):
    """Overlay histograms of recharge values with R/P bound markers.

    Reference lines drawn at 2%, 30%, 50% of P_annual_mm.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    bins = 40

    all_vals = np.concatenate([R.ravel() for R in method_results.values()])
    lo = float(np.min(all_vals))
    hi = float(np.max(all_vals))
    bin_edges = np.linspace(lo, hi, bins + 1)

    for i, (name, R) in enumerate(method_results.items()):
        ax.hist(R.ravel(), bins=bin_edges, alpha=0.45,
                label=name, color=colors[i % len(colors)],
                edgecolor="white", linewidth=0.4)

    # 참조선
    for frac, style in [(0.02, ":"), (0.30, "--"), (0.50, "-.")]:
        x = frac * P_annual_mm
        if lo <= x <= hi:
            ax.axvline(x, color="gray", linestyle=style, linewidth=1.2,
                       label=f"{int(frac*100)}% × P")

    ax.set_xlabel("Recharge (mm/yr)", fontsize=_FONT_LABEL)
    ax.set_ylabel("Cell count", fontsize=_FONT_LABEL)
    ax.set_title(f"Recharge distribution (P = {P_annual_mm:.0f} mm/yr)",
                 fontsize=_FONT_TITLE)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    ax.tick_params(axis="both", labelsize=_FONT_TICK)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════
# 4. Soil-class boxplots
# ══════════════════════════════════════════════════════════════════════
def plot_soil_class_boxplots(
    method_results: Dict[str, np.ndarray],
    soil_map: np.ndarray,
):
    """Boxplots of recharge per soil class, one panel per method.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = len(method_results)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n + 0.5, 4.2),
                             sharey=True)
    if n == 1:
        axes = [axes]

    unique_soils = sorted(int(s) for s in np.unique(soil_map))
    soil_names = []
    for si in unique_soils:
        try:
            soil_names.append(SOIL_DB[si].name)
        except (KeyError, IndexError):
            soil_names.append(f"soil-{si}")

    for ax, (name, R) in zip(axes, method_results.items()):
        data = []
        for si in unique_soils:
            mask = soil_map == si
            data.append(R[mask] if mask.any() else np.array([0.0]))
        bp = ax.boxplot(data, tick_labels=soil_names, patch_artist=True,
                        showmeans=True, meanline=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#bcd6e8")
            patch.set_edgecolor("#2c4a6a")
        ax.set_title(name, fontsize=_FONT_TITLE)
        ax.set_xlabel("Soil class", fontsize=_FONT_LABEL)
        ax.tick_params(axis="x", labelsize=8, rotation=30)
        ax.tick_params(axis="y", labelsize=_FONT_TICK)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Recharge (mm/yr)", fontsize=_FONT_LABEL)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════
# 5. Well consistency scatter
# ══════════════════════════════════════════════════════════════════════
def plot_well_consistency_scatter(
    method_results: Dict[str, np.ndarray],
    observations: Dict,
    domain,
):
    """Estimated vs observed-dh-implied recharge at each well.

    1:1 line included.  Points off the line indicate cells where the
    spatial mapping differs from what the well alone would suggest.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = len(method_results)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n + 0.5, 4.2),
                             sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    # 모든 점에서 통일된 축 범위
    all_obs = []
    all_est = []
    summaries = {}
    for name, R in method_results.items():
        wc = well_consistency(R, observations, domain, method_name=name)
        summaries[name] = wc
        for rec in wc.records:
            all_obs.append(rec.obs_implied_R)
            all_est.append(rec.estimated_R)

    if not all_obs:
        # 안전장치
        lim_lo, lim_hi = 0.0, 1.0
    else:
        lim_lo = 0.9 * min(min(all_obs), min(all_est))
        lim_hi = 1.1 * max(max(all_obs), max(all_est))
        if lim_hi - lim_lo < 1e-6:
            lim_hi = lim_lo + 1.0

    for ax, (name, wc) in zip(axes, summaries.items()):
        obs_arr = np.array([r.obs_implied_R for r in wc.records])
        est_arr = np.array([r.estimated_R for r in wc.records])
        soils = np.array([r.soil_type_idx for r in wc.records])

        sc = ax.scatter(obs_arr, est_arr, c=soils, cmap="tab10",
                        s=60, edgecolors="k", linewidths=0.5, alpha=0.85)
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1.0,
                label="1:1 line")

        ax.set_xlim(lim_lo, lim_hi)
        ax.set_ylim(lim_lo, lim_hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(name, fontsize=_FONT_TITLE)
        ax.set_xlabel("Obs-implied R (mm/yr)", fontsize=_FONT_LABEL)
        ax.tick_params(axis="both", labelsize=_FONT_TICK)
        ax.grid(alpha=0.3)
        # 요약 주석
        text = (
            f"median Δ = {wc.median_relative_diff*100:+.1f}%\n"
            f"|Δ|<20%: {wc.fraction_within_20pct*100:.0f}%"
        )
        ax.text(0.05, 0.95, text, transform=ax.transAxes,
                fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", alpha=0.8, edgecolor="gray"))
        ax.legend(loc="lower right", fontsize=8)

    axes[0].set_ylabel("Estimated R at well (mm/yr)", fontsize=_FONT_LABEL)
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
  h2 {{ color: #2c4a6a; margin-top: 36px; border-left: 4px solid #2c4a6a;
        padding-left: 10px; }}
  h3 {{ color: #444; }}
  .meta {{ color: #666; font-size: 0.9em; margin-bottom: 24px; }}
  .meta table {{ border-collapse: collapse; }}
  .meta td {{ padding: 2px 12px 2px 0; }}
  .summary-pre {{
    background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 6px;
    padding: 14px 16px; font-family: "SF Mono", Menlo, Consolas, monospace;
    font-size: 0.85em; overflow-x: auto; white-space: pre;
  }}
  .figure {{ margin: 16px 0 28px; }}
  .figure img {{ max-width: 100%; height: auto; border: 1px solid #d0d7de;
                 border-radius: 4px; }}
  .figure .caption {{
    font-size: 0.9em; color: #555; margin-top: 6px; font-style: italic;
  }}
  .flag-pass {{ color: #1a7f37; font-weight: 600; }}
  .flag-warn {{ color: #b35900; font-weight: 600; }}
  .flag-fail {{ color: #b00020; font-weight: 600; }}
  table.metrics {{ border-collapse: collapse; margin: 12px 0; }}
  table.metrics th, table.metrics td {{
    border: 1px solid #d0d7de; padding: 6px 12px; text-align: right;
  }}
  table.metrics th {{ background: #f6f8fa; }}
  table.metrics td:first-child, table.metrics th:first-child {{ text-align: left; }}
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
    <tr><td><b>Annual precipitation</b></td><td>{P_annual:.0f} mm/yr</td></tr>
    <tr><td><b>Domain</b></td><td>{ny} × {nx} cells, {n_wells} observation wells</td></tr>
    <tr><td><b>Methods compared</b></td><td>{methods_csv}</td></tr>
    <tr><td><b>Observation period</b></td><td>{n_days} days</td></tr>
  </table>
</div>

<div class="disclaimer">
  <b>Notes on interpretation.</b> This report applies <i>field-mode consistency
  metrics</i> — internal coherence checks that do not require ground-truth
  recharge.  Values reflect defensibility under stated assumptions, not validation
  against true recharge (which is generally unknown for real watersheds).
</div>

<h2>1. Method comparison maps</h2>
<div class="figure">
  <img src="data:image/png;base64,{img_comparison}" alt="Method comparison maps">
  <div class="caption">
    Figure 1. Estimated annual recharge from each method on a shared color
    scale.  Triangles mark observation wells.
  </div>
</div>

<h2>2. Method-to-method spread (epistemic uncertainty)</h2>
{spread_section}

<h2>3. Physical plausibility</h2>
<div class="figure">
  <img src="data:image/png;base64,{img_histogram}" alt="Recharge histograms">
  <div class="caption">
    Figure 3. Recharge value distributions per method.  Reference lines:
    2%, 30%, 50% of annual precipitation (typical bounds in humid climates,
    Healy 2010).
  </div>
</div>
{plausibility_table}

<h2>4. Soil-class coherence</h2>
<div class="figure">
  <img src="data:image/png;base64,{img_boxplot}" alt="Soil-class boxplots">
  <div class="caption">
    Figure 4. Recharge distribution by soil texture class for each method.
    Dashed line: class mean.  Tight, well-separated boxes indicate that the
    method respects soil-driven heterogeneity.
  </div>
</div>
{coherence_table}

<h2>5. Well-level consistency</h2>
<div class="figure">
  <img src="data:image/png;base64,{img_wellscatter}" alt="Well consistency scatter">
  <div class="caption">
    Figure 5. Estimated recharge at well-containing cells vs an
    observation-driven proxy (max-dh × Sy in 5-day windows).  The proxy is
    biased high (no recession correction) and serves only as a coarse sanity
    check.  Markers colored by soil class.
  </div>
</div>

<h2>6. Text summary</h2>
<pre class="summary-pre">{text_summary}</pre>

<div class="meta" style="margin-top:48px;">
  Report produced by hybrid-recharge field_report.py.  See evaluation/field_metrics.py
  for metric definitions.  References: Healy (2010), Scanlon et al. (2002),
  Beven (2006).
</div>
</body>
</html>
"""


def _format_plausibility_table(method_results, P_annual_mm) -> str:
    rows = []
    for name, R in method_results.items():
        rep = plausibility_check(R, P_annual_mm, method_name=name)
        if rep.pass_basic and not rep.flags:
            status = '<span class="flag-pass">✓ PASS</span>'
        elif rep.pass_basic:
            status = '<span class="flag-warn">△ WARN</span>'
        else:
            status = '<span class="flag-fail">✗ FAIL</span>'
        rows.append(
            f"<tr><td>{escape(name)}</td>"
            f"<td>{rep.mean_R:.1f}</td>"
            f"<td>{rep.R_over_P*100:.1f}%</td>"
            f"<td>{rep.min_R:.1f} – {rep.max_R:.1f}</td>"
            f"<td>{rep.n_negative}</td>"
            f"<td>{rep.n_above_precip}</td>"
            f"<td>{status}</td></tr>"
        )
    return (
        '<table class="metrics">'
        '<tr><th>Method</th><th>Mean R<br/>(mm/yr)</th><th>R/P</th>'
        '<th>Range</th><th>R&lt;0</th><th>R&gt;P</th><th>Status</th></tr>'
        + "".join(rows) + "</table>"
    )


def _format_coherence_table(method_results, soil_map) -> str:
    rows = []
    for name, R in method_results.items():
        coh = soil_class_coherence(R, soil_map, method_name=name)
        rows.append(
            f"<tr><td>{escape(name)}</td>"
            f"<td>{coh.coherence_ratio:.3f}</td>"
            f"<td>{coh.n_classes}</td>"
            f"<td>{coh.between_class_variance:.1f}</td>"
            f"<td>{coh.within_class_variance:.1f}</td></tr>"
        )
    return (
        '<table class="metrics">'
        '<tr><th>Method</th><th>Coherence ratio</th><th>Classes</th>'
        '<th>Between var</th><th>Within var</th></tr>'
        + "".join(rows) + "</table>"
    )


def _format_spread_section(spread: Optional[MethodSpread], img_b64: str) -> str:
    if spread is None:
        return (
            '<p><i>Spread analysis requires at least 2 methods. '
            'Only one method provided — section skipped.</i></p>'
        )
    method_means_html = "".join(
        f"<tr><td>{escape(n)}</td><td>{v:.1f}</td></tr>"
        for n, v in spread.method_domain_means.items()
    )
    return f"""
<div class="figure">
  <img src="data:image/png;base64,{img_b64}" alt="Method spread map">
  <div class="caption">
    Figure 2. (a) Cell-wise mean recharge across methods.  (b) Cell-wise
    standard deviation across methods (epistemic uncertainty proxy under
    method-choice ambiguity, Beven 2006).
  </div>
</div>
<table class="metrics">
  <tr><th>Quantity</th><th>Value</th></tr>
  <tr><td>Domain-mean recharge (across methods)</td>
      <td>{spread.domain_mean:.1f} mm/yr</td></tr>
  <tr><td>Domain-mean spread (std across methods)</td>
      <td>{spread.domain_mean_std:.1f} mm/yr</td></tr>
  <tr><td>Domain-mean CV across methods</td>
      <td>{spread.domain_mean_cv:.3f}</td></tr>
</table>
<h3>Per-method domain mean</h3>
<table class="metrics">
  <tr><th>Method</th><th>Mean R (mm/yr)</th></tr>
  {method_means_html}
</table>
"""


def build_html_report(
    method_results: Dict[str, np.ndarray],
    observations: Dict,
    domain,
    P_annual_mm: float,
    site_name: str = "Untitled site",
    output_path: Optional[str] = None,
    title: Optional[str] = None,
) -> str:
    """Build a self-contained HTML field-mode report.

    Parameters
    ----------
    method_results : dict
        {method_name: recharge_map (ny, nx) [mm/yr]}.
    observations : dict
        Must contain 'P', 'ho_obs', 'well_soil_types'.
    domain : SyntheticDomain
        Must expose well_rows, well_cols, x_centers, y_centers, soil_map,
        config (with nx, ny, dx, dy).
    P_annual_mm : float
        Annual precipitation [mm/yr].
    site_name : str
        Identifier for the site (appears in report header).
    output_path : str, optional
        If given, write HTML here.
    title : str, optional
        Report title.  Defaults to "Field-mode Recharge Report — <site>".

    Returns
    -------
    str
        The HTML content (also written to output_path if given).
    """
    if not method_results:
        raise ValueError("method_results is empty")
    if title is None:
        title = f"Field-mode Recharge Report — {site_name}"

    P = np.asarray(observations["P"])
    ho_obs = np.asarray(observations["ho_obs"])
    n_days = int(len(P))
    n_wells = int(ho_obs.shape[0])

    # 1. comparison maps
    fig1 = plot_method_comparison_maps(method_results, domain)
    img1 = _fig_to_base64(fig1)

    # 2. spread map (>= 2 methods)
    if len(method_results) >= 2:
        spread = between_method_spread(method_results)
        fig2 = plot_method_spread_map(spread, domain)
        img2 = _fig_to_base64(fig2)
    else:
        spread = None
        img2 = ""

    # 3. histograms
    fig3 = plot_recharge_histograms(method_results, P_annual_mm)
    img3 = _fig_to_base64(fig3)

    # 4. boxplots
    fig4 = plot_soil_class_boxplots(method_results, domain.soil_map)
    img4 = _fig_to_base64(fig4)

    # 5. well consistency scatter
    fig5 = plot_well_consistency_scatter(method_results, observations, domain)
    img5 = _fig_to_base64(fig5)

    # 텍스트 요약
    text = field_summary(method_results, observations, domain, P_annual_mm)

    html = _HTML_TEMPLATE.format(
        title=escape(title),
        site_name=escape(site_name),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
        P_annual=P_annual_mm,
        ny=domain.config.ny,
        nx=domain.config.nx,
        n_wells=n_wells,
        n_days=n_days,
        methods_csv=", ".join(escape(n) for n in method_results.keys()),
        img_comparison=img1,
        spread_section=_format_spread_section(spread, img2),
        img_histogram=img3,
        img_boxplot=img4,
        img_wellscatter=img5,
        plausibility_table=_format_plausibility_table(method_results, P_annual_mm),
        coherence_table=_format_coherence_table(method_results, domain.soil_map),
        text_summary=escape(text),
    )

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

    return html


# ══════════════════════════════════════════════════════════════════════
# 자체 시연 (모듈을 직접 실행 시)
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from synthetic.generate_domain import generate_domain, DomainConfig
    from synthetic.generate_data import generate_data
    from methods.wtf_lumped import estimate_recharge as est_lumped
    from methods.wtf_soil_weighted import estimate_recharge as est_soil

    print("=== field_report.py demo (S3 scenario) ===\n")
    domain = generate_domain(DomainConfig.S3())
    data = generate_data(domain)
    observations = {
        "P": data.P, "ET": data.ET,
        "ho_obs": data.ho_obs, "well_soil_types": data.well_soil_types,
    }
    results = {
        "Lumped": est_lumped(domain, observations),
        "Soil-weighted": est_soil(domain, observations),
    }
    P_annual = float(np.sum(data.P)) * 1000.0 / (data.n_days / 365.0)

    out = "/tmp/field_report_S3.html"
    build_html_report(
        results, observations, domain,
        P_annual_mm=P_annual,
        site_name="Synthetic-S3",
        output_path=out,
    )
    print(f"Report written: {out}")
    print(f"Open in browser, then File → Print → Save as PDF.")
