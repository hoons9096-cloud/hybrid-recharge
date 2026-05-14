"""benchmark_figures.py — Phase 4 논문용 그림 (Figure 3~5).

생성 그림:
    Figure 3 — RMSE bar chart  (S1~S5 × 3 methods × 2 truth)
    Figure 4 — Scatter plot    (true vs estimated, 방법별)
    Figure 5 — Spatial map     (S3 시나리오 — true / Lumped / Soil-weighted / EnKF)
    Figure 6 — Computation time (방법별 평균 + 표준편차)

Usage
-----
    python -m evaluation.benchmark_figures \
        --csv benchmark_results.csv \
        --outdir paper_figures/
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# 프로젝트 루트
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 논문용 폰트/스타일 — unified rcParams
from evaluation.figure_style import apply_publication_style
apply_publication_style()

# 방법별 색상 (color-blind friendly)
METHOD_COLORS = {
    "Lumped":        "#9CA3AF",   # 회색 — baseline
    "Soil-weighted": "#DC2626",   # 빨강 — 제안 방법
    "Hierarchical":  "#7C3AED",   # 보라 — Phase 3
    "EnKF":          "#2563EB",   # 파랑 — 비교
}
METHOD_MARKERS = {"Lumped": "o", "Soil-weighted": "s",
                  "Hierarchical": "D", "EnKF": "^"}


# ---------------------------------------------------------------------------
# Figure 3 — RMSE bar chart
# ---------------------------------------------------------------------------
def fig3_rmse_bars(df: pd.DataFrame, save_path: Optional[str] = None):
    truths = sorted(df["truth_model"].unique())
    fig, axes = plt.subplots(1, len(truths), figsize=(11, 4.5),
                             sharey=False)
    if len(truths) == 1:
        axes = [axes]
    methods = ["Lumped", "Soil-weighted", "Hierarchical", "EnKF"]
    scenarios = sorted(df["scenario"].unique())
    n_m = len(methods); width = 0.25
    x = np.arange(len(scenarios))

    panel_letters = ["(a)", "(b)", "(c)", "(d)"]
    for p, (ax, truth) in enumerate(zip(axes, truths)):
        sub = df[df["truth_model"] == truth]
        for i, m in enumerate(methods):
            vals = [
                float(sub[(sub.scenario == s) & (sub.method == m)]["rmse_grid"].iloc[0])
                if not sub[(sub.scenario == s) & (sub.method == m)].empty else 0.0
                for s in scenarios
            ]
            bars = ax.bar(x + (i - 1) * width, vals, width=width,
                          label=m, color=METHOD_COLORS[m],
                          edgecolor="black", linewidth=0.4)
            # Numeric labels above each bar for direct readability
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height(),
                            f"{v:.0f}",
                            ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(scenarios)
        ax.set_xlabel("Scenario")
        ax.set_ylabel("RMSE (mm/yr)")
        ax.set_title(f"{panel_letters[p]} Truth model: {truth}",
                     loc="left", fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(loc="upper left", framealpha=0.9)
    # Caption-side note: panel y-axis scales differ
    fig.text(0.5, -0.02,
             "Note: y-axis scales differ between panels.",
             ha="center", fontsize=8.5, style="italic", color="dimgray")

    # (Figure number applied by paper builder; no in-figure title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  ✓ {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Figure 4 — Scatter (true vs estimated, 시나리오별)
# ---------------------------------------------------------------------------
def fig4_scatter(
    truth_model: str = "alpha",
    scenarios: List[str] = None,
    save_path: Optional[str] = None,
):
    """True vs estimated 산포도. 각 셀의 R 추정/정답 비교.

    benchmark CSV 만으로는 grid 레벨 R 이 없으니 직접 재실행.
    """
    if scenarios is None:
        scenarios = ["S2", "S3", "S5"]
    from synthetic.scenarios import _CONFIG_FACTORY
    from synthetic.generate_domain import generate_domain
    from synthetic.generate_data import generate_data
    from evaluation.benchmark_matrix import _run_single_method

    methods = ["Lumped", "Soil-weighted", "Hierarchical", "EnKF"]
    fig, axes = plt.subplots(1, len(scenarios),
                             figsize=(4.0 * len(scenarios), 4.0),
                             sharex=True, sharey=True)
    if len(scenarios) == 1:
        axes = [axes]

    for ax, scn in zip(axes, scenarios):
        cfg = _CONFIG_FACTORY[scn]()
        dom = generate_domain(cfg)
        data = generate_data(dom, n_days=730, recharge_model=truth_model)
        R_true = data.true_recharge_annual.flatten()

        v_max = float(np.max(R_true)) * 1.1
        ax.plot([0, v_max], [0, v_max], "k--", linewidth=1, alpha=0.6, zorder=0)

        for m in methods:
            R_est, _, _ = _run_single_method(m, dom, data)
            if R_est is None:
                continue
            R_est_flat = R_est.flatten()
            # 다운샘플 (시각적 잡음 줄이기)
            n_show = min(2000, len(R_true))
            idx = np.random.default_rng(0).choice(len(R_true), n_show,
                                                   replace=False)
            ax.scatter(R_true[idx], R_est_flat[idx], s=8, alpha=0.5,
                       color=METHOD_COLORS[m], marker=METHOD_MARKERS[m],
                       label=m, edgecolors="none")

        ax.set_xlabel("True recharge (mm/yr)")
        ax.set_ylabel("Estimated recharge (mm/yr)")
        ax.set_title(f"{scn}")
        ax.set_xlim(0, v_max); ax.set_ylim(0, v_max)
        ax.grid(linestyle="--", alpha=0.3)
        ax.legend(loc="upper left", markerscale=2.0, framealpha=0.9)
        ax.set_aspect("equal", adjustable="box")

    # (Figure number applied by paper builder)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  ✓ {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Figure 5 — Spatial map (S3, 4 panels: true / 3 methods)
# ---------------------------------------------------------------------------
def fig5_spatial_maps(
    scenario: str = "S3",
    truth_model: str = "alpha",
    save_path: Optional[str] = None,
):
    from synthetic.scenarios import _CONFIG_FACTORY
    from synthetic.generate_domain import generate_domain
    from synthetic.generate_data import generate_data
    from evaluation.benchmark_matrix import _run_single_method

    cfg = _CONFIG_FACTORY[scenario]()
    dom = generate_domain(cfg)
    data = generate_data(dom, n_days=730, recharge_model=truth_model)
    R_true = data.true_recharge_annual

    methods = ["Lumped", "Soil-weighted", "Hierarchical", "EnKF"]
    R_estimates = {}
    for m in methods:
        R_est, _, err = _run_single_method(m, dom, data)
        R_estimates[m] = R_est

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2), sharey=True)
    panels = [
        ("True", R_true),
        ("Lumped", R_estimates["Lumped"]),
        ("Soil-weighted", R_estimates["Soil-weighted"]),
        ("Hierarchical", R_estimates["Hierarchical"]),
        ("EnKF", R_estimates["EnKF"]),
    ]
    vmin = float(np.min(R_true)); vmax = float(np.max(R_true))

    for ax, (title, arr) in zip(axes, panels):
        if arr is None:
            ax.text(0.5, 0.5, "FAILED", transform=ax.transAxes, ha="center")
            ax.set_title(title)
            continue
        im = ax.imshow(arr, vmin=vmin, vmax=vmax, cmap="viridis", origin="upper")
        # Wells
        ax.scatter(dom.well_cols, dom.well_rows, s=18,
                   facecolor="white", edgecolor="black", linewidth=0.7)
        ax.set_title(title)
        ax.set_xlabel("Column");
        if ax is axes[0]:
            ax.set_ylabel("Row")
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, aspect=30, pad=0.02)
    cbar.set_label("Annual recharge (mm/yr)")
    # (Figure number applied by paper builder)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  ✓ {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Figure 6 — Computation time
# ---------------------------------------------------------------------------
def fig6_compute_time(df: pd.DataFrame, save_path: Optional[str] = None):
    methods = [m for m in ["Lumped", "Soil-weighted", "Hierarchical", "EnKF"]
               if m in df["method"].unique()]
    fig, ax = plt.subplots(figsize=(6, 4))
    means, sds = [], []
    for m in methods:
        vals = df[df.method == m]["elapsed_sec"].values
        means.append(np.mean(vals))
        sds.append(np.std(vals))
    bars = ax.bar(methods, means, yerr=sds, capsize=5,
                  color=[METHOD_COLORS[m] for m in methods],
                  edgecolor="black", linewidth=0.5)
    for bar, m in zip(bars, means):
        ax.annotate(f"{m:.3f}s",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Computation time (s)")
    # (Figure number applied by paper builder)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_yscale("log")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  ✓ {save_path}")
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Phase 4 paper figures")
    ap.add_argument("--csv", default="benchmark_results.csv")
    ap.add_argument("--outdir", default="paper_figures")
    ap.add_argument("--scenario_for_map", default="S3")
    ap.add_argument("--truth_for_map", default="alpha")
    ap.add_argument("--no_show", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        print(f"❌ CSV not found: {args.csv}")
        print(f"   먼저 benchmark_matrix.py 실행하세요.")
        sys.exit(1)

    df = pd.read_csv(args.csv)
    os.makedirs(args.outdir, exist_ok=True)
    print(f"📊 Generating figures from {args.csv} → {args.outdir}/")

    fig3_rmse_bars(df, save_path=os.path.join(args.outdir, "fig3_rmse.png"))
    fig4_scatter(
        truth_model=args.truth_for_map,
        scenarios=["S2", "S3", "S5"],
        save_path=os.path.join(args.outdir, "fig4_scatter.png"),
    )
    fig5_spatial_maps(
        scenario=args.scenario_for_map,
        truth_model=args.truth_for_map,
        save_path=os.path.join(args.outdir, "fig5_spatial.png"),
    )
    fig6_compute_time(df, save_path=os.path.join(args.outdir, "fig6_time.png"))
    print(f"\n✓ 4개 그림 생성 완료: {args.outdir}/")


if __name__ == "__main__":
    main()
