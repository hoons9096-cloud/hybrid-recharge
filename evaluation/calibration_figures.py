"""calibration_figures.py — Figure 8 (PIT/coverage) + Figure 9 (bias)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
from evaluation.figure_style import apply_publication_style
apply_publication_style()


def fig8_calibration(
    cal_json: str, samples_npz: str, save_path: str,
):
    """PIT histogram + reliability diagram."""
    with open(cal_json) as f:
        cal = json.load(f)
    data = np.load(samples_npz)
    posterior = data["posterior"]
    truths = data["truth"]
    pit = np.array(cal["pit_values"])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))

    # (a) PIT histogram
    ax = axes[0]
    ax.hist(pit, bins=10, range=(0, 1), color="#7C3AED",
            edgecolor="black", linewidth=0.5)
    ax.axhline(y=len(pit) / 10, color="black", linestyle="--",
               linewidth=1.2, label="Uniform (ideal)")
    ax.set_xlabel("PIT value"); ax.set_ylabel("Count")
    ax.set_title(f"(a) PIT histogram (KS p={cal['pit_ks_pvalue']:.3f})")
    ax.legend(loc="upper right")

    # (b) Reliability — coverage 곡선
    ax = axes[1]
    levels = np.linspace(0.05, 0.95, 19)
    cov = []
    for lv in levels:
        lo = (1 - lv) / 2
        hi = 1 - lo
        lo_q = np.quantile(posterior, lo, axis=1)
        hi_q = np.quantile(posterior, hi, axis=1)
        cov.append(float(np.mean((truths >= lo_q) & (truths <= hi_q))))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Ideal")
    ax.plot(levels, cov, "-o", color="#DC2626", markersize=4,
            label="Empirical")
    ax.set_xlabel("Nominal level"); ax.set_ylabel("Empirical coverage")
    ax.set_title("(b) Coverage probability")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # (c) Posterior vs truth scatter
    ax = axes[2]
    post_mean = posterior.mean(axis=1)
    post_lo = np.quantile(posterior, 0.025, axis=1)
    post_hi = np.quantile(posterior, 0.975, axis=1)
    yerr = np.array([post_mean - post_lo, post_hi - post_mean])
    v_max = max(float(np.max(post_hi)), float(np.max(truths))) * 1.1
    ax.errorbar(truths, post_mean, yerr=yerr, fmt="o",
                color="#7C3AED", ecolor="gray",
                markeredgecolor="black", markersize=5,
                capsize=3, linewidth=0.6, label="Posterior 95% CI")
    ax.plot([0, v_max], [0, v_max], "k--", alpha=0.6, label="1:1")
    ax.set_xlabel("True recharge (% of P)")
    ax.set_ylabel("Posterior recharge (% of P)")
    ax.set_title("(c) Posterior vs. true")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, v_max); ax.set_ylim(0, v_max)

    # (Figure number applied by paper builder)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  ✓ {save_path}")


def fig9_bias(bias_json: str, save_path: str):
    """Bias bar chart by scenario × truth model."""
    with open(bias_json) as f:
        b = json.load(f)
    rows = b["rows"]
    if not rows:
        return
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    scenarios = sorted(df.scenario.unique())
    truths = sorted(df.truth_model.unique())
    width = 0.35
    x = np.arange(len(scenarios))
    colors = {"alpha": "#9CA3AF", "cascade": "#7C3AED"}

    for i, tm in enumerate(truths):
        sub = df[df.truth_model == tm].set_index("scenario").reindex(scenarios)
        means = sub["mean_bias_mm"].values
        sds = sub["bias_sd_mm"].values
        ax.bar(x + (i - 0.5) * width, means, width=width, yerr=sds,
               label=f"truth = {tm}", color=colors.get(tm, "gray"),
               edgecolor="black", linewidth=0.5, capsize=4)

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Bias = $\\hat R_\\mathrm{est} - R_\\mathrm{true}$ (mm/yr)")
    ax.set_title(
        "WTF inherent bias quantification "
        "(Soil-weighted estimator, N=10 replicates per scenario)",
        fontweight="bold",
    )
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  ✓ {save_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal_json", required=True)
    ap.add_argument("--samples_npz", required=True)
    ap.add_argument("--bias_json", required=True)
    ap.add_argument("--outdir", default="paper_figures")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    fig8_calibration(
        args.cal_json, args.samples_npz,
        os.path.join(args.outdir, "fig8_calibration.png"),
    )
    fig9_bias(
        args.bias_json,
        os.path.join(args.outdir, "fig9_bias.png"),
    )
