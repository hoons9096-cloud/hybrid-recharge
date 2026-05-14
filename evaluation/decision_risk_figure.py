"""decision_risk_figure.py — Figure 10: Decision-theoretic value of CI."""
from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.figure_style import apply_publication_style
apply_publication_style()


def fig10_decision_risk(samples_npz: str, save_path: str):
    """3 panels:
       (a) probability of recharge < threshold (decision risk)
       (b) point-estimate vs CI band
       (c) operational decision matrix (under-allocation risk)
    """
    data = np.load(samples_npz)
    posterior = data["posterior"]   # (R, S)
    truths = data["truth"]
    R = posterior.shape[0]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))

    # (a) Probability of exceedance
    ax = axes[0]
    thresholds = np.linspace(0, max(posterior.max(), truths.max()) * 1.0, 60)
    p_below = np.zeros((R, len(thresholds)))
    for i in range(R):
        for k, t in enumerate(thresholds):
            p_below[i, k] = float(np.mean(posterior[i] < t))
    # Plot mean ± SD across replicates
    p_mean = p_below.mean(axis=0)
    p_sd = p_below.std(axis=0)
    ax.plot(thresholds, p_mean, color="#7C3AED", linewidth=2,
            label="Mean P(R < threshold)")
    ax.fill_between(thresholds, p_mean - p_sd, p_mean + p_sd,
                    color="#7C3AED", alpha=0.2, label="±1 SD")
    truth_mean = truths.mean()
    ax.axvline(truth_mean, color="black", linestyle=":",
               linewidth=1.0, label=f"Mean truth ({truth_mean:.1f}%)")
    ax.set_xlabel("Recharge threshold (% of P)")
    ax.set_ylabel("P(R_watershed < threshold)")
    ax.set_title("(a) Decision risk curve")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    # (b) Point estimate vs full distribution (single replicate)
    ax = axes[1]
    rep_idx = R // 2
    ax.hist(posterior[rep_idx], bins=40, color="#7C3AED", alpha=0.6,
            edgecolor="black", linewidth=0.4, density=True)
    pmean = posterior[rep_idx].mean()
    plo = np.percentile(posterior[rep_idx], 2.5)
    phi = np.percentile(posterior[rep_idx], 97.5)
    ax.axvline(pmean, color="#DC2626", linewidth=2,
               label=f"Point estimate ({pmean:.1f}%)")
    ax.axvspan(plo, phi, color="#DC2626", alpha=0.15,
               label=f"95% CI [{plo:.1f}, {phi:.1f}]")
    ax.axvline(truths[rep_idx], color="black", linestyle="--",
               linewidth=1.5, label=f"Truth ({truths[rep_idx]:.1f}%)")
    ax.set_xlabel("Recharge (% of P)")
    ax.set_ylabel("Density")
    ax.set_title(f"(b) Posterior vs. point estimate (rep {rep_idx})")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # (c) Decision matrix: under-allocation risk by threshold
    ax = axes[2]
    bands = [5.0, 10.0, 15.0, 20.0, 25.0]
    risk_point = []
    risk_lower = []      # P(under-allocation given 95% lower bound)
    for t in bands:
        # Point estimate ≥ t? — only single number, no probability
        # Lower 95% bound — conservative decision
        lo_bound = np.percentile(posterior, 5, axis=1)
        risk_point.append(float(np.mean(posterior.mean(axis=1) >= t)))
        risk_lower.append(float(np.mean(lo_bound >= t)))
    x = np.arange(len(bands))
    width = 0.35
    ax.bar(x - width / 2, risk_point, width=width,
           color="#DC2626", edgecolor="black", linewidth=0.4,
           label="Point estimate ≥ threshold")
    ax.bar(x + width / 2, risk_lower, width=width,
           color="#7C3AED", edgecolor="black", linewidth=0.4,
           label="95% lower CI ≥ threshold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.0f}%" for t in bands])
    ax.set_xlabel("Allocation threshold")
    ax.set_ylabel("Fraction of replicates passing decision")
    ax.set_title("(c) Conservative vs. point allocation decisions")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0, 1.05)

    # (Figure number applied by paper builder)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  ✓ {save_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_npz", required=True)
    ap.add_argument("--out", default="fig10_decision.png")
    args = ap.parse_args()
    fig10_decision_risk(args.samples_npz, args.out)
