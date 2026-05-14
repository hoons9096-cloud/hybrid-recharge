"""DEPRECATED — use fig13_decision_yeongcheon.py instead.

This original generator depended on case_gimcheon.run_all() and produced
fig13_decision.png with stale β values from the ET/P=0.5 calibration
(α=1 markers showed 22.5%/12.8%, contradicting the paper's corrected
values from ET/P=0.648). The replacement script reads the corrected
WATERSHEDS dict from fig12_proxy_figure.py and matches Table 8 exactly.

decision_envelope_figure.py — Figure 13 (legacy α-spectrum chart).
α-spectrum vs. recharge ratio overlaid with multi-proxy envelope.
A single decision-support plot that combines the conservatism parameter,
field results, and independent proxy bands into one operational chart.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for f in ["AppleGothic", "NanumGothic", "Malgun Gothic"]:
    if f in [x.name for x in matplotlib.font_manager.fontManager.ttflist]:
        matplotlib.rcParams["font.family"] = f
        break
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams.update({"font.size": 11, "figure.dpi": 300})


def fig13_decision_envelope(
    cases, save_path: str,
    alphas: np.ndarray = None,
):
    """α-spectrum decision plot — final operational figure.

    Each watershed → curve of recharge(α) for α ∈ [0, 1]
    Background bands: BFI / CMB / MOLIT proxies + union envelope.
    Annotates α = 0 / 0.3 / 1.0 marker points on each curve.
    """
    from evaluation.proxy_validation import proxy_envelope

    if alphas is None:
        alphas = np.linspace(0.0, 1.0, 51)
    valid = [c for c in cases if c.lumped_pct is not None
             and c.bias_factor_mean is not None]
    if not valid:
        print("⚠️ no valid cases")
        return

    name_en = {"감천": "Gamcheon", "감천상류": "Gam.-upper",
               "감천중류": "Gam.-middle", "부항천": "Buhang-cheon"}

    # 한국 climate normal P=1100 mm 기준 envelope
    env = proxy_envelope(P_annual_mm=1100.0)

    fig, ax = plt.subplots(figsize=(11, 6.5))

    # Proxy bands (background)
    ax.axhspan(env.molit_lo, env.molit_hi, color="#F59E0B", alpha=0.25,
               label=f"MOLIT atlas ({env.molit_lo:.0f}–{env.molit_hi:.0f}%)")
    ax.axhspan(env.cmb_lo, env.cmb_hi, color="#7C3AED", alpha=0.18,
               label=f"CMB lit. ({env.cmb_lo:.0f}–{env.cmb_hi:.0f}%)")
    ax.axhspan(env.bfi_lo, env.bfi_hi, color="#0891B2", alpha=0.12,
               label=f"BFI proxy ({env.bfi_lo:.0f}–{env.bfi_hi:.0f}%)")
    # Envelope (full union) - light green outer band
    ax.axhspan(env.envelope_lo, env.envelope_hi, color="#10B981",
               alpha=0.05, zorder=0,
               label=f"Multi-proxy envelope ({env.envelope_lo:.0f}–{env.envelope_hi:.0f}%)")

    # Watershed curves
    colors = {"Gamcheon": "#DC2626", "Gam.-upper": "#0891B2",
              "Gam.-middle": "#7C3AED", "Buhang-cheon": "#374151"}
    markers = {"Gamcheon": "o", "Gam.-upper": "s",
               "Gam.-middle": "D", "Buhang-cheon": "^"}

    for c in valid:
        label = name_en.get(c.name, c.name)
        sw_pct = c.soil_weighted_pct
        beta = c.bias_factor_mean
        # R(α) = SW / (1 + α β)
        R_alpha = np.array([sw_pct / max(1.0 + a * beta, 0.05) for a in alphas])
        ax.plot(alphas, R_alpha, "-",
                color=colors.get(label, "black"),
                linewidth=2.0, label=label, zorder=3)
        # Mark α = 0, 0.3, 1.0
        for a_mark, sym in zip([0.0, 0.3, 1.0], ["o", "s", "*"]):
            R_mark = sw_pct / max(1.0 + a_mark * beta, 0.05)
            ax.plot(a_mark, R_mark, sym, color=colors.get(label, "black"),
                    markersize=10 if sym == "*" else 7,
                    markeredgecolor="black", markeredgewidth=0.7,
                    zorder=4)
            if sym == "*":
                ax.annotate(f"{R_mark:.1f}",
                            (a_mark, R_mark),
                            xytext=(8, -2), textcoords="offset points",
                            fontsize=8.5, color=colors.get(label, "black"))

    # Recommended operational region (within envelope)
    ax.axvspan(0.0, 0.5, color="#10B981", alpha=0.02, zorder=0)
    ax.text(0.25, 47, "operationally\ndefensible α range",
            ha="center", fontsize=9, color="#065F46",
            style="italic", alpha=0.7)

    # Annotations
    ax.axvline(0.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.axvline(0.3, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.axvline(1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.text(0.0, -2.5, "α=0\n(no corr.)", ha="center", fontsize=8.5,
            color="dimgray")
    ax.text(0.3, -2.5, "α=0.3\n(default)", ha="center", fontsize=8.5,
            color="dimgray", weight="bold")
    ax.text(1.0, -2.5, "α=1.0\n(upper bound)", ha="center", fontsize=8.5,
            color="dimgray")

    ax.set_xlabel("Conservatism parameter α")
    ax.set_ylabel("Watershed-mean recharge ratio (% of P)")
    ax.set_title(
        "α-spectrum decision chart: "
        "field-scale recharge envelope vs. independent proxies",
        fontweight="bold",
    )
    ax.set_xlim(-0.05, 1.08)
    ax.set_ylim(-5, 50)
    ax.grid(linestyle="--", alpha=0.3)
    ax.legend(loc="upper left", fontsize=8.5, ncol=2, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {save_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fig13_decision.png")
    args = ap.parse_args()

    # Re-run case_gimcheon to obtain cases
    from evaluation.case_gimcheon import run_all
    cases = run_all(verbose=False)
    fig13_decision_envelope(cases, args.out)
