"""fig12_proxy_figure.py — Figure 12: Multi-proxy bracketing with α-spectrum curves.

Independent multi-proxy consistency check for the Yeongcheon field application.
Shows BFI / CMB / MOLIT proxy bands, their union envelope, α-spectrum curves
for two Yeongcheon alluvial watersheds, and the UZF lower bound.

Corrected α=0.3 values (2026-05-06):
  Jaho-cheon      : SW=9.33%, α=0.3 → 10.30%, α=1.0 → 13.58%
  Geumho-gang up. : SW=9.34%, α=0.3 → 10.36%, α=1.0 → 13.88%

Run:
  /opt/anaconda3/envs/myenv/bin/python evaluation/fig12_proxy_figure.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Hard-coded watershed values
# (derived from case_yeongcheon.json with corrected bias factors)
# R(α) = SW / (1 + α * β)
# ---------------------------------------------------------------------------
WATERSHEDS = [
    {
        "name": "Jaho-cheon",
        "label": "Jaho-cheon",
        "SW": 9.329,      # α=0 value (soil-weighted, % of P)
        "beta": -0.321,   # bias factor from case_yeongcheon.json (ET/P=0.648)
        "color": "#DC2626",
        "marker": "o",
    },
    {
        "name": "Geumho-gang upper",
        "label": "Geumho-gang upper",
        "SW": 9.344,      # α=0 value (soil-weighted, % of P)
        "beta": -0.338,   # bias factor from case_yeongcheon.json (ET/P=0.648)
        "color": "#2563EB",
        "marker": "s",
    },
]

# Proxy bands (% of P)
BFI_LO, BFI_HI   = 30.0, 45.0    # Lyne-Hollick BFI, Korean montane (KICT 2018)
CMB_LO, CMB_HI   = 12.0, 24.0    # Cl mass balance (Park et al. 2015)
MOLIT_LO, MOLIT_HI = 13.5, 22.9  # MOLIT 2016 atlas, Yeongcheon region

UNION_LO = min(BFI_LO, CMB_LO, MOLIT_LO)   # 12.0
UNION_HI = max(BFI_HI, CMB_HI, MOLIT_HI)   # 45.0

UZF_LOWER = 7.2     # UZF kinematic-wave (gravity-only) lower bound, Loam (§6.6)
RICHARDS_UPPER = 20.6  # HYDRUS-class Richards (capillary+gravity) upper bound, Loam (§6.6)
CASCADE_TRUTH = 24.6   # Cascade-class synthetic truth (Loam) — outside bracket

ALPHA_MARKS = [0.0, 0.3, 0.5, 1.0]
ALPHA_MARK_SYMBOLS = ["o", "s", "D", "*"]
ALPHA_MARK_LABELS  = ["α=0\n(uncorrected)", "α=0.3\n(recommended)",
                      "α=0.5", "α=1.0\n(upper bound)"]


def R_alpha(SW: float, beta: float, alphas: np.ndarray) -> np.ndarray:
    return SW / np.maximum(1.0 + alphas * beta, 0.05)


def main(save_path: str = "paper/fig12_proxy.png", dpi: int = 200):
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from figure_style import apply_publication_style
    apply_publication_style()

    alphas = np.linspace(0.0, 1.0, 201)

    fig, ax = plt.subplots(figsize=(11.0, 6.0))

    # ---- Proxy bands (back to front) ----------------------------------------
    ax.axhspan(UNION_LO, UNION_HI, color="#10B981", alpha=0.06, zorder=1,
               label=f"Multi-proxy union ({UNION_LO:.0f}–{UNION_HI:.0f}% of P)")
    ax.axhspan(BFI_LO, BFI_HI, color="#0891B2", alpha=0.12, zorder=2,
               label=f"BFI proxy ({BFI_LO:.0f}–{BFI_HI:.0f}% of P)")
    ax.axhspan(CMB_LO, CMB_HI, color="#7C3AED", alpha=0.15, zorder=3,
               label=f"CMB literature ({CMB_LO:.0f}–{CMB_HI:.0f}% of P)")
    ax.axhspan(MOLIT_LO, MOLIT_HI, color="#F59E0B", alpha=0.20, zorder=4,
               label=f"MOLIT atlas ({MOLIT_LO:.1f}–{MOLIT_HI:.1f}% of P)")

    # ---- Physical bracket: UZF lower / Richards upper -----------------------
    # Light fill across the UZF–Richards envelope
    ax.axhspan(UZF_LOWER, RICHARDS_UPPER, color="#1A7E43", alpha=0.04, zorder=0)
    ax.axhline(UZF_LOWER, color="#1A7E43", linestyle="--", linewidth=1.5,
               zorder=5,
               label=f"UZF lower bound ({UZF_LOWER}%, gravity-only)")
    ax.axhline(RICHARDS_UPPER, color="#7B3F99", linestyle="--", linewidth=1.5,
               zorder=5,
               label=f"Richards upper bound ({RICHARDS_UPPER}%, mixed-form)")
    ax.axhline(CASCADE_TRUTH, color="#B91C1C", linestyle=":", linewidth=1.4,
               zorder=5,
               label=f"Cascade truth ({CASCADE_TRUTH}%, outside bracket)")

    # ---- α-spectrum curves --------------------------------------------------
    # Offset annotations in opposite directions per watershed to avoid overlap
    # (Jaho-cheon and Geumho-gang have near-identical values).
    for idx, ws in enumerate(WATERSHEDS):
        Rv = R_alpha(ws["SW"], ws["beta"], alphas)
        ax.plot(alphas, Rv, "-", color=ws["color"], linewidth=2.2,
                label=ws["label"], zorder=6)

        # Markers at specific α values
        for a_mark, sym in zip(ALPHA_MARKS, ALPHA_MARK_SYMBOLS):
            R_mark = R_alpha(ws["SW"], ws["beta"], np.array([a_mark]))[0]
            ms = 12 if sym == "*" else 8
            ax.plot(a_mark, R_mark, sym, color=ws["color"],
                    markersize=ms, markeredgecolor="black",
                    markeredgewidth=0.7, zorder=7)
            # Watershed 0: label above-left; Watershed 1: label below-right.
            if idx == 0:
                xytext_pt = (-6, 8)
                ha, va = "right", "bottom"
            else:
                xytext_pt = (6, -10)
                ha, va = "left", "top"
            ax.annotate(
                f"{R_mark:.2f}%",
                xy=(a_mark, R_mark),
                xytext=xytext_pt, textcoords="offset points",
                fontsize=8, color=ws["color"],
                ha=ha, va=va,
            )

    # ---- α vertical guide lines --------------------------------------------
    for a_mark, lbl in zip(ALPHA_MARKS, ALPHA_MARK_LABELS):
        ax.axvline(a_mark, color="gray", linestyle=":", linewidth=0.9,
                   alpha=0.6, zorder=2)
        ax.text(a_mark, -2.8, lbl, ha="center", fontsize=8.5,
                color="dimgray",
                fontweight="bold" if a_mark == 0.3 else "normal")

    # ---- Axes / labels ------------------------------------------------------
    ax.set_xlabel("Conservatism parameter α", fontsize=12)
    ax.set_ylabel("Recharge ratio (% of P)", fontsize=12)
    ax.set_title(
        "Independent multi-proxy consistency check\n"
        "(Yeongcheon alluvial watersheds vs. BFI / CMB / MOLIT proxies "
        "+ UZF–Richards physical bracket)",
        fontweight="bold",
    )
    ax.set_xlim(-0.04, 1.08)
    ax.set_ylim(-5, 55)
    ax.grid(linestyle="--", alpha=0.3)
    # Place legend outside on the right so it does not occlude curves/bands
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=9, framealpha=0.95, ncol=1, borderaxespad=0.0)

    plt.tight_layout()
    # Reserve room for the right-hand legend
    plt.subplots_adjust(right=0.70, bottom=0.16)

    # ---- Save ---------------------------------------------------------------
    out_dir = os.path.dirname(os.path.abspath(save_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {save_path} (dpi={dpi})")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/fig12_proxy.png")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()
    main(save_path=args.out, dpi=args.dpi)
