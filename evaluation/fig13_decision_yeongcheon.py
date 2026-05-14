"""fig13_decision_yeongcheon.py — Figure 7 (fig13_decision.png)

α-spectrum decision chart for Yeongcheon alluvial watersheds with the
multi-proxy envelope (BFI / CMB / MOLIT) as background bands. Uses the
corrected β values consistent with §4.5 Table 8 and §4.6.

Run:
  /opt/anaconda3/envs/myenv/bin/python evaluation/fig13_decision_yeongcheon.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse hardcoded values from fig12 source-of-truth
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig12_proxy_figure import (
    WATERSHEDS, R_alpha,
    BFI_LO, BFI_HI, CMB_LO, CMB_HI, MOLIT_LO, MOLIT_HI,
    UNION_LO, UNION_HI,
    ALPHA_MARKS, ALPHA_MARK_SYMBOLS,
)


def main(save_path: str = "paper/fig13_decision.png", dpi: int = 200):
    from figure_style import apply_publication_style
    apply_publication_style()
    alphas = np.linspace(0.0, 1.0, 201)

    # Schematic-style: stripped-down decision artefact (curves + bands + α
    # recommendation only); detailed quantitative comparison lives in Figure 6.
    fig, ax = plt.subplots(figsize=(9.0, 5.0))

    # Proxy bands — single muted palette (de-emphasize compared to Figure 6)
    ax.axhspan(MOLIT_LO, MOLIT_HI, color="#F59E0B", alpha=0.18, zorder=1,
               label=f"MOLIT atlas ({MOLIT_LO:.1f}–{MOLIT_HI:.1f}%)")
    ax.axhspan(CMB_LO, CMB_HI, color="#7C3AED", alpha=0.12, zorder=2,
               label=f"CMB literature ({CMB_LO:.0f}–{CMB_HI:.0f}%)")
    ax.axhspan(BFI_LO, BFI_HI, color="#0891B2", alpha=0.08, zorder=3,
               label=f"BFI proxy ({BFI_LO:.0f}–{BFI_HI:.0f}%)")

    # α-spectrum curves with star markers at α=1 only (decision endpoints)
    for idx, ws in enumerate(WATERSHEDS):
        Rv = R_alpha(ws["SW"], ws["beta"], alphas)
        ax.plot(alphas, Rv, "-", color=ws["color"], linewidth=2.8,
                label=ws["label"], zorder=5)
        # Highlight the recommended α=0.3 with a bold square; α=1 with a star.
        for a, sym, ms in [(0.3, "s", 9), (1.0, "*", 14)]:
            r = R_alpha(ws["SW"], ws["beta"], np.array([a]))[0]
            ax.plot(a, r, sym, color=ws["color"], markersize=ms,
                    markeredgecolor="black", markeredgewidth=0.8, zorder=6)
            if idx == 0:
                xytext_pt, ha, va = (-7, 9), "right", "bottom"
            else:
                xytext_pt, ha, va = (7, -10), "left", "top"
            ax.annotate(f"{r:.1f}%", xy=(a, r),
                        xytext=xytext_pt, textcoords="offset points",
                        fontsize=9, color=ws["color"],
                        fontweight="bold" if a == 0.3 else "normal",
                        ha=ha, va=va)

    # α guide line at the recommended default only — schematic emphasis
    ax.axvline(0.3, color="#065F46", linestyle="--", linewidth=1.6,
               alpha=0.7, zorder=2)
    ax.annotate("recommended\nα = 0.3",
                xy=(0.3, 35), xytext=(0.55, 40),
                fontsize=10, color="#065F46", fontweight="bold",
                ha="left",
                arrowprops=dict(arrowstyle="->", color="#065F46",
                                lw=1.2, alpha=0.8))

    # Light tick labels for α=0 and α=1 endpoints only
    for a, lbl in [(0.0, "α=0\n(no corr.)"), (1.0, "α=1.0\n(upper bound)")]:
        ax.text(a, -2.5, lbl, ha="center", fontsize=9, color="dimgray")

    ax.set_xlabel("Conservatism parameter α", fontsize=12)
    ax.set_ylabel("Watershed-mean recharge ratio (% of P)", fontsize=12)
    ax.set_title("α-spectrum decision chart (operational artefact)",
                 fontweight="bold")
    ax.set_xlim(-0.05, 1.08)
    ax.set_ylim(-5, 50)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=9, framealpha=0.95, ncol=1)

    plt.tight_layout()
    plt.subplots_adjust(right=0.72, bottom=0.16)

    out_dir = os.path.dirname(os.path.abspath(save_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {save_path} (dpi={dpi})")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/fig13_decision.png")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()
    main(args.out, args.dpi)
