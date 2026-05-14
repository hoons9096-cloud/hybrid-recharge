"""graphical_abstract.py — JoH-style 1920×1080 graphical abstract.

Layout:
  Left  (40%): WTF identity → bias correction schema
  Right (60%): α-spectrum decision chart with UZF/Richards physical
              brackets and cascade-truth overshoot marker
  Bottom: KEY FINDING strip

All numerical values match draft.md / case_yeongcheon.json (ET/P=0.648
calibration, β̂ from regression).

Run:
  /opt/anaconda3/envs/myenv/bin/python evaluation/graphical_abstract.py
"""
from __future__ import annotations
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch


# ---------------------------------------------------------------------------
# Numerical values — keep in sync with case_yeongcheon.json + draft.md
# ---------------------------------------------------------------------------
ALPHA_MARKS = np.array([0.0, 0.3, 0.5, 1.0])
JAHO   = np.array([9.33, 10.32, 11.11, 13.73])    # Jaho-cheon
GEUMHO = np.array([9.34, 10.40, 11.24, 14.11])    # Geumho-gang upper

UZF_LOWER     = 7.2
RICHARDS_UPPER = 20.6
CASCADE_TRUTH  = 24.6

BFI_LO, BFI_HI    = 30.0, 45.0
CMB_LO, CMB_HI    = 12.0, 24.0
MOLIT_LO, MOLIT_HI = 13.5, 22.9


def main(save_png: str = "paper/graphical_abstract.png", dpi: int = 200):
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.unicode_minus": False,
    })

    # Elsevier graphical abstract guidance: minimum 1328 × 531 px (W × H),
    # ratio 2.5 : 1 landscape. We use 15.0 × 6.0 inches at 300 dpi → 4500 ×
    # 1800 px (well above minimum, exact 2.5 : 1).
    fig = plt.figure(figsize=(15.0, 6.0), dpi=dpi)
    gs = GridSpec(1, 2, width_ratios=[2, 3], wspace=0.16,
                  left=0.03, right=0.985, top=0.88, bottom=0.18)

    # ── Left: methodology schema ────────────────────────────────────────
    axL = fig.add_subplot(gs[0])
    axL.axis("off")

    schema = [
        (0.5, 0.92, r"$\bf{WTF\ Identity}$",                  14),
        (0.5, 0.83, r"$R = S_y \cdot \Delta h$",              19),
        (0.5, 0.74, r"$\downarrow$",                          22),
        (0.5, 0.66, r"$\bf{+5\!-\!49\%\ structural\ bias}$",  12),
        (0.5, 0.59, "under cascade vadose dynamics",           9),
        (0.5, 0.50, r"$\downarrow$",                          22),
        (0.5, 0.42, r"$\bf{Learned\ bias\ correction}$",      12),
        (0.5, 0.35, r"$\hat{\beta}(x)$ + conservatism $\alpha$", 11),
        (0.5, 0.26, r"$\downarrow$",                          22),
        (0.5, 0.17, r"$\bf{Decision\ envelope}$",             13),
        (0.5, 0.09, r"$\hat{R}_{\rm corr}(\alpha) \in$ [UZF, Richards]",
                                                              11),
    ]
    for x, y, t, sz in schema:
        axL.text(x, y, t, ha="center", va="center", fontsize=sz,
                 transform=axL.transAxes)
    axL.add_patch(FancyBboxPatch((0.04, 0.04), 0.92, 0.93,
        boxstyle="round,pad=0.02", linewidth=1.5,
        edgecolor="#444", facecolor="#f8f8f8",
        transform=axL.transAxes, zorder=-1))

    # ── Right: α-spectrum decision chart ────────────────────────────────
    axR = fig.add_subplot(gs[1])

    # Statistical-proxy bands (background)
    axR.axhspan(BFI_LO, BFI_HI, color="#0891B2", alpha=0.10,
                label=f"BFI proxy ({BFI_LO:.0f}–{BFI_HI:.0f}%)")
    axR.axhspan(CMB_LO, CMB_HI, color="#7C3AED", alpha=0.13,
                label=f"CMB literature ({CMB_LO:.0f}–{CMB_HI:.0f}%)")
    axR.axhspan(MOLIT_LO, MOLIT_HI, color="#F59E0B", alpha=0.18,
                label=f"MOLIT atlas ({MOLIT_LO:.1f}–{MOLIT_HI:.1f}%)")

    # Physical bracket — emphasized
    axR.axhspan(UZF_LOWER, RICHARDS_UPPER, color="#1A7E43", alpha=0.05)
    axR.axhline(UZF_LOWER, color="#1A7E43", linestyle="--", linewidth=2.0,
                label=f"UZF lower bound ({UZF_LOWER}%, gravity)")
    axR.axhline(RICHARDS_UPPER, color="#7B3F99", linestyle="--",
                linewidth=2.0,
                label=f"Richards upper bound ({RICHARDS_UPPER}%, mixed-form)")

    # Cascade truth marker (above Richards → outside bracket)
    axR.scatter([1.07], [CASCADE_TRUTH], marker="*", s=380,
                color="#B91C1C", zorder=10, edgecolor="black",
                linewidth=0.8,
                label=f"Cascade truth ({CASCADE_TRUTH}%, outside)")
    axR.annotate("above Richards\n→ overshoot confirmed",
                 xy=(1.07, CASCADE_TRUTH),
                 xytext=(0.62, 31),
                 fontsize=10, color="#B91C1C",
                 arrowprops=dict(arrowstyle="->", color="#B91C1C", lw=1.3))

    # α-spectrum curves — Yeongcheon watersheds
    axR.plot(ALPHA_MARKS, JAHO, "o-", color="#DC2626", linewidth=2.6,
             markersize=10, markeredgecolor="black", markeredgewidth=0.6,
             label="Jaho-cheon", zorder=8)
    axR.plot(ALPHA_MARKS, GEUMHO, "s-", color="#2563EB", linewidth=2.6,
             markersize=10, markeredgecolor="black", markeredgewidth=0.6,
             label="Geumho-gang upper", zorder=8)

    # α=0.3 default — emphasized
    axR.axvline(0.3, color="#065F46", linestyle=":", linewidth=1.2,
                alpha=0.7)
    axR.text(0.31, 49, "α=0.3\n(recommended\ndefault)", fontsize=10,
             color="#065F46", fontweight="bold", va="top")

    # Annotate values at α=0.3 and α=1.0
    for x, jh, gh in [(0.3, JAHO[1], GEUMHO[1]),
                      (1.0, JAHO[3], GEUMHO[3])]:
        axR.annotate(f"{jh:.2f}", xy=(x, jh), xytext=(8, -12),
                     textcoords="offset points",
                     fontsize=9, color="#DC2626", fontweight="bold")
        axR.annotate(f"{gh:.2f}", xy=(x, gh), xytext=(8, 6),
                     textcoords="offset points",
                     fontsize=9, color="#2563EB", fontweight="bold")

    axR.set_xlabel("Conservatism parameter α", fontsize=13)
    axR.set_ylabel("Watershed-mean recharge (% of P)", fontsize=13)
    axR.set_xlim(-0.05, 1.18)
    axR.set_ylim(0, 50)
    axR.grid(linestyle="--", alpha=0.3)
    axR.legend(loc="upper left", fontsize=8.5, ncol=2, framealpha=0.95,
               handlelength=1.6, columnspacing=0.9)

    # ── Title + bottom KEY FINDING strip ────────────────────────────────
    fig.suptitle(
        "Bias-Aware WTF: From Point Estimator to Decision Envelope",
        fontsize=14, fontweight="bold", y=0.95,
    )
    fig.text(0.5, 0.06,
        "KEY FINDING — All bias-corrected WTF estimates (α∈[0,1]: "
        "9.3–14.1% of P) lie inside the [UZF = 7.2%, Richards = 20.6%] "
        "physical envelope; the cascade truth (24.6%) lies above the "
        "Richards upper bound, model-independently confirming the "
        "conservative α<1 default.",
        ha="center", fontsize=9.5, style="italic", wrap=True,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#fff8dc",
                  edgecolor="#888", linewidth=0.9))

    # Output
    out_dir = os.path.dirname(os.path.abspath(save_png))
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(save_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    pdf_path = save_png.replace(".png", ".pdf")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {save_png} ({dpi} dpi)")
    print(f"  ✓ {pdf_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/graphical_abstract.png")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()
    main(args.out, args.dpi)
