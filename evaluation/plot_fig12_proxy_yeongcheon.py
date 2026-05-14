"""plot_fig12_proxy_yeongcheon.py

Regenerate fig12_proxy.png (Independent multi-proxy bracketing figure)
using the correct Yeongcheon field values and updated α=0.3 numbers.

Key values (from bias_corr_yeongcheon_v2.json and paper §4.5/§6.6):
  P_annual = 956.2 mm/yr
  Jaho-cheon   : SW=9.33%,  α0.3=10.30%, α0.5=11.06%, α1.0=13.58%
  Geumho-gang  : SW=9.34%,  α0.3=10.36%, α0.5=11.17%, α1.0=13.88%

Proxy bands:
  BFI  = 30–45% (Lyne-Hollick Korean montane, Lim et al. 2010)
  CMB  = 12–24% (Park et al. 2015)
  MOLIT= 13.6–23.0% (130–220 mm/yr ÷ 956 mm/yr)
  FAO-56 SWB = 16.0% (eto_yeongcheon.json)
  UZF lower bound = 7.2% (Loam, Yeongcheon climate, cascade_vs_uzf_yc.csv)
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
P_ANN = 956.2  # mm/yr

WATERSHEDS = ["Jaho-cheon", "Geumho-gang\nupper"]

# α-spectrum values (from bias_corr_yeongcheon_v2.json)
SW     = [9.33,  9.34 ]
BC_03  = [10.30, 10.36]
BC_05  = [11.06, 11.17]
BC_10  = [13.58, 13.88]

# Independent reference points
FAO56_PCT = 16.0   # FAO-56 SWB (eto_yeongcheon.json)
UZF_PCT   = 7.2    # UZF kinematic-wave Loam, Yeongcheon climate

# Proxy bands (all in % of P)
BFI_LO, BFI_HI   = 30.0, 45.0
CMB_LO, CMB_HI   = 12.0, 24.0
MOL_LO, MOL_HI   = 130/P_ANN*100, 220/P_ANN*100   # 13.6–23.0%
ENV_LO = min(BFI_LO, CMB_LO, MOL_LO)  # = 12%
ENV_HI = max(BFI_HI, CMB_HI, MOL_HI)  # = 45%

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 6),
                         sharey=True, gridspec_kw={"wspace": 0.08})

bar_colors = {
    "SW":    "#374151",
    "0.3":   "#93C5FD",
    "0.5":   "#3B82F6",
    "1.0":   "#1E3A8A",
}

x = np.array([0])
width = 0.18

for ax_idx, (ax, ws, sw_v, bc03, bc05, bc10) in enumerate(
        zip(axes, WATERSHEDS, SW, BC_03, BC_05, BC_10)):

    # ---- Proxy bands --------------------------------------------------------
    ax.axhspan(ENV_LO, ENV_HI, color="#10B981", alpha=0.07,
               label=f"Multi-proxy union ({ENV_LO:.0f}–{ENV_HI:.0f}%)")
    ax.axhspan(BFI_LO, BFI_HI, color="#0891B2", alpha=0.10,
               label=f"BFI proxy ({BFI_LO:.0f}–{BFI_HI:.0f}%)")
    ax.axhspan(CMB_LO, CMB_HI, color="#7C3AED", alpha=0.13,
               label=f"CMB ({CMB_LO:.0f}–{CMB_HI:.0f}%)")
    ax.axhspan(MOL_LO, MOL_HI, color="#F59E0B", alpha=0.18,
               label=f"MOLIT ({MOL_LO:.1f}–{MOL_HI:.1f}%)")

    # ---- Reference horizontals ----------------------------------------------
    ax.axhline(FAO56_PCT, color="#15803D", linewidth=1.4, linestyle="--",
               label=f"FAO-56 SWB ({FAO56_PCT:.1f}%)" if ax_idx == 0 else "_")
    ax.axhline(UZF_PCT, color="#DC2626", linewidth=1.4, linestyle=":",
               label=f"UZF lower bound ({UZF_PCT:.1f}%)" if ax_idx == 0 else "_")

    # ---- Bars ---------------------------------------------------------------
    bars_data = [
        ("SW",  sw_v,  bar_colors["SW"],  "Soil-weighted (α=0)"),
        ("0.3", bc03,  bar_colors["0.3"], "Bias-corr. α=0.3 ★"),
        ("0.5", bc05,  bar_colors["0.5"], "Bias-corr. α=0.5"),
        ("1.0", bc10,  bar_colors["1.0"], "Bias-corr. α=1.0"),
    ]
    offsets = np.array([-1.5, -0.5, +0.5, +1.5]) * width

    for (tag, val, col, lbl), off in zip(bars_data, offsets):
        label = lbl if ax_idx == 0 else "_"
        xpos = float(x[0]) + float(off)
        b = ax.bar(xpos, val, width, color=col, edgecolor="white",
                   linewidth=0.5, label=label)
        # value annotation
        ax.text(xpos, val + 0.4, f"{val:.1f}%",
                ha="center", va="bottom", fontsize=7.5, color=col)

    # Recommended α=0.3 annotation
    x0 = float(x[0])
    ax.annotate("★ recommended\ndefault",
                xy=(x0 - 0.5*width, bc03 + 0.4),
                xytext=(x0 - 0.5*width + 0.25, bc03 + 4.5),
                fontsize=7, color=bar_colors["0.3"], ha="center",
                arrowprops=dict(arrowstyle="->", color=bar_colors["0.3"],
                                lw=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels([ws], fontsize=10)
    ax.set_xlim(-0.55, 0.55)
    ax.set_ylim(0, 52)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    if ax_idx == 0:
        ax.set_ylabel("Recharge ratio (% of annual P)", fontsize=10)

axes[0].set_title("(a) Jaho-cheon", fontsize=10)
axes[1].set_title("(b) Geumho-gang upper", fontsize=10)

# ---- Shared legend ---------------------------------------------------------
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc="lower center", bbox_to_anchor=(0.5, -0.04),
           ncol=5, fontsize=8, framealpha=0.9)

fig.suptitle(
    "Independent multi-proxy bracketing: Yeongcheon alluvial domain\n"
    f"(P = {P_ANN:.0f} mm yr⁻¹; HSG-D Loam dominant; BFI/CMB/MOLIT = regional averages)",
    fontsize=10, y=1.01,
)
plt.tight_layout()

out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "paper", "fig12_proxy.png")
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"✓ saved → {out}")
