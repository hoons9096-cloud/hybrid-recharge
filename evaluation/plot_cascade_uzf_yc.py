"""plot_cascade_uzf_yc.py

Generate paper-quality figure comparing cascade truth vs UZF kinematic-wave
under the actual Yeongcheon climate.  Also adds the field WTF estimate as a
third reference point for Loam soil.

Output: paper/fig14_cascade_uzf_yc.png
"""
from __future__ import annotations

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_style import apply_publication_style
apply_publication_style()

# ---------------------------------------------------------------------------
# Hard-coded results from run_cascade_uzf_yeongcheon.py
# (P=956 mm/yr, ETo=943 mm/yr, Loam = actual Yeongcheon soil)
# ---------------------------------------------------------------------------
SOILS      = ["Loamy\nSand", "Sandy\nLoam", "Silt\nLoam", "Clay", "Loam"]
CASCADE_MM = [202.5,          304.7,          201.0,         0.0,    235.0]
UZF_MM     = [290.6,          181.0,           45.8,         0.0,     69.2]
RICHARDS_MM = [None,          260.5,          151.5,         8.2,    197.1]   # v2; LS diverged
FIELD_MM   = [None,           None,            None,         None,    82.3]   # YC-012 WTF
P_ANN      = 956.2  # mm/yr

fig, axes = plt.subplots(1, 2, figsize=(10.5, 5))

# ---- Panel (a): grouped bar chart ----------------------------------------
ax = axes[0]
x     = np.arange(len(SOILS))
w     = 0.20
col_c = "#2166AC"
col_u = "#D6604D"
col_r = "#7B3F99"
col_f = "#1A7E43"

ax.bar(x - 1.5*w, CASCADE_MM, w, label="Cascade truth", color=col_c, alpha=0.85)
ax.bar(x - 0.5*w, UZF_MM,     w, label="UZF kinematic-wave (gravity)", color=col_u, alpha=0.85)

# Richards bars (skip None for diverged Loamy Sand)
rich_x = [xi + 0.5*w for xi, v in zip(x, RICHARDS_MM) if v is not None]
rich_v = [v for v in RICHARDS_MM if v is not None]
ax.bar(rich_x, rich_v, w, label="Richards mixed-form (capillary+gravity)",
       color=col_r, alpha=0.85)

# Mark LS divergence
ls_idx = 0
ax.text(x[ls_idx] + 0.5*w, 8, "div.", ha="center", va="bottom",
        fontsize=8, color=col_r, fontweight="bold", rotation=90)

# Field estimate for Loam only
loam_idx = 4
ax.bar(x[loam_idx] + 1.5*w, FIELD_MM[loam_idx], w,
       label="Field WTF (Yeongcheon_012)", color=col_f, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(SOILS, fontsize=9)
ax.set_ylabel("Annual recharge (mm yr⁻¹)", fontsize=10)
ax.set_title("(a) Annual recharge by soil type\n"
             f"(P = {P_ANN:.0f} mm yr⁻¹, ETo = 943 mm yr⁻¹, Yeongcheon climate)",
             fontsize=9)
ax.legend(fontsize=7.5, loc="upper left")
ax.set_ylim(0, 380)
ax.axhline(0, color="k", lw=0.5)

# Annotate Loam bracket
ax.annotate("UZF–Richards bracket\n[7.2%, 20.6%]\nfield WTF 8.6% inside",
            xy=(x[loam_idx], 95),
            xytext=(x[loam_idx] - 0.6, 320),
            arrowprops=dict(arrowstyle="->", color="green", lw=1.2),
            fontsize=7.5, color="green", ha="center")

ax.spines[["top", "right"]].set_visible(False)

# ---- Panel (b): scatter Cascade vs UZF (excluding Clay) ------------------
ax2 = axes[1]
# Exclude Clay (both near 0)
valid = [(c, u, s) for c, u, s in zip(CASCADE_MM, UZF_MM, SOILS) if c > 1 or u > 1]
cs  = [v[0] for v in valid]
us  = [v[1] for v in valid]
sns = [v[2] for v in valid]

colors = [col_c if "Clay" not in s else "gray" for s in sns]
scatter_colors = ["#4D9DE0", "#E15554", "#3BB273", "#F0A202", "#7768AE"]

for i, (c, u, s, sc) in enumerate(zip(cs, us, sns, scatter_colors[:len(cs)])):
    ax2.scatter(c, u, s=90, color=sc, zorder=5, label=s.replace("\n", " "))

# 1:1 line
lim = max(max(cs), max(us)) * 1.1
ax2.plot([0, lim], [0, lim], "k--", lw=1, label="1:1 line")

# Field WTF point for Loam
ax2.scatter(CASCADE_MM[loam_idx], FIELD_MM[loam_idx],
            s=130, marker="*", color=col_f, zorder=6, label="Field WTF (Loam, YC_012)")

ax2.set_xlabel("Cascade truth (mm yr⁻¹)", fontsize=10)
ax2.set_ylabel("UZF kinematic-wave (mm yr⁻¹)", fontsize=10)
ax2.set_title("(b) Cascade vs UZF scatter\n(excluding Clay; star = field WTF for Loam)",
              fontsize=9)

# Add Richards markers (open diamond) where defined
for i, (c, u, r, s, sc) in enumerate(zip(cs, us, RICHARDS_MM[1:], sns, scatter_colors[:len(cs)])):
    if r is not None:
        ax2.scatter(c, r, s=80, marker="D", facecolor="none",
                    edgecolor=col_r, linewidth=1.6, zorder=4)
ax2.scatter([], [], s=80, marker="D", facecolor="none",
            edgecolor=col_r, linewidth=1.6, label="Richards (vs cascade)")
ax2.set_xlim(0, lim)
ax2.set_ylim(0, lim)
ax2.legend(fontsize=8, ncol=2)
ax2.spines[["top", "right"]].set_visible(False)

# Factor-of-2 envelope
fov_x = np.linspace(0, lim, 200)
ax2.fill_between(fov_x, fov_x / 2, fov_x * 2, alpha=0.08, color="gray",
                 label="±factor-of-2")

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "paper", "fig14_cascade_uzf_yc.png")
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"✓ saved → {out}")
