"""
visualizer.py — 파이프라인 결과 시각화
========================================
생성 그래프:
  1. overview.png      — 원본/보정 수위 + 강우 + 펌핑 구간 오버레이
  2. detection.png     — 탐지 방법별 마스크 비교
  3. recharge.png      — 전처리 전/후 함양 시계열 비교
  4. soil_scores.png   — 토양별 점수 막대그래프
  5. summary_table.png — 핵심 지표 요약 표 (이미지)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import to_rgba
import warnings

from pipeline import PipelineResult

# ─── 스타일 상수 ─────────────────────────────────────────
C_RAW     = "#2c7bb6"   # 원본 수위 (파랑)
C_CORR    = "#d7191c"   # 보정 수위 (빨강)
C_PUMP    = "#fdae61"   # 펌핑 구간 (주황, alpha)
C_RAIN    = "#74add1"   # 강우 (하늘)
C_RECH0   = "#1a9641"   # 전처리 없음 함양 (녹색)
C_RECH1   = "#d73027"   # 전처리 후 함양 (붉은녹색)
C_SOIL    = "#4575b4"   # 토양 점수 막대

FIGSIZE_WIDE = (14, 4)
FIGSIZE_TALL = (14, 10)

warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"matplotlib")


# ─────────────────────────────────────────────────────────
class PipelineVisualizer:
    """
    Parameters
    ----------
    out_dir : str
        그래프 저장 폴더
    dpi : int
        출력 해상도
    """

    def __init__(self, out_dir: str = ".", dpi: int = 150):
        self.out_dir = out_dir
        self.dpi = dpi
        os.makedirs(out_dir, exist_ok=True)

    # ─────────────────────────────────────────────────────
    def plot_all(self, result: PipelineResult):
        """모든 그래프 생성"""
        self.plot_overview(result)
        self.plot_detection(result)
        self.plot_recharge(result)
        self.plot_soil_scores(result)
        self.plot_summary_table(result)

    # ═════════════════════════════════════════════════════
    # 1. Overview — 수위 + 강우 + 펌핑 구간
    # ═════════════════════════════════════════════════════
    def plot_overview(self, result: PipelineResult):
        fig, axes = plt.subplots(
            2, 1, figsize=(14, 7),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True,
        )
        ax_wl, ax_rain = axes
        t = result.dates

        # 강우
        ax_rain.bar(t, result.rainfall, color=C_RAIN, width=1.0, alpha=0.8, label="Rainfall")
        ax_rain.invert_yaxis()
        ax_rain.set_ylabel("Rainfall (mm)", fontsize=9)
        ax_rain.set_ylim(ax_rain.get_ylim()[0], 0)

        # 수위
        ax_wl.plot(t, result.raw_wl, color=C_RAW,  lw=1.0, alpha=0.8, label="Raw WL")
        ax_wl.plot(t, result.corrected_wl, color=C_CORR, lw=1.2, label="Corrected WL")

        # 펌핑 구간 음영
        _shade_pump(ax_wl, t, result.detection.pump_mask, color=C_PUMP)
        _shade_pump(ax_rain, t, result.detection.pump_mask, color=C_PUMP)

        ax_wl.set_ylabel("Groundwater Level (m)", fontsize=10)
        ax_wl.legend(fontsize=9, loc="upper right")
        ax_wl.set_title(
            f"Pump Pre-processing Overview  "
            f"(pumping contamination: {result.detection.pump_fraction*100:.1f}%)",
            fontsize=11, fontweight="bold",
        )

        # 범례 — 펌핑 음영
        pump_patch = mpatches.Patch(color=C_PUMP, alpha=0.4, label="Pumping period")
        ax_wl.legend(handles=ax_wl.get_legend_handles_labels()[0] + [pump_patch],
                     fontsize=9, loc="upper right")

        ax_rain.xaxis_date()
        fig.autofmt_xdate(rotation=30, ha="right")
        fig.tight_layout()
        self._save(fig, "overview.png")

    # ═════════════════════════════════════════════════════
    # 2. Detection — 방법별 마스크
    # ═════════════════════════════════════════════════════
    def plot_detection(self, result: PipelineResult):
        det = result.detection
        method_names = list(det.method_masks.keys()) if det.method_masks else []

        n_rows = 2 + len(method_names)
        fig, axes = plt.subplots(n_rows, 1, figsize=(14, 2.0 * n_rows), sharex=True)

        t = result.dates

        # 원본 수위
        ax0 = axes[0]
        ax0.plot(t, result.raw_wl, color=C_RAW, lw=0.9, label="Raw WL")
        _shade_pump(ax0, t, det.pump_mask, color=C_PUMP)
        ax0.set_ylabel("WL (m)", fontsize=8)
        ax0.set_title("Pumping Detection Result", fontsize=11, fontweight="bold")
        ax0.legend(fontsize=8, loc="upper right")

        # 신뢰도
        ax1 = axes[1]
        ax1.fill_between(t, det.confidence, color="#e08214", alpha=0.7)
        ax1.axhline(0.5, color="gray", lw=0.8, linestyle="--", label="threshold 0.5")
        ax1.set_ylim(0, 1)
        ax1.set_ylabel("Confidence", fontsize=8)
        ax1.legend(fontsize=8)

        # 방법별 마스크
        colors_m = ["#d73027", "#4575b4", "#1a9641", "#762a83"]
        for i, mname in enumerate(method_names):
            ax = axes[2 + i]
            mask = det.method_masks[mname].astype(float)
            ax.fill_between(t, mask, color=colors_m[i % len(colors_m)], alpha=0.7)
            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel(mname, fontsize=8)

        fig.autofmt_xdate(rotation=30, ha="right")
        fig.tight_layout()
        self._save(fig, "detection.png")

    # ═════════════════════════════════════════════════════
    # 3. Recharge — 전처리 전/후 비교
    # ═════════════════════════════════════════════════════
    def plot_recharge(self, result: PipelineResult):
        r0 = result.result_raw
        r1 = result.result_corrected
        t = result.dates

        fig = plt.figure(figsize=(14, 9))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

        # ─ 수위 비교 ─
        ax_wl = fig.add_subplot(gs[0, :])
        ax_wl.plot(t, result.raw_wl,      color=C_RAW,  lw=0.9, alpha=0.7, label="Raw WL")
        ax_wl.plot(t, result.corrected_wl, color=C_CORR, lw=1.1, label="Corrected WL")
        _shade_pump(ax_wl, t, result.detection.pump_mask, color=C_PUMP)
        ax_wl.set_ylabel("GW Level (m)", fontsize=9)
        ax_wl.legend(fontsize=9)
        ax_wl.set_title("Water Level Comparison", fontsize=10)

        # ─ 함양 시계열 ─
        ax_rech = fig.add_subplot(gs[1, :])
        _plot_rech_bars(ax_rech, t, r0.rech_total, C_RECH0, "Recharge (no preproc)")
        _plot_rech_bars(ax_rech, t, r1.rech_total, C_RECH1, "Recharge (corrected)", alpha=0.6)
        ax_rech.set_ylabel("Recharge (mm/day)", fontsize=9)
        ax_rech.legend(fontsize=9)
        ax_rech.set_title("Daily Recharge Estimate", fontsize=10)

        # ─ 누적 함양 ─
        ax_cum = fig.add_subplot(gs[2, 0])
        cum0 = np.nancumsum(r0.rech_total)
        cum1 = np.nancumsum(r1.rech_total)
        ax_cum.plot(t, cum0, color=C_RECH0, lw=1.3, label=f"No preproc  ({r0.rech_rate_pct:.1f}%)")
        ax_cum.plot(t, cum1, color=C_RECH1, lw=1.3, label=f"Corrected   ({r1.rech_rate_pct:.1f}%)")
        ax_cum.set_ylabel("Cumulative Recharge (mm)", fontsize=9)
        ax_cum.legend(fontsize=8)
        ax_cum.set_title("Cumulative Recharge", fontsize=10)

        # ─ Kalman w_est 분포 ─
        ax_sy = fig.add_subplot(gs[2, 1])
        w_clean = r1.w_est[~np.isnan(r1.w_est)]
        ax_sy.hist(w_clean, bins=30, color=C_SOIL, edgecolor="white", alpha=0.8)
        ax_sy.axvline(float(np.nanmean(w_clean)), color="red", lw=1.5,
                      linestyle="--", label=f"mean={np.nanmean(w_clean):.3f}")
        ax_sy.set_xlabel("Kalman w_est (recharge forcing)", fontsize=9)
        ax_sy.set_ylabel("Count", fontsize=9)
        ax_sy.legend(fontsize=8)
        ax_sy.set_title(f"w_est Distribution  [{r1.best_soil_name}]", fontsize=10)

        fig.autofmt_xdate(rotation=30, ha="right")
        self._save(fig, "recharge.png")

    # ═════════════════════════════════════════════════════
    # 4. Soil Scores
    # ═════════════════════════════════════════════════════
    def plot_soil_scores(self, result: PipelineResult):
        from kalman.wtf_kalman import SOIL_NAMES

        scores_raw  = result.result_raw.soil_scores
        scores_corr = result.result_corrected.soil_scores
        ns = len(SOIL_NAMES)
        x = np.arange(ns)
        width = 0.38

        fig, ax = plt.subplots(figsize=(12, 5))
        bars0 = ax.bar(x - width/2, scores_raw,  width, color=C_RAW,  alpha=0.8, label="No preproc")
        bars1 = ax.bar(x + width/2, scores_corr, width, color=C_CORR, alpha=0.8, label="Corrected")

        # 최상위 토양 마크
        best_raw  = int(np.argmax(scores_raw))
        best_corr = int(np.argmax(scores_corr))
        ax.bar(best_raw  - width/2, scores_raw[best_raw],
               width, color=C_RAW,  edgecolor="black", lw=2.0, label="_")
        ax.bar(best_corr + width/2, scores_corr[best_corr],
               width, color=C_CORR, edgecolor="black", lw=2.0, label="_")

        ax.set_xticks(x)
        ax.set_xticklabels(SOIL_NAMES, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Composite Score", fontsize=10)
        ax.set_title("Soil Identification Score (12 Standard Soils)", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        fig.tight_layout()
        self._save(fig, "soil_scores.png")

    # ═════════════════════════════════════════════════════
    # 5. Summary Table (이미지)
    # ═════════════════════════════════════════════════════
    def plot_summary_table(self, result: PipelineResult):
        r0 = result.result_raw
        r1 = result.result_corrected
        imp = result.improvement

        rows = [
            ["Metric",                "Without Preproc",            "With Preproc",              "Change"],
            ["Best Soil",             r0.best_soil_name,            r1.best_soil_name,           "—"],
            ["RMSE (m)",              f"{r0.rmse:.4f}",             f"{r1.rmse:.4f}",            f"{imp['rmse_delta']:+.4f}"],
            ["NSE",                   f"{r0.nse:.3f}",              f"{r1.nse:.3f}",             f"{imp['nse_delta']:+.3f}"],
            ["CC",                    f"{r0.cc:.3f}",               f"{r1.cc:.3f}",              f"{r1.cc - r0.cc:+.3f}"],
            ["Recharge rate (%)",     f"{r0.rech_rate_pct:.1f}",   f"{r1.rech_rate_pct:.1f}",  f"{imp['rech_delta_pct']:+.1f}"],
            ["Pump fraction (%)",     f"{result.detection.pump_fraction*100:.1f}", "—",          "—"],
            ["Pump events",           f"{len(result.detection.drop_events)}", "—",              "—"],
        ]
        if r0.rech_bias_pct is not None:
            rows.append([
                "Recharge Bias (%)",
                f"{r0.rech_bias_pct:+.1f}",
                f"{r1.rech_bias_pct:+.1f}",
                f"{imp['bias_delta']:+.1f}",
            ])

        fig, ax = plt.subplots(figsize=(10, 0.45 * len(rows) + 1.2))
        ax.axis("off")

        col_widths = [0.28, 0.24, 0.24, 0.18]
        table = ax.table(
            cellText=rows[1:],
            colLabels=rows[0],
            colWidths=col_widths,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9.5)
        table.scale(1, 1.6)

        # 헤더 스타일
        for j in range(len(rows[0])):
            cell = table[0, j]
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")

        # 홀수 행 음영
        for i in range(1, len(rows) - 1):
            for j in range(len(rows[0])):
                if i % 2 == 0:
                    table[i, j].set_facecolor("#f0f4f8")

        # Change 컬럼 색상
        for i in range(1, len(rows) - 1):
            cell = table[i, 3]
            val = rows[i][3]
            if val not in ("—", ""):
                try:
                    v = float(val.replace("+", ""))
                    # RMSE는 낮을수록 좋음 → 음수가 good
                    if rows[i][0].startswith("RMSE") or rows[i][0].startswith("Recharge Bias"):
                        cell.set_facecolor("#d5f5e3" if v < 0 else "#fde8e8")
                    else:
                        cell.set_facecolor("#d5f5e3" if v > 0 else "#fde8e8")
                except ValueError:
                    pass

        ax.set_title("Pipeline Result Summary", fontsize=12, fontweight="bold", pad=14)
        fig.tight_layout()
        self._save(fig, "summary_table.png")

    # ─────────────────────────────────────────────────────
    def _save(self, fig, fname: str):
        path = os.path.join(self.out_dir, fname)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"    [plot] {path}")


# ─── 헬퍼 함수 ───────────────────────────────────────────
def _shade_pump(ax, dates, pump_mask, color="#fdae61", alpha=0.35):
    """펌핑 구간을 수직 음영으로 표시"""
    import pandas as pd
    t = np.array(dates)
    in_pump = False
    t_start = None
    for i, flag in enumerate(pump_mask):
        if flag and not in_pump:
            in_pump = True
            t_start = t[i]
        elif not flag and in_pump:
            in_pump = False
            ax.axvspan(t_start, t[i], color=color, alpha=alpha, lw=0)
    if in_pump:
        ax.axvspan(t_start, t[-1], color=color, alpha=alpha, lw=0)


def _plot_rech_bars(ax, t, recharge, color, label, alpha=0.85):
    """양수/음수 함양을 막대 그래프로"""
    rech = np.where(np.isnan(recharge), 0, recharge)
    ax.bar(t, np.where(rech >= 0, rech, 0),
           color=color, alpha=alpha, width=1.0, label=label)
    ax.bar(t, np.where(rech < 0, rech, 0),
           color=color, alpha=alpha * 0.5, width=1.0)
