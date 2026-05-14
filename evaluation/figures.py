"""
figures.py -- 논문용 평가 그래프 생성

추정 함양량 맵 비교, 산점도, 시나리오별 지표 요약 바 차트를 생성한다.
300 dpi, 영문 라벨, matplotlib 기반.

Usage:
    from evaluation.figures import plot_recharge_comparison, plot_scatter_comparison
"""
from __future__ import annotations

import sys
import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

# 프로젝트 루트 경로 설정
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthetic.generate_domain import SyntheticDomain
from evaluation.metrics import EvalMetrics, compute_metrics


# ──────────────────────────────────────────────────────────
# 공통 스타일 설정
# ──────────────────────────────────────────────────────────
_FONT_TITLE = 13
_FONT_LABEL = 11
_FONT_TICK = 9
_CMAP_RECHARGE = "YlGnBu"
_DPI = 300


def _apply_tick_style(ax):
    """축 틱 폰트 크기 적용."""
    ax.tick_params(axis="both", labelsize=_FONT_TICK)


# ──────────────────────────────────────────────────────────
# 함양량 맵 비교 (side-by-side)
# ──────────────────────────────────────────────────────────
def plot_recharge_comparison(
    true_map: np.ndarray,
    estimated_maps: dict,
    domain: SyntheticDomain,
    scenario_name: str,
    save_path: Optional[str] = None,
):
    """Side-by-side recharge maps: truth + each method.

    Layout: 1 row, N+1 columns (truth + N methods).
    Colorbar: shared scale across all panels.
    Well locations overlaid.

    Parameters
    ----------
    true_map : np.ndarray
        (ny, nx) true recharge [mm/yr].
    estimated_maps : dict
        {method_name: (ny, nx) estimated recharge}.
    domain : SyntheticDomain
        Domain object for spatial info and well locations.
    scenario_name : str
        Scenario label for the title.
    save_path : str, optional
        If given, save figure to this path.
    """
    n_methods = len(estimated_maps)
    n_panels = n_methods + 1

    # 공통 색상 범위 결정
    all_maps = [true_map] + list(estimated_maps.values())
    vmin = min(m.min() for m in all_maps)
    vmax = max(m.max() for m in all_maps)

    # 도메인 공간 범위 [km]
    cfg = domain.config
    extent = [0, cfg.nx * cfg.dx / 1000, 0, cfg.ny * cfg.dy / 1000]

    # 관측정 좌표 [km]
    wx = domain.x_centers[domain.well_cols] / 1000
    wy = domain.y_centers[domain.well_rows] / 1000

    fig, axes = plt.subplots(1, n_panels, figsize=(14, 4))
    if n_panels == 1:
        axes = [axes]

    # (a) 참값
    ax = axes[0]
    im = ax.imshow(true_map, origin="lower", extent=extent,
                   cmap=_CMAP_RECHARGE, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    ax.scatter(wx, wy, c="red", marker="^", s=25, edgecolors="k",
               linewidths=0.4, zorder=5)
    ax.set_title("(a) True recharge", fontsize=_FONT_TITLE)
    ax.set_xlabel("X (km)", fontsize=_FONT_LABEL)
    ax.set_ylabel("Y (km)", fontsize=_FONT_LABEL)
    _apply_tick_style(ax)

    # (b~) 각 방법
    labels = "bcdefghij"
    for i, (method_name, est_map) in enumerate(estimated_maps.items()):
        ax = axes[i + 1]
        ax.imshow(est_map, origin="lower", extent=extent,
                  cmap=_CMAP_RECHARGE, vmin=vmin, vmax=vmax,
                  interpolation="nearest")
        ax.scatter(wx, wy, c="red", marker="^", s=25, edgecolors="k",
                   linewidths=0.4, zorder=5)
        label_char = labels[i] if i < len(labels) else ""
        ax.set_title(f"({label_char}) {method_name}", fontsize=_FONT_TITLE)
        ax.set_xlabel("X (km)", fontsize=_FONT_LABEL)
        ax.set_ylabel("")
        ax.set_yticklabels([])
        _apply_tick_style(ax)

    # 공유 컬러바
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, aspect=30, pad=0.02)
    cbar.set_label("Recharge (mm/yr)", fontsize=_FONT_LABEL)
    cbar.ax.tick_params(labelsize=_FONT_TICK)

    fig.suptitle(f"Recharge estimation comparison -- {scenario_name}",
                 fontsize=_FONT_TITLE + 1, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=_DPI, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    else:
        plt.show()

    plt.close(fig)
    return fig


# ──────────────────────────────────────────────────────────
# 산점도 비교
# ──────────────────────────────────────────────────────────
def plot_scatter_comparison(
    true_map: np.ndarray,
    estimated_maps: dict,
    metrics_list: list[EvalMetrics],
    scenario_name: str,
    save_path: Optional[str] = None,
):
    """Scatter plots: estimated vs true recharge for each method.

    Layout: 1 row, N columns.
    Include 1:1 line, RMSE and r annotations.

    Parameters
    ----------
    true_map : np.ndarray
        (ny, nx) true recharge [mm/yr].
    estimated_maps : dict
        {method_name: (ny, nx) estimated recharge}.
    metrics_list : list[EvalMetrics]
        Pre-computed metrics (for annotation).
    scenario_name : str
        Scenario label.
    save_path : str, optional
        If given, save figure.
    """
    n_methods = len(estimated_maps)
    # 방법명 -> EvalMetrics 매핑
    metrics_by_name = {m.method_name: m for m in metrics_list}

    fig, axes = plt.subplots(1, n_methods, figsize=(12, 4))
    if n_methods == 1:
        axes = [axes]

    true_flat = true_map.ravel()

    # 전체 범위 결정 (1:1 직선용)
    all_vals = [true_flat]
    for est in estimated_maps.values():
        all_vals.append(est.ravel())
    global_min = min(v.min() for v in all_vals)
    global_max = max(v.max() for v in all_vals)
    margin = (global_max - global_min) * 0.05
    lim_lo = global_min - margin
    lim_hi = global_max + margin

    labels = "abcdefghij"
    for i, (method_name, est_map) in enumerate(estimated_maps.items()):
        ax = axes[i]
        est_flat = est_map.ravel()

        # 산점도 (투명도로 밀도 표현)
        ax.scatter(true_flat, est_flat, s=4, alpha=0.3, c="steelblue",
                   edgecolors="none")

        # 1:1 직선
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1.0,
                label="1:1 line")

        ax.set_xlim(lim_lo, lim_hi)
        ax.set_ylim(lim_lo, lim_hi)
        ax.set_aspect("equal", adjustable="box")

        label_char = labels[i] if i < len(labels) else ""
        ax.set_title(f"({label_char}) {method_name}", fontsize=_FONT_TITLE)
        ax.set_xlabel("True recharge (mm/yr)", fontsize=_FONT_LABEL)
        if i == 0:
            ax.set_ylabel("Estimated recharge (mm/yr)", fontsize=_FONT_LABEL)
        _apply_tick_style(ax)

        # RMSE, r 주석
        if method_name in metrics_by_name:
            m = metrics_by_name[method_name]
            text = f"RMSE = {m.rmse:.1f} mm/yr\nr = {m.r_spatial:.3f}"
            ax.text(0.05, 0.95, text, transform=ax.transAxes,
                    fontsize=9, verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              alpha=0.8, edgecolor="gray"))

        ax.legend(loc="lower right", fontsize=8)

    fig.suptitle(f"Scatter comparison -- {scenario_name}",
                 fontsize=_FONT_TITLE + 1, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=_DPI, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    else:
        plt.show()

    plt.close(fig)
    return fig


# ──────────────────────────────────────────────────────────
# 시나리오별 지표 요약 바 차트
# ──────────────────────────────────────────────────────────
def plot_metrics_summary(
    all_metrics: list[EvalMetrics],
    save_path: Optional[str] = None,
):
    """Bar chart comparing methods across scenarios.

    Grouped bars: x-axis = scenarios, groups = methods.
    y-axis = RMSE (mm/yr).

    Parameters
    ----------
    all_metrics : list[EvalMetrics]
        Metrics across all scenarios and methods.
    save_path : str, optional
        If given, save figure.
    """
    # 시나리오, 방법 목록 추출 (순서 유지)
    scenarios = []
    methods = []
    for m in all_metrics:
        if m.scenario_name not in scenarios:
            scenarios.append(m.scenario_name)
        if m.method_name not in methods:
            methods.append(m.method_name)

    n_scenarios = len(scenarios)
    n_methods = len(methods)

    # RMSE 매트릭스 구성 (시나리오 x 방법)
    rmse_matrix = np.full((n_scenarios, n_methods), np.nan)
    for m in all_metrics:
        si = scenarios.index(m.scenario_name)
        mi = methods.index(m.method_name)
        rmse_matrix[si, mi] = m.rmse

    # 바 차트
    x = np.arange(n_scenarios)
    bar_width = 0.8 / n_methods
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]

    fig, ax = plt.subplots(figsize=(10, 5))

    for mi, method_name in enumerate(methods):
        offset = (mi - n_methods / 2 + 0.5) * bar_width
        vals = rmse_matrix[:, mi]
        color = colors[mi % len(colors)]
        ax.bar(x + offset, vals, bar_width * 0.9,
               label=method_name, color=color, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=_FONT_LABEL)
    ax.set_xlabel("Scenario", fontsize=_FONT_LABEL)
    ax.set_ylabel("RMSE (mm/yr)", fontsize=_FONT_LABEL)
    ax.set_title("Method comparison across scenarios", fontsize=_FONT_TITLE)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    _apply_tick_style(ax)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=_DPI, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    else:
        plt.show()

    plt.close(fig)
    return fig


# ──────────────────────────────────────────────────────────
# 노이즈 민감도 분석 그래프
# ──────────────────────────────────────────────────────────
def plot_noise_sensitivity(
    noise_levels: list[float],
    rmse_by_method: dict,
    save_path: Optional[str] = None,
):
    """RMSE vs observation noise level curve for each method.

    Parameters
    ----------
    noise_levels : list[float]
        Observation noise σ values [m].
    rmse_by_method : dict
        {method_name: list[float]} RMSE values at each noise level.
    save_path : str, optional
        If given, save figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    markers = ['o', 's', '^', 'D', 'v']
    colors = ['#4C72B0', '#C44E52', '#55A868', '#8172B2']

    noise_mm = [n * 1000 for n in noise_levels]  # m -> mm

    for i, (method_name, rmse_vals) in enumerate(rmse_by_method.items()):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        ax.plot(noise_mm, rmse_vals, f'-{marker}', color=color,
                label=method_name, markersize=7, linewidth=1.5)

    ax.set_xlabel('Observation noise σ (mm)', fontsize=_FONT_LABEL)
    ax.set_ylabel('RMSE (mm/yr)', fontsize=_FONT_LABEL)
    ax.set_title('Noise sensitivity analysis (S3 domain)', fontsize=_FONT_TITLE)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    _apply_tick_style(ax)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=_DPI, bbox_inches='tight')
        print(f'  Saved: {save_path}')
    else:
        plt.show()

    plt.close(fig)
    return fig


# ──────────────────────────────────────────────────────────
# 테스트: 더미 데이터로 샘플 그래프 생성
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    from synthetic.generate_domain import generate_domain, DomainConfig

    # 더미 도메인 생성 (S3 시나리오)
    domain = generate_domain(DomainConfig.S3())
    ny, nx = domain.config.ny, domain.config.nx

    rng = np.random.default_rng(123)

    # 더미 참값 함양량 맵 (Sy 기반 공간 변이 반영)
    true_map = 80.0 + 40.0 * (domain.Sy_map / domain.Sy_map.max())
    true_map += 10.0 * rng.standard_normal((ny, nx))
    true_map = np.clip(true_map, 10.0, 200.0)

    # 더미 추정 결과
    # Lumped: 공간 평균으로 균일
    lumped = np.full((ny, nx), np.mean(true_map))

    # Soil-weighted: 참값에 가까운 추정 (중간 수준 노이즈)
    soil_weighted = true_map + 12.0 * rng.standard_normal((ny, nx))

    # EnKF: 참값에 더 가까운 추정 (작은 노이즈)
    enkf = true_map + 6.0 * rng.standard_normal((ny, nx))

    estimated_maps = {
        "Lumped WTF": lumped,
        "Soil-weighted WTF": soil_weighted,
        "EnKF spatial": enkf,
    }

    # 지표 계산
    from evaluation.metrics import compare_methods, metrics_table

    metrics_list = compare_methods(estimated_maps, true_map, scenario_name="S3")
    print(metrics_table(metrics_list))
    print()

    # 그래프 저장 경로
    out_dir = os.path.dirname(os.path.abspath(__file__))

    # (1) 함양량 맵 비교
    plot_recharge_comparison(
        true_map, estimated_maps, domain, scenario_name="S3",
        save_path=os.path.join(out_dir, "fig_recharge_comparison_S3.png"),
    )

    # (2) 산점도 비교
    plot_scatter_comparison(
        true_map, estimated_maps, metrics_list, scenario_name="S3",
        save_path=os.path.join(out_dir, "fig_scatter_comparison_S3.png"),
    )

    # (3) 시나리오별 요약 (S1~S3 더미)
    all_metrics = list(metrics_list)  # S3만 있지만 시연용
    # S1 더미 추가
    for m in metrics_list:
        s1_m = EvalMetrics(
            method_name=m.method_name, scenario_name="S1",
            rmse=m.rmse * 0.5, mae=m.mae * 0.5, bias=m.bias * 0.5,
            r_spatial=min(m.r_spatial * 1.1, 1.0), rmse_pct=m.rmse_pct * 0.5,
        )
        all_metrics.append(s1_m)

    plot_metrics_summary(
        all_metrics,
        save_path=os.path.join(out_dir, "fig_metrics_summary.png"),
    )

    print("Sample figures generated.")
