"""
generate_domain.py — 합성 유역 도메인 생성기

가상 2D 격자 도메인에 토양 유형을 배치하고, 관측정 위치를 설정한다.
시나리오별로 토양 불균질성 수준과 관측정 밀도를 제어한다.

Scenarios (CLAUDE.md):
    S1: 균질,       높은 관측정, 낮은 노이즈
    S2: 약한 불균질, 높은 관측정, 낮은 노이즈
    S3: 강한 불균질, 높은 관측정, 낮은 노이즈
    S4: 강한 불균질, 낮은 관측정, 낮은 노이즈
    S5: 강한 불균질, 높은 관측정, 높은 노이즈

Usage:
    from synthetic.generate_domain import generate_domain, DomainConfig
    domain = generate_domain(DomainConfig.S3())
"""
from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

# soil_db 임포트 경로 설정
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from soil_db import SOIL_DB, SoilRecord


# ──────────────────────────────────────────────────────────
# 도메인 설정
# ──────────────────────────────────────────────────────────
@dataclass
class DomainConfig:
    """Synthetic domain configuration."""
    name: str = "S3"
    nx: int = 100                     # 격자 수 (x방향)
    ny: int = 100                     # 격자 수 (y방향)
    dx: float = 100.0                 # 격자 크기 [m]
    dy: float = 100.0                 # 격자 크기 [m]

    # 토양 유형 (soil_db index 기반)
    soil_types: List[int] = field(default_factory=lambda: [1, 3, 6, 9, 12])

    # 불균질성: "homogeneous", "weak", "strong"
    heterogeneity: str = "strong"

    # 관측정 밀도: "high" (25개), "low" (4개)
    well_density: str = "high"

    # 관측 노이즈 σ [m] (수위 관측 오차)
    obs_noise_std: float = 0.01

    random_seed: int = 42

    @classmethod
    def S1(cls) -> "DomainConfig":
        """균질, 높은 관측정, 낮은 노이즈."""
        return cls(name="S1", soil_types=[12],
                   heterogeneity="homogeneous", well_density="high",
                   obs_noise_std=0.01)

    @classmethod
    def S2(cls) -> "DomainConfig":
        """약한 불균질, 높은 관측정, 낮은 노이즈."""
        return cls(name="S2", soil_types=[3, 12],
                   heterogeneity="weak", well_density="high",
                   obs_noise_std=0.01)

    @classmethod
    def S3(cls) -> "DomainConfig":
        """강한 불균질, 높은 관측정, 낮은 노이즈."""
        return cls(name="S3", soil_types=[1, 3, 6, 9, 12],
                   heterogeneity="strong", well_density="high",
                   obs_noise_std=0.01)

    @classmethod
    def S4(cls) -> "DomainConfig":
        """강한 불균질, 낮은 관측정, 낮은 노이즈."""
        return cls(name="S4", soil_types=[1, 3, 6, 9, 12],
                   heterogeneity="strong", well_density="low",
                   obs_noise_std=0.01)

    @classmethod
    def S5(cls) -> "DomainConfig":
        """강한 불균질, 높은 관측정, 높은 노이즈."""
        return cls(name="S5", soil_types=[1, 3, 6, 9, 12],
                   heterogeneity="strong", well_density="high",
                   obs_noise_std=0.03)


# ──────────────────────────────────────────────────────────
# 도메인 결과
# ──────────────────────────────────────────────────────────
@dataclass
class SyntheticDomain:
    """Generated synthetic domain."""
    config: DomainConfig

    # 격자 좌표 [m]
    x_centers: np.ndarray    # (nx,)
    y_centers: np.ndarray    # (ny,)

    # 토양 배치 — soil_db index 값
    soil_map: np.ndarray     # (ny, nx), int

    # 수리특성 맵 (토양별 문헌값에서 매핑)
    Sy_map: np.ndarray       # (ny, nx), specific yield [-]
    alpha_map: np.ndarray    # (ny, nx), max recharge fraction [-]
    tau_map: np.ndarray      # (ny, nx), drainage time [days]

    # 관측정 위치 — 격자 인덱스
    well_rows: np.ndarray    # (n_wells,)
    well_cols: np.ndarray    # (n_wells,)

    # 토양별 면적 비율
    soil_fractions: dict     # {soil_index: fraction}

    @property
    def n_wells(self) -> int:
        return len(self.well_rows)

    @property
    def domain_size_km(self) -> Tuple[float, float]:
        """Domain extent in km."""
        return (self.config.nx * self.config.dx / 1000,
                self.config.ny * self.config.dy / 1000)

    @property
    def well_xy(self) -> np.ndarray:
        """Well coordinates [m], shape (n_wells, 2)."""
        return np.column_stack([
            self.x_centers[self.well_cols],
            self.y_centers[self.well_rows],
        ])

    def get_soil_record(self, row: int, col: int) -> SoilRecord:
        """Get SoilRecord for a grid cell."""
        return SOIL_DB[int(self.soil_map[row, col])]

    def summary(self) -> str:
        """Print domain summary."""
        sx, sy = self.domain_size_km
        lines = [
            f"═══ Synthetic Domain: {self.config.name} ═══",
            f"  Grid     : {self.config.ny}×{self.config.nx} "
            f"({sy:.0f}×{sx:.0f} km, Δ={self.config.dx:.0f}m)",
            f"  Hetero.  : {self.config.heterogeneity}",
            f"  Wells    : {self.n_wells} ({self.config.well_density} density)",
            f"  Obs noise: σ = {self.config.obs_noise_std:.3f} m",
            "",
            "  Soil distribution:",
        ]
        for si, frac in sorted(self.soil_fractions.items()):
            rec = SOIL_DB[si]
            lines.append(
                f"    {rec.name:<20s} (Sy={rec.sy_lit:.2f}): "
                f"{frac*100:5.1f}%"
            )
        # Sy 통계
        lines += [
            "",
            f"  Sy range : [{self.Sy_map.min():.3f}, {self.Sy_map.max():.3f}]",
            f"  Sy mean  : {self.Sy_map.mean():.3f}  (std={self.Sy_map.std():.3f})",
        ]
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────
# 토양 배치 생성
# ──────────────────────────────────────────────────────────
def _place_soils_homogeneous(
    ny: int, nx: int, soil_types: List[int],
) -> np.ndarray:
    """균질: 전 도메인 단일 토양."""
    return np.full((ny, nx), soil_types[0], dtype=int)


def _place_soils_weak(
    ny: int, nx: int, soil_types: List[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """약한 불균질: 수평 밴드 형태로 2~3개 토양 배치.

    밴드 경계에 약간의 랜덤 변동을 주어 자연스러운 전이대 형성.
    """
    soil_map = np.full((ny, nx), soil_types[0], dtype=int)
    n_soils = len(soil_types)
    band_height = ny // n_soils

    for k, si in enumerate(soil_types):
        y_start = k * band_height
        y_end = (k + 1) * band_height if k < n_soils - 1 else ny
        soil_map[y_start:y_end, :] = si

    # 밴드 경계 ±5행 랜덤 교란
    for k in range(1, n_soils):
        y_boundary = k * band_height
        for col in range(nx):
            jitter = rng.integers(-5, 6)
            y_actual = np.clip(y_boundary + jitter, 0, ny - 1)
            if jitter > 0:
                soil_map[y_boundary:y_actual, col] = soil_types[k]
            elif jitter < 0:
                soil_map[y_actual:y_boundary, col] = soil_types[k - 1]

    return soil_map


def _place_soils_strong(
    ny: int, nx: int, soil_types: List[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """강한 불균질: Voronoi 기반 불규칙 패치.

    랜덤 시드 포인트에서 가장 가까운 점 할당 (nearest-neighbor)으로
    불규칙한 토양 패치를 생성한다. 패치 수 = 토양 유형 수 × 4~6개.
    """
    n_soils = len(soil_types)
    n_seeds = n_soils * 5  # 25개 시드 포인트

    # 시드 포인트 좌표 생성
    seed_y = rng.uniform(0, ny, n_seeds).astype(float)
    seed_x = rng.uniform(0, nx, n_seeds).astype(float)

    # 각 시드에 토양 유형 할당 (균등 분배)
    seed_soils = np.array([soil_types[i % n_soils] for i in range(n_seeds)])
    # 셔플해서 공간적으로 랜덤하게
    rng.shuffle(seed_soils)

    # 각 격자 셀을 가장 가까운 시드에 할당 (Voronoi)
    yy, xx = np.mgrid[0:ny, 0:nx]
    yy = yy.astype(float)
    xx = xx.astype(float)

    soil_map = np.zeros((ny, nx), dtype=int)
    min_dist = np.full((ny, nx), np.inf)

    for s in range(n_seeds):
        dist = (yy - seed_y[s])**2 + (xx - seed_x[s])**2
        closer = dist < min_dist
        soil_map[closer] = seed_soils[s]
        min_dist[closer] = dist[closer]

    return soil_map


# ──────────────────────────────────────────────────────────
# 관측정 배치
# ──────────────────────────────────────────────────────────
def _place_wells(
    ny: int, nx: int, density: str,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Place observation wells on a regular grid with slight jitter.

    Parameters
    ----------
    density : str
        "high" → 5×5 = 25 wells
        "low"  → 2×2 = 4 wells
    """
    if density == "high":
        n_side = 5
    else:
        n_side = 2

    margin = 0.1  # 도메인 경계에서 10% 여백
    y_pos = np.linspace(int(ny * margin), int(ny * (1 - margin)), n_side).astype(int)
    x_pos = np.linspace(int(nx * margin), int(nx * (1 - margin)), n_side).astype(int)

    rows, cols = [], []
    for yi in y_pos:
        for xi in x_pos:
            # ±2셀 지터
            jr = int(np.clip(yi + rng.integers(-2, 3), 0, ny - 1))
            jc = int(np.clip(xi + rng.integers(-2, 3), 0, nx - 1))
            rows.append(jr)
            cols.append(jc)

    return np.array(rows), np.array(cols)


# ──────────────────────────────────────────────────────────
# 메인 생성 함수
# ──────────────────────────────────────────────────────────
def generate_domain(config: DomainConfig | None = None) -> SyntheticDomain:
    """Generate a synthetic 2D domain with soil types and well locations.

    Parameters
    ----------
    config : DomainConfig, optional
        Domain configuration. Defaults to S3.

    Returns
    -------
    SyntheticDomain
        Generated domain with soil map, hydraulic property maps, and wells.
    """
    if config is None:
        config = DomainConfig.S3()

    rng = np.random.default_rng(config.random_seed)

    nx, ny = config.nx, config.ny

    # 격자 중심 좌표
    x_centers = np.arange(nx) * config.dx + config.dx / 2
    y_centers = np.arange(ny) * config.dy + config.dy / 2

    # 토양 배치
    if config.heterogeneity == "homogeneous":
        soil_map = _place_soils_homogeneous(ny, nx, config.soil_types)
    elif config.heterogeneity == "weak":
        soil_map = _place_soils_weak(ny, nx, config.soil_types, rng)
    elif config.heterogeneity == "strong":
        soil_map = _place_soils_strong(ny, nx, config.soil_types, rng)
    else:
        raise ValueError(f"Unknown heterogeneity: {config.heterogeneity}")

    # 수리특성 맵 (soil_db 문헌값 매핑)
    Sy_map = np.zeros((ny, nx))
    alpha_map = np.zeros((ny, nx))
    tau_map = np.zeros((ny, nx))

    for si in config.soil_types:
        rec = SOIL_DB[si]
        mask = soil_map == si
        Sy_map[mask] = rec.sy_lit
        alpha_map[mask] = rec.alpha_recharge
        tau_map[mask] = rec.tau

    # 토양별 면적 비율
    total_cells = ny * nx
    soil_fractions = {}
    for si in config.soil_types:
        count = int(np.sum(soil_map == si))
        soil_fractions[si] = count / total_cells

    # 관측정 배치
    well_rows, well_cols = _place_wells(ny, nx, config.well_density, rng)

    return SyntheticDomain(
        config=config,
        x_centers=x_centers,
        y_centers=y_centers,
        soil_map=soil_map,
        Sy_map=Sy_map,
        alpha_map=alpha_map,
        tau_map=tau_map,
        well_rows=well_rows,
        well_cols=well_cols,
        soil_fractions=soil_fractions,
    )


# ──────────────────────────────────────────────────────────
# CLI / 시각화
# ──────────────────────────────────────────────────────────
def plot_domain(domain: SyntheticDomain, save_path: str | None = None):
    """Plot soil map with well locations.

    논문용 그래프 품질: 300 dpi, 영문 라벨.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm

    cfg = domain.config
    soil_types = sorted(domain.soil_fractions.keys())
    n_soils = len(soil_types)

    # 컬러맵: 토양 유형별 구분 색상
    base_colors = ['#E8D44D', '#8FBC8F', '#8B4513', '#B0C4DE', '#CD853F']
    colors = base_colors[:n_soils]
    cmap = ListedColormap(colors)
    bounds = [soil_types[0] - 0.5] + [
        (soil_types[i] + soil_types[i+1]) / 2 for i in range(n_soils - 1)
    ] + [soil_types[-1] + 0.5]
    norm = BoundaryNorm(bounds, cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) 토양 분포
    ax = axes[0]
    extent = [0, cfg.nx * cfg.dx / 1000, 0, cfg.ny * cfg.dy / 1000]
    im = ax.imshow(domain.soil_map, origin='lower', extent=extent,
                   cmap=cmap, norm=norm, interpolation='nearest')

    # 관측정 위치
    wx = domain.x_centers[domain.well_cols] / 1000
    wy = domain.y_centers[domain.well_rows] / 1000
    ax.scatter(wx, wy, c='red', marker='^', s=40, edgecolors='k',
               linewidths=0.5, zorder=5, label=f'Wells (n={domain.n_wells})')
    ax.legend(loc='upper right', fontsize=9)

    # 컬러바 라벨
    cbar = fig.colorbar(im, ax=ax, ticks=soil_types, shrink=0.85)
    cbar_labels = [SOIL_DB[si].name for si in soil_types]
    cbar.ax.set_yticklabels(cbar_labels, fontsize=8)

    ax.set_xlabel('X (km)', fontsize=11)
    ax.set_ylabel('Y (km)', fontsize=11)
    ax.set_title(f'(a) Soil type distribution — {cfg.name}', fontsize=12)

    # (b) Sy 분포
    ax = axes[1]
    im2 = ax.imshow(domain.Sy_map, origin='lower', extent=extent,
                    cmap='viridis', interpolation='nearest')
    ax.scatter(wx, wy, c='red', marker='^', s=40, edgecolors='k',
               linewidths=0.5, zorder=5)
    cbar2 = fig.colorbar(im2, ax=ax, shrink=0.85)
    cbar2.set_label('Specific yield [-]', fontsize=10)
    ax.set_xlabel('X (km)', fontsize=11)
    ax.set_ylabel('Y (km)', fontsize=11)
    ax.set_title(f'(b) Specific yield (Sy) — {cfg.name}', fontsize=12)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    else:
        plt.show()

    plt.close(fig)
    return fig


if __name__ == "__main__":
    # S3 시나리오 생성 및 확인
    domain = generate_domain(DomainConfig.S3())
    print(domain.summary())
    print()

    # 그래프 저장
    out_dir = os.path.dirname(os.path.abspath(__file__))
    plot_domain(domain, save_path=os.path.join(out_dir, "domain_S3.png"))
