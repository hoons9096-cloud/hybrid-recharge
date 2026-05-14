"""
metrics.py -- 함양량 추정 결과 평가 지표

추정된 함양량 맵을 참값(true recharge)과 비교하여
RMSE, MAE, bias, 공간 상관계수 등을 산출한다.

Usage:
    from evaluation.metrics import compute_metrics, compare_methods
    m = compute_metrics(estimated, true, method_name="Lumped", scenario_name="S3")
"""
from __future__ import annotations

import sys
import os
from dataclasses import dataclass

import numpy as np

# 프로젝트 루트 경로 설정
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ──────────────────────────────────────────────────────────
# 평가 지표 데이터 클래스
# ──────────────────────────────────────────────────────────
@dataclass
class EvalMetrics:
    """Evaluation metrics for one method on one scenario."""
    method_name: str
    scenario_name: str
    rmse: float          # Root Mean Square Error [mm/yr]
    mae: float           # Mean Absolute Error [mm/yr]
    bias: float          # Mean bias (estimated - true) [mm/yr]
    r_spatial: float     # Spatial correlation coefficient [-]
    rmse_pct: float      # RMSE as % of mean true recharge


# ──────────────────────────────────────────────────────────
# 개별 지표 계산
# ──────────────────────────────────────────────────────────
def compute_metrics(
    estimated: np.ndarray,
    true: np.ndarray,
    method_name: str = "",
    scenario_name: str = "",
) -> EvalMetrics:
    """Compute all evaluation metrics.

    Parameters
    ----------
    estimated : np.ndarray
        (ny, nx) estimated recharge map [mm/yr].
    true : np.ndarray
        (ny, nx) true recharge map [mm/yr].
    method_name : str
        Name of the estimation method.
    scenario_name : str
        Name of the scenario (e.g. "S1", "S3").

    Returns
    -------
    EvalMetrics
        Computed metrics.
    """
    # 입력 검증
    if estimated.shape != true.shape:
        raise ValueError(
            f"Shape mismatch: estimated {estimated.shape} vs true {true.shape}"
        )

    diff = estimated - true

    # RMSE
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    # MAE
    mae = float(np.mean(np.abs(diff)))

    # Bias (평균 편향)
    bias = float(np.mean(diff))

    # 공간 상관계수
    r_spatial = float(np.corrcoef(estimated.ravel(), true.ravel())[0, 1])

    # RMSE를 참값 평균 대비 백분율로 환산
    mean_true = float(np.mean(true))
    if mean_true != 0.0:
        rmse_pct = rmse / abs(mean_true) * 100.0
    else:
        rmse_pct = float("inf")

    return EvalMetrics(
        method_name=method_name,
        scenario_name=scenario_name,
        rmse=rmse,
        mae=mae,
        bias=bias,
        r_spatial=r_spatial,
        rmse_pct=rmse_pct,
    )


# ──────────────────────────────────────────────────────────
# 복수 방법 비교
# ──────────────────────────────────────────────────────────
def compare_methods(
    results: dict,
    true: np.ndarray,
    scenario_name: str = "",
) -> list[EvalMetrics]:
    """Compare multiple methods against truth.

    Parameters
    ----------
    results : dict
        {method_name: estimated_map} where each map is (ny, nx) [mm/yr].
    true : np.ndarray
        (ny, nx) true recharge map [mm/yr].
    scenario_name : str
        Name of the scenario.

    Returns
    -------
    list[EvalMetrics]
        One EvalMetrics per method, sorted by RMSE ascending.
    """
    metrics_list = []
    for method_name, est_map in results.items():
        m = compute_metrics(est_map, true,
                            method_name=method_name,
                            scenario_name=scenario_name)
        metrics_list.append(m)

    # RMSE 기준 오름차순 정렬
    metrics_list.sort(key=lambda m: m.rmse)
    return metrics_list


# ──────────────────────────────────────────────────────────
# 텍스트 테이블 출력
# ──────────────────────────────────────────────────────────
def metrics_table(metrics_list: list[EvalMetrics]) -> str:
    """Format metrics as a text table for printing.

    Parameters
    ----------
    metrics_list : list[EvalMetrics]
        Metrics to display.

    Returns
    -------
    str
        Formatted text table.
    """
    # 헤더
    header = (
        f"{'Scenario':<10s} {'Method':<22s} "
        f"{'RMSE':>8s} {'MAE':>8s} {'Bias':>8s} "
        f"{'r_sp':>6s} {'RMSE%':>7s}"
    )
    sep = "-" * len(header)

    lines = [sep, header, sep]
    for m in metrics_list:
        line = (
            f"{m.scenario_name:<10s} {m.method_name:<22s} "
            f"{m.rmse:8.2f} {m.mae:8.2f} {m.bias:8.2f} "
            f"{m.r_spatial:6.3f} {m.rmse_pct:6.1f}%"
        )
        lines.append(line)
    lines.append(sep)

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────
# 테스트
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 더미 데이터로 동작 확인
    rng = np.random.default_rng(42)

    ny, nx = 50, 50
    true_map = 100.0 + 50.0 * rng.standard_normal((ny, nx))
    true_map = np.clip(true_map, 10.0, 300.0)

    # 방법 1: 전역 평균 (Lumped)
    lumped = np.full((ny, nx), np.mean(true_map))

    # 방법 2: 참값 + 작은 노이즈 (Soil-weighted 모사)
    soil_weighted = true_map + 10.0 * rng.standard_normal((ny, nx))

    # 방법 3: 참값 + 아주 작은 노이즈 (EnKF 모사)
    enkf = true_map + 5.0 * rng.standard_normal((ny, nx))

    results = {
        "Lumped WTF": lumped,
        "Soil-weighted WTF": soil_weighted,
        "EnKF spatial": enkf,
    }

    mlist = compare_methods(results, true_map, scenario_name="S3")
    print(metrics_table(mlist))
