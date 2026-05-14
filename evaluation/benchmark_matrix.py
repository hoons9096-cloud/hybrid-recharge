"""benchmark_matrix.py — Phase 4: S1~S5 × N methods 벤치마크.

논문 Table/Figure 의 핵심 수치 산출.

각 시나리오 × 각 방법의 함양 추정값을 정답(true_recharge_annual) 과 비교:
    RMSE_grid = sqrt( mean( (R_est - R_true)^2 ) )       — 격자 전체
    RMSE_well = sqrt( mean( (R_est_well - R_true_well)^2 ) )  — 관정 위치만
    MAE       = mean( |R_est - R_true| )
    Bias      = mean( R_est - R_true )
    Spatial r = Pearson corr(R_est_flat, R_true_flat)
    rRMSE     = RMSE / mean(R_true) × 100  [%]

방법:
    Lumped         — methods/wtf_lumped.py
    Soil-weighted  — methods/wtf_soil_weighted.py
    EnKF           — methods/wtf_enkf_spatial.py

정답 모델 (recharge_model):
    "alpha"   — 기존 단순 alpha × (P-ET)
    "cascade" — Phase 2 multi-layer vadose

Usage
-----
    python -m evaluation.benchmark_matrix --truth cascade --output benchmark.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# 프로젝트 루트
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Metric utilities
# ---------------------------------------------------------------------------
@dataclass
class MethodMetrics:
    scenario: str
    method: str
    truth_model: str
    rmse_grid: float
    rmse_well: float
    mae: float
    bias: float
    spatial_r: float
    rrmse_pct: float
    elapsed_sec: float
    mean_true: float
    mean_est: float
    notes: str = ""


def _compute_metrics(
    R_est: np.ndarray, R_true: np.ndarray,
    well_rows: np.ndarray, well_cols: np.ndarray,
) -> Dict[str, float]:
    """추정 vs 정답 격자 비교."""
    diff = R_est - R_true
    rmse_grid = float(np.sqrt(np.nanmean(diff ** 2)))
    mae = float(np.nanmean(np.abs(diff)))
    bias = float(np.nanmean(diff))
    mean_true = float(np.nanmean(R_true))
    mean_est = float(np.nanmean(R_est))
    rrmse = (rmse_grid / max(abs(mean_true), 1e-9)) * 100.0

    # 관정 위치만
    R_est_w = R_est[well_rows, well_cols]
    R_true_w = R_true[well_rows, well_cols]
    rmse_well = float(np.sqrt(np.nanmean((R_est_w - R_true_w) ** 2)))

    # 공간 상관
    flat_e = R_est.flatten(); flat_t = R_true.flatten()
    if np.std(flat_e) > 1e-9 and np.std(flat_t) > 1e-9:
        sr = float(np.corrcoef(flat_e, flat_t)[0, 1])
    else:
        sr = float("nan")

    return {
        "rmse_grid": rmse_grid,
        "rmse_well": rmse_well,
        "mae": mae,
        "bias": bias,
        "spatial_r": sr,
        "rrmse_pct": rrmse,
        "mean_true": mean_true,
        "mean_est": mean_est,
    }


# ---------------------------------------------------------------------------
# Benchmark 실행
# ---------------------------------------------------------------------------
def _build_observations(domain, data) -> Dict:
    """methods 가 기대하는 observations dict 구성."""
    well_soil_types = np.array([
        int(domain.soil_map[domain.well_rows[w], domain.well_cols[w]])
        for w in range(domain.n_wells)
    ])
    return {
        "P": data.P,
        "ET": data.ET,
        "ho_obs": data.ho_obs,
        "well_soil_types": well_soil_types,
    }


def _run_single_method(
    method_name: str, domain, data,
) -> tuple:
    """method 실행 → (R_est_map, elapsed_sec, error_msg)"""
    obs = _build_observations(domain, data)
    t0 = time.perf_counter()
    err = ""
    try:
        if method_name == "Lumped":
            from methods.wtf_lumped import estimate_recharge as fn
            R = fn(domain, obs)
        elif method_name == "Soil-weighted":
            from methods.wtf_soil_weighted import estimate_recharge as fn
            R = fn(domain, obs)
        elif method_name == "EnKF":
            from methods.wtf_enkf_spatial import estimate_recharge as fn
            R = fn(domain, obs)
        elif method_name == "Hierarchical":
            from methods.wtf_hierarchical import estimate_recharge as fn
            R = fn(domain, obs)
        elif method_name == "Bias-corrected":
            from methods.wtf_bias_corrected import estimate_recharge as fn
            R = fn(domain, obs)
        else:
            raise ValueError(f"Unknown method: {method_name}")
    except Exception as e:
        return None, time.perf_counter() - t0, str(e)
    return R, time.perf_counter() - t0, err


def run_benchmark(
    scenarios: Optional[List[str]] = None,
    methods: Optional[List[str]] = None,
    truth_models: Optional[List[str]] = None,
    n_days: int = 730,
    verbose: bool = True,
) -> pd.DataFrame:
    """전체 매트릭스 실행 → DataFrame."""
    from synthetic.scenarios import _CONFIG_FACTORY
    from synthetic.generate_domain import generate_domain
    from synthetic.generate_data import generate_data

    if scenarios is None:
        scenarios = ["S1", "S2", "S3", "S4", "S5"]
    if methods is None:
        methods = ["Lumped", "Soil-weighted", "Bias-corrected",
                   "Hierarchical", "EnKF"]
    if truth_models is None:
        truth_models = ["alpha", "cascade"]

    rows: List[MethodMetrics] = []

    for scn in scenarios:
        config = _CONFIG_FACTORY[scn]()
        domain = generate_domain(config)
        if verbose:
            print(f"\n=== {scn} ({domain.config.ny}x{domain.config.nx}, "
                  f"{domain.n_wells} wells) ===")

        for truth in truth_models:
            data = generate_data(domain, n_days=n_days, recharge_model=truth)
            R_true = data.true_recharge_annual

            for m in methods:
                R_est, dt, err = _run_single_method(m, domain, data)
                if R_est is None:
                    if verbose:
                        print(f"  {truth:7s} | {m:14s} : FAILED  ({err[:60]})")
                    rows.append(MethodMetrics(
                        scenario=scn, method=m, truth_model=truth,
                        rmse_grid=float("nan"), rmse_well=float("nan"),
                        mae=float("nan"), bias=float("nan"),
                        spatial_r=float("nan"), rrmse_pct=float("nan"),
                        elapsed_sec=dt, mean_true=float(np.mean(R_true)),
                        mean_est=float("nan"), notes=err[:200],
                    ))
                    continue

                metr = _compute_metrics(
                    R_est, R_true, domain.well_rows, domain.well_cols,
                )
                rows.append(MethodMetrics(
                    scenario=scn, method=m, truth_model=truth,
                    elapsed_sec=dt,
                    **metr,
                ))
                if verbose:
                    print(
                        f"  {truth:7s} | {m:14s} : "
                        f"RMSE={metr['rmse_grid']:6.1f}  "
                        f"MAE={metr['mae']:6.1f}  "
                        f"r={metr['spatial_r']:+.3f}  "
                        f"rRMSE={metr['rrmse_pct']:5.1f}%  "
                        f"({dt:.1f}s)"
                    )

    return pd.DataFrame([asdict(r) for r in rows])


# ---------------------------------------------------------------------------
# 결과 요약 출력
# ---------------------------------------------------------------------------
def print_summary(df: pd.DataFrame) -> None:
    """논문용 표 형식 출력."""
    print("\n" + "═" * 100)
    print("BENCHMARK MATRIX SUMMARY (RMSE in mm/yr, lower is better)")
    print("═" * 100)
    for truth in df["truth_model"].unique():
        sub = df[df["truth_model"] == truth]
        pivot = sub.pivot_table(
            index="scenario", columns="method",
            values="rmse_grid", aggfunc="first",
        )
        print(f"\nTruth model: {truth}")
        print(pivot.round(1).to_string())

    print("\n" + "─" * 100)
    print("Spatial correlation (higher is better, 1.0 = perfect)")
    for truth in df["truth_model"].unique():
        sub = df[df["truth_model"] == truth]
        pivot = sub.pivot_table(
            index="scenario", columns="method",
            values="spatial_r", aggfunc="first",
        )
        print(f"\nTruth model: {truth}")
        print(pivot.round(3).to_string())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Phase 4 benchmark matrix")
    ap.add_argument("--scenarios", nargs="+", default=None,
                    help="시나리오 (예: S1 S3 S5)")
    ap.add_argument("--methods", nargs="+", default=None,
                    help="방법 (Lumped, Soil-weighted, EnKF)")
    ap.add_argument("--truth", nargs="+", default=None,
                    help='정답 모델 ("alpha", "cascade")')
    ap.add_argument("--n_days", type=int, default=730)
    ap.add_argument("--output", type=str, default="benchmark_results.csv",
                    help="결과 CSV 저장 경로")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    df = run_benchmark(
        scenarios=args.scenarios, methods=args.methods,
        truth_models=args.truth, n_days=args.n_days,
        verbose=not args.quiet,
    )
    df.to_csv(args.output, index=False)
    print(f"\n✓ Saved: {args.output}  ({len(df)} rows)")
    print_summary(df)


if __name__ == "__main__":
    main()
