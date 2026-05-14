"""uncertainty_calibration.py — Bayesian posterior calibration diagnostics.

리뷰어 지적 #1 — "Are your uncertainties calibrated?" 에 대한 응답.

세 가지 표준 calibration 검증:

  1. Coverage probability      — 95% CI 가 실제로 95% truth 를 포함하는가?
  2. PIT (Probability Integral Transform) histogram — posterior 가 잘
                                  calibrated 되어 있으면 uniform 분포여야.
  3. CRPS (Continuous Ranked Probability Score) — 점추정 + 분포추정의
                                  종합 점수 (작을수록 좋음).

References
----------
Dawid, A. P. (1984). Statistical theory: the prequential approach. JRSS A.
Gneiting, T. & Raftery, A. E. (2007). Strictly proper scoring rules,
    prediction, and estimation. JASA, 102(477).
Hersbach, H. (2000). Decomposition of the continuous ranked probability
    score. Weather and Forecasting, 15(5).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CalibrationResult:
    n_truths: int                 # 평가 truth 개수
    coverage_50: float            # 50% CI 내 비율 (이상적: 0.50)
    coverage_90: float            # 90% (이상적: 0.90)
    coverage_95: float            # 95% (이상적: 0.95)
    pit_mean: float               # PIT 평균 (이상적: 0.5)
    pit_var: float                # PIT 분산 (이상적: 1/12 ≈ 0.083)
    pit_ks_pvalue: float          # KS test against uniform — high p = good
    crps_mean: float              # 평균 CRPS (작을수록 좋음)
    crps_per_truth: np.ndarray    # truth-level CRPS
    pit_values: np.ndarray        # PIT histogram raw values

    @property
    def well_calibrated(self) -> bool:
        """5%p 이내 coverage, KS p > 0.05 면 calibrated."""
        return (
            abs(self.coverage_95 - 0.95) < 0.05
            and abs(self.coverage_90 - 0.90) < 0.07
            and self.pit_ks_pvalue > 0.05
        )


# ---------------------------------------------------------------------------
# Coverage probability
# ---------------------------------------------------------------------------
def coverage(samples: np.ndarray, truth: np.ndarray, level: float = 0.95) -> float:
    """truth 가 (1-level)/2 ~ (1+level)/2 quantile 사이에 있는 비율.

    samples: (n_truths, n_samples) 또는 (n_samples,) — 단일 truth
    truth:   (n_truths,) 또는 scalar
    """
    samples = np.atleast_2d(samples)
    truth = np.atleast_1d(truth)
    if samples.shape[0] != len(truth):
        if samples.shape[1] == len(truth):
            samples = samples.T
        else:
            raise ValueError(
                f"samples {samples.shape} vs truth {truth.shape} 불일치")
    lo_q = (1 - level) / 2
    hi_q = 1 - lo_q
    lo = np.quantile(samples, lo_q, axis=1)
    hi = np.quantile(samples, hi_q, axis=1)
    inside = (truth >= lo) & (truth <= hi)
    return float(np.mean(inside))


# ---------------------------------------------------------------------------
# PIT (Probability Integral Transform)
# ---------------------------------------------------------------------------
def pit_values(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """각 truth 에 대해 P(X ≤ truth) 의 empirical 추정."""
    samples = np.atleast_2d(samples)
    truth = np.atleast_1d(truth)
    if samples.shape[0] != len(truth) and samples.shape[1] == len(truth):
        samples = samples.T
    pit = np.array([
        float(np.mean(samples[i] <= truth[i]))
        for i in range(len(truth))
    ])
    return pit


def pit_uniformity_test(pit: np.ndarray) -> Tuple[float, float]:
    """KS test against uniform[0,1].  Returns (statistic, p-value)."""
    from scipy.stats import kstest
    stat, p = kstest(pit, "uniform")
    return float(stat), float(p)


# ---------------------------------------------------------------------------
# CRPS — Hersbach 2000 closed form for empirical samples
# ---------------------------------------------------------------------------
def crps_ensemble(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """CRPS for ensemble forecast.

    CRPS(F, y) = E|X - y| - 0.5 E|X - X'|
              ≈ mean|samples - y| - 0.5 mean_pairs|s_i - s_j|
    """
    samples = np.atleast_2d(samples)
    truth = np.atleast_1d(truth)
    if samples.shape[0] != len(truth) and samples.shape[1] == len(truth):
        samples = samples.T
    crps = np.zeros(len(truth))
    for i in range(len(truth)):
        s = samples[i]
        y = truth[i]
        term1 = float(np.mean(np.abs(s - y)))
        # term2 — 효율적 형태: 정렬 후 weighted sum (O(N log N))
        s_sorted = np.sort(s)
        n = len(s_sorted)
        idx = np.arange(1, n + 1)
        term2 = float(2.0 / (n * n) * np.sum((idx - (n + 1) / 2.0) * s_sorted))
        crps[i] = term1 - term2
    return crps


# ---------------------------------------------------------------------------
# 종합 calibration 검증
# ---------------------------------------------------------------------------
def assess_calibration(
    posterior_samples: np.ndarray,    # (n_truths, n_samples)
    truth: np.ndarray,                # (n_truths,)
) -> CalibrationResult:
    samples = np.atleast_2d(posterior_samples)
    truth = np.atleast_1d(truth)
    if samples.shape[0] != len(truth) and samples.shape[1] == len(truth):
        samples = samples.T

    cov_50 = coverage(samples, truth, level=0.50)
    cov_90 = coverage(samples, truth, level=0.90)
    cov_95 = coverage(samples, truth, level=0.95)
    pit = pit_values(samples, truth)
    _, pit_p = pit_uniformity_test(pit)
    crps = crps_ensemble(samples, truth)

    return CalibrationResult(
        n_truths=len(truth),
        coverage_50=cov_50,
        coverage_90=cov_90,
        coverage_95=cov_95,
        pit_mean=float(np.mean(pit)),
        pit_var=float(np.var(pit)),
        pit_ks_pvalue=pit_p,
        crps_mean=float(np.mean(crps)),
        crps_per_truth=crps,
        pit_values=pit,
    )


# ---------------------------------------------------------------------------
# Synthetic experiment — 합성 도메인에서 hierarchical calibration 검증
# ---------------------------------------------------------------------------
def calibration_experiment(
    n_replicates: int = 50,
    scenario: str = "S3",
    truth_model: str = "alpha",
    n_walkers: int = 16,
    n_steps: int = 1000,
    burn_in: int = 300,
    verbose: bool = True,
) -> Dict:
    """합성 도메인에서 N 회 추정 → posterior vs known truth → calibration.

    각 replicate 마다 seed 가 다른 합성 데이터 생성 + hierarchical 적합 →
    유역-level 함양율 posterior 와 truth 비교.
    """
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from synthetic.scenarios import _CONFIG_FACTORY
    from synthetic.generate_domain import generate_domain
    from synthetic.generate_data import generate_data
    from soil_db import SOIL_DB
    from bayes_hierarchical import WellObservation, fit_hierarchical
    from methods.wtf_soil_weighted import _wtf_single_well

    posterior_samples = []   # (n_replicates, n_mcmc_samples) — 함양율 % 단위
    truths = []
    for rep in range(n_replicates):
        cfg = _CONFIG_FACTORY[scenario]()
        cfg.random_seed = 1000 + rep
        dom = generate_domain(cfg)
        data = generate_data(dom, n_days=730, recharge_model=truth_model)

        # 유역 수준 truth = grid 평균 함양율 % of P
        n_yr = max(data.n_days / 365.25, 1.0)
        P_annual_mm = float(np.sum(data.P)) * 1000.0 / n_yr
        true_R_annual = float(np.mean(data.true_recharge_annual))
        if P_annual_mm <= 0:
            continue
        truth_pct = true_R_annual / P_annual_mm * 100.0

        # 각 관정의 WTF 추정
        obs_list = []
        for w in range(dom.n_wells):
            soil_idx = int(dom.soil_map[dom.well_rows[w], dom.well_cols[w]])
            sy = SOIL_DB[soil_idx].sy_lit
            tau = SOIL_DB[soil_idx].tau
            R_w = _wtf_single_well(data.ho_obs[w], data.P, sy, tau)
            ho = data.ho_obs[w]
            dh = np.diff(ho)
            cum_rise = float(np.nansum(dh[dh > 0]))
            P_total = float(np.sum(data.P)) / n_yr
            # USDA → HSG (대략)
            hsg_map = {1: "A", 2: "A", 3: "B", 4: "C", 5: "C",
                       6: "D", 7: "D", 8: "C", 9: "D", 10: "C",
                       11: "B", 12: "B"}
            hsg = hsg_map.get(soil_idx, "B")
            obs_list.append(WellObservation(
                name=f"w{w}", hsg=hsg, aquifer="alluvial",
                sy_eff_obs=sy, cumulative_dh_m=cum_rise, P_total_m=P_total,
                soil_area_frac=1.0 / dom.n_wells,
            ))

        try:
            res = fit_hierarchical(
                obs_list, n_walkers=n_walkers, n_steps=n_steps,
                burn_in=burn_in, seed=rep, verbose=False,
            )
        except Exception:
            continue

        # 유역 평균 함양율 posterior — 각 sample별로
        # rech_per_well_sample(s) = sy_well[i] × cum_rise[i]/P_total[i] × 100
        # 가중 평균 (균등)
        n_wells = len(obs_list)
        rech_samples = np.zeros(res.samples_sy_well.shape[0])
        for i, o in enumerate(obs_list):
            if o.cumulative_dh_m and o.P_total_m and o.P_total_m > 0:
                rech_samples += (1.0 / n_wells) * (
                    res.samples_sy_well[:, i]
                    * o.cumulative_dh_m / o.P_total_m * 100.0
                )

        if len(rech_samples) == 0 or not np.isfinite(rech_samples).all():
            continue

        posterior_samples.append(rech_samples)
        truths.append(truth_pct)

        if verbose and (rep + 1) % 10 == 0:
            print(f"  [{rep+1}/{n_replicates}] truth={truth_pct:.2f}%  "
                  f"post mean={float(np.mean(rech_samples)):.2f}% "
                  f"[{float(np.percentile(rech_samples, 2.5)):.2f}, "
                  f"{float(np.percentile(rech_samples, 97.5)):.2f}]")

    if not posterior_samples:
        raise RuntimeError("No successful replicates")

    # truncate to common length
    min_n = min(s.size for s in posterior_samples)
    samples_arr = np.array([s[:min_n] for s in posterior_samples])  # (R, S)
    truth_arr = np.array(truths)

    cal = assess_calibration(samples_arr, truth_arr)
    return {
        "calibration": cal,
        "n_replicates": len(truths),
        "scenario": scenario,
        "truth_model": truth_model,
        "posterior_samples": samples_arr,
        "truths": truth_arr,
    }


# ---------------------------------------------------------------------------
# WTF bias quantification — Section 5.5
# ---------------------------------------------------------------------------
def quantify_wtf_bias(
    n_replicates: int = 30,
    scenarios: Optional[List[str]] = None,
    truth_models: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict:
    """합성 데이터에서 WTF 점추정 vs true recharge — bias 분포."""
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from synthetic.scenarios import _CONFIG_FACTORY
    from synthetic.generate_domain import generate_domain
    from synthetic.generate_data import generate_data

    if scenarios is None:
        scenarios = ["S2", "S3", "S5"]
    if truth_models is None:
        truth_models = ["alpha", "cascade"]

    rows = []
    for scn in scenarios:
        for tm in truth_models:
            true_minus_est = []
            true_vals = []
            est_vals = []
            for rep in range(n_replicates):
                cfg = _CONFIG_FACTORY[scn]()
                cfg.random_seed = 2000 + rep
                dom = generate_domain(cfg)
                data = generate_data(dom, n_days=730, recharge_model=tm)
                # Soil-weighted = 점추정 best
                from methods.wtf_soil_weighted import estimate_recharge as fn
                R_est = fn(dom, {
                    "P": data.P, "ET": data.ET,
                    "ho_obs": data.ho_obs,
                    "well_soil_types": np.array([
                        int(dom.soil_map[dom.well_rows[w], dom.well_cols[w]])
                        for w in range(dom.n_wells)
                    ]),
                })
                R_true = data.true_recharge_annual
                bias = float(np.mean(R_est) - np.mean(R_true))
                true_minus_est.append(bias)
                true_vals.append(float(np.mean(R_true)))
                est_vals.append(float(np.mean(R_est)))
            arr = np.array(true_minus_est)
            rows.append({
                "scenario": scn, "truth_model": tm,
                "mean_bias_mm": float(np.mean(arr)),
                "rel_bias_pct": float(np.mean(arr) / np.mean(true_vals) * 100),
                "bias_sd_mm": float(np.std(arr)),
                "n_rep": len(arr),
                "true_mean_mm": float(np.mean(true_vals)),
                "est_mean_mm": float(np.mean(est_vals)),
            })
            if verbose:
                print(f"  {scn}/{tm}: bias={np.mean(arr):+.1f} mm/yr "
                      f"({np.mean(arr)/np.mean(true_vals)*100:+.1f}%) "
                      f"σ={np.std(arr):.1f}  N={len(arr)}")
    return {"rows": rows}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, json, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="calibration",
                    choices=["calibration", "bias", "both"])
    ap.add_argument("--scenario", default="S3")
    ap.add_argument("--truth", default="alpha")
    ap.add_argument("--n_rep", type=int, default=50)
    ap.add_argument("--output", default="calibration")
    args = ap.parse_args()

    if args.mode in ("calibration", "both"):
        print(f"\n▶ Calibration experiment ({args.scenario}, {args.truth}, "
              f"N={args.n_rep})")
        out = calibration_experiment(
            n_replicates=args.n_rep,
            scenario=args.scenario, truth_model=args.truth,
        )
        cal = out["calibration"]
        print("\n=== Calibration Result ===")
        print(f"  N replicates : {cal.n_truths}")
        print(f"  Coverage 50% : {cal.coverage_50:.3f}  (ideal 0.500)")
        print(f"  Coverage 90% : {cal.coverage_90:.3f}  (ideal 0.900)")
        print(f"  Coverage 95% : {cal.coverage_95:.3f}  (ideal 0.950)")
        print(f"  PIT mean/var : {cal.pit_mean:.3f} / {cal.pit_var:.3f}  "
              f"(ideal 0.500 / 0.083)")
        print(f"  PIT KS p-val : {cal.pit_ks_pvalue:.3f}  (>0.05 = uniform)")
        print(f"  CRPS         : {cal.crps_mean:.3f}")
        print(f"  Calibrated?  : {'✅' if cal.well_calibrated else '⚠️'}")

        # JSON 저장
        with open(f"{args.output}.json", "w") as f:
            json.dump({
                "scenario": args.scenario,
                "truth_model": args.truth,
                "coverage_50": cal.coverage_50,
                "coverage_90": cal.coverage_90,
                "coverage_95": cal.coverage_95,
                "pit_mean": cal.pit_mean,
                "pit_var": cal.pit_var,
                "pit_ks_pvalue": cal.pit_ks_pvalue,
                "crps_mean": cal.crps_mean,
                "well_calibrated": cal.well_calibrated,
                "n_truths": cal.n_truths,
                "pit_values": cal.pit_values.tolist(),
                "truths": out["truths"].tolist(),
            }, f, indent=2)
        # samples 별도 저장 (그림용)
        np.savez(f"{args.output}_samples.npz",
                 posterior=out["posterior_samples"],
                 truth=out["truths"])
        print(f"\n✓ {args.output}.json + {args.output}_samples.npz")

    if args.mode in ("bias", "both"):
        print(f"\n▶ WTF bias quantification (N={args.n_rep})")
        b = quantify_wtf_bias(n_replicates=args.n_rep)
        with open(f"{args.output}_bias.json", "w") as f:
            json.dump(b, f, indent=2)
        print(f"\n✓ {args.output}_bias.json")
