"""bayes_sy.py — Bayesian posterior for specific yield (Sy) and recharge ratio.

Phase 1 (단순버전) — numpy + scipy 만 사용 (PyMC 없음).
방법: Importance sampling

수학적 모형
----------
1. Prior (HSG + 대수층 타입 → USDA texture → Carsel-Parrish 1988):
       Sy_prior ~ TruncatedNormal(μ, σ, [0.01, 0.45])
       μ = soil_db.SOIL_DB[sn-1].sy_lit   (sn = HSG·aquifer 매핑)
       σ = 0.05  (Healy 2010 권고 기본값)

2. Likelihood (관측 Sy_eff 와의 Gaussian fit):
       P(Sy_eff_obs | Sy) ∝ exp(-(Sy - Sy_eff_obs)² / 2σ_obs²)
       σ_obs = 0.02  (WTF 추정 Sy 의 불확실성)

3. (옵션) Pumping test Sy 추가 likelihood:
       P(Sy_pump | Sy) ∝ exp(-(Sy - Sy_pump)² / 2σ_pump²)
       σ_pump = 0.005  (양수시험은 strong evidence)

4. Posterior:
       w_i ∝ likelihood(Sy_i)
       Posterior summary: weighted mean, sd, 95% CI

5. Recharge ratio posterior (WTF identity R = Sy × Σdh / P):
       각 Sy_sample 별로 rech_pct 계산 → 분포

References
----------
Healy, R.W. (2010). Estimating Groundwater Recharge.  Cambridge UP.
Carsel, R.F. & Parrish, R.S. (1988). WRR, 24(5), 755-769.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.stats import truncnorm

from soil_db import SOIL_DB


# ---------------------------------------------------------------------------
# HSG + 대수층 → 대표 USDA texture sn_idx (watershed_aggregator 와 일관)
# ---------------------------------------------------------------------------
HSG_AQUIFER_TO_SN = {
    ("A", "alluvial"): 2,    # Loamy Sand
    ("B", "alluvial"): 3,    # Sandy Loam
    ("C", "alluvial"): 12,   # Loam
    ("D", "alluvial"): 4,    # Silt Loam
    ("A", "bedrock"):  2,    # Loamy Sand
    ("B", "bedrock"):  12,   # Loam
    ("C", "bedrock"):  10,   # Clay Loam
    ("D", "bedrock"):  6,    # Clay
}

# 기본 prior SD (Healy 2010, Table 5.2)
DEFAULT_SY_PRIOR_SD = 0.05
# WTF Sy_eff 의 관측 불확실성
DEFAULT_OBS_SD = 0.02
# 양수시험 Sy 의 불확실성 (strong likelihood)
DEFAULT_PUMP_SD = 0.005

# Sy 물리적 한계
SY_MIN, SY_MAX = 0.01, 0.45


@dataclass
class BayesSyResult:
    # Posterior on Sy
    sy_prior_mean: float
    sy_prior_sd: float
    sy_post_mean: float
    sy_post_sd: float
    sy_post_lo95: float
    sy_post_hi95: float

    # Posterior on recharge ratio (%)
    rech_pct_post_mean: float
    rech_pct_post_lo95: float
    rech_pct_post_hi95: float
    rech_pct_post_sd: float

    # 진단
    n_eff: float            # effective sample size
    n_samples: int
    converged: bool         # n_eff > 100 이면 True

    # 메타
    hsg: str
    aquifer: str
    sn_used: int
    pump_test_sy: Optional[float] = None
    notes: str = ""


# ---------------------------------------------------------------------------
# Prior 구성
# ---------------------------------------------------------------------------
def get_prior_params(
    hsg: str, aquifer: str = "bedrock",
    sy_prior_sd: float = DEFAULT_SY_PRIOR_SD,
) -> Tuple[float, float, int]:
    """HSG + 대수층 → (Sy_mean, Sy_sd, sn_idx)."""
    key = (hsg, aquifer)
    if key not in HSG_AQUIFER_TO_SN:
        raise ValueError(f"Unsupported (hsg, aquifer): {key}")
    sn = HSG_AQUIFER_TO_SN[key]
    sy_lit = SOIL_DB[sn - 1].sy_lit
    return float(sy_lit), float(sy_prior_sd), sn


def sample_prior(
    mean: float, sd: float, n: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """TruncatedNormal(mean, sd, [SY_MIN, SY_MAX]) 샘플링."""
    rng = rng or np.random.default_rng()
    a = (SY_MIN - mean) / sd
    b = (SY_MAX - mean) / sd
    return truncnorm.rvs(a, b, loc=mean, scale=sd, size=n, random_state=rng)


# ---------------------------------------------------------------------------
# Likelihood (log scale, 수치 안정성)
# ---------------------------------------------------------------------------
def _log_lik_obs(sy_samples: np.ndarray, sy_obs: float, sigma: float) -> np.ndarray:
    """log N(sy_obs ; sy, sigma)."""
    return -0.5 * ((sy_samples - sy_obs) / sigma) ** 2


# ---------------------------------------------------------------------------
# Importance sampling 핵심
# ---------------------------------------------------------------------------
def posterior_sy(
    hsg: str,
    aquifer: str = "bedrock",
    sy_eff_obs: Optional[float] = None,    # WTF 관측치 (없으면 likelihood 무력)
    pump_test_sy: Optional[float] = None,  # 양수시험 (있으면 strong likelihood)
    cumulative_dh_m: Optional[float] = None,  # Σ rise (수위 상승 합) — recharge 계산용
    P_total_m: Optional[float] = None,        # 같은 기간 누적 강수 (m)
    sy_prior_sd: float = DEFAULT_SY_PRIOR_SD,
    obs_sd: float = DEFAULT_OBS_SD,
    pump_sd: float = DEFAULT_PUMP_SD,
    n_samples: int = 5000,
    rng: Optional[np.random.Generator] = None,
) -> BayesSyResult:
    """Sy + recharge 후행분포 추정.

    sy_eff_obs:
        WTF 알고리즘이 추정한 Sy (예: result_v27['Sy_eff']).
        None 이면 likelihood 적용 안 함 → posterior = prior.

    pump_test_sy:
        양수시험에서 측정된 Sy (있으면 strong likelihood 추가).

    cumulative_dh_m, P_total_m:
        있으면 rech_pct posterior 도 계산:
            rech_pct = Sy × Σdh / P × 100
        없으면 Sy posterior 만 반환 (rech 필드는 NaN).
    """
    rng = rng or np.random.default_rng(0)

    # 1. Prior
    mu0, sd0, sn = get_prior_params(hsg, aquifer, sy_prior_sd)
    sy_samples = sample_prior(mu0, sd0, n_samples, rng=rng)

    # 2. Likelihood
    log_lik = np.zeros(n_samples)
    if sy_eff_obs is not None and np.isfinite(sy_eff_obs) and sy_eff_obs > 0:
        log_lik += _log_lik_obs(sy_samples, sy_eff_obs, obs_sd)
    if pump_test_sy is not None and np.isfinite(pump_test_sy) and pump_test_sy > 0:
        log_lik += _log_lik_obs(sy_samples, pump_test_sy, pump_sd)

    # 3. Importance weights
    log_lik -= log_lik.max()  # 수치 안정성
    weights = np.exp(log_lik)
    weights /= weights.sum()

    # ESS — Kong 1992
    n_eff = 1.0 / np.sum(weights ** 2)

    # 4. Posterior summary on Sy
    sy_post_mean = float(np.sum(weights * sy_samples))
    sy_post_var = float(np.sum(weights * (sy_samples - sy_post_mean) ** 2))
    sy_post_sd = float(np.sqrt(max(sy_post_var, 0.0)))
    sy_lo95, sy_hi95 = _weighted_quantiles(sy_samples, weights, [0.025, 0.975])

    # 5. Recharge posterior
    if (cumulative_dh_m is not None and P_total_m is not None
            and np.isfinite(cumulative_dh_m) and np.isfinite(P_total_m)
            and P_total_m > 0):
        rech_pct_samples = sy_samples * cumulative_dh_m / P_total_m * 100.0
        r_mean = float(np.sum(weights * rech_pct_samples))
        r_var = float(np.sum(weights * (rech_pct_samples - r_mean) ** 2))
        r_sd = float(np.sqrt(max(r_var, 0.0)))
        r_lo95, r_hi95 = _weighted_quantiles(rech_pct_samples, weights, [0.025, 0.975])
    else:
        r_mean = r_sd = r_lo95 = r_hi95 = float("nan")

    return BayesSyResult(
        sy_prior_mean=mu0,
        sy_prior_sd=sd0,
        sy_post_mean=sy_post_mean,
        sy_post_sd=sy_post_sd,
        sy_post_lo95=float(sy_lo95),
        sy_post_hi95=float(sy_hi95),
        rech_pct_post_mean=r_mean,
        rech_pct_post_sd=r_sd,
        rech_pct_post_lo95=r_lo95,
        rech_pct_post_hi95=r_hi95,
        n_eff=float(n_eff),
        n_samples=n_samples,
        converged=bool(n_eff > 100),
        hsg=hsg, aquifer=aquifer, sn_used=sn,
        pump_test_sy=pump_test_sy,
    )


# ---------------------------------------------------------------------------
# Helper: weighted quantiles
# ---------------------------------------------------------------------------
def _weighted_quantiles(
    values: np.ndarray, weights: np.ndarray, q_list,
) -> Tuple[float, ...]:
    """Weighted quantiles via cumulative weight."""
    order = np.argsort(values)
    v_sorted = values[order]
    w_sorted = weights[order]
    cum = np.cumsum(w_sorted) / w_sorted.sum()
    out = []
    for q in q_list:
        idx = int(np.searchsorted(cum, q))
        idx = min(max(idx, 0), len(v_sorted) - 1)
        out.append(float(v_sorted[idx]))
    return tuple(out)


# ---------------------------------------------------------------------------
# Convenience wrapper — result_v27 → BayesSyResult
# ---------------------------------------------------------------------------
def from_result_v27(
    result_v27: Dict,
    hsg: str,
    aquifer: str = "bedrock",
    pump_test_sy: Optional[float] = None,
    n_samples: int = 5000,
) -> BayesSyResult:
    """result_v27 dict 로부터 자동 추출 후 posterior 계산.

    함양율은 *result_v27 의 recharge_ratio 를 Sy 비례로 스케일링* —
    Σdh 단순 누적은 일별 노이즈까지 합산되어 비물리적 값을 만들기 때문.

    R_post% = (Sy_post / Sy_obs) × R_obs%
    """
    sy_eff = float(result_v27.get("Sy_eff", 0.0)) or None
    rech_pct_obs = float(result_v27.get("recharge_ratio", 0.0))

    rng = np.random.default_rng(0)
    # 1. Sy posterior
    res = posterior_sy(
        hsg=hsg, aquifer=aquifer,
        sy_eff_obs=sy_eff,
        pump_test_sy=pump_test_sy,
        cumulative_dh_m=None,    # 함양 분포는 별도 산출 (아래)
        P_total_m=None,
        n_samples=n_samples,
        rng=rng,
    )

    # 2. 함양율 posterior — Sy_post / Sy_obs * Rech_obs
    if sy_eff is not None and sy_eff > 0 and rech_pct_obs > 0:
        # Sy_post 샘플 재생성 (동일 seed → posterior 분포 동일)
        from scipy.stats import truncnorm
        rng2 = np.random.default_rng(0)
        mu0 = res.sy_prior_mean
        sd0 = res.sy_prior_sd
        a = (SY_MIN - mu0) / sd0
        b = (SY_MAX - mu0) / sd0
        sy_samples = truncnorm.rvs(a, b, loc=mu0, scale=sd0,
                                    size=n_samples, random_state=rng2)
        log_lik = np.zeros(n_samples)
        log_lik += -0.5 * ((sy_samples - sy_eff) / DEFAULT_OBS_SD) ** 2
        if pump_test_sy is not None and pump_test_sy > 0:
            log_lik += -0.5 * ((sy_samples - pump_test_sy)
                                / DEFAULT_PUMP_SD) ** 2
        log_lik -= log_lik.max()
        weights = np.exp(log_lik); weights /= weights.sum()

        rech_pct_samples = (sy_samples / sy_eff) * rech_pct_obs
        # 물리적 cap (recharge ≤ 100% of P)
        rech_pct_samples = np.clip(rech_pct_samples, 0.0, 100.0)

        r_mean = float(np.sum(weights * rech_pct_samples))
        r_var = float(np.sum(weights * (rech_pct_samples - r_mean) ** 2))
        r_sd = float(np.sqrt(max(r_var, 0.0)))
        r_lo, r_hi = _weighted_quantiles(rech_pct_samples, weights,
                                          [0.025, 0.975])
        # res 갱신
        res.rech_pct_post_mean = r_mean
        res.rech_pct_post_sd = r_sd
        res.rech_pct_post_lo95 = float(r_lo)
        res.rech_pct_post_hi95 = float(r_hi)

    return res
