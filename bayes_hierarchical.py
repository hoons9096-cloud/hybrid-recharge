"""bayes_hierarchical.py — Phase 3: 계층 베이지안 모델 (Hierarchical Bayesian).

3-level hierarchy:
    Level 1 (유역): μ_w ~ TruncNormal(prior_mean, prior_sd)
                    유역 전체 평균 Sy
    Level 2 (HSG): μ_h | μ_w, σ_h ~ Normal(μ_w + δ_h_HSG, σ_h)
                   HSG (A/B/C/D) 별 평균 Sy.  δ_h 는 HSG 별 prior 편차.
    Level 3 (관정): Sy_w | μ_h ~ TruncNormal(μ_h, σ_well, [0.01, 0.45])
                    관정별 Sy
    Likelihood: Sy_eff_obs[i] ~ Normal(Sy_w[i], obs_sd)
                + (옵션) pump_test_sy[i] ~ Normal(Sy_w[i], pump_sd)

샘플러: emcee (Foreman-Mackey et al. 2013) — affine-invariant ensemble.

References
----------
Foreman-Mackey, D. et al. (2013). emcee: The MCMC hammer.  PASP 125.
Gelman, A. et al. (2013). Bayesian Data Analysis (3rd ed).  Ch. 5.
Healy, R.W. (2010). Estimating Groundwater Recharge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import emcee
    HAS_EMCEE = True
except ImportError:
    HAS_EMCEE = False

from soil_db import SOIL_DB
from bayes_sy import HSG_AQUIFER_TO_SN


# ---------------------------------------------------------------------------
# 입출력
# ---------------------------------------------------------------------------
@dataclass
class WellObservation:
    """관정 1개의 hierarchical 입력."""
    name: str
    hsg: str                          # "A"/"B"/"C"/"D"
    aquifer: str                      # "alluvial"/"bedrock"
    sy_eff_obs: float                 # WTF 관측 Sy
    pump_test_sy: Optional[float] = None
    cumulative_dh_m: Optional[float] = None
    P_total_m: Optional[float] = None
    soil_area_frac: float = 1.0       # HSG 면적 가중에 사용 (옵션)


@dataclass
class HierarchicalResult:
    """MCMC 결과 + posterior summary."""
    well_names: List[str]
    hsgs: List[str]
    aquifers: List[str]

    # MCMC samples
    n_samples: int                    # walkers × steps (after burn-in)
    samples_mu_watershed: np.ndarray  # (n_samples,)
    samples_mu_hsg: Dict[str, np.ndarray]  # HSG → (n_samples,)
    samples_sy_well: np.ndarray        # (n_samples, n_wells)

    # Posterior summary
    mu_watershed_mean: float
    mu_watershed_lo95: float
    mu_watershed_hi95: float
    mu_hsg_summary: Dict[str, Tuple[float, float, float]]   # HSG → (mean, lo95, hi95)
    sy_well_mean: np.ndarray           # (n_wells,)
    sy_well_lo95: np.ndarray
    sy_well_hi95: np.ndarray

    # Recharge posterior (유역 평균, %)
    rech_pct_watershed_mean: float
    rech_pct_watershed_lo95: float
    rech_pct_watershed_hi95: float

    # MCMC 진단
    mean_acceptance_rate: float
    autocorr_time: Optional[np.ndarray] = None
    converged: bool = False
    n_burn_in: int = 0


# ---------------------------------------------------------------------------
# Prior 구성 — HSG/aquifer 별 sy_lit 평균 활용
# ---------------------------------------------------------------------------
def _hsg_prior_sy(hsg: str, aquifer: str) -> float:
    sn = HSG_AQUIFER_TO_SN[(hsg, aquifer)]
    return float(SOIL_DB[sn].sy_lit)


# ---------------------------------------------------------------------------
# log-posterior
# ---------------------------------------------------------------------------
def _make_log_posterior(
    obs: List[WellObservation],
    prior_watershed_mean: float,
    prior_watershed_sd: float,
    sigma_hsg: float,
    sigma_well: float,
    obs_sd: float,
    pump_sd: float,
):
    """파라미터 벡터 θ = [μ_w, μ_A, μ_B, μ_C, μ_D, Sy_1, Sy_2, ..., Sy_N]"""
    hsg_list = ["A", "B", "C", "D"]
    n_wells = len(obs)
    well_hsg_idx = np.array([hsg_list.index(o.hsg) for o in obs])

    sy_obs = np.array([o.sy_eff_obs for o in obs])
    pump_mask = np.array([o.pump_test_sy is not None for o in obs])
    pump_vals = np.array([o.pump_test_sy if o.pump_test_sy is not None else 0.0 for o in obs])

    # HSG별 prior δ (HSG 평균 - watershed 평균)
    hsg_priors = np.array([
        np.mean([_hsg_prior_sy(h, o.aquifer) for o in obs])
        for h in hsg_list
    ])
    delta_h = hsg_priors - np.mean(hsg_priors)   # HSG 별 편차 prior

    SY_MIN, SY_MAX = 0.01, 0.45

    def log_posterior(theta):
        mu_w = theta[0]
        mu_h = theta[1:5]    # 4개 HSG
        sy_w = theta[5:5 + n_wells]

        # 경계
        if not (SY_MIN <= mu_w <= SY_MAX):
            return -np.inf
        if np.any((sy_w < SY_MIN) | (sy_w > SY_MAX)):
            return -np.inf

        # Level 1: watershed prior — TruncNormal
        log_prior = -0.5 * ((mu_w - prior_watershed_mean) / prior_watershed_sd) ** 2

        # Level 2: HSG ~ Normal(μ_w + δ_h, σ_hsg)
        mu_h_expected = mu_w + delta_h
        log_prior += -0.5 * np.sum(((mu_h - mu_h_expected) / sigma_hsg) ** 2)

        # Level 3: Sy_w ~ TruncNormal(μ_h[hsg_idx], σ_well)
        mu_per_well = mu_h[well_hsg_idx]
        log_prior += -0.5 * np.sum(((sy_w - mu_per_well) / sigma_well) ** 2)

        # Likelihood: WTF 관측 Sy
        log_lik = -0.5 * np.sum(((sy_obs - sy_w) / obs_sd) ** 2)

        # Likelihood: 양수시험
        if pump_mask.any():
            sy_pump = sy_w[pump_mask]
            obs_pump = pump_vals[pump_mask]
            log_lik += -0.5 * np.sum(((obs_pump - sy_pump) / pump_sd) ** 2)

        return log_prior + log_lik

    return log_posterior, hsg_list


# ---------------------------------------------------------------------------
# 메인 샘플러
# ---------------------------------------------------------------------------
def fit_hierarchical(
    observations: List[WellObservation],
    prior_watershed_mean: Optional[float] = None,
    prior_watershed_sd: float = 0.10,
    sigma_hsg: float = 0.04,
    sigma_well: float = 0.03,
    obs_sd: float = 0.02,
    pump_sd: float = 0.005,
    n_walkers: int = 32,
    n_steps: int = 3000,
    burn_in: int = 800,
    seed: int = 0,
    verbose: bool = False,
) -> HierarchicalResult:
    """3-level hierarchical Bayesian 적합.

    n_walkers · (n_steps - burn_in) samples 반환.
    """
    if not HAS_EMCEE:
        raise RuntimeError(
            "emcee 가 설치되어 있지 않습니다.  pip install emcee"
        )
    if not observations:
        raise ValueError("관측 비어 있음")

    n_wells = len(observations)
    n_dim = 1 + 4 + n_wells   # μ_w, μ_A..D, Sy_well_i
    # emcee 권장: n_walkers ≥ 2 × n_dim
    if n_walkers < 2 * n_dim + 2:
        n_walkers = 2 * n_dim + 2

    # prior_watershed_mean 자동 결정 (관측 평균)
    if prior_watershed_mean is None:
        sy_obs_mean = float(np.mean([o.sy_eff_obs for o in observations]))
        prior_watershed_mean = sy_obs_mean
    pwm = float(np.clip(prior_watershed_mean, 0.05, 0.40))

    log_post, hsg_list = _make_log_posterior(
        observations, pwm, prior_watershed_sd,
        sigma_hsg, sigma_well, obs_sd, pump_sd,
    )

    # 초기값 — prior 주변 small jitter
    rng = np.random.default_rng(seed)
    init = np.zeros((n_walkers, n_dim))
    init[:, 0] = pwm + rng.normal(0, 0.02, n_walkers)
    for i in range(4):
        init[:, 1 + i] = pwm + rng.normal(0, 0.03, n_walkers)
    for j, o in enumerate(observations):
        init[:, 5 + j] = o.sy_eff_obs + rng.normal(0, 0.02, n_walkers)
    init = np.clip(init, 0.02, 0.40)

    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_post)
    sampler.run_mcmc(init, n_steps, progress=verbose)

    chain = sampler.get_chain(discard=burn_in, flat=True)   # (n_samples, n_dim)
    accept = float(np.mean(sampler.acceptance_fraction))
    try:
        tau = sampler.get_autocorr_time(tol=0)
    except Exception:
        tau = None

    # 샘플 분리
    s_mu_w = chain[:, 0]
    s_mu_h = {h: chain[:, 1 + i] for i, h in enumerate(hsg_list)}
    s_sy_well = chain[:, 5:5 + n_wells]

    # Posterior summary
    def _ci(arr):
        return float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    mu_w_m, mu_w_lo, mu_w_hi = _ci(s_mu_w)
    mu_h_summary = {h: _ci(s_mu_h[h]) for h in hsg_list}

    sy_w_mean = np.array([np.mean(s_sy_well[:, i]) for i in range(n_wells)])
    sy_w_lo = np.array([np.percentile(s_sy_well[:, i], 2.5) for i in range(n_wells)])
    sy_w_hi = np.array([np.percentile(s_sy_well[:, i], 97.5) for i in range(n_wells)])

    # 유역 함양 posterior (cumulative_dh, P 가 모두 있을 때)
    have_dh = all(o.cumulative_dh_m is not None and o.P_total_m is not None
                  for o in observations)
    if have_dh:
        # 면적 가중치
        weights = np.array([o.soil_area_frac for o in observations])
        if weights.sum() <= 0:
            weights = np.ones_like(weights)
        weights = weights / weights.sum()

        # 각 sample 별 유역 평균 함양율
        rech_per_sample = np.zeros(s_sy_well.shape[0])
        for i, o in enumerate(observations):
            rech_per_sample += weights[i] * (
                s_sy_well[:, i] * o.cumulative_dh_m / o.P_total_m * 100.0
            )
        rech_m = float(np.mean(rech_per_sample))
        rech_lo = float(np.percentile(rech_per_sample, 2.5))
        rech_hi = float(np.percentile(rech_per_sample, 97.5))
    else:
        rech_m = rech_lo = rech_hi = float("nan")

    # 수렴 진단
    converged = (accept > 0.15 and accept < 0.6)
    if tau is not None and len(tau):
        # autocorr time × 50 < n_samples → 수렴 가능성 높음
        eff_steps = (n_steps - burn_in)
        if np.any(tau * 50 > eff_steps):
            converged = False

    return HierarchicalResult(
        well_names=[o.name for o in observations],
        hsgs=[o.hsg for o in observations],
        aquifers=[o.aquifer for o in observations],
        n_samples=chain.shape[0],
        samples_mu_watershed=s_mu_w,
        samples_mu_hsg=s_mu_h,
        samples_sy_well=s_sy_well,
        mu_watershed_mean=mu_w_m,
        mu_watershed_lo95=mu_w_lo,
        mu_watershed_hi95=mu_w_hi,
        mu_hsg_summary=mu_h_summary,
        sy_well_mean=sy_w_mean,
        sy_well_lo95=sy_w_lo,
        sy_well_hi95=sy_w_hi,
        rech_pct_watershed_mean=rech_m,
        rech_pct_watershed_lo95=rech_lo,
        rech_pct_watershed_hi95=rech_hi,
        mean_acceptance_rate=accept,
        autocorr_time=tau,
        converged=converged,
        n_burn_in=burn_in,
    )


# ---------------------------------------------------------------------------
# 편의: stored well_results 로부터 직접 fit
# ---------------------------------------------------------------------------
def fit_from_stored(
    watershed: str,
    well_names: List[str],
    hsg_fractions: Optional[Dict[str, float]] = None,
    **kwargs,
) -> HierarchicalResult:
    """well_results/{name}.json + wells_registry → hierarchical 적합."""
    from well_results_store import load
    from wells_registry import WELLS

    obs = []
    for name in well_names:
        s = load(name)
        if s is None or s.recharge_ratio_pct is None:
            continue
        if name not in WELLS:
            continue
        w = WELLS[name]
        # P_total, dh: stored 에 없으면 0 (rech posterior 무력)
        # cumulative_dh = wtf_mm / sy / 1000 (역산) — 단순 근사
        sy_eff = s.Sy_eff or 0.1
        if s.P_annual_mm and s.wtf_mm and sy_eff > 0:
            P_total = s.P_annual_mm / 1000.0
            cum_dh = (s.wtf_mm / 1000.0) / sy_eff
        else:
            P_total = cum_dh = None

        soil_frac = 1.0
        if hsg_fractions and s.hydro_type:
            soil_frac = hsg_fractions.get(s.hydro_type, 1.0)

        obs.append(WellObservation(
            name=name,
            hsg=s.hydro_type or "B",
            aquifer=w.aquifer,
            sy_eff_obs=sy_eff,
            pump_test_sy=s.pump_test_sy,
            cumulative_dh_m=cum_dh,
            P_total_m=P_total,
            soil_area_frac=soil_frac,
        ))

    if not obs:
        raise RuntimeError(f"{watershed}: 사용 가능한 stored 결과 없음")
    return fit_hierarchical(obs, **kwargs)
