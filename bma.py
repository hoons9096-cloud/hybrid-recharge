"""
bma.py — Bayesian Model Averaging for soil type selection.

Instead of selecting a single "best" soil type, BMA assigns posterior
probabilities to all 12 USDA textural classes and computes probability-
weighted recharge estimates with credible intervals.

Theoretical basis
-----------------
Given K candidate soil models {M1, ..., MK}, each producing a simulated
water level time-series h_sim(Mk), the posterior probability of model Mk
given observed data D is (Hoeting et al., 1999):

    P(Mk | D)  ∝  P(D | Mk) * P(Mk)

where:
  - P(D | Mk) is the marginal likelihood (model evidence),
  - P(Mk)     is the prior probability of model Mk.

For computational tractability, we approximate the marginal likelihood
using the BIC approximation (Schwarz, 1978):

    log P(D | Mk) ≈ -0.5 * BIC_k

    BIC_k = n * ln(σ²_k) + p_k * ln(n)

where σ²_k is the residual variance, p_k the number of parameters,
and n the number of valid observations.

The BMA-weighted recharge estimate and its variance are:

    E[R | D] = Σ_k  P(Mk | D) * R_k

    Var[R | D] = Σ_k  P(Mk | D) * [ Var(R_k | Mk) + (R_k - E[R|D])² ]

This "between-model" variance component is the key advantage over
single-model selection: it captures structural uncertainty due to
unknown soil type.

References
----------
Hoeting, J.A. et al. (1999). Bayesian Model Averaging: A Tutorial.
    Statistical Science, 14(4), 382-417.
Schwarz, G. (1978). Estimating the dimension of a model.
    Annals of Statistics, 6(2), 461-464.
Neuman, S.P. (2003). Maximum likelihood Bayesian averaging of uncertain
    model predictions. Stochastic Environmental Research and Risk
    Assessment, 17(5), 291-305.
Schöniger, A. et al. (2014). Model selection on solid ground: Rigorous
    comparison of nine ways to evaluate Bayesian model evidence.
    Water Resources Research, 50(12), 9484-9513.
Ye, M. et al. (2008). Expert elicitation of recharge model probabilities
    for the Death Valley regional flow system. Journal of Hydrology,
    354(1-4), 102-115.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from wtf_logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────────────────
@dataclass
class BMAResult:
    """Bayesian Model Averaging result across soil types."""

    # Per-model posterior probabilities (length 12, sums to 1)
    posterior: np.ndarray           # P(Mk | D) for k = 1..12
    soil_names: List[str]           # soil name for each model

    # BMA-weighted recharge statistics
    recharge_mean: float            # E[R | D]  (% of total rainfall)
    recharge_std: float             # sqrt(Var[R | D])
    recharge_ci_lo: float           # lower bound of 90% credible interval
    recharge_ci_hi: float           # upper bound of 90% credible interval

    # Individual model recharge rates (for display)
    recharge_per_soil: np.ndarray   # R_k for each soil (%)

    # Diagnostics
    bic_values: np.ndarray          # BIC for each model
    log_likelihoods: np.ndarray     # log-likelihood approximation
    n_effective_models: float       # effective number of models (entropy-based)
    dominant_soil: int              # 1-based index of highest-posterior soil
    dominant_prob: float            # posterior of dominant soil
    confidence_label: str           # "높음", "보통", "낮음"

    # 총 분산 분해 (Law of Total Variance): Var = within + between
    # within_variance: 모수 불확실성 (각 모델 내 bootstrap 분산의 가중합)
    # between_variance: 구조적 불확실성 (모델 간 분산의 가중합)
    within_variance: float = 0.0    # E[Var(R|Mk,D)] — parameter uncertainty
    between_variance: float = 0.0   # Var[E(R|Mk,D)] — structural uncertainty
    within_model_uncertainty_source: str = "nrmse_approx"  # "bootstrap" or "nrmse_approx"


# ──────────────────────────────────────────────────────────
# Core BMA computation
# ──────────────────────────────────────────────────────────
def compute_bma(
    scan_df: pd.DataFrame,
    n_params: int = 2,
    prior: Optional[np.ndarray] = None,
) -> BMAResult:
    """Compute Bayesian Model Averaging from soil scan results.

    Parameters
    ----------
    scan_df : pd.DataFrame
        Output from the soil scan loop. Required columns:
        - Index : int (1-based soil number)
        - RMSE  : float (root mean square error in metres)
        - SigmaHo : float (observed WL standard deviation)
        - Recharge : float (recharge ratio, %)
        - Soil : str (soil name)
        - EvalN : int (number of valid evaluation points)
    n_params : int
        Number of calibrated parameters per model (default=2: k, z).
        Used in BIC computation.
    prior : np.ndarray or None
        Prior probabilities P(Mk), length = number of rows in scan_df.
        If None, uniform prior is used (non-informative).

    Returns
    -------
    BMAResult
        Posterior probabilities and BMA-weighted recharge estimates.
    """
    n_models = len(scan_df)
    if n_models == 0:
        raise ValueError("Empty scan DataFrame — cannot compute BMA.")

    # ── Extract data ──
    soil_indices = scan_df["Index"].values.astype(int)
    rmse_arr = scan_df["RMSE"].values.astype(float)
    sigma_ho = scan_df["SigmaHo"].values.astype(float) if "SigmaHo" in scan_df else \
               np.full(n_models, 0.1)
    rech_arr = scan_df["Recharge"].values.astype(float)
    soil_names = scan_df["Soil"].values.tolist() if "Soil" in scan_df else \
                 [f"Soil {i}" for i in soil_indices]
    eval_n = scan_df["EvalN"].values.astype(int) if "EvalN" in scan_df else \
             np.full(n_models, 365, dtype=int)

    # ── BIC approximation of log-marginal-likelihood ──
    # σ²_k = RMSE² (residual variance for model k)
    # BIC_k = n * ln(σ²_k) + p * ln(n)
    # log P(D|Mk) ≈ -0.5 * BIC_k
    residual_var = np.maximum(rmse_arr ** 2, 1e-12)
    n_obs = np.maximum(eval_n, 10).astype(float)

    bic = n_obs * np.log(residual_var) + n_params * np.log(n_obs)

    # ── Temperature scaling (Ye et al., 2008) ──
    # Raw BIC differences are amplified by large n, causing degenerate
    # posteriors (one model ≈ 100%).  Temperature scaling τ > 1 softens
    # the posterior, yielding more informative multi-model weights.
    #
    # τ = sqrt(n_mean) follows the heuristic from Schöniger et al. (2014)
    # that adjusts for the over-confidence of BIC with large datasets.
    n_mean = float(np.mean(n_obs))
    tau = max(np.sqrt(n_mean), 1.0)
    log_lik = -0.5 * bic / tau

    # ── Prior ──
    if prior is None:
        log_prior = np.zeros(n_models)  # uniform
    else:
        prior = np.maximum(prior, 1e-12)
        prior = prior / prior.sum()
        log_prior = np.log(prior)

    # ── Posterior (log-sum-exp for numerical stability) ──
    log_unnorm = log_lik + log_prior
    log_max = np.max(log_unnorm)
    log_denom = log_max + np.log(np.sum(np.exp(log_unnorm - log_max)))
    log_posterior = log_unnorm - log_denom
    posterior = np.exp(log_posterior)

    # Ensure proper normalisation (floating point safety)
    posterior = posterior / posterior.sum()

    # ── BMA recharge statistics ──
    # E[R | D] = Σ P(Mk|D) * R_k
    rech_mean = float(np.dot(posterior, rech_arr))

    # Var[R | D] = Σ P(Mk|D) * [(R_k - E[R])² + Var(R_k|Mk)]
    # Within-model variance approximated as (RMSE/σ_ho)² * R_k² (error propagation)
    nrmse_arr = rmse_arr / np.maximum(sigma_ho, 1e-6)
    within_var = (nrmse_arr * rech_arr) ** 2       # approximate within-model variance
    between_var = (rech_arr - rech_mean) ** 2       # between-model variance
    within_total = float(np.dot(posterior, within_var))
    between_total = float(np.dot(posterior, between_var))
    total_var = within_total + between_total
    rech_std = float(np.sqrt(max(total_var, 0.0)))

    # 90% credible interval (Gaussian approximation)
    z90 = 1.645
    ci_lo = max(0.0, rech_mean - z90 * rech_std)
    ci_hi = rech_mean + z90 * rech_std

    # ── Diagnostics ──
    # Effective number of models (Shannon entropy based)
    ent = -float(np.sum(posterior * np.log(np.maximum(posterior, 1e-12))))
    n_eff = float(np.exp(ent))  # 1 = single model dominates, 12 = uniform

    dominant_idx = int(np.argmax(posterior))
    dominant_soil = int(soil_indices[dominant_idx])
    dominant_prob = float(posterior[dominant_idx])

    # Confidence label
    if dominant_prob >= 0.6:
        conf = "높음"
    elif dominant_prob >= 0.35:
        conf = "보통"
    else:
        conf = "낮음"

    logger.info(
        "BMA: dominant=%s (%.1f%%), n_eff=%.1f, rech=%.1f±%.1f%%",
        soil_names[dominant_idx], dominant_prob * 100,
        n_eff, rech_mean, rech_std,
    )

    return BMAResult(
        posterior=posterior,
        soil_names=soil_names,
        recharge_mean=rech_mean,
        recharge_std=rech_std,
        recharge_ci_lo=ci_lo,
        recharge_ci_hi=ci_hi,
        recharge_per_soil=rech_arr,
        bic_values=bic,
        log_likelihoods=log_lik,
        n_effective_models=n_eff,
        dominant_soil=dominant_soil,
        dominant_prob=dominant_prob,
        confidence_label=conf,
        within_variance=within_total,
        between_variance=between_total,
        within_model_uncertainty_source="nrmse_approx",
    )


def compute_bma_integrated(
    scan_df: pd.DataFrame,
    within_var_per_model: np.ndarray,
    n_params: int = 2,
    prior: Optional[np.ndarray] = None,
) -> BMAResult:
    """BMA with bootstrap-derived within-model variance (Law of Total Variance).

    Replaces the NRMSE-based within-model variance approximation in
    :func:`compute_bma` with bootstrap confidence intervals from
    :func:`uncertainty.bootstrap_uncertainty`.

    This implements the full Law of Total Variance decomposition
    (Hoeting et al., 1999):

        Var[R | D] = E[Var(R | Mk, D)]   +   Var[E(R | Mk, D)]
                     ──────────────────────   ──────────────────
                     within-model variance    between-model variance
                     (parameter uncertainty)  (structural uncertainty)

    The within-model variance is obtained from bootstrap resampling, which
    properly accounts for parameter estimation error (Efron & Tibshirani,
    1993). The NRMSE-based approximation in ``compute_bma`` is a cruder
    proxy that tends to over-estimate within-model variance for well-
    constrained soils and under-estimate it for poorly constrained soils.

    Parameters
    ----------
    scan_df : pd.DataFrame
        Same format as :func:`compute_bma`.
    within_var_per_model : np.ndarray
        Bootstrap-derived within-model variance for each soil model.
        Obtain via :func:`uncertainty.bootstrap_uncertainty` for each
        soil type, using ``result.rech_std ** 2`` as within-model variance.
        Length must equal ``len(scan_df)``.
    n_params : int
        Number of calibrated parameters (default=2).
    prior : np.ndarray or None
        Prior probabilities. If None, uniform prior is used.

    Returns
    -------
    BMAResult
        Same structure as :func:`compute_bma`, but with:
        - ``within_variance`` sourced from bootstrap (more accurate)
        - ``within_model_uncertainty_source`` = "bootstrap"

    References
    ----------
    Hoeting, J.A. et al. (1999). Bayesian Model Averaging: A Tutorial.
        Statistical Science, 14(4), 382-417.
    Efron, B. & Tibshirani, R.J. (1993). An Introduction to the Bootstrap.
        Chapman & Hall/CRC. p. 52.
    Ye, M. et al. (2010). Maximum likelihood Bayesian averaging of spatial
        variability models in unsaturated fractured tuff.
        Water Resources Research, 46(5), W05539.
    """
    within_var_arr = np.asarray(within_var_per_model, dtype=float)
    if len(within_var_arr) != len(scan_df):
        raise ValueError(
            f"within_var_per_model length ({len(within_var_arr)}) must match "
            f"scan_df rows ({len(scan_df)})."
        )
    within_var_arr = np.maximum(within_var_arr, 0.0)  # 음수 방지

    # ── 동일한 BIC/posterior 계산 (compute_bma 로직 재사용) ──
    n_models = len(scan_df)
    soil_indices = scan_df["Index"].values.astype(int)
    rmse_arr = scan_df["RMSE"].values.astype(float)
    rech_arr = scan_df["Recharge"].values.astype(float)
    soil_names = scan_df["Soil"].values.tolist() if "Soil" in scan_df else \
                 [f"Soil {i}" for i in soil_indices]
    eval_n = scan_df["EvalN"].values.astype(int) if "EvalN" in scan_df else \
             np.full(n_models, 365, dtype=int)

    residual_var = np.maximum(rmse_arr ** 2, 1e-12)
    n_obs = np.maximum(eval_n, 10).astype(float)
    bic = n_obs * np.log(residual_var) + n_params * np.log(n_obs)

    n_mean = float(np.mean(n_obs))
    tau = max(np.sqrt(n_mean), 1.0)
    log_lik = -0.5 * bic / tau

    if prior is None:
        log_prior = np.zeros(n_models)
    else:
        prior = np.maximum(prior, 1e-12)
        prior = prior / prior.sum()
        log_prior = np.log(prior)

    log_unnorm = log_lik + log_prior
    log_max = np.max(log_unnorm)
    log_denom = log_max + np.log(np.sum(np.exp(log_unnorm - log_max)))
    posterior = np.exp(log_unnorm - log_denom)
    posterior = posterior / posterior.sum()

    # ── 총 분산 분해 (Law of Total Variance) ──
    # Bootstrap 분산을 within-model variance로 사용
    rech_mean = float(np.dot(posterior, rech_arr))
    between_var = (rech_arr - rech_mean) ** 2
    within_total = float(np.dot(posterior, within_var_arr))
    between_total = float(np.dot(posterior, between_var))
    total_var = within_total + between_total
    rech_std = float(np.sqrt(max(total_var, 0.0)))

    z90 = 1.645
    ci_lo = max(0.0, rech_mean - z90 * rech_std)
    ci_hi = rech_mean + z90 * rech_std

    ent = -float(np.sum(posterior * np.log(np.maximum(posterior, 1e-12))))
    n_eff = float(np.exp(ent))
    dominant_idx = int(np.argmax(posterior))
    dominant_soil = int(soil_indices[dominant_idx])
    dominant_prob = float(posterior[dominant_idx])

    if dominant_prob >= 0.6:
        conf = "높음"
    elif dominant_prob >= 0.35:
        conf = "보통"
    else:
        conf = "낮음"

    logger.info(
        "BMA-integrated: dominant=%s (%.1f%%), within_var=%.4f, "
        "between_var=%.4f, rech=%.1f±%.1f%%",
        soil_names[dominant_idx], dominant_prob * 100,
        within_total, between_total, rech_mean, rech_std,
    )

    return BMAResult(
        posterior=posterior,
        soil_names=soil_names,
        recharge_mean=rech_mean,
        recharge_std=rech_std,
        recharge_ci_lo=ci_lo,
        recharge_ci_hi=ci_hi,
        recharge_per_soil=rech_arr,
        bic_values=bic,
        log_likelihoods=log_lik,
        n_effective_models=n_eff,
        dominant_soil=dominant_soil,
        dominant_prob=dominant_prob,
        confidence_label=conf,
        within_variance=within_total,
        between_variance=between_total,
        within_model_uncertainty_source="bootstrap",
    )


def bma_summary_table(result: BMAResult) -> pd.DataFrame:
    """Create a display-friendly summary table from BMA results.

    Returns a DataFrame sorted by posterior probability (descending)
    with columns: Soil, Posterior(%), BIC.
    """
    df = pd.DataFrame({
        "토양": result.soil_names,
        "사후확률(%)": np.round(result.posterior * 100, 1),
        "BIC": np.round(result.bic_values, 1),
    })
    df = df.sort_values("사후확률(%)", ascending=False).reset_index(drop=True)
    return df
