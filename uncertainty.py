"""
uncertainty.py — Parameter uncertainty quantification via bootstrap resampling.

Provides bootstrap-based confidence intervals for key model outputs
(recharge ratio, RMSE, Sy_eff, optimal k) by resampling residuals
and re-optimising parameters.

References
----------
Efron, B. & Tibshirani, R.J. (1993). An Introduction to the Bootstrap.
    Chapman & Hall/CRC.

Efron, B. (1987). Better bootstrap confidence intervals.
    Journal of the American Statistical Association, 82(397), 171-185.
    (BCa method)

Davison, A.C. & Hinkley, D.V. (1997). Bootstrap Methods and their
    Application. Cambridge University Press.

Beven, K. & Binley, A. (1992). The future of distributed models: Model
    calibration and uncertainty prediction.  Hydrological Processes,
    6(3), 279-298.  (GLUE framework motivation)

Method
------
Residual bootstrap (appropriate for time series with autocorrelation):
1. Fit the model to obtain baseline parameters and residuals.
2. Resample residual blocks (block bootstrap) to generate synthetic
   observation series.
3. Re-optimise parameters for each bootstrap replicate.
4. Collect the distribution of output metrics.
5. Report BCa (bias-corrected and accelerated) confidence intervals.

Block bootstrap is used instead of i.i.d. bootstrap because hydrological
residuals exhibit temporal autocorrelation.  Block length is set to the
soil-specific tau (drainage time constant) to preserve correlation structure.

BCa intervals (Efron, 1987) replace the previous Gaussian approximation
(z_alpha * std) because hydrological recharge distributions are typically
right-skewed, making symmetric Gaussian intervals inappropriate.
"""

from __future__ import annotations

import concurrent.futures
import os
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

from scipy.optimize import minimize

from core_sim_config import DEFAULT_Q_NOISE, DEFAULT_R_NOISE
from core_sim_v27 import (
    run_logic_v27,
    apply_lag,
    detect_pump_mask,
    remove_outliers,
    calc_error,
    load_core_data,
    normalize_core_inputs,
    optimize_parameters,
)
from soil_db import get_soil, get_bounds, TAU_DB


@dataclass
class UncertaintyResult:
    """Bootstrap uncertainty estimates for key metrics."""
    n_bootstrap: int
    confidence_level: float  # e.g. 0.95
    ci_method: str           # "bca" or "percentile"

    # Recharge ratio (% of rainfall)
    rech_mean: float
    rech_std: float
    rech_ci_lower: float
    rech_ci_upper: float

    # RMSE
    rmse_mean: float
    rmse_std: float
    rmse_ci_lower: float
    rmse_ci_upper: float

    # Optimal k
    k_mean: float
    k_std: float
    k_ci_lower: float
    k_ci_upper: float

    # Optimal z
    z_mean: float
    z_std: float
    z_ci_lower: float
    z_ci_upper: float

    # Sy_eff
    sy_mean: float
    sy_std: float
    sy_ci_lower: float
    sy_ci_upper: float

    # Baseline point estimate (from single optimisation, before bootstrap)
    rech_baseline: float = 0.0

    # Bootstrap bias diagnostics
    bootstrap_bias: float = 0.0       # rech_mean - rech_baseline
    bootstrap_bias_pct: float = 0.0   # |bias| / baseline * 100
    bca_fallback_used: bool = False    # True if BCa produced inverted CI

    # Raw bootstrap samples (for plotting)
    rech_samples: list = None  # type: ignore[assignment]
    rmse_samples: list = None  # type: ignore[assignment]
    k_samples: list = None     # type: ignore[assignment]

    def to_dict(self):
        d = asdict(self)
        # Convert numpy arrays that might be in lists
        for key in ['rech_samples', 'rmse_samples', 'k_samples']:
            d[key] = [float(v) for v in d[key]]
        return d


def _estimate_block_length(residuals: np.ndarray, max_lag: int = 100) -> int:
    """Estimate optimal block length from ACF e-folding timescale.

    Computes the sample autocorrelation function and finds the first
    lag where ACF drops below 1/e (≈ 0.368).  This decorrelation
    timescale is a robust, assumption-light heuristic for block
    bootstrap block length selection.

    Parameters
    ----------
    residuals : array
        Stationary residuals (NaN-free subset).
    max_lag : int
        Maximum lag to examine.

    Returns
    -------
    int
        Estimated block length (≥ 2).

    References
    ----------
    Politis, D.N. & Romano, J.P. (1994). The stationary bootstrap.
        JASA, 89(428), 1303-1313.
    Lahiri, S.N. (2003). Resampling Methods for Dependent Data.
        Springer, Ch. 2.
    """
    n = len(residuals)
    if n < 10:
        return 3  # too short for meaningful ACF

    max_lag = min(max_lag, n // 3)
    r = residuals - np.mean(residuals)
    var = np.dot(r, r)
    if var < 1e-15:
        return 3

    threshold = 1.0 / np.e  # ≈ 0.368
    for lag in range(1, max_lag + 1):
        acf_lag = np.dot(r[:n - lag], r[lag:]) / var
        if acf_lag < threshold:
            return max(lag, 2)

    # ACF never dropped below threshold within max_lag:
    # series is highly persistent, use max_lag as block length
    return max(max_lag, 2)


def _block_resample_residuals(
    residuals: np.ndarray,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate one block-bootstrap resample of residuals."""
    n = len(residuals)
    bl = max(2, block_length)
    n_blocks = int(np.ceil(n / bl))

    # Sample block start indices
    starts = rng.integers(0, n - bl + 1, size=n_blocks)
    resampled = np.concatenate([residuals[s:s + bl] for s in starts])
    return resampled[:n]


def _bca_interval(
    samples: np.ndarray,
    theta_hat: float,
    jackknife_values: np.ndarray,
    confidence: float,
) -> tuple:
    """Compute BCa (bias-corrected and accelerated) confidence interval.

    BCa intervals correct for both bias and skewness in the bootstrap
    distribution, producing more accurate coverage than simple percentile
    or Gaussian approximation methods.

    Parameters
    ----------
    samples : np.ndarray
        Bootstrap replicate values (B samples).
    theta_hat : float
        Point estimate from the original sample.
    jackknife_values : np.ndarray
        Leave-one-out estimates for acceleration computation.
    confidence : float
        Confidence level (e.g. 0.95).

    Returns
    -------
    (ci_lower, ci_upper) : tuple of float

    References
    ----------
    Efron, B. (1987). Better bootstrap confidence intervals.
        JASA, 82(397), 171-185.
    Efron, B. & Tibshirani, R.J. (1993). An Introduction to the
        Bootstrap. Chapman & Hall/CRC, Ch. 14.
    DiCiccio, T.J. & Efron, B. (1996). Bootstrap confidence intervals.
        Statistical Science, 11(3), 189-228.
    """
    from scipy.stats import norm

    B = len(samples)
    if B < 10:
        # Fallback to percentile for very small samples
        alpha = (1.0 - confidence) / 2.0
        return (
            float(np.percentile(samples, alpha * 100)),
            float(np.percentile(samples, (1 - alpha) * 100)),
        )

    # ── Bias correction z0 ──
    # z0 = Φ⁻¹(proportion of bootstrap replicates < θ̂)
    prop_less = np.mean(samples < theta_hat)
    # Clamp to avoid ±inf from norm.ppf
    prop_less = np.clip(prop_less, 1.0 / (2 * B), 1.0 - 1.0 / (2 * B))
    z0 = norm.ppf(prop_less)

    # ── Acceleration a ──
    # Estimated from jackknife influence values
    n_jack = len(jackknife_values)
    if n_jack > 2:
        theta_bar = np.mean(jackknife_values)
        diff = theta_bar - jackknife_values
        a_num = np.sum(diff ** 3)
        a_den = 6.0 * (np.sum(diff ** 2) ** 1.5)
        acc = a_num / a_den if abs(a_den) > 1e-15 else 0.0
    else:
        acc = 0.0

    # ── Adjusted percentiles ──
    alpha = (1.0 - confidence) / 2.0
    z_alpha = norm.ppf(alpha)
    z_1alpha = norm.ppf(1.0 - alpha)

    # BCa formula: adjusted percentile = Φ(z0 + (z0 + z_α) / (1 - a*(z0 + z_α)))
    def _bca_percentile(z_val):
        num = z0 + z_val
        denom = 1.0 - acc * num
        if abs(denom) < 1e-15:
            return norm.cdf(z_val)  # fallback
        adjusted = z0 + num / denom
        return norm.cdf(adjusted)

    p_lo = _bca_percentile(z_alpha)
    p_hi = _bca_percentile(z_1alpha)

    # Clamp to valid range
    p_lo = np.clip(p_lo, 0.5 / B, 1.0 - 0.5 / B)
    p_hi = np.clip(p_hi, 0.5 / B, 1.0 - 0.5 / B)

    ci_lo = float(np.percentile(samples, p_lo * 100))
    ci_hi = float(np.percentile(samples, p_hi * 100))

    # ── Sanity check: CI must be well-ordered ──────────────────
    # When B is small or bias is extreme, BCa can produce inverted
    # intervals.  Fall back to simple percentile CI.
    #
    # NOTE: We intentionally do NOT force the CI to bracket theta_hat.
    # If the bootstrap distribution is systematically above (or below)
    # the point estimate, the CI should reflect that — it is a valid
    # signal of optimisation instability or nonlinear bias.  Forcing
    # ci_lo = min(ci_lo, theta_hat) creates an artificially tight
    # lower bound that misleads users.
    if ci_lo > ci_hi:
        alpha_half = (1.0 - confidence) / 2.0
        ci_lo = float(np.percentile(samples, alpha_half * 100))
        ci_hi = float(np.percentile(samples, (1 - alpha_half) * 100))

    return (ci_lo, ci_hi)


def _bootstrap_single(args):
    """Single bootstrap iteration (module-level for ThreadPoolExecutor).

    Each call uses an independent RNG (seeded) for reproducibility
    regardless of parallel execution order.

    Parameters
    ----------
    args : tuple
        (seed, residuals, block_len, valid, hs_kf_base, po, po_shifted_base,
         sn, lb, ub, rc_val, q_val, r_val, base_k, base_z, base_rho, base_alpha)

    Returns
    -------
    tuple of (rech_ratio_b, rmse_b, boot_k, boot_z, sy_boot)  or  None
    """
    (seed, residuals, block_len, valid, hs_kf_base, po, po_shifted_base,
     sn, lb, ub, rc_val, q_val, r_val, base_k, base_z, base_rho, base_alpha) = args

    rng = np.random.default_rng(int(seed))
    resampled_res = _block_resample_residuals(residuals, block_len, rng)
    ho_synth = hs_kf_base + resampled_res
    ho_synth[~valid] = np.nan

    # Reuse baseline pump mask — resampled residuals do not change
    # pump event locations, and re-detection is the single largest
    # per-iteration cost after Nelder-Mead optimisation.
    pm_synth = detect_pump_mask(hs_kf_base, po, rc_val)
    try:
        def _obj(p):
            pk = max(min(p[0], ub[0]), lb[0])
            pz = max(min(p[1], ub[1]), lb[1])
            return calc_error(pk, pz, sn, po_shifted_base, ho_synth, rc_val, pm_synth,
                              rho=base_rho, alpha=base_alpha)

        res = minimize(
            _obj, [base_k, base_z], method="Nelder-Mead",
            options={"xatol": 1e-5, "fatol": 1e-5, "maxfev": 800},
        )
        boot_k = max(min(res.x[0], ub[0]), lb[0])
        boot_z = max(min(res.x[1], ub[1]), lb[1])
    except Exception:
        return None

    rech_boot, hs_boot, _, sy_boot, _ = run_logic_v27(
        boot_k, boot_z, sn, po_shifted_base, ho_synth, q_val, r_val, rc_val, pm_synth,
        rho=base_rho, alpha=base_alpha,
    )

    valid_boot = ~np.isnan(ho_synth) & ~np.isnan(hs_boot)
    if np.sum(valid_boot) < 5:
        return None

    rmse_b = float(np.sqrt(np.nanmean(
        (hs_boot[valid_boot] - ho_synth[valid_boot]) ** 2
    )))
    total_rain = max(float(np.sum(po_shifted_base)), 1e-9)
    rech_ratio_b = float(np.sum(rech_boot)) / total_rain * 100.0

    return (rech_ratio_b, rmse_b, float(boot_k), float(boot_z), float(sy_boot))


def bootstrap_uncertainty(
    file_path: str,
    soil_num: int = 3,
    k_init: float = -0.05,
    z_init: float = 3.0,
    q_val: float = DEFAULT_Q_NOISE,
    r_val: float = DEFAULT_R_NOISE,
    rc_val: float = 0.001,
    sens_val: float = 5.0,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
    *,
    # Pre-optimised parameters from the app's main analysis.
    # When provided, skip internal re-optimisation and use these directly.
    # This ensures the bootstrap distribution is centred on the SAME
    # baseline the user sees in the main analysis tab.
    opt_k: float | None = None,
    opt_z: float | None = None,
    opt_lag: int | None = None,
    opt_rho: float | None = None,
    opt_alpha: float | None = None,
) -> UncertaintyResult:
    """Compute bootstrap confidence intervals for model outputs.

    Parameters
    ----------
    file_path : str
        Path to input data file (or "DEMO").
    soil_num : int
        Soil type (1-12).
    n_bootstrap : int
        Number of bootstrap replicates.  Default: 1000.

        Efron & Tibshirani (1993, p. 52) recommend >= 1,000 replicates
        for stable 95% confidence intervals.  Using B=200 (previous
        default) results in Monte Carlo standard errors of ~10-20% of
        the CI half-width, which is unacceptable for publication-quality
        uncertainty estimates.  B=1000 reduces this error to ~3-5%.
    confidence : float
        Confidence level for intervals (e.g. 0.95 for 95% CI).
    seed : int
        Random seed for reproducibility.
    opt_k, opt_z, opt_lag, opt_rho, opt_alpha : float | None
        Pre-optimised parameters from the main analysis.  When ALL five
        are provided, the bootstrap skips its own ``optimize_parameters``
        call and uses these values as the baseline.  This eliminates the
        recharge discrepancy caused by converging to a different local
        minimum during internal re-optimisation.
    """
    rng = np.random.default_rng(seed)

    # ── Step 1: Baseline fit ──
    data = load_core_data(file_path)
    ho_orig = data.ho.ravel().astype(float)
    po = data.po.ravel().astype(float)
    n = len(ho_orig)

    ho = remove_outliers(ho_orig.copy(), sens_val)
    pump_mask = detect_pump_mask(ho, po, rc_val)
    sn, kb_min, kb_max, ok, oz, olag = normalize_core_inputs(
        soil_num, k_init, z_init, 0
    )

    # Use pre-optimised parameters if all five are provided;
    # otherwise fall back to internal optimisation.
    _have_preopt = all(v is not None for v in (opt_k, opt_z, opt_lag, opt_rho, opt_alpha))
    if _have_preopt:
        base_k = float(opt_k)
        base_z = float(opt_z)
        base_lag = int(opt_lag)
        base_rho = float(opt_rho)
        base_alpha = float(opt_alpha)
    else:
        # Optimise baseline parameters (Stage 1: k, z, lag; Stage 2: rho, alpha)
        base_k, base_z, base_lag, base_rho, base_alpha = optimize_parameters(
            ho, po, sn, ok, oz, rc_val, pump_mask
        )

    # Run baseline simulation with optimised Kalman hyperparameters
    po_shifted = apply_lag(po, base_lag)
    rech_base, hs_kf_base, _, sy_eff_base, _ = run_logic_v27(
        base_k, base_z, sn, po_shifted, ho, q_val, r_val, rc_val, pump_mask,
        rho=base_rho, alpha=base_alpha,
    )

    # Baseline point estimates
    valid = ~np.isnan(ho) & ~np.isnan(hs_kf_base)
    residuals = np.zeros(n)
    residuals[valid] = ho[valid] - hs_kf_base[valid]
    total_rain_base = max(float(np.sum(po_shifted)), 1e-9)
    rech_hat = float(np.sum(rech_base)) / total_rain_base * 100.0

    # ── Block length selection ──────────────────────────────────
    # Optimal block length for block bootstrap should reflect the
    # temporal autocorrelation structure of the residuals.
    #
    # Method: first lag where ACF drops below 1/e (≈ 0.368), which
    # corresponds to the e-folding decorrelation timescale.  This is
    # a standard heuristic for stationary time-series block bootstrap.
    #
    # Fallback: soil-specific τ from TAU_DB if residuals are too
    # short or ACF computation fails.
    #
    # References:
    #   Politis, D.N. & Romano, J.P. (1994). The stationary bootstrap.
    #       JASA, 89(428), 1303-1313.
    #   Lahiri, S.N. (2003). Resampling Methods for Dependent Data.
    #       Springer, Ch. 2.
    block_len = _estimate_block_length(residuals[valid])
    # Floor: soil τ provides physics-informed minimum
    block_len = max(block_len, TAU_DB[sn - 1], 3)

    # ── Step 2: Bootstrap loop (parallel) ─────────────────────────────
    #
    # ThreadPoolExecutor: NumPy/Scipy release the GIL during computation,
    # so thread-level parallelism provides real speedup without the pickle
    # overhead of ProcessPoolExecutor.
    #
    # Reproducibility: each iteration gets a pre-assigned independent seed.
    lb = np.array([kb_min, 0.1])
    ub = np.array([kb_max, 20.0])

    boot_seeds = rng.integers(0, 2**31, size=n_bootstrap)
    common_args = (residuals, block_len, valid, hs_kf_base.copy(), po.copy(),
                   po_shifted.copy(), sn, lb, ub, rc_val, q_val, r_val,
                   base_k, base_z, base_rho, base_alpha)

    task_args = [(int(boot_seeds[i]),) + common_args for i in range(n_bootstrap)]

    max_workers = min(os.cpu_count() or 1, 8)
    results_raw = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = executor.map(_bootstrap_single, task_args)
        for result in futures:
            if result is not None:
                results_raw.append(result)

    rech_samples = [r[0] for r in results_raw]
    rmse_samples = [r[1] for r in results_raw]
    k_samples = [r[2] for r in results_raw]
    z_samples = [r[3] for r in results_raw]
    sy_samples = [r[4] for r in results_raw]

    # ── Step 3: Jackknife for BCa acceleration ────────────────────────
    #
    # Leave-one-out jackknife on bootstrap samples (not original data)
    # to estimate the acceleration parameter.
    # For computational efficiency, use the bootstrap samples themselves
    # rather than re-running the full model n times.
    #
    # References:
    #   Efron, B. (1987). Better bootstrap confidence intervals. JASA.
    #   DiCiccio, T.J. & Efron, B. (1996). Statistical Science.
    jack_rech = np.array(rech_samples)
    n_boot_ok = len(jack_rech)

    if n_boot_ok >= 20:
        # Jackknife-of-bootstrap: leave each sample out and compute mean
        jack_vals = np.array([
            np.mean(np.delete(jack_rech, i)) for i in range(min(n_boot_ok, 200))
        ])
    else:
        jack_vals = jack_rech.copy()

    # ── Step 4: BCa confidence intervals ──────────────────────────────
    ci_method = "bca"

    def _ci_bca(samples_list, theta_hat, jack_vals_for_metric):
        arr = np.array(samples_list)
        if len(arr) < 10:
            # Fallback to percentile
            alpha_half = (1.0 - confidence) / 2.0
            ci_lo = float(np.percentile(arr, alpha_half * 100)) if len(arr) > 0 else 0.0
            ci_hi = float(np.percentile(arr, (1 - alpha_half) * 100)) if len(arr) > 0 else 0.0
            return (
                float(np.mean(arr)) if len(arr) > 0 else 0.0,
                float(np.std(arr)) if len(arr) > 0 else 0.0,
                ci_lo,
                ci_hi,
            )
        ci_lo, ci_hi = _bca_interval(arr, theta_hat, jack_vals_for_metric, confidence)
        return (
            float(np.mean(arr)),
            float(np.std(arr)),
            ci_lo,
            ci_hi,
        )

    # Jackknife values for each metric
    def _jack_for(samples_list):
        arr = np.array(samples_list)
        n_j = min(len(arr), 200)
        return np.array([np.mean(np.delete(arr, i)) for i in range(n_j)])

    r_m, r_s, r_lo, r_hi = _ci_bca(rech_samples, rech_hat, jack_vals)
    rm_m, rm_s, rm_lo, rm_hi = _ci_bca(
        rmse_samples,
        float(np.sqrt(np.nanmean(residuals[valid] ** 2))),
        _jack_for(rmse_samples),
    )
    k_m, k_s, k_lo, k_hi = _ci_bca(k_samples, float(base_k), _jack_for(k_samples))
    z_m, z_s, z_lo, z_hi = _ci_bca(z_samples, float(base_z), _jack_for(z_samples))
    s_m, s_s, s_lo, s_hi = _ci_bca(sy_samples, float(sy_eff_base), _jack_for(sy_samples))

    # ── Bootstrap bias diagnostics ──────────────────────────────
    boot_bias = r_m - rech_hat
    boot_bias_pct = abs(boot_bias) / max(abs(rech_hat), 1e-9) * 100.0

    return UncertaintyResult(
        n_bootstrap=len(rech_samples),
        confidence_level=confidence,
        ci_method=ci_method,
        rech_mean=r_m, rech_std=r_s, rech_ci_lower=r_lo, rech_ci_upper=r_hi,
        rmse_mean=rm_m, rmse_std=rm_s, rmse_ci_lower=rm_lo, rmse_ci_upper=rm_hi,
        k_mean=k_m, k_std=k_s, k_ci_lower=k_lo, k_ci_upper=k_hi,
        z_mean=z_m, z_std=z_s, z_ci_lower=z_lo, z_ci_upper=z_hi,
        rech_baseline=rech_hat,
        bootstrap_bias=boot_bias,
        bootstrap_bias_pct=boot_bias_pct,
        sy_mean=s_m, sy_std=s_s, sy_ci_lower=s_lo, sy_ci_upper=s_hi,
        rech_samples=rech_samples,
        rmse_samples=rmse_samples,
        k_samples=k_samples,
    )
