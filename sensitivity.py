"""
sensitivity.py — Kalman hyperparameter & TOPSIS weight sensitivity analysis.

Provides systematic evaluation of how model outputs change under parameter
perturbation, addressing a key gap for SCI-level publication readiness.

Module 1: Kalman Hyperparameter Sensitivity
--------------------------------------------
Sweeps over ρ (persistence), Q/R ratio (process/measurement noise),
and blend α (observation/filter weighting) to quantify recharge sensitivity.

Module 2: TOPSIS Weight Sensitivity
------------------------------------
Perturbs TOPSIS criteria weights ±Δ and re-ranks soils to assess
robustness of the soil type recommendation.

References
----------
Morris, M.D. (1991). Factorial sampling plans for preliminary
    computational experiments. Technometrics, 33(2), 161-174.
Saltelli, A. et al. (2004). Sensitivity Analysis in Practice.
    Wiley.
Triantaphyllou, E. (2000). Multi-Criteria Decision Making Methods:
    A Comparative Study. Springer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core_sim_config import OBJ_W_FIT, OBJ_W_RESP, OBJ_W_RECH
from wtf_logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════
# 1. Kalman Hyperparameter Sensitivity
# ══════════════════════════════════════════════════════════════

@dataclass
class KalmanSensitivityResult:
    """Results of Kalman hyperparameter sensitivity sweep."""

    # Parameter grids used
    rho_values: List[float]
    qr_ratio_values: List[float]
    alpha_values: List[float]

    # Recharge ratio (%) for each parameter combination
    # Each is a dict: param_name -> list of (param_value, recharge_%)
    rho_sweep: List[Tuple[float, float]]       # (rho, recharge%)
    qr_ratio_sweep: List[Tuple[float, float]]  # (qr_ratio, recharge%)
    alpha_sweep: List[Tuple[float, float]]      # (alpha, recharge%)

    # Baseline values
    baseline_rho: float
    baseline_qr: float
    baseline_alpha: float
    baseline_recharge: float

    # Sensitivity indices (normalised partial derivative)
    # S_i = (ΔR/R) / (Δp/p) — elasticity of recharge to parameter
    sensitivity_rho: float
    sensitivity_qr: float
    sensitivity_alpha: float

    # Tornado data: (param_name, rech_at_low, rech_at_high, low_val, high_val)
    tornado_data: List[Tuple[str, float, float, float, float]]


def kalman_sensitivity_sweep(
    run_func,
    k: float,
    z_unsat: float,
    sn: int,
    po_in: np.ndarray,
    ho_in: np.ndarray,
    q_base: float,
    r_base: float,
    r_c: float,
    pump_mask: np.ndarray,
    rho_range: Tuple[float, float] = (0.5, 0.95),
    qr_range: Tuple[float, float] = (0.05, 0.8),
    alpha_range: Tuple[float, float] = (0.1, 0.9),
    n_steps: int = 9,
) -> KalmanSensitivityResult:
    """Sweep Kalman hyperparameters and measure recharge sensitivity.

    Parameters
    ----------
    run_func : callable
        The run_logic_v27 function (passed to avoid circular imports).
    k, z_unsat, sn : float, float, int
        Optimised recession constant, unsaturated zone depth, soil number.
    po_in, ho_in : np.ndarray
        Rainfall and observed water level arrays.
    q_base, r_base : float
        Baseline Q and R noise values.
    r_c : float
        Rainfall threshold.
    pump_mask : np.ndarray
        Boolean pump contamination mask.
    rho_range, qr_range, alpha_range : tuple
        (min, max) for each parameter sweep.
    n_steps : int
        Number of grid points per parameter.

    Returns
    -------
    KalmanSensitivityResult
    """
    import core_sim_config as cfg

    total_rain = max(float(np.sum(po_in)), 1e-9)

    # 모든 칼만 하이퍼파라미터를 키워드 인자로 직접 전달 — 전역 상태 변경 없음.
    # 이전 구현은 cfg.KALMAN_W_Q_RATIO를 직접 수정하여 race condition 위험이 있었음.
    # v31.1: run_logic_v27에 w_q_ratio 키워드 인자 추가로 해결.
    orig_rho = cfg.KALMAN_RHO
    orig_qr = cfg.KALMAN_W_Q_RATIO
    orig_alpha = cfg.KALMAN_WTF_BLEND_ALPHA

    def _run_and_get_recharge(rho_val, qr_val, alpha_val):
        """All Kalman hyperparameters passed as keyword args (thread-safe)."""
        try:
            rech, _, _, _, _ = run_func(
                k, z_unsat, sn, po_in, ho_in, q_base, r_base, r_c, pump_mask,
                rho=rho_val, alpha=alpha_val, w_q_ratio=qr_val,
            )
            return float(np.sum(rech)) / total_rain * 100.0
        except Exception as e:
            logger.warning("Sensitivity run failed: %s", e)
            return np.nan

    # Baseline recharge
    baseline_rech = _run_and_get_recharge(orig_rho, orig_qr, orig_alpha)

    # ── Sweep ρ ──
    rho_vals = np.linspace(rho_range[0], rho_range[1], n_steps).tolist()
    rho_sweep = []
    for rv in rho_vals:
        r = _run_and_get_recharge(rv, orig_qr, orig_alpha)
        rho_sweep.append((rv, r))

    # ── Sweep Q/R ratio ──
    qr_vals = np.linspace(qr_range[0], qr_range[1], n_steps).tolist()
    qr_sweep = []
    for qv in qr_vals:
        r = _run_and_get_recharge(orig_rho, qv, orig_alpha)
        qr_sweep.append((qv, r))

    # ── Sweep blend α ──
    alpha_vals = np.linspace(alpha_range[0], alpha_range[1], n_steps).tolist()
    alpha_sweep = []
    for av in alpha_vals:
        r = _run_and_get_recharge(orig_rho, orig_qr, av)
        alpha_sweep.append((av, r))

    # ── Sensitivity indices (elasticity) ──
    def _elasticity(sweep, baseline_param, baseline_rech_val):
        """Compute elasticity S = (ΔR/R) / (Δp/p) using finite difference."""
        vals = [(p, r) for p, r in sweep if np.isfinite(r)]
        if len(vals) < 2 or baseline_rech_val == 0:
            return 0.0
        # Use endpoints for range-based elasticity
        p_lo, r_lo = vals[0]
        p_hi, r_hi = vals[-1]
        dp = (p_hi - p_lo) / max(abs(baseline_param), 1e-9)
        dr = (r_hi - r_lo) / max(abs(baseline_rech_val), 1e-9)
        return abs(dr / dp) if dp != 0 else 0.0

    s_rho = _elasticity(rho_sweep, orig_rho, baseline_rech)
    s_qr = _elasticity(qr_sweep, orig_qr, baseline_rech)
    s_alpha = _elasticity(alpha_sweep, orig_alpha, baseline_rech)

    # ── Tornado data ──
    def _tornado_entry(name, sweep, lo_param, hi_param):
        vals = {p: r for p, r in sweep if np.isfinite(r)}
        # Find closest to lo and hi
        lo_r = vals.get(lo_param, baseline_rech)
        hi_r = vals.get(hi_param, baseline_rech)
        return (name, lo_r, hi_r, lo_param, hi_param)

    tornado = [
        _tornado_entry("ρ (persistence)", rho_sweep, rho_vals[0], rho_vals[-1]),
        _tornado_entry("Q/R ratio", qr_sweep, qr_vals[0], qr_vals[-1]),
        _tornado_entry("Blend α", alpha_sweep, alpha_vals[0], alpha_vals[-1]),
    ]
    # Sort by impact (largest range first)
    tornado.sort(key=lambda t: abs(t[2] - t[1]), reverse=True)

    logger.info(
        "Kalman sensitivity: S_rho=%.3f, S_qr=%.3f, S_alpha=%.3f",
        s_rho, s_qr, s_alpha,
    )

    return KalmanSensitivityResult(
        rho_values=rho_vals,
        qr_ratio_values=qr_vals,
        alpha_values=alpha_vals,
        rho_sweep=rho_sweep,
        qr_ratio_sweep=qr_sweep,
        alpha_sweep=alpha_sweep,
        baseline_rho=orig_rho,
        baseline_qr=orig_qr,
        baseline_alpha=orig_alpha,
        baseline_recharge=baseline_rech,
        sensitivity_rho=s_rho,
        sensitivity_qr=s_qr,
        sensitivity_alpha=s_alpha,
        tornado_data=tornado,
    )


# ══════════════════════════════════════════════════════════════
# 2. TOPSIS Weight Sensitivity
# ══════════════════════════════════════════════════════════════

@dataclass
class TopsisWeightSensitivityResult:
    """Results of TOPSIS weight perturbation analysis."""

    # Original weights and best soil
    original_weights: List[float]
    criteria_names: List[str]
    original_best_soil: str
    original_best_index: int

    # Per-criterion perturbation results
    # Each entry: (criterion_name, weight_lo, weight_hi, best_soil_lo, best_soil_hi)
    perturbation_results: List[Tuple[str, float, float, str, str]]

    # Stability metric: fraction of perturbations that keep the same best soil
    stability_ratio: float

    # Tornado data for TOPSIS score of the best soil
    tornado_topsis: List[Tuple[str, float, float]]


# ══════════════════════════════════════════════════════════════
# 3. Objective Function Weight Sensitivity
# ══════════════════════════════════════════════════════════════

@dataclass
class ObjectiveWeightSensitivityResult:
    """Results of objective function weight perturbation analysis.

    Measures how sensitive the optimised parameters (k, z) and derived
    recharge ratio are to changes in the three objective function weights:
    w_fit (NRMSE), w_resp (rain-response mismatch), w_rech (recharge range).

    References
    ----------
    Saltelli, A. et al. (2004). Sensitivity Analysis in Practice. Wiley.
    Morris, M.D. (1991). Factorial sampling plans for preliminary
        computational experiments. Technometrics, 33(2), 161-174.
    """
    # Baseline values
    baseline_w_fit: float
    baseline_w_resp: float
    baseline_w_rech: float
    baseline_recharge: float
    baseline_k: float
    baseline_z: float

    # Sweep results: (weight_value, recharge%) tuples for each component
    w_fit_sweep: List[Tuple[float, float]]    # (w_fit, recharge%)
    w_resp_sweep: List[Tuple[float, float]]   # (w_resp, recharge%)
    w_rech_sweep: List[Tuple[float, float]]   # (w_rech, recharge%)

    # Elasticity: S_i = |ΔR/R| / |Δw/w|
    sensitivity_w_fit: float
    sensitivity_w_resp: float
    sensitivity_w_rech: float

    # Tornado data: (name, rech_at_low_w, rech_at_high_w, low_w, high_w)
    tornado_data: List[Tuple[str, float, float, float, float]]

    # Diagnostic: objective component values at baseline optimum
    # Helps interpret zero-sensitivity results (if penalty ≈ 0,
    # changing its weight has no effect on the argmin).
    baseline_nrmse: float = 0.0
    baseline_resp_mismatch: float = 0.0
    baseline_rech_violation: float = 0.0


def objective_weight_sensitivity(
    k: float,
    z: float,
    sn: int,
    po_shifted: np.ndarray,
    ho: np.ndarray,
    rc: float,
    pump_mask: np.ndarray,
    calc_error_func,
    run_func,
    optimize_func,
    q_val: float,
    r_val: float,
    rho: Optional[float] = None,
    alpha: Optional[float] = None,
    delta: float = 0.5,
    n_steps: int = 7,
) -> ObjectiveWeightSensitivityResult:
    """Analyse sensitivity of recharge estimate to objective function weights.

    For each of the three weights (w_fit, w_resp, w_rech), the weight is
    varied over [max(0.01, base*(1-delta)), min(1.0, base*(1+delta))]
    while the other two weights are re-normalised to sum to 1. For each
    perturbed weight set, parameter optimisation is re-run and the
    resulting recharge ratio is recorded.

    Parameters
    ----------
    k, z : float
        Baseline optimised parameters (used as warm start).
    sn : int
        Soil number (1-12).
    po_shifted, ho : np.ndarray
        Lag-shifted rainfall and observed WL arrays.
    rc : float
        Rainfall threshold.
    pump_mask : np.ndarray
        Boolean pump contamination mask.
    calc_error_func : callable
        The ``calc_error`` function from ``core_sim_v27``.
    run_func : callable
        The ``run_logic_v27`` function from ``core_sim_v27``.
    optimize_func : callable
        The ``optimize_parameters`` function from ``core_sim_v27``.
    q_val, r_val : float
        Kalman noise parameters.
    delta : float
        Relative perturbation range (default ±50% of baseline weight).
    n_steps : int
        Number of weight grid points per component.

    Returns
    -------
    ObjectiveWeightSensitivityResult
    """
    from scipy.optimize import minimize
    import core_sim_config as cfg
    from soil_db import get_bounds

    total_rain = max(float(np.sum(po_shifted)), 1e-9)
    kb_min, kb_max = get_bounds(sn)
    lb = np.array([kb_min, cfg.MIN_Z_PARAM])
    ub = np.array([kb_max, cfg.MAX_Z_PARAM])

    def _recharge_for_weights(wf, wr, wc):
        """Re-optimise with perturbed weights using warm-start Nelder-Mead.

        Single warm-start from the baseline optimum is intentional: it
        answers the question "starting from the current best (k, z), does
        changing the objective weights shift the optimum?"  If penalties
        (resp_mismatch, rech_violation) ≈ 0 at the optimum, the answer is
        legitimately "no" — which indicates the model satisfies all
        constraints simultaneously and the weight choice is non-critical.
        """
        def _obj(p):
            pk = np.clip(p[0], lb[0], ub[0])
            pz = np.clip(p[1], lb[1], ub[1])
            return calc_error_func(
                pk, pz, sn, po_shifted, ho, rc, pump_mask,
                rho=rho, alpha=alpha,
                w_fit=wf, w_resp=wr, w_rech=wc,
            )

        res = minimize(
            _obj, x0=[k, z], method="Nelder-Mead",
            options={"xatol": 1e-4, "fatol": 1e-4, "maxfev": 500},
        )

        pk_opt = np.clip(res.x[0], lb[0], ub[0])
        pz_opt = np.clip(res.x[1], lb[1], ub[1])

        rech, _, _, _, _ = run_func(
            pk_opt, pz_opt, sn, po_shifted, ho, q_val, r_val, rc, pump_mask,
            rho=rho, alpha=alpha,
        )
        return float(np.sum(rech)) / total_rain * 100.0, pk_opt, pz_opt

    # ── Baseline ──
    base_wf = cfg.OBJ_W_FIT
    base_wr = cfg.OBJ_W_RESP
    base_wc = cfg.OBJ_W_RECH
    base_rech, base_k_opt, base_z_opt = _recharge_for_weights(base_wf, base_wr, base_wc)

    # ── Baseline diagnostics: decompose objective at optimum ──
    # Run once with unit weights to get raw component values
    _diag_nrmse = 0.0
    _diag_resp = 0.0
    _diag_rech_v = 0.0
    try:
        _e_total = calc_error_func(
            base_k_opt, base_z_opt, sn, po_shifted, ho, rc, pump_mask,
            rho=rho, alpha=alpha,
            w_fit=1.0, w_resp=0.0, w_rech=0.0,
        )
        _diag_nrmse = _e_total  # w_fit=1 → return = nrmse
        _e_resp = calc_error_func(
            base_k_opt, base_z_opt, sn, po_shifted, ho, rc, pump_mask,
            rho=rho, alpha=alpha,
            w_fit=0.0, w_resp=1.0, w_rech=0.0,
        )
        _diag_resp = _e_resp  # w_resp=1 → return = resp_mismatch
        _e_rech = calc_error_func(
            base_k_opt, base_z_opt, sn, po_shifted, ho, rc, pump_mask,
            rho=rho, alpha=alpha,
            w_fit=0.0, w_resp=0.0, w_rech=1.0,
        )
        _diag_rech_v = _e_rech  # w_rech=1 → return = rech_violation
    except Exception:
        pass  # diagnostics are best-effort

    # ── Sweep each weight ──
    def _sweep_one(idx, base_w, other1, other2):
        lo = max(0.01, base_w * (1 - delta))
        hi = min(0.99, base_w * (1 + delta))
        grid = np.linspace(lo, hi, n_steps)
        results = []
        for w_i in grid:
            remainder = 1.0 - w_i
            sum_oth = other1 + other2
            if sum_oth > 0:
                o1 = other1 / sum_oth * remainder
                o2 = other2 / sum_oth * remainder
            else:
                o1 = o2 = remainder / 2.0
            weights = [0.0, 0.0, 0.0]
            weights[idx] = w_i
            oidx = [j for j in range(3) if j != idx]
            weights[oidx[0]] = o1
            weights[oidx[1]] = o2
            rech_val, _, _ = _recharge_for_weights(*weights)
            results.append((float(w_i), rech_val))
        return results

    w_fit_sweep = _sweep_one(0, base_wf, base_wr, base_wc)
    w_resp_sweep = _sweep_one(1, base_wr, base_wf, base_wc)
    w_rech_sweep = _sweep_one(2, base_wc, base_wf, base_wr)

    # ── Elasticity: S = |ΔR/R| / |Δw/w| ──
    def _elasticity(sweep, base_w):
        if len(sweep) < 2 or abs(base_rech) < 1e-9 or abs(base_w) < 1e-9:
            return 0.0
        r_lo = sweep[0][1]
        r_hi = sweep[-1][1]
        w_lo = sweep[0][0]
        w_hi = sweep[-1][0]
        dr = abs(r_hi - r_lo) / max(abs(base_rech), 1e-9)
        dw = abs(w_hi - w_lo) / max(abs(base_w), 1e-9)
        return dr / max(dw, 1e-9)

    sens_wf = _elasticity(w_fit_sweep, base_wf)
    sens_wr = _elasticity(w_resp_sweep, base_wr)
    sens_wc = _elasticity(w_rech_sweep, base_wc)

    # ── Tornado data ──
    tornado = [
        ("w_fit", w_fit_sweep[0][1], w_fit_sweep[-1][1],
         w_fit_sweep[0][0], w_fit_sweep[-1][0]),
        ("w_resp", w_resp_sweep[0][1], w_resp_sweep[-1][1],
         w_resp_sweep[0][0], w_resp_sweep[-1][0]),
        ("w_rech", w_rech_sweep[0][1], w_rech_sweep[-1][1],
         w_rech_sweep[0][0], w_rech_sweep[-1][0]),
    ]
    tornado.sort(key=lambda t: abs(t[2] - t[1]), reverse=True)

    return ObjectiveWeightSensitivityResult(
        baseline_w_fit=base_wf,
        baseline_w_resp=base_wr,
        baseline_w_rech=base_wc,
        baseline_recharge=base_rech,
        baseline_k=base_k_opt,
        baseline_z=base_z_opt,
        w_fit_sweep=w_fit_sweep,
        w_resp_sweep=w_resp_sweep,
        w_rech_sweep=w_rech_sweep,
        sensitivity_w_fit=sens_wf,
        sensitivity_w_resp=sens_wr,
        sensitivity_w_rech=sens_wc,
        tornado_data=tornado,
        baseline_nrmse=_diag_nrmse,
        baseline_resp_mismatch=_diag_resp,
        baseline_rech_violation=_diag_rech_v,
    )


# ══════════════════════════════════════════════════════════════
# 4. TOPSIS Weight Sensitivity
# ══════════════════════════════════════════════════════════════

def topsis_weight_sensitivity(
    scan_df,
    delta: float = 0.20,
) -> TopsisWeightSensitivityResult:
    """Analyse TOPSIS ranking robustness under weight perturbation.

    For each of the 6 criteria, the weight is increased and decreased
    by +/-delta (other weights re-normalised to sum to 1). The soil
    ranking is recomputed for each perturbation.

    Parameters
    ----------
    scan_df : pd.DataFrame
        Soil scan results (same format as score_dataframe input).
    delta : float
        Weight perturbation fraction (default +/-20%).

    Returns
    -------
    TopsisWeightSensitivityResult
    """
    from scoring import (
        CRITERIA_WEIGHTS, CRITERIA_NAMES,
        compute_soil_scores, topsis_rank,
    )

    orig_weights = CRITERIA_WEIGHTS.copy()
    n_criteria = len(orig_weights)

    # Baseline ranking
    base_scores = [compute_soil_scores(row) for _, row in scan_df.iterrows()]
    base_scores = topsis_rank(base_scores)
    base_best_idx = max(base_scores, key=lambda s: s.topsis_score).soil_num
    base_best_name = max(base_scores, key=lambda s: s.topsis_score).soil_name

    perturbation_results = []
    tornado_topsis = []
    same_count = 0
    total_perturbations = 0

    for ci in range(n_criteria):
        crit_name = CRITERIA_NAMES[ci]
        lo_name = hi_name = base_best_name
        lo_weight = hi_weight = orig_weights[ci]
        lo_topsis = hi_topsis = max(s.topsis_score for s in base_scores
                                     if s.soil_num == base_best_idx)

        for direction in [-1, +1]:
            new_weights = orig_weights.copy()
            shift = direction * delta * orig_weights[ci]
            new_weights[ci] = max(0.01, new_weights[ci] + shift)

            # Re-normalise remaining weights proportionally
            remaining_sum = sum(new_weights[j] for j in range(n_criteria) if j != ci)
            if remaining_sum > 0:
                target_remaining = 1.0 - new_weights[ci]
                for j in range(n_criteria):
                    if j != ci:
                        new_weights[j] = new_weights[j] / remaining_sum * target_remaining

            import scoring as sc_mod
            saved_w = sc_mod.CRITERIA_WEIGHTS.copy()
            sc_mod.CRITERIA_WEIGHTS = new_weights

            try:
                perturbed_scores = [compute_soil_scores(row) for _, row in scan_df.iterrows()]
                perturbed_scores = topsis_rank(perturbed_scores)
                perturbed_best = max(perturbed_scores, key=lambda s: s.topsis_score)

                if perturbed_best.soil_num == base_best_idx:
                    same_count += 1
                total_perturbations += 1

                if direction == -1:
                    lo_name = perturbed_best.soil_name
                    lo_weight = new_weights[ci]
                    lo_topsis = max(s.topsis_score for s in perturbed_scores
                                    if s.soil_num == base_best_idx)
                else:
                    hi_name = perturbed_best.soil_name
                    hi_weight = new_weights[ci]
                    hi_topsis = max(s.topsis_score for s in perturbed_scores
                                    if s.soil_num == base_best_idx)
            finally:
                sc_mod.CRITERIA_WEIGHTS = saved_w

        perturbation_results.append((crit_name, lo_weight, hi_weight, lo_name, hi_name))
        tornado_topsis.append((crit_name, lo_topsis, hi_topsis))

    # Sort tornado by impact
    tornado_topsis.sort(key=lambda t: abs(t[2] - t[1]), reverse=True)

    stability = same_count / max(total_perturbations, 1)

    logger.info(
        "TOPSIS sensitivity: stability=%.1f%% (%d/%d same best soil)",
        stability * 100, same_count, total_perturbations,
    )

    return TopsisWeightSensitivityResult(
        original_weights=orig_weights.tolist(),
        criteria_names=CRITERIA_NAMES,
        original_best_soil=base_best_name,
        original_best_index=base_best_idx,
        perturbation_results=perturbation_results,
        stability_ratio=stability,
        tornado_topsis=tornado_topsis,
    )
 