"""
cross_validation.py — Split-sample and temporal cross-validation framework.

Provides temporal cross-validation methods appropriate for hydrological
time series, where random shuffling would violate temporal autocorrelation.

References
----------
Klemeš, V. (1986). Operational testing of hydrological simulation models.
    Hydrological Sciences Journal, 31(1), 13-24.
    (Split-sample test — the standard in hydrological model evaluation)

Refsgaard, J.C. & Knudsen, J. (1996). Operational validation and
    intercomparison of different types of hydrological models.
    Water Resources Research, 32(7), 2189-2202.

Usage
-----
    from cross_validation import split_sample_test, temporal_kfold_cv
    report = split_sample_test("SH11.txt", soil_num=3)
    print(report)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

from core_sim_config import DEFAULT_Q_NOISE, DEFAULT_R_NOISE
from core_sim_v27 import (
    core_sim_v27,
    run_logic_v27,
    apply_lag,
    detect_pump_mask,
    remove_outliers,
    load_core_data,
    normalize_core_inputs,
    optimize_parameters,
    build_core_metrics,
)


# ──────────────────────────────────────────────────────────
# Result dataclasses
# ──────────────────────────────────────────────────────────
@dataclass
class FoldResult:
    """Metrics for one calibration/validation split."""
    fold_id: int
    cal_start: int
    cal_end: int
    val_start: int
    val_end: int

    # Calibration metrics
    cal_rmse: float
    cal_cc: float
    cal_rech: float

    # Validation metrics
    val_rmse: float
    val_cc: float
    val_rech: float

    # Optimised parameters (from calibration period)
    opt_k: float
    opt_z: float
    opt_lag: int


@dataclass
class CVReport:
    """Aggregated cross-validation report."""
    method: str              # "split_sample" or "temporal_kfold"
    file_path: str
    soil_num: int
    n_folds: int
    folds: list = field(default_factory=list)

    # Summary statistics (computed after all folds)
    cal_rmse_mean: float = 0.0
    cal_rmse_std: float = 0.0
    val_rmse_mean: float = 0.0
    val_rmse_std: float = 0.0
    val_cc_mean: float = 0.0
    val_cc_std: float = 0.0
    val_rech_mean: float = 0.0
    val_rech_std: float = 0.0
    generalisation_ratio: float = 0.0  # val_rmse / cal_rmse (>1 = overfit)

    def to_dict(self):
        d = asdict(self)
        return d

    def summarise(self):
        """Compute summary statistics from fold results."""
        if not self.folds:
            return
        cal_rmses = [f.cal_rmse for f in self.folds]
        val_rmses = [f.val_rmse for f in self.folds]
        val_ccs = [f.val_cc for f in self.folds]
        val_rechs = [f.val_rech for f in self.folds]

        self.cal_rmse_mean = float(np.mean(cal_rmses))
        self.cal_rmse_std = float(np.std(cal_rmses))
        self.val_rmse_mean = float(np.mean(val_rmses))
        self.val_rmse_std = float(np.std(val_rmses))
        self.val_cc_mean = float(np.mean(val_ccs))
        self.val_cc_std = float(np.std(val_ccs))
        self.val_rech_mean = float(np.mean(val_rechs))
        self.val_rech_std = float(np.std(val_rechs))

        if self.cal_rmse_mean > 1e-9:
            self.generalisation_ratio = self.val_rmse_mean / self.cal_rmse_mean
        else:
            self.generalisation_ratio = 1.0


def _evaluate_period(ho_full, po_full, start, end, opt_k, opt_z, opt_lag,
                     sn, q_val, r_val, rc_val, pump_mask,
                     opt_rho=None, opt_alpha=None):
    """Run simulation on a sub-period and return (rmse, cc, recharge_ratio)."""
    ho_sub = ho_full[start:end].copy()
    po_sub = po_full[start:end].copy()
    pm_sub = pump_mask[start:end].copy()
    n = len(ho_sub)

    if n < 10:
        return np.nan, np.nan, np.nan

    # rho/alpha를 run_logic_v27에 직접 전달 — 전역 상태 변경 불필요
    po_shifted = apply_lag(po_sub, opt_lag)
    rech, hs_kf, hs_pure, sy_eff, n_f_avg = run_logic_v27(
        opt_k, opt_z, sn, po_shifted, ho_sub, q_val, r_val, rc_val, pm_sub,
        rho=opt_rho, alpha=opt_alpha,
    )

    valid = ~pm_sub & ~np.isnan(hs_kf) & ~np.isnan(ho_sub)
    if np.sum(valid) < 5:
        valid = ~np.isnan(hs_kf) & ~np.isnan(ho_sub)
    if np.sum(valid) < 3:
        return np.nan, np.nan, np.nan

    rmse = float(np.sqrt(np.nanmean((hs_kf[valid] - ho_sub[valid]) ** 2)))

    cc = 0.0
    ho_v = ho_sub[valid]
    hs_v = hs_kf[valid]
    if len(ho_v) >= 3:
        cc_mat = np.corrcoef(hs_v, ho_v)
        if cc_mat.size > 1:
            cc = float(cc_mat[0, 1])

    total_rain = max(float(np.sum(po_shifted)), 1e-9)
    rech_ratio = float(np.sum(rech)) / total_rain * 100.0

    return rmse, cc, rech_ratio


def _evaluate_discontiguous_cal(
    ho_full, po_full, val_s, val_e, n,
    opt_k, opt_z, opt_lag, sn, q_val, r_val, rc_val, pump_mask,
    opt_rho=None, opt_alpha=None,
):
    """비연속 calibration 구간을 각 세그먼트별로 평가 후 가중 평균을 반환.

    temporal_kfold_cv에서 validation 구간(val_s:val_e)을 제외한
    calibration 구간은 최대 2개의 연속 세그먼트로 구성됨:
      - 전반 세그먼트: [0, val_s)
      - 후반 세그먼트: [val_e, n)

    각 세그먼트를 독립 시뮬레이션으로 평가한 후,
    유효 관측 수를 가중치로 하는 가중 평균을 사용함.

    RMSE 가중 평균: sqrt(sum(n_i * RMSE_i^2) / sum(n_i))
    CC, recharge_ratio: sum(n_i * metric_i) / sum(n_i)

    References
    ----------
    Klemeš, V. (1986). Operational testing of hydrological simulation models.
        Hydrological Sciences Journal, 31(1), 13-24.
    """
    segments = []
    if val_s > 0:
        segments.append((0, val_s))
    if val_e < n:
        segments.append((val_e, n))

    if not segments:
        return np.nan, np.nan, np.nan

    rmse_wsum = 0.0
    cc_wsum = 0.0
    rech_wsum = 0.0
    w_total = 0.0

    for seg_s, seg_e in segments:
        seg_len = seg_e - seg_s
        rmse_s, cc_s, rech_s = _evaluate_period(
            ho_full, po_full, seg_s, seg_e,
            opt_k, opt_z, opt_lag, sn, q_val, r_val, rc_val, pump_mask,
            opt_rho=opt_rho, opt_alpha=opt_alpha,
        )
        if np.isnan(rmse_s):
            continue
        w = float(seg_len)
        rmse_wsum += w * rmse_s ** 2
        cc_wsum += w * cc_s
        rech_wsum += w * rech_s
        w_total += w

    if w_total < 1.0:
        return np.nan, np.nan, np.nan

    cal_rmse = float(np.sqrt(rmse_wsum / w_total))
    cal_cc = float(cc_wsum / w_total)
    cal_rech = float(rech_wsum / w_total)
    return cal_rmse, cal_cc, cal_rech


# ──────────────────────────────────────────────────────────
# Split-sample test (Klemeš, 1986)
# ──────────────────────────────────────────────────────────
def split_sample_test(
    file_path: str,
    soil_num: int = 3,
    k_init: float = -0.05,
    z_init: float = 3.0,
    q_val: float = DEFAULT_Q_NOISE,
    r_val: float = DEFAULT_R_NOISE,
    rc_val: float = 0.001,
    sens_val: float = 5.0,
    split_ratio: float = 0.5,
) -> CVReport:
    """Classic split-sample test: calibrate on first half, validate on second.

    Also performs the reverse split (calibrate second half, validate first)
    as recommended by Klemeš (1986), yielding 2 folds.
    """
    data = load_core_data(file_path)
    ho = data.ho.ravel().astype(float)
    po = data.po.ravel().astype(float)
    n = len(ho)

    ho = remove_outliers(ho, sens_val)
    pump_mask = detect_pump_mask(ho, po, rc_val)
    sn, kb_min, kb_max, ok, oz, olag = normalize_core_inputs(soil_num, k_init, z_init, 0)

    split_idx = int(n * split_ratio)

    report = CVReport(
        method="split_sample",
        file_path=file_path,
        soil_num=soil_num,
        n_folds=2,
    )

    for fold_id, (cal_s, cal_e, val_s, val_e) in enumerate([
        (0, split_idx, split_idx, n),       # Forward split
        (split_idx, n, 0, split_idx),        # Reverse split
    ]):
        # Calibrate on cal period
        ho_cal = ho[cal_s:cal_e].copy()
        po_cal = po[cal_s:cal_e].copy()
        pm_cal = pump_mask[cal_s:cal_e].copy()

        opt_k, opt_z, opt_lag, opt_rho, opt_alpha = optimize_parameters(
            ho_cal, po_cal, sn, ok, oz, rc_val, pm_cal
        )

        # Evaluate calibration period
        cal_rmse, cal_cc, cal_rech = _evaluate_period(
            ho, po, cal_s, cal_e, opt_k, opt_z, opt_lag,
            sn, q_val, r_val, rc_val, pump_mask,
            opt_rho=opt_rho, opt_alpha=opt_alpha
        )

        # Evaluate validation period (same parameters, different data)
        val_rmse, val_cc, val_rech = _evaluate_period(
            ho, po, val_s, val_e, opt_k, opt_z, opt_lag,
            sn, q_val, r_val, rc_val, pump_mask,
            opt_rho=opt_rho, opt_alpha=opt_alpha
        )

        report.folds.append(FoldResult(
            fold_id=fold_id,
            cal_start=cal_s, cal_end=cal_e,
            val_start=val_s, val_end=val_e,
            cal_rmse=cal_rmse, cal_cc=cal_cc, cal_rech=cal_rech,
            val_rmse=val_rmse, val_cc=val_cc, val_rech=val_rech,
            opt_k=opt_k, opt_z=opt_z, opt_lag=opt_lag,
        ))

    report.summarise()
    return report


# ──────────────────────────────────────────────────────────
# Temporal k-fold CV
# ──────────────────────────────────────────────────────────
def temporal_kfold_cv(
    file_path: str,
    soil_num: int = 3,
    k_init: float = -0.05,
    z_init: float = 3.0,
    q_val: float = DEFAULT_Q_NOISE,
    r_val: float = DEFAULT_R_NOISE,
    rc_val: float = 0.001,
    sens_val: float = 5.0,
    n_folds: int = 5,
) -> CVReport:
    """Temporal k-fold cross-validation.

    The record is divided into n_folds contiguous blocks.  For each fold,
    one block is held out for validation and the remaining blocks are used
    for calibration.

    This respects temporal structure: each fold validates on a different
    period, testing the model's ability to generalise across seasons.
    """
    data = load_core_data(file_path)
    ho = data.ho.ravel().astype(float)
    po = data.po.ravel().astype(float)
    n = len(ho)

    ho = remove_outliers(ho, sens_val)
    pump_mask = detect_pump_mask(ho, po, rc_val)
    sn, kb_min, kb_max, ok, oz, olag = normalize_core_inputs(soil_num, k_init, z_init, 0)

    fold_size = n // n_folds
    if fold_size < 30:
        raise ValueError(
            f"Record too short ({n} days) for {n_folds}-fold CV. "
            f"Need at least {30 * n_folds} days."
        )

    report = CVReport(
        method="temporal_kfold",
        file_path=file_path,
        soil_num=soil_num,
        n_folds=n_folds,
    )

    for fold_id in range(n_folds):
        val_s = fold_id * fold_size
        val_e = val_s + fold_size if fold_id < n_folds - 1 else n

        # Calibration = everything outside validation window
        cal_indices = np.concatenate([
            np.arange(0, val_s),
            np.arange(val_e, n),
        ])
        if len(cal_indices) < 30:
            continue

        ho_cal = ho[cal_indices].copy()
        po_cal = po[cal_indices].copy()
        pm_cal = pump_mask[cal_indices].copy()

        opt_k, opt_z, opt_lag, opt_rho, opt_alpha = optimize_parameters(
            ho_cal, po_cal, sn, ok, oz, rc_val, pm_cal
        )

        # Calibration 성능: validation 기간을 제외한 구간만 평가
        # (수정 전: 전체 기간 0~n 사용 → data leakage 발생)
        # 비연속 cal 구간을 각 세그먼트별 독립 평가 후 가중 평균.
        # Ref: Klemeš (1986) HSJ 31(1):13-24
        cal_rmse, cal_cc, cal_rech = _evaluate_discontiguous_cal(
            ho, po, val_s, val_e, n, opt_k, opt_z, opt_lag,
            sn, q_val, r_val, rc_val, pump_mask,
            opt_rho=opt_rho, opt_alpha=opt_alpha,
        )
        val_rmse, val_cc, val_rech = _evaluate_period(
            ho, po, val_s, val_e, opt_k, opt_z, opt_lag,
            sn, q_val, r_val, rc_val, pump_mask,
            opt_rho=opt_rho, opt_alpha=opt_alpha
        )

        report.folds.append(FoldResult(
            fold_id=fold_id,
            cal_start=int(cal_indices[0]), cal_end=int(cal_indices[-1]),
            val_start=val_s, val_end=val_e,
            cal_rmse=cal_rmse, cal_cc=cal_cc, cal_rech=cal_rech,
            val_rmse=val_rmse, val_cc=val_cc, val_rech=val_rech,
            opt_k=opt_k, opt_z=opt_z, opt_lag=opt_lag,
        ))

    report.summarise()
    return report
