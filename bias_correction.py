"""bias_correction.py — Phase 5: Bias-aware WTF correction model.

리뷰어 #1 정면 대응: "You improved inference, not the model".

목적
----
WTF 점추정을 truth 와 가깝게 만드는 *학습 기반 bias factor* β를 정량화.

수학적 모형
----------
관찰: 합성 벤치마크에서  R_WTF = R_true · (1 + β),
      β = β(soil_class, ET/P, drainage_τ, noise_level, ...)

단순 모형 (operationally usable):

    β̂(x) = β₀ + β₁ · (ET/P) + β₂ · log(τ_drainage) + β₃ · (HSG_index)
                + β₄ · σ_obs

학습 데이터: cascade truth 에서 N 개 합성 도메인 × M 개 셀
  → β_observed[i] = (R_WTF[i] - R_true[i]) / R_true[i]

검증: 5-fold cross-validation, RMSE / R²

적용:
    R_corrected = R_WTF / (1 + β̂(x))

이 모형은 WTF identity 자체가 가진 systematic 편향을 *외부 정보*
(토양·ET·관측노이즈) 로 보정한다.  Section 5 (논문 핵심 contribution).

References
----------
Crosbie et al. (2010) — texture-based recharge regression.
Healy (2010) §5.6 — WTF bias mitigation strategies.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# 결과 컨테이너
# ---------------------------------------------------------------------------
@dataclass
class BiasCorrectionResult:
    coefs: np.ndarray              # (n_features+1,) [intercept, β₁, β₂, ...]
    feature_names: List[str]
    r2_train: float
    r2_cv: float                   # 5-fold cross-validation R²
    rmse_cv_mm: float              # CV RMSE on β
    n_train: int
    bias_mean_before: float        # 평균 bias (보정 전)
    bias_mean_after: float         # 평균 bias (보정 후)
    rmse_before_mm: float          # truth-vs-WTF RMSE 보정 전
    rmse_after_mm: float           # 보정 후


# ---------------------------------------------------------------------------
# 데이터셋 구성 — synthetic 벤치마크에서 (R_WTF, R_true, x) 추출
# ---------------------------------------------------------------------------
def build_dataset(
    n_replicates: int = 30,
    scenarios: Optional[List[str]] = None,
    truth_model: str = "cascade",
    n_days: int = 730,
) -> pd.DataFrame:
    """N 합성 도메인에서 (per-cell) R_WTF, R_true, features → DataFrame."""
    from synthetic.scenarios import _CONFIG_FACTORY
    from synthetic.generate_domain import generate_domain
    from synthetic.generate_data import generate_data
    from methods.wtf_soil_weighted import estimate_recharge as fn
    from soil_db import SOIL_DB

    if scenarios is None:
        scenarios = ["S2", "S3", "S4", "S5"]

    rows = []
    for scn in scenarios:
        for rep in range(n_replicates):
            cfg = _CONFIG_FACTORY[scn]()
            cfg.random_seed = 5000 + rep
            dom = generate_domain(cfg)
            data = generate_data(dom, n_days=n_days, recharge_model=truth_model)
            n_yr = max(data.n_days / 365.25, 1.0)
            P_total_mm = float(np.sum(data.P)) * 1000.0 / n_yr
            ET_total_mm = float(np.sum(data.ET)) * 1000.0 / n_yr
            ET_over_P = ET_total_mm / max(P_total_mm, 1.0)
            obs_sigma = float(dom.config.obs_noise_std)

            obs = {
                "P": data.P, "ET": data.ET, "ho_obs": data.ho_obs,
                "well_soil_types": np.array([
                    int(dom.soil_map[dom.well_rows[w], dom.well_cols[w]])
                    for w in range(dom.n_wells)
                ]),
            }
            R_est = fn(dom, obs)
            R_true = data.true_recharge_annual

            ny, nx = R_true.shape
            for i in range(ny):
                for j in range(nx):
                    if R_true[i, j] < 1.0:    # 정답 0 근처 → 비율 정의 안 됨
                        continue
                    soil_idx = int(dom.soil_map[i, j])
                    sr = SOIL_DB[soil_idx]
                    rows.append({
                        "scenario": scn,
                        "replicate": rep,
                        "soil_idx": soil_idx,
                        "sy_lit": sr.sy_lit,
                        "tau_log": float(np.log(sr.tau)),
                        "ET_over_P": ET_over_P,
                        "obs_sigma": obs_sigma,
                        "R_true_mm": float(R_true[i, j]),
                        "R_est_mm": float(R_est[i, j]),
                        "beta": float((R_est[i, j] - R_true[i, j]) / R_true[i, j]),
                    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 학습: 단순 다중회귀 + 5-fold CV
# ---------------------------------------------------------------------------
def fit_bias_model(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    n_folds: int = 5,
    seed: int = 0,
) -> BiasCorrectionResult:
    if features is None:
        features = ["sy_lit", "tau_log", "ET_over_P", "obs_sigma"]

    df = df.dropna(subset=["beta"] + features).copy()
    # 극단치 클립 (β > 5 또는 < -2 — 비현실적인 ratio)
    df = df[(df["beta"] > -5.0) & (df["beta"] < 20.0)]

    X = df[features].values
    y = df["beta"].values
    n = len(y)

    # Augment with intercept
    X_aug = np.column_stack([np.ones(n), X])

    # Closed-form least squares
    coefs, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    y_pred = X_aug @ coefs
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2_train = 1.0 - ss_res / max(ss_tot, 1e-12)

    # 5-fold CV
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    fold_size = n // n_folds
    cv_pred = np.zeros(n)
    for k in range(n_folds):
        test_idx = idx[k * fold_size:(k + 1) * fold_size]
        train_idx = np.setdiff1d(np.arange(n), test_idx)
        Xt = X_aug[train_idx]; yt = y[train_idx]
        Xe = X_aug[test_idx]
        c, *_ = np.linalg.lstsq(Xt, yt, rcond=None)
        cv_pred[test_idx] = Xe @ c
    ss_res_cv = float(np.sum((y - cv_pred) ** 2))
    r2_cv = 1.0 - ss_res_cv / max(ss_tot, 1e-12)
    rmse_cv = float(np.sqrt(np.mean((y - cv_pred) ** 2)))

    # Before/after correction RMSE on R (mm/yr)
    R_true = df["R_true_mm"].values
    R_est = df["R_est_mm"].values
    bias_before = R_est / np.maximum(R_true, 1e-3) - 1.0
    bias_mean_before = float(np.mean(bias_before))
    rmse_before = float(np.sqrt(np.mean((R_est - R_true) ** 2)))

    R_corr = R_est / np.maximum(1.0 + cv_pred, 0.05)
    bias_after = R_corr / np.maximum(R_true, 1e-3) - 1.0
    bias_mean_after = float(np.mean(bias_after))
    rmse_after = float(np.sqrt(np.mean((R_corr - R_true) ** 2)))

    return BiasCorrectionResult(
        coefs=coefs,
        feature_names=["intercept"] + features,
        r2_train=float(r2_train),
        r2_cv=float(r2_cv),
        rmse_cv_mm=rmse_cv,
        n_train=n,
        bias_mean_before=bias_mean_before,
        bias_mean_after=bias_mean_after,
        rmse_before_mm=rmse_before,
        rmse_after_mm=rmse_after,
    )


# ---------------------------------------------------------------------------
# 적용: 학습된 β̂(x) 로 새 추정 보정
# ---------------------------------------------------------------------------
def apply_correction(
    R_est: np.ndarray,
    soil_indices: np.ndarray,
    ET_over_P: float,
    obs_sigma: float,
    coefs: np.ndarray,
    features: List[str],
    alpha: float = 1.0,
) -> np.ndarray:
    """학습된 β̂ 로 R_est 보정 (conservatism α 적용 가능).

    R_corr = R_est / (1 + α · β̂)

    α 의미:
        0.0  → no correction (원래 WTF)
        0.3  → 약한 보정 (conservative)
        0.5  → 절반 보정 (균형)
        1.0  → full cascade-truth 기준 (default)

    coefs: [intercept, sy_lit_coef, tau_log_coef, ET_over_P_coef, obs_sigma_coef]
    """
    from soil_db import SOIL_DB
    R_corr = np.zeros_like(R_est, dtype=float)
    soil_idx_unique = np.unique(soil_indices)
    for si in soil_idx_unique:
        sr = SOIL_DB[int(si)]
        x = np.array([1.0, sr.sy_lit, np.log(sr.tau), ET_over_P, obs_sigma])
        beta_hat = float(np.dot(coefs, x))
        denom = max(1.0 + alpha * beta_hat, 0.05)
        mask = (soil_indices == si)
        R_corr[mask] = R_est[mask] / denom
    return R_corr


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def cross_truth_validation(
    n_replicates: int = 8,
    n_folds: int = 5,
    seed: int = 0,
) -> Dict:
    """일반화 검증 — train on cascade, test on alpha (and vice versa).

    또한 leave-one-scenario-out (LOSO) 검증 — train on 3 scenarios, test on 4th.
    """
    print("▶ Building datasets for cross-truth validation…")
    df_cas = build_dataset(n_replicates=n_replicates, truth_model="cascade")
    df_alp = build_dataset(n_replicates=n_replicates, truth_model="alpha")

    out = {}
    # 1. Train cascade → test cascade (in-distribution baseline)
    res_in = fit_bias_model(df_cas, n_folds=n_folds, seed=seed)
    out["cascade_in_dist"] = {
        "r2_train": res_in.r2_train, "r2_cv": res_in.r2_cv,
        "rmse_before": res_in.rmse_before_mm, "rmse_after": res_in.rmse_after_mm,
        "n": res_in.n_train,
    }
    # 2. Train cascade → apply on alpha (out-of-distribution)
    coefs = res_in.coefs
    feats = ["sy_lit", "tau_log", "ET_over_P", "obs_sigma"]
    df_alp_clean = df_alp.dropna(subset=["beta"] + feats).copy()
    df_alp_clean = df_alp_clean[(df_alp_clean["beta"] > -5) & (df_alp_clean["beta"] < 20)]
    X = np.column_stack([np.ones(len(df_alp_clean)),
                         df_alp_clean[feats].values])
    beta_pred = X @ coefs
    R_est = df_alp_clean["R_est_mm"].values
    R_true = df_alp_clean["R_true_mm"].values
    R_corr = R_est / np.maximum(1.0 + beta_pred, 0.05)
    rmse_before = float(np.sqrt(np.mean((R_est - R_true) ** 2)))
    rmse_after = float(np.sqrt(np.mean((R_corr - R_true) ** 2)))
    bias_before = float(np.mean(R_est / np.maximum(R_true, 1e-3) - 1.0))
    bias_after = float(np.mean(R_corr / np.maximum(R_true, 1e-3) - 1.0))
    ss_tot = float(np.sum((df_alp_clean["beta"].values - df_alp_clean["beta"].mean()) ** 2))
    ss_res = float(np.sum((df_alp_clean["beta"].values - beta_pred) ** 2))
    r2_oo = 1.0 - ss_res / max(ss_tot, 1e-12)
    out["cascade_to_alpha_ood"] = {
        "r2_oo": r2_oo, "rmse_before": rmse_before, "rmse_after": rmse_after,
        "bias_before": bias_before, "bias_after": bias_after,
        "n": len(df_alp_clean),
    }

    # 3. Symmetric test (alpha-trained → cascade-tested)
    res_alp = fit_bias_model(df_alp, n_folds=n_folds, seed=seed)
    coefs_alp = res_alp.coefs
    df_cas_clean = df_cas.dropna(subset=["beta"] + feats).copy()
    df_cas_clean = df_cas_clean[(df_cas_clean["beta"] > -5) & (df_cas_clean["beta"] < 20)]
    Xc = np.column_stack([np.ones(len(df_cas_clean)),
                          df_cas_clean[feats].values])
    beta_pred_c = Xc @ coefs_alp
    R_est_c = df_cas_clean["R_est_mm"].values
    R_true_c = df_cas_clean["R_true_mm"].values
    R_corr_c = R_est_c / np.maximum(1.0 + beta_pred_c, 0.05)
    rmse_before_c = float(np.sqrt(np.mean((R_est_c - R_true_c) ** 2)))
    rmse_after_c = float(np.sqrt(np.mean((R_corr_c - R_true_c) ** 2)))
    bias_before_c = float(np.mean(R_est_c / np.maximum(R_true_c, 1e-3) - 1.0))
    bias_after_c = float(np.mean(R_corr_c / np.maximum(R_true_c, 1e-3) - 1.0))
    out["alpha_to_cascade_ood"] = {
        "rmse_before": rmse_before_c, "rmse_after": rmse_after_c,
        "bias_before": bias_before_c, "bias_after": bias_after_c,
        "n": len(df_cas_clean),
    }

    # 4. Leave-one-scenario-out (LOSO) on cascade truth
    scenarios = sorted(df_cas["scenario"].unique())
    loso = {}
    for held in scenarios:
        train = df_cas[df_cas["scenario"] != held]
        test = df_cas[df_cas["scenario"] == held]
        res_train = fit_bias_model(train, n_folds=3, seed=seed)
        cf = res_train.coefs
        test_clean = test.dropna(subset=["beta"] + feats).copy()
        test_clean = test_clean[(test_clean["beta"] > -5) & (test_clean["beta"] < 20)]
        Xt = np.column_stack([np.ones(len(test_clean)),
                              test_clean[feats].values])
        bp = Xt @ cf
        R_e = test_clean["R_est_mm"].values
        R_t = test_clean["R_true_mm"].values
        R_c = R_e / np.maximum(1.0 + bp, 0.05)
        rmse_b = float(np.sqrt(np.mean((R_e - R_t) ** 2)))
        rmse_a = float(np.sqrt(np.mean((R_c - R_t) ** 2)))
        loso[held] = {
            "rmse_before": rmse_b, "rmse_after": rmse_a,
            "improvement_pct": (1 - rmse_a / max(rmse_b, 1e-9)) * 100,
        }
    out["loso_cascade"] = loso

    return out


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_rep", type=int, default=20)
    ap.add_argument("--truth", default="cascade")
    ap.add_argument("--output", default="bias_model")
    args = ap.parse_args()

    print(f"▶ Building bias dataset (truth={args.truth}, n_rep={args.n_rep})…")
    df = build_dataset(n_replicates=args.n_rep, truth_model=args.truth)
    print(f"  N rows: {len(df)}")

    print("▶ Fitting bias correction model (5-fold CV)…")
    res = fit_bias_model(df)
    print(f"\n=== Bias Correction Model ===")
    for fn_, c in zip(res.feature_names, res.coefs):
        print(f"  {fn_:12s}: {c:+.4f}")
    print(f"\n  N_train      : {res.n_train}")
    print(f"  R² (train)   : {res.r2_train:.3f}")
    print(f"  R² (5-fold CV): {res.r2_cv:.3f}")
    print(f"  RMSE_CV(β)    : {res.rmse_cv_mm:.3f}")
    print(f"\n  Mean bias before: {res.bias_mean_before*100:+.1f}%")
    print(f"  Mean bias after : {res.bias_mean_after*100:+.1f}%")
    print(f"  RMSE  before    : {res.rmse_before_mm:.1f} mm/yr")
    print(f"  RMSE  after     : {res.rmse_after_mm:.1f} mm/yr  "
          f"(Δ {(1 - res.rmse_after_mm/res.rmse_before_mm)*100:.1f}% reduction)")

    # 저장
    out = {
        "coefs": res.coefs.tolist(),
        "feature_names": res.feature_names,
        "r2_train": res.r2_train,
        "r2_cv": res.r2_cv,
        "rmse_cv": res.rmse_cv_mm,
        "n_train": res.n_train,
        "bias_mean_before": res.bias_mean_before,
        "bias_mean_after": res.bias_mean_after,
        "rmse_before_mm": res.rmse_before_mm,
        "rmse_after_mm": res.rmse_after_mm,
        "truth_model": args.truth,
    }
    with open(f"{args.output}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✓ {args.output}.json")
