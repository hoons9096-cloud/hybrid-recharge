"""run_block_cv.py — Compare random vs block (group) CV for the bias model.

Reviewer #4 concern (paper review):
  "270k cells 5-fold CV가 random split이면 spatial autocorrelation 때문에
   R²=0.61이 inflated."

Mitigation:
  Hold out entire (scenario, replicate) blocks from training. Each
  block is one independent synthetic domain — no cell from a held-out
  block appears in the training fold, so spatial leakage is impossible.

Output:
  - Random CV R² (baseline, current paper value)
  - Block CV R² (defensible)
  - ΔR² → reported alongside Table 4 in revision

Usage:
    python run_block_cv.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from bias_correction import build_dataset, fit_bias_model


def fit_with_groupkfold(df: pd.DataFrame,
                          features: list[str] | None = None,
                          n_folds: int = 5) -> dict:
    """Fit linear bias model under group-level CV. Groups = (scenario, replicate)."""
    if features is None:
        features = ["sy_lit", "tau_log", "ET_over_P", "obs_sigma"]

    df = df.dropna(subset=["beta"] + features).copy()
    df = df[(df["beta"] > -5.0) & (df["beta"] < 20.0)]
    df["group"] = df["scenario"].astype(str) + "_" + df["replicate"].astype(str)

    X = df[features].values
    y = df["beta"].values
    groups = df["group"].values
    n = len(y)

    X_aug = np.column_stack([np.ones(n), X])

    # All-data fit (training R²)
    coefs_all, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    y_pred_train = X_aug @ coefs_all
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2_train = 1.0 - float(np.sum((y - y_pred_train) ** 2)) / max(ss_tot, 1e-12)

    # GroupKFold
    try:
        from sklearn.model_selection import GroupKFold
    except ImportError:
        raise RuntimeError("sklearn required for GroupKFold")

    kf = GroupKFold(n_splits=n_folds)
    cv_pred = np.zeros(n)

    fold_metrics = []
    for fold_i, (tr, te) in enumerate(kf.split(X_aug, y, groups)):
        c, *_ = np.linalg.lstsq(X_aug[tr], y[tr], rcond=None)
        cv_pred[te] = X_aug[te] @ c
        rmse_te = float(np.sqrt(np.mean((y[te] - cv_pred[te]) ** 2)))
        held_out_groups = sorted(set(groups[te]))
        fold_metrics.append({
            "fold": fold_i + 1,
            "n_train": len(tr), "n_test": len(te),
            "n_groups_test": len(held_out_groups),
            "rmse_test": rmse_te,
        })

    ss_res_cv = float(np.sum((y - cv_pred) ** 2))
    r2_cv = 1.0 - ss_res_cv / max(ss_tot, 1e-12)
    rmse_cv = float(np.sqrt(np.mean((y - cv_pred) ** 2)))

    return {
        "method": "GroupKFold",
        "n": n, "n_groups": len(set(groups)),
        "r2_train": r2_train,
        "r2_cv": r2_cv,
        "rmse_cv": rmse_cv,
        "folds": fold_metrics,
    }


def main():
    print("=" * 70)
    print("Bias-correction CV comparison: random vs block (group)")
    print("=" * 70)

    print("\nBuilding dataset (S2-S5, 30 reps each, n_days=730)…")
    print("  (this matches the paper's random-CV setup; ~10 min)")
    t0 = time.time()
    df = build_dataset(
        n_replicates=30,
        scenarios=["S2", "S3", "S4", "S5"],
        truth_model="cascade",
        n_days=730,
    )
    print(f"  built in {time.time()-t0:.1f}s, n={len(df)}, "
          f"groups={df['scenario'].astype(str) + '_' + df['replicate'].astype(str)} "
          f"({(df['scenario'].astype(str) + '_' + df['replicate'].astype(str)).nunique()} unique)")

    # ── Random CV (baseline) ──────────────────────────────
    print("\n[1/2] Random 5-fold CV (current paper method)…")
    t0 = time.time()
    res_rand = fit_bias_model(df, n_folds=5)
    print(f"  R²_train = {res_rand.r2_train:.3f}")
    print(f"  R²_cv    = {res_rand.r2_cv:.3f}     ← paper value")
    print(f"  RMSE_cv  = {res_rand.rmse_cv_mm:.3f}")
    print(f"  elapsed: {time.time()-t0:.1f}s")

    # ── Block CV (proposed) ───────────────────────────────
    print("\n[2/2] Block (GroupKFold) 5-fold CV (defensible)…")
    t0 = time.time()
    res_blk = fit_with_groupkfold(df, n_folds=5)
    print(f"  R²_train  = {res_blk['r2_train']:.3f}")
    print(f"  R²_cv     = {res_blk['r2_cv']:.3f}     ← block CV")
    print(f"  RMSE_cv   = {res_blk['rmse_cv']:.3f}")
    print(f"  n_groups  = {res_blk['n_groups']}")
    print(f"  elapsed: {time.time()-t0:.1f}s")
    print("  Per-fold:")
    for f in res_blk["folds"]:
        print(f"    Fold {f['fold']}: n_test={f['n_test']:6d}  "
              f"n_groups_test={f['n_groups_test']:2d}  "
              f"RMSE={f['rmse_test']:.3f}")

    # ── Comparison ────────────────────────────────────────
    delta_r2 = res_blk["r2_cv"] - res_rand.r2_cv
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"  R²_cv (random) : {res_rand.r2_cv:.3f}")
    print(f"  R²_cv (block ) : {res_blk['r2_cv']:.3f}")
    print(f"  ΔR²            : {delta_r2:+.3f}")
    if delta_r2 < -0.05:
        verdict = "INFLATED — random CV substantially overstates fit"
    elif delta_r2 < -0.02:
        verdict = "MILDLY INFLATED — random CV slightly optimistic"
    elif delta_r2 < 0.02:
        verdict = "ROBUST — random and block CV agree"
    else:
        verdict = "BLOCK CV BETTER — unusual, may indicate data quirk"
    print(f"  VERDICT        : {verdict}")

    # Save results for inclusion in paper revision
    out = {
        "random_cv_r2": float(res_rand.r2_cv),
        "block_cv_r2": float(res_blk["r2_cv"]),
        "delta_r2": float(delta_r2),
        "n_obs": int(res_blk["n"]),
        "n_groups": int(res_blk["n_groups"]),
        "verdict": verdict,
        "random_rmse_cv": float(res_rand.rmse_cv_mm),
        "block_rmse_cv": float(res_blk["rmse_cv"]),
        "fold_metrics": res_blk["folds"],
    }
    import json
    out_path = ROOT / "paper" / "block_cv_comparison.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
