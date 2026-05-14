"""wtf_bias_corrected.py — Phase 5: Soil-weighted WTF + bias correction.

Pipeline:
    1. Soil-weighted WTF point estimate (methods/wtf_soil_weighted)
    2. Apply learned bias correction β̂(soil, ET/P, τ, σ_obs)
       → R_corrected = R_WTF / (1 + β̂)

학습된 β̂(coefficients) 는 cascade-truth 합성 데이터에서 사전 적합되며
`/tmp/bias_model.json` 또는 환경변수 `WTF_BIAS_MODEL` 로부터 로드한다.
모델 파일이 없으면 재학습 (build_dataset + fit_bias_model).

이 method 는 WTF identity 자체의 systematic bias 를 *학습 기반* 으로 보정해
"inference 만 개선했지 model 은 안 했다" 는 비판에 정면 대응한다.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methods.wtf_soil_weighted import estimate_recharge as soil_weighted_recharge


# ---------------------------------------------------------------------------
# 사전 학습된 β̂ 로드 (없으면 학습)
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH = os.environ.get(
    "WTF_BIAS_MODEL",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "bias_model.json"),
)


def _load_or_train_model():
    """β̂ 모델 로드.  없으면 cascade-truth 8 replicates 로 학습."""
    if os.path.exists(DEFAULT_MODEL_PATH):
        with open(DEFAULT_MODEL_PATH) as f:
            return json.load(f)
    # On-the-fly 학습
    from bias_correction import build_dataset, fit_bias_model
    df = build_dataset(n_replicates=8, truth_model="cascade")
    res = fit_bias_model(df)
    out = {
        "coefs": res.coefs.tolist(),
        "feature_names": res.feature_names,
        "r2_cv": res.r2_cv,
    }
    with open(DEFAULT_MODEL_PATH, "w") as f:
        json.dump(out, f, indent=2)
    return out


# ---------------------------------------------------------------------------
# 인터페이스 — methods 표준 (estimate_recharge(domain, observations))
# ---------------------------------------------------------------------------
def estimate_recharge(domain, observations):
    """Soil-weighted WTF + 학습 기반 bias correction."""
    from soil_db import SOIL_DB

    R_est = soil_weighted_recharge(domain, observations)

    # 보정 입력
    P = observations["P"]
    ET = observations["ET"]
    P_total_mm = float(np.sum(P)) * 1000.0
    ET_total_mm = float(np.sum(ET)) * 1000.0
    n_yr = max(len(P) / 365.25, 1.0)
    ET_over_P = ET_total_mm / max(P_total_mm, 1.0)
    obs_sigma = float(domain.config.obs_noise_std)

    model = _load_or_train_model()
    coefs = np.array(model["coefs"])

    # Apply correction per soil class
    R_corr = np.zeros_like(R_est, dtype=float)
    soil_map = domain.soil_map
    for si in np.unique(soil_map):
        sr = SOIL_DB[int(si)]
        x = np.array([1.0, sr.sy_lit, np.log(sr.tau), ET_over_P, obs_sigma])
        beta_hat = float(np.dot(coefs, x))
        denom = max(1.0 + beta_hat, 0.05)
        mask = (soil_map == si)
        R_corr[mask] = R_est[mask] / denom
    return R_corr
