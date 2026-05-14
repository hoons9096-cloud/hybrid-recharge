"""
scoring.py — Multi-Criteria Decision Making (MCDM) for soil type ranking.

Replaces the ad-hoc weighted-sum scoring in the original app with a
TOPSIS-based framework (Technique for Order of Preference by Similarity
to Ideal Solution).

References
----------
Hwang, C.L. & Yoon, K. (1981). Multiple Attribute Decision Making:
    Methods and Applications. Springer-Verlag.

Moriasi, D.N. et al. (2007). Model evaluation guidelines for systematic
    quantification of accuracy in watershed simulations.
    Trans. ASABE, 50(3), 885-900.
    (NSE performance ratings used for score_fit thresholds)

Design rationale
----------------
1. Each sub-criterion is normalised to [0, 100] with explicit formulas.
2. Criteria weights are documented with justification and sum to 1.0.
3. TOPSIS ranking provides a principled alternative to raw weighted sum.
4. Penalty logic for clay/pump contamination is preserved but documented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from soil_db import SOIL_DB, get_soil, CLAY_LIKE_SET


# ──────────────────────────────────────────────────────────
# Criteria weights — MUST sum to 1.0
# ──────────────────────────────────────────────────────────
# Justification table (based on expert judgement and sensitivity analysis):
#
# | Criterion        | Weight | Rationale                                    |
# |------------------|--------|----------------------------------------------|
# | k-stress         | 0.20   | k near boundary → unreliable parameter       |
# | Sy match         | 0.10   | Sy within literature confirms soil class      |
# | Goodness-of-fit  | 0.25   | NSE-based; primary calibration metric         |
# | Rain response    | 0.20   | Physical plausibility of timing               |
# | Recharge range   | 0.15   | Constraint: literature recharge bounds         |
# | Data cleanliness | 0.10   | Pump contamination degrades all other scores  |
#
# Total = 1.00
#
CRITERIA_WEIGHTS = np.array([0.20, 0.10, 0.25, 0.20, 0.15, 0.10])
CRITERIA_NAMES = [
    "k-Stress", "Sy Match", "Goodness-of-Fit",
    "Rain Response", "Recharge Range", "Cleanliness",
]

# All criteria are "benefit" type (higher = better)
_CRITERIA_BENEFICIAL = np.ones(len(CRITERIA_WEIGHTS), dtype=bool)


# ──────────────────────────────────────────────────────────
# Individual score functions — each returns [0, 100]
# ──────────────────────────────────────────────────────────
def score_k_stress(k_opt: float, soil_num: int) -> float:
    """Score how well k_opt sits within the allowed bounds.

    100 = exactly at centre of [k_min, k_max].
     20 = at the boundary.
      0 = outside bounds.
    Uses a linear mapping from boundary distance to score.
    """
    soil = get_soil(soil_num)
    k_min, k_max = soil.k_bounds

    if not (k_min <= k_opt <= k_max):
        violation = max(k_opt - k_max, k_min - k_opt)
        range_k = abs(k_max - k_min)
        return max(0.0, 50.0 - (violation / range_k) * 100.0)

    centre = (k_min + k_max) / 2.0
    half_range = abs(k_max - k_min) / 2.0
    if half_range < 1e-12:
        return 100.0
    normalised_distance = abs(k_opt - centre) / half_range  # [0, 1]
    return 100.0 - normalised_distance * 80.0  # 100→20 linear


def score_sy_match(sy_eff: float, soil_num: int) -> float:
    """Score how closely effective Sy matches literature Sy.

    100 = exact match.
      0 = ratio deviates by >= 50% from literature.
    """
    soil = get_soil(soil_num)
    sy_ref = soil.sy_lit
    if sy_ref < 1e-6:
        return 50.0
    ratio = sy_eff / sy_ref
    return max(0.0, 100.0 - abs(1.0 - ratio) * 200.0)


def score_goodness_of_fit(
    pure_rmse: float,
    kalman_rmse: float,
    sigma_ho: float,
) -> float:
    """NSE-based fit score mixing physical (pure WTF) and Kalman fits.

    Following Moriasi et al. (2007):
      NSE > 0.75  →  "very good"   →  score ~75–100
      NSE 0.50–0.75  →  "good"     →  score ~50–75
      NSE < 0.36  →  "unsatisfactory" → score < 36

    The mix (70% pure + 30% Kalman) prioritises physical fidelity over
    Kalman-corrected accuracy, avoiding rewarding filters that mask
    poor physics.
    """
    sigma_sq = max(sigma_ho ** 2, 1e-12)
    nse_pure = 1.0 - (pure_rmse ** 2) / sigma_sq
    nse_kf = 1.0 - (kalman_rmse ** 2) / sigma_sq
    nse_mixed = nse_pure * 0.70 + nse_kf * 0.30
    return max(0.0, min(100.0, nse_mixed * 100.0))


def score_rain_response(resp_obs: float, resp_sim: float) -> float:
    """Score similarity of observed vs simulated rain-rise fractions.

    Both inputs are fractions [0, 1].  A difference of 0.71 yields
    score = 0 (since 100 - 0.71*140 ≈ 0).
    """
    if np.isnan(resp_obs) or np.isnan(resp_sim):
        return 50.0  # no data → neutral score
    return max(0.0, 100.0 - abs(resp_obs - resp_sim) * 140.0)


def score_recharge_range(recharge_pct: float, soil_num: int) -> float:
    """Score whether recharge ratio falls within literature bounds.

    100 = inside range.
    Decays linearly outside, scaled by range width for fairness across
    soils with narrow vs wide expected ranges.
    """
    soil = get_soil(soil_num)
    lo, hi = soil.rech_range
    if lo <= recharge_pct <= hi:
        return 100.0
    range_width = max(hi - lo, 1.0)
    distance = min(abs(recharge_pct - lo), abs(recharge_pct - hi))
    normalised = distance / range_width
    return max(0.0, 100.0 - normalised * 100.0)


def score_cleanliness(
    pump_contam_idx: float,
    pump_events: int,
    pump_max_run: int,
) -> float:
    """Score data cleanliness (low pumping contamination = high score).

    Combines three pump indicators with documented sub-weights:
      - contamination fraction (dominant factor)
      - number of events
      - longest continuous run
    """
    pen = (
        pump_contam_idx * 90.0
        + min(pump_events * 3.0, 20.0)
        + min(max(pump_max_run - 2, 0) * 4.0, 20.0)
    )
    return max(0.0, 100.0 - pen)


# ──────────────────────────────────────────────────────────
# Composite scoring
# ──────────────────────────────────────────────────────────
@dataclass
class SoilScore:
    """Scoring result for one soil type."""
    soil_num: int
    soil_name: str

    # Individual criterion scores [0, 100]
    s_stress: float
    s_sy: float
    s_fit: float
    s_resp: float
    s_rech: float
    s_clean: float

    # Composite
    weighted_sum: float     # simple weighted sum
    topsis_score: float     # TOPSIS closeness coefficient [0, 1] × 100

    # Penalties and flags
    clay_penalty: float
    contam_penalty: float
    final_score: float      # after penalties, [0, 100]
    flag: str               # "양호", "주의", "보류권장"


def compute_soil_scores(row: pd.Series, *,
                        weights: np.ndarray | None = None) -> SoilScore:
    """Compute all sub-scores and composite for one soil scan row.

    Parameters
    ----------
    row : pd.Series with keys:
    weights : np.ndarray, optional
        Criteria weight vector (length 6, sums to 1.0).
        Default: ``CRITERIA_WEIGHTS`` module constant.
        Passing explicit weights avoids reliance on mutable global state
        and makes sensitivity analysis (weight perturbation) cleaner.
        Index, Recharge, OptK, SyEff, PureRMSE, RMSE, SigmaHo,
        RainRespObs, RainRespSim, PumpIdx, PumpEvents, PumpRun, Soil
    """
    sn = int(row["Index"])
    soil = get_soil(sn)

    s_stress = score_k_stress(
        float(row.get("OptK", 0)),
        sn,
    ) if "OptK" in row and pd.notna(row.get("OptK")) else 50.0

    s_sy = score_sy_match(
        float(row.get("SyEff", soil.sy_lit)),
        sn,
    )

    s_fit = score_goodness_of_fit(
        pure_rmse=float(row.get("PureRMSE", row["RMSE"])),
        kalman_rmse=float(row["RMSE"]),
        sigma_ho=max(float(row.get("SigmaHo", 0.1)), 1e-6),
    )

    s_resp = score_rain_response(
        float(row.get("RainRespObs", np.nan)),
        float(row.get("RainRespSim", np.nan)),
    )

    s_rech = score_recharge_range(
        float(row["Recharge"]),
        sn,
    )

    pump_idx = float(row.get("PumpIdx", 0))
    pump_events = int(row.get("PumpEvents", 0))
    pump_run = int(row.get("PumpRun", 0))

    s_clean = score_cleanliness(pump_idx, pump_events, pump_run)

    # ── Weighted sum ──
    _w = weights if weights is not None else CRITERIA_WEIGHTS
    scores = np.array([s_stress, s_sy, s_fit, s_resp, s_rech, s_clean])
    weighted_sum = float(np.dot(_w, scores))

    # ── Penalties (domain-specific, documented) ──
    # Clay soils with pump contamination: scoring is unreliable because
    # pump signatures in clay resemble natural recession behaviour.
    clay_pen = 0.0
    if sn in CLAY_LIKE_SET and pump_idx >= 0.15:
        clay_pen = 12.0 + 45.0 * min(pump_idx, 0.6)

    # High contamination: systematic negative bias on all criteria.
    contam_pen = 8.0 if pump_idx >= 0.35 else 0.0

    final = max(0.0, min(100.0, weighted_sum - clay_pen - contam_pen))

    flag = (
        "보류권장" if pump_idx >= 0.45
        else ("주의" if pump_idx >= 0.25 else "양호")
    )

    return SoilScore(
        soil_num=sn,
        soil_name=row.get("Soil", soil.name),
        s_stress=s_stress,
        s_sy=s_sy,
        s_fit=s_fit,
        s_resp=s_resp,
        s_rech=s_rech,
        s_clean=s_clean,
        weighted_sum=weighted_sum,
        topsis_score=0.0,  # filled by batch TOPSIS below
        clay_penalty=clay_pen,
        contam_penalty=contam_pen,
        final_score=final,
        flag=flag,
    )


def topsis_rank(score_list: List[SoilScore], *,
                weights: np.ndarray | None = None) -> List[SoilScore]:
    """Apply TOPSIS to the batch of soil scores and update topsis_score.

    Parameters
    ----------
    score_list : list of SoilScore
    weights : np.ndarray, optional
        Criteria weight vector (length 6, sums to 1.0).
        Default: ``CRITERIA_WEIGHTS`` module constant.

    TOPSIS steps (Hwang & Yoon, 1981):
    1. Construct the decision matrix (alternatives × criteria).
    2. Normalise by vector norm.
    3. Apply criteria weights.
    4. Determine ideal best (A+) and ideal worst (A-).
    5. Compute Euclidean distances to A+ and A-.
    6. Closeness coefficient C = d- / (d+ + d-), in [0, 1].

    The closeness coefficient is scaled to [0, 100] for display.
    """
    n = len(score_list)
    if n == 0:
        return score_list
    if n == 1:
        score_list[0].topsis_score = score_list[0].final_score
        return score_list

    # Step 1: decision matrix
    matrix = np.array([
        [s.s_stress, s.s_sy, s.s_fit, s.s_resp, s.s_rech, s.s_clean]
        for s in score_list
    ])  # shape (n, 6)

    # Step 2: vector normalisation
    norms = np.sqrt(np.sum(matrix ** 2, axis=0))
    norms[norms < 1e-12] = 1.0  # avoid division by zero
    normalised = matrix / norms  # (n, 6)

    # Step 3: weighted normalised matrix
    _w = weights if weights is not None else CRITERIA_WEIGHTS
    weighted = normalised * _w  # broadcast (n, 6)

    # Step 4: ideal best and worst
    # All criteria are benefit-type → ideal best = max, worst = min
    ideal_best = np.max(weighted, axis=0)
    ideal_worst = np.min(weighted, axis=0)

    # Step 5: Euclidean distances
    dist_best = np.sqrt(np.sum((weighted - ideal_best) ** 2, axis=1))
    dist_worst = np.sqrt(np.sum((weighted - ideal_worst) ** 2, axis=1))

    # Step 6: closeness coefficient
    denominator = dist_best + dist_worst
    denominator[denominator < 1e-12] = 1.0
    closeness = dist_worst / denominator  # [0, 1], higher = better

    # Apply penalties and scale to [0, 100]
    for i, s in enumerate(score_list):
        raw_topsis = closeness[i] * 100.0
        s.topsis_score = max(0.0, min(100.0,
            raw_topsis - s.clay_penalty - s.contam_penalty
        ))

    return score_list


def score_dataframe(df_scan: pd.DataFrame, *,
                    weights: np.ndarray | None = None) -> pd.DataFrame:
    """Score all rows, apply TOPSIS, and return an enriched DataFrame.

    Parameters
    ----------
    df_scan : pd.DataFrame
        Soil scan results from all 12 texture classes.
    weights : np.ndarray, optional
        Criteria weight vector (length 6). Propagated to
        ``compute_soil_scores`` and ``topsis_rank``.

    This is the main entry point replacing the old compute_hybrid_score_row.
    """
    scores = [compute_soil_scores(row, weights=weights) for _, row in df_scan.iterrows()]
    scores = topsis_rank(scores, weights=weights)

    df = df_scan.copy()
    df["HybridScore"] = [s.final_score for s in scores]
    df["TopsisScore"] = [s.topsis_score for s in scores]
    df["StressScore"] = [s.s_stress for s in scores]
    df["SyScore"] = [s.s_sy for s in scores]
    df["FitScore"] = [s.s_fit for s in scores]
    df["RespScore"] = [s.s_resp for s in scores]
    df["RechScore"] = [s.s_rech for s in scores]
    df["CleanScore"] = [s.s_clean for s in scores]
    df["ClayPenalty"] = [s.clay_penalty for s in scores]
    df["ContamPenalty"] = [s.contam_penalty for s in scores]
    df["RecoFlag"] = [s.flag for s in scores]

    df = df.sort_values("TopsisScore", ascending=False).reset_index(drop=True)
    return df
