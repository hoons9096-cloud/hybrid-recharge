"""Tests for scoring.py — TOPSIS-based soil scoring."""

import unittest
import numpy as np
import pandas as pd


class TestIndividualScores(unittest.TestCase):
    def test_k_stress_centre_is_high(self):
        from scoring import score_k_stress
        from soil_db import get_bounds
        # Use actual bounds for Sand (soil 1)
        k_min, k_max = get_bounds(1)
        centre = (k_min + k_max) / 2.0
        s = score_k_stress(centre, 1)
        self.assertGreater(s, 80)

    def test_k_stress_boundary_is_low(self):
        from scoring import score_k_stress
        from soil_db import get_bounds
        k_min, k_max = get_bounds(1)
        s = score_k_stress(k_max, 1)  # at upper boundary
        self.assertLess(s, 50)
        self.assertGreater(s, 0)

    def test_k_stress_outside_is_very_low(self):
        from scoring import score_k_stress
        s = score_k_stress(0.0, 1)  # outside bounds
        self.assertLess(s, 50)

    def test_sy_match_exact(self):
        from scoring import score_sy_match
        s = score_sy_match(0.33, 1)  # exact match for Sand
        self.assertAlmostEqual(s, 100.0, places=1)

    def test_sy_match_far_off(self):
        from scoring import score_sy_match
        s = score_sy_match(0.01, 1)  # very low Sy for Sand
        self.assertLess(s, 10)

    def test_goodness_of_fit_perfect(self):
        from scoring import score_goodness_of_fit
        s = score_goodness_of_fit(0.001, 0.001, 0.3)  # near-perfect
        self.assertGreater(s, 95)

    def test_goodness_of_fit_bad(self):
        from scoring import score_goodness_of_fit
        s = score_goodness_of_fit(0.5, 0.5, 0.3)  # RMSE > sigma
        self.assertAlmostEqual(s, 0.0, places=1)

    def test_recharge_in_range(self):
        from scoring import score_recharge_range
        s = score_recharge_range(20.0, 1)  # Sand range 10-38
        self.assertAlmostEqual(s, 100.0, places=1)

    def test_recharge_out_of_range(self):
        from scoring import score_recharge_range
        s = score_recharge_range(60.0, 1)  # far above Sand range
        self.assertLess(s, 50)

    def test_rain_response_nan(self):
        from scoring import score_rain_response
        s = score_rain_response(np.nan, 0.5)
        self.assertAlmostEqual(s, 50.0, places=1)

    def test_cleanliness_clean(self):
        from scoring import score_cleanliness
        s = score_cleanliness(0.0, 0, 0)
        self.assertAlmostEqual(s, 100.0, places=1)


class TestCriteriaWeights(unittest.TestCase):
    def test_weights_sum_to_one(self):
        from scoring import CRITERIA_WEIGHTS
        self.assertAlmostEqual(float(np.sum(CRITERIA_WEIGHTS)), 1.0, places=6)


class TestTOPSIS(unittest.TestCase):
    def test_topsis_rank_ordering(self):
        from scoring import score_dataframe
        df = pd.DataFrame([
            {"Soil": "Sand", "Index": 1, "RMSE": 0.05, "PureRMSE": 0.06,
             "SigmaHo": 0.3, "Recharge": 20.0, "OptK": -0.20, "SyEff": 0.30,
             "RainRespObs": 0.7, "RainRespSim": 0.65,
             "PumpIdx": 0.05, "PumpEvents": 1, "PumpRun": 2},
            {"Soil": "Clay", "Index": 6, "RMSE": 0.25, "PureRMSE": 0.28,
             "SigmaHo": 0.2, "Recharge": 5.0, "OptK": -0.01, "SyEff": 0.04,
             "RainRespObs": 0.4, "RainRespSim": 0.2,
             "PumpIdx": 0.50, "PumpEvents": 8, "PumpRun": 10},
        ])
        result = score_dataframe(df)
        # Sand should rank higher than heavily contaminated Clay
        self.assertEqual(result.iloc[0]["Soil"], "Sand")

    def test_score_dataframe_columns(self):
        from scoring import score_dataframe
        df = pd.DataFrame([
            {"Soil": "Loam", "Index": 12, "RMSE": 0.08, "PureRMSE": 0.09,
             "SigmaHo": 0.25, "Recharge": 12.0, "OptK": -0.08, "SyEff": 0.12,
             "RainRespObs": 0.6, "RainRespSim": 0.55,
             "PumpIdx": 0.10, "PumpEvents": 2, "PumpRun": 3},
        ])
        result = score_dataframe(df)
        for col in ["HybridScore", "TopsisScore", "StressScore", "RecoFlag"]:
            self.assertIn(col, result.columns)


if __name__ == "__main__":
    unittest.main()
