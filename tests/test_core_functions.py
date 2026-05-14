"""Tests for core simulation functions — calc_error, filpor, detect_pump_mask."""

import unittest
import numpy as np


class TestFilporV27(unittest.TestCase):
    def test_sand_high_sy(self):
        from core_sim_v27 import filpor_v27
        sy = filpor_v27(1, 3.0, 30)  # Sand, 3m depth, 30 days dry
        self.assertGreater(sy, 0.1)
        self.assertLess(sy, 0.45)

    def test_clay_low_sy(self):
        from core_sim_v27 import filpor_v27
        sy = filpor_v27(6, 10.0, 3)  # Clay, 10m depth, 3 days dry
        self.assertLess(sy, 0.05)

    def test_sy_monotonic_with_dry_time(self):
        from core_sim_v27 import filpor_v27
        sy_short = filpor_v27(3, 5.0, 1)
        sy_long = filpor_v27(3, 5.0, 30)
        self.assertLessEqual(sy_short, sy_long)

    def test_sy_clamped_above_minimum(self):
        from core_sim_v27 import filpor_v27
        sy = filpor_v27(6, 0.01, 0)  # minimal conditions
        self.assertGreaterEqual(sy, 0.001)


class TestDetectPumpMask(unittest.TestCase):
    def test_clean_data_no_pumping(self):
        from core_sim_v27 import detect_pump_mask
        np.random.seed(42)
        n = 200
        ho = 100.0 + np.cumsum(np.random.normal(0, 0.01, n))
        po = np.maximum(0, np.random.normal(0.005, 0.005, n))
        mask = detect_pump_mask(ho, po, 0.001)
        self.assertLess(mask.sum() / n, 0.2)  # <20% flagged

    def test_pump_spike_detected(self):
        from core_sim_v27 import detect_pump_mask
        n = 200
        ho = np.full(n, 100.0)
        po = np.full(n, 0.0)
        # Insert a sharp drop during dry period
        ho[100] = 98.0  # -2m drop
        ho[101] = 98.5
        mask = detect_pump_mask(ho, po, 0.001)
        # At least some days around the spike should be flagged
        self.assertTrue(mask[99:104].any())


class TestCalcError(unittest.TestCase):
    def test_returns_finite(self):
        from core_sim_v27 import calc_error
        np.random.seed(42)
        n = 365
        ho = 100.0 + np.sin(np.arange(n) / 50.0) * 0.5
        po = np.maximum(0, np.random.normal(0.005, 0.005, n))
        pm = np.zeros(n, dtype=bool)
        err = calc_error(-0.05, 3.0, 3, po, ho, 0.001, pm)
        self.assertTrue(np.isfinite(err))
        self.assertGreater(err, 0)

    def test_perfect_fit_low_error(self):
        """A simulation compared against itself should have low error.

        Note: calc_error returns a dimensionless multi-objective cost:
            w_fit(0.70)*NRMSE + w_resp(0.15)*resp_mismatch + w_rech(0.15)*rech_violation
        Even with perfect RMSE (NRMSE≈0), the rain-response and recharge-range
        penalty components may contribute up to ~0.30, so the threshold is set
        at 0.55 rather than near-zero.
        """
        from core_sim_v27 import run_logic_v27, calc_error, apply_lag
        np.random.seed(42)
        n = 365
        po = np.maximum(0, np.random.normal(0.005, 0.005, n))
        ho_seed = 100.0 + np.sin(np.arange(n) / 50.0) * 0.5
        pm = np.zeros(n, dtype=bool)
        # Generate simulated data and use as observations
        rech, hs, _, _, _ = run_logic_v27(-0.05, 3.0, 3, po, ho_seed, 0.005, 0.1, 0.001, pm)
        err = calc_error(-0.05, 3.0, 3, po, hs, 0.001, pm)
        self.assertLess(err, 0.55)  # dimensionless cost: NRMSE≈0 + penalties ≤0.30


class TestApplyLag(unittest.TestCase):
    def test_zero_lag(self):
        from core_sim_v27 import apply_lag
        po = np.array([1.0, 2.0, 3.0, 4.0])
        result = apply_lag(po, 0)
        np.testing.assert_array_equal(result, po)

    def test_positive_lag(self):
        from core_sim_v27 import apply_lag
        po = np.array([1.0, 2.0, 3.0, 4.0])
        result = apply_lag(po, 2)
        np.testing.assert_array_equal(result, [0.0, 0.0, 1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
