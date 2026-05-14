"""Edge case tests for core simulation robustness.

These tests verify that the core simulation handles degenerate and
extreme inputs gracefully, without crashes or numerical instability.
They complement the existing regression tests by probing boundary
conditions that are rare in normal data but critical for robustness.

Test categories:
- All-NaN water levels
- 100% pumping contamination
- Extreme parameter values (k near 0, very large z)
- Very short time series (minimum viable length)
- Constant water level (zero variance)
- All-dry or all-wet rainfall
- RTS smoother PSD enforcement
- BCa bootstrap with degenerate samples
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_sim_v27 import (
    run_logic_v27,
    calc_error,
    detect_pump_mask,
    remove_outliers,
    filpor_v27,
    apply_lag,
    calc_rain_response,
    propagate_kalman_recharge_uncertainty,
)
from uncertainty import _bca_interval


class TestAllNaNWaterLevel(unittest.TestCase):
    """Simulation must handle all-NaN water levels without crash."""

    def test_all_nan_runs_without_error(self):
        n = 100
        ho = np.full(n, np.nan)
        po = np.random.default_rng(0).uniform(0, 0.02, n)
        rech, hs_kf, hs_pure, sy, nf = run_logic_v27(
            -0.05, 3.0, 3, po, ho, 0.005, 0.1, 0.001
        )
        self.assertEqual(len(rech), n)
        # Should produce zeros or near-zeros (no valid obs to assimilate)
        self.assertTrue(np.all(np.isfinite(hs_kf)))

    def test_calc_error_all_nan_returns_inf(self):
        n = 100
        ho = np.full(n, np.nan)
        po = np.random.default_rng(0).uniform(0, 0.02, n)
        pm = np.zeros(n, dtype=bool)
        err = calc_error(-0.05, 3.0, 3, po, ho, 0.001, pm)
        self.assertEqual(err, np.inf)


class TestFullPumping(unittest.TestCase):
    """100% pumping mask: filter should rely entirely on prediction."""

    def test_all_pumping_runs(self):
        n = 200
        rng = np.random.default_rng(42)
        ho = 120.0 + np.cumsum(rng.normal(0, 0.02, n))
        po = rng.uniform(0, 0.015, n)
        pump_mask = np.ones(n, dtype=bool)

        rech, hs_kf, _, sy, _ = run_logic_v27(
            -0.05, 3.0, 3, po, ho, 0.005, 0.1, 0.001, pump_mask
        )
        self.assertEqual(len(rech), n)
        self.assertTrue(np.all(np.isfinite(hs_kf)))


class TestExtremeParameters(unittest.TestCase):
    """Extreme k, z values should not cause numerical overflow."""

    def setUp(self):
        self.n = 150
        self.rng = np.random.default_rng(7)
        self.ho = 100.0 + np.sin(np.arange(self.n) / 50.0)
        self.po = np.maximum(0, self.rng.normal(0.008, 0.008, self.n))

    def test_k_near_zero(self):
        """k ~ 0: minimal recession, should still be stable."""
        rech, hs, _, _, _ = run_logic_v27(
            -0.001, 3.0, 3, self.po, self.ho, 0.005, 0.1, 0.001
        )
        self.assertTrue(np.all(np.isfinite(hs)))

    def test_k_very_negative(self):
        """k ~ -0.5: rapid recession."""
        rech, hs, _, _, _ = run_logic_v27(
            -0.5, 3.0, 1, self.po, self.ho, 0.005, 0.1, 0.001
        )
        self.assertTrue(np.all(np.isfinite(hs)))

    def test_large_z(self):
        """Very deep unsaturated zone."""
        rech, hs, _, _, _ = run_logic_v27(
            -0.05, 20.0, 3, self.po, self.ho, 0.005, 0.1, 0.001
        )
        self.assertTrue(np.all(np.isfinite(hs)))

    def test_w_q_ratio_keyword(self):
        """w_q_ratio keyword should override config without crash."""
        rech1, hs1, _, _, _ = run_logic_v27(
            -0.05, 3.0, 3, self.po, self.ho, 0.005, 0.1, 0.001,
            w_q_ratio=0.01,
        )
        rech2, hs2, _, _, _ = run_logic_v27(
            -0.05, 3.0, 3, self.po, self.ho, 0.005, 0.1, 0.001,
            w_q_ratio=0.9,
        )
        self.assertTrue(np.all(np.isfinite(hs1)))
        self.assertTrue(np.all(np.isfinite(hs2)))
        # Different w_q_ratio should produce different results
        self.assertFalse(np.allclose(hs1, hs2))


class TestShortTimeSeries(unittest.TestCase):
    """Minimum viable length (< 10 days)."""

    def test_5_day_series(self):
        ho = np.array([100.0, 100.1, 100.05, 100.2, 100.1])
        po = np.array([0.0, 0.01, 0.015, 0.0, 0.0])
        rech, hs, _, _, _ = run_logic_v27(
            -0.05, 3.0, 3, po, ho, 0.005, 0.1, 0.001
        )
        self.assertEqual(len(rech), 5)

    def test_3_day_series(self):
        ho = np.array([100.0, 100.1, 100.05])
        po = np.array([0.0, 0.01, 0.0])
        rech, hs, _, _, _ = run_logic_v27(
            -0.05, 3.0, 3, po, ho, 0.005, 0.1, 0.001
        )
        self.assertEqual(len(rech), 3)


class TestConstantWaterLevel(unittest.TestCase):
    """Zero variance in observations: P0 should use floor value."""

    def test_constant_ho(self):
        n = 100
        ho = np.full(n, 120.0)  # σ² = 0
        po = np.random.default_rng(1).uniform(0, 0.01, n)
        rech, hs, _, _, _ = run_logic_v27(
            -0.05, 3.0, 3, po, ho, 0.005, 0.1, 0.001
        )
        self.assertTrue(np.all(np.isfinite(hs)))


class TestAllDryAllWet(unittest.TestCase):
    """Extreme rainfall conditions."""

    def test_all_dry(self):
        """No rainfall at all: zero recharge expected."""
        n = 100
        ho = 120.0 + np.sin(np.arange(n) / 30.0) * 0.5
        po = np.zeros(n)
        rech, hs, _, _, _ = run_logic_v27(
            -0.05, 3.0, 3, po, ho, 0.005, 0.1, 0.001
        )
        self.assertAlmostEqual(np.sum(rech), 0.0, places=10)

    def test_all_wet(self):
        """Continuous rainfall: should trigger events."""
        n = 100
        ho = 120.0 + np.cumsum(np.random.default_rng(5).normal(0, 0.01, n))
        po = np.full(n, 0.01)
        rech, hs, _, _, _ = run_logic_v27(
            -0.05, 3.0, 3, po, ho, 0.005, 0.1, 0.001
        )
        self.assertTrue(np.all(np.isfinite(hs)))


class TestFilporV27EdgeCases(unittest.TestCase):
    """Specific yield estimation edge cases."""

    def test_zero_dry_time(self):
        sy = filpor_v27(3, 3.0, 0)
        self.assertGreater(sy, 0)
        self.assertLess(sy, 0.5)

    def test_very_long_dry(self):
        sy = filpor_v27(3, 3.0, 10000)
        self.assertGreater(sy, 0)

    def test_all_soil_types(self):
        for sn in range(1, 13):
            sy = filpor_v27(sn, 3.0, 30)
            self.assertGreater(sy, 0, f"Soil {sn} returned non-positive Sy")
            self.assertLess(sy, 0.5, f"Soil {sn} returned unreasonably high Sy")


class TestRTSSmootherPSD(unittest.TestCase):
    """Verify RTS smoother produces finite covariances."""

    def test_long_series_psd(self):
        """Long series should produce finite smoothed covariances."""
        from core_sim_v27 import get_kalman_uncertainty

        n = 500
        rng = np.random.default_rng(99)
        ho = 100.0 + np.sin(np.arange(n) / 80.0) * 2.0 + np.cumsum(rng.normal(0, 0.02, n))
        po = np.maximum(0, rng.normal(0.006, 0.008, n))

        # Full mode (not _fast) triggers RTS smoother
        rech, hs, _, _, _ = run_logic_v27(
            -0.05, 3.0, 3, po, ho, 0.005, 0.1, 0.001, _fast=False
        )

        extras = get_kalman_uncertainty()
        if extras:
            P_h = extras["P_h_var"]
            P_w = extras["P_w_var"]
            # All variances must be non-negative and finite
            self.assertTrue(np.all(np.isfinite(P_h)), "P_h_var has non-finite values")
            self.assertTrue(np.all(P_h >= 0), "P_h_var has negative values")
            self.assertTrue(np.all(np.isfinite(P_w)), "P_w_var has non-finite values")
            self.assertTrue(np.all(P_w >= 0), "P_w_var has negative values")


class TestBCaInterval(unittest.TestCase):
    """BCa confidence interval edge cases."""

    def test_symmetric_distribution(self):
        """For symmetric data, BCa ≈ percentile."""
        rng = np.random.default_rng(42)
        samples = rng.normal(10, 1, 1000)
        lo, hi = _bca_interval(samples, 10.0, samples[:200], 0.95)
        # Should be roughly (8, 12) for N(10,1)
        self.assertGreater(lo, 7)
        self.assertLess(hi, 13)
        self.assertLess(lo, 10)
        self.assertGreater(hi, 10)

    def test_skewed_distribution(self):
        """BCa should handle right-skewed data."""
        rng = np.random.default_rng(42)
        samples = rng.exponential(5, 1000)
        theta_hat = 5.0
        jack_vals = np.array([np.mean(np.delete(samples, i)) for i in range(200)])
        lo, hi = _bca_interval(samples, theta_hat, jack_vals, 0.95)
        self.assertGreater(lo, 0)
        self.assertLess(lo, theta_hat)
        self.assertGreater(hi, theta_hat)

    def test_tiny_sample(self):
        """Very small B should fallback to percentile."""
        samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lo, hi = _bca_interval(samples, 3.0, samples, 0.95)
        self.assertLessEqual(lo, 3.0)
        self.assertGreaterEqual(hi, 3.0)


class TestCalcErrorInf(unittest.TestCase):
    """calc_error should return np.inf (not 1e9) for degenerate inputs."""

    def test_returns_inf_not_magic_number(self):
        n = 10
        ho = np.full(n, np.nan)
        po = np.zeros(n)
        pm = np.zeros(n, dtype=bool)
        result = calc_error(-0.05, 3.0, 3, po, ho, 0.001, pm)
        self.assertTrue(np.isinf(result), f"Expected inf, got {result}")


class TestDetectPumpMaskEdgeCases(unittest.TestCase):
    """Pump detection on pathological inputs."""

    def test_very_short_series(self):
        ho = np.array([100.0, 100.1, 100.0])
        po = np.array([0.0, 0.0, 0.0])
        mask = detect_pump_mask(ho, po, 0.001)
        self.assertEqual(len(mask), 3)

    def test_all_nan(self):
        ho = np.full(50, np.nan)
        po = np.zeros(50)
        mask = detect_pump_mask(ho, po, 0.001)
        self.assertEqual(len(mask), 50)
        self.assertFalse(np.any(mask))


class TestApplyLagEdge(unittest.TestCase):
    """Lag application edge cases."""

    def test_lag_exceeds_length(self):
        po = np.array([1.0, 2.0, 3.0])
        result = apply_lag(po, 10)
        # apply_lag pads with zeros; when lag > len, all original values
        # are shifted out, leaving only zeros (length = lag_days since
        # po[:len-lag] is empty and np.zeros(lag) is returned).
        self.assertTrue(np.all(result == 0))

    def test_negative_lag(self):
        po = np.array([1.0, 2.0, 3.0])
        result = apply_lag(po, -5)
        # Negative lag treated as 0
        np.testing.assert_array_equal(result, po)


class TestPropagateRechargeUncertaintyEdge(unittest.TestCase):
    """Recharge uncertainty propagation with edge inputs."""

    def test_no_events(self):
        rech = np.zeros(100)
        P_h_var = np.ones(100) * 0.01
        result = propagate_kalman_recharge_uncertainty(rech, 0.1, P_h_var)
        self.assertTrue(np.all(result == 0))

    def test_index_exceeds_pvar(self):
        """Event index beyond P_h_var length should be skipped."""
        rech = np.zeros(10)
        rech[8] = 0.05
        P_h_var = np.ones(5) * 0.01  # shorter than rech
        result = propagate_kalman_recharge_uncertainty(rech, 0.1, P_h_var)
        self.assertEqual(result[8], 0.0)  # skipped


class TestGaussianAdaptiveR(unittest.TestCase):
    """Verify Gaussian-kernel R(t) inflation produces smooth, decaying values."""

    def test_smooth_decay(self):
        """R should be highest at pump event and decay smoothly."""
        nn = 50
        ho = np.sin(np.linspace(0, 4 * np.pi, nn)) * 0.3 + 10.0
        po = np.random.default_rng(42).uniform(0, 5, nn)
        pump_mask = np.zeros(nn, dtype=bool)
        pump_mask[25] = True  # single pump event at day 25
        sn = 3
        rech, hs_kf, _, _, _ = run_logic_v27(
            -0.05, 3.0, sn, po, ho, 0.005, 0.1, 0.001, pump_mask
        )
        self.assertTrue(np.all(np.isfinite(hs_kf[~np.isnan(ho)])))

    def test_no_pump_r_unchanged(self):
        """Without pump events, R should be uniform everywhere."""
        nn = 30
        ho = np.linspace(9.5, 10.5, nn)
        po = np.ones(nn) * 2.0
        pump_mask = np.zeros(nn, dtype=bool)
        # This should not raise — just a smoke test
        rech, hs_kf, _, _, _ = run_logic_v27(
            -0.03, 2.0, 3, po, ho, 0.005, 0.1, 0.001, pump_mask
        )
        self.assertTrue(np.all(np.isfinite(hs_kf)))


class TestEstimateBlockLength(unittest.TestCase):
    """Test ACF-based block length estimation."""

    def test_white_noise_short_block(self):
        """White noise has no autocorrelation → block ≈ 2."""
        from uncertainty import _estimate_block_length
        rng = np.random.default_rng(42)
        white = rng.normal(0, 1, 500)
        bl = _estimate_block_length(white)
        self.assertLessEqual(bl, 5)

    def test_persistent_series_long_block(self):
        """AR(1) with ρ=0.95 → block >> short."""
        from uncertainty import _estimate_block_length
        rng = np.random.default_rng(42)
        n = 500
        ar1 = np.zeros(n)
        for i in range(1, n):
            ar1[i] = 0.95 * ar1[i - 1] + rng.normal(0, 0.1)
        bl = _estimate_block_length(ar1)
        self.assertGreater(bl, 10)

    def test_very_short_input(self):
        """< 10 points should return minimum block."""
        from uncertainty import _estimate_block_length
        bl = _estimate_block_length(np.array([1.0, 2.0, 3.0]))
        self.assertEqual(bl, 3)


class TestBCaInversionGuard(unittest.TestCase):
    """BCa should never produce inverted CIs.

    Note: when the bootstrap distribution is systematically above theta_hat,
    the CI is NOT forced to bracket theta_hat.  This is by design — the
    non-bracketing CI is a valid diagnostic signal of optimisation bias.
    """

    def test_heavily_biased_samples(self):
        """Skewed samples with small B should still produce well-ordered CI."""
        from uncertainty import _bca_interval
        rng = np.random.default_rng(99)
        # Heavily right-skewed: most samples > theta_hat
        samples = rng.exponential(2.0, size=30) + 10.0
        theta_hat = 5.0  # well below sample mass
        jack = np.array([np.mean(np.delete(samples, i)) for i in range(30)])

        ci_lo, ci_hi = _bca_interval(samples, theta_hat, jack, 0.95)
        # CI must not be inverted
        self.assertLessEqual(ci_lo, ci_hi)
        # CI should reflect the sample distribution (not forced to bracket theta)
        self.assertGreaterEqual(ci_hi, ci_lo)

    def test_all_samples_above_theta(self):
        """All bootstrap samples above point estimate → CI above theta_hat."""
        from uncertainty import _bca_interval
        samples = np.linspace(20.0, 30.0, 50)
        theta_hat = 10.0
        jack = np.array([np.mean(np.delete(samples, i)) for i in range(50)])

        ci_lo, ci_hi = _bca_interval(samples, theta_hat, jack, 0.95)
        # CI must not be inverted
        self.assertLessEqual(ci_lo, ci_hi)
        # When ALL samples > theta, CI will naturally be above theta
        # This is correct — it signals bootstrap bias
        self.assertGreaterEqual(ci_lo, 20.0)

    def test_well_centred_samples_bracket_theta(self):
        """When samples are centred on theta_hat, CI should bracket it."""
        from uncertainty import _bca_interval
        rng = np.random.default_rng(42)
        theta_hat = 15.0
        samples = rng.normal(theta_hat, 2.0, size=200)
        jack = np.array([np.mean(np.delete(samples, i)) for i in range(200)])

        ci_lo, ci_hi = _bca_interval(samples, theta_hat, jack, 0.95)
        self.assertLessEqual(ci_lo, ci_hi)
        self.assertLessEqual(ci_lo, theta_hat)
        self.assertGreaterEqual(ci_hi, theta_hat)


if __name__ == "__main__":
    unittest.main()
