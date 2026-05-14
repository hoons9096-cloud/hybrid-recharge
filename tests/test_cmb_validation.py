"""Tests for CMB (Chloride Mass Balance) validation module."""

import unittest
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmb_validation import cmb_recharge, cmb_timeseries, cmb_multi_well


class TestCMBRechargeBasic(unittest.TestCase):
    """Basic CMB formula verification."""

    def test_simple_case(self):
        """R = Cl_p × P / Cl_gw = 1.5 × 1200 / 15 = 120 mm/yr."""
        r = cmb_recharge(cl_precip_mg_l=1.5, cl_gw_mg_l=15.0, precip_mm_yr=1200)
        self.assertAlmostEqual(r.recharge_mm_yr, 120.0, places=1)
        self.assertAlmostEqual(r.recharge_ratio_pct, 10.0, places=1)

    def test_with_dry_deposition(self):
        """Dry deposition adds to effective Cl_p."""
        # D = 300 mg/m²/yr, P = 1000 mm/yr → D/P = 0.3 mg/L added
        # Cl_total = 1.0 + 0.3 = 1.3, R = 1.3 × 1000 / 13 = 100
        r = cmb_recharge(
            cl_precip_mg_l=1.0, cl_gw_mg_l=13.0,
            precip_mm_yr=1000, dry_deposition_mg_m2_yr=300,
        )
        self.assertAlmostEqual(r.recharge_mm_yr, 100.0, places=1)

    def test_enrichment_factor(self):
        r = cmb_recharge(cl_precip_mg_l=2.0, cl_gw_mg_l=40.0, precip_mm_yr=800)
        self.assertAlmostEqual(r.cl_ratio, 20.0)


class TestCMBWarnings(unittest.TestCase):
    """Assumption violation warnings."""

    def test_cl_gw_less_than_cl_p(self):
        """Cl_gw < Cl_p should trigger warning."""
        r = cmb_recharge(cl_precip_mg_l=10.0, cl_gw_mg_l=5.0, precip_mm_yr=1000)
        self.assertTrue(any("Cl_gw" in w and "≤" in w for w in r.assumption_warnings))

    def test_very_high_cl_gw(self):
        """Cl_gw > 250 should warn about contamination."""
        r = cmb_recharge(cl_precip_mg_l=2.0, cl_gw_mg_l=300.0, precip_mm_yr=1000)
        self.assertTrue(any("very high" in w.lower() or "halite" in w.lower()
                            for w in r.assumption_warnings))

    def test_low_enrichment(self):
        """Enrichment < 2× should warn about sensitivity."""
        r = cmb_recharge(cl_precip_mg_l=5.0, cl_gw_mg_l=8.0, precip_mm_yr=1000)
        self.assertTrue(any("enrichment" in w.lower() for w in r.assumption_warnings))

    def test_implausible_recharge(self):
        """R > 80% of P should warn."""
        r = cmb_recharge(cl_precip_mg_l=10.0, cl_gw_mg_l=11.0, precip_mm_yr=1000)
        self.assertTrue(any("implausible" in w.lower() for w in r.assumption_warnings))


class TestCMBvsWTF(unittest.TestCase):
    """WTF comparison metrics."""

    def test_good_agreement(self):
        r = cmb_recharge(
            cl_precip_mg_l=1.5, cl_gw_mg_l=15.0,
            precip_mm_yr=1200, wtf_recharge_mm_yr=130,
        )
        self.assertIsNotNone(r.ratio_cmb_to_wtf)
        self.assertIsNotNone(r.relative_error_pct)
        # CMB=120, WTF=130 → ratio ≈ 0.92, rel_err ≈ -7.7%
        self.assertAlmostEqual(r.ratio_cmb_to_wtf, 120 / 130, places=2)
        self.assertTrue(abs(r.relative_error_pct) < 20)

    def test_no_wtf(self):
        """Without WTF input, comparison fields should be None."""
        r = cmb_recharge(cl_precip_mg_l=1.5, cl_gw_mg_l=15.0, precip_mm_yr=1200)
        self.assertIsNone(r.ratio_cmb_to_wtf)
        self.assertIsNone(r.relative_error_pct)


class TestCMBInputValidation(unittest.TestCase):

    def test_negative_cl_gw(self):
        with self.assertRaises(ValueError):
            cmb_recharge(cl_precip_mg_l=1.0, cl_gw_mg_l=-5.0, precip_mm_yr=1000)

    def test_zero_precip(self):
        with self.assertRaises(ValueError):
            cmb_recharge(cl_precip_mg_l=1.0, cl_gw_mg_l=10.0, precip_mm_yr=0)


class TestCMBTimeseries(unittest.TestCase):
    """Time-series CMB with seasonal variation."""

    def test_uniform_series(self):
        """Uniform Cl and P should match simple CMB."""
        n = 12
        result = cmb_timeseries(
            cl_precip_series=[1.5] * n,
            cl_gw_series=[15.0] * n,
            precip_series=[100.0] * n,  # 1200 mm/yr total
        )
        self.assertAlmostEqual(result["recharge_total_mm"], 120.0, places=1)
        self.assertAlmostEqual(result["recharge_ratio_pct"], 10.0, places=1)

    def test_varying_cl_gw(self):
        """Higher Cl_gw → lower recharge for that period."""
        result = cmb_timeseries(
            cl_precip_series=[1.0, 1.0],
            cl_gw_series=[10.0, 20.0],
            precip_series=[500.0, 500.0],
        )
        # Period 1: 1×500/10=50, Period 2: 1×500/20=25
        self.assertAlmostEqual(result["recharge_total_mm"], 75.0, places=1)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            cmb_timeseries([1.0], [10.0, 20.0], [500.0])


class TestCMBMultiWell(unittest.TestCase):
    """Multi-well CMB comparison."""

    def test_returns_list(self):
        results = cmb_multi_well(
            cl_precip_mg_l=1.5,
            cl_gw_values=[10.0, 15.0, 20.0],
            precip_mm_yr=1200,
        )
        self.assertEqual(len(results), 3)
        # Lower Cl_gw → higher recharge
        self.assertGreater(results[0].recharge_mm_yr, results[1].recharge_mm_yr)
        self.assertGreater(results[1].recharge_mm_yr, results[2].recharge_mm_yr)

    def test_with_wtf_values(self):
        results = cmb_multi_well(
            cl_precip_mg_l=1.5,
            cl_gw_values=[15.0, 20.0],
            precip_mm_yr=1200,
            wtf_recharge_values=[130.0, 85.0],
        )
        for r in results:
            self.assertIsNotNone(r.ratio_cmb_to_wtf)


class TestCMBSummary(unittest.TestCase):
    """Summary string generation."""

    def test_summary_contains_key_info(self):
        r = cmb_recharge(
            cl_precip_mg_l=1.5, cl_gw_mg_l=15.0,
            precip_mm_yr=1200, wtf_recharge_mm_yr=130,
        )
        s = r.summary()
        self.assertIn("CMB recharge", s)
        self.assertIn("WTF", s)
        self.assertIn("agreement", s.lower())


if __name__ == "__main__":
    unittest.main()
