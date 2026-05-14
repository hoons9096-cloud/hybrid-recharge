"""test_field_metrics.py — Field-mode metrics 테스트.

Ground-truth 없이 동작하는 4가지 일관성 지표가 합리적으로 동작하는지 검증한다.
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

# 프로젝트 루트 경로
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestBetweenMethodSpread(unittest.TestCase):
    def test_identical_methods_zero_spread(self):
        """완전히 같은 결과 → spread 0."""
        from evaluation.field_metrics import between_method_spread

        R = np.full((20, 20), 100.0)
        sp = between_method_spread({"A": R, "B": R.copy(), "C": R.copy()})
        self.assertAlmostEqual(sp.domain_mean_std, 0.0, places=6)
        self.assertAlmostEqual(sp.domain_mean, 100.0, places=6)
        np.testing.assert_array_equal(sp.range_map, np.zeros_like(R))

    def test_spread_increases_with_disagreement(self):
        from evaluation.field_metrics import between_method_spread

        rng = np.random.default_rng(0)
        base = np.full((30, 30), 100.0)
        a = base + 5.0 * rng.standard_normal(base.shape)
        b = base + 5.0 * rng.standard_normal(base.shape)
        c = base + 50.0 * rng.standard_normal(base.shape)

        sp_low = between_method_spread({"A": a, "B": b})
        sp_high = between_method_spread({"A": a, "B": b, "C": c})

        self.assertGreater(sp_high.domain_mean_std, sp_low.domain_mean_std)

    def test_requires_two_methods(self):
        from evaluation.field_metrics import between_method_spread

        with self.assertRaises(ValueError):
            between_method_spread({"only_one": np.zeros((5, 5))})

    def test_shape_mismatch_raises(self):
        from evaluation.field_metrics import between_method_spread

        with self.assertRaises(ValueError):
            between_method_spread({
                "A": np.zeros((5, 5)),
                "B": np.zeros((6, 6)),
            })


class TestPlausibilityCheck(unittest.TestCase):
    def test_clean_map_no_flags(self):
        """전형적 R/P=15% 결과는 모두 통과해야 한다."""
        from evaluation.field_metrics import plausibility_check

        rng = np.random.default_rng(0)
        # 1200 mm/yr 강수, R 평균 ~180 mm/yr (15%)
        R = 180.0 + 20.0 * rng.standard_normal((30, 30))
        rep = plausibility_check(R, P_annual_mm=1200.0)
        self.assertEqual(rep.n_negative, 0)
        self.assertEqual(rep.n_above_precip, 0)
        self.assertTrue(rep.pass_basic)

    def test_negative_flagged(self):
        from evaluation.field_metrics import plausibility_check

        R = np.full((10, 10), 100.0)
        R[0, 0] = -5.0
        rep = plausibility_check(R, P_annual_mm=1000.0)
        self.assertEqual(rep.n_negative, 1)
        self.assertFalse(rep.pass_basic)

    def test_above_precip_flagged(self):
        from evaluation.field_metrics import plausibility_check

        R = np.full((10, 10), 100.0)
        R[0, 0] = 2000.0  # > P
        rep = plausibility_check(R, P_annual_mm=1000.0)
        self.assertEqual(rep.n_above_precip, 1)
        self.assertFalse(rep.pass_basic)

    def test_zero_p_raises(self):
        from evaluation.field_metrics import plausibility_check

        with self.assertRaises(ValueError):
            plausibility_check(np.zeros((5, 5)), P_annual_mm=0.0)


class TestSoilClassCoherence(unittest.TestCase):
    def test_perfect_separation(self):
        """클래스별로 완전히 다른 R → coherence ≈ 1."""
        from evaluation.field_metrics import soil_class_coherence

        soil = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]])
        # 각 클래스 내부는 완전 동일
        R = np.array([[10, 10, 50, 50],
                      [10, 10, 50, 50],
                      [100, 100, 200, 200],
                      [100, 100, 200, 200]], dtype=float)
        coh = soil_class_coherence(R, soil)
        # within = 0 → coherence = 1
        self.assertAlmostEqual(coh.within_class_variance, 0.0, places=6)
        self.assertAlmostEqual(coh.coherence_ratio, 1.0, places=6)

    def test_pure_noise_low_coherence(self):
        """노이즈만 → coherence 낮음."""
        from evaluation.field_metrics import soil_class_coherence

        rng = np.random.default_rng(0)
        soil = rng.integers(1, 4, (40, 40))
        R = 100.0 + 30.0 * rng.standard_normal((40, 40))  # 토양과 무관
        coh = soil_class_coherence(R, soil)
        self.assertLess(coh.coherence_ratio, 0.2)

    def test_shape_mismatch_raises(self):
        from evaluation.field_metrics import soil_class_coherence

        with self.assertRaises(ValueError):
            soil_class_coherence(np.zeros((5, 5)), np.zeros((6, 6)))


class TestWellConsistency(unittest.TestCase):
    def test_runs_on_s3_synthetic(self):
        """End-to-end: S3 합성 데이터에서 well_consistency가 합리적 결과를 낸다."""
        from synthetic.generate_domain import generate_domain, DomainConfig
        from synthetic.generate_data import generate_data
        from methods.wtf_soil_weighted import estimate_recharge
        from evaluation.field_metrics import well_consistency

        domain = generate_domain(DomainConfig.S3())
        data = generate_data(domain)
        observations = {
            "P": data.P,
            "ET": data.ET,
            "ho_obs": data.ho_obs,
            "well_soil_types": data.well_soil_types,
        }
        R = estimate_recharge(domain, observations)

        wc = well_consistency(R, observations, domain, method_name="Soil-weighted")
        self.assertEqual(wc.n_wells, domain.n_wells)
        self.assertEqual(len(wc.records), domain.n_wells)
        # 각 record에 추정값이 채워져 있어야 함
        for rec in wc.records:
            self.assertGreaterEqual(rec.estimated_R, 0.0)


class TestFieldSummary(unittest.TestCase):
    def test_summary_runs_end_to_end(self):
        """field_summary가 S3 시나리오에서 문자열을 반환하고 4 섹션 포함."""
        from synthetic.generate_domain import generate_domain, DomainConfig
        from synthetic.generate_data import generate_data
        from methods.wtf_lumped import estimate_recharge as est_lumped
        from methods.wtf_soil_weighted import estimate_recharge as est_soil
        from evaluation.field_metrics import field_summary

        domain = generate_domain(DomainConfig.S3())
        data = generate_data(domain)
        observations = {
            "P": data.P,
            "ET": data.ET,
            "ho_obs": data.ho_obs,
            "well_soil_types": data.well_soil_types,
        }
        results = {
            "Lumped": est_lumped(domain, observations),
            "Soil-weighted": est_soil(domain, observations),
        }
        P_annual = float(np.sum(data.P)) * 1000.0 / (data.n_days / 365.0)
        report = field_summary(results, observations, domain, P_annual_mm=P_annual)

        self.assertIsInstance(report, str)
        self.assertIn("[1] Between-method spread", report)
        self.assertIn("[2] Physical plausibility", report)
        self.assertIn("[3] Soil-class coherence", report)
        self.assertIn("[4] Well-level consistency", report)


if __name__ == "__main__":
    unittest.main()
