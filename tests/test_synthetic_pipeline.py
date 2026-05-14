"""test_synthetic_pipeline.py — 합성 벤치마크 파이프라인 테스트."""

import sys
import os
import unittest

import numpy as np

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestGenerateDomain(unittest.TestCase):
    """generate_domain.py 테스트."""

    def test_s3_domain_shape(self):
        from synthetic.generate_domain import generate_domain, DomainConfig
        domain = generate_domain(DomainConfig.S3())
        self.assertEqual(domain.soil_map.shape, (100, 100))
        self.assertEqual(domain.Sy_map.shape, (100, 100))
        self.assertEqual(domain.n_wells, 25)

    def test_s1_homogeneous(self):
        from synthetic.generate_domain import generate_domain, DomainConfig
        domain = generate_domain(DomainConfig.S1())
        # 균질: 단일 토양 유형
        unique = np.unique(domain.soil_map)
        self.assertEqual(len(unique), 1)

    def test_s3_strong_heterogeneity(self):
        from synthetic.generate_domain import generate_domain, DomainConfig
        domain = generate_domain(DomainConfig.S3())
        unique = np.unique(domain.soil_map)
        self.assertEqual(len(unique), 5)

    def test_s4_low_density_wells(self):
        from synthetic.generate_domain import generate_domain, DomainConfig
        domain = generate_domain(DomainConfig.S4())
        self.assertEqual(domain.n_wells, 4)

    def test_sy_range_physical(self):
        from synthetic.generate_domain import generate_domain, DomainConfig
        domain = generate_domain(DomainConfig.S3())
        self.assertGreater(domain.Sy_map.min(), 0.0)
        self.assertLess(domain.Sy_map.max(), 0.5)

    def test_reproducibility(self):
        from synthetic.generate_domain import generate_domain, DomainConfig
        d1 = generate_domain(DomainConfig.S3())
        d2 = generate_domain(DomainConfig.S3())
        np.testing.assert_array_equal(d1.soil_map, d2.soil_map)

    def test_all_scenarios_create(self):
        from synthetic.generate_domain import generate_domain, DomainConfig
        for factory in [DomainConfig.S1, DomainConfig.S2, DomainConfig.S3,
                        DomainConfig.S4, DomainConfig.S5]:
            domain = generate_domain(factory())
            self.assertEqual(domain.soil_map.shape, (100, 100))


class TestGenerateData(unittest.TestCase):
    """generate_data.py 테스트."""

    @classmethod
    def setUpClass(cls):
        from synthetic.generate_domain import generate_domain, DomainConfig
        from synthetic.generate_data import generate_data
        cls.domain = generate_domain(DomainConfig.S3())
        cls.data = generate_data(cls.domain)

    def test_precipitation_shape(self):
        self.assertEqual(len(self.data.P), 730)

    def test_precipitation_nonnegative(self):
        self.assertTrue(np.all(self.data.P >= 0))

    def test_et_shape(self):
        self.assertEqual(len(self.data.ET), 730)

    def test_et_positive(self):
        self.assertTrue(np.all(self.data.ET > 0))

    def test_true_recharge_shape(self):
        self.assertEqual(self.data.true_recharge_annual.shape, (100, 100))

    def test_true_recharge_nonnegative(self):
        self.assertTrue(np.all(self.data.true_recharge_annual >= 0))

    def test_ho_obs_shape(self):
        self.assertEqual(self.data.ho_obs.shape[0], 25)  # n_wells
        self.assertEqual(self.data.ho_obs.shape[1], 730)

    def test_ho_obs_has_noise(self):
        # ho_obs != ho_true (노이즈 추가됨)
        diff = np.abs(self.data.ho_obs - self.data.ho_true)
        self.assertGreater(diff.max(), 0)

    def test_well_soil_types(self):
        self.assertEqual(len(self.data.well_soil_types), 25)
        for st in self.data.well_soil_types:
            self.assertIn(int(st), [1, 3, 6, 9, 12])

    def test_reproducibility(self):
        from synthetic.generate_data import generate_data
        data2 = generate_data(self.domain)
        np.testing.assert_array_equal(self.data.P, data2.P)


class TestMethods(unittest.TestCase):
    """3가지 방법론 테스트."""

    @classmethod
    def setUpClass(cls):
        from synthetic.generate_domain import generate_domain, DomainConfig
        from synthetic.generate_data import generate_data
        cls.domain = generate_domain(DomainConfig.S3())
        cls.data = generate_data(cls.domain)
        cls.obs = {
            'P': cls.data.P,
            'ET': cls.data.ET,
            'ho_obs': cls.data.ho_obs,
            'well_soil_types': cls.data.well_soil_types,
        }

    def test_lumped_shape_and_uniform(self):
        from methods.wtf_lumped import estimate_recharge
        r = estimate_recharge(self.domain, self.obs)
        self.assertEqual(r.shape, (100, 100))
        # Lumped: 모든 셀 동일
        self.assertAlmostEqual(r.min(), r.max(), places=5)

    def test_lumped_positive(self):
        from methods.wtf_lumped import estimate_recharge
        r = estimate_recharge(self.domain, self.obs)
        self.assertGreater(r.mean(), 0)

    def test_soil_weighted_shape(self):
        from methods.wtf_soil_weighted import estimate_recharge
        r = estimate_recharge(self.domain, self.obs)
        self.assertEqual(r.shape, (100, 100))

    def test_soil_weighted_spatial_variation(self):
        from methods.wtf_soil_weighted import estimate_recharge
        r = estimate_recharge(self.domain, self.obs)
        # 공간 변이 있어야 함
        self.assertGreater(r.std(), 0)
        self.assertNotAlmostEqual(r.min(), r.max(), places=1)

    def test_enkf_shape(self):
        from methods.wtf_enkf_spatial import estimate_recharge
        r = estimate_recharge(self.domain, self.obs)
        self.assertEqual(r.shape, (100, 100))

    def test_enkf_nonnegative(self):
        from methods.wtf_enkf_spatial import estimate_recharge
        r = estimate_recharge(self.domain, self.obs)
        self.assertTrue(np.all(r >= 0))

    def test_enkf_spatial_variation(self):
        from methods.wtf_enkf_spatial import estimate_recharge
        r = estimate_recharge(self.domain, self.obs)
        self.assertGreater(r.std(), 0)


class TestMetrics(unittest.TestCase):
    """evaluation/metrics.py 테스트."""

    def test_perfect_match(self):
        from evaluation.metrics import compute_metrics
        true = np.random.default_rng(0).uniform(50, 200, (10, 10))
        m = compute_metrics(true, true, "test", "S0")
        self.assertAlmostEqual(m.rmse, 0.0, places=5)
        self.assertAlmostEqual(m.mae, 0.0, places=5)
        self.assertAlmostEqual(m.bias, 0.0, places=5)
        self.assertAlmostEqual(m.r_spatial, 1.0, places=5)

    def test_uniform_vs_varied(self):
        from evaluation.metrics import compute_metrics
        true = np.random.default_rng(1).uniform(50, 200, (10, 10))
        uniform = np.full((10, 10), true.mean())
        m = compute_metrics(uniform, true, "uniform", "S0")
        # 균일 추정은 공간상관 0 (또는 NaN)
        self.assertLess(abs(m.r_spatial), 0.01)

    def test_compare_methods_sorted(self):
        from evaluation.metrics import compare_methods
        true = np.random.default_rng(2).uniform(50, 200, (10, 10))
        good = true + np.random.default_rng(3).normal(0, 5, (10, 10))
        bad = np.full((10, 10), 100.0)
        results = compare_methods({'good': good, 'bad': bad}, true)
        # RMSE 순 정렬: good이 먼저
        self.assertEqual(results[0].method_name, 'good')


class TestScenarios(unittest.TestCase):
    """scenarios.py 테스트."""

    def test_run_single_scenario(self):
        from synthetic.scenarios import run_scenario
        result = run_scenario("S3")
        self.assertEqual(result.name, "S3")
        self.assertIsNotNone(result.domain)
        self.assertIsNotNone(result.data)

    def test_run_all_scenarios(self):
        from synthetic.scenarios import run_all_scenarios
        results = run_all_scenarios(["S1", "S3"])
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].name, "S1")
        self.assertEqual(results[1].name, "S3")


if __name__ == "__main__":
    unittest.main()
