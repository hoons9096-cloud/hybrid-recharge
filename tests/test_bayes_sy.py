"""test_bayes_sy.py — Phase 1 Bayesian Sy posterior 검증."""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPriorMapping(unittest.TestCase):
    def test_hsg_aquifer_returns_valid_sn(self):
        from bayes_sy import HSG_AQUIFER_TO_SN, get_prior_params
        for (hsg, aq), sn in HSG_AQUIFER_TO_SN.items():
            self.assertIn(sn, range(1, 13))
            mu, sd, sn2 = get_prior_params(hsg, aq)
            self.assertEqual(sn, sn2)
            self.assertGreater(mu, 0.0)
            self.assertLess(mu, 0.5)

    def test_unsupported_combination_raises(self):
        from bayes_sy import get_prior_params
        with self.assertRaises(ValueError):
            get_prior_params("X", "alluvial")

    def test_alluvial_A_higher_sy_than_bedrock_D(self):
        """충적 A 의 prior Sy > 암반 D 의 prior Sy 여야."""
        from bayes_sy import get_prior_params
        mu_A, _, _ = get_prior_params("A", "alluvial")
        mu_D, _, _ = get_prior_params("D", "bedrock")
        self.assertGreater(mu_A, mu_D)


class TestPosterior(unittest.TestCase):
    def test_no_likelihood_recovers_prior(self):
        """관측 없이 sampling 만 하면 posterior ≈ prior."""
        from bayes_sy import posterior_sy
        r = posterior_sy(hsg="A", aquifer="alluvial", n_samples=20000)
        # truncnorm 평균은 prior μ 와 약간 다를 수 있음 — 0.05 안에서 일치
        self.assertAlmostEqual(r.sy_post_mean, r.sy_prior_mean, delta=0.05)

    def test_strong_obs_pulls_posterior_to_obs(self):
        """관측이 prior 와 다르면 posterior 가 관측 쪽으로 이동."""
        from bayes_sy import posterior_sy
        # HSG D bedrock prior μ ≈ 0.06 (Clay sy_lit)
        # Sy_eff_obs = 0.20 (반대 방향) → posterior 가 prior 와 obs 사이
        r = posterior_sy(
            hsg="D", aquifer="bedrock",
            sy_eff_obs=0.20,
            n_samples=20000,
        )
        self.assertGreater(r.sy_post_mean, r.sy_prior_mean)
        self.assertLess(r.sy_post_mean, 0.20)

    def test_pump_test_dominates(self):
        """양수시험은 strong likelihood — posterior 가 양수시험 값에 강하게 수렴."""
        from bayes_sy import posterior_sy
        r = posterior_sy(
            hsg="A", aquifer="alluvial",
            sy_eff_obs=0.30,           # weak signal
            pump_test_sy=0.15,         # strong signal
            n_samples=20000,
        )
        # posterior 는 양수시험 (0.15) 근처
        self.assertAlmostEqual(r.sy_post_mean, 0.15, delta=0.02)

    def test_posterior_sd_smaller_than_prior_with_data(self):
        """데이터 추가 시 posterior 분산 < prior 분산."""
        from bayes_sy import posterior_sy
        r = posterior_sy(
            hsg="A", aquifer="alluvial",
            sy_eff_obs=0.18,
            n_samples=20000,
        )
        self.assertLess(r.sy_post_sd, r.sy_prior_sd)

    def test_ess_reasonable(self):
        from bayes_sy import posterior_sy
        r = posterior_sy(hsg="B", aquifer="bedrock",
                         sy_eff_obs=0.12, n_samples=10000)
        # 양호한 sample → ESS > 100
        self.assertGreater(r.n_eff, 100)
        self.assertTrue(r.converged)

    def test_recharge_posterior_finite(self):
        """cumulative_dh + P_total 주면 rech_pct 분포 산출."""
        from bayes_sy import posterior_sy
        r = posterior_sy(
            hsg="A", aquifer="alluvial",
            sy_eff_obs=0.20,
            cumulative_dh_m=1.0, P_total_m=1.0,  # 1m / 1m
            n_samples=5000,
        )
        self.assertTrue(np.isfinite(r.rech_pct_post_mean))
        self.assertGreater(r.rech_pct_post_hi95, r.rech_pct_post_lo95)


class TestWeightedQuantile(unittest.TestCase):
    def test_uniform_weights_match_numpy(self):
        from bayes_sy import _weighted_quantiles
        rng = np.random.default_rng(0)
        v = rng.normal(size=10000)
        w = np.ones_like(v) / len(v)
        lo, hi = _weighted_quantiles(v, w, [0.025, 0.975])
        np_lo = np.quantile(v, 0.025)
        np_hi = np.quantile(v, 0.975)
        self.assertAlmostEqual(lo, np_lo, delta=0.05)
        self.assertAlmostEqual(hi, np_hi, delta=0.05)


if __name__ == "__main__":
    unittest.main()
