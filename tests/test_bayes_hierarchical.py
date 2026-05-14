"""test_bayes_hierarchical.py — Phase 3 hierarchical Bayesian 검증."""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _basic_obs():
    from bayes_hierarchical import WellObservation
    return [
        WellObservation("w1", "A", "alluvial", sy_eff_obs=0.22, cumulative_dh_m=2.0, P_total_m=1.0),
        WellObservation("w2", "A", "alluvial", sy_eff_obs=0.20, cumulative_dh_m=1.8, P_total_m=1.0),
        WellObservation("w3", "D", "alluvial", sy_eff_obs=0.08, cumulative_dh_m=1.5, P_total_m=1.0),
    ]


class TestHierarchicalBasic(unittest.TestCase):
    def test_emcee_available(self):
        from bayes_hierarchical import HAS_EMCEE
        self.assertTrue(HAS_EMCEE, "emcee 가 설치되어 있어야 합니다")

    def test_runs_without_error(self):
        from bayes_hierarchical import fit_hierarchical
        obs = _basic_obs()
        r = fit_hierarchical(obs, n_walkers=16, n_steps=500, burn_in=200, seed=0)
        self.assertEqual(len(r.well_names), 3)
        self.assertEqual(len(r.sy_well_mean), 3)

    def test_acceptance_in_target_range(self):
        from bayes_hierarchical import fit_hierarchical
        r = fit_hierarchical(_basic_obs(), n_walkers=24, n_steps=1000, burn_in=200, seed=0)
        self.assertGreater(r.mean_acceptance_rate, 0.10)
        self.assertLess(r.mean_acceptance_rate, 0.80)

    def test_posterior_summary_consistency(self):
        from bayes_hierarchical import fit_hierarchical
        r = fit_hierarchical(_basic_obs(), n_walkers=16, n_steps=600, burn_in=200, seed=0)
        for i in range(len(r.well_names)):
            self.assertLessEqual(r.sy_well_lo95[i], r.sy_well_mean[i])
            self.assertLessEqual(r.sy_well_mean[i], r.sy_well_hi95[i])
        self.assertLessEqual(r.mu_watershed_lo95, r.mu_watershed_mean)
        self.assertLessEqual(r.mu_watershed_mean, r.mu_watershed_hi95)


class TestHierarchicalPhysics(unittest.TestCase):
    def test_pump_test_narrows_ci(self):
        """양수시험 추가된 관정은 CI 가 더 좁아야."""
        from bayes_hierarchical import WellObservation, fit_hierarchical
        obs_no_pump = [
            WellObservation("w1", "A", "alluvial", sy_eff_obs=0.20),
            WellObservation("w2", "A", "alluvial", sy_eff_obs=0.20),
        ]
        obs_with_pump = [
            WellObservation("w1", "A", "alluvial", sy_eff_obs=0.20, pump_test_sy=0.18),
            WellObservation("w2", "A", "alluvial", sy_eff_obs=0.20),
        ]
        r0 = fit_hierarchical(obs_no_pump, n_walkers=24, n_steps=2000, burn_in=500, seed=0)
        r1 = fit_hierarchical(obs_with_pump, n_walkers=24, n_steps=2000, burn_in=500, seed=0)
        ci_no = r0.sy_well_hi95[0] - r0.sy_well_lo95[0]
        ci_w = r1.sy_well_hi95[0] - r1.sy_well_lo95[0]
        self.assertLess(ci_w, ci_no)

    def test_hsg_a_higher_than_hsg_d(self):
        """관측이 일관되면 HSG A 의 posterior 가 HSG D 보다 높아야."""
        from bayes_hierarchical import WellObservation, fit_hierarchical
        obs = [
            WellObservation("a1", "A", "alluvial", sy_eff_obs=0.25),
            WellObservation("a2", "A", "alluvial", sy_eff_obs=0.23),
            WellObservation("d1", "D", "alluvial", sy_eff_obs=0.07),
            WellObservation("d2", "D", "alluvial", sy_eff_obs=0.08),
        ]
        r = fit_hierarchical(obs, n_walkers=24, n_steps=1500, burn_in=500, seed=0)
        mu_A = r.mu_hsg_summary["A"][0]
        mu_D = r.mu_hsg_summary["D"][0]
        self.assertGreater(mu_A, mu_D)

    def test_recharge_posterior_finite_when_dh_p_provided(self):
        from bayes_hierarchical import fit_hierarchical
        r = fit_hierarchical(_basic_obs(), n_walkers=16, n_steps=600, burn_in=200, seed=0)
        self.assertTrue(np.isfinite(r.rech_pct_watershed_mean))
        self.assertGreater(r.rech_pct_watershed_hi95, r.rech_pct_watershed_lo95)


if __name__ == "__main__":
    unittest.main()
