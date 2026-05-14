"""test_vadose_cascade.py — Phase 2 multi-layer cascade vadose model 검증."""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestLayerBuild(unittest.TestCase):
    def test_root_fractions_sum_to_one(self):
        from synthetic.vadose_cascade import build_layers_from_sn
        for n in [1, 5, 10]:
            layers = build_layers_from_sn(12, n_layers=n, root_decay=1.5)
            total = sum(L.root_frac for L in layers)
            self.assertAlmostEqual(total, 1.0, places=6)

    def test_uniform_root_when_decay_zero(self):
        from synthetic.vadose_cascade import build_layers_from_sn
        layers = build_layers_from_sn(12, n_layers=5, root_decay=0.0)
        rfs = [L.root_frac for L in layers]
        for rf in rfs:
            self.assertAlmostEqual(rf, 1/5, places=6)

    def test_field_capacity_above_wilting(self):
        from synthetic.vadose_cascade import build_layers_from_sn
        for sn in [1, 2, 6, 12]:
            layers = build_layers_from_sn(sn, n_layers=3)
            for L in layers:
                self.assertGreater(L.theta_fc, L.theta_wp)
                self.assertGreater(L.theta_s, L.theta_fc)
                self.assertGreaterEqual(L.theta_wp, L.theta_r)


class TestMassConservation(unittest.TestCase):
    """캐스케이드가 mass-conservative 인지 — 핵심 합성정답 요건."""

    def _run_default(self, sn=12, days=365):
        from synthetic.vadose_cascade import build_layers_from_sn, simulate_cascade
        rng = np.random.default_rng(0)
        P = rng.exponential(5.0, days)
        ET = np.full(days, 2.0)
        layers = build_layers_from_sn(sn, L_total_m=2.0, n_layers=5)
        r = simulate_cascade(P, ET, layers)
        return P, ET, r

    def test_per_day_mass_balance_zero(self):
        _, _, r = self._run_default()
        self.assertLess(np.max(np.abs(r.mass_balance_err_mm)), 1e-6)

    def test_recharge_plus_et_plus_runoff_minus_dstor_equals_p(self):
        P, ET, r = self._run_default(sn=12, days=200)
        S_start = r.storage_mm[0].sum()
        S_end = r.storage_mm[-1].sum()
        residual = (
            P.sum() - r.recharge.sum() - r.ET_actual.sum()
            - r.runoff.sum() - (S_end - S_start)
        )
        self.assertLess(abs(residual), 1e-3)


class TestPhysicalReasonability(unittest.TestCase):
    def test_sandy_more_recharge_than_clay(self):
        """모래(sn=1) 보다 점토(sn=6) recharge 가 적어야 (당연)."""
        from synthetic.vadose_cascade import build_layers_from_sn, simulate_cascade
        rng = np.random.default_rng(0)
        P = rng.exponential(5.0, 365)
        ET = np.full(365, 1.5)
        r_sand = simulate_cascade(P, ET, build_layers_from_sn(1))
        r_clay = simulate_cascade(P, ET, build_layers_from_sn(6))
        self.assertGreater(r_sand.annual_recharge_mm, r_clay.annual_recharge_mm)

    def test_zero_precipitation_zero_recharge(self):
        from synthetic.vadose_cascade import build_layers_from_sn, simulate_cascade
        n = 60
        P = np.zeros(n); ET = np.full(n, 2.0)
        r = simulate_cascade(P, ET, build_layers_from_sn(12))
        self.assertEqual(r.recharge.sum(), 0.0)

    def test_high_et_reduces_recharge(self):
        from synthetic.vadose_cascade import build_layers_from_sn, simulate_cascade
        rng = np.random.default_rng(0)
        P = rng.exponential(5.0, 365)
        layers = build_layers_from_sn(12)
        r_low = simulate_cascade(P, np.full(365, 0.5), layers)
        r_high = simulate_cascade(P, np.full(365, 5.0), layers)
        self.assertGreater(r_low.annual_recharge_mm, r_high.annual_recharge_mm)


class TestETBehavior(unittest.TestCase):
    def test_actual_et_below_potential(self):
        from synthetic.vadose_cascade import build_layers_from_sn, simulate_cascade
        n = 200
        P = np.zeros(n); ET_pot = np.full(n, 5.0)
        # 강수 없이 시작 → 토양 마름 → actual ET < potential
        r = simulate_cascade(P, ET_pot, build_layers_from_sn(12),
                             init_theta_frac=0.5)
        # 모든 일 actual ET ≤ potential
        self.assertTrue(np.all(r.ET_actual <= ET_pot + 1e-9))
        # 누적 actual < 누적 potential (마른 토양)
        self.assertLess(r.ET_actual.sum(), ET_pot.sum())


if __name__ == "__main__":
    unittest.main()
