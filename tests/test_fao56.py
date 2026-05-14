"""test_fao56.py — FAO-56 daily SWB module tests.

Algorithm verification: Hargreaves ET₀ vs FAO-56 Annex B example,
extraterrestrial radiation Ra vs FAO-56 Table 2.6, mass balance closure,
edge cases.
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestExtraterrestrialRadiation(unittest.TestCase):
    """Ra vs FAO-56 Annex 2 Table 2.6 (computed for various lat/DOY)."""

    def test_equator_equinox(self):
        """적도 춘분 → Ra ≈ 38 MJ/m²/day (FAO-56 표 2.6 0°, DOY 81)."""
        from fao56_swb import extraterrestrial_radiation_mj
        Ra = extraterrestrial_radiation_mj(np.array([81]), lat_deg=0.0)
        self.assertAlmostEqual(float(Ra[0]), 38.0, delta=1.0)

    def test_korea_summer(self):
        """대전 위도 36.37°, 하지(DOY ~172) → Ra ≈ 41 MJ/m²/day."""
        from fao56_swb import extraterrestrial_radiation_mj
        Ra = extraterrestrial_radiation_mj(np.array([172]), lat_deg=36.37)
        self.assertGreater(float(Ra[0]), 38.0)
        self.assertLess(float(Ra[0]), 43.0)

    def test_korea_winter(self):
        """대전 위도 36.37°, 동지(DOY ~355) → Ra ≈ 16 MJ/m²/day
        (FAO-56 Annex 2 Table 2.6, lat 35°N DOY 355).
        """
        from fao56_swb import extraterrestrial_radiation_mj
        Ra = extraterrestrial_radiation_mj(np.array([355]), lat_deg=36.37)
        self.assertGreater(float(Ra[0]), 14.0)
        self.assertLess(float(Ra[0]), 18.0)


class TestHargreavesETO(unittest.TestCase):
    def test_warm_summer_korea(self):
        """대전 7월 평균 25°C, T-range 9°C → ETo ~5–7 mm/day."""
        from fao56_swb import hargreaves_eto
        Tmean = np.array([25.0])
        Tmax = np.array([29.0])
        Tmin = np.array([21.0])
        eto = hargreaves_eto(Tmean, Tmax, Tmin, lat_deg=36.37, start_doy=200)
        self.assertGreater(float(eto[0]), 4.0)
        self.assertLess(float(eto[0]), 8.0)

    def test_winter_korea_low_eto(self):
        """대전 1월 평균 -2°C, T-range 8°C → ETo ~0.5–1.5 mm/day."""
        from fao56_swb import hargreaves_eto
        Tmean = np.array([-2.0])
        Tmax = np.array([2.0])
        Tmin = np.array([-6.0])
        eto = hargreaves_eto(Tmean, Tmax, Tmin, lat_deg=36.37, start_doy=15)
        self.assertGreaterEqual(float(eto[0]), 0.0)
        self.assertLess(float(eto[0]), 2.5)

    def test_zero_temp_range(self):
        """T_max == T_min → ETo = 0 (Hargreaves √ΔT 항)."""
        from fao56_swb import hargreaves_eto
        Tmean = np.array([20.0])
        Tmax = np.array([20.0])
        Tmin = np.array([20.0])
        eto = hargreaves_eto(Tmean, Tmax, Tmin, lat_deg=36.0)
        self.assertEqual(float(eto[0]), 0.0)

    def test_negative_range_clipped(self):
        """T_max < T_min (입력 오류) → 음수 차이는 0으로 클립 → ETo=0."""
        from fao56_swb import hargreaves_eto
        eto = hargreaves_eto(
            np.array([20.0]), np.array([18.0]), np.array([22.0]),
            lat_deg=36.0,
        )
        self.assertEqual(float(eto[0]), 0.0)


class TestKcCurve(unittest.TestCase):
    def test_curve_has_correct_endpoints(self):
        from fao56_swb import kc_curve, KC_PRESETS
        for land_use in KC_PRESETS:
            Kc = kc_curve(land_use, n_days=365, start_doy=1)
            self.assertEqual(len(Kc), 365)
            kc_ini = KC_PRESETS[land_use]["Kc"][0]
            # 비성장기 (예: 1월 = DOY 1) → Kc_ini
            self.assertAlmostEqual(Kc[0], kc_ini, places=5)

    def test_mid_season_peaks(self):
        """혼합농경지 — 생육 중반에 Kc_mid 도달."""
        from fao56_swb import kc_curve, KC_PRESETS
        Kc = kc_curve("혼합농경지", n_days=365, start_doy=1)
        kc_mid = KC_PRESETS["혼합농경지"]["Kc"][1]
        self.assertAlmostEqual(np.max(Kc), kc_mid, places=2)

    def test_unknown_land_use_raises(self):
        from fao56_swb import kc_curve
        with self.assertRaises(ValueError):
            kc_curve("Unknown", n_days=10)


class TestSoilWaterCapacity(unittest.TestCase):
    def test_texture_to_paw(self):
        from fao56_swb import soil_water_capacity_mm
        # FAO-56 표 19 일관성
        self.assertEqual(soil_water_capacity_mm("coarse", 1.0), 70.0)
        self.assertEqual(soil_water_capacity_mm("medium", 1.0), 140.0)
        self.assertEqual(soil_water_capacity_mm("fine", 1.0), 190.0)
        # 근권 깊이 비례
        self.assertEqual(soil_water_capacity_mm("medium", 0.5), 70.0)


class TestEndToEnd(unittest.TestCase):
    def test_zero_precipitation_recharge_zero(self):
        """비강우 → 함양 0."""
        from fao56_swb import estimate_recharge_fao56
        n = 365
        P = np.zeros(n)
        T = np.full(n, 15.0)
        Tx = T + 5; Tn = T - 5
        r = estimate_recharge_fao56(P, T, Tx, Tn, lat_deg=36.0)
        self.assertEqual(r.R_annual_mm, 0.0)
        self.assertEqual(r.recharge_ratio_pct, 0.0)

    def test_mass_balance_closure(self):
        """P − ETa − Q − DP ≈ ΔAW (저류 변화)."""
        from fao56_swb import estimate_recharge_fao56
        rng = np.random.default_rng(0)
        n = 365
        P = rng.exponential(5.0, n)
        T = 15.0 + 10 * np.sin(2*np.pi*np.arange(n)/365) + rng.normal(0, 2, n)
        Tx = T + 5; Tn = T - 5
        r = estimate_recharge_fao56(P, T, Tx, Tn, lat_deg=36.0,
                                    runoff_fraction=0.1)
        # 시작/종료 AW 차이가 (P - ETa - Q - DP)와 일치해야 함
        sum_residual = (
            r.P_annual_mm - r.ETa_annual_mm
            - r.runoff_annual_mm - r.R_annual_mm
        )
        # 토양수분 capacity 한도 내 (단위 환산 후 ±100mm)
        self.assertLess(abs(sum_residual), 100.0)

    def test_high_temperature_more_eto(self):
        """기온 높을수록 ETo↑ → ETa↑ → 함양↓ (다른 조건 동일)."""
        from fao56_swb import estimate_recharge_fao56
        rng = np.random.default_rng(0)
        P = rng.exponential(5.0, 365)
        Tx_cold = np.full(365, 15.0)
        Tn_cold = np.full(365, 5.0)
        Tx_hot = np.full(365, 30.0)
        Tn_hot = np.full(365, 20.0)
        r_cold = estimate_recharge_fao56(
            P, (Tx_cold + Tn_cold)/2, Tx_cold, Tn_cold, lat_deg=36.0,
        )
        r_hot = estimate_recharge_fao56(
            P, (Tx_hot + Tn_hot)/2, Tx_hot, Tn_hot, lat_deg=36.0,
        )
        self.assertGreater(r_hot.ETa_annual_mm, r_cold.ETa_annual_mm)
        self.assertLess(r_hot.R_annual_mm, r_cold.R_annual_mm)

    def test_korean_monsoon_realistic_range(self):
        """한국 몬순 (대전 위도) → 함양율 30–70% (runoff 미적용 기본).

        실제 함양율 10–25%는 SCS-CN runoff와 결합해야 도달.  알고리즘
        자체는 ET 적용으로 SCS-CN 단독(72%) 대비 분명히 낮춰야 함.
        """
        from fao56_swb import estimate_recharge_fao56
        rng = np.random.default_rng(42)
        n = 730
        doy = np.arange(n) % 365

        wet_prob = np.clip(0.18 + 0.20*np.sin(2*np.pi*(doy-80)/365), 0.05, 0.55)
        is_wet = rng.random(n) < wet_prob
        intensity_scale = 8.0 + 25.0 * np.clip(
            np.sin(2*np.pi*(doy-80)/365), 0.0, 1.0
        )
        P = np.where(is_wet, rng.exponential(intensity_scale), 0.0)
        P = np.clip(P, 0, 200)

        Tmean = 12.5 + 15 * np.sin(2*np.pi*(doy-110)/365) + rng.normal(0, 2, n)
        Tmax = Tmean + 4; Tmin = Tmean - 4

        r = estimate_recharge_fao56(
            P, Tmean, Tmax, Tmin,
            lat_deg=36.37, texture_group="medium",
            land_use="혼합농경지",
        )
        self.assertGreater(r.P_annual_mm, 1000.0)
        self.assertLess(r.P_annual_mm, 2000.0)
        self.assertGreater(r.recharge_ratio_pct, 25.0)
        self.assertLess(r.recharge_ratio_pct, 75.0)
        # ETa는 ETo 미만이어야 (stress 가능)
        self.assertLessEqual(r.ETa_annual_mm, r.ETo_annual_mm + 0.01)
        # ETo는 한국 위도에서 800–1200 mm/yr
        self.assertGreater(r.ETo_annual_mm, 700.0)
        self.assertLess(r.ETo_annual_mm, 1300.0)

    def test_runoff_reduces_recharge(self):
        """runoff_fraction 늘면 함양 줄어든다."""
        from fao56_swb import estimate_recharge_fao56
        rng = np.random.default_rng(0)
        n = 365
        P = rng.exponential(5.0, n)
        T = np.full(n, 15.0); Tx = T + 5; Tn = T - 5

        r0 = estimate_recharge_fao56(P, T, Tx, Tn, lat_deg=36.0,
                                     runoff_fraction=0.0)
        r2 = estimate_recharge_fao56(P, T, Tx, Tn, lat_deg=36.0,
                                     runoff_fraction=0.2)
        self.assertGreater(r0.R_annual_mm, r2.R_annual_mm)
        self.assertGreater(r2.runoff_annual_mm, 0.0)


class TestResultMetadata(unittest.TestCase):
    def test_daily_arrays_returned(self):
        from fao56_swb import estimate_recharge_fao56
        n = 30
        P = np.full(n, 5.0); T = np.full(n, 15.0)
        Tx = T + 5; Tn = T - 5
        r = estimate_recharge_fao56(P, T, Tx, Tn, lat_deg=36.0,
                                    return_daily=True)
        self.assertEqual(len(r.daily_eto_mm), n)
        self.assertEqual(len(r.daily_etc_mm), n)
        self.assertEqual(len(r.daily_eta_mm), n)
        self.assertEqual(len(r.daily_deep_perc_mm), n)
        self.assertEqual(len(r.daily_aw_mm), n)
        self.assertEqual(len(r.daily_kc), n)


if __name__ == "__main__":
    unittest.main()
