"""test_watershed.py — shp_soil_mapper + watershed_aggregator 통합 테스트.

토양도 .shp 가 없으면 skip 한다.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SHP_PATH = "/Users/choejeonghun/정밀토양도/전국_정밀토양도_GRS80.shp"


def _shp_available() -> bool:
    return os.path.exists(SHP_PATH)


@unittest.skipUnless(_shp_available(), "soil shapefile not present")
class TestSoilMapper(unittest.TestCase):
    def test_point_query_returns_valid_hsg(self):
        from shp_soil_mapper import query_point
        # 김천 일원 임의 좌표
        r = query_point("test", lat=36.05, lon=128.13)
        self.assertIn(r.hydro_type, ["A", "B", "C", "D"])
        self.assertGreater(r.Sy, 0.0)
        self.assertLess(r.Sy, 0.5)
        self.assertIn(r.texture_group, ["coarse", "medium", "fine"])

    def test_point_outside_korea_falls_to_nearest(self):
        from shp_soil_mapper import query_point
        # 동해 한가운데 — contains 실패 시 nearest fallback
        r = query_point("offshore", lat=37.0, lon=131.0)
        self.assertIn(r.hydro_type, ["A", "B", "C", "D"])

    def test_watershed_profile_fractions_sum_to_one(self):
        from shp_soil_mapper import watershed_profile_from_wells
        wells = [("a", 36.05, 128.13), ("b", 36.13, 128.11)]
        p = watershed_profile_from_wells("test_ws", wells, buffer_km=2.0)
        self.assertAlmostEqual(sum(p.hsg_fractions.values()), 1.0, places=6)
        self.assertGreater(p.total_area_km2, 0.0)


@unittest.skipUnless(_shp_available(), "soil shapefile not present")
class TestAggregator(unittest.TestCase):
    def test_gamcheon_watershed_runs(self):
        """감천 유역 — HSG A/D 혼합 → soil-weighted ≠ lumped 여야."""
        from watershed_aggregator import estimate_watershed
        cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        files = {
            "김천남면": os.path.join(cwd, "김천남면.txt"),
            "김천지좌": os.path.join(cwd, "김천지좌.txt"),
        }
        if not all(os.path.exists(f) for f in files.values()):
            self.skipTest("well .txt files not present")

        r = estimate_watershed("감천", file_paths=files, buffer_km=2.0)
        self.assertEqual(len(r.wells), 2)
        self.assertIsNotNone(r.lumped_wtf_pct)
        self.assertIsNotNone(r.soil_weighted_wtf_pct)
        # HSG 분포가 2종 이상이어야 (A 와 D)
        nonzero_hsg = [k for k, v in r.profile.hsg_fractions.items() if v > 0.05]
        self.assertGreaterEqual(len(nonzero_hsg), 2)


class TestHSGToParams(unittest.TestCase):
    """수문 매핑 일관성 — .shp 없어도 실행."""

    def test_hsg_sy_monotone(self):
        from shp_soil_mapper import HSG_TO_SY
        # 거친 토양 → 비산출률 高 (단조 감소)
        self.assertGreater(HSG_TO_SY["A"], HSG_TO_SY["B"])
        self.assertGreater(HSG_TO_SY["B"], HSG_TO_SY["C"])
        self.assertGreater(HSG_TO_SY["C"], HSG_TO_SY["D"])

    def test_hsg_cn_monotone(self):
        from shp_soil_mapper import HSG_TO_CN
        # CN: 거친 토양 → 침투 高 → CN 低 (단조 증가)
        self.assertLess(HSG_TO_CN["A"], HSG_TO_CN["B"])
        self.assertLess(HSG_TO_CN["B"], HSG_TO_CN["C"])
        self.assertLess(HSG_TO_CN["C"], HSG_TO_CN["D"])

    def test_hsg_to_sn_in_valid_range(self):
        from watershed_aggregator import HSG_TO_SN
        for hsg, sn in HSG_TO_SN.items():
            self.assertIn(sn, range(1, 13))


if __name__ == "__main__":
    unittest.main()
