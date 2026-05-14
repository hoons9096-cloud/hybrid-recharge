"""Tests for soil_db.py — single source of truth for soil properties."""

import unittest
import numpy as np


class TestSoilDB(unittest.TestCase):
    def test_all_12_soils_present(self):
        from soil_db import SOIL_DB
        self.assertEqual(len(SOIL_DB), 12)
        for i in range(1, 13):
            self.assertIn(i, SOIL_DB)

    def test_get_soil_clamping(self):
        from soil_db import get_soil
        s = get_soil(0)  # below range → clamp to 1
        self.assertEqual(s.index, 1)
        s = get_soil(99)  # above range → clamp to 12
        self.assertEqual(s.index, 12)

    def test_get_bounds_returns_tuple(self):
        from soil_db import get_bounds
        lo, hi = get_bounds(1)
        self.assertLess(lo, hi)
        self.assertLess(lo, 0)
        self.assertLess(hi, 0)

    def test_vg_db_shape(self):
        from soil_db import VG_DB
        self.assertEqual(VG_DB.shape, (12, 4))

    def test_clay_like_set(self):
        from soil_db import CLAY_LIKE_SET
        self.assertIn(6, CLAY_LIKE_SET)
        self.assertNotIn(1, CLAY_LIKE_SET)

    def test_gap_allow_coarse_lt_fine(self):
        from soil_db import gap_allow_for_soil
        self.assertLess(gap_allow_for_soil(1), gap_allow_for_soil(6))

    def test_peak_window_fast_lt_slow(self):
        from soil_db import peak_window_for_soil
        self.assertLess(peak_window_for_soil(1), peak_window_for_soil(6))

    def test_rech_range_positive(self):
        from soil_db import SOIL_DB
        for sn, soil in SOIL_DB.items():
            lo, hi = soil.rech_range
            self.assertGreater(lo, 0)
            self.assertGreater(hi, lo)

    def test_sy_lit_reasonable(self):
        from soil_db import SOIL_DB
        for sn, soil in SOIL_DB.items():
            self.assertGreater(soil.sy_lit, 0)
            self.assertLess(soil.sy_lit, soil.theta_s)


if __name__ == "__main__":
    unittest.main()
