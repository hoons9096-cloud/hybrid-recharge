import unittest
from importlib.util import find_spec

import numpy as np
import pandas as pd

SCIPY_AVAILABLE = find_spec("scipy") is not None

if SCIPY_AVAILABLE:
    from pump_preprocess.preprocess.corrector import WaterLevelCorrector
    from pump_preprocess.preprocess.detector import PumpingDetector


@unittest.skipUnless(SCIPY_AVAILABLE, "scipy is required for preprocessing tests")
class PumpingDetectorTests(unittest.TestCase):
    def test_sigma_detector_flags_obvious_dry_drop(self):
        dates = pd.date_range("2024-01-01", periods=12, freq="D")
        rainfall = np.zeros(12)
        water_level = np.array(
            [10.0, 10.0, 9.99, 9.98, 9.2, 8.7, 8.2, 8.1, 8.1, 8.1, 8.1, 8.1],
            dtype=float,
        )

        detector = PumpingDetector(methods=["sigma"], sigma_drop=2.0, sigma_run=0.8)
        result = detector.detect(dates, water_level, rainfall)

        self.assertGreater(result.n_pump_days, 0)
        self.assertTrue(result.pump_mask[4:8].any())
        self.assertEqual(result.confidence.shape[0], water_level.shape[0])


@unittest.skipUnless(SCIPY_AVAILABLE, "scipy is required for preprocessing tests")
class WaterLevelCorrectorTests(unittest.TestCase):
    def test_corrector_returns_input_when_no_pumping(self):
        dates = pd.date_range("2024-01-01", periods=6, freq="D")
        wl = np.array([10.0, 9.9, 9.8, 9.7, 9.6, 9.5], dtype=float)
        mask = np.zeros(6, dtype=bool)

        result = WaterLevelCorrector().correct(dates, wl, mask, np.zeros(6))

        self.assertEqual(result.strategy_used, "none")
        np.testing.assert_allclose(result.corrected_wl, wl)
        self.assertFalse(result.filled_mask.any())

    def test_corrector_fills_pumping_segment(self):
        dates = pd.date_range("2024-01-01", periods=8, freq="D")
        wl = np.array([10.0, 9.95, 9.9, 8.7, 8.4, 9.7, 9.65, 9.6], dtype=float)
        mask = np.array([False, False, False, True, True, False, False, False])

        result = WaterLevelCorrector(strategy="spline_fill").correct(
            dates, wl, mask, np.zeros(8)
        )

        self.assertTrue(result.filled_mask[3:5].all())
        self.assertTrue(np.isfinite(result.corrected_wl[3:5]).all())
        self.assertNotEqual(result.strategy_used, "none")


if __name__ == "__main__":
    unittest.main()
