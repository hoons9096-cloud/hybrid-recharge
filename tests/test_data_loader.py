import tempfile
import unittest
from pathlib import Path

from data_loader import load_timeseries_file


class DataLoaderTests(unittest.TestCase):
    def write_text(self, text: str) -> str:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8")
        tmp.write(text)
        tmp.flush()
        tmp.close()
        self.addCleanup(lambda: Path(tmp.name).unlink(missing_ok=True))
        return tmp.name

    def test_load_valid_csv(self):
        path = self.write_text(
            "2024-01-01,10.0,5\n"
            "2024-01-02,10.2,0\n"
            "2024-01-03,10.4,1\n"
        )

        data = load_timeseries_file(path, interpolate_water_level=True, rainfall_unit="mm")

        self.assertEqual(len(data.water_level), 3)
        self.assertAlmostEqual(float(data.rainfall_mm[0]), 5.0)

    def test_reject_negative_rainfall(self):
        path = self.write_text(
            "2024-01-01,10.0,5\n"
            "2024-01-02,10.2,-1\n"
            "2024-01-03,10.4,1\n"
        )

        with self.assertRaisesRegex(ValueError, "non-negative"):
            load_timeseries_file(path, interpolate_water_level=True)

    def test_reject_duplicate_dates(self):
        path = self.write_text(
            "2024-01-01,10.0,5\n"
            "2024-01-01,10.2,1\n"
            "2024-01-03,10.4,1\n"
        )

        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_timeseries_file(path, interpolate_water_level=True, require_dates=False)

    def test_interpolates_water_level_when_requested(self):
        path = self.write_text(
            "2024-01-01,10.0,5\n"
            "2024-01-02,,1\n"
            "2024-01-03,10.4,1\n"
            "2024-01-04,10.6,0\n"
        )

        data = load_timeseries_file(path, interpolate_water_level=True)
        self.assertAlmostEqual(float(data.water_level[1]), 10.2, places=6)


if __name__ == "__main__":
    unittest.main()
