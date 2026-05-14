import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

SCIPY_AVAILABLE = find_spec("scipy") is not None

if SCIPY_AVAILABLE:
    from core_sim_v27 import core_sim_v27


@unittest.skipUnless(SCIPY_AVAILABLE, "scipy is required for core simulation smoke tests")
class CoreSimSmokeTests(unittest.TestCase):
    def write_input(self, text: str) -> str:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8")
        tmp.write(text)
        tmp.flush()
        tmp.close()
        self.addCleanup(lambda: Path(tmp.name).unlink(missing_ok=True))
        return tmp.name

    def test_demo_mode_returns_metrics(self):
        result = core_sim_v27(
            "DEMO",
            -0.05,
            3.0,
            0.0,
            3.0,
            1e-4,
            1e-3,
            0.001,
            1.0,
            5.0,
            0.0,
        )

        self.assertIn("rmse", result)
        self.assertIn("recharge_ratio", result)
        self.assertIn("pump_mask", result)

    def test_file_input_uses_shared_loader_path(self):
        path = self.write_input(
            "2024-01-01,10.0,5\n"
            "2024-01-02,10.2,0\n"
            "2024-01-03,10.3,1\n"
            "2024-01-04,10.1,0\n"
            "2024-01-05,10.0,0\n"
            "2024-01-06,10.2,2\n"
        )

        result = core_sim_v27(
            path,
            -0.05,
            3.0,
            0.0,
            3.0,
            1e-4,
            1e-3,
            0.001,
            1.0,
            5.0,
            0.0,
        )

        self.assertNotIn("error", result)
        self.assertIn("rmse", result)
        self.assertTrue(len(result["ho"]) >= 6)


if __name__ == "__main__":
    unittest.main()
