import tempfile
import unittest
from pathlib import Path

from evaluation_runner import EvalCase, _has_baseline_failures, run_cases


class EvaluationRunnerTests(unittest.TestCase):
    def test_run_cases_returns_reproducible_report_shape(self):
        report = run_cases([EvalCase(label="demo", input_path="DEMO")])

        self.assertIn("generated_at_utc", report)
        self.assertIn("platform", report)
        self.assertIn("summary", report)
        self.assertEqual(len(report["results"]), 1)
        self.assertEqual(report["results"][0]["case"]["label"], "demo")
        self.assertIn(report["results"][0]["status"], {"ok", "error"})
        self.assertIn("baseline_check", report["results"][0])
        self.assertEqual(report["summary"]["total_cases"], 1)

    def test_run_cases_accepts_real_input_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as tmp:
            tmp.write(
                "2024-01-01,10.0,5\n"
                "2024-01-02,10.2,0\n"
                "2024-01-03,10.3,1\n"
                "2024-01-04,10.1,0\n"
                "2024-01-05,10.0,0\n"
                "2024-01-06,10.2,2\n"
            )
            tmp_path = Path(tmp.name)

        try:
            report = run_cases([EvalCase(label="tmp", input_path=str(tmp_path))])
            self.assertEqual(report["results"][0]["status"], "ok")
            self.assertIn("rmse", report["results"][0]["summary"])
            self.assertIn(report["results"][0]["baseline_check"]["status"], {"pass", "fail", "no_baseline"})
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_has_baseline_failures_detects_failures(self):
        report = {
            "summary": {
                "error_cases": 0,
                "baseline": {"pass": 0, "fail": 1, "no_baseline": 0},
            }
        }
        self.assertTrue(_has_baseline_failures(report))

    def test_has_baseline_failures_detects_clean_report(self):
        report = {
            "summary": {
                "error_cases": 0,
                "baseline": {"pass": 2, "fail": 0, "no_baseline": 0},
            }
        }
        self.assertFalse(_has_baseline_failures(report))

    def test_run_cases_applies_richer_metric_rules(self):
        baseline = {
            "demo": {
                "rmse": {"max": 1.0},
                "cc": {"min": 0.5},
                "recharge_ratio": {"min": 1.0, "max": 20.0},
                "pump_contam_idx": {"max": 0.1},
            }
        }
        report = run_cases([EvalCase(label="demo", input_path="DEMO")], baseline=baseline)

        self.assertEqual(report["results"][0]["baseline_check"]["status"], "pass")
        metrics = {item["metric"] for item in report["results"][0]["baseline_check"]["checks"]}
        self.assertIn("recharge_ratio", metrics)
        self.assertIn("pump_contam_idx", metrics)


if __name__ == "__main__":
    unittest.main()
