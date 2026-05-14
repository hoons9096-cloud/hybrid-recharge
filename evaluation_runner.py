"""Reproducible evaluation runner for the v27 hybrid-recharge core."""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy

from core_sim_config import DEFAULT_Q_NOISE, DEFAULT_R_NOISE
from core_sim_v27 import core_sim_v27


@dataclass
class EvalCase:
    label: str
    input_path: str
    k_val: float = -0.05
    z_val: float = 3.0
    lag_val: float = 0.0
    sn_idx: float = 3.0
    q_val: float = DEFAULT_Q_NOISE
    r_val: float = DEFAULT_R_NOISE
    rc_val: float = 0.001
    ignore_pump: float = 1.0
    sens_val: float = 5.0
    do_optimize: float = 0.0


def _default_cases() -> list[EvalCase]:
    return [
        EvalCase(label="demo", input_path="DEMO"),
        EvalCase(label="sh11", input_path="SH11.txt"),
        EvalCase(label="sh22", input_path="SH22.txt"),
        EvalCase(label="sh28", input_path="SH28.txt"),
    ]


def _load_cases(case_file: str | None) -> list[EvalCase]:
    if case_file is None:
        return _default_cases()

    raw = json.loads(Path(case_file).read_text(encoding="utf-8"))
    return [EvalCase(**item) for item in raw]


def _load_baseline(baseline_file: str | None) -> dict:
    if baseline_file is None:
        baseline_path = Path(__file__).with_name("evaluation_baseline.json")
        if baseline_path.exists():
            return json.loads(baseline_path.read_text(encoding="utf-8"))
        return {}

    return json.loads(Path(baseline_file).read_text(encoding="utf-8"))


def _result_summary(result: dict) -> dict:
    summary_keys = [
        "rmse",
        "cc",
        "nse",
        "kge",
        "pbias",
        "recharge_ratio",
        "Sy_eff",
        "stress",
        "pump_contam_idx",
        "pump_event_count",
        "pump_max_run",
        "eval_n",
    ]
    return {key: result.get(key) for key in summary_keys}


def _json_safe_value(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _evaluate_against_baseline(case_label: str, summary: dict, baseline: dict) -> dict:
    rules = baseline.get(case_label)
    if not rules:
        return {"status": "no_baseline", "checks": []}

    checks = []
    overall = "pass"

    for metric, metric_rules in rules.items():
        value = summary.get(metric)

        if not isinstance(metric_rules, dict):
            continue

        if "min" in metric_rules:
            passed = bool(value is not None and value >= metric_rules["min"])
            checks.append(
                {
                    "metric": metric,
                    "rule": f">= {metric_rules['min']}",
                    "value": _json_safe_value(value),
                    "passed": passed,
                }
            )
            if not passed:
                overall = "fail"

        if "max" in metric_rules:
            passed = bool(value is not None and value <= metric_rules["max"])
            checks.append(
                {
                    "metric": metric,
                    "rule": f"<= {metric_rules['max']}",
                    "value": _json_safe_value(value),
                    "passed": passed,
                }
            )
            if not passed:
                overall = "fail"

    return {"status": overall, "checks": checks}


def _aggregate_results(results: list[dict]) -> dict:
    baseline_counts = {"pass": 0, "fail": 0, "no_baseline": 0}
    for item in results:
        baseline_status = item["baseline_check"]["status"]
        baseline_counts[baseline_status] = baseline_counts.get(baseline_status, 0) + 1

    return {
        "total_cases": len(results),
        "ok_cases": sum(1 for item in results if item["status"] == "ok"),
        "error_cases": sum(1 for item in results if item["status"] == "error"),
        "baseline": baseline_counts,
    }


def _has_baseline_failures(report: dict) -> bool:
    baseline_summary = report.get("summary", {}).get("baseline", {})
    return baseline_summary.get("fail", 0) > 0 or report.get("summary", {}).get("error_cases", 0) > 0


def run_cases(cases: Iterable[EvalCase], baseline: dict | None = None) -> dict:
    baseline = baseline or {}
    results = []
    for case in cases:
        payload = core_sim_v27(
            case.input_path,
            case.k_val,
            case.z_val,
            case.lag_val,
            case.sn_idx,
            case.q_val,
            case.r_val,
            case.rc_val,
            case.ignore_pump,
            case.sens_val,
            case.do_optimize,
        )
        summary = _result_summary(payload)
        results.append(
            {
                "case": asdict(case),
                "status": "error" if "error" in payload else "ok",
                "summary": summary,
                "baseline_check": _evaluate_against_baseline(case.label, summary, baseline),
                "error": payload.get("error"),
            }
        )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "system": platform.platform(),
        },
        "summary": _aggregate_results(results),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run reproducible v27 evaluation cases.")
    parser.add_argument(
        "--cases",
        help="Optional JSON file containing a list of evaluation cases.",
    )
    parser.add_argument(
        "--output",
        default="evaluation_report.json",
        help="Path to write the JSON evaluation report.",
    )
    parser.add_argument(
        "--baseline",
        help="Optional JSON file containing baseline thresholds.",
    )
    parser.add_argument(
        "--fail-on-baseline",
        action="store_true",
        help="Exit with code 1 when any baseline check fails or a case errors.",
    )
    args = parser.parse_args()

    report = run_cases(_load_cases(args.cases), baseline=_load_baseline(args.baseline))
    output_path = Path(args.output)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote evaluation report to {output_path}")
    if args.fail_on_baseline and _has_baseline_failures(report):
        print("Baseline gate failed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
