"""test_well_report.py — 단일 관측정 리포트 figure & HTML 빌더 테스트.

실제 streamlit 실행 없이 모듈 단위로 검증한다.  세션 데이터는 DEMO 모드의
core_sim_v27 결과를 사용한다.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np
import matplotlib
matplotlib.use("Agg")

# 프로젝트 루트
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _get_demo_result_v27():
    """DEMO 모드에서 core_sim_v27 결과 dict 반환."""
    from core_sim_v27 import core_sim_v27
    result = core_sim_v27(
        file_path="DEMO", k_val=-0.015, z_val=3.0, lag_val=0,
        sn_idx=6, q_val=0.005, r_val=0.10, rc_val=0.005,
        ignore_pump=0.0, sens_val=1.0, do_optimize=True,
    )
    if hasattr(result, "to_dict"):
        result = result.to_dict()
    return result


class TestPlausibility(unittest.TestCase):
    def test_plausibility_runs_on_demo(self):
        from evaluation.well_report import well_plausibility_check
        r = _get_demo_result_v27()
        plaus = well_plausibility_check(r)
        self.assertGreaterEqual(plaus.recharge_ratio_pct, 0)
        self.assertGreater(plaus.n_obs_days, 0)
        self.assertGreater(plaus.P_annual_mm, 0)

    def test_negative_recharge_flagged(self):
        from evaluation.well_report import well_plausibility_check
        r = _get_demo_result_v27()
        r["recharge_ratio"] = -5.0
        plaus = well_plausibility_check(r)
        self.assertFalse(plaus.pass_basic)
        self.assertTrue(any("음수" in f for f in plaus.flags))

    def test_high_ratio_flagged(self):
        from evaluation.well_report import well_plausibility_check
        r = _get_demo_result_v27()
        r["recharge_ratio"] = 80.0  # > 50%
        plaus = well_plausibility_check(r)
        self.assertTrue(any("이례적" in f or "높음" in f for f in plaus.flags))


class TestFigures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = _get_demo_result_v27()

    def test_time_series(self):
        from evaluation.well_report import plot_well_time_series
        import matplotlib.figure
        fig = plot_well_time_series(self.result)
        self.assertIsInstance(fig, matplotlib.figure.Figure)

    def test_cumulative(self):
        from evaluation.well_report import plot_well_recharge_cumulative
        import matplotlib.figure
        fig = plot_well_recharge_cumulative(self.result)
        self.assertIsInstance(fig, matplotlib.figure.Figure)

    def test_uncertainty_histogram_with_none(self):
        """uc_result가 없으면 None 반환."""
        from evaluation.well_report import plot_uncertainty_histogram
        # rech_samples가 없는 mock
        class MockUC:
            rech_samples = None
        self.assertIsNone(plot_uncertainty_histogram(MockUC()))

    def test_uncertainty_histogram_with_samples(self):
        from evaluation.well_report import plot_uncertainty_histogram
        import matplotlib.figure
        rng = np.random.default_rng(0)
        class MockUC:
            rech_samples = rng.normal(15.0, 2.0, 100).tolist()
            rech_mean = 15.0
            rech_ci_lower = 11.5
            rech_ci_upper = 18.5
            confidence_level = 0.95
            n_bootstrap = 100
        fig = plot_uncertainty_histogram(MockUC())
        self.assertIsInstance(fig, matplotlib.figure.Figure)

    def test_sensitivity_tornado_with_none(self):
        from evaluation.well_report import plot_sensitivity_tornado
        class MockKS:
            tornado_data = None
        self.assertIsNone(plot_sensitivity_tornado(MockKS()))

    def test_bma_posterior_with_none(self):
        from evaluation.well_report import plot_bma_posterior
        class MockBMA:
            posterior = None
            soil_names = None
        self.assertIsNone(plot_bma_posterior(MockBMA()))


class TestHTMLBuilder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = _get_demo_result_v27()

    def test_minimal_report(self):
        """result_v27만으로 HTML 생성 — 옵션 섹션은 'skipped' 표시."""
        from evaluation.well_report import build_well_html_report
        html = build_well_html_report(
            result_v27=self.result,
            site_name="DEMO-Test",
            soil_label="Loam (sn=6)",
        )
        self.assertIsInstance(html, str)
        self.assertIn("DEMO-Test", html)
        self.assertIn("Loam (sn=6)", html)
        # 항상 포함 섹션
        self.assertIn("Core estimate", html)
        self.assertIn("Plausibility", html)
        self.assertIn("Time-series", html)
        # 옵션 섹션 — skipped 마커
        self.assertIn("Bootstrap", html)
        self.assertIn("실행되지 않았습니다", html)
        # 임베드된 figure (>= 2: time series + cumulative)
        n_imgs = html.count("data:image/png;base64,")
        self.assertGreaterEqual(n_imgs, 2)

    def test_writes_to_file(self):
        from evaluation.well_report import build_well_html_report
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            out = f.name
        try:
            build_well_html_report(
                result_v27=self.result,
                site_name="Disk-Test",
                output_path=out,
            )
            self.assertTrue(os.path.exists(out))
            with open(out, encoding="utf-8") as f:
                content = f.read()
            self.assertIn("Disk-Test", content)
        finally:
            if os.path.exists(out):
                os.remove(out)

    def test_full_report_with_all_optional(self):
        """모든 보조 분석이 있을 때 모든 섹션이 활성화된다."""
        from evaluation.well_report import build_well_html_report

        rng = np.random.default_rng(0)
        class MockUC:
            n_bootstrap = 100
            confidence_level = 0.95
            rech_mean = 15.0; rech_ci_lower = 12.0; rech_ci_upper = 18.0
            rmse_mean = 0.05; rmse_ci_lower = 0.04; rmse_ci_upper = 0.06
            sy_mean = 0.15; sy_ci_lower = 0.12; sy_ci_upper = 0.18
            rech_samples = rng.normal(15.0, 2.0, 100).tolist()

        class MockBMA:
            posterior = np.array([0.1, 0.05, 0.05, 0.4, 0.1, 0.1,
                                  0.05, 0.05, 0.05, 0.02, 0.02, 0.01])
            soil_names = [f"Soil-{i}" for i in range(1, 13)]
            recharge_mean = 16.0
            recharge_ci_lo = 13.0
            recharge_ci_hi = 19.0
            recharge_per_soil = np.linspace(10, 20, 12)
            bic_values = np.zeros(12)
            log_likelihoods = np.zeros(12)
            n_effective_models = 4.5
            dominant_soil = 4
            dominant_prob = 0.4
            confidence_label = "보통"

        class MockKS:
            tornado_data = [
                ("rho", 14.0, 16.0, 0.5, 0.95),
                ("Q/R", 13.5, 16.5, 0.05, 0.8),
                ("alpha", 14.5, 15.5, 0.1, 0.9),
            ]
            baseline_recharge = 15.0
            sensitivity_rho = 0.13
            sensitivity_qr = 0.20
            sensitivity_alpha = 0.07

        mock_pump = {
            "v27_orig": {"rmse": 0.06, "cc": 0.85, "rech_rate": 14.0},
            "v27_corr": {"rmse": 0.05, "cc": 0.88, "rech_rate": 15.5},
        }

        html = build_well_html_report(
            result_v27=self.result,
            site_name="Full-Report",
            soil_label="Loam (sn=6)",
            uc_result=MockUC(),
            bma_result=MockBMA(),
            kalman_sens=MockKS(),
            pump_result=mock_pump,
        )

        # 모든 섹션이 활성 상태 (skipped 마커가 없어야 함)
        self.assertNotIn("Bootstrap CI 분석이 실행되지", html)
        self.assertNotIn("BMA 분석이 실행되지", html)
        self.assertNotIn("민감도 분석이 실행되지", html)
        self.assertNotIn("펌핑 전처리가 실행되지", html)

        # figure 5개 (time, cum, uc, bma, sens) + 임베드 확인
        n_imgs = html.count("data:image/png;base64,")
        self.assertGreaterEqual(n_imgs, 5)

        # 펌핑 비교 표
        self.assertIn("0.0600", html)  # rmse_orig
        self.assertIn("14.00", html)   # rech_orig

    def test_none_result_raises(self):
        from evaluation.well_report import build_well_html_report
        with self.assertRaises(ValueError):
            build_well_html_report(result_v27=None)

    def test_three_method_comparison_section(self):
        """SCS-CN + FAO-56 결과를 함께 넣으면 §3 비교 섹션 활성화."""
        from evaluation.well_report import build_well_html_report
        from scs_cn import estimate_recharge_scs_cn, derive_cn
        from fao56_swb import estimate_recharge_fao56
        from kma_adapter import fetch_mock_korean_climate

        # 같은 강수 시계열 사용 (mock)
        kma = fetch_mock_korean_climate(
            stn_id=133, start_date="2024-01-01", end_date="2024-12-31",
            seed=7,
        )

        scs = estimate_recharge_scs_cn(
            P_daily_mm=kma.P_mm,
            CN=derive_cn("B", "혼합농경지"),
            soil_hydro_group="B", land_use="혼합농경지",
        )
        fao = estimate_recharge_fao56(
            P_daily_mm=kma.P_mm,
            Tmean_C=kma.Tmean_C, Tmax_C=kma.Tmax_C, Tmin_C=kma.Tmin_C,
            lat_deg=kma.lat_deg,
            texture_group="medium", land_use="혼합농경지",
            runoff_fraction=0.20,
        )

        html = build_well_html_report(
            result_v27=self.result,
            site_name="Three-Method-Test",
            soil_label="Loam (sn=6)",
            scs_result=scs,
            fao56_result=fao,
        )

        # 3-method 섹션이 들어 있어야 함
        self.assertIn("Method comparison", html)
        self.assertIn("hybrid-recharge", html)
        self.assertIn("SCS-CN", html)
        self.assertIn("FAO-56", html)
        # 카테고리 분리 표시 (ET-반영 vs 침투-only)
        self.assertIn("핵심 수렴 판정", html)
        self.assertIn("ET-반영", html)
        self.assertIn("침투 상한", html)

        # 'skipped' 표시는 없어야 함
        self.assertNotIn("SCS-CN 또는 FAO-56 결과가", html)

    def test_et_aware_methods_drive_convergence(self):
        """SCS-CN이 발산해도 WTF+FAO-56이 일치하면 'converged' verdict."""
        from evaluation.well_report import _convergence_verdict
        # 실제 김천 자료 모사: WTF 24.7%, FAO-56 26.1%, SCS-CN 98.6%
        estimates = [
            {"name": "hybrid-recharge", "rech_pct": 24.7, "rech_mm": 90,
             "lo_pct": None, "hi_pct": None, "category": "et_aware",
             "method_type": ""},
            {"name": "SCS-CN", "rech_pct": 98.6, "rech_mm": 361,
             "lo_pct": 92, "hi_pct": 99, "category": "infiltration_only",
             "method_type": ""},
            {"name": "FAO-56 SWB", "rech_pct": 26.1, "rech_mm": 95,
             "lo_pct": None, "hi_pct": None, "category": "et_aware",
             "method_type": ""},
        ]
        conv = _convergence_verdict(estimates)
        # SCS-CN 무시하고 WTF+FAO-56만 평가 → 수렴 (1.4%p 차이, 5.6% 상대차)
        self.assertEqual(conv["verdict"], "converged")
        self.assertEqual(conv["n_primary"], 2)
        self.assertEqual(conv["n_supplementary"], 1)
        # 권장 보고값 = ET-aware median ≈ 25.4%
        self.assertAlmostEqual(conv["recommended_pct"], 25.4, places=1)

    def test_method_comparison_skipped_when_only_wtf(self):
        """SCS-CN/FAO-56 없이 WTF만 → 안내 문구 표시."""
        from evaluation.well_report import build_well_html_report
        html = build_well_html_report(
            result_v27=self.result,
            site_name="WTF-Only-Test",
            scs_result=None,
            fao56_result=None,
        )
        self.assertIn("SCS-CN 또는 FAO-56 결과가", html)


class TestTabImport(unittest.TestCase):
    def test_module_imports(self):
        from ui import tab_well_report
        self.assertTrue(hasattr(tab_well_report, "render"))

    def test_app_v30_wires_tab(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "app_v30.py",
        )
        with open(path, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("tab_well_report", content)
        self.assertIn("tab_well_report.render", content)
        self.assertIn("관측정 리포트", content)


if __name__ == "__main__":
    unittest.main()
