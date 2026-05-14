"""test_field_report.py — field_report.py figure & HTML builder 테스트.

Figure 함수들이 matplotlib Figure 객체를 정상 반환하는지, HTML 빌더가
필수 섹션을 모두 포함하는지 검증한다.  실제 그림 품질 확인은 smoke test
산출물(/tmp/field_report_S3.html)을 직접 열어 본다.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np
import matplotlib
matplotlib.use("Agg")  # GUI 백엔드 사용 안 함

# 프로젝트 루트 경로
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _setup_synthetic():
    """공통 합성 데이터 (S3) 셋업."""
    from synthetic.generate_domain import generate_domain, DomainConfig
    from synthetic.generate_data import generate_data
    from methods.wtf_lumped import estimate_recharge as est_lumped
    from methods.wtf_soil_weighted import estimate_recharge as est_soil

    domain = generate_domain(DomainConfig.S3())
    data = generate_data(domain)
    observations = {
        "P": data.P, "ET": data.ET,
        "ho_obs": data.ho_obs, "well_soil_types": data.well_soil_types,
    }
    results = {
        "Lumped": est_lumped(domain, observations),
        "Soil-weighted": est_soil(domain, observations),
    }
    P_annual = float(np.sum(data.P)) * 1000.0 / (data.n_days / 365.0)
    return domain, data, observations, results, P_annual


class TestFieldFigures(unittest.TestCase):
    """모든 figure 함수가 Figure 객체를 반환하는지 확인."""

    @classmethod
    def setUpClass(cls):
        cls.domain, cls.data, cls.obs, cls.results, cls.P = _setup_synthetic()

    def test_method_comparison_maps(self):
        from evaluation.field_report import plot_method_comparison_maps
        import matplotlib.figure
        fig = plot_method_comparison_maps(self.results, self.domain)
        self.assertIsInstance(fig, matplotlib.figure.Figure)

    def test_spread_map(self):
        from evaluation.field_report import plot_method_spread_map
        from evaluation.field_metrics import between_method_spread
        import matplotlib.figure
        sp = between_method_spread(self.results)
        fig = plot_method_spread_map(sp, self.domain)
        self.assertIsInstance(fig, matplotlib.figure.Figure)

    def test_histograms(self):
        from evaluation.field_report import plot_recharge_histograms
        import matplotlib.figure
        fig = plot_recharge_histograms(self.results, self.P)
        self.assertIsInstance(fig, matplotlib.figure.Figure)

    def test_boxplots(self):
        from evaluation.field_report import plot_soil_class_boxplots
        import matplotlib.figure
        fig = plot_soil_class_boxplots(self.results, self.domain.soil_map)
        self.assertIsInstance(fig, matplotlib.figure.Figure)

    def test_well_scatter(self):
        from evaluation.field_report import plot_well_consistency_scatter
        import matplotlib.figure
        fig = plot_well_consistency_scatter(self.results, self.obs, self.domain)
        self.assertIsInstance(fig, matplotlib.figure.Figure)

    def test_comparison_with_single_method(self):
        from evaluation.field_report import plot_method_comparison_maps
        import matplotlib.figure
        single = {"Lumped": self.results["Lumped"]}
        fig = plot_method_comparison_maps(single, self.domain)
        self.assertIsInstance(fig, matplotlib.figure.Figure)

    def test_comparison_empty_raises(self):
        from evaluation.field_report import plot_method_comparison_maps
        with self.assertRaises(ValueError):
            plot_method_comparison_maps({}, self.domain)


class TestHTMLBuilder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.domain, cls.data, cls.obs, cls.results, cls.P = _setup_synthetic()

    def test_html_returns_string(self):
        from evaluation.field_report import build_html_report
        html = build_html_report(
            self.results, self.obs, self.domain,
            P_annual_mm=self.P, site_name="Test-Site",
        )
        self.assertIsInstance(html, str)
        self.assertGreater(len(html), 1000)

    def test_html_contains_required_sections(self):
        from evaluation.field_report import build_html_report
        html = build_html_report(
            self.results, self.obs, self.domain,
            P_annual_mm=self.P, site_name="Test-Site",
        )
        for section in [
            "Method comparison maps",
            "Method-to-method spread",
            "Physical plausibility",
            "Soil-class coherence",
            "Well-level consistency",
            "Test-Site",
        ]:
            self.assertIn(section, html, f"missing section: {section}")

    def test_html_embeds_images(self):
        """모든 figure가 base64로 임베드되어 있어야 함."""
        from evaluation.field_report import build_html_report
        html = build_html_report(
            self.results, self.obs, self.domain,
            P_annual_mm=self.P, site_name="Test-Site",
        )
        # data:image/png;base64 마커 카운트 (>=5 figure 기대)
        n_imgs = html.count("data:image/png;base64,")
        self.assertGreaterEqual(n_imgs, 5)

    def test_html_writes_to_file(self):
        from evaluation.field_report import build_html_report
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            out = f.name
        try:
            build_html_report(
                self.results, self.obs, self.domain,
                P_annual_mm=self.P, site_name="Test-Site",
                output_path=out,
            )
            self.assertTrue(os.path.exists(out))
            with open(out, encoding="utf-8") as f:
                content = f.read()
            self.assertGreater(len(content), 1000)
            self.assertIn("Test-Site", content)
        finally:
            if os.path.exists(out):
                os.remove(out)

    def test_html_with_single_method_skips_spread(self):
        """1개 method만 있을 때 spread 섹션은 안내 문구로 대체된다."""
        from evaluation.field_report import build_html_report
        single = {"Lumped": self.results["Lumped"]}
        html = build_html_report(
            single, self.obs, self.domain,
            P_annual_mm=self.P, site_name="Test-Site",
        )
        self.assertIn("Spread analysis requires at least 2 methods", html)

    def test_empty_methods_raises(self):
        from evaluation.field_report import build_html_report
        with self.assertRaises(ValueError):
            build_html_report({}, self.obs, self.domain, P_annual_mm=1000.0)


if __name__ == "__main__":
    unittest.main()
