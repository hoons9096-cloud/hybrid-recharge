"""test_tab_field_report.py — Streamlit 탭 모듈 import/interface 스모크 테스트.

Streamlit UI를 헤드리스로 완전 시뮬레이션하기는 어렵지만, 최소한 다음을
보장한다:
  1. ui.tab_field_report 모듈이 정상 import 된다.
  2. render(tab, ctx) 시그니처가 존재한다.
  3. 내부 _run_pipeline()이 합성 시나리오에서 session_state 키 3개를 채운다
     (Streamlit session_state를 dict로 monkey-patch하여 간접 검증).
"""
from __future__ import annotations

import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import matplotlib
matplotlib.use("Agg")

# 프로젝트 루트 경로
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTabImport(unittest.TestCase):
    def test_module_imports(self):
        from ui import tab_field_report
        self.assertTrue(hasattr(tab_field_report, "render"))

    def test_render_signature(self):
        import inspect
        from ui import tab_field_report
        sig = inspect.signature(tab_field_report.render)
        # render(tab, ctx) — 2개 인자
        self.assertEqual(len(sig.parameters), 2)
        self.assertIn("tab", sig.parameters)
        self.assertIn("ctx", sig.parameters)


class TestPipelineEndToEnd(unittest.TestCase):
    """_run_pipeline()이 session_state에 결과를 저장하는지 검증.

    Streamlit의 st.session_state를 일반 dict로 대체하고 함수를 직접 호출.
    """

    def test_pipeline_populates_session_state(self):
        from ui import tab_field_report

        fake_state: dict = {}

        # st.session_state를 dict-like 객체로 패치
        with patch.object(tab_field_report.st, "session_state", fake_state):
            tab_field_report._run_pipeline(
                scenario="S3",
                method_labels=["Lumped", "Soil-weighted"],
                site_name="Test-Site",
            )

        # 3개 키가 채워져야 함
        self.assertIn("field_report_html", fake_state)
        self.assertIn("field_report_results", fake_state)
        self.assertIn("field_report_meta", fake_state)

        # HTML 내용 검증
        html = fake_state["field_report_html"]
        self.assertIsInstance(html, str)
        self.assertIn("Test-Site", html)

        # results: 2개 method
        results = fake_state["field_report_results"]
        self.assertEqual(set(results.keys()), {"Lumped", "Soil-weighted"})
        for R in results.values():
            self.assertEqual(R.shape, (100, 100))  # S3 default domain

        # meta: 필수 필드
        meta = fake_state["field_report_meta"]
        for key in ("scenario", "site_name", "P_annual_mm",
                    "n_days", "n_wells", "domain", "observations"):
            self.assertIn(key, meta)


class TestAppV30Wiring(unittest.TestCase):
    """app_v30.py가 새 탭을 포함하도록 수정되었는지 검증."""

    def test_app_v30_imports_new_tab(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "app_v30.py",
        )
        with open(path, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("tab_field_report", content)
        self.assertIn("tab_field_report.render", content)
        self.assertIn("Field 리포트", content)


if __name__ == "__main__":
    unittest.main()
