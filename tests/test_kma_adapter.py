"""test_kma_adapter.py — KMA APIHub ASOS adapter tests.

Live API 호출은 활용신청 승인 후 별도 통합 테스트로.  여기서는
mock 응답 / 합성 데이터로 어댑터 인터페이스를 검증한다.
"""
from __future__ import annotations

import os
import sys
import unittest
from datetime import date

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestStationCatalog(unittest.TestCase):
    def test_known_korean_stations_present(self):
        from kma_adapter import KMA_STATIONS
        # 대전, 추풍령, 김천 권역
        self.assertIn(133, KMA_STATIONS)
        self.assertIn(135, KMA_STATIONS)
        self.assertEqual(KMA_STATIONS[133]["name"], "대전")
        self.assertEqual(KMA_STATIONS[135]["name"], "추풍령")

    def test_lat_lon_in_korea_range(self):
        from kma_adapter import KMA_STATIONS
        for stn, meta in KMA_STATIONS.items():
            self.assertGreater(meta["lat"], 33.0, f"stn {stn}")
            self.assertLess(meta["lat"], 38.5, f"stn {stn}")
            self.assertGreater(meta["lon"], 124.0, f"stn {stn}")
            self.assertLess(meta["lon"], 132.0, f"stn {stn}")

    def test_list_stations(self):
        from kma_adapter import list_stations
        out = list_stations()
        self.assertIsInstance(out, list)
        self.assertGreater(len(out), 5)
        for s in out:
            self.assertIn("stn_id", s)
            self.assertIn("name", s)
            self.assertIn("lat", s)
            self.assertIn("lon", s)


class TestNearestStation(unittest.TestCase):
    def test_daejeon_coordinates(self):
        from kma_adapter import nearest_station
        # 대전 시청 ~ 36.35, 127.39 → 대전(133)이 가장 가까움
        stn_id, _ = nearest_station(36.35, 127.39)
        self.assertEqual(stn_id, 133)

    def test_kimcheon_coordinates(self):
        """김천(약 36.14, 128.11) → 추풍령(135) 또는 구미(279) 후보."""
        from kma_adapter import nearest_station
        stn_id, meta = nearest_station(36.14, 128.11)
        self.assertIn(stn_id, [135, 279, 137])  # 추풍령/구미/상주
        self.assertIn("name", meta)


class TestDateParsing(unittest.TestCase):
    def test_iso_format(self):
        from kma_adapter import _parse_date
        d = _parse_date("2024-06-15")
        self.assertEqual(d, date(2024, 6, 15))

    def test_compact_format(self):
        from kma_adapter import _parse_date
        d = _parse_date("20240615")
        self.assertEqual(d, date(2024, 6, 15))

    def test_invalid_raises(self):
        from kma_adapter import _parse_date
        with self.assertRaises(ValueError):
            _parse_date("2024-06")
        with self.assertRaises(ValueError):
            _parse_date("not-a-date")


class TestMockFetch(unittest.TestCase):
    def test_mock_returns_kmadaily(self):
        from kma_adapter import fetch_mock_korean_climate, KMADaily
        d = fetch_mock_korean_climate(
            stn_id=133, start_date="2024-01-01", end_date="2024-01-31",
        )
        self.assertIsInstance(d, KMADaily)
        self.assertEqual(d.stn_id, 133)
        self.assertEqual(d.n_days, 31)
        self.assertEqual(len(d.P_mm), 31)
        self.assertEqual(len(d.Tmean_C), 31)
        self.assertEqual(d.fetched_from, "mock")

    def test_mock_korean_climate_realistic(self):
        from kma_adapter import fetch_mock_korean_climate
        d = fetch_mock_korean_climate(
            stn_id=133, start_date="2024-01-01", end_date="2024-12-31",
        )
        # 한국 연 평균 기온 8–17°C
        self.assertGreater(d.Tmean_C.mean(), 8.0)
        self.assertLess(d.Tmean_C.mean(), 17.0)
        # 한국 연 강수 800–2000mm
        self.assertGreater(d.P_mm.sum(), 800.0)
        self.assertLess(d.P_mm.sum(), 2500.0)
        # Tmax >= Tmean >= Tmin
        self.assertTrue(np.all(d.Tmax_C >= d.Tmean_C - 0.01))
        self.assertTrue(np.all(d.Tmean_C >= d.Tmin_C - 0.01))

    def test_mock_unknown_station_raises(self):
        from kma_adapter import fetch_mock_korean_climate
        with self.assertRaises(ValueError):
            fetch_mock_korean_climate(stn_id=999, start_date="2024-01-01",
                                       end_date="2024-01-31")


class TestResponseParsing(unittest.TestCase):
    """가짜 KMA 응답으로 parser 검증."""

    def _fake_response(
        self, days, base_date="20240601",
        Tmean=18.5, Tmax=24.3, Tmin=13.1, P=5.2,
    ):
        """KMA APIHub kma_sfcdd3 위치 기반 형식 합성 응답.

        실제 39+ 컬럼 중 사용하는 위치만 의미 있는 값,
        나머지는 0/-9 채움.  KMA_SFCDD3_COLS:
            0=TM, 1=STN, 10=TA_AVG, 11=TA_MAX, 13=TA_MIN, 38=RN_DAY
        """
        header = "#START7777"
        rows = []
        from datetime import datetime, timedelta
        d0 = datetime.strptime(base_date, "%Y%m%d")
        for i in range(days):
            d = (d0 + timedelta(days=i)).strftime("%Y%m%d")
            cols = [d, "133"]
            cols += ["0.0"] * 8                 # 2-9: WS/WD 등
            cols.append(f"{Tmean}")             # 10: TA_AVG
            cols.append(f"{Tmax}")              # 11: TA_MAX
            cols.append("1500")                 # 12: TA_MAX_TM
            cols.append(f"{Tmin}")              # 13: TA_MIN
            cols.append("500")                  # 14: TA_MIN_TM
            cols += ["0.0"] * 23                # 15-37: 기타 컬럼
            cols.append(f"{P}")                 # 38: RN_DAY
            rows.append(" ".join(cols))
        return "\n".join([header] + rows + ["#7777END"])

    def test_parse_clean_response(self):
        from kma_adapter import parse_kma_response
        body = self._fake_response(days=7)
        result = parse_kma_response(body, stn_id=133,
                                    start=date(2024, 6, 1),
                                    end=date(2024, 6, 7))
        self.assertEqual(result.n_days, 7)
        self.assertEqual(len(result.P_mm), 7)
        self.assertAlmostEqual(result.Tmean_C[0], 18.5, places=2)
        self.assertAlmostEqual(result.Tmax_C[0], 24.3, places=2)
        self.assertAlmostEqual(result.Tmin_C[0], 13.1, places=2)
        self.assertAlmostEqual(result.P_mm[0], 5.2, places=2)

    def test_error_payload_raises(self):
        """JSON 오류 응답 (활용신청 미승인)이 KMAAdapterError로 변환."""
        from kma_adapter import parse_kma_response, KMAAdapterError
        body = '{"result": {"status": 403, "message": "활용신청 필요"}}'
        with self.assertRaises(KMAAdapterError):
            parse_kma_response(body, stn_id=133,
                               start=date(2024, 6, 1),
                               end=date(2024, 6, 7))

    def test_missing_temperature_interpolated(self):
        """결측 sentinel(-9.0 / -99.0 등)이 선형 보간되는지."""
        from kma_adapter import parse_kma_response
        # 위치 기반 — 가운데 일자만 결측
        rows = [
            "#START7777",
            self._make_row("20240601", Tmean=10.0, Tmax=15.0, Tmin=5.0),
            self._make_row("20240602", Tmean=-9.0, Tmax=-9.0, Tmin=-9.0),
            self._make_row("20240603", Tmean=20.0, Tmax=25.0, Tmin=15.0),
            "#7777END",
        ]
        body = "\n".join(rows)
        result = parse_kma_response(body, stn_id=133,
                                    start=date(2024, 6, 1),
                                    end=date(2024, 6, 3))
        # 가운데 값은 (10+20)/2 = 15에 선형 보간
        self.assertAlmostEqual(result.Tmean_C[1], 15.0, places=2)
        self.assertEqual(result.n_missing_T, 1)

    def _make_row(self, date_str, Tmean=20.0, Tmax=25.0, Tmin=15.0, P=0.0):
        """위치 기반 KMA 응답 한 줄 (0-base 인덱스 39+ 컬럼)."""
        cols = [date_str, "133"]
        cols += ["0.0"] * 8
        cols += [f"{Tmean}", f"{Tmax}", "1500", f"{Tmin}", "500"]
        cols += ["0.0"] * 23
        cols += [f"{P}"]
        return " ".join(cols)


class TestAuthHandling(unittest.TestCase):
    def test_no_key_raises(self):
        from kma_adapter import fetch_kma_daily, KMAAdapterError
        # 환경변수 보호 — 임시로 제거
        original = os.environ.pop("KMA_API_KEY", None)
        try:
            with self.assertRaises(KMAAdapterError):
                fetch_kma_daily(
                    stn_id=133, start_date="2024-01-01",
                    end_date="2024-01-07",
                )
        finally:
            if original is not None:
                os.environ["KMA_API_KEY"] = original

    def test_unknown_station_raises(self):
        from kma_adapter import fetch_kma_daily
        with self.assertRaises(ValueError):
            fetch_kma_daily(
                stn_id=99999, start_date="2024-01-01", end_date="2024-01-07",
                auth_key="dummy",
            )

    def test_invalid_date_range_raises(self):
        from kma_adapter import fetch_kma_daily
        with self.assertRaises(ValueError):
            fetch_kma_daily(
                stn_id=133, start_date="2024-01-31", end_date="2024-01-01",
                auth_key="dummy",
            )


class TestEndToEndMockToFAO56(unittest.TestCase):
    def test_mock_pipeline_to_fao56(self):
        """mock KMA → FAO-56 → 한국 합리적 함양율."""
        from kma_adapter import fetch_mock_korean_climate
        from fao56_swb import estimate_recharge_fao56

        d = fetch_mock_korean_climate(
            stn_id=133, start_date="2024-01-01", end_date="2024-12-31",
            seed=42,
        )
        r = estimate_recharge_fao56(
            P_daily_mm=d.P_mm,
            Tmean_C=d.Tmean_C, Tmax_C=d.Tmax_C, Tmin_C=d.Tmin_C,
            lat_deg=d.lat_deg,
            texture_group="medium", land_use="혼합농경지",
            runoff_fraction=0.20,
        )
        self.assertGreater(r.recharge_ratio_pct, 5.0)
        self.assertLess(r.recharge_ratio_pct, 60.0)
        self.assertGreater(r.ETo_annual_mm, 700.0)
        self.assertLess(r.ETo_annual_mm, 1300.0)


@unittest.skipUnless(
    os.environ.get("KMA_API_KEY"),
    "KMA_API_KEY 미설정 — 라이브 API 테스트 스킵 (단위 테스트만 실행).",
)
class TestLiveKMAFetch(unittest.TestCase):
    """실제 KMA APIHub 호출 — KMA_API_KEY 환경변수가 있을 때만 실행.

    네트워크 + API 쿼터 사용.  CI에서는 비활성화 권장.
    """

    def test_daejeon_one_week(self):
        from kma_adapter import fetch_kma_daily
        data = fetch_kma_daily(
            stn_id=133, start_date="2024-06-01", end_date="2024-06-07",
        )
        self.assertEqual(data.fetched_from, "kma_apihub")
        self.assertEqual(data.stn_name, "대전")
        self.assertEqual(data.n_days, 7)
        # 6월 초 대전 평균기온 18-25°C 합리적
        self.assertGreater(data.Tmean_C.mean(), 15.0)
        self.assertLess(data.Tmean_C.mean(), 28.0)
        # Tmax >= Tmean >= Tmin
        self.assertTrue(np.all(data.Tmax_C >= data.Tmean_C - 0.01))
        self.assertTrue(np.all(data.Tmean_C >= data.Tmin_C - 0.01))

    def test_live_pipeline_to_fao56(self):
        """라이브 KMA → FAO-56 1년 통합."""
        from kma_adapter import fetch_kma_daily
        from fao56_swb import estimate_recharge_fao56
        from datetime import date

        data = fetch_kma_daily(
            stn_id=133,
            start_date="2023-06-01", end_date="2024-05-31",
        )
        self.assertEqual(data.n_days, 366)
        start_doy = date(2023, 6, 1).timetuple().tm_yday

        r = estimate_recharge_fao56(
            P_daily_mm=data.P_mm,
            Tmean_C=data.Tmean_C, Tmax_C=data.Tmax_C, Tmin_C=data.Tmin_C,
            lat_deg=data.lat_deg, texture_group="medium",
            land_use="혼합농경지", runoff_fraction=0.20,
            start_doy=start_doy,
        )
        # 한국 ETo 800-1300 mm/yr
        self.assertGreater(r.ETo_annual_mm, 800.0)
        self.assertLess(r.ETo_annual_mm, 1300.0)
        # 함양율 비음수, 80% 미만
        self.assertGreaterEqual(r.recharge_ratio_pct, 0.0)
        self.assertLess(r.recharge_ratio_pct, 80.0)


if __name__ == "__main__":
    unittest.main()
