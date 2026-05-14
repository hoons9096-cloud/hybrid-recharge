"""coord_utils.py — Korean TM ↔ WGS84 좌표 변환 유틸.

사용 시나리오
-----------
- 한국 토양도 (.shp) 는 EPSG:5186 (Korean 2000 / Central Belt 2010) 좌표.
- wells_registry 는 WGS84 (lat, lon) 으로 저장.
- 사용자가 현장에서 받는 좌표는 TM (X, Y) 인 경우가 흔함.
- 두 좌표계 간 양방향 변환을 한 곳에서 처리.

좌표계 메모
----------
- EPSG:4326 = WGS84 (lat, lon, 도)
- EPSG:5186 = Korean 2000 / Central Belt 2010
              · false_easting = 200000
              · false_northing = 600000
              · central_meridian = 127°E
              · lat_origin = 38°N
              · 한국 중부원점 (typical X ~ 150–250 km, Y ~ 400–600 km)

전국 통합 ITRF2010 (TM-K) 도 거의 동일하나, 토양도(.shp) 가 5186 기준이므로
이 모듈은 5186 을 표준 TM 으로 가정한다.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Tuple

WGS84 = "EPSG:4326"
KOREAN_TM = "EPSG:5186"


@lru_cache(maxsize=2)
def _transformer(src: str, dst: str):
    from pyproj import Transformer
    return Transformer.from_crs(src, dst, always_xy=True)


def tm_to_wgs84(x: float, y: float) -> Tuple[float, float]:
    """Korean TM (X, Y) [m] → (lat, lon) [deg]."""
    tr = _transformer(KOREAN_TM, WGS84)
    lon, lat = tr.transform(float(x), float(y))
    return float(lat), float(lon)


def wgs84_to_tm(lat: float, lon: float) -> Tuple[float, float]:
    """(lat, lon) [deg] → Korean TM (X, Y) [m]."""
    tr = _transformer(WGS84, KOREAN_TM)
    x, y = tr.transform(float(lon), float(lat))
    return float(x), float(y)


def looks_like_tm(x: float, y: float) -> bool:
    """입력값이 TM 으로 보이는지 휴리스틱 판정.

    한국 EPSG:5186 의 일반적 범위:
      X ≈ 100 000 ~ 350 000
      Y ≈ 250 000 ~ 700 000
    """
    return 50_000 <= float(x) <= 500_000 and 200_000 <= float(y) <= 800_000


def looks_like_wgs84(lat: float, lon: float) -> bool:
    """한국 영토 범위의 WGS84 lat/lon 인지."""
    return 33.0 <= float(lat) <= 39.0 and 124.0 <= float(lon) <= 132.0
