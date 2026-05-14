"""shp_soil_mapper.py — 정밀토양도 (.shp) 공간 조인.

전국_정밀토양도_GRS80.shp 의 컬럼:
  Soil_Code   — 토양형 코드 (예: AaB)
  Hydro_Type  — HSG (A/B/C/D)
  Hydro_Ty_N  — HSG 숫자형 (1~4)
  K           — 수리전도도 또는 토양침식인자
  geometry    — Polygon (CRS: ITRF2000 TM, 한국 중부원점)

기능:
  1. 관정 점좌표(WGS84) → Korean TM 변환 → 점이 속한 폴리곤 검색
  2. 유역 폴리곤(또는 관정 리스트의 convex hull) 내 HSG 면적 분포 계산
  3. HSG → FAO-56 texture_group / WTF Sy 매핑
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple

# 지연 임포트 (geopandas가 없는 환경에서도 다른 모듈 import 가능하도록)


SHP_PATH_DEFAULT = "/Users/choejeonghun/정밀토양도/전국_정밀토양도_GRS80.shp"

# .prj 파일은 PCS_ITRF2000_TM (false_easting=200000, false_northing=600000,
# central_meridian=127, lat_origin=38) 인데 EPSG 코드가 비어 있어
# 직접 명시해야 함.  이 정의는 EPSG:5186 (Korea 2000 / Central Belt 2010) 와 동일.
SHP_CRS_EPSG = "EPSG:5186"


# ---------------------------------------------------------------------------
# HSG → 수문 파라미터 매핑
# ---------------------------------------------------------------------------
HSG_TO_TEXTURE = {
    "A": "coarse",   # 모래질, 침투성 高
    "B": "medium",   # 양토, 약간 거친
    "C": "medium",   # 양토, 약간 미세 (보수적: medium)
    "D": "fine",     # 점토질, 침투성 低
}

# WTF Sy 추정값 (HSG별 — 문헌 평균)
HSG_TO_SY = {
    "A": 0.25,
    "B": 0.18,
    "C": 0.12,
    "D": 0.07,
}

# SCS-CN AMC II 평균 CN (혼합농경지 가정)
HSG_TO_CN = {
    "A": 64,
    "B": 75,
    "C": 82,
    "D": 85,
}


@dataclass
class SoilQuery:
    """관정 한 지점의 토양 조회 결과."""
    well_name: str
    lat: float
    lon: float
    soil_code: str
    hydro_type: str          # A/B/C/D
    K: float
    texture_group: str       # FAO-56 입력
    Sy: float                # WTF 입력
    CN: int                  # SCS-CN 입력


@dataclass
class WatershedSoilProfile:
    """유역 또는 관정군의 토양 분포."""
    name: str
    hsg_fractions: Dict[str, float]   # {"A": 0.1, "B": 0.4, ...}, 합 = 1
    dominant_hsg: str
    weighted_Sy: float
    weighted_CN: float
    weighted_texture: str             # 면적 가중 majority
    n_wells: int
    total_area_km2: Optional[float] = None


# ---------------------------------------------------------------------------
# Shapefile 로딩 (캐시)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=2)
def _load_shp(path: str = SHP_PATH_DEFAULT):
    import geopandas as gpd
    gdf = gpd.read_file(path)
    # CRS 명시 (Korean TM Central, GRS80, false_easting=200000, false_northing=600000)
    # → EPSG:5179 와 동일
    if gdf.crs is None or gdf.crs.to_epsg() is None:
        gdf = gdf.set_crs(SHP_CRS_EPSG, allow_override=True)
    return gdf


def _to_korean_tm(lat: float, lon: float):
    """WGS84 (lat, lon) → 토양도 CRS (x, y) 변환."""
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", SHP_CRS_EPSG, always_xy=True)
    x, y = tr.transform(lon, lat)
    return x, y


# ---------------------------------------------------------------------------
# 점 조회
# ---------------------------------------------------------------------------
def query_point(
    well_name: str, lat: float, lon: float,
    shp_path: str = SHP_PATH_DEFAULT,
) -> SoilQuery:
    """관정 좌표에 해당하는 토양 폴리곤 조회."""
    from shapely.geometry import Point
    gdf = _load_shp(shp_path)
    x, y = _to_korean_tm(lat, lon)
    pt = Point(x, y)

    # spatial index 활용
    idx = list(gdf.sindex.query(pt, predicate="contains"))
    if not idx:
        # contains 실패 시 nearest
        idx = [int(gdf.sindex.nearest(pt)[1][0])]
    row = gdf.iloc[idx[0]]
    hsg = str(row["Hydro_Type"])
    return SoilQuery(
        well_name=well_name,
        lat=lat, lon=lon,
        soil_code=str(row["Soil_Code"]),
        hydro_type=hsg,
        K=float(row["K"]),
        texture_group=HSG_TO_TEXTURE.get(hsg, "medium"),
        Sy=HSG_TO_SY.get(hsg, 0.15),
        CN=HSG_TO_CN.get(hsg, 75),
    )


# ---------------------------------------------------------------------------
# 유역(또는 관정군) 면적 분포
# ---------------------------------------------------------------------------
def watershed_profile_from_wells(
    name: str,
    wells: Iterable[Tuple[str, float, float]],   # (well_name, lat, lon)
    buffer_km: float = 2.0,
    shp_path: str = SHP_PATH_DEFAULT,
) -> WatershedSoilProfile:
    """관정들 주위 buffer 영역의 HSG 면적 분포.

    유역 폴리곤이 없을 때 대안 — 각 관정에 buffer_km 반경 원을 그려
    union 한 영역을 토양도와 교차해 면적 가중치 계산.
    """
    from shapely.geometry import Point
    from shapely.ops import unary_union
    gdf = _load_shp(shp_path)

    well_list = list(wells)
    if not well_list:
        raise ValueError("wells empty")
    buffers = []
    for _, lat, lon in well_list:
        x, y = _to_korean_tm(lat, lon)
        buffers.append(Point(x, y).buffer(buffer_km * 1000.0))
    region = unary_union(buffers)

    # 토양 폴리곤과 교차
    candidate_idx = list(gdf.sindex.query(region, predicate="intersects"))
    if not candidate_idx:
        raise RuntimeError(f"{name}: no soil polygons within {buffer_km} km")
    sub = gdf.iloc[candidate_idx].copy()
    sub["clipped"] = sub.geometry.intersection(region)
    sub["area_m2"] = sub["clipped"].area

    total = float(sub["area_m2"].sum())
    if total <= 0:
        raise RuntimeError(f"{name}: zero area after intersection")

    fractions: Dict[str, float] = {}
    for hsg in ["A", "B", "C", "D"]:
        a = float(sub.loc[sub["Hydro_Type"] == hsg, "area_m2"].sum())
        if a > 0:
            fractions[hsg] = a / total

    dominant = max(fractions, key=fractions.get)
    w_sy = sum(HSG_TO_SY[h] * f for h, f in fractions.items())
    w_cn = sum(HSG_TO_CN[h] * f for h, f in fractions.items())

    # 면적 majority texture_group
    tex_area: Dict[str, float] = {}
    for h, f in fractions.items():
        tex = HSG_TO_TEXTURE[h]
        tex_area[tex] = tex_area.get(tex, 0.0) + f
    w_tex = max(tex_area, key=tex_area.get)

    return WatershedSoilProfile(
        name=name,
        hsg_fractions=fractions,
        dominant_hsg=dominant,
        weighted_Sy=w_sy,
        weighted_CN=w_cn,
        weighted_texture=w_tex,
        n_wells=len(well_list),
        total_area_km2=total / 1e6,
    )


def query_wells(
    wells: Iterable[Tuple[str, float, float]],
    shp_path: str = SHP_PATH_DEFAULT,
) -> List[SoilQuery]:
    """관정 여러 개 일괄 조회."""
    return [query_point(n, lat, lon, shp_path) for n, lat, lon in wells]
