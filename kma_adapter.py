"""
kma_adapter.py -- KMA APIHub ASOS daily-data adapter.

Fetches daily mean / max / min air temperature and precipitation from the
Korea Meteorological Administration's APIHub for a chosen station and
date range.  Output is normalised to a clean dict-of-arrays consumable by
fao56_swb.estimate_recharge_fao56().

Requires
--------
- Free account at https://apihub.kma.go.kr (issues `authKey`)
- Service activation: "지상관측 (ASOS) 일자료" (활용신청 후 자동 승인)
- API key passed via env var KMA_API_KEY (recommended), an explicit
  argument, or a Streamlit sidebar text input.

The adapter is intentionally thin: it does HTTP, parses the KMA fixed-
width text response (no auth-required JSON endpoint), validates rows,
and gracefully fills gaps.

References
----------
KMA APIHub documentation:
    https://apihub.kma.go.kr — "지상(종관, ASOS) 일자료" detail page.
"""
from __future__ import annotations

import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np


# ══════════════════════════════════════════════════════════════════════
# 한국 주요 ASOS 지점 (대전 / 추풍령 / 김천 권역 + 자주 쓰이는 보조)
# ══════════════════════════════════════════════════════════════════════
# stnIds, name, latitude, longitude
KMA_STATIONS: Dict[int, Dict] = {
    133: {"name": "대전",     "lat": 36.3724, "lon": 127.3720},
    135: {"name": "추풍령",   "lat": 36.2204, "lon": 127.9941},
    137: {"name": "상주",     "lat": 36.4054, "lon": 128.1572},
    143: {"name": "대구",     "lat": 35.8779, "lon": 128.6533},
    232: {"name": "천안",     "lat": 36.7766, "lon": 127.1213},
    279: {"name": "구미",     "lat": 36.1303, "lon": 128.3208},
    136: {"name": "Yeongcheon",     "lat": 35.9774, "lon": 128.9514},
    281: {"name": "영주",     "lat": 36.8717, "lon": 128.5167},
    108: {"name": "서울",     "lat": 37.5714, "lon": 126.9658},
    159: {"name": "부산",     "lat": 35.1047, "lon": 129.0319},
    105: {"name": "강릉",     "lat": 37.7515, "lon": 128.8910},
    184: {"name": "제주",     "lat": 33.5141, "lon": 126.5297},
}


def list_stations() -> List[Dict]:
    """전체 지점 목록 (UI 드롭다운용)."""
    return [
        {"stn_id": k, **v} for k, v in sorted(KMA_STATIONS.items())
    ]


def nearest_station(lat: float, lon: float) -> Tuple[int, Dict]:
    """좌표 → 가장 가까운 ASOS 지점 (구면 거리 근사)."""
    def dist2(s):
        dlat = s["lat"] - lat
        dlon = (s["lon"] - lon) * np.cos(np.deg2rad((s["lat"] + lat) / 2))
        return dlat * dlat + dlon * dlon
    best = min(KMA_STATIONS.items(), key=lambda kv: dist2(kv[1]))
    return best[0], best[1]


# ══════════════════════════════════════════════════════════════════════
# 결과 dataclass
# ══════════════════════════════════════════════════════════════════════
@dataclass
class KMADaily:
    """Normalised daily ASOS data."""
    stn_id: int
    stn_name: str
    lat_deg: float
    lon_deg: float
    dates: List[str]                     # "YYYY-MM-DD"
    P_mm: np.ndarray = field(default_factory=lambda: np.array([]))
    Tmean_C: np.ndarray = field(default_factory=lambda: np.array([]))
    Tmax_C: np.ndarray = field(default_factory=lambda: np.array([]))
    Tmin_C: np.ndarray = field(default_factory=lambda: np.array([]))
    n_days: int = 0
    n_missing_T: int = 0                 # 결측 보간된 일수
    n_missing_P: int = 0
    fetched_from: str = "kma_apihub"     # or "mock"


# ══════════════════════════════════════════════════════════════════════
# 핵심 다운로드 함수
# ══════════════════════════════════════════════════════════════════════
KMA_BASE_URL = "https://apihub.kma.go.kr/api/typ01/url/kma_sfcdd3.php"


def fetch_kma_daily(
    stn_id: int,
    start_date: str,
    end_date: str,
    auth_key: Optional[str] = None,
    timeout_s: float = 30.0,
) -> KMADaily:
    """Fetch daily ASOS data for [start_date, end_date] inclusive.

    Parameters
    ----------
    stn_id : int
        KMA ASOS station number (e.g., 133 for 대전).
    start_date, end_date : str
        "YYYY-MM-DD".
    auth_key : str, optional
        KMA APIHub authKey.  If None, reads from env var KMA_API_KEY.
    timeout_s : float
        HTTP timeout in seconds.

    Returns
    -------
    KMADaily

    Raises
    ------
    ValueError
        If station unknown or dates invalid.
    KMAAdapterError
        On HTTP / parsing failure (subclass of RuntimeError).
    """
    if stn_id not in KMA_STATIONS:
        raise ValueError(
            f"Unknown station {stn_id}. "
            f"Known: {sorted(KMA_STATIONS)}"
        )
    key = auth_key or os.environ.get("KMA_API_KEY")
    if not key:
        raise KMAAdapterError(
            "KMA_API_KEY not set.  Pass auth_key= or set env var."
        )

    sd = _parse_date(start_date)
    ed = _parse_date(end_date)
    if ed < sd:
        raise ValueError(f"end_date {end_date} < start_date {start_date}")

    # KMA APIHub 일자료 endpoint (kma_sfcdd3): tm1, tm2, stn, help, authKey
    params = {
        "tm1": sd.strftime("%Y%m%d"),
        "tm2": ed.strftime("%Y%m%d"),
        "stn": str(stn_id),
        "help": "0",
        "authKey": key,
    }
    qs = urllib.parse.urlencode(params)
    url = f"{KMA_BASE_URL}?{qs}"

    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            err_body = ""
        raise KMAAdapterError(
            f"HTTP {e.code} from KMA APIHub: {e.reason}.  "
            f"Body: {err_body[:200]}"
        ) from e
    except Exception as e:
        raise KMAAdapterError(f"Network error: {e}") from e

    return parse_kma_response(body, stn_id, sd, ed)


# ══════════════════════════════════════════════════════════════════════
# 응답 파싱
# ══════════════════════════════════════════════════════════════════════
class KMAAdapterError(RuntimeError):
    """Raised on KMA API errors (HTTP, auth, parse)."""


# kma_sfcdd3 응답 컬럼 인덱스 (0-base, KMA APIHub 문서 'help=1' 기준)
# 응답이 위치 기반 고정 컬럼이므로 헤더 이름 대신 인덱스로 파싱
#
#   0  TM         YYYYMMDD          (날짜, KST)
#   1  STN        지점번호
#   2  WS_AVG     일 평균 풍속
#   ...
#  10  TA_AVG     일 평균 기온 (°C)
#  11  TA_MAX     최고 기온 (°C)
#  12  TA_MAX_TM  최고기온 시각 (시분)
#  13  TA_MIN     최저 기온 (°C)
#  14  TA_MIN_TM  최저기온 시각 (시분)
#  ...
#  38  RN_DAY     일 강수량 (mm)
#
# 결측 sentinel: -9, -9.0, -9.00, -99.9 등 음수
KMA_SFCDD3_COLS = {
    "TM": 0,
    "STN": 1,
    "TA_AVG": 10,
    "TA_MAX": 11,
    "TA_MIN": 13,
    "RN_DAY": 38,
}


def parse_kma_response(
    body: str,
    stn_id: int,
    start: date,
    end: date,
) -> KMADaily:
    """Parse KMA APIHub kma_sfcdd3.php fixed-width text response into KMADaily.

    Response format:
      #START7777
      #--- info lines ---
      # YYMMDD STN   WS  ... TA  TA  TA  TA  TA  ... RN  ...   (column header lines)
      20240601 133  2.6 ...  20.0 25.3 1504 16.6 529  ... 0.0 ...  (data rows)
      ...
      #7777END

    위치 기반 파싱 (KMA_SFCDD3_COLS).
    """
    lines = body.splitlines()

    # 1. 오류 응답 감지 (JSON payload — 활용신청 미승인 등)
    if not lines or lines[0].lstrip().startswith("{"):
        raise KMAAdapterError(
            f"KMA API returned non-data payload: {body[:300]}"
        )

    idx_tm  = KMA_SFCDD3_COLS["TM"]
    idx_avg = KMA_SFCDD3_COLS["TA_AVG"]
    idx_max = KMA_SFCDD3_COLS["TA_MAX"]
    idx_min = KMA_SFCDD3_COLS["TA_MIN"]
    idx_rn  = KMA_SFCDD3_COLS["RN_DAY"]
    needed_cols = max(idx_tm, idx_avg, idx_max, idx_min, idx_rn) + 1

    parsed: Dict[str, Dict[str, float]] = {}
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        toks = s.split()
        if len(toks) < needed_cols:
            continue
        try:
            d = _parse_kma_date(toks[idx_tm])
        except Exception:
            continue
        rec: Dict[str, float] = {
            "Tmean": _to_float_or_nan(toks[idx_avg]),
            "Tmax":  _to_float_or_nan(toks[idx_max]),
            "Tmin":  _to_float_or_nan(toks[idx_min]),
        }
        rn = _to_float_or_nan(toks[idx_rn])
        # 강수: 결측은 NaN, 음수는 0(무강수) 처리
        rec["P"] = 0.0 if np.isnan(rn) else max(rn, 0.0)
        parsed[d.isoformat()] = rec

    # 일자별 정렬 + 결측 보간
    n_days = (end - start).days + 1
    dates = [(start + timedelta(days=i)).isoformat() for i in range(n_days)]
    Tmean = np.full(n_days, np.nan)
    Tmax = np.full(n_days, np.nan)
    Tmin = np.full(n_days, np.nan)
    P = np.full(n_days, 0.0)

    n_miss_T = 0
    n_miss_P = 0
    for i, ds in enumerate(dates):
        rec = parsed.get(ds)
        if rec is None:
            n_miss_T += 1
            n_miss_P += 1
            continue
        Tmean[i] = rec.get("Tmean", np.nan)
        Tmax[i] = rec.get("Tmax", np.nan)
        Tmin[i] = rec.get("Tmin", np.nan)
        rn = rec.get("P", np.nan)
        P[i] = 0.0 if np.isnan(rn) else rn
        if np.isnan(Tmean[i]) or np.isnan(Tmax[i]) or np.isnan(Tmin[i]):
            n_miss_T += 1
        if np.isnan(rn):
            n_miss_P += 1

    # 기온 결측은 선형 보간 (양 끝은 가장 가까운 값으로 채움)
    Tmean = _interpolate_nan(Tmean)
    Tmax = _interpolate_nan(Tmax)
    Tmin = _interpolate_nan(Tmin)

    meta = KMA_STATIONS[stn_id]
    return KMADaily(
        stn_id=stn_id,
        stn_name=meta["name"],
        lat_deg=meta["lat"],
        lon_deg=meta["lon"],
        dates=dates,
        P_mm=P,
        Tmean_C=Tmean,
        Tmax_C=Tmax,
        Tmin_C=Tmin,
        n_days=n_days,
        n_missing_T=n_miss_T,
        n_missing_P=n_miss_P,
        fetched_from="kma_apihub",
    )


# ══════════════════════════════════════════════════════════════════════
# 헬퍼
# ══════════════════════════════════════════════════════════════════════
def _parse_date(s: str) -> date:
    """YYYY-MM-DD or YYYYMMDD."""
    s = s.strip().replace("-", "").replace("/", "")
    if len(s) != 8:
        raise ValueError(f"bad date '{s}', expected YYYY-MM-DD or YYYYMMDD")
    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def _parse_kma_date(s: str) -> date:
    s = s.strip()
    if len(s) >= 8:
        return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    raise ValueError(f"bad KMA date '{s}'")


def _to_float_or_nan(s: str) -> float:
    """Convert a token to float, treating KMA missing-value sentinels as NaN.

    KMA APIHub uses -9.0 / -99.0 / -99.9 as missing markers in the kma_sfcdd3
    response.  Note: real Korean Tmin can occasionally hit -9.0°C in deep
    winter, which would be falsely flagged as missing — the cost is one
    interpolated day, acceptable for daily-aggregate use.
    """
    try:
        v = float(s)
    except (ValueError, TypeError):
        return np.nan
    # 안전 하한 (-50°C 미만은 한국에서 불가능)
    if v <= -50.0:
        return np.nan
    # KMA 명시적 결측 sentinel (정확 매칭, 0.05 tolerance)
    for sentinel in (-9.0, -99.0, -99.9):
        if abs(v - sentinel) < 0.05:
            return np.nan
    return v


def _interpolate_nan(arr: np.ndarray) -> np.ndarray:
    """선형 보간, 양 끝은 가장 가까운 유효값."""
    out = arr.copy()
    n = len(out)
    if n == 0 or not np.any(np.isnan(out)):
        return out
    if np.all(np.isnan(out)):
        out[:] = 0.0
        return out
    valid_idx = np.where(~np.isnan(out))[0]
    nan_idx = np.where(np.isnan(out))[0]
    out[nan_idx] = np.interp(nan_idx, valid_idx, out[valid_idx])
    return out


# ══════════════════════════════════════════════════════════════════════
# Mock — 테스트용 (활용신청 승인 전에도 통합 가능)
# ══════════════════════════════════════════════════════════════════════
def fetch_mock_korean_climate(
    stn_id: int,
    start_date: str,
    end_date: str,
    seed: int = 42,
) -> KMADaily:
    """API 승인 전 / 오프라인 테스트용 합성 한국 기후 데이터.

    실 API와 동일한 KMADaily 인터페이스 반환.
    """
    if stn_id not in KMA_STATIONS:
        raise ValueError(f"Unknown station {stn_id}")

    sd = _parse_date(start_date)
    ed = _parse_date(end_date)
    n = (ed - sd).days + 1
    rng = np.random.default_rng(seed)
    doy = np.array([(sd + timedelta(days=i)).timetuple().tm_yday for i in range(n)])

    # 기온 (대전 평균 12.5°C, 진폭 15°C)
    Tmean = 12.5 + 15.0 * np.sin(2*np.pi*(doy-110)/365) + rng.normal(0, 2, n)
    Tmax = Tmean + 4.0 + rng.normal(0, 1, n)
    Tmin = Tmean - 4.0 + rng.normal(0, 1, n)

    # 강수 (몬순)
    wet_prob = np.clip(0.18 + 0.20*np.sin(2*np.pi*(doy-80)/365), 0.05, 0.55)
    is_wet = rng.random(n) < wet_prob
    intensity_scale = 8.0 + 25.0 * np.clip(np.sin(2*np.pi*(doy-80)/365), 0.0, 1.0)
    P = np.where(is_wet, rng.exponential(intensity_scale), 0.0)
    P = np.clip(P, 0, 200)

    meta = KMA_STATIONS[stn_id]
    dates = [(sd + timedelta(days=i)).isoformat() for i in range(n)]
    return KMADaily(
        stn_id=stn_id,
        stn_name=meta["name"],
        lat_deg=meta["lat"],
        lon_deg=meta["lon"],
        dates=dates,
        P_mm=P,
        Tmean_C=Tmean,
        Tmax_C=Tmax,
        Tmin_C=Tmin,
        n_days=n,
        n_missing_T=0,
        n_missing_P=0,
        fetched_from="mock",
    )


# ══════════════════════════════════════════════════════════════════════
# 자체 시연
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=== kma_adapter.py self-test (mock data) ===\n")
    data = fetch_mock_korean_climate(
        stn_id=133, start_date="2024-01-01", end_date="2024-12-31",
    )
    print(f"  Station:  {data.stn_id} {data.stn_name} "
          f"({data.lat_deg:.4f}°N, {data.lon_deg:.4f}°E)")
    print(f"  Days:     {data.n_days}")
    print(f"  Annual P: {data.P_mm.sum():.0f} mm")
    print(f"  Tmean:    {data.Tmean_C.mean():.1f} °C "
          f"(min {data.Tmean_C.min():.1f}, max {data.Tmean_C.max():.1f})")
    print(f"  Source:   {data.fetched_from}")

    # FAO-56과 결합 데모
    print(f"\n=== FAO-56 with mock KMA data ===")
    from fao56_swb import estimate_recharge_fao56
    r = estimate_recharge_fao56(
        P_daily_mm=data.P_mm,
        Tmean_C=data.Tmean_C, Tmax_C=data.Tmax_C, Tmin_C=data.Tmin_C,
        lat_deg=data.lat_deg,
        texture_group="medium", land_use="혼합농경지",
        runoff_fraction=0.20,        # 한국 몬순 표면유출 ~20%
    )
    print(f"  P_annual:    {r.P_annual_mm:.0f} mm/yr")
    print(f"  ETo:         {r.ETo_annual_mm:.0f} mm/yr")
    print(f"  ETa:         {r.ETa_annual_mm:.0f} mm/yr")
    print(f"  Runoff:      {r.runoff_annual_mm:.0f} mm/yr")
    print(f"  Recharge:    {r.R_annual_mm:.0f} mm/yr ({r.recharge_ratio_pct:.1f}%)")
