"""wells_registry.py — 관정 좌표 + 유역 그룹 (JSON 기반).

저장 위치: wells_registry.json (프로젝트 루트)
파일이 없으면 김천 기본값으로 자동 생성.

UI 에서 add_well / remove_well / update_well / add_watershed 함수로
관리 가능 — 호출 즉시 JSON 갱신.

기존 호환성: 모듈 레벨 WELLS, WATERSHEDS dict 그대로 노출.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


REGISTRY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "wells_registry.json",
)


@dataclass
class WellInfo:
    name: str
    lat: float
    lon: float
    watershed: str
    aquifer: str = "bedrock"     # "alluvial" | "bedrock"
    nearest_kma: int = 135       # ASOS 관측소 ID


# ---------------------------------------------------------------------------
# 기본값 (JSON 없을 때 초기화)
# ---------------------------------------------------------------------------
_DEFAULT_WELLS = [
    WellInfo("김천남면", 36.0500, 128.1300, "감천",     "alluvial", 135),
    WellInfo("김천지좌", 36.1300, 128.1100, "감천",     "alluvial", 135),
    WellInfo("대덕",     35.9300, 127.9700, "감천상류", "bedrock",  135),
    WellInfo("대덕충적", 35.9320, 127.9720, "감천상류", "alluvial", 135),
    WellInfo("동좌암반", 35.9650, 128.0500, "감천중류", "bedrock",  135),
    WellInfo("동좌충적", 35.9670, 128.0520, "감천중류", "alluvial", 135),
    WellInfo("부항암반", 35.9000, 127.9300, "부항천",   "bedrock",  135),
]


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------
def _load_from_json(path: str = REGISTRY_PATH) -> Dict[str, WellInfo]:
    if not os.path.exists(path):
        # 기본값으로 초기 JSON 생성
        wells = {w.name: w for w in _DEFAULT_WELLS}
        _save_to_json(wells, path)
        return wells
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    wells: Dict[str, WellInfo] = {}
    for name, attr in data.get("wells", {}).items():
        # name 필드 보정
        attr["name"] = name
        # 알 수 없는 키는 무시
        valid = {k: v for k, v in attr.items()
                 if k in WellInfo.__dataclass_fields__}
        wells[name] = WellInfo(**valid)
    return wells


def _save_to_json(wells: Dict[str, WellInfo], path: str = REGISTRY_PATH) -> None:
    data = {
        "wells": {name: {k: v for k, v in asdict(w).items() if k != "name"}
                  for name, w in wells.items()},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _build_watersheds(wells: Dict[str, WellInfo]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for name, w in wells.items():
        out.setdefault(w.watershed, []).append(name)
    # 정렬해서 안정적 순서 유지
    return {k: sorted(v) for k, v in sorted(out.items())}


# ---------------------------------------------------------------------------
# 모듈 레벨 — 항상 최신 JSON 반영
# ---------------------------------------------------------------------------
WELLS: Dict[str, WellInfo] = _load_from_json()
WATERSHEDS: Dict[str, List[str]] = _build_watersheds(WELLS)


def reload() -> None:
    """JSON 다시 읽기. 외부에서 파일 직접 편집 시 호출."""
    global WELLS, WATERSHEDS
    WELLS = _load_from_json()
    WATERSHEDS = _build_watersheds(WELLS)


# ---------------------------------------------------------------------------
# CRUD API
# ---------------------------------------------------------------------------
def add_well(
    name: str, lat: float, lon: float, watershed: str,
    aquifer: str = "bedrock", nearest_kma: int = 135,
    overwrite: bool = False,
) -> WellInfo:
    """새 관정 등록. overwrite=False 이면 중복 시 ValueError."""
    if not name.strip():
        raise ValueError("관정명이 비어 있음")
    if name in WELLS and not overwrite:
        raise ValueError(f"이미 등록된 관정: {name}")
    info = WellInfo(
        name=name, lat=float(lat), lon=float(lon),
        watershed=watershed.strip() or "기본",
        aquifer=aquifer if aquifer in ("alluvial", "bedrock") else "bedrock",
        nearest_kma=int(nearest_kma),
    )
    WELLS[name] = info
    _save_to_json(WELLS)
    reload()
    return info


def remove_well(name: str) -> bool:
    """관정 삭제. 성공 True."""
    if name not in WELLS:
        return False
    del WELLS[name]
    _save_to_json(WELLS)
    reload()
    return True


def update_well(name: str, **changes) -> WellInfo:
    """기존 관정 일부 필드 갱신.

    예: update_well("김천남면", watershed="감천하류", aquifer="bedrock")
    """
    if name not in WELLS:
        raise KeyError(f"등록되지 않은 관정: {name}")
    cur = asdict(WELLS[name])
    for k, v in changes.items():
        if k in cur and k != "name":
            cur[k] = v
    cur["name"] = name
    WELLS[name] = WellInfo(**cur)
    _save_to_json(WELLS)
    reload()
    return WELLS[name]


def rename_watershed(old: str, new: str) -> int:
    """유역명 일괄 변경. 변경된 관정 수 반환."""
    if not new.strip():
        raise ValueError("새 유역명이 비어 있음")
    n = 0
    for name, w in list(WELLS.items()):
        if w.watershed == old:
            WELLS[name] = WellInfo(**{**asdict(w), "watershed": new.strip()})
            n += 1
    if n > 0:
        _save_to_json(WELLS)
        reload()
    return n


# ---------------------------------------------------------------------------
# 조회 헬퍼
# ---------------------------------------------------------------------------
def get_well(name: str) -> WellInfo:
    if name not in WELLS:
        raise KeyError(f"Unknown well: {name}.  Registered: {list(WELLS)}")
    return WELLS[name]


def wells_in_watershed(ws: str) -> List[WellInfo]:
    if ws not in WATERSHEDS:
        raise KeyError(f"Unknown watershed: {ws}.  Registered: {list(WATERSHEDS)}")
    return [WELLS[n] for n in WATERSHEDS[ws]]
