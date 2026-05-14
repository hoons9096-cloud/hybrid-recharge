"""well_results_store.py — 단일 관정 분석 결과 영구 저장소.

Tab 1 에서 분석한 결과를 JSON 으로 저장하면 Tab 10 (유역 함양율) 이
이를 읽어 면적 가중 집계에 사용한다.

저장 파일: well_results/{well_name}.json
형식 안정성을 위해 schema_version 필드 포함.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional


SCHEMA_VERSION = 1
DEFAULT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "well_results",
)


@dataclass
class StoredWellResult:
    """저장 가능한 단일 관정 분석 결과."""
    schema_version: int = SCHEMA_VERSION
    well_name: str = ""
    file_path: str = ""
    analyzed_at: str = ""

    # 결과 (Tab 10 이 사용)
    recharge_ratio_pct: Optional[float] = None
    P_annual_mm: Optional[float] = None
    wtf_mm: Optional[float] = None
    rmse: Optional[float] = None
    cc: Optional[float] = None

    # 사용된 파라미터 (재현용)
    sn_idx: Optional[int] = None
    soil_name: Optional[str] = None
    k_val: Optional[float] = None
    z_val: Optional[float] = None
    lag_val: Optional[int] = None
    Sy_eff: Optional[float] = None
    optimized: bool = False
    pump_corrected: bool = False

    # 메타 (Tab 10 매핑용)
    aquifer: Optional[str] = None       # "alluvial" | "bedrock"
    hydro_type: Optional[str] = None    # "A"/"B"/"C"/"D"
    soil_code: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    notes: str = ""

    # Bayesian Sy / 함양율 후행분포 (Phase 1)
    bayes_sy_post_mean: Optional[float] = None
    bayes_sy_post_sd: Optional[float] = None
    bayes_sy_post_lo95: Optional[float] = None
    bayes_sy_post_hi95: Optional[float] = None
    bayes_rech_pct_post_mean: Optional[float] = None
    bayes_rech_pct_post_lo95: Optional[float] = None
    bayes_rech_pct_post_hi95: Optional[float] = None
    bayes_n_eff: Optional[float] = None
    pump_test_sy: Optional[float] = None


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def _ensure_dir(path: str = DEFAULT_DIR) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save(result: StoredWellResult, store_dir: str = DEFAULT_DIR) -> str:
    """저장. 파일명 = {well_name}.json (덮어쓰기)."""
    if not result.well_name:
        raise ValueError("well_name 이 비어 있음")
    if not result.analyzed_at:
        result.analyzed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _ensure_dir(store_dir)
    path = os.path.join(store_dir, f"{result.well_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, ensure_ascii=False, indent=2)
    return path


def load(well_name: str, store_dir: str = DEFAULT_DIR) -> Optional[StoredWellResult]:
    """관정명으로 로드. 없으면 None."""
    path = os.path.join(store_dir, f"{well_name}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # schema_version 호환 처리 (현재는 v1 만)
    return StoredWellResult(**{k: v for k, v in data.items()
                                if k in StoredWellResult.__dataclass_fields__})


def list_stored(store_dir: str = DEFAULT_DIR) -> List[StoredWellResult]:
    """저장된 모든 결과 목록."""
    if not os.path.isdir(store_dir):
        return []
    out = []
    for fn in sorted(os.listdir(store_dir)):
        if fn.endswith(".json"):
            try:
                with open(os.path.join(store_dir, fn), "r", encoding="utf-8") as f:
                    data = json.load(f)
                out.append(StoredWellResult(
                    **{k: v for k, v in data.items()
                       if k in StoredWellResult.__dataclass_fields__}
                ))
            except Exception:
                continue
    return out


def delete(well_name: str, store_dir: str = DEFAULT_DIR) -> bool:
    """삭제. 성공 True, 없으면 False."""
    path = os.path.join(store_dir, f"{well_name}.json")
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


# ---------------------------------------------------------------------------
# Convenience: result_v27 dict → StoredWellResult
# ---------------------------------------------------------------------------
def from_result_v27(
    well_name: str,
    result_v27: Dict,
    file_path: str = "",
    sn_idx: Optional[int] = None,
    soil_name: Optional[str] = None,
    pump_corrected: bool = False,
    aquifer: Optional[str] = None,
    hydro_type: Optional[str] = None,
    soil_code: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    notes: str = "",
) -> StoredWellResult:
    """result_v27 dict 로부터 StoredWellResult 구성."""
    import numpy as np
    rr = float(result_v27.get("recharge_ratio_corrected")
               or result_v27.get("recharge_ratio") or 0.0)
    po = result_v27.get("po_shifted") or result_v27.get("po", [])
    po_arr = np.asarray(po, dtype=float)
    if len(po_arr) > 0:
        n_yr = max(len(po_arr) / 365.25, 1.0)
        P_annual = float(np.nansum(po_arr)) * 1000.0 / n_yr
    else:
        P_annual = 0.0
    return StoredWellResult(
        well_name=well_name,
        file_path=file_path,
        recharge_ratio_pct=rr,
        P_annual_mm=P_annual,
        wtf_mm=P_annual * rr / 100.0,
        rmse=float(result_v27.get("rmse", 0.0)) or None,
        cc=float(result_v27.get("cc", 0.0)) or None,
        sn_idx=int(sn_idx) if sn_idx is not None else None,
        soil_name=soil_name,
        k_val=float(result_v27["opt_k"]) if "opt_k" in result_v27 else None,
        z_val=float(result_v27["opt_z"]) if "opt_z" in result_v27 else None,
        lag_val=int(result_v27["opt_lag"]) if "opt_lag" in result_v27 else None,
        Sy_eff=float(result_v27.get("Sy_eff", 0.0)) or None,
        optimized=bool(result_v27.get("optimized", False))
                  or "opt_k" in result_v27,
        pump_corrected=pump_corrected,
        aquifer=aquifer,
        hydro_type=hydro_type,
        soil_code=soil_code,
        lat=lat, lon=lon,
        notes=notes,
    )
