"""watershed_aggregator.py — 다중 관정 → 면적 가중 유역 함양율.

흐름:
  1. wells_registry.WELLS 에서 유역 내 관정 목록 읽기
  2. 각 관정에서 토양도 점 조회 → HSG → (sn_idx, Sy, texture_group)
  3. 각 관정 .txt → core_sim_v27 (WTF) + fao56_swb (FAO-56) 실행
  4. 유역 HSG 면적 분포로 면적 가중 함양율 계산 (Soil-weighted WTF)

Soil-weighted aggregation (논문 핵심 방법):
  R_watershed = Σ_h f_h · R_h
  where f_h = HSG h 의 면적 비율,  R_h = HSG h 를 가진 관정들의 평균 함양율
  (해당 HSG 관정이 없으면 인접 HSG 의 회귀 또는 전체 평균 fallback)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from wells_registry import WELLS, WATERSHEDS, WellInfo
from shp_soil_mapper import (
    query_point, watershed_profile_from_wells,
    HSG_TO_SY, HSG_TO_CN, HSG_TO_TEXTURE,
    SoilQuery, WatershedSoilProfile,
)


# ---------------------------------------------------------------------------
# HSG → USDA sn_idx (Carsel-Parrish 1988, soil_db.py 인덱스)
#
# WTF 의 Sy 는 *포화 대수층(스크린 위치)* 의 비산출률이므로 표면 토양 직접 적용은
# 부적합.  대수층 타입에 따라 두 매핑 테이블을 분리한다:
#
# 1. Bedrock 관정 — 표면 토양 ≈ 풍화대(saprolite) 토양 → 표면 HSG 직접 매핑
# 2. Alluvial 관정 — 충적 대수층은 모래/자갈 dominant 이지만 표면이 fines (HSG D)
#    이면 충적층 내에 silt/clay 베드 끼워 있을 가능성 高 → "그라데이션" 적용:
#       표면 A → sn=2 (Loamy Sand) — 가장 거친 충적
#       표면 B → sn=3 (Sandy Loam)
#       표면 C → sn=12 (Loam)
#       표면 D → sn=4 (Silt Loam) — Clay(sn=6) 직행은 충적에 부적합
# ---------------------------------------------------------------------------
HSG_TO_SN_BEDROCK = {
    "A": 2,    # Loamy Sand
    "B": 12,   # Loam
    "C": 10,   # Clay Loam
    "D": 6,    # Clay
}
HSG_TO_SN_ALLUVIAL = {
    "A": 2,    # Loamy Sand   — 거친 충적 (점사주, 자갈층)
    "B": 3,    # Sandy Loam
    "C": 12,   # Loam         — 일반 충적
    "D": 4,    # Silt Loam    — fines 우세 충적 (배후 습지, 점토베드)
}
# 하위호환 별칭
HSG_TO_SN = HSG_TO_SN_BEDROCK


def select_wtf_sn(aquifer: str, hydro_type: str) -> int:
    """관정 대수층 타입 + 표면 HSG → WTF 용 sn_idx 결정."""
    if aquifer == "alluvial":
        return HSG_TO_SN_ALLUVIAL.get(hydro_type, 12)
    return HSG_TO_SN_BEDROCK.get(hydro_type, 12)


# ---------------------------------------------------------------------------
# 결과 컨테이너
# ---------------------------------------------------------------------------
@dataclass
class WellRechargeResult:
    well: WellInfo
    soil: SoilQuery
    P_annual_mm: float
    wtf_pct: Optional[float] = None
    wtf_mm: Optional[float] = None
    fao_pct: Optional[float] = None
    fao_mm: Optional[float] = None
    sn_used: Optional[int] = None
    notes: str = ""


@dataclass
class WatershedRechargeResult:
    watershed: str
    profile: WatershedSoilProfile
    wells: List[WellRechargeResult] = field(default_factory=list)
    # 단순 산술평균 (Lumped)
    lumped_wtf_pct: Optional[float] = None
    lumped_fao_pct: Optional[float] = None
    # 면적 가중 (Soil-weighted) — 논문 제안 방법
    soil_weighted_wtf_pct: Optional[float] = None
    soil_weighted_fao_pct: Optional[float] = None
    # 메타
    P_annual_mm: Optional[float] = None
    method_notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 단일 관정 추정 — 기존 core_sim_v27 + fao56_swb 사용
# ---------------------------------------------------------------------------
def estimate_single_well(
    well: WellInfo,
    file_path: str,
    soil: Optional[SoilQuery] = None,
    run_fao: bool = True,
    fao_kwargs: Optional[Dict] = None,
) -> WellRechargeResult:
    """관정 1개의 WTF + (옵션) FAO-56 함양율.

    soil 이 None 이면 토양도 점 조회로 자동 결정.
    """
    if soil is None:
        soil = query_point(well.name, well.lat, well.lon)

    # WTF — alluvial 이면 sn=3 강제, bedrock 만 표면 HSG 매핑
    sn_idx = select_wtf_sn(well.aquifer, soil.hydro_type)
    from core_sim_v27 import core_sim_v27
    wtf = core_sim_v27(
        file_path=file_path,
        k_val=0.0, z_val=0.5, lag_val=0,
        sn_idx=sn_idx, q_val=1.0, r_val=1.0,
        rc_val=0.0, ignore_pump=0, sens_val=3.0,
        do_optimize=True,
    )
    wtf_ok = isinstance(wtf, dict) and not wtf.get("error")
    P_annual_mm = float("nan")
    wtf_pct = wtf_mm = None
    if wtf_ok:
        po = np.asarray(wtf.get("po_shifted", []), dtype=float)
        n_yr = max(len(po) / 365.25, 1.0) if len(po) else 1.0
        P_annual_mm = float(np.nansum(po)) * 1000.0 / n_yr if len(po) else 0.0
        wtf_pct = float(wtf.get("recharge_ratio", 0.0))
        wtf_mm = P_annual_mm * wtf_pct / 100.0

    fao_pct = fao_mm = None
    notes = "" if wtf_ok else f"WTF failed: {wtf.get('error','?')}"

    if run_fao and wtf_ok:
        # FAO-56 는 일별 P, T 가 필요하지만 .txt 에 T 가 없으므로
        # KMA adapter 로 보강.  여기서는 호출 측이 fao_kwargs 로 주입.
        if fao_kwargs:
            try:
                from fao56_swb import estimate_recharge_fao56
                fao = estimate_recharge_fao56(
                    texture_group=soil.texture_group,
                    **fao_kwargs,
                )
                fao_pct = float(fao.recharge_ratio_pct)
                fao_mm = float(fao.R_annual_mm)
            except Exception as e:
                notes += f" | FAO-56 failed: {e}"

    return WellRechargeResult(
        well=well, soil=soil,
        P_annual_mm=P_annual_mm,
        wtf_pct=wtf_pct, wtf_mm=wtf_mm,
        fao_pct=fao_pct, fao_mm=fao_mm,
        sn_used=sn_idx,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# 면적 가중 집계 (Soil-weighted)
# ---------------------------------------------------------------------------
def _soil_weighted(
    profile: WatershedSoilProfile,
    well_results: List[WellRechargeResult],
    field_name: str,
    sy_scale: bool = False,
) -> Optional[float]:
    """HSG 면적 비율로 가중 평균.

    각 HSG 의 대표값 = 그 HSG 토양에 위치한 관정들의 평균.
    관측 관정이 없는 HSG 의 대표값 추정:
      - sy_scale=True (WTF): 관측 관정의 'Sy 단위당 함양'(R/Sy)을 평균한 뒤
        대상 HSG 의 문헌 Sy 로 다시 스케일 → grid 방법(wtf_soil_weighted)과 동일.
        토양 구조를 반영하므로 grand-mean fallback 보다 이론적으로 타당.
      - sy_scale=False (FAO-56 등 Sy 무관 추정): 전체 관정 평균으로 fallback.
    """
    all_vals = [getattr(w, field_name) for w in well_results
                if getattr(w, field_name) is not None]
    if not all_vals:
        return None
    overall_mean = float(np.mean(all_vals))

    # WTF: 관측 관정의 Sy 정규화 함양 (R/Sy) 평균 — 미관측 HSG 추정용
    r_per_sy = None
    if sy_scale:
        ratios = [
            getattr(w, field_name) / HSG_TO_SY[w.soil.hydro_type]
            for w in well_results
            if getattr(w, field_name) is not None
            and HSG_TO_SY.get(w.soil.hydro_type, 0) > 0
        ]
        if ratios:
            r_per_sy = float(np.mean(ratios))

    weighted = 0.0
    weight_sum = 0.0
    for hsg, frac in profile.hsg_fractions.items():
        hsg_wells = [w for w in well_results if w.soil.hydro_type == hsg]
        hsg_vals = [getattr(w, field_name) for w in hsg_wells
                    if getattr(w, field_name) is not None]
        if hsg_vals:
            rep = float(np.mean(hsg_vals))
        elif r_per_sy is not None and HSG_TO_SY.get(hsg, 0) > 0:
            rep = r_per_sy * HSG_TO_SY[hsg]      # Sy 비율 스케일링 (grand mean 아님)
        else:
            rep = overall_mean
        weighted += frac * rep
        weight_sum += frac
    return weighted / weight_sum if weight_sum > 0 else None


# ---------------------------------------------------------------------------
# 유역 단위 추정
# ---------------------------------------------------------------------------
def estimate_watershed(
    watershed: str,
    file_paths: Optional[Dict[str, str]] = None,     # {well_name: file_path}
    run_fao: bool = False,
    fao_kwargs_per_well: Optional[Dict[str, Dict]] = None,
    buffer_km: float = 2.0,
    use_cached: bool = False,
) -> WatershedRechargeResult:
    """유역 1개의 다중 관정 → 면적 가중 함양율.

    use_cached=True 이면 well_results/{well_name}.json 에서 결과를 로드.
    file_paths 는 무시되고, 누락된 관정은 method_notes 에 기록.
    """
    if file_paths is None:
        file_paths = {}
    if watershed not in WATERSHEDS:
        raise KeyError(f"Unknown watershed: {watershed}")
    well_infos = [WELLS[n] for n in WATERSHEDS[watershed]]

    # 유역 토양 분포
    profile = watershed_profile_from_wells(
        watershed,
        [(w.name, w.lat, w.lon) for w in well_infos],
        buffer_km=buffer_km,
    )

    # 각 관정 추정
    well_results: List[WellRechargeResult] = []
    notes: List[str] = []
    for w in well_infos:
        if use_cached:
            # well_results 저장소에서 로드
            try:
                from well_results_store import load as load_stored
                stored = load_stored(w.name)
            except Exception as e:
                stored = None
                notes.append(f"{w.name}: cache load error — {e}")
            if stored is None:
                notes.append(f"{w.name}: 저장된 결과 없음 — Tab 1 에서 분석 후 저장 필요")
                continue
            soil = query_point(w.name, w.lat, w.lon)
            wr = WellRechargeResult(
                well=w, soil=soil,
                P_annual_mm=stored.P_annual_mm or 0.0,
                wtf_pct=stored.recharge_ratio_pct,
                wtf_mm=stored.wtf_mm,
                sn_used=stored.sn_idx,
                notes=f"cached @ {stored.analyzed_at}"
                      + (" (pump-corrected)" if stored.pump_corrected else ""),
            )
            well_results.append(wr)
            continue

        if w.name not in file_paths:
            notes.append(f"{w.name}: no file_path supplied — skipped")
            continue
        fk = (fao_kwargs_per_well or {}).get(w.name)
        try:
            wr = estimate_single_well(
                w, file_paths[w.name],
                run_fao=run_fao, fao_kwargs=fk,
            )
            well_results.append(wr)
        except Exception as e:
            notes.append(f"{w.name}: {e}")

    if not well_results:
        raise RuntimeError(f"{watershed}: no successful well estimates")

    # Lumped — 단순 평균
    wtf_vals = [w.wtf_pct for w in well_results if w.wtf_pct is not None]
    fao_vals = [w.fao_pct for w in well_results if w.fao_pct is not None]
    lumped_wtf = float(np.mean(wtf_vals)) if wtf_vals else None
    lumped_fao = float(np.mean(fao_vals)) if fao_vals else None

    # Soil-weighted — WTF 는 Sy 비율 스케일링, FAO-56 은 Sy 무관이라 평균 fallback
    sw_wtf = _soil_weighted(profile, well_results, "wtf_pct", sy_scale=True)
    sw_fao = _soil_weighted(profile, well_results, "fao_pct", sy_scale=False)

    P_vals = [w.P_annual_mm for w in well_results if np.isfinite(w.P_annual_mm)]
    P_mean = float(np.mean(P_vals)) if P_vals else None

    return WatershedRechargeResult(
        watershed=watershed,
        profile=profile,
        wells=well_results,
        lumped_wtf_pct=lumped_wtf,
        lumped_fao_pct=lumped_fao,
        soil_weighted_wtf_pct=sw_wtf,
        soil_weighted_fao_pct=sw_fao,
        P_annual_mm=P_mean,
        method_notes=notes,
    )
