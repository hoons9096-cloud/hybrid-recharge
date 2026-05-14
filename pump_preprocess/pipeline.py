"""
pipeline.py — 전처리 + Kalman WTF 통합 파이프라인
=====================================================
전체 워크플로우:
    1. 데이터 로드 및 검증
    2. 펌핑 탐지 (PumpingDetector)
    3. 수위 보정 (WaterLevelCorrector)
    4. Kalman WTF 실행 (AugmentedKalmanWTF)
    5. 결과 집계 및 비교 (전처리 전/후)

사용 예시:
    from pipeline import PumpWTFPipeline
    pipe = PumpWTFPipeline()
    result = pipe.run("data/sample.csv")
    pipe.print_summary(result)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List, Tuple
import time

from preprocess.detector import PumpingDetector, DetectionResult
from preprocess.corrector import WaterLevelCorrector, CorrectionResult
from kalman.wtf_kalman import AugmentedKalmanWTF, KalmanWTFResult
from kalman.soil_identifier import SoilIdentifier


@dataclass
class PipelineResult:
    # 입력
    dates: pd.DatetimeIndex
    raw_wl: np.ndarray
    rainfall: np.ndarray

    # 전처리
    detection: DetectionResult
    correction: CorrectionResult
    corrected_wl: np.ndarray

    # Kalman WTF — 원본 수위 적용
    result_raw: KalmanWTFResult

    # Kalman WTF — 보정 수위 적용
    result_corrected: KalmanWTFResult

    # 비교 지표
    improvement: dict

    # 토양 식별
    top3_soils: list
    ensemble_rech: dict

    # 소요 시간
    elapsed_sec: float


class PumpWTFPipeline:
    """
    Parameters
    ----------
    detect_methods : list
        펌핑 탐지 방법 조합. ["sigma", "rolling_baseline", "fourier"]
    correction_strategy : str
        보정 전략. "auto" | "recession_fill" | "spline_fill" | "baseline_shift"
    soil_num : int or None
        고정 토양 번호 (None = 자동 탐색)
    auto_optimize : bool
        Kalman 매개변수 자동 최적화 여부
    sigma_drop : float
        급강하 탐지 임계값 승수
    known_pump_times : list of (str, str), optional
        알려진 펌핑 기간. 예: [("2021-06-01", "2021-06-30")]
    """

    def __init__(
        self,
        detect_methods: List[str] = None,
        correction_strategy: str = "auto",
        soil_num: Optional[int] = None,
        auto_optimize: bool = True,
        sigma_drop: float = 2.5,
        known_pump_times: Optional[List[Tuple]] = None,
    ):
        self.detect_methods = detect_methods or ["sigma", "rolling_baseline"]
        self.correction_strategy = correction_strategy
        self.soil_num = soil_num
        self.auto_optimize = auto_optimize
        self.sigma_drop = sigma_drop
        self.known_pump_times = known_pump_times

    # ─────────────────────────────────────────
    def run(
        self,
        source,
        date_col: str = "date",
        wl_col: str = "water_level",
        rain_col: str = "rainfall",
        true_rech_col: Optional[str] = None,
    ) -> PipelineResult:
        """
        Parameters
        ----------
        source : str (CSV 경로) 또는 pd.DataFrame
        date_col, wl_col, rain_col : 컬럼명
        true_rech_col : 합성 검증용 실제 함양 컬럼 (선택)
        """
        t0 = time.time()
        print("=" * 60)
        print("  Pump Pre-processing + Kalman WTF Pipeline")
        print("=" * 60)

        # ── 1. 데이터 로드 ──
        df = self._load_data(source, date_col, wl_col, rain_col, true_rech_col)
        dates = df.index
        raw_wl  = df["water_level"].values
        rainfall = df["rainfall"].values
        true_rech = df["true_recharge"].values if "true_recharge" in df.columns else None

        print(f"\n[1] Data loaded: {len(dates)} days  "
              f"({dates[0].date()} ~ {dates[-1].date()})")
        print(f"    WL range:  {np.nanmin(raw_wl):.3f} ~ {np.nanmax(raw_wl):.3f} m")
        print(f"    Rainfall:  {np.nansum(rainfall):.1f} mm total")

        # ── 2. 펌핑 탐지 ──
        print(f"\n[2] Pumping Detection  (methods: {self.detect_methods})")
        detector = PumpingDetector(
            methods=self.detect_methods,
            sigma_drop=self.sigma_drop,
        )
        detection = detector.detect(
            dates, raw_wl, rainfall,
            known_pump_times=self.known_pump_times,
        )
        print(f"    Pumping days detected: {detection.n_pump_days} "
              f"({detection.pump_fraction * 100:.1f}% of record)")
        print(f"    Events: {len(detection.drop_events)}")
        if detection.dominant_period:
            print(f"    Dominant pumping period: {detection.dominant_period:.0f} days")

        # ── 3. 수위 보정 ──
        print(f"\n[3] Water Level Correction  (strategy: {self.correction_strategy})")
        corrector = WaterLevelCorrector(strategy=self.correction_strategy)
        correction = corrector.correct(dates, raw_wl, detection.pump_mask, rainfall)
        corrected_wl = correction.corrected_wl
        print(f"    Strategy applied: {correction.strategy_used}")
        print(f"    Days corrected: {correction.diagnostics.get('total_filled_days', 0)}")
        print(f"    Recession k: {correction.recession_k:.4f}"
              if correction.recession_k else "")

        # ── 4. Kalman WTF — 원본 수위 ──
        print(f"\n[4] Kalman WTF  (raw water level, no preprocessing)")
        kalman_raw = AugmentedKalmanWTF(
            soil_num=self.soil_num,
            auto_optimize=self.auto_optimize,
            exclude_pump_from_kalman=False,  # 전처리 없이
        )
        result_raw = kalman_raw.run(raw_wl, rainfall, None, true_rech)
        print(f"    Soil: {result_raw.best_soil_name}  "
              f"| RMSE: {result_raw.rmse:.4f} m  "
              f"| NSE: {result_raw.nse:.3f}  "
              f"| Recharge: {result_raw.rech_rate_pct:.1f}%")

        # ── 5. Kalman WTF — 보정 수위 ──
        # 보정 수위는 이미 펌핑 신호가 제거된 상태이므로
        # exclude_pump_from_kalman=False + pump_mask=None 으로 실행.
        # (exclude=True + pump_mask 동시 전달 시 이중처벌 → h_sim 하방 발산)
        print(f"\n[5] Kalman WTF  (corrected water level, no mask exclusion)")
        kalman_corr = AugmentedKalmanWTF(
            soil_num=self.soil_num,
            auto_optimize=self.auto_optimize,
            exclude_pump_from_kalman=False,  # 보정 완료 신호 → 전 구간 갱신
        )
        result_corrected = kalman_corr.run(
            corrected_wl, rainfall, None, true_rech  # pump_mask=None
        )
        print(f"    Soil: {result_corrected.best_soil_name}  "
              f"| RMSE: {result_corrected.rmse:.4f} m  "
              f"| NSE: {result_corrected.nse:.3f}  "
              f"| Recharge: {result_corrected.rech_rate_pct:.1f}%")

        # ── 6. 비교 및 개선도 ──
        improvement = self._compute_improvement(result_raw, result_corrected)
        print(f"\n[6] Improvement Summary")
        print(f"    RMSE:     {result_raw.rmse:.4f} → {result_corrected.rmse:.4f} m  "
              f"({'▼' if improvement['rmse_delta'] < 0 else '▲'}{abs(improvement['rmse_delta']):.4f})")
        print(f"    NSE:      {result_raw.nse:.3f} → {result_corrected.nse:.3f}  "
              f"({'▲' if improvement['nse_delta'] > 0 else '▼'}{abs(improvement['nse_delta']):.3f})")
        print(f"    Recharge: {result_raw.rech_rate_pct:.1f}% → {result_corrected.rech_rate_pct:.1f}%")
        if true_rech is not None:
            print(f"    Rech Bias: {result_raw.rech_bias_pct:+.1f}% → {result_corrected.rech_bias_pct:+.1f}%")

        # ── 7. 토양 앙상블 ──
        top3 = SoilIdentifier.top_k(result_corrected.soil_scores, k=3)
        ensemble = {"note": "Run with top-3 recharges for ensemble"}

        elapsed = time.time() - t0
        print(f"\n✓ Pipeline completed in {elapsed:.1f}s")
        print("=" * 60)

        return PipelineResult(
            dates=dates,
            raw_wl=raw_wl,
            rainfall=rainfall,
            detection=detection,
            correction=correction,
            corrected_wl=corrected_wl,
            result_raw=result_raw,
            result_corrected=result_corrected,
            improvement=improvement,
            top3_soils=top3,
            ensemble_rech=ensemble,
            elapsed_sec=elapsed,
        )

    # ─────────────────────────────────────────
    def _load_data(self, source, date_col, wl_col, rain_col, true_rech_col):
        if isinstance(source, pd.DataFrame):
            df = source.copy()
        else:
            df = pd.read_csv(source, parse_dates=[date_col])

        df = df.set_index(date_col).sort_index()
        df = df.rename(columns={wl_col: "water_level", rain_col: "rainfall"})
        if true_rech_col and true_rech_col in df.columns:
            df = df.rename(columns={true_rech_col: "true_recharge"})

        df["water_level"] = pd.to_numeric(df["water_level"], errors="coerce")
        df["rainfall"]    = pd.to_numeric(df["rainfall"], errors="coerce").fillna(0)

        return df[["water_level", "rainfall"] +
                  (["true_recharge"] if "true_recharge" in df.columns else [])]

    def _compute_improvement(
        self,
        raw: KalmanWTFResult,
        corr: KalmanWTFResult,
    ) -> dict:
        rmse_imp = (raw.rmse - corr.rmse) / raw.rmse * 100 if raw.rmse > 0 else 0
        return {
            "rmse_delta": corr.rmse - raw.rmse,
            "rmse_improvement_pct": rmse_imp,
            "nse_delta": corr.nse - raw.nse,
            "rech_delta_pct": corr.rech_rate_pct - raw.rech_rate_pct,
            "bias_delta": corr.rech_bias_pct - raw.rech_bias_pct,
        }

    def print_summary(self, result: PipelineResult):
        """최종 결과 요약 출력"""
        print("\n" + "=" * 60)
        print("  FINAL RESULTS SUMMARY")
        print("=" * 60)
        print(f"\n▶ Site period: {result.dates[0].date()} ~ {result.dates[-1].date()}")
        print(f"▶ Pumping contamination: {result.detection.pump_fraction * 100:.1f}% of record")
        print(f"\n  {'Metric':<25} {'Without Preproc':>15} {'With Preproc':>15} {'Change':>10}")
        print("  " + "-" * 65)
        r0, r1 = result.result_raw, result.result_corrected
        rows = [
            ("RMSE (m)",       f"{r0.rmse:.4f}",         f"{r1.rmse:.4f}",         f"{r1.rmse-r0.rmse:+.4f}"),
            ("NSE",            f"{r0.nse:.3f}",           f"{r1.nse:.3f}",           f"{r1.nse-r0.nse:+.3f}"),
            ("CC",             f"{r0.cc:.3f}",            f"{r1.cc:.3f}",            f"{r1.cc-r0.cc:+.3f}"),
            ("Recharge (%)",   f"{r0.rech_rate_pct:.1f}", f"{r1.rech_rate_pct:.1f}", f"{r1.rech_rate_pct-r0.rech_rate_pct:+.1f}"),
            ("Best Soil",      r0.best_soil_name,         r1.best_soil_name,         ""),
        ]
        for name, v0, v1, delta in rows:
            print(f"  {name:<25} {v0:>15} {v1:>15} {delta:>10}")

        print(f"\n▶ Top-3 Soils (with preprocessing):")
        for s in result.top3_soils:
            print(f"   #{s['rank']} {s['soil_name']:<18} score={s['score']:.1f}")
