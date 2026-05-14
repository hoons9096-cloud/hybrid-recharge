"""
main.py — Pump Pre-processing + Kalman WTF CLI Entry Point
===========================================================
사용법:
    python main.py --input data/sample_pumping.csv --plot
    python main.py --input data/sample_pumping.csv \\
        --methods sigma pelt --strategy recession_fill \\
        --soil 3 --no-optimize --plot --output results/

    # 합성 데이터 생성 후 즉시 실행:
    python main.py --generate --n-days 1460 --pump-fraction 0.15 --plot
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

# 경로 설정
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import PumpWTFPipeline


# ─────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pump_wtf",
        description="Pump-contaminated groundwater recharge estimation via Kalman WTF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # CSV 파일 직접 실행
  python main.py --input data/sample_pumping.csv --plot

  # 합성 데이터 생성 + 실행
  python main.py --generate --pump-fraction 0.20 --plot

  # 고급 옵션
  python main.py --input mydata.csv \\
      --date-col date --wl-col wl_m --rain-col rain_mm \\
      --methods sigma pelt fourier \\
      --strategy auto \\
      --sigma-drop 3.0 \\
      --soil 5 \\
      --output ./results --plot
        """,
    )

    # ── 입력 ──────────────────────────────────────────────
    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument(
        "--input", "-i",
        metavar="CSV",
        help="입력 CSV 파일 경로 (date, water_level, rainfall 컬럼 필요)",
    )
    inp.add_argument(
        "--generate", "-g",
        action="store_true",
        help="합성 데이터를 생성하여 즉시 실행",
    )

    # ── 컬럼명 ────────────────────────────────────────────
    p.add_argument("--date-col",  default="date",        metavar="COL")
    p.add_argument("--wl-col",    default="water_level", metavar="COL")
    p.add_argument("--rain-col",  default="rainfall",    metavar="COL")
    p.add_argument("--true-rech", default=None,          metavar="COL",
                   help="검증용 실제 함양 컬럼명 (선택)")

    # ── 탐지 설정 ─────────────────────────────────────────
    det = p.add_argument_group("Detection")
    det.add_argument(
        "--methods", nargs="+",
        choices=["sigma", "rolling_baseline", "pelt", "fourier"],  # pelt: deprecated alias
        default=["sigma", "rolling_baseline"],
        help="펌핑 탐지 방법 (복수 선택 가능, default: sigma rolling_baseline). "
             "'pelt'는 'rolling_baseline'의 deprecated alias.",
    )
    det.add_argument(
        "--sigma-drop", type=float, default=2.5,
        metavar="MULT",
        help="급강하 탐지 임계값 승수 (default: 2.5)",
    )
    det.add_argument(
        "--known-pump", nargs="+", metavar="YYYY-MM-DD",
        help="알려진 펌핑 기간 쌍: --known-pump 2021-06-01 2021-06-30 2022-01-01 2022-01-15",
    )

    # ── 보정 설정 ─────────────────────────────────────────
    cor = p.add_argument_group("Correction")
    cor.add_argument(
        "--strategy",
        choices=["auto", "recession_fill", "spline_fill", "baseline_shift"],
        default="auto",
        help="보정 전략 (default: auto)",
    )

    # ── Kalman 설정 ───────────────────────────────────────
    kal = p.add_argument_group("Kalman WTF")
    kal.add_argument(
        "--soil", type=int, default=None,
        metavar="N",
        help="고정 토양 번호 1-12 (None = 자동 탐색)",
    )
    kal.add_argument(
        "--no-optimize", action="store_true",
        help="Kalman 매개변수 자동 최적화 비활성화",
    )

    # ── 합성 데이터 옵션 ──────────────────────────────────
    syn = p.add_argument_group("Synthetic Data (--generate)")
    syn.add_argument("--n-days",        type=int,   default=1095, metavar="N")
    syn.add_argument("--pump-fraction", type=float, default=0.12, metavar="F",
                     help="펌핑 오염 비율 0~1 (default: 0.12)")
    syn.add_argument("--seed",          type=int,   default=42)
    syn.add_argument("--true-recharge", action="store_true",
                     help="합성 데이터에 실제 함양 컬럼 포함")

    # ── 출력 ──────────────────────────────────────────────
    out = p.add_argument_group("Output")
    out.add_argument(
        "--output", "-o", default=None, metavar="DIR",
        help="결과 저장 폴더 (미지정 시 콘솔만 출력)",
    )
    out.add_argument(
        "--plot", action="store_true",
        help="결과 그래프 저장 (--output 필요, 미지정 시 현재 폴더)",
    )
    out.add_argument(
        "--save-csv", action="store_true",
        help="보정된 수위 및 추정 함양을 CSV로 저장",
    )
    out.add_argument(
        "--no-summary", action="store_true",
        help="최종 요약 표 출력 생략",
    )

    return p


# ─────────────────────────────────────────────────────────
def parse_known_pump_times(raw: list):
    """--known-pump 2021-06-01 2021-06-30 2022-01-01 2022-01-15 → list of tuples"""
    if not raw or len(raw) % 2 != 0:
        if raw:
            print(f"[WARNING] --known-pump 인수는 쌍으로 입력해야 합니다. 무시됩니다.")
        return None
    return [(raw[i], raw[i + 1]) for i in range(0, len(raw), 2)]


# ─────────────────────────────────────────────────────────
def generate_sample(n_days, pump_fraction, seed, include_true_recharge):
    """합성 펌핑-오염 수위 데이터 생성"""
    from data.generate_sample import SyntheticPumpData
    gen = SyntheticPumpData(seed=seed)
    df = gen.generate(
        n_days=n_days,
        pump_fraction=pump_fraction,
        include_true_recharge=include_true_recharge,
    )
    print(f"[Synthetic] Generated {n_days}-day record  "
          f"(pump fraction target: {pump_fraction*100:.0f}%)")
    return df


# ─────────────────────────────────────────────────────────
def save_results(result, out_dir: str, save_csv: bool):
    os.makedirs(out_dir, exist_ok=True)

    if save_csv:
        df_out = pd.DataFrame(
            {
                "date": result.dates,
                "raw_wl": result.raw_wl,
                "corrected_wl": result.corrected_wl,
                "pump_mask": result.detection.pump_mask.astype(int),
                "rainfall": result.rainfall,
                "recharge_raw": result.result_raw.recharge,
                "recharge_corrected": result.result_corrected.recharge,
            }
        )
        csv_path = os.path.join(out_dir, "pipeline_output.csv")
        df_out.to_csv(csv_path, index=False)
        print(f"  [CSV saved] {csv_path}")


# ─────────────────────────────────────────────────────────
def main():
    parser = build_parser()
    args = parser.parse_args()

    # ── 소스 결정 ──
    if args.generate:
        source = generate_sample(
            n_days=args.n_days,
            pump_fraction=args.pump_fraction,
            seed=args.seed,
            include_true_recharge=args.true_recharge,
        )
        true_rech_col = "true_recharge" if args.true_recharge else None
    else:
        if not os.path.exists(args.input):
            print(f"[ERROR] 파일을 찾을 수 없습니다: {args.input}")
            sys.exit(1)
        source = args.input
        true_rech_col = args.true_rech

    # ── known_pump_times 파싱 ──
    known_pump_times = parse_known_pump_times(args.known_pump) if args.known_pump else None

    # ── 파이프라인 실행 ──
    pipeline = PumpWTFPipeline(
        detect_methods=args.methods,
        correction_strategy=args.strategy,
        soil_num=args.soil,
        auto_optimize=not args.no_optimize,
        sigma_drop=args.sigma_drop,
        known_pump_times=known_pump_times,
    )

    if args.generate:
        result = pipeline.run(
            source,
            true_rech_col=true_rech_col,
        )
    else:
        result = pipeline.run(
            source,
            date_col=args.date_col,
            wl_col=args.wl_col,
            rain_col=args.rain_col,
            true_rech_col=true_rech_col,
        )

    # ── 요약 출력 ──
    if not args.no_summary:
        pipeline.print_summary(result)

    # ── 결과 저장 ──
    out_dir = args.output or "."
    if args.save_csv or args.output:
        save_results(result, out_dir, args.save_csv)

    # ── 시각화 ──
    if args.plot:
        try:
            from visualizer import PipelineVisualizer
            vis = PipelineVisualizer(out_dir=out_dir)
            vis.plot_all(result)
            print(f"\n  [Plots saved] → {out_dir}/")
        except ImportError as e:
            print(f"[WARNING] visualizer 로드 실패: {e}")
        except Exception as e:
            print(f"[WARNING] 그래프 생성 중 오류: {e}")

    print(f"\n  총 소요 시간: {result.elapsed_sec:.1f}s")
    return result


# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
