"""
generate_sample.py — 합성 펌핑 오염 수위 데이터 생성기
=======================================================
물리 기반 시뮬레이션:
  1. Sy(토양) × 강우 이벤트 → 실제 함양
  2. 지수 감쇠 기저유출 모델 → 자연 수위 시계열
  3. 무작위 펌핑 이벤트 삽입 (지수 감쇠 급하강 + 회복)
  4. 측정 잡음 추가

단독 실행:
    python generate_sample.py
    python generate_sample.py --n-days 730 --pump-fraction 0.20 --out my_data.csv
"""

import numpy as np
import pandas as pd
import argparse
import os


class SyntheticPumpData:
    """
    Parameters
    ----------
    seed : int
        난수 시드
    sy : float
        비수율 (0.05 ~ 0.25)
    k_recession : float
        기저유출 감쇠계수 (음수, 기본 -0.02/day)
    noise_std : float
        측정 잡음 표준편차 (m)
    rain_mean : float
        일평균 강우 (mm/day, wet-day)
    rain_prob : float
        강우 발생 확률 (일)
    """

    def __init__(
        self,
        seed: int = 42,
        sy: float = 0.08,
        k_recession: float = -0.018,
        noise_std: float = 0.03,
        rain_mean: float = 12.0,
        rain_prob: float = 0.18,
    ):
        self.rng = np.random.default_rng(seed)
        self.sy = sy
        self.k_recession = k_recession
        self.noise_std = noise_std
        self.rain_mean = rain_mean
        self.rain_prob = rain_prob

    # ─────────────────────────────────────────
    def generate(
        self,
        n_days: int = 1095,
        pump_fraction: float = 0.12,
        include_true_recharge: bool = True,
        start_date: str = "2020-01-01",
    ) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame
            date, water_level, rainfall [, true_recharge]
        """
        dates = pd.date_range(start_date, periods=n_days, freq="D")

        # 1. 강우 생성
        rainfall = self._generate_rainfall(n_days)

        # 2. 자연 수위 + 실제 함양
        wl_natural, true_rech = self._generate_natural_wl(rainfall, n_days)

        # 3. 펌핑 이벤트 삽입
        wl_pumped, pump_true_mask = self._insert_pumping(
            wl_natural.copy(), n_days, pump_fraction
        )

        # 4. 측정 잡음
        noise = self.rng.normal(0, self.noise_std, n_days)
        wl_obs = wl_pumped + noise

        df = pd.DataFrame({
            "date": dates,
            "water_level": np.round(wl_obs, 4),
            "rainfall": np.round(rainfall, 2),
        })
        if include_true_recharge:
            df["true_recharge"] = np.round(true_rech, 3)

        # 메타 정보 (검증용)
        actual_pump_frac = pump_true_mask.sum() / n_days
        print(f"    [Synthetic] n={n_days}d | "
              f"pump_frac={actual_pump_frac*100:.1f}% | "
              f"total_rain={rainfall.sum():.0f}mm | "
              f"total_rech={true_rech.sum():.0f}mm "
              f"({true_rech.sum()/rainfall.sum()*100:.1f}%)")

        return df

    # ─────────────────────────────────────────
    def _generate_rainfall(self, n: int) -> np.ndarray:
        """감마 분포 기반 강우 이벤트"""
        rain = np.zeros(n)
        wet_days = self.rng.random(n) < self.rain_prob
        n_wet = wet_days.sum()
        if n_wet > 0:
            # 감마 분포: 평균=rain_mean, 형태 집중
            rain[wet_days] = self.rng.gamma(
                shape=1.5,
                scale=self.rain_mean / 1.5,
                size=n_wet,
            )
        # 계절성 — 여름 강우 집중 (한국 기후 모사)
        t = np.arange(n)
        seasonal = 1.0 + 0.8 * np.sin(2 * np.pi * (t - 150) / 365.25)
        rain *= np.maximum(0.2, seasonal)
        return rain

    # ─────────────────────────────────────────
    def _generate_natural_wl(
        self, rainfall: np.ndarray, n: int
    ):
        """
        물 균형 모델:
          dh/dt = R(t)/Sy - k_drain × (h - h_eq)
          R(t): 이벤트 기반 함양 (강우의 일부)
        """
        wl = np.zeros(n)
        rech = np.zeros(n)

        # 초기 수위
        h0 = 10.0
        h_eq = 8.5    # 평형 수위 (m)
        k_drain = abs(self.k_recession) * 0.5  # 1/day

        # 함양 지연 (이동 평균으로 근사)
        lag = 5  # days
        eff_rain = np.convolve(rainfall, np.ones(lag) / lag, mode="same")

        # 함양률: 강우의 8~20% (van Genuchten soil 계열)
        rech_frac = 0.12
        h = h0

        for t in range(n):
            r_t = eff_rain[t] * rech_frac
            rech[t] = r_t
            # 오일러 적분
            h = h + r_t / self.sy - k_drain * (h - h_eq)
            wl[t] = h

        return wl, rech

    # ─────────────────────────────────────────
    def _insert_pumping(
        self, wl: np.ndarray, n: int, pump_fraction: float
    ):
        """
        펌핑 이벤트 삽입:
          - 급격한 수위 하강 (지수 감쇠)
          - 이벤트 길이: 5~30일
          - 이벤트 강도: 0.3~1.5m 급락
        """
        pump_mask = np.zeros(n, dtype=bool)
        target_days = int(n * pump_fraction)
        total_pumped = 0

        # 평균 이벤트 길이
        mean_event_len = 12
        min_gap = 20  # 이벤트 간 최소 간격 (일)

        attempts = 0
        last_end = -min_gap

        while total_pumped < target_days and attempts < 300:
            attempts += 1

            # 이벤트 시작 시점
            lo = last_end + min_gap
            hi = n - mean_event_len * 2
            if lo >= hi:
                break
            start = int(self.rng.integers(lo, hi))
            if start >= n - 5:
                continue

            # 이벤트 길이
            ev_len = int(self.rng.exponential(mean_event_len))
            ev_len = max(3, min(ev_len, 45))
            end = min(start + ev_len, n)

            # 중복 방지
            if pump_mask[start:end].any():
                continue

            # 급하강 크기
            drop_mag = float(self.rng.uniform(0.3, 1.5))
            k_pump = float(self.rng.uniform(0.08, 0.25))  # 급강하 속도

            # 원래 수위 기준으로 하강 적용
            wl_before = wl[start]
            for i, idx in enumerate(range(start, end)):
                t_rel = i + 1
                pump_drawdown = drop_mag * (1 - np.exp(-k_pump * t_rel))
                wl[idx] = wl_before - pump_drawdown

            # 회복 (이벤트 종료 후)
            recovery_len = int(ev_len * 0.6)
            for i in range(recovery_len):
                idx = end + i
                if idx >= n:
                    break
                alpha = (i + 1) / (recovery_len + 1)
                target = wl[end - 1] + (wl[min(end + recovery_len, n - 1)] - wl[end - 1]) * alpha
                wl[idx] = (1 - alpha) * wl[idx] + alpha * target

            pump_mask[start:end] = True
            total_pumped += (end - start)
            last_end = end

        return wl, pump_mask


# ─────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="합성 펌핑 오염 수위 데이터 생성")
    p.add_argument("--n-days",        type=int,   default=1095)
    p.add_argument("--pump-fraction", type=float, default=0.12)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--out",           type=str,   default="sample_pumping.csv")
    p.add_argument("--no-true-rech",  action="store_true")
    args = p.parse_args()

    gen = SyntheticPumpData(seed=args.seed)
    df = gen.generate(
        n_days=args.n_days,
        pump_fraction=args.pump_fraction,
        include_true_recharge=not args.no_true_rech,
    )

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out)
    df.to_csv(out_path, index=False)
    print(f"  → Saved: {out_path}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
