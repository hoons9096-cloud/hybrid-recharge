"""
soil_identifier.py — 독립 토양 식별 유틸리티
Top-K 토양 목록, 앙상블 함양 추정
"""
import numpy as np
from .wtf_kalman import SOIL_NAMES, RECH_RANGE


class SoilIdentifier:
    """
    AugmentedKalmanWTF 결과에서 토양 식별 정보를 추가 분석합니다.
    """

    @staticmethod
    def top_k(scores: np.ndarray, k: int = 3):
        """상위 k개 토양 목록 반환"""
        idx = np.argsort(scores)[::-1][:k]
        return [
            {
                "rank": i + 1,
                "soil_num": int(idx[i]) + 1,
                "soil_name": SOIL_NAMES[idx[i]],
                "score": float(scores[idx[i]]),
            }
            for i in range(len(idx))
        ]

    @staticmethod
    def ensemble_recharge(
        top_soils: list,
        rech_totals: list,
        rainfall_total: float,
    ) -> dict:
        """
        상위 3개 토양의 가중 평균 함양 추정 (불확실성 정량화)
        rech_totals: 각 토양별 total recharge (mm)
        """
        weights = np.array([s["score"] for s in top_soils])
        weights = weights / weights.sum()

        rech_arr = np.array(rech_totals)
        mean_rech = float(np.average(rech_arr, weights=weights))
        std_rech = float(np.sqrt(np.average((rech_arr - mean_rech) ** 2, weights=weights)))
        rech_rate = mean_rech / rainfall_total * 100 if rainfall_total > 0 else 0.0

        return {
            "ensemble_recharge_mm": mean_rech,
            "ensemble_std_mm": std_rech,
            "ensemble_rate_pct": rech_rate,
            "ci_lower_pct": (mean_rech - 1.96 * std_rech) / rainfall_total * 100 if rainfall_total > 0 else 0,
            "ci_upper_pct": (mean_rech + 1.96 * std_rech) / rainfall_total * 100 if rainfall_total > 0 else 0,
        }

    @staticmethod
    def k_overlap_group(soil_num: int) -> str:
        """k-값 중복 그룹 레이블"""
        groups = {
            frozenset([1, 2]): "[1, 2]",
            frozenset([3, 11]): "[3, 11]",
            frozenset([4, 5]): "[4, 5]",
            frozenset([6, 7, 9]): "[6, 7, 9]",
            frozenset([8, 10]): "[8, 10]",
            frozenset([12]): "[12]",
        }
        for grp, label in groups.items():
            if soil_num in grp:
                return label
        return f"[{soil_num}]"
