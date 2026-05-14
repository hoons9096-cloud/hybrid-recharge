"""
scenarios.py -- 합성 벤치마크 시나리오 매트릭스 관리

시나리오별 도메인 및 합성 데이터를 일괄 생성하고 관리한다.
5개 시나리오(S1-S5)를 전부 또는 선택적으로 실행할 수 있다.

Scenario matrix (CLAUDE.md):
    S1: 균질,       높은 관측정, 낮은 노이즈
    S2: 약한 불균질, 높은 관측정, 낮은 노이즈
    S3: 강한 불균질, 높은 관측정, 낮은 노이즈
    S4: 강한 불균질, 낮은 관측정, 낮은 노이즈
    S5: 강한 불균질, 높은 관측정, 높은 노이즈

Usage:
    from synthetic.scenarios import run_all_scenarios, run_scenario
    results = run_all_scenarios()
    result = run_scenario("S3")
"""
from __future__ import annotations

import sys
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# 프로젝트 루트 경로 설정
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from synthetic.generate_domain import DomainConfig, SyntheticDomain, generate_domain
from synthetic.generate_data import generate_data


# ──────────────────────────────────────────────────────────
# 시나리오 이름 목록
# ──────────────────────────────────────────────────────────
SCENARIO_NAMES: List[str] = ["S1", "S2", "S3", "S4", "S5"]

# DomainConfig 클래스 메서드 매핑 (시나리오 이름 → 생성자)
_CONFIG_FACTORY: Dict[str, callable] = {
    "S1": DomainConfig.S1,
    "S2": DomainConfig.S2,
    "S3": DomainConfig.S3,
    "S4": DomainConfig.S4,
    "S5": DomainConfig.S5,
}


# ──────────────────────────────────────────────────────────
# 시나리오 결과 컨테이너
# ──────────────────────────────────────────────────────────
@dataclass
class ScenarioResult:
    """Container for a single scenario's generated domain and data.

    Parameters
    ----------
    name : str
        Scenario identifier (e.g., "S1", "S3").
    domain : SyntheticDomain
        Generated synthetic domain (soil map, wells, hydraulic properties).
    data : object or None
        Synthetic observation data (will be populated when generate_data.py
        is implemented). None until then.
    elapsed_sec : float
        Wall-clock time for domain + data generation [seconds].
    """
    name: str
    domain: SyntheticDomain
    data: object = None           # SyntheticData (generate_data.py 구현 후 교체)
    elapsed_sec: float = 0.0
    # 각 방법론 결과를 나중에 추가할 딕셔너리
    method_results: Dict[str, object] = field(default_factory=dict)

    def summary_line(self) -> str:
        """Return a one-line summary for table display."""
        cfg = self.domain.config
        sy = self.domain.Sy_map
        return (
            f"{self.name:>4s}  "
            f"{cfg.heterogeneity:<10s}  "
            f"{cfg.well_density:<6s}  "
            f"{cfg.obs_noise_std:>6.3f}  "
            f"{self.domain.n_wells:>5d}  "
            f"{sy.mean():>6.3f}  "
            f"[{sy.min():.3f}-{sy.max():.3f}]  "
            f"{self.elapsed_sec:>6.3f}s"
        )


# ──────────────────────────────────────────────────────────
# 시나리오 실행 함수
# ──────────────────────────────────────────────────────────
def run_scenario(name: str) -> ScenarioResult:
    """Generate domain (+ data when available) for a single scenario.

    Parameters
    ----------
    name : str
        Scenario name, one of SCENARIO_NAMES ("S1" through "S5").

    Returns
    -------
    ScenarioResult
        Container with generated domain and metadata.

    Raises
    ------
    ValueError
        If the scenario name is not recognized.
    """
    name = name.upper()
    if name not in _CONFIG_FACTORY:
        raise ValueError(
            f"Unknown scenario '{name}'. "
            f"Choose from: {', '.join(SCENARIO_NAMES)}"
        )

    t0 = time.perf_counter()

    # 도메인 생성
    config = _CONFIG_FACTORY[name]()
    domain = generate_domain(config)

    # 합성 데이터 생성
    data = generate_data(domain)

    elapsed = time.perf_counter() - t0

    return ScenarioResult(
        name=name,
        domain=domain,
        data=data,
        elapsed_sec=elapsed,
    )


def run_all_scenarios(
    names: Optional[List[str]] = None,
) -> List[ScenarioResult]:
    """Generate domain (+ data) for all or selected scenarios.

    Parameters
    ----------
    names : list of str, optional
        Subset of scenario names to run. Defaults to all 5 scenarios.

    Returns
    -------
    list of ScenarioResult
        Results for each requested scenario, in order.
    """
    if names is None:
        names = list(SCENARIO_NAMES)

    results = []
    for name in names:
        result = run_scenario(name)
        results.append(result)

    return results


# ──────────────────────────────────────────────────────────
# 요약 테이블 출력
# ──────────────────────────────────────────────────────────
def print_summary_table(results: List[ScenarioResult]) -> None:
    """Print a formatted summary table of scenario results.

    Parameters
    ----------
    results : list of ScenarioResult
        Scenario results to summarize.
    """
    # 헤더
    header = (
        f"{'Scen':>4s}  "
        f"{'Hetero.':<10s}  "
        f"{'Wells':<6s}  "
        f"{'Noise':>6s}  "
        f"{'nWell':>5s}  "
        f"{'Sy_mu':>6s}  "
        f"{'Sy_range':<15s}  "
        f"{'Time':>7s}"
    )
    sep = "-" * len(header)

    print()
    print("=" * len(header))
    print("  Synthetic Benchmark — Scenario Summary")
    print("=" * len(header))
    print(header)
    print(sep)

    for r in results:
        print(r.summary_line())

    print(sep)
    total_time = sum(r.elapsed_sec for r in results)
    print(f"  Total generation time: {total_time:.3f}s")
    print()


# ──────────────────────────────────────────────────────────
# CLI 실행
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 전체 시나리오 생성 및 요약 출력
    print("Generating all scenarios (S1-S5)...")
    results = run_all_scenarios()

    # 개별 도메인 요약 출력
    for r in results:
        print()
        print(r.domain.summary())

    # 요약 테이블
    print_summary_table(results)
