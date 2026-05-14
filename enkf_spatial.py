"""
enkf_spatial.py  —  공간 지하수 함양 추정
                    (Ensemble Kalman Filter, R-state 방식)

핵심 설계 원리
--------------
상태벡터 = R (연간 함양률, mm/yr)  ← h 시계열이 아님

기존 h-state 방식의 문제:
  - 격자점에 수위 관측 없음 → 강수마다 h 상승 누적 → 함양 과대추정
  - 관정 수위가 바뀌어도 격자 함양이 변하지 않음 (공분산 붕괴)

R-state 방식의 장점:
  - 관정: run_logic_v27로 정확한 함양률 산출 → "관측값"으로 사용
  - 격자: k, z, Sy를 섭동한 앙상블로 R 사전분포 형성
  - EnKF: 관정 R 관측을 동화해 격자 R 추정
  - 국소화: 거리에 따라 관정 영향력 감쇠 → 물리적으로 올바름

결과:
  - 관정 근처 격자: 관정 R에 수렴, 불확실성 작음
  - 관정 먼 격자: 사전분포(파라미터 섭동 기반), 불확실성 큼
  - 관정 함양 바뀌면 격자도 연동 변화
  - 과대추정 없음

References
----------
Evensen, G. (2003). Ocean Dynamics, 53(4), 343-367.
Gaspari & Cohn (1999). Q.J.R. Meteorol. Soc., 125, 723-757.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


# ─────────────────────────────────────────────────────────────
# 공간 위치
# ─────────────────────────────────────────────────────────────
@dataclass
class SpatialPoint:
    x: float
    y: float
    name: str = ""
    is_well: bool = False

    def dist(self, other: "SpatialPoint") -> float:
        return float(np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2))


# ─────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────
@dataclass
class EnKFConfig:
    n_ensemble: int = 200             # 앙상블 크기 (R-state는 200+ 권장)
    localization_radius: float = 8.0  # 국소화 반경 (좌표 단위, km 권장)
    perturb_k_std: float = 0.002      # k 섭동 σ
    perturb_z_std: float = 1.0        # z_unsat 섭동 σ (m)
    obs_noise_mm: float = 20.0        # 관정 함양률 관측 불확실성 (mm/yr)
                                      # run_logic_v27의 추정 오차 반영
    r_c: float = 0.001                # 강수 임계 (m = 1mm)
    random_seed: int = 42


# ─────────────────────────────────────────────────────────────
# Gaspari-Cohn 국소화
# ─────────────────────────────────────────────────────────────
def _gc(d: float, r: float) -> float:
    """Gaspari-Cohn 5차 다항식. d=0→1, d≥r→0."""
    if r <= 0 or d >= r:
        return 0.0
    c = r / 2.0; z = d / c
    if z < 1.0:
        return 1-(5/4)*z**2+(5/3)*z**3+(5/8)*z**4-(1/2)*z**5
    return 4-5*z+(5/3)*z**2+(5/8)*z**3-(1/2)*z**4+(1/12)*z**5-2/(3*z)


# ─────────────────────────────────────────────────────────────
# 단일 파라미터 세트로 run_logic_v27 실행 → R 반환
# ─────────────────────────────────────────────────────────────
def _run_R(
    ho: np.ndarray, po: np.ndarray,
    sn: int, k: float, z: float,
    r_c: float = 0.001,
) -> float:
    """주어진 k, z로 run_logic_v27 실행 → 연간 함양률 (mm/yr)."""
    from core_sim_v27 import run_logic_v27, detect_pump_mask
    pump_mask = detect_pump_mask(ho, po, r_c)
    rech, _, _, _, _ = run_logic_v27(
        k, z, sn, po, ho, 0.005, 0.10, r_c, pump_mask,
        _fast=True,   # 최적화 속도용
    )
    n_yr = max(len(ho) / 365.25, 1.0)
    return float(rech.sum() / n_yr * 1000)  # mm/yr


# ─────────────────────────────────────────────────────────────
# 관정 최적화 (사전 실행)
# ─────────────────────────────────────────────────────────────
def _optimize_well(
    ho: np.ndarray, po: np.ndarray, sn: int, r_c: float,
) -> dict:
    """관정 최적 파라미터 탐색 + 기준 함양률 산출."""
    from core_sim_v27 import (
        detect_pump_mask, optimize_parameters, normalize_core_inputs,
        apply_lag, run_logic_v27, _estimate_equilibrium_head,
    )
    from soil_db import ALPHA_SOIL_LIST, SY_LIT_LIST

    pump_mask = detect_pump_mask(ho, po, r_c)
    _, _, _, opt_k, opt_z, _ = normalize_core_inputs(sn-1, -0.015, 3.0, 0)
    best_k, best_z, best_lag, best_rho, best_alpha = optimize_parameters(
        ho, po, sn, opt_k, opt_z, r_c, pump_mask)
    po_lag = apply_lag(po, best_lag)

    rech, hs_kf, _, sy_eff, _ = run_logic_v27(
        best_k, best_z, sn, po_lag, ho, 0.005, 0.10, r_c,
        pump_mask, rho=best_rho, alpha=best_alpha)

    n_yr = max(len(ho) / 365.25, 1.0)
    ref_mm = float(rech.sum() / n_yr * 1000)
    ann_prec_mm = float(po.sum() / n_yr * 1000)

    return dict(
        k=best_k, z=best_z, lag=best_lag,
        rho=best_rho, alpha_wtf=best_alpha,
        pump_mask=pump_mask, po_lag=po_lag,
        sy_eff=float(sy_eff),
        alpha_soil=float(ALPHA_SOIL_LIST[sn-1]),
        rech=rech, hs_kf=hs_kf,
        ref_mm=ref_mm,
        ann_prec_mm=ann_prec_mm,
    )


# ─────────────────────────────────────────────────────────────
# 결과 데이터클래스
# ─────────────────────────────────────────────────────────────
@dataclass
class EnKFResult:
    points: List[SpatialPoint]

    # (n_pts,) 연간 추정값
    ann_rech_mm: np.ndarray     # mm/yr
    ann_rech_pct: np.ndarray    # % of precip
    ann_rech_std: np.ndarray    # 불확실성 σ
    ann_rech_ci_lo: np.ndarray  # 95% CI 하한
    ann_rech_ci_hi: np.ndarray  # 95% CI 상한

    # (n_pts, N) 앙상블 분포
    ann_rech_ens: np.ndarray = field(default_factory=lambda: np.array([]))

    # 사전분포 (EnKF 업데이트 전)
    prior_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    prior_std: np.ndarray  = field(default_factory=lambda: np.array([]))

    well_params: dict = field(default_factory=dict)
    n_ensemble: int = 0
    localization_radius: float = 0.0

    def summary(self) -> str:
        n_w = sum(p.is_well for p in self.points)
        n_g = len(self.points) - n_w
        lines = [
            "═══ EnKF Spatial Recharge Summary (R-state) ═══",
            f"  Points   : {len(self.points)} ({n_w} wells / {n_g} grid)",
            f"  Ensemble : {self.n_ensemble} | "
            f"Loc. radius: {self.localization_radius:.1f} km",
            "",
            f"  {'Name':<12}{'Type':<7}{'R(mm/yr)':<11}"
            f"{'R(%)':<9}{'±Std':<9}{'95% CI'}",
            "  " + "─"*62,
        ]
        for i, pt in enumerate(self.points):
            ptype = "Well" if pt.is_well else "Grid"
            lines.append(
                f"  {pt.name:<12}{ptype:<7}"
                f"{self.ann_rech_mm[i]:<11.1f}"
                f"{self.ann_rech_pct[i]:<9.1f}"
                f"{self.ann_rech_std[i]:<9.1f}"
                f"[{self.ann_rech_ci_lo[i]:.0f}, {self.ann_rech_ci_hi[i]:.0f}]"
            )

        lines += ["", "  ── 검증 (관정) ──"]
        for name, wp in self.well_params.items():
            ref = wp.get('ref_mm')
            i = next(j for j,p in enumerate(self.points) if p.name==name)
            ev = self.ann_rech_mm[i]
            lo = self.ann_rech_ci_lo[i]
            hi = self.ann_rech_ci_hi[i]
            if ref is not None:
                ok = "✓" if lo <= ref <= hi else "✗"
                lines.append(
                    f"  {name}: EnKF={ev:.0f} Ref={ref:.0f} mm/yr "
                    f"CI=[{lo:.0f},{hi:.0f}] {ok}"
                )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# 메인 클래스
# ─────────────────────────────────────────────────────────────
class SpatialEnKF:
    """R-state 앙상블 Kalman 필터.

    올바른 사용법
    -------------
    >>> points = [
    ...     SpatialPoint(x=0, y=0, name='SH11', is_well=True),
    ...     SpatialPoint(x=2, y=1, name='G_01', is_well=False),
    ...     SpatialPoint(x=5, y=3, name='G_02', is_well=False),
    ... ]
    >>> enkf = SpatialEnKF(points, EnKFConfig(n_ensemble=200))
    >>> ho_m, po_m = load_sh_m('SH11.txt')
    >>> enkf.add_well('SH11', ho_m, po_m, sn=12)
    >>> po_raw = enkf._wells['SH11']['po']
    >>> enkf.add_grid('G_01', po_raw)
    >>> enkf.add_grid('G_02', po_raw)
    >>> result = enkf.run()
    >>> print(result.summary())
    """

    def __init__(
        self,
        points: List[SpatialPoint],
        config: Optional[EnKFConfig] = None,
    ):
        self.points = points
        self.cfg    = config or EnKFConfig()
        self.n_pts  = len(points)
        self.rng    = np.random.default_rng(self.cfg.random_seed)
        self._wells: dict = {}
        self._grids: dict = {}   # 현재는 강수 데이터만 저장 (R-state에서는 직접 미사용)

    def add_well(self, name: str, ho: np.ndarray, po: np.ndarray, sn: int = 12):
        """관정 데이터 등록 및 최적화 실행."""
        assert any(p.name == name and p.is_well for p in self.points), \
            f"'{name}' not found with is_well=True"
        print(f"  [EnKF] Optimizing {name} (sn={sn})...", end=" ", flush=True)
        p = _optimize_well(np.asarray(ho,float), np.asarray(po,float), sn, self.cfg.r_c)
        print(
            f"k={p['k']:.5f} z={p['z']:.1f}m "
            f"R={p['ref_mm']:.0f}mm/yr ({p['ref_mm']/p['ann_prec_mm']*100:.1f}%)"
        )
        self._wells[name] = dict(
            ho=np.asarray(ho,float),
            po=np.asarray(po,float),
            sn=sn, params=p,
        )

    def add_grid(self, name: str, po: np.ndarray):
        """격자 데이터 등록 (강수 공유)."""
        assert any(p.name == name and not p.is_well for p in self.points), \
            f"'{name}' not found with is_well=False"
        self._grids[name] = dict(po=np.asarray(po,float))

    def _nearest_well(self, pt: SpatialPoint) -> str:
        """격자점에서 가장 가까운 관정 이름."""
        wpts = [p for p in self.points if p.is_well and p.name in self._wells]
        if not wpts:
            return list(self._wells.keys())[0]
        dists = [pt.dist(wp) for wp in wpts]
        return wpts[int(np.argmin(dists))].name

    def run(self) -> EnKFResult:
        """R-state EnKF 실행.

        단계:
        1. 각 점마다 k, z를 섭동한 N개 앙상블으로 R 사전분포 형성
        2. 관정: run_logic_v27 최적값 R을 관측값으로 사용
        3. EnKF 업데이트: 각 격자점에 거리 기반 국소화 Kalman gain 적용
        4. 결과: 격자점 R 사후분포 (평균, std, 95% CI)
        """
        if not self._wells:
            raise ValueError("add_well()로 관정 데이터를 먼저 등록하세요.")

        N = self.cfg.n_ensemble
        loc_r = self.cfg.localization_radius
        obs_noise = self.cfg.obs_noise_mm

        print(f"\n  [EnKF] R-state: {self.n_pts}pts × {N}members")

        # ── 관정 기준값 (관측값) ──
        well_R = {}  # name → ref_mm
        well_prec = {}
        for name, wd in self._wells.items():
            well_R[name] = wd['params']['ref_mm']
            well_prec[name] = wd['params']['ann_prec_mm']

        # ── 각 점 앙상블 R 계산 ──
        R_ens_all = np.zeros((self.n_pts, N))  # 사전분포
        ann_prec_all = np.zeros(self.n_pts)

        for i, pt in enumerate(self.points):
            if pt.is_well and pt.name in self._wells:
                wd  = self._wells[pt.name]
                ho  = wd['ho']
                po  = wd['params']['po_lag']   # lag 적용된 강수
                sn  = wd['sn']
                k0  = wd['params']['k']
                z0  = wd['params']['z']
                ann_prec_all[i] = wd['params']['ann_prec_mm']
            else:
                # 격자: 가장 가까운 관정 데이터 사용 (강수, 수위 패턴 대리)
                nw_name = self._nearest_well(pt)
                nwd = self._wells[nw_name]
                ho  = nwd['ho']
                # 격자 강수: add_grid로 등록된 것 사용, 없으면 관정 것
                if pt.name in self._grids:
                    po = self._grids[pt.name]['po']
                else:
                    po = nwd['params']['po_lag']
                sn  = nwd['sn']
                k0  = nwd['params']['k']
                z0  = nwd['params']['z']
                ann_prec_all[i] = nwd['params']['ann_prec_mm']

            # k, z 섭동 앙상블
            k_ens = np.clip(
                k0 + self.rng.normal(0, self.cfg.perturb_k_std, N),
                -0.3, -0.0001,
            )
            z_ens = np.clip(
                z0 + self.rng.normal(0, self.cfg.perturb_z_std, N),
                0.1, 30.0,
            )

            print(f"    {pt.name} ({'Well' if pt.is_well else 'Grid'}): "
                  f"앙상블 R 계산 중 ({N}개)...", end="\r")

            R_ens = np.array([
                _run_R(ho, po, sn, k_ens[m], z_ens[m], self.cfg.r_c)
                for m in range(N)
            ])
            R_ens_all[i] = R_ens

        print(f"    앙상블 계산 완료.{' '*30}")

        # ── EnKF 업데이트 ──
        # 각 관정 R을 순서대로 동화 (sequential update)
        R_post = R_ens_all.copy()  # (n_pts, N)

        for w_name, R_obs in well_R.items():
            w_pt = next(p for p in self.points if p.name == w_name)

            # 관정 점 인덱스
            w_idx = next(j for j,p in enumerate(self.points) if p.name == w_name)

            # 관정 앙상블 R의 공분산
            X_w = R_post[w_idx]                    # (N,)
            X_w_anom = X_w - X_w.mean()
            PHT_ww = float(np.dot(X_w_anom, X_w_anom) / (N-1))
            S = PHT_ww + obs_noise**2              # scalar

            if S < 1e-10:
                continue  # 공분산 붕괴 방지

            # 관정-격자 공분산 및 국소화 적용
            for i in range(self.n_pts):
                pt = self.points[i]
                dist = pt.dist(w_pt)
                L = _gc(dist, loc_r)

                if L < 1e-6:
                    continue  # 국소화 반경 밖 → 업데이트 없음

                # PHT(i, w): i번 점과 관정의 교차 공분산
                X_i_anom = R_post[i] - R_post[i].mean()
                PHT_iw = float(np.dot(X_i_anom, X_w_anom) / (N-1))

                # Kalman gain (국소화 적용)
                K = L * PHT_iw / S

                # 혁신: 관측값 - 관정 앙상블 예측
                # stochastic: 관측에 노이즈 추가
                innov = (R_obs + self.rng.normal(0, obs_noise, N)) - X_w

                R_post[i] = R_post[i] + K * innov

        print(f"  [EnKF] 업데이트 완료.")

        # ── 결과 집계 ──
        ann_mean  = R_post.mean(axis=1)
        ann_std   = R_post.std(axis=1)
        ann_ci_lo = np.percentile(R_post, 2.5,  axis=1)
        ann_ci_hi = np.percentile(R_post, 97.5, axis=1)
        ann_pct   = ann_mean / np.maximum(ann_prec_all, 1.0) * 100

        prior_mean = R_ens_all.mean(axis=1)
        prior_std  = R_ens_all.std(axis=1)

        well_params = {
            name: dict(
                k=wd['params']['k'], z=wd['params']['z'],
                sy_eff=wd['params']['sy_eff'],
                ref_mm=wd['params']['ref_mm'],
            )
            for name, wd in self._wells.items()
        }

        return EnKFResult(
            points=self.points,
            ann_rech_mm=ann_mean, ann_rech_pct=ann_pct,
            ann_rech_std=ann_std,
            ann_rech_ci_lo=ann_ci_lo, ann_rech_ci_hi=ann_ci_hi,
            ann_rech_ens=R_post,
            prior_mean=prior_mean, prior_std=prior_std,
            well_params=well_params,
            n_ensemble=N,
            localization_radius=loc_r,
        )


# ─────────────────────────────────────────────────────────────
# 파일 로더
# ─────────────────────────────────────────────────────────────
def load_sh_m(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """SH*.txt → (ho_m, po_m) — m 단위."""
    from core_sim_v27 import load_core_data
    cdata = load_core_data(filepath)
    return cdata.ho.ravel(), cdata.po.ravel()


# ─────────────────────────────────────────────────────────────
# 데모
# ─────────────────────────────────────────────────────────────
def run_demo(data_dir: str = ".") -> Tuple["SpatialEnKF", EnKFResult]:
    """SH11 관정 + 거리별 격자 4개 데모."""
    points = [
        SpatialPoint(x=0.0, y=0.0, name="SH11", is_well=True),
        SpatialPoint(x=2.0, y=0.0, name="G_02km", is_well=False),
        SpatialPoint(x=4.0, y=0.0, name="G_04km", is_well=False),
        SpatialPoint(x=6.0, y=0.0, name="G_06km", is_well=False),
        SpatialPoint(x=9.0, y=0.0, name="G_09km", is_well=False),
    ]

    cfg = EnKFConfig(
        n_ensemble=200,
        localization_radius=8.0,
        perturb_k_std=0.002,
        perturb_z_std=1.0,
        obs_noise_mm=20.0,
        random_seed=42,
    )
    enkf = SpatialEnKF(points, cfg)

    fpath = os.path.join(data_dir, "SH11.txt")
    if os.path.exists(fpath):
        ho_m, po_m = load_sh_m(fpath)
        enkf.add_well("SH11", ho_m, po_m, sn=12)
    else:
        print("  SH11.txt 없음 — 합성 데이터 사용")
        n_ = 344
        ho_m = np.clip(np.random.normal(0.7, 0.15, n_), 0.05, 2.0)
        po_m = np.where(np.random.rand(n_) < 0.2,
                        np.random.exponential(0.005, n_), 0.0)
        enkf.add_well("SH11", ho_m, po_m, sn=12)

    po_raw = enkf._wells["SH11"]["po"]
    for pt in points[1:]:
        enkf.add_grid(pt.name, po_raw)

    result = enkf.run()
    print("\n" + result.summary())
    return enkf, result


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    enkf, result = run_demo(data_dir)
