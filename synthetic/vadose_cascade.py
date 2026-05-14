"""vadose_cascade.py — Multi-layer cascade vadose zone model (Phase 2).

Hewlett & Hibbert (1967), FAO-56 (Allen et al. 1998), SWAT (Arnold et al. 1998)
계열의 표준 N-layer bucket cascade.

수식
----
각 층 i, 일 단위:

    S_i^{t+1} = S_i^t + (q_in - q_out - ET_i) · 1day

    q_in     = q_drain_{i-1}     (i=0: P_daily)
    q_drain  = (S_i - S_fc_i) / τ_i,   if S_i > S_fc_i  else 0
    ET_i     = root_frac_i · ET_p · stress(S_i)
    stress   = clip((θ_i - θ_wp) / (θ_fc - θ_wp), 0, 1)

마지막 층의 q_drain = **true recharge** (true percolation across vadose).

수치적 안정 — 일 단위 implicit:
    drainage timescale τ ≥ 1day 인 토양에서는 그대로 explicit OK.
    τ < 1day (sand 류) 는 sub-daily step 분할.

soil_db.py 의 (theta_s, theta_r, tau, sy_lit) 를 직접 사용.
Field capacity / wilting point 는 van Genuchten 으로부터 도출:
    θ_fc = θ(h = -3.3 m)         # 1/3 atm
    θ_wp = θ(h = -150 m)          # 15 atm
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# van Genuchten θ(h) 헬퍼 (richards_1d 와 동일)
# ---------------------------------------------------------------------------
def _vg_theta_at_head(theta_s, theta_r, alpha, n, h_m):
    """Pressure head h(m, 음수=불포화) → θ."""
    m = 1.0 - 1.0 / n
    if h_m >= 0:
        return theta_s
    Se = (1.0 + (alpha * abs(h_m)) ** n) ** (-m)
    return theta_r + (theta_s - theta_r) * Se


# ---------------------------------------------------------------------------
# Layer descriptor + cascade 결과
# ---------------------------------------------------------------------------
@dataclass
class LayerSoil:
    name: str
    dz_m: float                 # 층 두께 [m]
    theta_s: float
    theta_r: float
    theta_fc: float             # field capacity
    theta_wp: float             # wilting point
    tau_day: float              # drainage time constant
    root_frac: float            # 0~1 (모든 층 합 = 1)


@dataclass
class CascadeResult:
    days: np.ndarray            # (Nt+1,) 시간 인덱스 (포함 t=0)
    theta: np.ndarray           # (Nt+1, n_layers) 일별 층별 θ
    storage_mm: np.ndarray      # (Nt+1, n_layers) 일별 층별 저장량 [mm]
    q_drain_layer: np.ndarray   # (Nt, n_layers) 일별 층별 drainage [mm/day]
    ET_actual: np.ndarray       # (Nt,) 실제 ET [mm/day]
    recharge: np.ndarray        # (Nt,) 일별 함양 = 마지막 층 drainage [mm/day]
    runoff: np.ndarray          # (Nt,) ponding overflow (top 층 포화 시) [mm/day]
    mass_balance_err_mm: np.ndarray  # (Nt,) 누적 mass balance 잔차 [mm]

    @property
    def annual_recharge_mm(self) -> float:
        n_yr = max(len(self.recharge) / 365.25, 1.0)
        return float(np.sum(self.recharge) / n_yr)

    @property
    def annual_ET_mm(self) -> float:
        n_yr = max(len(self.ET_actual) / 365.25, 1.0)
        return float(np.sum(self.ET_actual) / n_yr)


# ---------------------------------------------------------------------------
# Layer 빌더 — soil_db.py 기반
# ---------------------------------------------------------------------------
def build_layers_from_sn(
    sn_idx: int,
    L_total_m: float = 2.0,
    n_layers: int = 5,
    root_decay: float = 1.0,    # 0=균등, 양수=상층 root 우세
) -> List[LayerSoil]:
    """soil_db.py 단일 토양 + N층 동질 구성."""
    from soil_db import SOIL_DB
    s = SOIL_DB[sn_idx]
    theta_fc = _vg_theta_at_head(s.theta_s, s.theta_r, s.alpha_vg, s.n_vg, -3.3)
    theta_wp = _vg_theta_at_head(s.theta_s, s.theta_r, s.alpha_vg, s.n_vg, -150.0)
    if theta_fc <= theta_wp + 0.01:
        # 비현실 케이스 (실수 방지) — 기본값
        theta_fc = max(theta_wp + 0.05, 0.20)

    dz = L_total_m / n_layers

    # root_frac: 지수 감소 (상층 우세)
    if root_decay <= 0:
        rf = np.ones(n_layers) / n_layers
    else:
        weights = np.exp(-root_decay * np.arange(n_layers))
        rf = weights / weights.sum()

    layers = []
    for i in range(n_layers):
        # 층마다 τ 를 두께 비례로 분배 (전체 τ_lit = 단일층 기준 → 다층화 시 분할)
        tau_i = max(s.tau / n_layers, 0.5)
        layers.append(LayerSoil(
            name=f"{s.name}-L{i+1}",
            dz_m=dz,
            theta_s=s.theta_s, theta_r=s.theta_r,
            theta_fc=theta_fc, theta_wp=theta_wp,
            tau_day=tau_i,
            root_frac=float(rf[i]),
        ))
    return layers


# ---------------------------------------------------------------------------
# Cascade solver
# ---------------------------------------------------------------------------
def simulate_cascade(
    P_daily_mm: np.ndarray,
    ET_daily_mm: np.ndarray,
    layers: List[LayerSoil],
    init_theta_frac: float = 0.6,   # 초기 θ = θ_wp + frac·(θ_fc - θ_wp)
    n_subdt: Optional[int] = None,  # None → tau 기반 자동 결정
) -> CascadeResult:
    """일별 P, ET → multi-layer cascade → 일별 함양량."""
    P = np.asarray(P_daily_mm, dtype=float)
    ETp = np.asarray(ET_daily_mm, dtype=float)
    Nt = len(P)
    nL = len(layers)
    if not nL:
        raise ValueError("layers empty")

    # sub-time step 자동 결정 (안정성: dt < 0.5·τ_min)
    tau_min = min(L.tau_day for L in layers)
    if n_subdt is None:
        n_subdt = max(1, int(np.ceil(2.0 / tau_min)))
    dt = 1.0 / n_subdt

    # θ → S 변환: S [mm] = θ × dz × 1000
    dz_mm = np.array([L.dz_m * 1000.0 for L in layers])
    theta_s = np.array([L.theta_s for L in layers])
    theta_r = np.array([L.theta_r for L in layers])
    theta_fc = np.array([L.theta_fc for L in layers])
    theta_wp = np.array([L.theta_wp for L in layers])
    tau = np.array([L.tau_day for L in layers])
    rf = np.array([L.root_frac for L in layers])

    # 초기 θ (모든 층 동일)
    theta = theta_wp + init_theta_frac * (theta_fc - theta_wp)
    S = theta * dz_mm                # mm
    S_fc = theta_fc * dz_mm
    S_wp = theta_wp * dz_mm
    S_sat = theta_s * dz_mm

    # 출력 buffer
    theta_traj = [theta.copy()]
    S_traj = [S.copy()]
    q_drain_daily = np.zeros((Nt, nL))
    ET_act_daily = np.zeros(Nt)
    rech_daily = np.zeros(Nt)
    runoff_daily = np.zeros(Nt)
    mb_err_daily = np.zeros(Nt)

    for t in range(Nt):
        # 일 누적 추적용
        in_total = P[t]              # mm/day
        out_drain_bottom = 0.0
        out_ET = 0.0
        out_runoff = 0.0
        S_start = S.copy()

        for sub in range(n_subdt):
            # 1. ET 잠재 (균등 분배 후 stress 적용)
            stress = np.clip((S - S_wp) / np.maximum(S_fc - S_wp, 1e-6), 0.0, 1.0)
            ET_layer = rf * ETp[t] * stress * dt   # mm in this sub-step
            ET_layer = np.minimum(ET_layer, np.maximum(S - S_wp, 0.0))
            S = S - ET_layer
            out_ET += float(np.sum(ET_layer))

            # 2. drainage (top → bottom 순 cascade, gravity)
            q_in = P[t] * dt          # mm in this sub-step (top inflow)
            for i in range(nL):
                S[i] += q_in
                # 포화 초과 → runoff (위로 튕겨나가지 않고 표면 유출 처리)
                if S[i] > S_sat[i]:
                    overflow = S[i] - S_sat[i]
                    if i == 0:
                        out_runoff += overflow
                    else:
                        # 하층의 backflow 는 가까운 층으로 — 단순화: 표면 유출
                        out_runoff += overflow
                    S[i] = S_sat[i]
                # field capacity 초과분 drainage
                if S[i] > S_fc[i]:
                    q_out = (S[i] - S_fc[i]) * (dt / tau[i])
                    q_out = min(q_out, S[i] - S_fc[i])
                    S[i] -= q_out
                    q_drain_daily[t, i] += q_out
                    if i == nL - 1:
                        out_drain_bottom += q_out
                    q_in = q_out
                else:
                    q_in = 0.0

        # θ 갱신
        theta = S / dz_mm
        theta_traj.append(theta.copy())
        S_traj.append(S.copy())
        ET_act_daily[t] = out_ET
        rech_daily[t] = out_drain_bottom
        runoff_daily[t] = out_runoff

        # mass balance: ΔS = in - ET - drain_bottom - runoff
        dS = float(np.sum(S - S_start))
        mb = in_total - out_ET - out_drain_bottom - out_runoff - dS
        mb_err_daily[t] = mb

    return CascadeResult(
        days=np.arange(Nt + 1),
        theta=np.array(theta_traj),
        storage_mm=np.array(S_traj),
        q_drain_layer=q_drain_daily,
        ET_actual=ET_act_daily,
        recharge=rech_daily,
        runoff=runoff_daily,
        mass_balance_err_mm=mb_err_daily,
    )
