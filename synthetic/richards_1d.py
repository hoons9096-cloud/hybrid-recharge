"""richards_1d.py — Mixed-form Richards equation 1D solver (Phase 2).

수치 모형
--------
Mixed-form Richards (Celia et al. 1990, WRR):

    ∂θ/∂t = ∂/∂z [K(h) (∂h/∂z - 1)]

van Genuchten 1980 (θ-h, K-h 관계):

    Se(h)   = (1 + (α|h|)^n)^(-m),         m = 1 - 1/n   (h<0)
    θ(h)    = θr + (θs - θr) Se
    K(h)    = Ks * Se^0.5 * (1 - (1 - Se^(1/m))^m)^2

특이성 분리 형태(SOLID):
    C(h)·∂h/∂t + ∂θ_old/∂t = ...

Picard iteration 으로 비선형 해결, implicit time stepping.

Boundary:
- Top:    flux BC,  q_top = P - ET (mm/day → m/s)  (양수: 침투)
- Bottom: free drainage,  q_bot = -K(h_bot)   (수직 중력류)

Output:
- 깊이별 θ(z, t)
- 매일 percolation (= true recharge) at lower boundary

References
----------
Celia, M.A., Bouloutas, E.T., & Zarba, R.L. (1990).
    A general mass-conservative numerical solution for the unsaturated
    flow equation. Water Resources Research, 26(7), 1483–1496.
van Genuchten, M.Th. (1980). A closed-form equation for predicting
    the hydraulic conductivity of unsaturated soils. SSSAJ, 44(5).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# van Genuchten constitutive relations
# ---------------------------------------------------------------------------
def vg_theta(
    h: np.ndarray, theta_s: float, theta_r: float,
    alpha: float, n: float,
) -> np.ndarray:
    """h(m, 음수=불포화) → θ.  α 단위는 1/m."""
    h = np.asarray(h, dtype=float)
    m = 1.0 - 1.0 / n
    Se = np.where(h < 0,
                  (1.0 + (alpha * np.abs(h)) ** n) ** (-m),
                  1.0)
    return theta_r + (theta_s - theta_r) * Se


def vg_capacity(
    h: np.ndarray, theta_s: float, theta_r: float,
    alpha: float, n: float,
) -> np.ndarray:
    """C(h) = dθ/dh."""
    h = np.asarray(h, dtype=float)
    m = 1.0 - 1.0 / n
    abs_h = np.abs(h)
    # C = -α m n (θs-θr) [1 + (α|h|)^n]^(-m-1) (α|h|)^(n-1) sign(h)
    # 포화 영역 (h>=0): C = 0
    base = (1.0 + (alpha * abs_h) ** n) ** (-(m + 1))
    factor = (alpha * abs_h) ** (n - 1)
    C = alpha * m * n * (theta_s - theta_r) * base * factor
    return np.where(h < 0, C, 0.0)


def vg_K(
    h: np.ndarray, Ks: float, theta_s: float, theta_r: float,
    alpha: float, n: float,
) -> np.ndarray:
    """h → K (van Genuchten-Mualem)."""
    h = np.asarray(h, dtype=float)
    m = 1.0 - 1.0 / n
    abs_h = np.abs(h)
    Se = np.where(h < 0,
                  (1.0 + (alpha * abs_h) ** n) ** (-m),
                  1.0)
    Se_clip = np.clip(Se, 1e-9, 1.0)
    inner = 1.0 - (1.0 - Se_clip ** (1.0 / m)) ** m
    return Ks * np.sqrt(Se_clip) * inner ** 2


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
@dataclass
class RichardsResult:
    z: np.ndarray                 # (Nz,) cell-center depths [m, 0=top]
    theta: np.ndarray             # (Nt+1, Nz) θ at each output time
    h: np.ndarray                 # (Nt+1, Nz) pressure head [m]
    flux_bottom: np.ndarray       # (Nt,) daily percolation [m/day]
    flux_top: np.ndarray          # (Nt,) actual top infiltration [m/day]
    storage_change: np.ndarray    # (Nt,) ΔStorage per day [m]
    mass_balance_err: np.ndarray  # (Nt,) per-day mass balance residual [m]

    @property
    def percolation_mm_per_day(self) -> np.ndarray:
        return self.flux_bottom * 1000.0

    @property
    def annual_recharge_mm(self) -> float:
        n_yr = max(len(self.flux_bottom) / 365.25, 1.0)
        return float(np.sum(self.flux_bottom) * 1000.0 / n_yr)


def solve_richards_1d(
    P_daily_m: np.ndarray,        # (Nt,) 강수 [m/day]
    ET_daily_m: np.ndarray,       # (Nt,) 잠재 ET [m/day]
    L: float = 3.0,               # 영역 깊이 [m]
    Nz: int = 60,                 # 수직 격자 수
    soil: dict = None,            # van Genuchten 파라미터
    h_init_m: float = -0.5,       # 초기 압력수두 [m]
    n_subdt: int = 48,            # 일일 sub-time steps
    picard_max: int = 30,
    picard_tol: float = 1e-4,
    h_max: float = 0.05,          # 압력수두 상한 (포화 클리핑)
    verbose: bool = False,
) -> RichardsResult:
    """일별 P/ET 입력 → 수치 Richards → 매일 percolation.

    soil: dict with keys: theta_s, theta_r, alpha (1/m), n, Ks (m/day)
    """
    if soil is None:
        # Default: USDA Loam (mid texture)
        soil = dict(theta_s=0.43, theta_r=0.078, alpha=3.6, n=1.56,
                    Ks=0.25)  # m/day
    theta_s = soil["theta_s"]; theta_r = soil["theta_r"]
    alpha = soil["alpha"]; nv = soil["n"]; Ks = soil["Ks"]

    P = np.asarray(P_daily_m, dtype=float)
    ET = np.asarray(ET_daily_m, dtype=float)
    Nt = len(P)

    dz = L / Nz
    z = (np.arange(Nz) + 0.5) * dz   # cell center
    dt = 1.0 / n_subdt               # day fraction
    n_steps = Nt * n_subdt

    h = np.full(Nz, h_init_m)
    theta_old_full = vg_theta(h, theta_s, theta_r, alpha, nv)

    flux_bot_daily = np.zeros(Nt)
    flux_top_daily = np.zeros(Nt)
    storage_d = np.zeros(Nt)
    mb_err_d = np.zeros(Nt)

    # 출력 buffer
    h_out = [h.copy()]
    theta_out = [theta_old_full.copy()]

    for t_idx in range(n_steps):
        day = t_idx // n_subdt
        # Daily flux (m/day) → m/sub-dt
        net_top_flux = (P[day] - ET[day])  # 양수=침투, 음수=증발

        # Picard iteration
        h_new = h.copy()
        h_n = h.copy()
        theta_old = theta_old_full.copy()

        for it in range(picard_max):
            # 압력수두 클리핑 (포화 폭주 방지)
            h_new = np.clip(h_new, -1e6, h_max)
            # 면 K — 기하 평균 (불포화 흐름에 안정적)
            K_cell = vg_K(h_new, Ks, theta_s, theta_r, alpha, nv)
            K_cell = np.clip(K_cell, 1e-12, None)
            K_face = np.sqrt(K_cell[:-1] * K_cell[1:])  # geometric mean
            C = vg_capacity(h_new, theta_s, theta_r, alpha, nv)
            theta_iter = vg_theta(h_new, theta_s, theta_r, alpha, nv)

            # 선형계수 (tridiagonal):
            # mixed form: C·(h^{k+1}-h^n)/dt + (θ^{k}-θ^n)/dt = ∂/∂z[K(h^k)·(∂h/∂z - 1)]
            # → A·h^{k+1} = b
            a = np.zeros(Nz)   # sub
            b_diag = np.zeros(Nz)
            c = np.zeros(Nz)   # super
            rhs = np.zeros(Nz)

            for i in range(Nz):
                # face fluxes:
                # interior: q_face = -K_face * ((h_{i+1}-h_i)/dz - 1)
                Kf_up = K_face[i - 1] if i > 0 else 0.0
                Kf_dn = K_face[i] if i < Nz - 1 else 0.0

                if i == 0:
                    # Top BC: prescribed flux net_top_flux (positive=infil downward)
                    # q_top_into_cell = net_top_flux  (양수 → cell 0 채움)
                    # implicit only on bottom face
                    coef_dn = Kf_dn / dz**2
                    a[i] = 0.0
                    c[i] = -coef_dn
                    b_diag[i] = C[i] / dt + coef_dn
                    rhs[i] = (
                        C[i] * h_new[i] / dt
                        - (theta_iter[i] - theta_old[i]) / dt
                        + net_top_flux / dz                # top flux in
                        - Kf_dn / dz                       # gravity term at bottom face
                    )
                elif i == Nz - 1:
                    # Bottom BC: free drainage,  q_bot = -K_cell[Nz-1]
                    coef_up = Kf_up / dz**2
                    a[i] = -coef_up
                    c[i] = 0.0
                    b_diag[i] = C[i] / dt + coef_up
                    rhs[i] = (
                        C[i] * h_new[i] / dt
                        - (theta_iter[i] - theta_old[i]) / dt
                        + Kf_up / dz                        # gravity at top face
                        - K_cell[i] / dz                    # bottom drainage
                    )
                else:
                    coef_up = Kf_up / dz**2
                    coef_dn = Kf_dn / dz**2
                    a[i] = -coef_up
                    c[i] = -coef_dn
                    b_diag[i] = C[i] / dt + coef_up + coef_dn
                    rhs[i] = (
                        C[i] * h_new[i] / dt
                        - (theta_iter[i] - theta_old[i]) / dt
                        + Kf_up / dz - Kf_dn / dz
                    )

            # Tridiagonal solve (Thomas)
            h_next = _thomas(a, b_diag, c, rhs)
            err = np.max(np.abs(h_next - h_new))
            h_new = h_next
            if err < picard_tol:
                break

        # Update for next sub-step
        h = h_new
        theta_new = vg_theta(h, theta_s, theta_r, alpha, nv)

        # Bottom flux (m/day equivalent for this sub-dt) accumulate
        K_bot = vg_K(np.array([h[-1]]), Ks, theta_s, theta_r, alpha, nv)[0]
        q_bot_sub = K_bot  # free drainage: flux out = K(h_bot) [m/day]
        flux_bot_daily[day] += q_bot_sub * dt
        flux_top_daily[day] += net_top_flux * dt

        # Storage change accumulate
        dStor = np.sum(theta_new - theta_old_full) * dz
        storage_d[day] += dStor

        # Mass balance: in - out - dStor
        mb = (net_top_flux - q_bot_sub) * dt - dStor
        mb_err_d[day] += mb

        theta_old_full = theta_new

        # 출력 buffer (매일 마지막 sub-step)
        if (t_idx + 1) % n_subdt == 0:
            h_out.append(h.copy())
            theta_out.append(theta_new.copy())

    return RichardsResult(
        z=z,
        theta=np.array(theta_out),
        h=np.array(h_out),
        flux_bottom=flux_bot_daily,
        flux_top=flux_top_daily,
        storage_change=storage_d,
        mass_balance_err=mb_err_d,
    )


# ---------------------------------------------------------------------------
# Thomas algorithm (tridiagonal solver)
# ---------------------------------------------------------------------------
def _thomas(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Solve tridiagonal A x = d.
    a: subdiagonal (a[0] ignored), b: diagonal, c: superdiagonal (c[-1] ignored),
    d: RHS.  Length n.
    """
    n = len(b)
    cp = np.empty(n)
    dp = np.empty(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / denom if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom
    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


# ---------------------------------------------------------------------------
# soil_db.py 인덱스 → Richards solver 입력 변환
# ---------------------------------------------------------------------------
def soil_params_from_sn(sn_idx: int) -> dict:
    """soil_db.py SOIL_DB[sn-1] → Richards 입력 dict.

    Ks 는 textbook (Carsel-Parrish median) 사용.
    """
    from soil_db import SOIL_DB
    s = SOIL_DB[sn_idx]
    # Carsel-Parrish 1988 Table 3 — median Ksat (cm/h → m/day)
    Ks_table_cm_per_hr = {
        1: 29.7,    # Sand
        2: 14.59,   # Loamy Sand
        3: 4.42,    # Sandy Loam
        4: 0.45,    # Silt Loam
        5: 0.25,    # Silt
        6: 0.20,    # Clay
        7: 0.020,   # Silty Clay
        8: 0.12,    # Sandy Clay
        9: 0.07,    # Silty Clay Loam
        10: 0.26,   # Clay Loam
        11: 0.55,   # Sandy Clay Loam
        12: 1.04,   # Loam
    }
    Ks_cm_per_hr = Ks_table_cm_per_hr.get(sn_idx, 1.0)
    Ks_m_per_day = Ks_cm_per_hr * 0.01 * 24.0  # cm/h → m/day
    return dict(
        theta_s=s.theta_s, theta_r=s.theta_r,
        alpha=s.alpha_vg, n=s.n_vg,
        Ks=Ks_m_per_day,
    )


# ===========================================================================
# v2 — Properly separated surface BC + root-zone ET sink
# ===========================================================================
# 기존 v1 솔버의 문제:
#   · 상단 BC를 net = P - ET 로 lumped → ET>P 인 날에 표면에서 물을
#     위로 끌어올리려는 unphysical flux로 발산
#   · ET가 root zone 분포되지 않고 표면에만 적용 → 토양 동역학 왜곡
#   · 침투 한계(Ks) 미적용 → 강수가 그대로 토양에 강제 주입되어
#     포화 폭주 → 질량보존 오차 누적
#
# v2 수정:
#   · 표면: q_in = min(P, Ks) (침투 한계), 초과분은 runoff
#   · ET: root zone (z ∈ [0, root_depth]) 에 지수 분포 sink 항 S(z,θ)
#         물스트레스 보정: stress = (θ - θr) / (θs - θr) ∈ [0, 1]
#   · Picard 수렴 강화 + sub-step 분할 fallback
#   · 질량보존: in - runoff - ET_actual - drainage = ΔS
# ---------------------------------------------------------------------------


@dataclass
class RichardsResultV2:
    z: np.ndarray
    theta: np.ndarray
    h: np.ndarray
    flux_bottom: np.ndarray       # (Nt,) drainage [m/day]
    flux_top_actual: np.ndarray   # (Nt,) actual infiltration [m/day]
    runoff: np.ndarray            # (Nt,) overflow [m/day]
    ET_actual: np.ndarray         # (Nt,) realized ET [m/day]
    storage_change: np.ndarray
    mass_balance_err: np.ndarray  # (Nt,) per-day MB residual [m]


def solve_richards_1d_v2(
    P_daily_m: np.ndarray,
    ETp_daily_m: np.ndarray,
    L: float = 3.0,
    Nz: int = 60,
    soil: dict = None,
    h_init_m: float = -1.0,
    n_subdt: int = 48,
    picard_max: int = 50,
    picard_tol: float = 1e-5,
    root_depth_m: float = 1.0,
    root_decay: float = 1.5,
    stress_h_wilting: float = -150.0,   # m (≈ -15 bar permanent wilting)
    stress_h_field: float = -3.3,       # m (≈ -0.33 bar field capacity)
    verbose: bool = False,
) -> RichardsResultV2:
    """Mixed-form Richards 1D with proper surface BC and root-zone ET sink.

    상단 BC (infiltration-limited Neumann):
        q_in = min(P, K_top)  with overflow → runoff
    하단 BC: free drainage (q = -K(h_bot))
    Root-zone sink:
        S(z) = ETp · w_root(z) · stress(h)
        w_root(z) ∝ exp(-root_decay · z / root_depth)   for z ≤ root_depth
        stress(h) = clip((h - h_wilt) / (h_field - h_wilt), 0, 1)
    """
    if soil is None:
        soil = dict(theta_s=0.43, theta_r=0.078, alpha=3.6, n=1.56, Ks=0.25)
    theta_s = soil["theta_s"]; theta_r = soil["theta_r"]
    alpha = soil["alpha"]; nv = soil["n"]; Ks = soil["Ks"]

    P = np.asarray(P_daily_m, dtype=float)
    ETp = np.asarray(ETp_daily_m, dtype=float)
    Nt = len(P)

    dz = L / Nz
    z = (np.arange(Nz) + 0.5) * dz
    # CFL-aware sub-stepping: dt ≤ dz / (safety·Ks)
    # Safety scales with vG-n (stiffer curves need finer dt)
    cfl_safety = 4.0 + 4.0 * max(nv - 1.5, 0.0)   # n=1.5 → 4×, n=2.5 → 8×
    n_subdt_cfl = int(np.ceil(cfl_safety * Ks * Nz / L))
    n_subdt = max(n_subdt, n_subdt_cfl)
    dt = 1.0 / n_subdt
    n_steps = Nt * n_subdt
    if verbose:
        print(f"  [Richards v2] n_subdt = {n_subdt} (Ks={Ks:.2f} m/day, "
              f"dz={dz:.3f} m → CFL min={n_subdt_cfl})")

    # Root weights — exponential, zero below root depth, integrates to 1
    in_root = z <= root_depth_m
    w = np.zeros(Nz)
    if np.any(in_root):
        w_raw = np.exp(-root_decay * z[in_root] / root_depth_m)
        w[in_root] = w_raw / (w_raw.sum() * dz)

    h = np.full(Nz, h_init_m)
    theta = vg_theta(h, theta_s, theta_r, alpha, nv)

    flux_bot_d = np.zeros(Nt)
    flux_top_d = np.zeros(Nt)
    runoff_d = np.zeros(Nt)
    ET_act_d = np.zeros(Nt)
    storage_d = np.zeros(Nt)
    mb_d = np.zeros(Nt)

    h_out = [h.copy()]
    theta_out = [theta.copy()]

    for t_idx in range(n_steps):
        day = t_idx // n_subdt
        P_day = P[day]
        ETp_day = ETp[day]

        # ----- Surface partition: infiltration vs. runoff -----------------
        # Conservative cap: infil ≤ Ks. (At fully saturated top h≥0 we'd
        # use K(h=0)=Ks; otherwise even less. Cap at Ks is safe upper.)
        q_in = min(P_day, Ks)
        q_runoff = max(P_day - q_in, 0.0)

        h_old = h.copy()
        theta_old = theta.copy()
        h_new = h.copy()

        for it in range(picard_max):
            h_new = np.clip(h_new, -1e6, 0.0)
            K_cell = vg_K(h_new, Ks, theta_s, theta_r, alpha, nv)
            K_cell = np.clip(K_cell, 1e-14, None)
            K_face = np.sqrt(K_cell[:-1] * K_cell[1:])
            C = vg_capacity(h_new, theta_s, theta_r, alpha, nv)
            C = np.clip(C, 1e-9, None)
            theta_iter = vg_theta(h_new, theta_s, theta_r, alpha, nv)

            # ET sink term S_i [m³/m³/day] = ETp · w_i · stress_i
            stress = np.clip(
                (h_new - stress_h_wilting)
                / (stress_h_field - stress_h_wilting),
                0.0, 1.0,
            )
            S = ETp_day * w * stress    # per-cell (1/day)

            a = np.zeros(Nz); b_diag = np.zeros(Nz); c = np.zeros(Nz)
            rhs = np.zeros(Nz)

            for i in range(Nz):
                Kf_up = K_face[i - 1] if i > 0 else 0.0
                Kf_dn = K_face[i] if i < Nz - 1 else 0.0
                if i == 0:
                    coef_dn = Kf_dn / dz**2
                    a[i] = 0.0
                    c[i] = -coef_dn
                    b_diag[i] = C[i] / dt + coef_dn
                    rhs[i] = (
                        C[i] * h_new[i] / dt
                        - (theta_iter[i] - theta_old[i]) / dt
                        + q_in / dz
                        - Kf_dn / dz
                        - S[i]
                    )
                elif i == Nz - 1:
                    coef_up = Kf_up / dz**2
                    a[i] = -coef_up
                    c[i] = 0.0
                    b_diag[i] = C[i] / dt + coef_up
                    rhs[i] = (
                        C[i] * h_new[i] / dt
                        - (theta_iter[i] - theta_old[i]) / dt
                        + Kf_up / dz
                        - K_cell[i] / dz
                        - S[i]
                    )
                else:
                    coef_up = Kf_up / dz**2
                    coef_dn = Kf_dn / dz**2
                    a[i] = -coef_up
                    c[i] = -coef_dn
                    b_diag[i] = C[i] / dt + coef_up + coef_dn
                    rhs[i] = (
                        C[i] * h_new[i] / dt
                        - (theta_iter[i] - theta_old[i]) / dt
                        + Kf_up / dz - Kf_dn / dz
                        - S[i]
                    )

            h_next = _thomas(a, b_diag, c, rhs)
            err = np.max(np.abs(h_next - h_new))
            h_new = h_next
            if err < picard_tol:
                break

        # Limit if top oversaturates (extra → runoff)
        if h_new[0] > 0.0:
            # Estimate excess infiltration that pushed h>0
            # Simplification: clip and route excess as runoff this sub-step
            excess_theta = vg_theta(np.array([h_new[0]]),
                                    theta_s, theta_r, alpha, nv)[0] - theta_s
            extra_runoff = max(excess_theta, 0.0) * dz / dt   # m/day
            q_runoff += extra_runoff
            q_in -= extra_runoff
            h_new[0] = 0.0

        h = h_new
        theta = vg_theta(h, theta_s, theta_r, alpha, nv)

        # Bottom drainage
        K_bot = vg_K(np.array([h[-1]]), Ks, theta_s, theta_r, alpha, nv)[0]
        q_bot_sub = K_bot

        # Realized ET (per-cell sink integrated over column)
        stress_post = np.clip(
            (h - stress_h_wilting) / (stress_h_field - stress_h_wilting),
            0.0, 1.0,
        )
        S_post = ETp_day * w * stress_post
        ET_realized = float(np.sum(S_post) * dz)   # m/day

        # Daily accumulators
        flux_top_d[day] += q_in * dt
        runoff_d[day]   += q_runoff * dt
        flux_bot_d[day] += q_bot_sub * dt
        ET_act_d[day]   += ET_realized * dt

        dStor = float(np.sum(theta - theta_old) * dz)
        storage_d[day] += dStor

        # MB this sub-step: in - drainage - ET - dStor (runoff already
        # excluded from q_in)
        mb_d[day] += (q_in - q_bot_sub - ET_realized) * dt - dStor

        if (t_idx + 1) % n_subdt == 0:
            h_out.append(h.copy())
            theta_out.append(theta.copy())

    return RichardsResultV2(
        z=z,
        theta=np.array(theta_out),
        h=np.array(h_out),
        flux_bottom=flux_bot_d,
        flux_top_actual=flux_top_d,
        runoff=runoff_d,
        ET_actual=ET_act_d,
        storage_change=storage_d,
        mass_balance_err=mb_d,
    )
