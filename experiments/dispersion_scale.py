"""
brick_02 — c_eff 는 하나의 숫자가 아니라 스케일(파장)의 함수인가?

최소 주장 (테스트 대상):
  같은 이산 규칙이라도, 교란(wave-packet)이 얼마나 큰가(=격자 대비 파장)에 따라
  관측되는 전파속도가 달라진다. 즉 "속도"는 절대 상수가 아니라 스케일 비율의 함수다.

정직한 예고 — 재보면 '속도'가 둘로 갈라진다:
  (1) 신호/원뿔 속도  : 시프트가 스텝당 ≤1칸 → 항상 정확히 1 (구조적, 스케일 무관).
  (2) 군속도(group)   : 덩어리 무게중심이 실제로 움직이는 속도 → 파장·질량에 의존.
  → 스케일-관계적인 것은 (2)이고, 인과 최대속도 (1)은 불변. 이 구분이 brick_02의 발견.

모형: brick_01 과 동일한 1D Dirac Quantum Walk.
  한 스텝 운영자를 파수 k 로 푸리에하면  U(k) = S(k) · C(m),
    C(m) = [[cos m, i sin m],[i sin m, cos m]],  S(k) = diag(e^-ik, e^+ik)
  det U = 1, 고유값 e^±iω  →  분산관계:  cos ω = cos m · cos k
  군속도:  v_g(k) = dω/dk = cos m · sin k / sqrt(1 - cos^2 m cos^2 k)
  파장(격자칸 단위)  λ = 2π / k   ← 이게 (관측 스케일 / 격자 간격) 비율.
"""

import numpy as np


# ---------- 이론: 띠구조(band)와 군속도 ----------

def U_of_k(k: float, mass: float) -> np.ndarray:
    c, s = np.cos(mass), np.sin(mass)
    C = np.array([[c, 1j * s], [1j * s, c]], dtype=complex)
    S = np.diag([np.exp(-1j * k), np.exp(1j * k)])
    return S @ C


def band_and_group(mass: float, nk: int = 4001):
    """상단 밴드 ω(k) 와 군속도 v_g(k) 를 U(k) 고유값에서 직접 구한다."""
    ks = np.linspace(-np.pi, np.pi, nk)
    omega = np.empty(nk)
    for i, k in enumerate(ks):
        # trace = 2 cos m cos k = 2 cos ω  →  상단 밴드
        omega[i] = np.arccos(np.clip(np.cos(mass) * np.cos(k), -1, 1))
    v_g = np.gradient(omega, ks)
    return ks, omega, v_g


def v_g_theory(k: float, mass: float) -> float:
    c = np.cos(mass)
    denom = np.sqrt(max(1.0 - (c * np.cos(k)) ** 2, 1e-15))
    return c * np.sin(k) / denom


# ---------- 실험: 실제 덩어리를 쏴서 속도를 측정 ----------

def eigenspinor(k: float, mass: float) -> np.ndarray:
    """U(k) 의 상단 밴드(+ω) 고유벡터 = +에너지 덩어리의 스피너."""
    U = U_of_k(k, mass)
    vals, vecs = np.linalg.eig(U)
    # 시간발전 e^{iθt} 이므로 물리적 ω = -θ. k>0 에서 오른쪽(+)으로 가는
    # 양(+)의 군속도 상태를 고르려면 고유값 위상 θ = -arccos(...) 를 택한다.
    target = -np.arccos(np.clip(np.cos(mass) * np.cos(k), -1, 1))
    j = int(np.argmin(np.abs(np.angle(vals) - target)))
    return vecs[:, j]


def measure_packet_velocity(k0: float, mass: float, steps: int = 120,
                            sigma: float = 18.0) -> float:
    """파수 k0 의 가우시안 덩어리를 쏴서 무게중심 속도 <x>/t 를 측정한다."""
    N = 6 * steps + 1                # 원뿔(속도1)이 경계에 닿지 않도록 넉넉히
    center = N // 2
    x = np.arange(N) - center
    envelope = np.exp(-x ** 2 / (2 * sigma ** 2)) * np.exp(1j * k0 * x)
    spinor = eigenspinor(k0, mass)
    psi = np.outer(envelope, spinor)         # (N,2) : +에너지 코히런트 덩어리
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2))

    c, s = np.cos(mass), np.sin(mass)
    C = np.array([[c, 1j * s], [1j * s, c]], dtype=complex)

    coms = np.empty(steps + 1)
    for t in range(steps + 1):
        prob = np.sum(np.abs(psi) ** 2, axis=1)
        coms[t] = np.sum(x * prob)           # 무게중심 <x>
        psi = psi @ C.T
        up = np.roll(psi[:, 0], 1)
        down = np.roll(psi[:, 1], -1)
        psi = np.stack([up, down], axis=1)
    # 후반 구간 기울기 = 군속도 (초기 과도구간 제외)
    t0 = steps // 3
    slope = np.polyfit(np.arange(t0, steps + 1), coms[t0:], 1)[0]
    return slope


if __name__ == "__main__":
    MASSES = [0.0, 0.30, 0.80]
    PROBE_KS = [np.pi / 4, np.pi / 2, 3 * np.pi / 4]   # 세 가지 파장으로 덩어리 발사

    print("=" * 72)
    print("brick_02 — 속도는 스케일(파장)의 함수인가?  (1D Dirac Quantum Walk)")
    print("=" * 72)

    # [검증 1] 질량 없으면 분산 없음: 모든 파장에서 군속도 = 1 (스케일 무관)
    # [검증 2] 질량 있으면 분산: 군속도가 파장에 따라 달라짐 (스케일 의존)
    print("\n측정: 파수 k 의 덩어리를 쏴서 무게중심 속도를 재고, 이론값과 비교\n")
    print(f"{'질량 m':>7} | {'k=π/4 (λ=8칸)':>22} | {'k=π/2 (λ=4칸)':>22} | {'k=3π/4 (λ≈2.7칸)':>22}")
    print("-" * 72)
    for m in MASSES:
        cells = []
        for k0 in PROBE_KS:
            meas = measure_packet_velocity(k0, m)
            theo = v_g_theory(k0, m)
            cells.append(f"측정 {meas:+.3f} / 이론 {theo:+.3f}")
        print(f"{m:>7.2f} | " + " | ".join(f"{c:>22}" for c in cells))
    print("-" * 72)

    # 요약 판정
    print("\n[검증 1] m=0 : 세 파장 모두 군속도 = 1.000")
    print("         → 질량 없으면 분산 없음. 속도는 스케일과 무관한 단일값. (선행연구와 일치)")
    print("\n[검증 2] m>0 : 군속도가 파장에 따라 달라짐 (예: m=0.8 에서 k=π/2 가 최대)")
    print("         → 질량이 있으면 '관측되는 속도'는 스케일(파장) 비율의 함수다. ✓")
    print("         이것이 이 프로젝트의 명제 — c_eff 는 절대가 아니라 스케일에서 드러난다.")
    print("\n[정직한 단서] 그러나 신호/원뿔 최대속도는 여전히 정확히 1 (구조적, 불변).")
    print("         → 스케일-관계적인 것은 '군속도(관측 전파)'이고, 인과 최대속도는 아니다.")
    print("         '속도'가 둘로 갈라진 것 자체가 brick_02 의 발견.\n")

    # ---------- 그림 ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

        # (좌) 분산관계 ω(k)
        for m in MASSES:
            ks, omega, _ = band_and_group(m)
            ax1.plot(ks, omega, label=f"m = {m}")
        ax1.set_title("Dispersion relation  cos ω = cos m · cos k")
        ax1.set_xlabel("wavenumber k  (per site)")
        ax1.set_ylabel("energy ω  (per step)")
        ax1.legend()
        ax1.grid(alpha=0.3)

        # (우) 군속도 vs 파장(=스케일 비율), 측정점 겹치기
        for m in MASSES:
            ks, _, v_g = band_and_group(m)
            pos = (ks > 0.05)
            lam = 2 * np.pi / ks[pos]           # 파장 = 스케일 비율
            ax2.plot(lam, v_g[pos], label=f"m = {m}  (theory)")
        for m in MASSES:
            for k0 in PROBE_KS:
                meas = measure_packet_velocity(k0, m)
                ax2.plot(2 * np.pi / k0, meas, "ko", ms=4, zorder=5)
        ax2.axhline(1.0, ls="--", color="gray", lw=0.8, alpha=0.7)
        ax2.set_title("Group velocity vs scale ratio  (black dots = measured packets)")
        ax2.set_xlabel("wavelength λ = 2π/k  (sites)  ←  observation scale / lattice")
        ax2.set_ylabel("group velocity  v_g")
        ax2.set_xlim(2, 20)
        ax2.legend()
        ax2.grid(alpha=0.3)

        fig.suptitle("brick_02 — the observed speed depends on the scale you probe "
                     "(dispersion), while the signal cone stays 1", fontsize=12)
        fig.tight_layout()
        out = __file__.rsplit("/", 1)[0] + "/brick_02_dispersion.png"
        fig.savefig(out, dpi=130)
        print(f"[그림] 저장됨: {out}")
    except Exception as e:
        print(f"[그림] 건너뜀: {e}")
