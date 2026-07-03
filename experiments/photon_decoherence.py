"""
brick_11 — 광자(CMB·열복사)라는 '가장 공평한 잣대'로 본 양자/고전 경계

직감: 기체는 진공으로 빼낼 수 있지만, 광자(열복사)는 못 뺀다. 온도 있는 모든 것이 광자를
방출·산란·흡수하고, 궁극의 바닥은 우주배경복사(CMB, 2.725 K) — 우주 어디든 모든 것이
잠긴 광자 목욕탕. 그러니 '광자 에너지 기준'이 가장 보편적(제거 불가능)인 양자/고전 잣대다.

여기서 열광자 산란에 의한 결어긋남 수명 τ(질량)을 계산한다 (Joos-Zeh, Rayleigh 영역):
    국소화율  Λ ≈ K · a⁶ · (k_B T)⁹ / (ℏ⁹ c⁸)      [1/(m²·s)]
    결어긋남율 Γ = Λ · (Δx)²,   여기서 Δx = a  (제 크기만큼 떨어진 '거시적으로 구별되는' 중첩)
    수명       τ = 1/Γ
  K = (기하 전치인자 ~1.1e4) × (유전율 인자 [(ε−1)/(ε+2)]², ε=2 → 0.0625)

핵심 스케일: Γ ∝ a⁸ T⁹ ∝ m^(8/3) T⁹  → 무거울수록·뜨거울수록 급격히 양자를 잃는다.
정직 표기: Rayleigh(장파장) 근사 + Δx=a 기준의 차수(order-of-magnitude) 추정.
"""

import numpy as np

HBAR = 1.0546e-34
KB = 1.381e-23
C = 2.998e8
AMU = 1.6605e-27
RHO = 2000.0                     # 물질 밀도 [kg/m^3]

K_JZ = 1.1e4 * 0.0625            # 기하 전치인자 × 유전율 인자 (ε=2)

T_CMB = 2.725                    # 우주배경복사 [K]
T_ROOM = 300.0                   # 상온 열복사 [K]


def radius(m):
    return (3 * m / (4 * np.pi * RHO)) ** (1 / 3)


def tau_photon(m, T):
    """열광자(온도 T) 산란 결어긋남 수명 [s]. Δx = 물체 크기 a."""
    a = radius(m)
    Lambda = K_JZ * a ** 6 * (KB * T) ** 9 / (HBAR ** 9 * C ** 8)
    Gamma = Lambda * a ** 2
    return 1.0 / Gamma


def thermal_wavelength(T):
    return HBAR * C / (KB * T)


def crossing_mass(T, tau_target=1.0):
    lo, hi = 1e-24, 1e-6
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if tau_photon(mid, T) > tau_target:
            lo = mid
        else:
            hi = mid
    return np.sqrt(lo * hi)


if __name__ == "__main__":
    print("=" * 78)
    print("brick_11 — 광자(CMB·열복사) 결어긋남 vs 질량: 가장 공평한 양자/고전 잣대")
    print(f"밀도 {RHO:.0f} kg/m³, Δx=물체크기, CMB {T_CMB} K / 상온 {T_ROOM} K")
    print("=" * 78)
    print(f"\nCMB 광자 파장 ~{thermal_wavelength(T_CMB)*1e3:.2f} mm, 광자에너지 ~{KB*T_CMB/1.602e-19*1e3:.2f} meV")

    rows = [
        ("분자 ~1e5 amu",     1e5 * AMU),
        ("바이러스 ~1e9 amu",  1e9 * AMU),
        ("나노 ~1e-18 kg",     1e-18),
        ("나노 ~1e-15 kg",     1e-15),
        ("~1e-12 kg (1pg)",    1e-12),
        ("~1e-10 kg",          1e-10),
    ]
    print(f"\n{'대상':<18} | {'질량[kg]':>10} | {'반지름[m]':>10} | {'τ_CMB[s]':>12} | {'τ_상온[s]':>12}")
    print("-" * 78)
    for name, m in rows:
        print(f"{name:<18} | {m:>10.1e} | {radius(m):>10.1e} | "
              f"{tau_photon(m, T_CMB):>12.1e} | {tau_photon(m, T_ROOM):>12.1e}")
    print("-" * 78)

    m_cmb = crossing_mass(T_CMB)
    m_room = crossing_mass(T_ROOM)
    print(f"\n[경계] τ = 1초 코히런스 유지 최대 질량 (광자 결어긋남):")
    print(f"       CMB(2.725 K, 제거 불가능) : {m_cmb:.1e} kg  (~{m_cmb/AMU:.0e} amu)")
    print(f"       상온(300 K, 물체가 따뜻하면): {m_room:.1e} kg  (~{m_room/AMU:.0e} amu)")

    print(f"\n[해석] 광자는 못 뺀다 — 기체를 다 빼도 CMB가 남는다. CMB는 매우 차가워서(2.7K)")
    print(f"       결어긋남이 느려, 무려 ~{m_cmb*1e12:.0f} ng 급까지 양자 중첩을 허용한다(궁극의 바닥).")
    print(f"       그러나 물체가 상온이면 제 열복사가 {m_cmb/m_room:.0e}배 강해 훨씬 작은 질량에서")
    print(f"       고전화 → 그래서 실험은 물체를 극저온으로 식혀 CMB 바닥에 다가간다.")
    print(f"\n[네 직감 확인] Γ ∝ 질량^(8/3)·T⁹ → 무거울수록 낮은 광자에너지(차가운 CMB)에도")
    print(f"       양자를 잃는다. CMB 광자에너지는 ~0.23 meV로 극히 낮은데도 무거운 물체는")
    print(f"       (긴 파장으로도 구별되고 단면적도 커서) 결국 고전화된다. 정확히 네 주장.")
    print(f"\n[정직한 단서] Joos-Zeh 열광자 산란(Rayleigh·Δx=a) 차수 추정. 광자는 실질적으로")
    print(f"       가장 공평(제거 불가능)하지만, 원리적으로 더 보편적인 건 더 약한 중력(brick_10).\n")

    # ---------- 그림 ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ms = np.logspace(-24, -9, 400)
        tC = np.array([tau_photon(m, T_CMB) for m in ms])
        tR = np.array([tau_photon(m, T_ROOM) for m in ms])

        fig, ax = plt.subplots(figsize=(9, 5.6))
        ax.loglog(ms, tC, lw=2.2, color="tab:cyan",
                  label="CMB photons (2.725 K) — universal floor, unremovable")
        ax.loglog(ms, tR, lw=2, color="tab:orange",
                  label="room-temperature thermal photons (300 K)")
        ax.axhline(1.0, ls="--", color="gray", lw=1, label="τ = 1 s (observable)")
        ax.axvline(m_cmb, ls=":", color="tab:cyan", lw=0.9, alpha=0.8)
        ax.axvline(m_room, ls=":", color="tab:orange", lw=0.9, alpha=0.8)
        ax.set_xlabel("mass  [kg]")
        ax.set_ylabel("photon-decoherence lifetime  τ  [s]")
        ax.set_title("brick_11 — the most universal yardstick: photon (thermal) decoherence\n"
                     "CMB is the unremovable floor; warmth (self-emission) is far stronger")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3, which="both")
        ax.set_ylim(1e-10, 1e25)
        fig.tight_layout()
        out = __file__.rsplit("/", 1)[0] + "/brick_11_photon_decoherence.png"
        fig.savefig(out, dpi=130)
        print(f"[그림] 저장됨: {out}")
    except Exception as e:
        print(f"[그림] 건너뜀: {e}")
