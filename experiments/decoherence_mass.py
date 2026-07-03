"""
brick_10 — "코히런스를 유지하는 최소 에너지"와 질량의 관계: 양자/고전 경계

네 직감: 물체가 두 위치에 중첩돼 있을 때, 그 두 갈래를 "구별"하는 최소 에너지와
대상 질량 사이에 관계가 있지 않을까 — 그리고 그 관계가 양자/고전 경계를 정하지 않을까.

여기서 두 메커니즘의 '코히런스 수명 τ(질량)'을 계산해 비교한다.

(1) 펜로즈-디오시 중력 자기붕괴 (speculative, 검증 중):
    두 갈래를 구별하는 최소 에너지 = 중력 자체에너지  E_G ≈ (6/5)·G·m²/R  (균일 구, 완전 분리)
    중첩이 스스로 무너지는 시간            τ_G ≈ ℏ / E_G
    → 질량↑ ⇒ E_G↑ ⇒ τ_G↓ (더 빨리 고전으로). 질량에 대한 구체적 함수.

(2) 환경 결어긋남 (established, 실제로 큰 물체를 고전으로 만드는 주된 힘):
    잔류 기체 분자가 물체(단면 πR²)에 충돌하는 비율이 결어긋남을 일으킴
    τ_env ≈ 1 / (n·v·πR²),   n=기체 수밀도, v=열속도  (포화 영역 근사)

정직 표기: (1)은 미확립 제안(지하·부양 실험으로 시험 중), (2)는 확립. 관측 가능한 영역에선
대개 (2)가 (1)보다 빨라 지배한다 → 펜로즈를 시험하려면 환경 결어긋남을 (1) 아래로 낮춰야.
"""

import numpy as np

G = 6.674e-11         # 중력상수
HBAR = 1.0546e-34    # 플랑크 상수/2π
KB = 1.381e-23       # 볼츠만
AMU = 1.6605e-27     # 원자질량단위 [kg]

RHO = 2000.0          # 물질 밀도 [kg/m^3] (유기물·실리카급)

# 환경(초고진공 UHV) 설정
P_VAC = 1e-6          # 압력 [Pa] (=1e-8 mbar, 좋은 UHV)
T_ENV = 300.0         # 온도 [K]
M_GAS = 28 * AMU      # 잔류 기체 분자질량 (N2)


def radius(m):
    """질량 m 의 구 반지름 (밀도 RHO)."""
    return (3 * m / (4 * np.pi * RHO)) ** (1 / 3)


def tau_penrose(m):
    """펜로즈-디오시 중력 붕괴 코히런스 수명 [s]."""
    R = radius(m)
    E_G = (6 / 5) * G * m ** 2 / R      # 최소 구별 에너지 (중력 자체에너지)
    return HBAR / E_G


def tau_env(m):
    """환경(UHV 잔류기체) 결어긋남 코히런스 수명 [s]."""
    R = radius(m)
    n = P_VAC / (KB * T_ENV)                     # 기체 수밀도
    v = np.sqrt(8 * KB * T_ENV / (np.pi * M_GAS))  # 평균 열속도
    rate = n * v * np.pi * R ** 2                 # 충돌(결어긋남) 비율
    return 1.0 / rate


def crossing_mass(tau_func, tau_target):
    """tau_func(m) = tau_target 인 질량을 이분법으로."""
    lo, hi = 1e-27, 1e-6
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if tau_func(mid) > tau_target:   # τ는 질량에 대해 감소함수
            lo = mid
        else:
            hi = mid
    return np.sqrt(lo * hi)


if __name__ == "__main__":
    print("=" * 76)
    print("brick_10 — 코히런스 수명 τ vs 질량 (양자/고전 경계)")
    print(f"밀도 {RHO:.0f} kg/m³, UHV {P_VAC:.0e} Pa, {T_ENV:.0f} K")
    print("=" * 76)

    rows = [
        ("원자 ~1 amu",        1 * AMU),
        ("분자 ~1e4 amu",      1e4 * AMU),
        ("최대중첩분자 ~1e5",   1e5 * AMU),
        ("바이러스 ~1e9 amu",   1e9 * AMU),
        ("나노입자 ~1e-18 kg",  1e-18),
        ("나노입자 ~1e-15 kg",  1e-15),
        ("먼지 ~1e-12 kg",      1e-12),
    ]
    print(f"\n{'대상':<20} | {'질량[kg]':>10} | {'반지름[m]':>10} | {'τ_펜로즈[s]':>12} | {'τ_환경[s]':>12} | 지배")
    print("-" * 76)
    for name, m in rows:
        tG, tE = tau_penrose(m), tau_env(m)
        dom = "환경" if tE < tG else "펜로즈"
        print(f"{name:<20} | {m:>10.1e} | {radius(m):>10.1e} | {tG:>12.1e} | {tE:>12.1e} | {dom}")
    print("-" * 76)

    # 관측 가능(τ = 1 s) 경계
    m_env_1s = crossing_mass(tau_env, 1.0)
    m_pen_1s = crossing_mass(tau_penrose, 1.0)
    print(f"\n[경계] τ = 1초 코히런스를 유지할 수 있는 최대 질량:")
    print(f"       환경 결어긋남 한계 : {m_env_1s:.1e} kg  (~{m_env_1s/AMU:.0e} amu)")
    print(f"       펜로즈 중력붕괴 한계: {m_pen_1s:.1e} kg  (~{m_pen_1s/AMU:.0e} amu)")
    print(f"\n[해석] 관측 영역에선 환경 결어긋남이 훨씬 빨라 지배한다 — 큰 물체가 고전인")
    print(f"       주된 이유는 환경(확립). 펜로즈 중력붕괴는 '완벽히 고립해도 남는 내재적 바닥'")
    print(f"       이며, 환경 한계({m_env_1s:.0e} kg)와 펜로즈 한계({m_pen_1s:.0e} kg) 사이의 큰 간극이")
    print(f"       곧 '펜로즈를 검증하려면 넘어야 할 고립의 난이도'다.")
    print(f"\n[정직한 단서] (1) 펜로즈-디오시는 speculative(부양 나노입자·지하 실험으로 시험 중).")
    print(f"       (2) 환경 결어긋남은 established. (3) 실제 최대 물질파 중첩은 ~1e4~1e5 amu")
    print(f"       (분자 간섭계) — 표의 환경 한계와 대략 부합. 네 '최소에너지↔질량'은 (1)의 E_G.\n")

    # ---------- 그림 ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ms = np.logspace(-27, -11, 400)
        tG = np.array([tau_penrose(m) for m in ms])
        tE = np.array([tau_env(m) for m in ms])

        fig, ax = plt.subplots(figsize=(9, 5.6))
        ax.loglog(ms, tG, lw=2, color="tab:purple",
                  label="Penrose-Diósi gravitational collapse (speculative)")
        ax.loglog(ms, tE, lw=2, color="tab:blue",
                  label="environmental decoherence, UHV (established)")
        ax.axhline(1.0, ls="--", color="gray", lw=1, label="τ = 1 s (observable)")
        # 경계 표시
        ax.axvline(m_env_1s, ls=":", color="tab:blue", lw=0.8, alpha=0.7)
        ax.axvline(m_pen_1s, ls=":", color="tab:purple", lw=0.8, alpha=0.7)
        # 실제 실험 위치
        ax.axvspan(1e4 * AMU, 1e5 * AMU, color="tab:green", alpha=0.12)
        ax.text(3e4 * AMU, 1e-9, "largest matter-wave\nsuperpositions (~10⁴–10⁵ amu)",
                fontsize=7, color="tab:green", ha="center")
        ax.set_xlabel("mass  [kg]")
        ax.set_ylabel("coherence lifetime  τ  [s]")
        ax.set_title("brick_10 — quantum coherence lifetime vs mass\n"
                     "which mechanism forces classicality, and where the boundary sits")
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(alpha=0.3, which="both")
        ax.set_ylim(1e-12, 1e20)
        fig.tight_layout()
        out = __file__.rsplit("/", 1)[0] + "/brick_10_decoherence_mass.png"
        fig.savefig(out, dpi=130)
        print(f"[그림] 저장됨: {out}")
    except Exception as e:
        print(f"[그림] 건너뜀: {e}")
