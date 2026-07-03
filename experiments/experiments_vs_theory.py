"""
brick_12 — 실측 데이터 vs 이론: 양자/고전 경계의 실제 수치들

네 제안: 관측(실험) 데이터로 "질량 대 임계 에너지"의 실제 수치를 확보하고, 이론(brick 10·11)과
비교해 관계를 확인하자. 여기서 두 가지를 한다.

(A) 물질파 간섭 '질량 기록'들을 이론 곡선(τ vs 질량) 위에 얹어, 데이터가 어디 있고
    미도달 최전선(나노그램·CMB·펜로즈)이 어디 비어 있는지 본다.

(B) '에너지를 넣으니 양자가 고전이 되는' 직접 측정 — Hackermüller 2004의 C70 열 결어긋남
    (내부온도 ↑ → 간섭 소멸)을 재현한다.

데이터 출처(웹 검색 확인):
  - C60 720 amu: Arndt et al., Nature 401, 680 (1999).
  - C70 840 amu 열결어긋남: Hackermüller et al., Nature 427, 711 (2004).
      "~1500 K까지 완전 대비, 2000 K 위 급감, ~3000 K 소멸; 경로분리 990 nm; 광자당 +140 K."
  - 유기분자 >1e4 amu: Eibenberger et al. (2013).
  - 올리고포르피린 27 kDa (~2000원자): Fein et al., Nat. Phys. (2019), LUMI 2 m 기선. 질량 기록.
정직 표기: 질량은 출처 수치, Hackermüller 곡선은 보고된 특징으로부터의 재구성(원 데이터 아님).
"""

import numpy as np

HBAR = 1.0546e-34
KB = 1.381e-23
C = 2.998e8
G = 6.674e-11
AMU = 1.6605e-27
RHO = 2000.0
T_CMB, T_ROOM = 2.725, 300.0
K_JZ = 1.1e4 * 0.0625


def radius(m):
    return (3 * m / (4 * np.pi * RHO)) ** (1 / 3)


def tau_photon(m, T):
    a = radius(m)
    Lam = K_JZ * a ** 6 * (KB * T) ** 9 / (HBAR ** 9 * C ** 8)
    return 1.0 / (Lam * a ** 2)


def tau_penrose(m):
    a = radius(m)
    return HBAR / ((6 / 5) * G * m ** 2 / a)


# 실측 질량 기록 (출처: 위 주석)  (이름, 질량[amu], 연도)
EXPERIMENTS = [
    ("electron (1927)",      5.49e-4, 1927),
    ("atom, Na (1991)",      23.0,    1991),
    ("C60 (1999)",           720.0,   1999),
    ("C70 thermal (2004)",   840.0,   2004),
    ("organic >1e4 (2013)",  1.0e4,   2013),
    ("27 kDa, ~2000 atoms (2019)", 2.7e4, 2019),
]


def hackermuller_visibility(T):
    """C70 열 결어긋남 재현 (보고 특징: 1500K 완전, 2000K↑ 급감, 3000K 소멸)."""
    return np.exp(-(T / 2500.0) ** 8)


if __name__ == "__main__":
    print("=" * 76)
    print("brick_12 — 실측 데이터 vs 이론 (양자/고전 경계의 실제 수치)")
    print("=" * 76)

    print("\n[A] 물질파 간섭 질량 기록 (실측) — 그 질량에서 이론이 주는 코히런스 수명:")
    print(f"{'실험':<30} | {'질량[amu]':>9} | {'τ_CMB[s]':>10} | {'τ_상온[s]':>10}")
    print("-" * 76)
    for name, amu, yr in EXPERIMENTS:
        m = amu * AMU
        print(f"{name:<30} | {amu:>9.1e} | {tau_photon(m, T_CMB):>10.1e} | {tau_photon(m, T_ROOM):>10.1e}")
    print("-" * 76)
    print("→ 모든 실측은 저질량(~1~3e4 amu)에 몰려 있고, 이론상 진공+냉각이면 양자 유지 가능한")
    print("  영역이다. CMB·펜로즈가 무는 나노그램(~1e-14~1e-11 kg) 경계는 아직 데이터 없음(최전선).")

    print("\n[B] '에너지 넣으니 양자→고전' 직접 측정 (Hackermüller 2004, C70 840 amu):")
    for T in [1000, 1500, 2000, 2500, 3000]:
        print(f"       내부온도 {T:>5} K → 간섭 대비(재현) {hackermuller_visibility(T):>5.2f}")
    print("       측정 사실: ~1500K 완전한 양자 대비 → 2000K 위 급감 → ~3000K 소멸.")
    print("       이것이 '질량 840 amu 물체를 고전화시키는 에너지 임계'의 실제 수치.")

    print("\n[관계] 실측은 이론의 스케일링(무거울수록·뜨거울수록 결어긋남 급증)과 정성적으로 일치.")
    print("       그러나 데이터가 성겨(1~3e4 amu) 정량적 '관계식'을 넓게 고정하진 못한다 —")
    print("       나노그램 최전선(부양 나노입자)이 채워지면 비로소 관계가 넓게 확정된다.")
    print("\n[정직한 단서] 질량은 출처 수치, Hackermüller 곡선은 보고 특징의 재구성(원 데이터 아님).")
    print("       정확한 값은 원 논문 대조 필요.\n")

    # ---------- 그림 ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

        # (좌) 질량-지도: 이론 곡선 + 실측 질량
        ms = np.logspace(-27, -9, 400)
        ax1.loglog(ms, [tau_photon(m, T_CMB) for m in ms], color="tab:cyan", lw=2,
                   label="CMB photons (2.725 K)")
        ax1.loglog(ms, [tau_photon(m, T_ROOM) for m in ms], color="tab:orange", lw=2,
                   label="room thermal photons (300 K)")
        ax1.loglog(ms, [tau_penrose(m) for m in ms], color="tab:purple", lw=1.5, ls="-.",
                   label="Penrose-Diósi (speculative)")
        ax1.axhline(1.0, ls="--", color="gray", lw=1, label="τ = 1 s")
        for name, amu, yr in EXPERIMENTS:
            m = amu * AMU
            ax1.axvline(m, color="tab:green", ls=":", lw=0.8, alpha=0.6)
        ax1.scatter([amu * AMU for _, amu, _ in EXPERIMENTS],
                    [1e-2] * len(EXPERIMENTS), color="tab:green", s=30, zorder=5,
                    label="real experiments (measured mass)")
        ax1.text(3e4 * AMU, 3e-2, "27 kDa\n(2019 record)", fontsize=7, color="tab:green")
        ax1.axvspan(1e-14, 1e-9, color="red", alpha=0.06)
        ax1.text(3e-13, 1e12, "unreached\nfrontier\n(no data)", fontsize=8,
                 color="tab:red", ha="center")
        ax1.set_xlabel("mass  [kg]")
        ax1.set_ylabel("coherence lifetime  τ  [s]")
        ax1.set_title("(A) real interference records vs theory")
        ax1.legend(fontsize=7, loc="lower left")
        ax1.grid(alpha=0.3, which="both")
        ax1.set_ylim(1e-8, 1e35)

        # (우) Hackermüller 직접 측정 재현: 간섭 대비 vs 내부온도
        Ts = np.linspace(500, 3500, 300)
        ax2.plot(Ts, hackermuller_visibility(Ts), lw=2.4, color="tab:red")
        ax2.axvspan(2000, 3000, color="gray", alpha=0.12)
        ax2.annotate("full quantum\ncontrast (~1500 K)", xy=(1500, hackermuller_visibility(1500)),
                     xytext=(700, 0.4), fontsize=8,
                     arrowprops=dict(arrowstyle="->", color="gray"))
        ax2.annotate("interference gone\n(~3000 K)", xy=(3000, hackermuller_visibility(3000)),
                     xytext=(2600, 0.55), fontsize=8,
                     arrowprops=dict(arrowstyle="->", color="gray"))
        ax2.set_xlabel("internal temperature of C70  [K]  (= added energy)")
        ax2.set_ylabel("interference visibility  (quantum-ness)")
        ax2.set_title("(B) adding energy turns quantum → classical\n"
                      "C70 thermal decoherence (Hackermüller 2004, reconstructed)")
        ax2.grid(alpha=0.3)
        ax2.set_ylim(0, 1.05)

        fig.tight_layout()
        out = __file__.rsplit("/", 1)[0] + "/brick_12_experiments_vs_theory.png"
        fig.savefig(out, dpi=130)
        print(f"[그림] 저장됨: {out}")
    except Exception as e:
        print(f"[그림] 건너뜀: {e}")
