"""
brick_05 — SR의 고정된 c 를 '관찰자 스케일의 함수' c(E) 로 바꾸면, 실제 관측은 뭐라 하나?

네 질문 그대로: 특수상대성은 빛의 속도를 고정값으로 못박는다. 그것을 관찰하는
스케일(에너지)의 함수로 치환하면 — 즉 잘게 파고들수록(고에너지) c 가 살짝 달라진다면 —
무엇이 예상되고, 실제 데이터는 그걸 허용하는가?

이건 이 프로젝트에서 처음으로 '실제 관측'과 부딪히는 벽돌이다. 그리고 새 물리를
발견하는 게 아니라, 이미 있는 양자중력 현상론(이중특수상대성/DSR)의 예측을 재현하고
Fermi 망원경의 감마선 폭발 관측과 비교한다.

모형 (정직 표기: c(E) 를 손으로 넣는다, 창발이 아니다):
  수정된 빛의 속도:  v(E) = c0 · [ 1 - (E / E_QG)^n ]   (n=1: 선형, 가장 많이 검증된 경우)
  → 고에너지 광자가 아주 살짝 느리다. 우주적 거리를 지나면 시간차가 누적된다.
  선형(n=1) 시간지연 (Jacob-Piran 2008, 우주팽창 포함):
    Δt = (ΔE / E_QG) · (1/H0) · ∫_0^z (1+z') / sqrt(Ωm(1+z')^3 + ΩΛ) dz'

실제 사건: GRB 090510 (적색편이 z≈0.903). Fermi 가 31 GeV 광자를 폭발 시작
약 0.83초 안에 관측 → "c 의 스케일 의존이 이보다 크면 안 된다"는 한계.
"""

import numpy as np

# --- 상수 ---
E_PLANCK = 1.2209e19        # 플랑크 에너지 [GeV] : 스케일 의존이 '켜질' 자연 눈금
H0 = 2.184e-18             # 허블 상수 [1/s]  (67.4 km/s/Mpc)
INV_H0 = 1.0 / H0          # 허블 시간 [s] ≈ 4.58e17
OMEGA_M, OMEGA_L = 0.315, 0.685

# --- GRB 090510 (실제 관측) ---
Z_SRC = 0.903              # 적색편이
E_HIGH = 31.0             # 관측된 고에너지 광자 [GeV]
E_LOW = 1e-4              # 저에너지 기준 (100 keV) [GeV], E_HIGH 에 비해 무시할 수준
DT_OBSERVED = 0.83        # 이 광자가 도착한 시간 상한 [s] → 지연이 이보다 작아야 함


def cosmo_integral(z: float, npts: int = 20000) -> float:
    """I(z) = ∫_0^z (1+z') / sqrt(Ωm(1+z')^3 + ΩΛ) dz'  (선형 n=1 우주 거리인자)."""
    zp = np.linspace(0, z, npts)
    integrand = (1 + zp) / np.sqrt(OMEGA_M * (1 + zp) ** 3 + OMEGA_L)
    return np.trapezoid(integrand, zp)


def time_delay(E_high, E_low, z, E_QG, n: int = 1) -> float:
    """스케일 의존 c 로 인한 고-저 에너지 광자의 도착 시간차 [s] (선형 n=1)."""
    I = cosmo_integral(z)
    return ((E_high ** n - E_low ** n) / E_QG ** n) * INV_H0 * I


if __name__ == "__main__":
    print("=" * 72)
    print("brick_05 — 고정 c 를 c(E) 로 바꾸면? 실제 GRB 관측과의 대조")
    print("사건: GRB 090510  (z=0.903, 31 GeV 광자, Fermi)")
    print("=" * 72)

    I = cosmo_integral(Z_SRC)
    print(f"\n우주 거리인자 I(z={Z_SRC}) = {I:.3f}   (허블시간 1/H0 = {INV_H0:.2e} s)")

    # 1) 스케일 의존이 플랑크 스케일에서 켜진다고 가정하면, 예상 시간차는?
    dt_planck = time_delay(E_HIGH, E_LOW, Z_SRC, E_PLANCK)
    print(f"\n[예상] c 의 스케일 의존이 플랑크 스케일(E_QG=E_Planck)에서 켜진다면:")
    print(f"       31 GeV 광자의 지연 Δt ≈ {dt_planck:.2f} 초")

    # 2) 실제로는 그 광자가 {DT_OBSERVED}초 안에 도착했다 → 한계
    print(f"\n[관측] 그런데 실제로 그 광자는 약 {DT_OBSERVED}초 안에 도착했다.")
    print(f"       즉 지연은 {DT_OBSERVED}초보다 작아야 한다.")

    # 3) 관측 한계를 만족하려면 E_QG 가 얼마나 커야 하나?
    #    Δt = (ΔE/E_QG)·(1/H0)·I  ≤  DT_OBSERVED  →  E_QG ≥ ΔE·(1/H0)·I / DT_OBSERVED
    E_QG_limit = (E_HIGH - E_LOW) * INV_H0 * I / DT_OBSERVED
    print(f"\n[결론] 관측을 만족하려면  E_QG ≥ {E_QG_limit:.3e} GeV")
    print(f"       = {E_QG_limit / E_PLANCK:.2f} × 플랑크 에너지")
    print(f"       → 스케일 의존이 '켜지는' 눈금은 플랑크 스케일보다도 위로 밀렸다.")
    print(f"       즉 가장 단순한 선형 형태의 '스케일 의존 c' 는 플랑크 스케일에서")
    print(f"       이미 배제되는 쪽이다. 네 아이디어가 처음으로 실제 데이터에 부딪혔고,")
    print(f"       이 형태로는 살아남지 못한다(다른 형태·고차항은 아직 열려 있음).")

    print(f"\n[정직한 단서] (1) c(E) 는 손으로 넣었다 — 창발이 아니라 삽입. (2) 이건 표준")
    print(f"       DSR 현상론의 재현이지 새 발견이 아니다. (3) 그러나 '실제 관측과의 대조'")
    print(f"       라는 점에서 앞의 네 벽돌(가능성 시연)과 성격이 근본적으로 다르다.\n")

    # ---------- 그림 ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

        # (좌) 예상 지연 Δt vs E_QG, 플랑크선과 관측한계 표시
        E_QGs = np.logspace(17, 21, 400)
        dts = np.array([time_delay(E_HIGH, E_LOW, Z_SRC, e) for e in E_QGs])
        ax1.loglog(E_QGs, dts, lw=2, color="tab:blue",
                   label="predicted delay of 31 GeV photon")
        ax1.axhline(DT_OBSERVED, color="tab:red", ls="--", lw=1.5,
                    label=f"observed limit (~{DT_OBSERVED}s)")
        ax1.axvline(E_PLANCK, color="gray", ls=":", lw=1.2, label="Planck energy")
        ax1.fill_between(E_QGs, DT_OBSERVED, dts, where=dts > DT_OBSERVED,
                         color="tab:red", alpha=0.12)
        ax1.set_xlabel("E_QG  —  scale where c(E) turns on  [GeV]")
        ax1.set_ylabel("arrival-time delay Δt  [s]")
        ax1.set_title("Scale-dependent c vs the real GRB 090510 limit\n"
                      "(red band = excluded: predicts more delay than observed)")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3, which="both")

        # (우) E_QG=Planck 일 때, 광자 에너지별 예상 지연 (관측 가능한 신호 모양)
        Es = np.logspace(-3, 1.6, 200)   # 1 MeV ~ 40 GeV
        dts_E = np.array([time_delay(e, E_LOW, Z_SRC, E_PLANCK) for e in Es])
        ax2.loglog(Es, dts_E, lw=2, color="tab:green")
        ax2.axhline(DT_OBSERVED, color="tab:red", ls="--", lw=1.2,
                    label=f"observed limit (~{DT_OBSERVED}s)")
        ax2.axvline(E_HIGH, color="gray", ls=":", lw=1.0, label="31 GeV photon")
        ax2.set_xlabel("photon energy E  [GeV]")
        ax2.set_ylabel("predicted delay Δt  [s]  (if E_QG = Planck)")
        ax2.set_title("The observable signature:\nhigher-energy light would arrive later")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3, which="both")

        fig.tight_layout()
        out = __file__.rsplit("/", 1)[0] + "/brick_05_scale_dependent_c.png"
        fig.savefig(out, dpi=130)
        print(f"[그림] 저장됨: {out}")
    except Exception as e:
        print(f"[그림] 건너뜀: {e}")
