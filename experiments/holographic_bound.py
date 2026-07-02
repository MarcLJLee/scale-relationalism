"""
brick_06 — 정보적 관점이 표준 직감과 다른 예측을 하는 곳: 홀로그래피 경계

물리를 '정보'로 보면 표준(소박한) 직감과 다른 예측이 나오는 대표 지점이 이것이다.

  표준/소박한 직감:  어떤 영역에 담을 수 있는 최대 정보량 ∝ 부피 (R^3).
                    (상자가 크면 그만큼 많은 비트를 넣을 수 있다 — 국소 장론적 직감)
  정보적 관점(홀로그래피):  최대 정보량 ∝ 표면적 (R^2).
                    (베켄슈타인 한계 / 'tHooft-Susskind 홀로그래피 원리)

두 예측은 R/ℓ_P 배만큼 어긋난다. 즉 실제 세계는 소박한 직감보다 어마어마하게
'적은' 정보만 담을 수 있다는 것 — 이것이 정보적 관점의 서로 다른, 정량적 예측이다.

왜 표면적이 이기나 (블랙홀 논증):
  한 영역에 정보(=에너지)를 계속 채워 넣으면, 부피를 다 채우기 전에 블랙홀이 된다.
  블랙홀은 그 크기가 담을 수 있는 '최대 엔트로피' 상태이고, 그 엔트로피는
  표면적에 비례한다(S = A / 4ℓ_P^2). 그래서 부피가 아니라 표면적이 진짜 한계다.
"""

import numpy as np

L_P = 1.616e-35        # 플랑크 길이 [m]
LN2 = np.log(2)


def bits_volume(R):
    """소박한 직감: 플랑크 부피당 1비트  →  (R/ℓ_P)^3."""
    return (R / L_P) ** 3


def bits_holographic(R):
    """홀로그래피: S = A/(4ℓ_P^2), A=4πR^2  →  비트수 = πR^2/(ℓ_P^2·ln2)."""
    return np.pi * (R / L_P) ** 2 / LN2


if __name__ == "__main__":
    regions = [
        ("양성자  ~1e-15 m", 1e-15),
        ("원자    ~1e-10 m", 1e-10),
        ("세포    ~1e-5 m",  1e-5),
        ("1 cm 공",          1e-2),
        ("1 m 공",           1.0),
        ("지구    ~6.4e6 m",  6.4e6),
        ("태양    ~7e8 m",    7e8),
        ("관측우주 ~4.4e26 m", 4.4e26),
    ]

    print("=" * 78)
    print("brick_06 — 정보 용량: 부피(표준 직감) vs 표면적(정보적/홀로그래피)")
    print("=" * 78)
    print(f"\n{'영역':<18} | {'부피 예측 (log10 비트)':>20} | {'표면적 예측 (log10 비트)':>22} | {'과대계상 배율':>12}")
    print("-" * 78)
    for name, R in regions:
        v = bits_volume(R)
        h = bits_holographic(R)
        ratio = v / h
        print(f"{name:<18} | {np.log10(v):>20.1f} | {np.log10(h):>22.1f} | 10^{np.log10(ratio):>6.1f}")
    print("-" * 78)

    print("\n[핵심] 두 예측은 R/ℓ_P 배만큼 벌어진다. 1 m 공만 해도 부피 직감은")
    print("       홀로그래피 한계보다 약 10^34 배 많은 정보를 담을 수 있다고 '틀리게' 말한다.")
    print("       → 실제 세계가 담을 수 있는 정보는 소박한 직감보다 어마어마하게 적다.")
    print("\n[왜 표면적이 이기나] 부피를 정보로 채우려 하면 다 채우기 전에 블랙홀이 된다.")
    print("       블랙홀 = 그 크기의 최대 엔트로피 = 표면적 법칙(S=A/4). 그래서 표면적이")
    print("       진짜 한계이고, 부피 직감은 어떤 거시 영역에서도 이 한계를 위반한다.")
    print("\n[표준과 다른 예측인가?] 그렇다. 국소 장론의 소박한 자유도 세기는 부피(R^3)를,")
    print("       정보적/홀로그래피 관점은 표면적(R^2)을 예측한다 — 정성적으로 다른 스케일링.")
    print("\n[정직한 단서] (1) 재현이지 발견이 아니다(베켄슈타인·홀로그래피 원리). (2) 직접")
    print("       실험은 불가(플랑크 규모 저장장치를 못 만든다). 단 블랙홀 엔트로피 면적법칙")
    print("       으로 이론적으로는 견고하다. (3) Fermilab '홀로미터'가 관련 홀로그래픽 잡음을")
    print("       탐색했으나 null(2015) — 특정 모델을 배제. 즉 부분적으로 실측이 건드린 영역.\n")

    # ---------- 그림 ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        Rs = np.logspace(-15, 27, 400)
        v = bits_volume(Rs)
        h = bits_holographic(Rs)

        fig, ax = plt.subplots(figsize=(9, 5.4))
        ax.loglog(Rs, v, lw=2, color="tab:orange",
                  label="naive: capacity ∝ volume (R³)")
        ax.loglog(Rs, h, lw=2, color="tab:blue",
                  label="informational (holography): ∝ area (R²)")
        ax.fill_between(Rs, h, v, where=v > h, color="tab:orange", alpha=0.10)
        # 몇몇 실제 영역 표시
        for name, R in [("proton", 1e-15), ("1 m", 1.0),
                        ("Earth", 6.4e6), ("obs. universe", 4.4e26)]:
            ax.axvline(R, color="gray", ls=":", lw=0.7, alpha=0.6)
            ax.text(R, 1e5, name, rotation=90, fontsize=7,
                    va="bottom", ha="right", color="gray")
        ax.set_xlabel("region size R  [m]")
        ax.set_ylabel("maximum information capacity  [bits]")
        ax.set_title("brick_06 — how much information fits in a region?\n"
                     "naive volume-scaling (R³) vs informational area-scaling (R²)")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        out = __file__.rsplit("/", 1)[0] + "/brick_06_holographic_bound.png"
        fig.savefig(out, dpi=130)
        print(f"[그림] 저장됨: {out}")
    except Exception as e:
        print(f"[그림] 건너뜀: {e}")
