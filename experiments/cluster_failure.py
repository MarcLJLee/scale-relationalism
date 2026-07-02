"""
brick_08 — 이론이 깨지는 곳: 은하단에서 스케일-수정 중력이 실패한다

brick_07 에서 '중력이 스케일 a0 아래에서 변한다'(MOND)를 은하에 대입하니 암흑물질
없이 회전곡선이 맞았다. 그러나 좋은 이론은 어디서 깨지는지도 정직하게 보여야 한다.

여기서는 같은 규칙을 은하단(galaxy cluster)에 대입하면 어떻게 되는지 본다.

핵심 이유 — 왜 은하에선 성공하고 은하단에선 실패하나:
  MOND 보정(부족 질량 배율)은  b(g) = 1 / (1 - exp(-sqrt(g/a0)))  꼴이고,
  아주 낮은 가속도(g << a0)에서만 크게 커진다(b ≈ sqrt(a0/g)).
    - 은하 외곽: g ≪ a0  → 보정이 커서 부족분을 다 메움. (성공)
    - 은하단 중심부: g ~ a0 → 보정이 작아서, 관측이 요구하는 부족분의 절반밖에 못 메움.
  결과: 은하단은 MOND 를 써도 여전히 '약 2배'의 질량이 빈다 → 여전히 암흑물질(또는
  중성미자 등)을 요구한다. 이것이 잘 알려진 'MOND 은하단 문제'다.

주의(정직 표기): 아래 은하단 점들은 특정 데이터 적합이 아니라, 문헌에서 잘 알려진
'MOND 잔여 부족 질량 배율 ~2'(Sanders 1999; Pointecouteau & Silk 2005; Angus+ 2008)를
대표하는 값이다. 은하 점들은 관측된 촘촘한 RAR 위에 놓인다.
"""

import numpy as np

A0 = 1.2e-10   # 특징적 가속도 스케일 [m/s^2]


def mond_boost(g_bar):
    """스케일-수정 중력이 예측하는 부족 질량 배율 (관측가속도/바리온가속도)."""
    return 1.0 / (1.0 - np.exp(-np.sqrt(g_bar / A0)))


# 대표 시스템: (이름, 바리온 가속도 g_bar[m/s^2], 관측된 부족질량 배율, 종류)
#   은하: 관측된 RAR 위에 놓임 (MOND 예측 = 관측)
#   은하단: 관측 배율이 MOND 예측보다 ~2.2배 위 (잔여 부족)
CLUSTER_RESIDUAL = 2.2
systems = []
for name, en, g in [("왜소은하 외곽", "dwarf outskirt", 3e-12),
                    ("나선은하 외곽", "spiral outskirt", 1e-11),
                    ("나선은하 중간", "spiral mid", 8e-11)]:
    systems.append((name, en, g, mond_boost(g), "galaxy"))
for name, en, g in [("은하단 외곽", "cluster outskirt", 8e-11),
                    ("은하단 중간", "cluster mid", 2e-10),
                    ("은하단 중심부", "cluster core", 5e-10)]:
    systems.append((name, en, g, CLUSTER_RESIDUAL * mond_boost(g), "cluster"))


if __name__ == "__main__":
    print("=" * 76)
    print("brick_08 — 이론이 깨지는 곳: 은하단에서의 스케일-수정 중력")
    print("부족 질량 배율 = 관측이 요구하는 중력 / 보이는 물질의 중력")
    print("=" * 76)
    print(f"\n{'시스템':<14} | {'g_bar/a0':>9} | {'MOND 예측 배율':>13} | {'관측 배율':>9} | {'잔여(관측/MOND)':>14}")
    print("-" * 76)
    for name, en, g, obs, kind in systems:
        pred = mond_boost(g)
        residual = obs / pred
        tag = "" if kind == "galaxy" else "  ← 여전히 부족"
        print(f"{name:<14} | {g/A0:>9.2f} | {pred:>13.2f} | {obs:>9.2f} | {residual:>14.2f}{tag}")
    print("-" * 76)

    print("\n[은하] g ≪ a0 (깊은 MOND). 보정이 커서 관측 배율 = MOND 예측. 잔여 ≈ 1 → 성공.")
    print("[은하단] g ~ a0. 보정이 작아서 MOND 예측이 관측의 절반쯤. 잔여 ≈ 2.2 → 실패.")
    print("        → 은하단은 스케일-수정 중력을 써도 여전히 '약 2배' 질량이 빈다.")
    print("        여전히 암흑물질(또는 중성미자 등)을 요구 → 주류가 입자 암흑물질을 놓지 못하는 이유.")

    print("\n[왜 갈리나] MOND 보정은 g가 a0보다 훨씬 낮을 때만 크게 작동한다(b≈sqrt(a0/g)).")
    print("        은하 외곽은 그 조건을 만족하지만, 은하단 핵심 영역은 가속도가 a0 근처라")
    print("        보정이 약하다. 즉 '스케일이 달라지는 문턱(a0)'이 은하엔 맞고 은하단엔 안 맞는다.")

    print("\n[의미] 하나의 특징적 스케일 a0 로 은하와 은하단을 동시에 설명하지 못한다.")
    print("        스케일-관계 아이디어가 틀렸다는 게 아니라, '단일 문턱' 형태로는 부족하다는 것.")
    print("        brick_07(성공)과 brick_08(실패)를 나란히 두는 것이 정직한 이력이다.\n")

    # ---------- 그림 ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        g = np.logspace(-12.5, -8.5, 400)
        curve = mond_boost(g)

        fig, ax = plt.subplots(figsize=(9, 5.6))
        ax.loglog(g / A0, curve, lw=2, color="tab:blue",
                  label="scale-modified prediction (MOND/RAR)")
        ax.axhline(1.0, ls=":", color="gray", lw=0.8)
        ax.axvline(1.0, ls=":", color="gray", lw=0.8)
        ax.text(1.05, 1.05, "g = a₀", color="gray", fontsize=8)

        for name, en, gg, obs, kind in systems:
            color = "tab:green" if kind == "galaxy" else "tab:red"
            marker = "o" if kind == "galaxy" else "s"
            ax.scatter(gg / A0, obs, color=color, marker=marker, s=70, zorder=5,
                       edgecolor="k", linewidth=0.5)
            ax.annotate(en, (gg / A0, obs), fontsize=7,
                        textcoords="offset points", xytext=(6, 4))

        ax.scatter([], [], color="tab:green", marker="o", edgecolor="k",
                   label="galaxies — on the curve (success)")
        ax.scatter([], [], color="tab:red", marker="s", edgecolor="k",
                   label="clusters — above the curve (residual ~2× → fails)")
        ax.set_xlabel("baryonic acceleration  g_bar / a₀")
        ax.set_ylabel("missing-mass factor  (M_dyn / M_bar)")
        ax.set_title("brick_08 — one scale a₀ fits galaxies but not clusters\n"
                     "clusters need ~2× more mass than scale-modified gravity gives")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        out = __file__.rsplit("/", 1)[0] + "/brick_08_cluster_failure.png"
        fig.savefig(out, dpi=130)
        print(f"[그림] 저장됨: {out}")
    except Exception as e:
        print(f"[그림] 건너뜀: {e}")
