"""
brick_07 — 이론을 대입했더니 흥미로운 현상: 은하 회전과 암흑물질

지금까지의 벽돌은 대체로 '당연한 재기술'이었다. 여기서는 스케일-관계 아이디어를
실제 수수께끼에 '대입'하면 놀라운 결과가 나오는지 본다.

수수께끼: 은하 바깥쪽 별들은 보이는 물질만으론 설명 안 될 만큼 빨리 돈다.
  표준 해법 → 보이지 않는 '암흑물질'을 잔뜩 넣는다(우주의 대부분).
  대입할 아이디어 → "중력이 어떤 특징적 가속도 스케일 a0 아래에서 달라진다."
                    (스케일에 따라 물리 규칙이 변한다는 이 프로젝트의 명제)

두 관점이 같은 데이터를 다르게 읽는다:
  (A) 뉴턴 중력 + 안 보이는 물질을 추가        → 암흑물질
  (B) 물질은 보이는 것뿐 + 중력이 스케일에 따라 변함 → 스케일-관계적 수정

(B)의 구체형: 관측 가속도 g_obs 가 보이는 물질의 가속도 g_bar 만의 함수라는
경험식(Radial Acceleration Relation, McGaugh+2016):
    g_obs = g_bar / (1 - exp(-sqrt(g_bar / a0))),   a0 ≈ 1.2e-10 m/s^2
  높은 가속도(g_bar >> a0): g_obs ≈ g_bar (뉴턴 그대로)
  낮은 가속도(g_bar << a0): g_obs ≈ sqrt(g_bar·a0)  → 회전곡선이 평평해짐
                                                    → v^4 = G·M·a0 (터리-피셔 관계)
"""

import numpy as np

G = 6.674e-11            # 중력상수
A0 = 1.2e-10            # 특징적 가속도 스케일 [m/s^2]
MSUN = 1.989e30         # 태양질량 [kg]
KPC = 3.086e19          # 킬로파섹 [m]
KMS = 1e3              # km/s


def enclosed_mass(r, M_disk, R_d):
    """지수원반 근사의 반지름 r 안쪽 질량 (구대칭 근사)."""
    x = r / R_d
    return M_disk * (1 - (1 + x) * np.exp(-x))


def newtonian_v(r, M_disk, R_d):
    """보이는 물질만으로 뉴턴 중력이 예측하는 회전속도."""
    g_bar = G * enclosed_mass(r, M_disk, R_d) / r ** 2
    return np.sqrt(g_bar * r), g_bar


def scale_modified_v(r, M_disk, R_d):
    """중력이 스케일(a0)에 따라 변한다고 볼 때의 회전속도 (RAR)."""
    _, g_bar = newtonian_v(r, M_disk, R_d)
    g_obs = g_bar / (1 - np.exp(-np.sqrt(g_bar / A0)))
    return np.sqrt(g_obs * r)


if __name__ == "__main__":
    M_disk = 6e10 * MSUN       # 은하수급 원반 질량
    R_d = 3 * KPC              # 원반 스케일 반지름
    r = np.linspace(0.3, 30, 300) * KPC

    v_newton = newtonian_v(r, M_disk, R_d)[0] / KMS
    v_scale = scale_modified_v(r, M_disk, R_d) / KMS

    print("=" * 74)
    print("brick_07 — 이론 대입: 은하 회전과 암흑물질")
    print("보이는 물질만: 뉴턴 예측 vs 스케일-수정 예측")
    print("=" * 74)
    print(f"\n{'반지름[kpc]':>10} | {'뉴턴 v[km/s]':>14} | {'스케일수정 v[km/s]':>18}")
    print("-" * 74)
    for rk in [2, 5, 10, 20, 30]:
        i = int(np.argmin(np.abs(r / KPC - rk)))
        print(f"{rk:>10} | {v_newton[i]:>14.0f} | {v_scale[i]:>18.0f}")
    print("-" * 74)

    v_flat_pred = (G * M_disk * A0) ** 0.25 / KMS
    print(f"\n[핵심] 뉴턴 예측은 바깥으로 갈수록 '떨어진다'(케플러 하강).")
    print(f"       스케일-수정 예측은 '평평하게' 유지된다 — 실제 은하 관측 그대로.")
    print(f"       평평구간 속도 이론값 v=(G·M·a0)^(1/4) ≈ {v_flat_pred:.0f} km/s (관측과 부합).")
    print(f"\n[흥미로운 점] 같은 '보이는 물질'인데, 뉴턴으로 읽으면 부족한 중력을 메우려")
    print(f"       '안 보이는 암흑물질'을 넣어야 한다. 그러나 '중력이 스케일 a0 아래에서")
    print(f"       변한다'고 대입하면 암흑물질 없이 곡선이 맞는다. → 암흑물질이 '입자'가")
    print(f"       아니라 '스케일에서 드러나는 현상'일 수 있다는 것.")

    # 터리-피셔: 질량이 다른 은하들의 평평속도 v ∝ M^(1/4)
    masses = np.logspace(9, 11.5, 40) * MSUN
    v_flats = (G * masses * A0) ** 0.25 / KMS
    slope = np.polyfit(np.log(masses), np.log(v_flats), 1)[0]
    print(f"\n[뾰족한 예측] 질량 다른 은하들에서 v_flat ∝ M^{slope:.3f}  (즉 M ∝ v^4).")
    print(f"       이 '바리온 터리-피셔 관계'는 실제로 아주 좁은 산포로 관측된다 —")
    print(f"       암흑물질 모형이 자연스럽게 내기 어려운, 스케일-수정의 뾰족한 성공.")

    print(f"\n[정직한 단서] (1) 이건 MOND / RAR(McGaugh+2016) 재현이다. (2) 은하에선")
    print(f"       놀랍게 잘 맞지만, 은하단·우주배경복사·총알성단에서는 여전히 암흑물질을")
    print(f"       요구한다 → 주류는 아직 '입자 암흑물질'. (3) 베를린데의 엔트로피/정보")
    print(f"       중력은 이 MOND식 법칙을 정보 원리에서 유도한다(논쟁 중). 즉 확정된 승리가")
    print(f"       아니라, 이 이론이 실제로 '일하는' 살아있는 논쟁 지점이다.\n")

    # ---------- 그림 ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

        # (좌) 회전곡선
        ax1.plot(r / KPC, v_newton, lw=2, color="tab:orange",
                 label="Newton, visible matter only (falls)")
        ax1.plot(r / KPC, v_scale, lw=2, color="tab:blue",
                 label="scale-modified gravity (stays flat)")
        ax1.fill_between(r / KPC, v_newton, v_scale, color="tab:blue", alpha=0.08)
        ax1.text(20, (v_newton[-1] + v_scale[-1]) / 2,
                 "gap that standard\nphysics fills with\n'dark matter'",
                 fontsize=8, color="gray", ha="center", va="center")
        ax1.set_xlabel("radius  [kpc]")
        ax1.set_ylabel("rotation speed  [km/s]")
        ax1.set_title("Galaxy rotation: same visible matter, two readings")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)
        ax1.set_ylim(0, max(v_scale) * 1.3)

        # (우) 바리온 터리-피셔
        ax2.loglog(masses / MSUN, v_flats, lw=2, color="tab:green")
        ax2.set_xlabel("visible (baryonic) mass  [M_sun]")
        ax2.set_ylabel("flat rotation speed  [km/s]")
        ax2.set_title(f"Baryonic Tully-Fisher: v ∝ M^{slope:.2f}  (M ∝ v⁴)\n"
                      "a sharp prediction, observed with tight scatter")
        ax2.grid(alpha=0.3, which="both")

        fig.suptitle("brick_07 — apply 'gravity changes at a characteristic scale a0' "
                     "→ flat rotation curves without dark matter", fontsize=12)
        fig.tight_layout()
        out = __file__.rsplit("/", 1)[0] + "/brick_07_rotation_curves.png"
        fig.savefig(out, dpi=130)
        print(f"[그림] 저장됨: {out}")
    except Exception as e:
        print(f"[그림] 건너뜀: {e}")
