"""
brick_09 — 미시세계에서 스케일에 따라 힘이 변한다: 세 힘의 (거의) 통일

이 프로젝트의 명제 "스케일에 따라 물리 규칙이 변한다"가 미시세계에서 가장 문자
그대로 실현되는 곳: 힘의 세기(결합상수)가 관측 에너지(=스케일)에 따라 변한다(running).

세 힘 — 전자기(U(1)), 약(SU(2)), 강(SU(3)) — 의 세기를 낮은 에너지(M_Z)에서
높은 에너지로 재규격화군 방정식으로 굴린다:
    α_i^{-1}(μ) = α_i^{-1}(M_Z) - (b_i / 2π) · ln(μ / M_Z)
힘의 세기는 스케일에 따라 흘러가고(running), 높은 에너지로 가면 세 값이 한 점 근처로
모인다 → "세 힘이 원래 하나였다"는 대통일(GUT)의 단서.

정직한 반전(brick_08 처럼):
  표준모형(SM) 계수로는 세 선이 '거의' 모이지만 정확히 한 점에서 만나지는 않는다
  (작은 삼각형을 이룬다). 초대칭(MSSM)을 넣으면 한 점(~2e16 GeV)에서 만난다 —
  그러나 초대칭 입자는 아직 관측되지 않았다. 성공(거의 통일)과 미결(정확히는 새 물리 필요).
"""

import numpy as np

M_Z = 91.19            # Z 보손 질량 [GeV] — 낮은 에너지 기준점
TWO_PI = 2 * np.pi

# M_Z 에서 측정된 역결합상수 (GUT 정규화, 표준 교과서 값)
AINV_MZ = {"U(1)": 59.0, "SU(2)": 29.6, "SU(3)": 8.5}

# 1-루프 베타 계수 (b_i)
B_SM = {"U(1)": 41 / 10, "SU(2)": -19 / 6, "SU(3)": -7}        # 표준모형
B_MSSM = {"U(1)": 33 / 5, "SU(2)": 1.0,    "SU(3)": -3}         # 초대칭


def run(ainv_mz, b, mu):
    """역결합상수 α^{-1}(μ)."""
    t = np.log(mu / M_Z)
    return ainv_mz - (b / TWO_PI) * t


def pairwise_crossings(b):
    """세 힘의 쌍별 교차 에너지 [GeV] (선형이라 해석적으로)."""
    keys = list(AINV_MZ)
    out = {}
    for i in range(3):
        for j in range(i + 1, 3):
            a, b1 = keys[i], keys[j]
            # AINV_MZ[a] - (b[a]/2π)t = AINV_MZ[b1] - (b[b1]/2π)t
            slope = (b[a] - b[b1]) / TWO_PI
            t = (AINV_MZ[a] - AINV_MZ[b1]) / slope
            out[f"{a}={b1}"] = M_Z * np.exp(t)
    return out


if __name__ == "__main__":
    print("=" * 72)
    print("brick_09 — 미시세계: 스케일에 따라 힘이 변하고, 높은 에너지에서 (거의) 통일")
    print("=" * 72)

    print("\n힘의 세기(역결합상수)가 스케일에 따라 흐른다:")
    print(f"{'에너지 μ [GeV]':>16} | {'U(1) 전자기':>10} | {'SU(2) 약':>9} | {'SU(3) 강':>9}")
    print("-" * 60)
    for mu in [M_Z, 1e6, 1e10, 1e14, 1e16]:
        a1 = run(AINV_MZ["U(1)"], B_SM["U(1)"], mu)
        a2 = run(AINV_MZ["SU(2)"], B_SM["SU(2)"], mu)
        a3 = run(AINV_MZ["SU(3)"], B_SM["SU(3)"], mu)
        print(f"{mu:>16.2e} | {a1:>10.1f} | {a2:>9.1f} | {a3:>9.1f}")
    print("-" * 60)
    print("낮은 에너지에선 8.5 vs 29.6 vs 59.0 로 제각각인데, 높은 에너지로 갈수록 모인다.")

    print("\n[표준모형] 세 힘의 쌍별 교차 에너지:")
    for pair, mu in pairwise_crossings(B_SM).items():
        print(f"        {pair:>14} 에서 만남:  {mu:>10.2e} GeV")
    print("        → 10^13 ~ 10^17 GeV 사이에서 '거의' 모이지만 한 점은 아니다(삼각형).")

    print("\n[초대칭(MSSM)] 세 힘의 쌍별 교차 에너지:")
    for pair, mu in pairwise_crossings(B_MSSM).items():
        print(f"        {pair:>14} 에서 만남:  {mu:>10.2e} GeV")
    print("        → 셋이 ~2e16 GeV 한 점에서 만난다. 단, 초대칭 입자는 아직 미관측.")

    print("\n[의미] 미시세계에서 힘의 세기는 스케일의 함수다(측정된 사실). 그것을 높은")
    print("        에너지로 외삽하면 세 힘이 거의 하나가 된다 — 스케일-의존이 낳는 흥미로운")
    print("        결과. 그러나 '정확한' 통일은 표준모형만으로는 안 되고 새 물리를 요구한다.")
    print("        (brick_07 은하 성공 / brick_08 은하단 실패 처럼, 성공과 미결의 짝.)\n")

    # ---------- 그림 ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mu = np.logspace(np.log10(M_Z), 18, 400)
        x = np.log10(mu)
        colors = {"U(1)": "tab:blue", "SU(2)": "tab:green", "SU(3)": "tab:red"}
        labels = {"U(1)": "U(1) electromagnetic", "SU(2)": "SU(2) weak",
                  "SU(3)": "SU(3) strong"}

        fig, ax = plt.subplots(figsize=(9, 5.6))
        for k in AINV_MZ:
            ax.plot(x, run(AINV_MZ[k], B_SM[k], mu), lw=2, color=colors[k],
                    label=f"{labels[k]} (Standard Model)")
            ax.plot(x, run(AINV_MZ[k], B_MSSM[k], mu), lw=1.2, ls="--",
                    color=colors[k], alpha=0.7)
        ax.plot([], [], "k--", alpha=0.7, label="with supersymmetry (MSSM)")

        ax.set_xlabel("energy scale  log₁₀(μ / GeV)   →  finer resolution")
        ax.set_ylabel("inverse force strength  α⁻¹")
        ax.set_title("brick_09 — force strengths run with scale, and (nearly) unify\n"
                     "solid = Standard Model (miss), dashed = SUSY (meet at ~2×10¹⁶ GeV)")
        ax.legend(fontsize=8, loc="center left")
        ax.grid(alpha=0.3)
        ax.annotate("they almost meet here", xy=(13.5, 44), fontsize=8, color="gray")
        fig.tight_layout()
        out = __file__.rsplit("/", 1)[0] + "/brick_09_coupling_unification.png"
        fig.savefig(out, dpi=130)
        print(f"[그림] 저장됨: {out}")
    except Exception as e:
        print(f"[그림] 건너뜀: {e}")
