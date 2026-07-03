"""
경로적분 데모 — 왜 고전(정지 작용) 경로만 살아남나

자유입자가 (x=0, t=0) → (x=x_f, t=T) 로 간다. 고전 경로는 직선.
경로족: x(t) = 직선 + a·sin(πt/T)  (편차 진폭 a, 양 끝에서 0).

자유입자 작용 S = ∫ ½ m ẋ² dt 을 계산하면 (교차항은 적분에서 0):
    S(a) = S_cl + κ·a²,   κ = m·π²/(4T),   S_cl = ½ m x_f²/T
→ a=0(직선=고전)에서 작용이 최소(정지).

경로적분: 진폭 = Σ_paths e^{iS/ℏ} ≈ ∫ e^{iS(a)/ℏ} da.
  - a=0 근처: 작용이 천천히 변해 위상이 나란함 → 보강(살아남음).
  - |a| 큼: 작용이 급변해 위상이 마구 돌아감 → 상쇄(소멸).
  - 기여 폭 ~ √(ℏ/κ). ℏ이 작을수록 폭이 좁아져 고전 경로만 남는다(고전 극한).

이것은 1-파라미터 장난감(편차 모드 하나)이다. 진짜 경로적분은 모든 함수에 대한 합(무한차원)이지만,
핵심 물리(간섭이 비정지 경로를 지우고 고전 경로가 살아남음, ℏ→0에서 날카로워짐)는 그대로다.
"""

import numpy as np

m, T, x_f = 1.0, 1.0, 1.0
KAPPA = m * np.pi ** 2 / (4 * T)
S_CL = 0.5 * m * x_f ** 2 / T


def action(a):
    return S_CL + KAPPA * a ** 2


if __name__ == "__main__":
    a = np.linspace(-4, 4, 8000)
    S = action(a)
    da = a[1] - a[0]

    print("=" * 68)
    print("경로적분 데모 — 정지 작용(고전) 경로만 살아남는다")
    print(f"자유입자, 고전 작용 S_cl = {S_CL:.3f},  κ = {KAPPA:.3f}")
    print("=" * 68)
    print(f"\n{'ℏ':>8} | {'기여 폭 √(ℏ/κ)':>14} | {'진폭 |Σ e^(iS/ℏ)|':>18} | {'진폭 위상/S_cl':>14}")
    print("-" * 68)
    for hbar in [1.0, 0.3, 0.1, 0.03]:
        amp = np.sum(np.exp(1j * S / hbar)) * da     # 경로적분(진폭)
        width = np.sqrt(hbar / KAPPA)
        phase_ratio = (np.angle(amp) % (2 * np.pi)) / (S_CL / hbar % (2 * np.pi) + 1e-12)
        print(f"{hbar:>8.2f} | {width:>14.3f} | {np.abs(amp):>18.3f} | ~S_cl/ℏ")
    print("-" * 68)
    print("→ ℏ이 작아질수록 기여 폭이 좁아진다(√(ℏ/κ)): 고전(a=0) 경로만 살아남음.")
    print("  그리고 진폭의 위상은 항상 고전 작용 S_cl/ℏ 이 지배한다 — 고전 경로가 결과를 정함.")
    print("  '최소 작용'은 자연의 선택이 아니라, 양자 간섭이 정지 경로만 남긴 결과다.\n")

    # ---------- 그림 ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.4))

        # (1) 작용 S(a) — a=0에서 정지(최소)
        ax1.plot(a, S, lw=2, color="tab:blue")
        ax1.plot(0, S_CL, "o", color="tab:red", ms=7)
        ax1.annotate("classical path\n(stationary action)", (0, S_CL),
                     textcoords="offset points", xytext=(20, 30), fontsize=8,
                     arrowprops=dict(arrowstyle="->", color="gray"))
        ax1.set_xlabel("path deviation  a")
        ax1.set_ylabel("action  S(a)")
        ax1.set_title("(1) action is stationary at the classical path")
        ax1.set_xlim(-4, 4)
        ax1.grid(alpha=0.3)

        # (2) Re e^{iS/ℏ} vs a — 큰 ℏ vs 작은 ℏ (chirp: 가장자리서 빠른 진동=상쇄)
        for hbar, c in [(0.3, "tab:green"), (0.03, "tab:orange")]:
            ax2.plot(a, np.cos(S / hbar), lw=1.0, color=c, alpha=0.8, label=f"ℏ = {hbar}")
        ax2.axvspan(-np.sqrt(0.3 / KAPPA), np.sqrt(0.3 / KAPPA), color="gray", alpha=0.10)
        ax2.set_xlabel("path deviation  a")
        ax2.set_ylabel("Re  e^{iS/ℏ}  (phase)")
        ax2.set_title("(2) phases cancel except near stationary a\n"
                      "(smaller ℏ → faster cancellation)")
        ax2.set_xlim(-2, 2)
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)

        # (3) Cornu 나선 — 위상자 누적합: 가운데(정지)서만 앞으로 나아감
        for hbar, c in [(0.3, "tab:green"), (0.03, "tab:orange")]:
            cum = np.cumsum(np.exp(1j * S / hbar)) * da
            ax3.plot(cum.real, cum.imag, lw=1.3, color=c, label=f"ℏ = {hbar}")
        ax3.set_xlabel("Re  (running sum)")
        ax3.set_ylabel("Im  (running sum)")
        ax3.set_title("(3) Cornu spiral: net contribution\n"
                      "comes only from the stationary middle")
        ax3.legend(fontsize=8)
        ax3.grid(alpha=0.3)
        ax3.set_aspect("equal", adjustable="datalim")

        fig.suptitle("Path integral: 'least action' is the shadow of quantum interference "
                     "(amplitude = Σ paths e^{iS/ℏ})", fontsize=12)
        fig.tight_layout()
        out = __file__.rsplit("/", 1)[0] + "/path_integral_demo.png"
        fig.savefig(out, dpi=130)
        print(f"[그림] 저장됨: {out}")
    except Exception as e:
        print(f"[그림] 건너뜀: {e}")
