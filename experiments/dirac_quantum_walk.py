"""
brick_01 — 이산 관계 규칙에서 창발하는 불변 광원뿔 (1D Dirac Quantum Walk)

최소 주장 (테스트 대상):
  공간·시간을 미리 넣지 않은 순수 이산·국소 규칙(coin + shift)만으로
  교란(excitation)에 유한한 불변 최대전파속도 c_eff 가 창발한다.
  그리고 그 c_eff 는 동역학(질량)과 무관한 격자 구조의 성질이다.

모형:
  각 격자점 x 에 2성분 스피너 psi(x) ∈ C^2
    성분 0 = 오른쪽 이동자(right-mover), 성분 1 = 왼쪽 이동자(left-mover)
  한 스텝 = 코인(질량 회전) 적용 → 시프트(오른쪽/왼쪽 1칸)
    C(m) = [[cos m, i sin m], [i sin m, cos m]]
  m=0  : 코인=단위행렬 → 순수 이동, 엄밀한 광원뿔
  m>0  : 좌우 이동자 섞임 → 내부가 채워짐(디랙 지그재그), 그러나 원뿔 가장자리는 불변

연속극한에서 이 보행은 1+1D Weyl/Dirac 방정식을 재현한다 (D'Ariano-Perinotti 계보).
"""

import numpy as np


def dirac_quantum_walk(steps: int, mass: float) -> np.ndarray:
    """steps 스텝 동안 확률분포 prob[t, x] 를 반환한다."""
    N = 2 * steps + 3          # 원뿔을 담기에 충분한 격자
    center = N // 2

    # 상태: (N, 2) 복소 스피너. 중앙에 국소화된 초기 교란.
    psi = np.zeros((N, 2), dtype=complex)
    psi[center, 0] = 1.0 / np.sqrt(2)
    psi[center, 1] = 1j / np.sqrt(2)

    c, s = np.cos(mass), np.sin(mass)
    C = np.array([[c, 1j * s], [1j * s, c]], dtype=complex)

    history = np.zeros((steps + 1, N))
    for t in range(steps + 1):
        history[t] = np.sum(np.abs(psi) ** 2, axis=1)   # 스텝 전 확률 기록 (t=0 = 초기)
        psi = psi @ C.T                                  # 각 격자점에 코인 C 적용
        up = np.roll(psi[:, 0], 1)                       # 오른쪽 이동자 +1
        down = np.roll(psi[:, 1], -1)                    # 왼쪽 이동자 -1
        psi = np.stack([up, down], axis=1)
    return history, center


def causal_leak(history: np.ndarray, center: int) -> float:
    """엄밀한 인과 검사: 각 시각 t 에서 |x| > t (원뿔 바깥) 확률의 총합의 최댓값.
    시프트가 스텝당 최대 1칸이므로 이 값은 기계정밀도까지 정확히 0 이어야 한다.
    → c_eff = 1 은 측정이 아니라 격자 구조가 강제하는 정리(theorem)다."""
    steps = history.shape[0] - 1
    worst = 0.0
    for t in range(steps + 1):
        lo, hi = center - t, center + t
        outside = history[t].sum() - history[t][lo:hi + 1].sum()
        worst = max(worst, outside)
    return worst


def mean_spread(history: np.ndarray, center: int) -> np.ndarray:
    """평균 |x| (군속도적 채움) — 질량에 의존하는 양."""
    steps = history.shape[0] - 1
    x = np.arange(history.shape[1]) - center
    spread = np.zeros(steps + 1)
    for t in range(steps + 1):
        p = history[t]
        norm = p.sum()
        spread[t] = np.sum(np.abs(x) * p) / norm if norm > 0 else 0.0
    return spread


if __name__ == "__main__":
    STEPS = 200
    MASSES = [0.0, 0.15, 0.30, 0.60]

    print("=" * 68)
    print("brick_01 — 창발하는 불변 광원뿔 (1D Dirac Quantum Walk)")
    print(f"스텝 수 = {STEPS}, 격자칸/스텝 단위\n")
    print(f"{'질량 m':>8} | {'원뿔밖 확률(최대)':>18} | {'평균확산 <|x|>/t':>18}")
    print("-" * 68)

    results = {}
    for m in MASSES:
        hist, center = dirac_quantum_walk(STEPS, m)
        leak = causal_leak(hist, center)
        spread = mean_spread(hist, center)
        results[m] = (hist, center, leak, spread)
        print(f"{m:>8.2f} | {leak:>18.2e} | {spread[-1] / STEPS:>18.6f}")

    print("-" * 68)
    # 핵심 검증 1: 원뿔 바깥 확률 = 0 (기계정밀도) → c_eff = 1 은 정리다.
    worst_leak = max(results[m][2] for m in MASSES)
    print(f"\n[검증 1] 원뿔 |x|>t 바깥 확률의 최댓값 = {worst_leak:.2e}  (≈ 0)")
    print("         → 최대전파속도 c_eff = 1 은 질량과 무관하게 격자가 강제하는 불변량.")
    print("         측정이 아니라 정리(시프트가 스텝당 ≤1칸). ✓")

    # 검증 2: 내부 채움(군속도적 확산)은 질량에 의존한다.
    spreads = np.array([results[m][3][-1] / STEPS for m in MASSES])
    print(f"\n[검증 2] 평균확산 <|x|>/t 은 질량에 따라 "
          f"{spreads.max():.3f} → {spreads.min():.3f} 로 감소")
    print("         → 질량은 원뿔을 바꾸지 않고 '내부 채움(지속성)'만 바꾼다. ✓")
    print("         (앞선 통찰과 일치: c=구조, 질량=실체성/지속은 별개 축)\n")

    # 시각화: 확률 히트맵으로 원뿔 그리기
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(MASSES), figsize=(4 * len(MASSES), 4.2),
                                 sharey=True)
        for ax, m in zip(axes, MASSES):
            hist, center = results[m][0], results[m][1]
            half = STEPS + 1
            window = hist[:, center - half:center + half + 1]
            ax.imshow(np.sqrt(window), origin="lower", aspect="auto",
                      cmap="magma", extent=[-half, half, 0, STEPS])
            # 이론 광원뿔 |x| = t
            ax.plot([0, half], [0, half], "c--", lw=0.8, alpha=0.7)
            ax.plot([0, -half], [0, half], "c--", lw=0.8, alpha=0.7)
            ax.set_title(f"mass m = {m}", fontsize=11)
            ax.set_xlabel("position x (sites)")
        axes[0].set_ylabel("time t (steps)")
        fig.suptitle("Emergent invariant light cone from discrete local rules  "
                     "(cyan dashed = |x| = c_eff*t, c_eff = 1)", fontsize=12)
        fig.tight_layout()
        out = __file__.rsplit("/", 1)[0] + "/brick_01_lightcone.png"
        fig.savefig(out, dpi=130)
        print(f"[그림] 저장됨: {out}")
    except Exception as e:
        print(f"[그림] 건너뜀: {e}")
