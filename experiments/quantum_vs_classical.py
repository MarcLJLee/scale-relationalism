"""
brick_04 — 양자적 움직임은 무엇이며, 언제 평범한 움직임으로 바뀌는가?

일상적인 말로:
  - 평범한(고전적) 걷기 = 술 취한 사람의 갈지자 걸음. 매 걸음 왼/오른쪽을 무작위로.
    → 천천히 퍼지고, 출발점 근처에 종 모양으로 모인다.
  - 양자적 걷기 = 걷는 이가 양쪽을 '동시에' 밟고, 그 가능성들이 서로 간섭한다.
    → 훨씬 빨리 바깥으로 몰려가, 가장자리에 두 개의 뿔로 쌓인다.
  - 결어긋남(decoherence) = 누군가 자꾸 "지금 어디 있어?" 하고 들여다보는 것.
    들여다볼 때마다 '동시에'가 깨져서, 자주 볼수록(=거칠게/크게 볼수록) 평범한 걸음이 된다.

측정하는 숫자: 시간이 갈수록 '퍼짐(표준편차 σ)'이 얼마나 빨리 자라는가(지수 α).
  σ(t) ~ t^α  에서
    α ≈ 1.0  → 양자(탄도적, 시간에 비례해 빠르게 퍼짐)
    α ≈ 0.5  → 고전(확산적, 시간의 제곱근으로 느리게 퍼짐)
  들여다보는 정도 γ 를 0 → 1 로 키우며 α 가 1 에서 0.5 로 미끄러지는지 본다.

이것은 '증명'이 아니라 '시연'이다. 코드는 양자적 운동이 고전적 운동과 어떻게 다른지,
그리고 스케일(들여다봄)에 따라 어떻게 넘어가는지를 보여줄 뿐,
우리 우주가 양자라는 것을 증명하지는 못한다(그건 실물 실험의 몫).
"""

import numpy as np


def build_operators(M: int):
    """위치 -M..M 격자 위 동전 붙은 걷기의 한 걸음 연산자 U 와 위치 인덱스."""
    N = 2 * M + 1
    dim = 2 * N
    pos_of = np.repeat(np.arange(N), 2)          # 각 상태의 위치 인덱스 (길이 dim)

    # 동전: 각 자리에서 아다마르(Hadamard) — '동시에 양쪽'을 만드는 부분
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    C = np.zeros((dim, dim), dtype=complex)
    for p in range(N):
        C[2 * p:2 * p + 2, 2 * p:2 * p + 2] = H

    # 이동: 동전 0 이면 오른쪽(+1), 동전 1 이면 왼쪽(-1)
    S = np.zeros((dim, dim), dtype=complex)
    for p in range(N):
        if p + 1 < N:
            S[2 * (p + 1) + 0, 2 * p + 0] = 1     # 오른쪽 이동자
        if p - 1 >= 0:
            S[2 * (p - 1) + 1, 2 * p + 1] = 1     # 왼쪽 이동자
    U = S @ C
    return U, pos_of, N


def run_walk(steps: int, gamma: float):
    """gamma = 매 걸음 '들여다보는 정도'(0=순수 양자, 1=완전 고전). σ(t) 를 반환."""
    M = steps + 3
    U, pos_of, N = build_operators(M)
    dim = 2 * N
    center = M

    # 초기 상태: 가운데, 동전 (|0>+i|1>)/√2 (좌우 대칭 퍼짐)
    psi0 = np.zeros(dim, dtype=complex)
    psi0[2 * center + 0] = 1 / np.sqrt(2)
    psi0[2 * center + 1] = 1j / np.sqrt(2)
    rho = np.outer(psi0, psi0.conj())            # 밀도행렬 (섞임까지 다루려면 필요)

    # 위치 결맞음을 깨는 마스크: 같은 위치는 1, 다른 위치는 (1-gamma)
    same_pos = (pos_of[:, None] == pos_of[None, :])
    dephase = np.where(same_pos, 1.0, 1.0 - gamma)

    x = np.arange(N) - center
    sigma = np.empty(steps + 1)
    for t in range(steps + 1):
        diag = np.real(np.diag(rho))
        Px = diag[0::2] + diag[1::2]             # 위치별 확률 (동전 두 성분 합)
        Px = Px / Px.sum()
        mean = np.sum(x * Px)
        sigma[t] = np.sqrt(np.sum((x - mean) ** 2 * Px))
        rho = U @ rho @ U.conj().T               # 한 걸음
        rho = rho * dephase                      # 들여다보기(결어긋남)
    return sigma


def spreading_exponent(sigma: np.ndarray, t_lo: int = 10) -> float:
    """σ(t) ~ t^α 의 지수 α 를 로그-로그 기울기로 추정."""
    ts = np.arange(len(sigma))
    m = ts >= t_lo
    return np.polyfit(np.log(ts[m]), np.log(sigma[m]), 1)[0]


if __name__ == "__main__":
    STEPS = 60
    GAMMAS = [0.0, 0.03, 0.15, 1.0]

    print("=" * 70)
    print("brick_04 — 양자적 움직임과, 그것이 평범해지는 지점")
    print("들여다보는 정도 γ 를 키우며 '퍼짐 지수 α' 를 측정")
    print("  α≈1.0 = 양자(빠르게 퍼짐) / α≈0.5 = 고전(느리게 퍼짐)")
    print("=" * 70)
    print(f"\n{'들여다봄 γ':>10} | {'퍼짐 지수 α':>14} | 해석")
    print("-" * 70)

    curves = {}
    for g in GAMMAS:
        sig = run_walk(STEPS, g)
        a = spreading_exponent(sig)
        curves[g] = sig
        if a > 0.85:
            tag = "양자적 (탄도적)"
        elif a < 0.6:
            tag = "고전적 (확산적)"
        else:
            tag = "중간 — 넘어가는 중"
        print(f"{g:>10.2f} | {a:>14.3f} | {tag}")
    print("-" * 70)

    print("\n[검증] γ=0 일 때 α≈1 (양자), γ=1 일 때 α≈0.5 (고전).")
    print("       그 사이에서 α 가 미끄러진다 → '양자냐 고전이냐'는 켜고 끄는 스위치가")
    print("       아니라, 얼마나 들여다보는가(=스케일)에 따라 이어진 눈금이다.")
    print("\n[정직한 단서] 이것은 시연이지 증명이 아니다. 양자 운동이 고전 운동과 어떻게")
    print("       다른지, 스케일에 따라 어떻게 넘어가는지를 보일 뿐, 우리 우주가 양자라는")
    print("       것은 실물 실험(이중슬릿·벨 부등식)만 증명할 수 있다.\n")

    # ---------- 그림 ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

        # (좌) 마지막 시점 위치 분포: 양자(두 뿔) vs 고전(종 모양)
        for g, style in [(0.0, "-"), (1.0, "--")]:
            M = STEPS + 3
            U, pos_of, N = build_operators(M)
            center = M
            psi0 = np.zeros(2 * N, dtype=complex)
            psi0[2 * center] = 1 / np.sqrt(2)
            psi0[2 * center + 1] = 1j / np.sqrt(2)
            rho = np.outer(psi0, psi0.conj())
            same = (pos_of[:, None] == pos_of[None, :])
            dep = np.where(same, 1.0, 1.0 - g)
            for _ in range(STEPS):
                rho = U @ rho @ U.conj().T
                rho = rho * dep
            diag = np.real(np.diag(rho))
            Px = diag[0::2] + diag[1::2]
            x = np.arange(N) - center
            label = "quantum (γ=0): two horns" if g == 0 else "classical (γ=1): bell"
            ax1.plot(x, Px, style, lw=2, label=label)
        ax1.set_title("Where the walker ends up (after 60 steps)")
        ax1.set_xlabel("position")
        ax1.set_ylabel("probability")
        ax1.set_xlim(-STEPS, STEPS)
        ax1.legend()
        ax1.grid(alpha=0.3)

        # (우) 퍼짐 σ(t) vs 시간, 로그-로그 — 기울기 = 지수 α
        ts = np.arange(STEPS + 1)
        for g in GAMMAS:
            a = spreading_exponent(curves[g])
            ax2.loglog(ts[1:], curves[g][1:], lw=2, label=f"γ={g}  (α≈{a:.2f})")
        ax2.set_title("How fast the spread grows  (slope = exponent α)")
        ax2.set_xlabel("time t  (steps)")
        ax2.set_ylabel("spread σ")
        ax2.legend()
        ax2.grid(alpha=0.3, which="both")

        fig.suptitle("brick_04 — quantum motion (α≈1) slides to classical (α≈0.5) "
                     "as you look more (decoherence)", fontsize=12)
        fig.tight_layout()
        out = __file__.rsplit("/", 1)[0] + "/brick_04_quantum_classical.png"
        fig.savefig(out, dpi=130)
        print(f"[그림] 저장됨: {out}")
    except Exception as e:
        print(f"[그림] 건너뜀: {e}")
