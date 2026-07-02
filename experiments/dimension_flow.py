"""
brick_03 — 유효 차원은 스케일에 따라 흐르는가? (실패할 수 있는 첫 시험)

앞선 벽돌들은 규칙적 격자에서 "당연히 나올" 결과를 확인했다. 격자를 규칙적으로 깔면
차원·속도·국소성이 전부 입력값이라, 스케일 의존을 물어도 통과가 보장돼 있었다.

여기서는 판을 바꾼다. 규칙적 격자 대신, 연결 구조가 다른 여러 바탕(그래프)을 깔고
'유효 차원'이 관측 스케일에 따라 변하는지(흐르는지) 잰다. 이번 결과는 미리 정해져
있지 않다 — 안 흐를 수도 있다. 그래서 처음으로 진짜 시험이다.

측정 도구: 스펙트럴 차원.
  그래프 위 확산(연속시간 random walk)의 되돌아올 확률(heat-kernel return prob)
    P(t) = (1/N) Σ_k exp(-λ_k t),   λ_k = 정규화 라플라시안 고유값
  d차원에서 P(t) ~ t^(-d/2)  이므로  d_s(t) = -2 · d(log P)/d(log t).
  작은 t = 작은 스케일(국소),  큰 t = 큰 스케일(전역).
  → d_s(t) 가 곧 "관측 스케일에 따른 유효 차원".

무엇이 입력이고 무엇이 출력인가 (정직 표기):
  입력 = 그래프의 연결 규칙(고리/격자/무작위 지름길).
  출력 = d_s(t) 곡선. 이 곡선의 모양(흐르는지 아닌지)은 넣지 않았다.
"""

import numpy as np

RNG = np.random.default_rng(42)


# ---------- 바탕(그래프) 만들기 : 입력 ----------

def ring_adjacency(n: int, k: int = 2) -> np.ndarray:
    """1차원 고리: 각 점이 양옆 k개와 연결. 규칙적 → 대조군(차원 ≈ 1)."""
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(1, k + 1):
            A[i, (i + j) % n] = 1
            A[i, (i - j) % n] = 1
    return A


def torus_adjacency(L: int) -> np.ndarray:
    """2차원 토러스 격자(L×L, 주기경계). 규칙적 → 대조군(차원 ≈ 2)."""
    n = L * L
    A = np.zeros((n, n))
    idx = lambda r, c: (r % L) * L + (c % L)
    for r in range(L):
        for c in range(L):
            v = idx(r, c)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                A[v, idx(r + dr, c + dc)] = 1
    return A


def small_world_adjacency(n: int, k: int = 2, p: float = 0.06) -> np.ndarray:
    """고리에 무작위 지름길을 섞음(Watts-Strogatz). 작은 스케일=고리(1D처럼),
    큰 스케일=지름길이 지배 → 유효 차원이 흐를 수 있다(미정)."""
    A = ring_adjacency(n, k)
    for i in range(n):
        for j in range(1, k + 1):
            if RNG.random() < p:
                nb = (i + j) % n
                A[i, nb] = A[nb, i] = 0
                t = int(RNG.integers(n))
                while t == i or A[i, t]:
                    t = int(RNG.integers(n))
                A[i, t] = A[t, i] = 1
    return A


# ---------- 측정 : 출력 ----------

def spectral_dimension(A: np.ndarray, ts: np.ndarray):
    """정규화 라플라시안 고유값에서 되돌아올 확률 P(t)와 d_s(t)를 계산."""
    deg = A.sum(1)
    dinv = 1.0 / np.sqrt(deg)
    L = np.eye(A.shape[0]) - (dinv[:, None] * A * dinv[None, :])
    evals = np.linalg.eigvalsh(L)                       # [0, 2]
    P = np.exp(-ts[:, None] * evals[None, :]).mean(1)   # heat-kernel return prob
    logt, logP = np.log(ts), np.log(P)
    d_s = -2.0 * np.gradient(logP, logt)
    return P, d_s


if __name__ == "__main__":
    N = 2025                       # 고리/지름길
    L = 45                         # 토러스 45×45 = 2025 (같은 크기)
    ts = np.logspace(-0.3, 2.3, 70)   # 확산 시간 = 관측 스케일 (작은→큰)

    graphs = {
        "ring (1D, regular)":       ring_adjacency(N, k=2),
        "torus (2D, regular)":      torus_adjacency(L),
        "small-world (ring+links)": small_world_adjacency(N, k=2, p=0.06),
    }

    print("=" * 74)
    print("brick_03 — 유효 차원은 스케일에 따라 흐르는가?")
    print("d_s(t): 작은 t=작은 스케일, 큰 t=큰 스케일  (실패 가능한 시험)")
    print("=" * 74)
    # 유한크기 포화 전 구간에서 세 스케일의 d_s 보고
    probe = [8, 30, 100]           # 관측 스케일(확산시간) 세 지점
    pj = [int(np.argmin(np.abs(ts - p))) for p in probe]
    print(f"\n{'바탕(그래프)':<26} | {'작은스케일 t≈8':>14} | {'중간 t≈30':>12} | {'큰스케일 t≈100':>14}")
    print("-" * 74)
    results = {}
    for name, A in graphs.items():
        P, d_s = spectral_dimension(A, ts)
        results[name] = (P, d_s)
        print(f"{name:<26} | {d_s[pj[0]]:>14.2f} | {d_s[pj[1]]:>12.2f} | {d_s[pj[2]]:>14.2f}")
    print("-" * 74)

    print("\n[대조군] ring / torus : d_s 가 스케일에 무관하게 거의 1 / 2 로 평평.")
    print("         → 규칙적 격자에선 차원이 입력값. 안 흐름 (예상대로, 리깅된 결과).")
    print("\n[시험]   small-world : d_s 가 스케일에 따라 변하는가?")
    sw = results["small-world (ring+links)"][1]
    flow = sw[pj[2]] - sw[pj[0]]
    print(f"         작은 스케일 d_s≈{sw[pj[0]]:.2f}  →  큰 스케일 d_s≈{sw[pj[2]]:.2f}  "
          f"(변화 {flow:+.2f})")
    if abs(flow) > 0.3:
        print("         → 흐른다. 유효 차원은 고정된 수가 아니라 관측 스케일의 함수.")
        print("           그리고 이 흐름은 넣지 않았다 — 지름길만 넣었을 뿐 곡선 모양은 출력.")
    else:
        print("         → 거의 안 흐른다. 강한 판본이 여기서는 성립하지 않음(그것도 결과).")
    print("\n[정직한 단서] 흐름의 '방향'(스케일↑에서 차원↑)은 실제 양자중력(작은 스케일에서")
    print("         차원↓)과 반대다. 바탕이 다르기 때문. 여기서 보인 건 '차원이 스케일에")
    print("         의존할 수 있다'는 정성적 사실이지, 우리 우주가 그렇다는 게 아니다.")
    print("         큰 t 끝에서 d_s가 0으로 떨어지는 건 유한크기(N) 포화 아티팩트.\n")

    # ---------- 그림 ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8.5, 5))
        for name, (P, d_s) in results.items():
            ax.semilogx(ts, d_s, label=name, lw=2)
        ax.axhline(1, ls=":", color="gray", lw=0.8)
        ax.axhline(2, ls=":", color="gray", lw=0.8)
        ax.set_xlabel("diffusion time t   =   observation scale  (small → large)")
        ax.set_ylabel("spectral dimension  d_s(t)")
        ax.set_title("brick_03 — does effective dimension flow with scale?\n"
                     "regular grids stay flat (rigged); small-world flows (not hardcoded)")
        ax.set_ylim(0, 4)
        ax.legend()
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        out = __file__.rsplit("/", 1)[0] + "/brick_03_dimension_flow.png"
        fig.savefig(out, dpi=130)
        print(f"[그림] 저장됨: {out}")
    except Exception as e:
        print(f"[그림] 건너뜀: {e}")
