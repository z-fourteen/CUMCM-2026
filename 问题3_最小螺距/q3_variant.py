# -*- coding: utf-8 -*-
"""验证猜想：全量碰撞检验能否简化为「仅检验龙头与其外层一圈」？

对同一套几何、同一套 SAT、同一套二分流程，只替换碰撞判据的板凳对集合：
  A. 全量判据    ：所有 |i-j|>=2 的板凳对（基准，已得 p* = 0.450337 m）
  B. 龙头-外层一圈：i = 1（龙头），j 取角度上位于龙头外一圈的板凳
  C. 龙头-全部    ：i = 1，j >= 3（参照，用于分离“只看龙头”与“只看外一圈”的影响）
"""
import numpy as np
from q3_core import (handles, obb, corners, _pt_seg_many, _cand_pairs,
                     signed_gap, HALF_W, RAD, R_TURN, N_BENCH)

R_HI = 6.0                       # r_head 扫描上界
TWO_PI = 2 * np.pi


def bench_theta(P, pitch):
    """各节板凳中心的（展开）极角。等距螺线上 theta = r / b，r 沿队伍单调递增。"""
    b = pitch / TWO_PI
    th = np.linalg.norm(P, axis=1) / b
    return 0.5 * (th[:-1] + th[1:])


def signed_gap_pairs(P, I, J):
    """指定板凳对集合上的最小带符号间隙（>0 精确距离，<=0 为 -穿透深度）。"""
    C, u, nv, hl, _ = obb(P)
    if len(I) == 0:
        return np.inf, None
    dC = C[J] - C[I]
    s = np.full(len(I), -np.inf)
    for ax in (u[I], nv[I], u[J], nv[J]):
        ri = np.abs(np.einsum('ij,ij->i', ax, u[I])) * hl[I] \
           + np.abs(np.einsum('ij,ij->i', ax, nv[I])) * HALF_W
        rj = np.abs(np.einsum('ij,ij->i', ax, u[J])) * hl[J] \
           + np.abs(np.einsum('ij,ij->i', ax, nv[J])) * HALF_W
        s = np.maximum(s, np.abs(np.einsum('ij,ij->i', dC, ax)) - ri - rj)
    val = s.copy()
    far = s > 0.0
    if far.any():
        V = corners(C, u, nv, hl)
        Vi, Vj = V[I[far]], V[J[far]]
        best = np.full(int(far.sum()), np.inf)
        for X, Y in ((Vi, Vj), (Vj, Vi)):
            for k in range(4):
                for m in range(4):
                    best = np.minimum(best, _pt_seg_many(
                        X[:, k], Y[:, m], Y[:, (m + 1) % 4]))
        val[far] = best
    k = int(np.argmin(val))
    return float(val[k]), (int(I[k]) + 1, int(J[k]) + 1)


def outer_turn_j(P, pitch, lo=np.pi, hi=3 * np.pi):
    """龙头外层一圈的板凳下标（0-based）：中心极角比龙头超出 (pi, 3pi)，
    即以 +2pi（整一圈）为中心、宽度一圈的角度窗，并排除铰接邻居 j<2。"""
    d = bench_theta(P, pitch)
    d = d - d[0]
    J = np.nonzero((d > lo) & (d < hi))[0]
    return J[J >= 2]


# ---- 三种判据，签名统一为 crit(P, pitch) -> (gap, pair) ----
def crit_full(P, pitch):
    return signed_gap(P, slack=0.6)


def crit_head_outer(P, pitch):
    J = outer_turn_j(P, pitch)
    return signed_gap_pairs(P, np.zeros(len(J), int), J)


def crit_head_all(P, pitch):
    J = np.arange(2, len(P) - 1)
    return signed_gap_pairs(P, np.zeros(len(J), int), J)


def worst_gap(pitch, crit, dr=0.005, r_hi=R_HI, refine=True):
    """盘入 [4.5, r_hi] 全程的最小带符号间隙 -> (G, 最紧 r_head, 板凳对)。"""
    best = (np.inf, None, None)
    for rh in np.arange(r_hi, R_TURN - 1e-12, -dr):
        g, pr = crit(handles(pitch, float(rh)), pitch)
        if g < best[0]:
            best = (g, float(rh), pr)
        if g <= 0.0:
            return best
    if refine and best[1] is not None:
        a, b = max(R_TURN, best[1] - dr), min(r_hi, best[1] + dr)
        for rh in np.linspace(a, b, 25):
            g, pr = crit(handles(pitch, float(rh)), pitch)
            if g < best[0]:
                best = (g, float(rh), pr)
    return best


def solve_pstar(crit, lo=0.43, hi=0.48, iters=26, dr=0.005):
    """二分 G(p)=0 求最小螺距。"""
    assert worst_gap(lo, crit, 0.01, refine=False)[0] < 0, 'lo 应碰撞'
    assert worst_gap(hi, crit, 0.01, refine=False)[0] > 0, 'hi 应可行'
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if worst_gap(mid, crit, dr)[0] > 0:
            hi = mid
        else:
            lo = mid
    return hi
