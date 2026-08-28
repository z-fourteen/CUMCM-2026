# -*- coding: utf-8 -*-
"""问题 3 核心：把手递推 + 精确矩形(OBB)碰撞判定，向量化实现。"""
import numpy as np
from math import cos, sin, pi, sqrt, asinh

W       = 0.30                    # 板宽
HALF_W  = W / 2                   # 0.15
D_END   = 0.275                   # 孔心到最近板头
L_HEAD  = 3.41
L_BODY  = 2.20
CH_HEAD = L_HEAD - 2 * D_END      # 2.86 龙头两孔间距
CH_BODY = L_BODY - 2 * D_END      # 1.65 龙身两孔间距
N_BENCH = 223                     # 1 龙头 + 221 龙身 + 1 龙尾
R_TURN  = 4.5                     # 调头空间半径

CHORDS = np.array([CH_HEAD] + [CH_BODY] * (N_BENCH - 1))
PLEN   = np.array([L_HEAD]  + [L_BODY]  * (N_BENCH - 1))
HL     = PLEN / 2                 # 沿板长半长（已含两端 0.275 外伸）
RAD    = np.hypot(HL, HALF_W)     # 外接圆半径


def handles(pitch, r_head, nb=N_BENCH):
    """把手中心 P_1..P_{nb+1}：龙头前把手在半径 r_head，沿 r=b*theta 向外逐节递推。
    牛顿法解 b^2(t^2+t0^2-2*t*t0*cos(t-t0))=L^2 在 t>t0 的第一个根。"""
    b = pitch / (2 * pi)
    th = np.empty(nb + 1)
    th[0] = r_head / b
    for i in range(nb):
        L, t0 = CHORDS[i], th[i]
        t = t0 + L / (b * t0)
        for _ in range(60):
            d = t - t0
            c, s = cos(d), sin(d)
            f = b * b * (t * t + t0 * t0 - 2 * t * t0 * c) - L * L
            fp = 2 * b * b * (t - t0 * c + t * t0 * s)
            st = f / fp
            t -= st
            if abs(st) < 1e-14:
                break
        assert 0.0 < t - t0 < pi, (i, t - t0)
        th[i + 1] = t
    r = b * th
    return np.column_stack([r * np.cos(th), r * np.sin(th)])


def obb(P):
    """把手点列 -> 各节板凳的有向矩形 (中心 C, 长轴 u, 短轴 n, 半长 hl, 半宽 hw)。
    矩形 = 把手连线两端各外伸 0.275、横向各外扩 0.15。"""
    A, B = P[:-1], P[1:]
    d = B - A
    u = d / np.linalg.norm(d, axis=1, keepdims=True)
    nv = np.column_stack([-u[:, 1], u[:, 0]])
    C = 0.5 * (A + B)
    n = len(C)
    return C, u, nv, HL[:n], HALF_W


def corners(C, u, nv, hl):
    """OBB -> 四角点 (n,4,2)。"""
    hu = u * hl[:, None]
    hn = nv * HALF_W
    return np.stack([C + hu + hn, C + hu - hn, C - hu - hn, C - hu + hn], axis=1)


def _sat_pairs(C, u, nv, hl, I, J):
    """向量化 SAT：对给定索引对 (I,J) 判定 OBB 是否相交（接触算相交）。
    四条分离轴 = 两矩形各自的长轴与短轴；任一轴上投影分离即不相交。"""
    dC = C[J] - C[I]
    res = np.ones(len(I), bool)
    for ax in (u[I], nv[I], u[J], nv[J]):
        # OBB 在轴 ax 上的投影半径 = |ax·u|*hl + |ax·n|*hw
        ri = np.abs(np.einsum('ij,ij->i', ax, u[I])) * hl[I] \
           + np.abs(np.einsum('ij,ij->i', ax, nv[I])) * HALF_W
        rj = np.abs(np.einsum('ij,ij->i', ax, u[J])) * hl[J] \
           + np.abs(np.einsum('ij,ij->i', ax, nv[J])) * HALF_W
        dist = np.abs(np.einsum('ij,ij->i', dC, ax))
        res &= dist <= ri + rj          # 该轴未分离
        if not res.any():
            break
    return res


def _cand_pairs(C, rad, slack=0.0):
    """外接圆粗筛，返回 j>=i+2 的候选索引对。"""
    n = len(C)
    D = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=2)
    thr = rad[:, None] + rad[None, :] + slack
    iu = np.triu(np.ones((n, n), bool), 2)
    ij = np.argwhere(iu & (D < thr))
    return ij[:, 0], ij[:, 1]


def collides(P):
    """是否存在 |i-j|>=2 的板凳相交。返回 (bool, (i,j) 1-based)。"""
    C, u, nv, hl, _ = obb(P)
    I, J = _cand_pairs(C, RAD[:len(C)])
    if len(I) == 0:
        return False, None
    hit = _sat_pairs(C, u, nv, hl, I, J)
    if not hit.any():
        return False, None
    k = np.argmax(hit)
    return True, (int(I[k]) + 1, int(J[k]) + 1)


def _pt_seg_many(p, a, b):
    """点到线段距离（逐对向量化）。p,a,b 形状 (m,2)。"""
    ab = b - a
    denom = np.einsum('ij,ij->i', ab, ab)
    t = np.where(denom > 1e-15,
                 np.einsum('ij,ij->i', p - a, ab) / np.maximum(denom, 1e-300), 0.0)
    t = np.clip(t, 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def min_gap(P, slack=0.5):
    """最小间隙及最近对。相交则间隙为 0。slack 须大于真实最小间隙方保证正确。"""
    C, u, nv, hl, _ = obb(P)
    I, J = _cand_pairs(C, RAD[:len(C)], slack)
    if len(I) == 0:
        return np.inf, None
    hit = _sat_pairs(C, u, nv, hl, I, J)
    if hit.any():
        k = np.argmax(hit)
        return 0.0, (int(I[k]) + 1, int(J[k]) + 1)
    V = corners(C, u, nv, hl)
    Vi, Vj = V[I], V[J]                       # (m,4,2)
    best = np.full(len(I), np.inf)
    for X, Y in ((Vi, Vj), (Vj, Vi)):
        for k in range(4):
            for m in range(4):
                best = np.minimum(best, _pt_seg_many(X[:, k], Y[:, m], Y[:, (m + 1) % 4]))
    k = int(np.argmin(best))
    return float(best[k]), (int(I[k]) + 1, int(J[k]) + 1)


def signed_gap(P, slack=0.5):
    """带符号间隙（连续函数，便于求根）：
      >0  两矩形分离，值 = 精确最短距离
      <=0 两矩形相交，值 = -穿透深度（SAT 四轴最小重叠）
    返回 (全队最小值, 对应板凳对 1-based)。"""
    C, u, nv, hl, _ = obb(P)
    I, J = _cand_pairs(C, RAD[:len(C)], slack)
    if len(I) == 0:
        return np.inf, None
    dC = C[J] - C[I]
    s = np.full(len(I), -np.inf)                  # 各轴 (dist-ri-rj) 的最大值
    for ax in (u[I], nv[I], u[J], nv[J]):
        ri = np.abs(np.einsum('ij,ij->i', ax, u[I])) * hl[I] \
           + np.abs(np.einsum('ij,ij->i', ax, nv[I])) * HALF_W
        rj = np.abs(np.einsum('ij,ij->i', ax, u[J])) * hl[J] \
           + np.abs(np.einsum('ij,ij->i', ax, nv[J])) * HALF_W
        dist = np.abs(np.einsum('ij,ij->i', dC, ax))
        s = np.maximum(s, dist - ri - rj)
    val = s.copy()                                # s<=0 即穿透，直接用作负间隙
    far = s > 0.0
    if far.any():                                 # 分离对改用精确距离
        V = corners(C, u, nv, hl)
        Vi, Vj = V[I[far]], V[J[far]]
        best = np.full(far.sum(), np.inf)
        for X, Y in ((Vi, Vj), (Vj, Vi)):
            for k in range(4):
                for m in range(4):
                    best = np.minimum(best, _pt_seg_many(
                        X[:, k], Y[:, m], Y[:, (m + 1) % 4]))
        val[far] = best
    k = int(np.argmin(val))
    return float(val[k]), (int(I[k]) + 1, int(J[k]) + 1)


def arc(theta, b):
    """等距螺线 r=b*theta 弧长。"""
    return 0.5 * b * (theta * sqrt(1 + theta * theta) + asinh(theta))
