# -*- coding: utf-8 -*-
"""问题 4 / 5 共享内核：S 形调头曲线、复合路径、刚性链跟随、碰撞判据。

坐标与约定
----------
盘入螺线   P(t) = b*t*(cos t, sin t)，b = 1.7/(2*pi)，龙头沿 t 减小方向盘入
行进方向   u(t) = -P'(t)/|P'(t)|
盘出螺线   Q(t) = -P(t)（与盘入螺线中心对称），沿 t 增大盘出，行进方向亦为 u(t)
调头曲线   弧1(半径 R1) 切盘入螺线于 A = P(tA)
           弧2(半径 R2 = R1/2) 切盘出螺线于 B = Q(tB)
           两弧外切于 C，转向相反（S 形）

弧长坐标 s：s = 0 在 A，s 增大为行进方向
    s < 0        盘入螺线
    0 <= s <= L1 弧1
    L1 < s <= L  弧2
    s > L        盘出螺线
龙头速度 1 m/s 时 s_head = t（秒），故弧长坐标与题目时间同刻度。
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '问题3_最小螺距'))
from q3_core import (CHORDS, HL, HALF_W, RAD, N_BENCH,      # noqa: E402
                     corners, _pt_seg_many)

PITCH  = 1.7
B_SP   = PITCH / (2 * np.pi)
R_TURN = 4.5                      # 调头空间半径
RATIO  = 2.0                      # R1 : R2
CH_HEAD, CH_BODY = float(CHORDS[0]), float(CHORDS[1])
BODY = float(CHORDS.sum())        # 全队把手弦长之和 369.16 m


# ======================================================================
# 一、S 形调头曲线的构造（两自由度：等价于 (R1, alpha1) 或 (tA, tB)）
# ======================================================================
def rot90(v):
    return np.array([-v[1], v[0]])


def spiral(t):
    """盘入螺线上的点 P(t) 与行进方向 u(t)。"""
    c, s = np.cos(t), np.sin(t)
    P = B_SP * t * np.array([c, s])
    T = B_SP * np.array([c - t * s, s + t * c])
    return P, -T / np.linalg.norm(T)


def sweep(O, P0, P1, sense):
    """P0 绕 O 按 sense(+1 逆时针) 转到 P1 的正扫角 ∈ [0, 2*pi)。"""
    v0, v1 = P0 - O, P1 - O
    a = (np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])) * sense
    return float(np.mod(a, 2 * np.pi))


def build(tA, tB, sigma=-1.0, ratio=RATIO):
    """由两切点极角构造 S 形曲线；外切条件是 R1 的一元二次方程。"""
    A, uA = spiral(tA)
    Pb, uB = spiral(tB)
    B, uBq = -Pb, uB                      # 盘出螺线 Q=-P，行进方向同为 u
    n1 = sigma * rot90(uA)
    n2 = -sigma * rot90(uBq)
    w = n2 / ratio - n1                   # O2-O1 = (B-A) + R1*w
    d = B - A
    k = 1.0 + 1.0 / ratio

    aa, bb, cc = w @ w - k * k, 2.0 * (d @ w), d @ d
    if abs(aa) < 1e-13:
        roots = [-cc / bb] if abs(bb) > 1e-13 else []
    else:
        disc = bb * bb - 4 * aa * cc
        if disc < -1e-12:
            return None
        sq = np.sqrt(max(disc, 0.0))
        roots = [(-bb + sq) / (2 * aa), (-bb - sq) / (2 * aa)]

    best = None
    for R1 in roots:
        if not np.isfinite(R1) or R1 <= 1e-6:
            continue
        R2 = R1 / ratio
        O1, O2 = A + R1 * n1, B + R2 * n2
        if abs(np.linalg.norm(O2 - O1) - (R1 + R2)) > 1e-7:
            continue
        C = O1 + R1 * (O2 - O1) / np.linalg.norm(O2 - O1)
        a1 = sweep(O1, A, C, sigma)
        a2 = sweep(O2, C, B, -sigma)
        g = dict(tA=tA, tB=tB, sigma=sigma, A=A, B=B, C=C, O1=O1, O2=O2,
                 R1=R1, R2=R2, a1=a1, a2=a2, L=R1 * a1 + R2 * a2,
                 uA=uA, uB=uBq, n1=n1, n2=n2)
        if best is None or g['L'] < best['L']:
            best = g
    return best


def arc_pts(O, P0, ang, sense, m):
    v0 = P0 - O
    R = np.linalg.norm(v0)
    ph = np.arctan2(v0[1], v0[0]) + sense * np.linspace(0.0, ang, m)
    return O + R * np.column_stack([np.cos(ph), np.sin(ph)])


def sample(g, m=3000):
    p1 = arc_pts(g['O1'], g['A'], g['a1'], g['sigma'], m)
    p2 = arc_pts(g['O2'], g['C'], g['a2'], -g['sigma'], m)
    return np.vstack([p1, p2[1:]])


def validate(g, tol=1e-7):
    """相切 + 行进方向连续 + 不出调头空间。方向判据滤掉几何相切但反向的伪解。"""
    A, B, C = g['A'], g['B'], g['C']
    R1, R2, s = g['R1'], g['R2'], g['sigma']
    e = {}
    e['tan_A'] = abs((A - g['O1']) @ g['uA']) / R1
    e['tan_B'] = abs((B - g['O2']) @ g['uB']) / R2
    e['ext_tan'] = abs(np.linalg.norm(g['O2'] - g['O1']) - (R1 + R2))
    e['dir_A'] = float((s * rot90(A - g['O1']) / R1) @ g['uA'])
    e['dir_B'] = float((-s * rot90(B - g['O2']) / R2) @ g['uB'])
    e['dir_C'] = float((s * rot90(C - g['O1']) / R1) @
                       (-s * rot90(C - g['O2']) / R2))
    e['r_max'] = float(np.linalg.norm(sample(g), axis=1).max())
    ok = (e['tan_A'] < 1e-9 and e['tan_B'] < 1e-9 and e['ext_tan'] < 1e-7
          and e['dir_A'] > 1 - 1e-6 and e['dir_B'] > 1 - 1e-6
          and e['dir_C'] > 1 - 1e-6 and e['r_max'] <= R_TURN + 1e-9)
    return ok, e


def make(rA, rB, sigma=-1.0):
    """按两切点半径构造并校验，不合格返回 None。"""
    g = build(rA / B_SP, rB / B_SP, sigma)
    if g is None:
        return None
    return g if validate(g)[0] else None


# ======================================================================
# 二、复合路径的弧长参数化
# ======================================================================
def spiral_arc(t, b=B_SP):
    """等距螺线 r=b*t 自 t=0 起的弧长。"""
    return 0.5 * b * (t * np.sqrt(1 + t * t) + np.arcsinh(t))


def t_of_arc(S, b=B_SP):
    """由弧长反解极角 t。牛顿法，d(arc)/dt = b*sqrt(1+t^2)。"""
    S = np.atleast_1d(np.asarray(S, float))
    t = np.sqrt(np.maximum(2.0 * S / b, 0.0)) + 1e-9
    for _ in range(60):
        step = (spiral_arc(t, b) - S) / (b * np.sqrt(1.0 + t * t))
        t = np.maximum(t - step, 1e-12)
        if np.max(np.abs(step)) < 1e-13:
            break
    return t


def _chord_out(t0, L, b=B_SP):
    """螺线上自 t0 向外(t 增大)满足弦长 = L 的第一个 t。"""
    t = t0 + L / (b * t0)
    for _ in range(60):
        d = t - t0
        c, s = np.cos(d), np.sin(d)
        f = b * b * (t * t + t0 * t0 - 2 * t * t0 * c) - L * L
        fp = 2 * b * b * (t - t0 * c + t * t0 * s)
        st = f / fp
        t -= st
        if abs(st) < 1e-14:
            break
    return t


class Path:
    """复合调头路径：盘入螺线 -> 弧1 -> 弧2 -> 盘出螺线，按弧长参数化。"""

    def __init__(self, g, b=B_SP):
        self.g, self.b = g, b
        self.L1 = g['R1'] * g['a1']
        self.L2 = g['R2'] * g['a2']
        self.L = self.L1 + self.L2
        self.sA = spiral_arc(g['tA'], b)
        self.sB = spiral_arc(g['tB'], b)

    def pos(self, s):
        """弧长坐标 -> 平面坐标 (m,2)。"""
        s = np.atleast_1d(np.asarray(s, float))
        out = np.empty((len(s), 2))
        g, b = self.g, self.b

        m = s < 0.0                                   # 盘入螺线
        if m.any():
            t = t_of_arc(self.sA - s[m], b)
            out[m] = (b * t)[:, None] * np.column_stack([np.cos(t), np.sin(t)])
        m = (s >= 0.0) & (s <= self.L1)               # 弧1
        if m.any():
            ph = np.arctan2(*(g['A'] - g['O1'])[::-1]) + g['sigma'] * s[m] / g['R1']
            out[m] = g['O1'] + g['R1'] * np.column_stack([np.cos(ph), np.sin(ph)])
        m = (s > self.L1) & (s <= self.L)             # 弧2
        if m.any():
            ph = (np.arctan2(*(g['C'] - g['O2'])[::-1])
                  - g['sigma'] * (s[m] - self.L1) / g['R2'])
            out[m] = g['O2'] + g['R2'] * np.column_stack([np.cos(ph), np.sin(ph)])
        m = s > self.L                                # 盘出螺线 Q = -P
        if m.any():
            t = t_of_arc(self.sB + (s[m] - self.L), b)
            out[m] = -((b * t)[:, None] * np.column_stack([np.cos(t), np.sin(t)]))
        return out

    def tangent(self, s):
        """弧长坐标 -> 单位行进方向 (m,2)。"""
        s = np.atleast_1d(np.asarray(s, float))
        out = np.empty((len(s), 2))
        g, b = self.g, self.b

        def spir_dir(t):
            c, sn = np.cos(t), np.sin(t)
            T = b * np.column_stack([c - t * sn, sn + t * c])
            return -T / np.linalg.norm(T, axis=1, keepdims=True)

        m = s < 0.0
        if m.any():
            out[m] = spir_dir(t_of_arc(self.sA - s[m], b))
        m = (s >= 0.0) & (s <= self.L1)
        if m.any():
            v = self.pos(s[m]) - g['O1']
            out[m] = g['sigma'] * np.column_stack([-v[:, 1], v[:, 0]]) / g['R1']
        m = (s > self.L1) & (s <= self.L)
        if m.any():
            v = self.pos(s[m]) - g['O2']
            out[m] = -g['sigma'] * np.column_stack([-v[:, 1], v[:, 0]]) / g['R2']
        m = s > self.L
        if m.any():
            out[m] = spir_dir(t_of_arc(self.sB + (s[m] - self.L), b))
        return out


# ======================================================================
# 三、刚性链的连续跟随
# ======================================================================
# 定位后继把手要解 |P(s) - P(s')| = L。曾用"沿路径向后的第一个根"，
# 但当圆弧小到板凳可横跨整段弧(2*R2 < 弦长)时该判据不连续：根成对生灭，
# "第一个根"发生跳变，把手位置突变、速度出现负值。
# 正确判据是**连续延拓**：队伍自远处盘入(全队在螺线上、根唯一)出发，
# 每个把手的 s' 沿同一分支连续变化，用上一步的解作牛顿初值即可锁定分支。
def _newton_back(p, P0, L, guess, iters=60):
    """在 guess 附近解 |P(s') - P0| = L，返回同一分支上的根。"""
    s = float(guess)
    for _ in range(iters):
        d = p.pos(s)[0] - P0
        r = float(np.hypot(d[0], d[1]))
        fp = float(d @ p.tangent(s)[0]) / r
        if abs(fp) < 1e-14:
            break
        st = (r - L) / fp
        s -= st
        if abs(st) < 1e-14:
            break
    return s


def phi_scan(p, L, s_lo, s_hi, ds=0.01):
    """长度 L 的板凳的连续延拓映射 phi_L: s -> s'，及其导数 phi'。

    链上每个把手都走同一条路径，故同长度板凳共用一个 phi，与队伍多长无关。
    phi'(s) = (d·u(s)) / (d·u(s')),  d = P(s) - P(s')，即相邻把手速度比。
    -> (s, s', phi')
    """
    s = np.arange(s_lo, s_hi + 0.5 * ds, ds)
    t0 = float(t_of_arc(p.sA - s[0], p.b)[0])
    sp = np.empty_like(s)
    sp[0] = p.sA - spiral_arc(_chord_out(t0, L, p.b), p.b)   # 螺线上根唯一
    for k in range(1, len(s)):
        sp[k] = _newton_back(p, p.pos(s[k])[0], L,
                             sp[k - 1] + (s[k] - s[k - 1]))
    U0, U1 = p.tangent(s), p.tangent(sp)
    d = p.pos(s) - p.pos(sp)
    der = (np.einsum('ij,ij->i', d, U0) /
           np.einsum('ij,ij->i', d, U1))
    return s, sp, der


def follow_stats(p, ds=0.01, margin=8.0):
    """两种弦长的 phi' 统计。"""
    out = {}
    for tag, L in (('head', CH_HEAD), ('body', CH_BODY)):
        s, sp, der = phi_scan(p, L, -L - margin, p.L + margin, ds)
        out[tag] = dict(L=L, s=s, sp=sp, der=der,
                        dmin=float(der.min()), dmax=float(der.max()),
                        s_dmin=float(s[der.argmin()]),
                        s_dmax=float(s[der.argmax()]))
    out['dmin'] = min(out['head']['dmin'], out['body']['dmin'])
    out['dmax'] = max(out['head']['dmax'], out['body']['dmax'])
    return out


def followable(p, ds=0.01):
    """可跟随性：phi' 全程 > 0。phi' <= 0 意味着后把手须倒退才能维持杆长，
    队伍无法沿该路径整体前进。这是几何相切之外的独立可行性条件。"""
    st = follow_stats(p, ds)
    return st['dmin'] > 0.0, st


def _chord_in_spiral(t0, L, b=B_SP):
    """螺线上自 t0 向内(t 减小)满足弦长 = L 的第一个 t；越过 t=0 则返回 None。"""
    if b * t0 < L * 0.5:
        return None
    t = t0 - L / (b * t0)
    if t <= 0.0:
        return None
    for _ in range(60):
        d = t - t0
        c, s = np.cos(d), np.sin(d)
        f = b * b * (t * t + t0 * t0 - 2 * t * t0 * c) - L * L
        fp = 2 * b * b * (t - t0 * c + t * t0 * s)
        if abs(fp) < 1e-14:
            return None
        st = f / fp
        t -= st
        if t <= 0.0:
            return None
        if abs(st) < 1e-14:
            break
    return t if 0.0 < t < t0 else None


class Follower:
    """长度 L 的板凳的连续延拓映射 phi_L: s -> s'，带查表加速。

    直接对每个把手做"march + 牛顿"代价极高（整队 224 节 x 上万时间步）。
    但 phi_L 是一条**固定的一维函数**，与时间、与队伍长度无关，
    故只需在调头区附近离散化一次，之后查表取初值 + 一步牛顿精修即可，
    精度与逐步 march 相同而无须时间推进。

    两侧远场有解析捷径：
      s < 0        前后把手同在盘入螺线 -> 螺线弦长方程
      s' > L       前后把手同在盘出螺线 -> 螺线弦长方程（向内）
    只有跨越 弧/螺线 接缝的一小段需要查表。
    """

    def __init__(self, p, L, ds=0.002, margin=1.5):
        self.p, self.L = p, float(L)
        self.s_hi = p.L + margin * self.L + 2.0      # 表域上界
        s, sp, der = phi_scan(p, self.L, -self.L - 6.0, self.s_hi, ds)
        self.tab_s, self.tab_sp, self.tab_der = s, sp, der
        self.dmin, self.dmax = float(der.min()), float(der.max())

    def _polish(self, s, guess):
        return _newton_back(self.p, self.p.pos(s)[0], self.L, guess)

    def __call__(self, s):
        """单点求后继把手弧长坐标。"""
        p, b, L = self.p, self.p.b, self.L
        s = float(s)
        if s < 0.0:                                  # 同在盘入螺线
            t0 = float(t_of_arc(p.sA - s, b)[0])
            return p.sA - spiral_arc(_chord_out(t0, L, b), b)
        if s > self.s_hi:                            # 同在盘出螺线
            t0 = float(t_of_arc(p.sB + (s - p.L), b)[0])
            t1 = _chord_in_spiral(t0, L, b)
            if t1 is not None and t1 >= p.g['tB']:
                return p.L + (spiral_arc(t1, b) - p.sB)
        g = float(np.interp(s, self.tab_s, self.tab_sp))
        return self._polish(s, g)


class Team:
    """整队把手定位：给定龙头弧长坐标，直接逐节应用 phi（无须时间推进）。

    可行时 phi' > 0，phi 单值单调，故 S_i = phi^i(s_head) 唯一确定队形。
    """

    def __init__(self, p, chords=CHORDS, ds=0.002):
        self.p, self.ch = p, np.asarray(chords, float)
        uniq = sorted(set(np.round(self.ch, 12)))
        self.fol = {u: Follower(p, u, ds=ds) for u in uniq}
        self.dmin = min(f.dmin for f in self.fol.values())
        self.dmax = max(f.dmax for f in self.fol.values())

    def S_of(self, s_head):
        S = np.empty(len(self.ch) + 1)
        S[0] = float(s_head)
        for i, L in enumerate(self.ch):
            S[i + 1] = self.fol[round(float(L), 12)](S[i])
        return S

    def ratios(self, S):
        p = self.p
        P, U = p.pos(S), p.tangent(S)
        d = P[:-1] - P[1:]
        return (np.einsum('ij,ij->i', d, U[:-1]) /
                np.einsum('ij,ij->i', d, U[1:]))

    def state(self, s_head, v_head=1.0):
        """-> (S, P[n+1,2], v[n+1])"""
        S = self.S_of(s_head)
        v = v_head * np.concatenate([[1.0], np.cumprod(self.ratios(S))])
        return S, self.p.pos(S), v


class Chain:
    """沿路径连续跟随的把手链（逐步 march，作为 Team 的独立参照）。"""

    def __init__(self, p, chords=CHORDS):
        self.p, self.ch = p, np.asarray(chords, float)
        self.n = len(self.ch)
        self.S = None

    def init(self, s_head):
        """在全队都位于盘入螺线（根唯一）处建立初始条件。"""
        p, b = self.p, self.p.b
        S = np.empty(self.n + 1)
        S[0] = s_head
        for i in range(self.n):
            t0 = float(t_of_arc(p.sA - S[i], b)[0])
            S[i + 1] = p.sA - spiral_arc(_chord_out(t0, self.ch[i], b), b)
        assert S[0] < 0.0 and S[-1] < 0.0, '初始位置未全落在盘入螺线上'
        self.S = S
        return S

    def ratios(self, S=None):
        """相邻把手速度比 v_{i+1}/v_i。"""
        S = self.S if S is None else S
        p = self.p
        P, U = p.pos(S), p.tangent(S)
        d = P[:-1] - P[1:]
        return (np.einsum('ij,ij->i', d, U[:-1]) /
                np.einsum('ij,ij->i', d, U[1:]))

    def speeds(self, v_head=1.0, S=None):
        """各把手速率。由 |P_i - P_{i+1}| = const 微分得递推速度比。"""
        return v_head * np.concatenate([[1.0], np.cumprod(self.ratios(S))])

    def step(self, s_head):
        """推进龙头到 s_head，用上一步解逐节延拓（锁定同一分支）。"""
        p, S_old = self.p, self.S
        rt = self.ratios(S_old)
        S = np.empty_like(S_old)
        S[0] = s_head
        dh = s_head - S_old[0]
        for i in range(self.n):
            dh = dh * rt[i]                 # 用速度比预估该节位移作初值
            S[i + 1] = _newton_back(p, p.pos(S[i])[0], self.ch[i],
                                    S_old[i + 1] + dh)
        self.S = S
        return S

    def run(self, s0, s1, ds, record=None):
        """自 s0 march 到 s1；record 为需精确落点的 s_head 升序数组。
        -> {s_head: S}"""
        self.init(s0)
        out = {}
        rec = None if record is None else np.asarray(record, float)
        ri = 0
        if rec is not None and abs(rec[0] - s0) < 1e-12:
            out[float(rec[0])] = self.S.copy()
            ri = 1
        n = int(np.ceil((s1 - s0) / ds))
        s = s0
        for k in range(1, n + 1):
            nxt = min(s0 + k * ds, s1)
            while rec is not None and ri < len(rec) and rec[ri] <= nxt + 1e-12:
                self.step(float(rec[ri]))
                out[float(rec[ri])] = self.S.copy()
                s = float(rec[ri])
                ri += 1
            if nxt > s + 1e-12:
                self.step(nxt)
                s = nxt
            if rec is None:
                out[s] = self.S.copy()
        return out


# ======================================================================
# 四、碰撞判据（沿用问题 2/3 的精确矩形 OBB + SAT，|i-j| >= 2）
# ======================================================================
def obb_of(P):
    A, B = P[:-1], P[1:]
    d = B - A
    u = d / np.linalg.norm(d, axis=1, keepdims=True)
    nv = np.column_stack([-u[:, 1], u[:, 0]])
    return 0.5 * (A + B), u, nv, HL[:len(A)]


def signed_gap_P(P, slack=0.6):
    """全队最小带符号间隙：>0 为精确距离，<=0 为 -穿透深度。"""
    C, u, nv, hl = obb_of(P)
    n = len(C)
    D = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=2)
    thr = RAD[:n][:, None] + RAD[:n][None, :] + slack
    I, J = np.nonzero(np.triu(np.ones((n, n), bool), 2) & (D < thr))
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
