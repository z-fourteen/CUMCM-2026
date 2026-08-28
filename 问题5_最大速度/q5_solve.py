# -*- coding: utf-8 -*-
"""问题 5：确定龙头最大行进速度，使各把手速度均不超过 2 m/s。

模型
----
沿问题 4 设定的路径行进。路径按弧长参数化，故龙头速率 v0 恒定时
    s_head(t) = s0 + v0 * t
相邻把手速度比只取决于队形几何：
    v_{i+1}/v_i = phi'(S_i) = (d·u(S_i)) / (d·u(S_{i+1})),   d = P(S_i) - P(S_{i+1})
于是
    v_i(s_head) = v0 * K_i(s_head),   K_i = prod_{j<i} phi'(S_j)
**放大系数 K 与 v0 无关**（改变 v0 只是重新参数化时间，不改变队形轨迹）。
因此约束 max_i v_i <= 2 等价于
    v0 <= 2 / Kmax,     Kmax = max over 全程、over 全部把手 of K_i

于是问题 5 化为求一个纯几何量 Kmax，不需要对 v0 做搜索或迭代。

扫描范围
--------
K 只在队伍与调头曲线有交集时才偏离 1（纯螺线段上 K ≈ 1）。
全队长 369.16 m，故 s_head 自 -(BODY+20) 扫到 L+BODY+20 即完整覆盖。
"""
import sys, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '问题4_调头曲线'))
from dragon4 import (make, Path, Team, follow_stats,          # noqa: E402
                     signed_gap_P, CHORDS, BODY, N_BENCH)

V_LIMIT = 2.0


def load_best():
    f = os.path.join(HERE, '..', '问题4_调头曲线', 'q4_best.npy')
    if os.path.exists(f):
        rA, rB = np.load(f)
        return float(rA), float(rB)
    return 4.279, 4.479


def amp_scan(tm, s_lo, s_hi, ds):
    """全程放大系数扫描 -> (s, Kmax_of_s, argmax_handle)。"""
    S = np.arange(s_lo, s_hi + 1e-9, ds)
    kmax = np.empty(len(S))
    karg = np.empty(len(S), int)
    for i, sh in enumerate(S):
        Si = tm.S_of(float(sh))
        K = np.concatenate([[1.0], np.cumprod(tm.ratios(Si))])
        kmax[i] = K.max()
        karg[i] = int(K.argmax())
    return S, kmax, karg


def refine(tm, s0, ds, iters=40):
    """在 s0 附近用黄金分割细化 Kmax 的峰值位置。"""
    def f(sh):
        Si = tm.S_of(float(sh))
        return np.concatenate([[1.0], np.cumprod(tm.ratios(Si))]).max()

    a, b = s0 - ds, s0 + ds
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc > fd:                      # 求最大值
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)
    sh = 0.5 * (a + b)
    return sh, f(sh)


def all_peaks(S, kmax, n_top=10):
    """kmax(s) 的全部局部极大，按高度降序。粗网格可能对窄峰欠采样，
    故每个候选峰都要单独细化，不能只细化最高的那个。"""
    idx = np.nonzero((kmax[1:-1] >= kmax[:-2]) &
                     (kmax[1:-1] >= kmax[2:]))[0] + 1
    order = idx[np.argsort(kmax[idx])[::-1]]
    return order[:n_top]


if __name__ == '__main__':
    rA, rB = load_best()
    g = make(rA, rB)
    p = Path(g)
    tm = Team(p)
    print('=' * 76)
    print('问题 5  龙头最大行进速度')
    print('=' * 76)
    print('沿问题 4 路径: rA=%.6f rB=%.6f  R1=%.6f R2=%.6f  L=%.6f m'
          % (rA, rB, g['R1'], g['R2'], p.L))
    print('单杆速度比 phi\' ∈ [%.6f, %.6f]' % (tm.dmin, tm.dmax))

    print('\n[1] 关键性质：放大系数 K 与 v0 无关（数值验证）')
    for v0 in (0.5, 1.0, 1.7):
        Si, P, v = tm.state(3.0, v0)
        print('  v0=%.2f m/s: max v_i = %.6f,  max v_i / v0 = %.6f'
              % (v0, v.max(), v.max() / v0))

    print('\n[2] 定位 K 偏离 1 的区段（远离调头区时 K ≈ 1，无须细扫）')
    s_lo, s_hi = -(BODY + 20.0), p.L + BODY + 20.0
    S0, k0, _ = amp_scan(tm, s_lo, s_hi, 2.0)
    act = np.nonzero(k0 > 1.02)[0]
    print('  全域粗扫 s_head ∈ [%.1f, %.1f]，K>1.02 的区段:'
          % (s_lo, s_hi))
    print('    s_head ∈ [%.1f, %.1f]  (其余处 K<=1.02)'
          % (S0[act].min(), S0[act].max()))
    w_lo = max(s_lo, S0[act].min() - 6.0)
    w_hi = min(s_hi, S0[act].max() + 6.0)

    print('\n[3] 有效区段粗扫 + 峰值细化')
    # 全域细扫代价高且无必要：K 的峰很窄，粗扫定位候选峰后逐个细化即可。
    DS_C = 0.5
    S, kmax, karg = amp_scan(tm, w_lo, w_hi, DS_C)
    k = int(kmax.argmax())
    print('  粗扫 ds=%.2f: Kmax=%.6f @ s_head=%.3f 第 %d 把手'
          % (DS_C, kmax[k], S[k], karg[k]))

    tops = all_peaks(S, kmax, n_top=8)
    print('  候选局部极大 %d 个，逐个黄金分割细化（窄峰在粗网格上被低估）:'
          % len(tops))
    best = (-np.inf, None)
    for j in tops:
        sh_j, K_j = refine(tm, float(S[j]), 1.5 * DS_C)
        flag = ''
        if K_j > best[0]:
            best = (K_j, sh_j)
            flag = '  <- 当前最大'
        print('    粗峰 s=%9.4f K=%.6f  ->  细化 s=%9.4f K=%.6f%s'
              % (S[j], kmax[j], sh_j, K_j, flag))
    Kmax, sh = best

    print('\n[4] 最高峰邻域细扫复核（确认细化未落入次峰）')
    S4, k4, a4 = amp_scan(tm, sh - 1.0, sh + 1.0, 0.01)
    j4 = int(k4.argmax())
    print('  ds=0.01 局部细扫: Kmax=%.6f @ s_head=%.4f 第 %d 把手'
          % (k4[j4], S4[j4], a4[j4]))
    print('  与黄金分割结果之差 = %+.2e' % (k4[j4] - Kmax))
    if k4[j4] > Kmax:
        Kmax, sh = float(k4[j4]), float(S4[j4])
    Si = tm.S_of(sh)
    K = np.concatenate([[1.0], np.cumprod(tm.ratios(Si))])
    i_max = int(K.argmax())
    v_max = V_LIMIT / Kmax

    seg = lambda x: ('盘入螺线' if x < 0 else
                     ('弧1' if x <= p.L1 else
                      ('弧2' if x <= p.L else '盘出螺线')))
    print('\n[5] 结论')
    print('  最不利时刻 s_head = %.6f m  (龙头位于 %s)' % (sh, seg(sh)))
    print('  Kmax = %.6f  出现在第 %d 把手，其 S = %.6f m (%s)'
          % (Kmax, i_max, Si[i_max], seg(Si[i_max])))
    print('  相邻把手速度比 phi\' 沿链的最大值 = %.6f' % tm.ratios(Si).max())
    print('\n  >>> 龙头最大行进速度 v0_max = 2 / Kmax = %.6f m/s' % v_max)

    print('\n[6] 复核：以 v0_max 行进，有效区段内全程速度上界')
    vv = v_max * kmax
    print('  max v_i = %.6f m/s  (限值 %.1f)  满足约束: %s'
          % (vv.max(), V_LIMIT, bool(vv.max() <= V_LIMIT + 1e-6)))
    print('  该时刻各把手速度（以 v0_max 计）: max=%.6f, min=%.6f'
          % ((v_max * K).max(), (v_max * K).min()))
    print('  放大系数最大的前 6 个把手（最不利时刻）:')
    for i in np.argsort(K)[::-1][:6]:
        print('    第 %3d 把手: K=%.6f -> v=%.6f m/s' % (i, K[i], v_max * K[i]))

    np.savez(os.path.join(HERE, 'q5_result.npz'),
             rA=rA, rB=rB, Kmax=Kmax, s_peak=sh, i_peak=i_max, v_max=v_max)
    print('\n已保存 -> q5_result.npz')
