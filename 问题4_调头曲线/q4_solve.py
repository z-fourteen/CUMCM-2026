# -*- coding: utf-8 -*-
"""问题 4：在两自由度内求最短调头曲线，并输出 result4.xlsx。

设计变量（2 个，与题设约束逐一对应）
------------------------------------
题目已固定：螺距 1.7 m、调头空间半径 4.5 m、R1:R2 = 2:1、三处相切。
在这些约束下 S 形曲线仍有 2 个自由度，取 (rA, rB) = 两切点的极径
（与 (R1, alpha1) 互为坐标变换）。盘入/盘出螺线中心对称，但调头路径本身
不必中心对称，故 rA 与 rB 可独立。

可行性约束
----------
1. 几何：三处相切 + 行进方向连续 + 曲线不超出 r <= 4.5 的调头空间；
2. 碰撞：全程各节板凳(精确矩形)互不相撞，|i-j| >= 2；
3. 可跟随：连续延拓映射的 phi' 全程 > 0。

第 3 条是纯几何优化会忽略的关键约束。弧长最短的解 (rA≈1.81, L≈5.20 m)
把 R2 压到 0.61 m，此时 1.65 m 的板凳横跨整段小弧，后把手须倒退才能维持
杆长，队伍根本无法沿该路径前进。故本题的最优解由**可跟随性**定界，
而非碰撞。
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dragon4 import (make, Path, Chain, follow_stats, followable,   # noqa: E402
                     signed_gap_P, CHORDS, CH_HEAD, CH_BODY, BODY,
                     N_BENCH, R_TURN, PITCH, B_SP)

OUT = os.path.dirname(os.path.abspath(__file__))


def min_gap(g, ds_march=0.25, ds_check=1.0, s_lo=None, s_hi=None,
            early=True, verbose=False):
    """调头全程最小带符号间隙 -> (G, s_head*, 板凳对)。

    march 步长与碰撞检测步长分离：连续延拓需要细步(ds_march)才不跳支，
    但碰撞判据无须每步都算（间隙随 s_head 缓变），故每隔 ds_check 检测一次。
    扫描区间只需覆盖"队伍与调头区有交集"的时段：龙头进入前 BODY 米到
    龙尾驶出调头曲线，即 s_head ∈ [-(BODY+20), L+BODY+20]。
    """
    p = Path(g)
    lo = -(BODY + 20.0) if s_lo is None else s_lo
    hi = (p.L + BODY + 20.0) if s_hi is None else s_hi
    ch = Chain(p)
    ch.init(lo)
    best = (np.inf, None, None)
    n_check = max(1, int(round(ds_check / ds_march)))
    for k, sh in enumerate(np.arange(lo + ds_march, hi + 1e-9, ds_march), 1):
        S = ch.step(float(sh))
        if k % n_check:
            continue
        gap, pr = signed_gap_P(p.pos(S))
        if gap < best[0]:
            best = (gap, float(sh), pr)
            if verbose:
                print('        G=%+.6f @ s_head=%.2f 对 %s' % (gap, sh, pr))
        if early and gap <= 0.0:
            break
    return best


def report(g, tag=''):
    p = Path(g)
    st = follow_stats(p, ds=0.01)
    print('  %s rA=%.6f rB=%.6f' % (tag, np.linalg.norm(g['A']),
                                    np.linalg.norm(g['B'])))
    print('     R1=%.6f R2=%.6f  a1=%.6f a2=%.6f' %
          (g['R1'], g['R2'], g['a1'], g['a2']))
    print('     L1=%.6f L2=%.6f  L=%.6f m' % (p.L1, p.L2, p.L))
    print("     phi' ∈ [%.4f, %.4f]  可跟随=%s" %
          (st['dmin'], st['dmax'], st['dmin'] > 0.0))
    return p, st


if __name__ == '__main__':
    print('=' * 76)
    print('问题 4  最短调头曲线')
    print('=' * 76)
    print('螺距 %.1f m, b=%.6f, 调头空间半径 %.1f m, R1:R2 = 2:1'
          % (PITCH, B_SP, R_TURN))

    print('\n[1] 题给基准：两切点均在 r = 4.5')
    p0, st0 = report(make(R_TURN, R_TURN), '基准')
    L0 = p0.L

    print('\n[2] 纯几何最优（只管相切/不越界，不问运动学）')
    print('    沿对称族 rA=rB 减小 rA，L 单调减小，弧长可一直压到退化，')
    print('    说明"最短"必须由别的约束定界。')
    print('  %8s %9s %9s %10s %11s %11s %6s' %
          ('rA', 'R1', 'R2', 'L', "min phi'", "max phi'", '可跟随'))
    for rA in [4.5, 4.0, 3.5, 3.0, 2.5, 2.2, 2.0, 1.8]:
        g = make(rA, rA)
        if g is None:
            print('  %8.4f  几何不可行' % rA)
            continue
        p = Path(g)
        st = follow_stats(p, ds=0.02)
        print('  %8.4f %9.6f %9.6f %10.6f %11.4f %11.4f %6s'
              % (rA, g['R1'], g['R2'], p.L, st['dmin'], st['dmax'],
                 st['dmin'] > 0.0))

    print('\n[3] 可跟随性临界点：二分 min phi\' = 0（对称族 rA=rB）')
    f = lambda r: follow_stats(Path(make(r, r)), ds=0.01)['dmin']
    lo, hi = 4.0, 4.5
    flo = f(lo)
    print('    f(%.4f)=%+.6f  f(%.4f)=%+.6f' % (lo, flo, hi, f(hi)))
    for _ in range(24):
        m = 0.5 * (lo + hi)
        if f(m) * flo > 0:
            lo = m
        else:
            hi = m
    rc = hi                                  # 取可行侧
    print('    临界 rA ≈ %.6f' % rc)
    # 细化：在可行侧留出安全余量，取 phi' 有明确正下界的最小 rA
    for pad in (1e-4, 1e-3, 1e-2):
        st = follow_stats(Path(make(rc + pad, rc + pad)), ds=0.005)
        print("      rA=%.6f: min phi'=%+.6f" % (rc + pad, st['dmin']))

    print('\n[4] 二维搜索：rA、rB 独立，在三条约束下最小化 L')
    best = None
    grid = np.arange(4.20, 4.52, 0.02)
    for rA in grid:
        for rB in grid:
            g = make(rA, rB)
            if g is None:
                continue
            p = Path(g)
            st = follow_stats(p, ds=0.05)
            if st['dmin'] <= 0.0:
                continue
            if best is None or p.L < best[0]:
                best = (p.L, rA, rB, g, st['dmin'])
    if best is None:
        print('    粗网格内无可行解')
    else:
        print('    粗网格最优: L=%.6f  rA=%.4f rB=%.4f  min phi\'=%.4f'
              % (best[0], best[1], best[2], best[4]))
        # 局部细化
        L_best, rA_b, rB_b, g_b, _ = best
        for step in (0.01, 0.004, 0.001):
            improved = True
            while improved:
                improved = False
                for dA in (-step, 0.0, step):
                    for dB in (-step, 0.0, step):
                        if dA == 0.0 and dB == 0.0:
                            continue
                        g = make(rA_b + dA, rB_b + dB)
                        if g is None:
                            continue
                        p = Path(g)
                        if p.L >= L_best - 1e-9:
                            continue
                        if follow_stats(p, ds=0.02)['dmin'] <= 0.0:
                            continue
                        L_best, rA_b, rB_b, g_b = p.L, rA_b + dA, rB_b + dB, g
                        improved = True
        print('    细化后:   L=%.6f  rA=%.6f rB=%.6f' % (L_best, rA_b, rB_b))
        p_b, st_b = report(g_b, '最优')
        print('    碰撞复核 (ds=0.5):')
        G, sh, pr = min_gap(g_b, ds=0.5)
        print('      最小间隙 G=%+.6f m @ s_head=%.1f, 板凳对 %s' % (G, sh, pr))
        print('\n    vs 基准 %.6f m: 缩短 %.6f m (%.2f%%)'
              % (L0, L0 - L_best, 100.0 * (L0 - L_best) / L0))
        np.save(os.path.join(OUT, 'q4_best.npy'),
                np.array([rA_b, rB_b], float))
        print('    已保存最优参数 -> q4_best.npy')
