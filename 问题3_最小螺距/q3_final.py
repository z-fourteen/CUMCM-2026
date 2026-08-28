# -*- coding: utf-8 -*-
"""问题 3 最终求解：最小螺距 p*（精确矩形模型，判据 = 盘入全程无碰撞）。

G(p) = min_{r_head in [4.5, R_HI]} signed_gap(p, r_head)
  signed_gap > 0 : 分离，值为精确最短距离
  signed_gap <= 0: 相交，值为 -穿透深度
G(p) 连续、随 p 单调递增，二分求 G(p*) = 0。

r_head 扫描窗口取 [4.5, R_HI]：碰撞最紧处必在最内圈（脚本末尾用全程扫描验证）。
"""
import numpy as np
from q3_core import handles, signed_gap, min_gap, arc, R_TURN, CHORDS

R_HI = 6.0          # 扫描上界（>= 3 个螺距，覆盖最内圈全部临界构型）


def worst_gap(pitch, dr=0.005, r_hi=R_HI, refine=True):
    """盘入 [4.5, r_hi] 段的最小带符号间隙 -> (G, 最紧 r_head, 板凳对)。"""
    best = (np.inf, None, None)
    for rh in np.arange(r_hi, R_TURN - 1e-12, -dr):
        g, pr = signed_gap(handles(pitch, float(rh)), slack=0.4)
        if g < best[0]:
            best = (g, float(rh), pr)
        if g <= 0.0:                       # 已碰撞，G<0 已确定
            return best
    if refine and best[1] is not None:     # 最紧点附近细化一档
        a, b = max(R_TURN, best[1] - dr), min(r_hi, best[1] + dr)
        for rh in np.linspace(a, b, 25):
            g, pr = signed_gap(handles(pitch, float(rh)), slack=0.4)
            if g < best[0]:
                best = (g, float(rh), pr)
    return best


if __name__ == '__main__':
    print('=' * 74, flush=True)
    print('问题 3  最小螺距（精确矩形碰撞模型，全程无碰撞判据）', flush=True)
    print('=' * 74, flush=True)

    print('\n[1] G(p) 粗表 (dr=0.01)', flush=True)
    print(f'{"pitch(m)":>9} {"G(p) (m)":>12} {"最紧r_head":>11}  {"临界对":>12}', flush=True)
    for p in [0.43, 0.44, 0.45, 0.452, 0.455, 0.46, 0.48, 0.55]:
        g, rh, pr = worst_gap(p, dr=0.01, refine=False)
        rs = '  -  ' if rh is None else f'{rh:.3f}'
        print(f'{p:9.3f} {g:12.6f} {rs:>11}  {str(pr):>12}', flush=True)

    print('\n[2] 二分 G(p)=0 (dr=0.005 + 局部细化)', flush=True)
    lo, hi = 0.43, 0.48
    assert worst_gap(lo, 0.01, refine=False)[0] < 0, 'lo 应碰撞'
    assert worst_gap(hi, 0.01, refine=False)[0] > 0, 'hi 应可行'
    for it in range(26):
        mid = 0.5 * (lo + hi)
        if worst_gap(mid, 0.005)[0] > 0:
            hi = mid
        else:
            lo = mid
    p_star = hi
    print(f'    p* = {p_star:.8f} m  (区间宽 {hi-lo:.2e})', flush=True)

    print('\n[3] 复核：更细 dr 下 p* 两侧符号', flush=True)
    for p in [p_star - 0.004, p_star - 0.001, p_star, p_star + 0.001, p_star + 0.004]:
        g, rh, pr = worst_gap(p, dr=0.001)
        rs = '  -  ' if rh is None else f'{rh:.4f}'
        print(f'    p={p:.6f}  G={g: .6f}  @r={rs}  对{pr}', flush=True)

    print('\n[4] 验证扫描窗口：p* 处 r_head 由 12 m 一路扫到 4.5 m', flush=True)
    print(f'{"r_head(m)":>10} {"min gap(m)":>12}  {"最近对":>12}', flush=True)
    gw = (np.inf, None, None)
    for rh in np.arange(12.0, R_TURN - 1e-12, -0.01):
        g, pr = signed_gap(handles(p_star, float(rh)), slack=0.45)
        if g < gw[0]:
            gw = (g, float(rh), pr)
        if abs(round(rh, 2) % 1.0) < 1e-9:
            print(f'{rh:10.2f} {g:12.6f}  {str(pr):>12}', flush=True)
    print(f'\n    全程最紧 G = {gw[0]:.6f} m @ r_head = {gw[1]:.3f} m, 对 {gw[2]}', flush=True)
    print(f'    最紧点是否落在窗口 [4.5, {R_HI}] 内: {gw[1] <= R_HI}', flush=True)

    print('\n[5] 最终结果', flush=True)
    print(f'    最小螺距 p* = {p_star:.6f} m ≈ {p_star*100:.4f} cm', flush=True)
    gm, prm = min_gap(handles(p_star, R_TURN), slack=0.45)
    print(f'    到达边界 r=4.5 m 时最小间隙 = {gm:.6f} m, 最近对 = 第{prm[0]}/第{prm[1]}节', flush=True)

    P = handles(p_star, R_TURN)
    b = p_star / (2 * np.pi)
    r_tail = float(np.linalg.norm(P[-1]))
    print(f'    龙头前把手 ({P[0,0]:.6f}, {P[0,1]:.6f}), r = {np.hypot(*P[0]):.6f} m', flush=True)
    print(f'    龙头位于第 {R_TURN/p_star:.4f} 圈；龙尾后把手 r = {r_tail:.4f} m', flush=True)
    print(f'    队伍跨 {(r_tail-R_TURN)/p_star:.2f} 圈', flush=True)
    print(f'    弧长校验 {arc(r_tail/b, b)-arc(R_TURN/b, b):.3f} m (弦长和 {CHORDS.sum():.3f} m)', flush=True)
