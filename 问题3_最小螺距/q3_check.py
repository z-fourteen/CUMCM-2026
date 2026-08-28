# -*- coding: utf-8 -*-
"""自检：递推精度、SAT 正确性、并用同一模型复算问题 2（对标 ~412.47 s）。"""
import time
import numpy as np
from q3_core import (handles, obb, collides, min_gap, arc, corners,
                     CHORDS, R_TURN, N_BENCH)

print('[A] 把手递推精度 (pitch=0.55, r_head=4.5)')
P = handles(0.55, 4.5)
seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
r = np.linalg.norm(P, axis=1)
print(f'  弦长最大误差 = {np.abs(seg - CHORDS).max():.3e} m')
print(f'  半径单调 = {bool(np.all(np.diff(r) > 0))}, r: {r[0]:.4f} -> {r[-1]:.4f} m')
b = 0.55 / (2 * np.pi)
print(f'  总弧长 = {arc(r[-1]/b, b) - arc(r[0]/b, b):.3f} m, 弦长和 = {CHORDS.sum():.3f} m')

print('\n[B] SAT 自检（构造已知算例）')
# 两个平行矩形，中心距 0.29 < 0.30 应相交；0.31 应分离
from q3_core import HALF_W, HL
def two(dy):
    return np.array([[0., 0.], [1.65, 0.], [1.65, dy], [0., dy]])  # 2 节，第2节平移 dy
for dy in (0.29, 0.31):
    Q = np.array([[0., 0.], [1.65, 0.], [1.65+1e-9, dy], [1e-9, dy]])
    C, u, nv, hl, _ = obb(Q)
    from q3_core import _sat_pairs
    hit = _sat_pairs(C, u, nv, hl, np.array([0]), np.array([2]))
    print(f'  平行错开 dy={dy}: 相交={bool(hit[0])} (期望 {dy < 0.30})')

print('\n[C] 用同一模型复算问题 2')
def r_head_of_t(pitch, t, v=1.0, turn0=16):
    bb = pitch / (2 * np.pi)
    s = arc(turn0 * 2 * np.pi, bb) - v * t
    lo, hi = 1e-12, turn0 * 2 * np.pi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if arc(mid, bb) < s:
            lo = mid
        else:
            hi = mid
    return bb * 0.5 * (lo + hi)

t0 = time.time()
lo, hi = 0.0, None
for t in np.arange(300.0, 442.0, 1.0):
    if collides(handles(0.55, r_head_of_t(0.55, float(t))))[0]:
        hi, lo = float(t), float(t) - 1.0
        break
for _ in range(45):
    mid = 0.5 * (lo + hi)
    if collides(handles(0.55, r_head_of_t(0.55, mid)))[0]:
        hi = mid
    else:
        lo = mid
print(f'  问题2 终止时刻 t* = {lo:.4f} s   (公认答案 ~412.47 s)')
g, pr = min_gap(handles(0.55, r_head_of_t(0.55, lo)), slack=0.3)
print(f'  该时刻最小间隙 = {g:.6f} m, 最近对 = {pr}')
print(f'  单次 collides 耗时 ≈ {(time.time()-t0)/70*1000:.1f} ms')
