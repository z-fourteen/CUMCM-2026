# -*- coding: utf-8 -*-
"""跑三种判据，对比最小螺距与临界构型，判定猜想是否成立。"""
import numpy as np
from q3_core import handles, R_TURN
from q3_variant import (crit_full, crit_head_outer, crit_head_all,
                        worst_gap, solve_pstar, outer_turn_j, bench_theta)

P_REF = 0.450337          # 基准（全量判据）已知结果

print('=' * 76, flush=True)
print('猜想验证：碰撞判据能否简化为「仅龙头 vs 外层一圈」', flush=True)
print('=' * 76, flush=True)

print('\n[0] 外层一圈窗口自检 (p=0.45, r_head=4.57 临界位置)', flush=True)
P = handles(0.45, 4.57)
J = outer_turn_j(P, 0.45)
d = bench_theta(P, 0.45); d = d - d[0]
print(f'    龙头外一圈候选节 (1-based) = {(J+1)[:12]} ... 共 {len(J)} 节', flush=True)
print(f'    其角度超出量/2pi = {np.round(d[J][:12]/(2*np.pi), 3)}', flush=True)

print('\n[1] 三种判据下的 G(p) 对比 (dr=0.01)', flush=True)
print(f'{"p(m)":>8} | {"全量 G":>10} {"最紧r":>7} {"对":>9} | '
      f'{"龙头-外圈 G":>11} {"最紧r":>7} {"对":>9} | {"龙头-全部 G":>11} {"对":>9}', flush=True)
for p in [0.43, 0.44, 0.45, 0.452, 0.455, 0.46, 0.48, 0.55]:
    a = worst_gap(p, crit_full, 0.01, refine=False)
    b = worst_gap(p, crit_head_outer, 0.01, refine=False)
    c = worst_gap(p, crit_head_all, 0.01, refine=False)
    fa = '  -  ' if a[1] is None else f'{a[1]:.3f}'
    fb = '  -  ' if b[1] is None else f'{b[1]:.3f}'
    print(f'{p:8.3f} | {a[0]:10.6f} {fa:>7} {str(a[2]):>9} | '
          f'{b[0]:11.6f} {fb:>7} {str(b[2]):>9} | {c[0]:11.6f} {str(c[2]):>9}', flush=True)

print('\n[2] 各判据独立二分求 p*', flush=True)
res = {}
for name, crit in [('全量 (基准)', crit_full),
                   ('龙头-外层一圈', crit_head_outer),
                   ('龙头-全部', crit_head_all)]:
    p = solve_pstar(crit)
    res[name] = p
    g, rh, pr = worst_gap(p, crit, 0.002)
    rs = '  -  ' if rh is None else f'{rh:.4f}'
    print(f'    {name:<14} p* = {p:.6f} m ({p*100:.4f} cm)  '
          f'临界 r={rs} 对={pr}', flush=True)

print('\n[3] 与基准比较', flush=True)
base = res['全量 (基准)']
for name, p in res.items():
    d_mm = (p - base) * 1000
    print(f'    {name:<14} p* = {p:.6f}  与基准差 = {d_mm:+.4f} mm  '
          f'相对 {abs(d_mm)/(base*1000)*100:.4f}%', flush=True)

print('\n[4] 关键检验：简化判据的解，在全量判据下是否真的无碰撞？', flush=True)
for name in ('龙头-外层一圈', '龙头-全部'):
    p = res[name]
    g, rh, pr = worst_gap(p, crit_full, 0.002)
    rs = '  -  ' if rh is None else f'{rh:.4f}'
    verdict = '通过（仍无碰撞）' if g > 0 else f'不通过：全量判据下碰撞 @r={rs} 对{pr}'
    print(f'    {name:<14} p*={p:.6f} -> 全量 G = {g:+.6f} m  {verdict}', flush=True)

print('\n[5] 基准 p* 处：全量最紧对 vs 简化判据能否捕获', flush=True)
for rh in [4.51, 4.55, 4.5726, 4.60, 4.65, 4.70, 5.00, 5.42]:
    Pp = handles(P_REF, rh)
    ga, pa = crit_full(Pp, P_REF)
    gb, pb = crit_head_outer(Pp, P_REF)
    same = '同' if pa == pb else '异'
    print(f'    r={rh:6.4f}  全量 {ga:+.6f} {str(pa):>9} | '
          f'龙头-外圈 {gb:+.6f} {str(pb):>9}  [{same}]', flush=True)
