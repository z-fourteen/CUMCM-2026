"""主脚本：运行问题1~4的完整流程"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.unicode_minus'] = False

from simulator import (simulate, zone_center, zone_end,
                        position_to_time, build_Tf)
from constraints import compute_indicators, check_feasible, check_feasible_detail
from optimize import max_feasible_speed, optimize_problem3, optimize_problem4


def mkdir(path):
    os.makedirs(path, exist_ok=True)


# ============================================================
# 公共绘图
# ============================================================
def plot_temperature_curve(t, T, title, save_path, ylabel='温度 (°C)'):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, T, 'b-', linewidth=0.8)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ============================================================
# 问题1
# ============================================================
def solve_problem1():
    print("=" * 60)
    print("  问题1：炉温曲线仿真")
    print("=" * 60)

    # 温区设定
    T1, T6, T7, T8 = 173, 198, 230, 257
    speed = 78.0  # cm/min

    # 仿真
    t, T_pcb = simulate(T1, T6, T7, T8, speed)

    # 保存结果目录
    out_dir = 'problem1_result'
    mkdir(out_dir)

    # 1. 绘制炉温曲线
    plot_temperature_curve(t, T_pcb, '焊接区域中心炉温曲线 (78 cm/min)',
                           os.path.join(out_dir, '炉温曲线.png'))

    # 2. 输出指定位置温度
    positions = {
        '小温区3中点': zone_center(3),
        '小温区6中点': zone_center(6),
        '小温区7中点': zone_center(7),
        '小温区8结束处': zone_end(8),
    }

    print(f"\n  指定位置温度 (v={speed} cm/min):")
    print(f"  {'位置':<16} {'x (cm)':<10} {'t (s)':<10} {'温度 (°C)':<12}")
    print(f"  {'-'*48}")

    results = []
    for name, x in positions.items():
        t_pos = position_to_time(x, speed)
        # 插值获取该时刻温度
        T_at_pos = np.interp(t_pos, t, T_pcb)
        print(f"  {name:<16} {x:<10.2f} {t_pos:<10.2f} {T_at_pos:<12.2f}")
        results.append({'位置': name, 'x_cm': x, 't_s': t_pos, '温度_°C': T_at_pos})

    # 保存位置温度到CSV
    df_pos = pd.DataFrame(results)
    df_pos.to_csv(os.path.join(out_dir, '指定位置温度.csv'), index=False,
                  encoding='utf-8-sig')

    # 3. 保存每隔0.5s的温度
    df_full = pd.DataFrame({'time': t, 'temperature': T_pcb})
    df_full.to_csv(os.path.join(out_dir, 'result.csv'), index=False,
                   encoding='utf-8-sig')

    # 输出指标
    ind = compute_indicators(t, T_pcb)
    print(f"\n  工艺指标:")
    print(f"    最大升温斜率: {ind['v_up_max']:.4f} °C/s")
    print(f"    最大降温斜率: {ind['v_down_max']:.4f} °C/s")
    print(f"    150~190°C持续时间: {ind['t_150_190']:.2f} s")
    print(f"    >217°C持续时间: {ind['t_above_217']:.2f} s")
    print(f"    峰值温度: {ind['T_peak']:.2f} °C")

    print(f"\n  结果已保存至 {out_dir}/")
    return t, T_pcb


# ============================================================
# 问题2
# ============================================================
def solve_problem2():
    print("\n" + "=" * 60)
    print("  问题2：最大过炉速度搜索")
    print("=" * 60)

    # 温区设定
    zone_temps = (182, 203, 237, 254)

    # 二分搜索
    v_opt, result = max_feasible_speed(zone_temps)

    out_dir = 'problem2_result'
    mkdir(out_dir)

    if v_opt is None:
        print(f"\n  {result['msg']}")
        print("  原因分析: 峰值温度与150~190°C时间约束矛盾")
        print("  - 低速(v<70): 峰值OK但150~190°C时间过长(>120s)")
        print("  - 高速(v>85): 150~190°C时间OK但峰值过低(<240°C)")
        print("  → 需通过问题3优化温区设定来解决")
        # 保存诊断信息
        with open(os.path.join(out_dir, '结果.txt'), 'w', encoding='utf-8') as f:
            f.write(f"{result['msg']}\n")
            f.write("原因分析:\n")
            f.write("  给定温区设定(182/203/237/254°C)下，65~100cm/min范围内无可行解:\n")
            f.write("  - 低速时150~190°C持续时间超出120s上限\n")
            f.write("  - 高速时峰值温度低于240°C下限\n")
            f.write("  需通过问题3优化温区温度设定\n")
        print(f"\n  诊断信息已保存至 {out_dir}/")
        return

    print(f"\n  最大允许速度: {v_opt:.2f} cm/min")
    print(f"\n  对应工艺指标:")

    ind = result['indicators']
    detail = check_feasible_detail(ind)
    for name, (ok, val) in detail.items():
        status = '✓' if ok else '✗'
        print(f"    {status} {name}: {val}")

    # 保存曲线图
    plot_temperature_curve(
        result['t'], result['T'],
        f'最大速度炉温曲线 (v={v_opt:.2f} cm/min)',
        os.path.join(out_dir, '最优速度炉温曲线.png')
    )

    # 保存数值结果
    with open(os.path.join(out_dir, '结果.txt'), 'w', encoding='utf-8') as f:
        f.write(f"最大允许速度: {v_opt:.2f} cm/min\n\n")
        f.write("工艺指标:\n")
        for name, (ok, val) in detail.items():
            f.write(f"  {'✓' if ok else '✗'} {name}: {val}\n")

    df = pd.DataFrame({'time': result['t'], 'temperature': result['T']})
    df.to_csv(os.path.join(out_dir, '炉温曲线.csv'), index=False,
              encoding='utf-8-sig')

    print(f"\n  结果已保存至 {out_dir}/")
    return v_opt, result


# ============================================================
# 问题3
# ============================================================
def solve_problem3():
    print("\n" + "=" * 60)
    print("  问题3：温区设定与速度优化（最小化>217°C面积）")
    print("=" * 60)

    result = optimize_problem3()

    out_dir = 'problem3_result'
    mkdir(out_dir)

    print(f"\n  优化结果:")
    print(f"    T1 (温区1~5) = {result['T1']:.2f} °C")
    print(f"    T6 (温区6)   = {result['T6']:.2f} °C")
    print(f"    T7 (温区7)   = {result['T7']:.2f} °C")
    print(f"    T8 (温区8~9) = {result['T8']:.2f} °C")
    print(f"    v (速度)     = {result['v']:.2f} cm/min")
    print(f"    超过217°C面积 = {result['area']:.4f}")

    ind = result['indicators']
    print(f"\n  工艺指标:")
    detail = check_feasible_detail(ind)
    for name, (ok, val) in detail.items():
        status = '✓' if ok else '✗'
        print(f"    {status} {name}: {val}")

    # 绘图
    plot_temperature_curve(
        result['t'], result['T'],
        f'优化炉温曲线 (面积={result["area"]:.2f})',
        os.path.join(out_dir, '优化炉温曲线.png')
    )

    # 保存结果
    with open(os.path.join(out_dir, '结果.txt'), 'w', encoding='utf-8') as f:
        f.write("问题3优化结果\n")
        f.write(f"T1 (温区1~5) = {result['T1']:.4f} °C\n")
        f.write(f"T6 (温区6)   = {result['T6']:.4f} °C\n")
        f.write(f"T7 (温区7)   = {result['T7']:.4f} °C\n")
        f.write(f"T8 (温区8~9) = {result['T8']:.4f} °C\n")
        f.write(f"v  (速度)    = {result['v']:.4f} cm/min\n")
        f.write(f">217°C面积   = {result['area']:.4f}\n\n")
        f.write("工艺指标:\n")
        for name, (ok, val) in detail.items():
            f.write(f"  {'✓' if ok else '✗'} {name}: {val}\n")

    df = pd.DataFrame({'time': result['t'], 'temperature': result['T']})
    df.to_csv(os.path.join(out_dir, '炉温曲线.csv'), index=False,
              encoding='utf-8-sig')

    print(f"\n  结果已保存至 {out_dir}/")
    return result


# ============================================================
# 问题4
# ============================================================
def solve_problem4():
    print("\n" + "=" * 60)
    print("  问题4：对称性优化")
    print("=" * 60)

    result = optimize_problem4(alpha=0.5)

    out_dir = 'problem4_result'
    mkdir(out_dir)

    print(f"\n  优化结果 (α={result['alpha']}):")
    print(f"    T1 (温区1~5) = {result['T1']:.2f} °C")
    print(f"    T6 (温区6)   = {result['T6']:.2f} °C")
    print(f"    T7 (温区7)   = {result['T7']:.2f} °C")
    print(f"    T8 (温区8~9) = {result['T8']:.2f} °C")
    print(f"    v (速度)     = {result['v']:.2f} cm/min")
    print(f"    超过217°C面积 = {result['area']:.4f}")
    print(f"    对称误差      = {result['symmetry_error']:.4f}")
    print(f"    J (目标)     = {result['fun']:.6f}")

    ind = result['indicators']
    print(f"\n  工艺指标:")
    detail = check_feasible_detail(ind)
    for name, (ok, val) in detail.items():
        status = '✓' if ok else '✗'
        print(f"    {status} {name}: {val}")

    # 绘图
    plot_temperature_curve(
        result['t'], result['T'],
        f'对称优化炉温曲线 (面积={result["area"]:.2f}, D={result["symmetry_error"]:.2f})',
        os.path.join(out_dir, '优化炉温曲线.png')
    )

    # 保存结果
    with open(os.path.join(out_dir, '结果.txt'), 'w', encoding='utf-8') as f:
        f.write("问题4对称性优化结果\n")
        f.write(f"alpha  = {result['alpha']}\n")
        f.write(f"T1 (温区1~5) = {result['T1']:.4f} °C\n")
        f.write(f"T6 (温区6)   = {result['T6']:.4f} °C\n")
        f.write(f"T7 (温区7)   = {result['T7']:.4f} °C\n")
        f.write(f"T8 (温区8~9) = {result['T8']:.4f} °C\n")
        f.write(f"v  (速度)    = {result['v']:.4f} cm/min\n")
        f.write(f">217°C面积   = {result['area']:.4f}\n")
        f.write(f"对称误差     = {result['symmetry_error']:.4f}\n")
        f.write(f"A_norm       = {result['A_norm']:.6f}\n")
        f.write(f"D_norm       = {result['D_norm']:.6f}\n")
        f.write(f"J            = {result['fun']:.6f}\n\n")
        f.write("工艺指标:\n")
        for name, (ok, val) in detail.items():
            f.write(f"  {'✓' if ok else '✗'} {name}: {val}\n")

    df = pd.DataFrame({'time': result['t'], 'temperature': result['T']})
    df.to_csv(os.path.join(out_dir, '炉温曲线.csv'), index=False,
              encoding='utf-8-sig')

    print(f"\n  结果已保存至 {out_dir}/")
    return result


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 60)
    print("  2020年数学建模A题《炉温曲线》- 完整求解")
    print("=" * 60)

    # 问题1
    solve_problem1()

    # 问题2
    solve_problem2()

    # 问题3
    solve_problem3()

    # 问题4
    solve_problem4()

    print("\n" + "=" * 60)
    print("  所有问题求解完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
