"""问题3：差分进化与模拟退火结果对比。

两种算法均使用 optimize.PROBLEM3_BOUNDS，即原题 ±10°C 温区调节范围。
"""
import os
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl

from constraints import check_feasible_detail
from optimize import optimize_problem3, simulated_annealing_problem3


OUT_DIR = 'problem3_compare_result'
SA_SEEDS = [1, 7, 21, 42, 99]
SA_MAX_ITER = 5000
SA_INITIAL_TEMP = 500.0
SA_COOLING_RATE = 0.998
SA_FINAL_TEMP = 1e-3
SA_BASE_STEP_FRAC = 0.20
SA_MIN_STEP_FRAC = 0.03

mpl.rcParams['font.sans-serif'] = [
    'SimHei', 'Microsoft YaHei', 'SimSun', 'STSong', 'FangSong'
]
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.unicode_minus'] = False


def mkdir(path):
    os.makedirs(path, exist_ok=True)


def summarize_result(name, result):
    ind = result['indicators']
    return {
        'method': name,
        'T1': result['T1'],
        'T6': result['T6'],
        'T7': result['T7'],
        'T8': result['T8'],
        'v': result['v'],
        'area': result['area'],
        'feasible': result.get('feasible', True),
        'v_up_max': ind['v_up_max'],
        'v_down_max': ind['v_down_max'],
        't_150_190': ind['t_150_190'],
        't_above_217': ind['t_above_217'],
        'T_peak': ind['T_peak'],
        't_peak': ind['t_peak'],
    }


def save_curve(result, filename):
    path = os.path.join(OUT_DIR, filename)
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'temperature'])
        writer.writerows(zip(result['t'], result['T']))


def plot_curves(results):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, result in results:
        ax.plot(
            result['t'], result['T'], linewidth=1.0,
            label=f'{name}: A={result["area"]:.2f}'
        )
    ax.axhline(217, color='k', linestyle='--', linewidth=0.8)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('温度 (°C)')
    ax.set_title('问题3两种优化算法炉温曲线对比')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '炉温曲线对比.png'), dpi=150)
    plt.close(fig)


def save_history(result, filename):
    path = os.path.join(OUT_DIR, filename)
    if not result.get('history'):
        return
    fieldnames = [
        'iter', 'temperature', 'objective', 'area', 'violation',
        'feasible', 'best_area'
    ]
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result['history'])


def plot_energy_curve(result):
    history = result.get('history', [])
    if not history:
        return

    iters = [h['iter'] for h in history]
    objective = [h['objective'] for h in history]
    best_area = [h['best_area'] for h in history]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(iters, objective, linewidth=0.6, alpha=0.55, label='当前内能')
    ax.plot(iters, best_area, linewidth=1.2, label='历史最优阴影面积')
    ax.set_xlabel('迭代次数')
    ax.set_ylabel('内能 / 面积 (°C·s)')
    ax.set_title('问题3模拟退火内能曲线')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '模拟退火内能曲线.png'), dpi=150)
    plt.close(fig)


def save_summary(rows, best_sa_seed):
    fieldnames = [
        'method', 'T1', 'T6', 'T7', 'T8', 'v', 'area', 'feasible',
        'v_up_max', 'v_down_max', 't_150_190', 't_above_217',
        'T_peak', 't_peak'
    ]
    with open(os.path.join(OUT_DIR, 'summary.csv'), 'w', newline='',
              encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(os.path.join(OUT_DIR, '结果对比.txt'), 'w', encoding='utf-8') as f:
        f.write('问题3：差分进化与模拟退火对比\n')
        f.write('变量范围来自原文：各温区在实验设定基础上 ±10°C，速度65~100 cm/min\n\n')
        f.write('面积口径：首次超过217°C到峰值温度之间的题图阴影面积\n')
        f.write('模拟退火参数:\n')
        f.write(f'  seeds={SA_SEEDS}\n')
        f.write(f'  max_iter={SA_MAX_ITER}, initial_temp={SA_INITIAL_TEMP}, '
                f'cooling_rate={SA_COOLING_RATE}, final_temp={SA_FINAL_TEMP}\n')
        f.write(f'  base_step_frac={SA_BASE_STEP_FRAC}, '
                f'min_step_frac={SA_MIN_STEP_FRAC}, '
                f'dt_search=2.0s, dt_final=0.5s\n\n')
        for row in rows:
            f.write(f"{row['method']}\n")
            f.write(f"  T1={row['T1']:.4f} °C, T6={row['T6']:.4f} °C, "
                    f"T7={row['T7']:.4f} °C, T8={row['T8']:.4f} °C\n")
            f.write(f"  v={row['v']:.4f} cm/min, area={row['area']:.4f}, "
                    f"feasible={row['feasible']}\n")
            f.write(f"  v_up={row['v_up_max']:.4f} °C/s, "
                    f"v_down={row['v_down_max']:.4f} °C/s\n")
            f.write(f"  t_150_190={row['t_150_190']:.2f} s, "
                    f"t_above_217={row['t_above_217']:.2f} s\n")
            f.write(f"  T_peak={row['T_peak']:.2f} °C @ {row['t_peak']:.2f} s\n\n")
        f.write(f"模拟退火最优种子: {best_sa_seed}\n")


def print_detail(name, result):
    print(f'\n{name}')
    print(f"  T1={result['T1']:.4f}, T6={result['T6']:.4f}, "
          f"T7={result['T7']:.4f}, T8={result['T8']:.4f}, "
          f"v={result['v']:.4f}")
    print(f"  area={result['area']:.4f}")
    detail = check_feasible_detail(result['indicators'])
    for item, (ok, value) in detail.items():
        print(f"  {'OK' if ok else 'NO'} {item}: {value}")


def main():
    mkdir(OUT_DIR)

    print('Running differential evolution baseline...')
    de_result = optimize_problem3()
    print_detail('Differential Evolution', de_result)

    print('\nRunning simulated annealing seeds...')
    sa_results = []
    for seed in SA_SEEDS:
        result = simulated_annealing_problem3(
            seed=seed,
            max_iter=SA_MAX_ITER,
            initial_temp=SA_INITIAL_TEMP,
            final_temp=SA_FINAL_TEMP,
            cooling_rate=SA_COOLING_RATE,
            dt_search=2.0,
            dt_final=0.5,
            base_step_frac=SA_BASE_STEP_FRAC,
            min_step_frac=SA_MIN_STEP_FRAC,
        )
        sa_results.append((seed, result))
        print_detail(f'Simulated Annealing seed={seed}', result)

    feasible_sa = [(seed, r) for seed, r in sa_results if r.get('feasible')]
    if feasible_sa:
        best_seed, best_sa = min(feasible_sa, key=lambda item: item[1]['area'])
    else:
        best_seed, best_sa = min(sa_results, key=lambda item: item[1]['violation'])

    rows = [
        summarize_result('Differential Evolution', de_result),
        summarize_result(f'Simulated Annealing best(seed={best_seed})', best_sa),
    ]
    save_summary(rows, best_seed)
    save_curve(de_result, '差分进化_炉温曲线.csv')
    save_curve(best_sa, '模拟退火_炉温曲线.csv')
    save_history(best_sa, '模拟退火内能历史.csv')
    plot_curves([
        ('Differential Evolution', de_result),
        (f'Simulated Annealing seed={best_seed}', best_sa),
    ])
    plot_energy_curve(best_sa)

    print(f'\nSaved comparison to {OUT_DIR}/')


if __name__ == '__main__':
    main()
