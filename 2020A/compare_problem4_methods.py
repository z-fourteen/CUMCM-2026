"""Compare differential evolution and simulated annealing for problem 4."""
import os
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl

from constraints import check_feasible_detail
from optimize import optimize_problem4, simulated_annealing_problem4


OUT_DIR = 'problem4_compare_result'
ALPHA = 0.7

mpl.rcParams['font.sans-serif'] = [
    'SimHei', 'Microsoft YaHei', 'SimSun', 'STSong', 'FangSong'
]
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.unicode_minus'] = False


def mkdir(path):
    os.makedirs(path, exist_ok=True)


def result_J(result):
    return result['alpha'] * result['A_norm'] + (1 - result['alpha']) * result['D_norm']


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
        'total_area_above_217': result.get('total_area_above_217'),
        'symmetry_error': result['symmetry_error'],
        'A_norm': result['A_norm'],
        'D_norm': result['D_norm'],
        'J': result_J(result),
        'alpha': result['alpha'],
        'area_scale': result.get('area_scale'),
        'feasible': result.get('feasible', True),
        'violation': result.get('violation', 0.0),
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
        J = result_J(result)
        ax.plot(
            result['t'], result['T'], linewidth=1.0,
            label=f'{name}: J={J:.4f}, D={result["symmetry_error"]:.4f}'
        )
    ax.axhline(217, color='k', linestyle='--', linewidth=0.8)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('temperature (C)')
    ax.set_title('Problem 4 Optimized Reflow Curves')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '炉温曲线对比.png'), dpi=150)
    plt.close(fig)


def save_energy_curve(result):
    history = result.get('history', [])
    fieldnames = [
        'iter', 'temperature', 'objective', 'J', 'best_J',
        'best_objective', 'area', 'total_area_above_217',
        'symmetry_error', 'A_norm', 'D_norm', 'violation', 'feasible'
    ]
    with open(os.path.join(OUT_DIR, '模拟退火_内能曲线.csv'), 'w', newline='',
              encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(history)

    if not history:
        return

    iters = [row['iter'] for row in history]
    objective = [row['objective'] for row in history]
    best_objective = [row['best_objective'] for row in history]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(iters, objective, linewidth=0.8, alpha=0.75, label='current energy')
    ax.plot(iters, best_objective, linewidth=1.2, label='best energy')
    ax.set_xlabel('iteration')
    ax.set_ylabel('energy')
    ax.set_title('Simulated Annealing Energy Curve')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '模拟退火_内能曲线.png'), dpi=150)
    plt.close(fig)


def save_summary(rows, best_sa_seed):
    fieldnames = [
        'method', 'T1', 'T6', 'T7', 'T8', 'v', 'area',
        'total_area_above_217', 'symmetry_error', 'A_norm', 'D_norm',
        'J', 'alpha', 'area_scale', 'feasible', 'violation',
        'v_up_max', 'v_down_max', 't_150_190', 't_above_217',
        'T_peak', 't_peak'
    ]
    with open(os.path.join(OUT_DIR, 'summary.csv'), 'w', newline='',
              encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(os.path.join(OUT_DIR, '结果对比.txt'), 'w', encoding='utf-8') as f:
        f.write('问题4：差分进化与模拟退火对比\n')
        f.write('变量范围：各温区在实验设定基础上 +/-10 C，速度65~100 cm/min\n')
        f.write('面积项复用问题3：S3 = integral from first 217 C crossing to peak of (T-217) dt\n')
        f.write('对称项：D_sym = normalized mirror area error around peak time above 217 C\n')
        f.write(f'综合目标: J = alpha*(S3/{rows[0]["area_scale"]:.1f}) + (1-alpha)*D_sym, alpha={ALPHA}\n\n')
        for row in rows:
            f.write(f"{row['method']}\n")
            f.write(f"  T1={row['T1']:.4f} C, T6={row['T6']:.4f} C, "
                    f"T7={row['T7']:.4f} C, T8={row['T8']:.4f} C\n")
            f.write(f"  v={row['v']:.4f} cm/min, feasible={row['feasible']}, "
                    f"violation={row['violation']:.6f}\n")
            f.write(f"  S3={row['area']:.4f}, total_above_217={row['total_area_above_217']:.4f}, "
                    f"D_sym={row['symmetry_error']:.6f}\n")
            f.write(f"  S3_norm={row['A_norm']:.6f}, D_norm={row['D_norm']:.6f}, "
                    f"J={row['J']:.6f}\n")
            f.write(f"  v_up={row['v_up_max']:.4f} C/s, "
                    f"v_down={row['v_down_max']:.4f} C/s\n")
            f.write(f"  t_150_190={row['t_150_190']:.2f} s, "
                    f"t_above_217={row['t_above_217']:.2f} s\n")
            f.write(f"  T_peak={row['T_peak']:.2f} C @ {row['t_peak']:.2f} s\n\n")
        f.write(f"模拟退火最优种子: {best_sa_seed}\n")


def print_detail(name, result):
    print(f'\n{name}')
    print(f"  T1={result['T1']:.4f}, T6={result['T6']:.4f}, "
          f"T7={result['T7']:.4f}, T8={result['T8']:.4f}, "
          f"v={result['v']:.4f}")
    print(f"  S3={result['area']:.4f}, total_above_217={result['total_area_above_217']:.4f}, "
          f"D_sym={result['symmetry_error']:.6f}, J={result_J(result):.6f}")
    detail = check_feasible_detail(result['indicators'])
    for item, (ok, value) in detail.items():
        print(f"  {'OK' if ok else 'NO'} {item}: {value}")


def main():
    mkdir(OUT_DIR)

    print('Running differential evolution baseline...')
    de_result = optimize_problem4(alpha=ALPHA)
    print_detail('Differential Evolution', de_result)

    print('\nRunning simulated annealing seeds...')
    sa_results = []
    for seed in [1, 7, 21, 42, 99]:
        result = simulated_annealing_problem4(
            alpha=ALPHA,
            seed=seed,
            max_iter=1800,
            initial_temp=3.0,
            cooling_rate=0.995,
            dt_search=2.0,
            dt_final=0.5,
        )
        sa_results.append((seed, result))
        print_detail(f'Simulated Annealing seed={seed}', result)

    feasible_sa = [(seed, r) for seed, r in sa_results if r.get('feasible')]
    if feasible_sa:
        best_seed, best_sa = min(feasible_sa, key=lambda item: item[1]['J'])
    else:
        best_seed, best_sa = min(sa_results, key=lambda item: item[1]['violation'])

    rows = [
        summarize_result('Differential Evolution', de_result),
        summarize_result(f'Simulated Annealing best(seed={best_seed})', best_sa),
    ]
    save_summary(rows, best_seed)
    save_curve(de_result, '差分进化_炉温曲线.csv')
    save_curve(best_sa, '模拟退火_炉温曲线.csv')
    save_energy_curve(best_sa)
    plot_curves([
        ('Differential Evolution', de_result),
        (f'Simulated Annealing seed={best_seed}', best_sa),
    ])

    print(f'\nSaved comparison to {OUT_DIR}/')


if __name__ == '__main__':
    main()
