"""优化函数：问题2、3、4"""
import numpy as np
from scipy.optimize import differential_evolution

from simulator import simulate
from constraints import (compute_indicators, check_feasible,
                          area_above_217, area_217_to_peak,
                          symmetry_error)


# 由原题“在上述实验设定温度的基础上，各小温区设定温度可以进行±10°C范围内的调整”得到
PROBLEM3_BOUNDS = [
    (165, 185),   # T1: 小温区1~5, 175±10
    (185, 205),   # T6: 小温区6, 195±10
    (225, 245),   # T7: 小温区7, 235±10
    (245, 265),   # T8: 小温区8~9, 255±10
    (65, 100),    # v: 传送带速度
]

PROBLEM4_AREA_SCALE = 800.0


# ==================== 问题2：最大速度二分搜索 ====================

def max_feasible_speed(zone_temps, v_min=65, v_max=100, tol=0.1):
    """二分搜索满足制程约束的最大传送带速度

    先全范围扫描是否有可行解。若无可行解则返回None（附诊断）。

    参数
    ----
    zone_temps : (T1, T6, T7, T8) — 温区温度
    v_min, v_max : float — 速度搜索范围 (cm/min)
    tol : float — 收敛容差

    返回
    ----
    v_opt : float or None — 最大可行速度，无解则None
    result : dict — 详细结果
    """
    T1, T6, T7, T8 = zone_temps

    # 全范围扫描找可行区间
    feasible_list = []
    for v_test in np.arange(v_min, v_max + 1, 1):
        t_sim, T_sim = simulate(T1, T6, T7, T8, v_test)
        ind = compute_indicators(t_sim, T_sim)
        if check_feasible(ind):
            feasible_list.append(v_test)

    if not feasible_list:
        return None, {
            'feasible': False,
            'msg': f'在 [{v_min}, {v_max}] cm/min 范围内无可行解',
            'zone_temps': zone_temps
        }

    # 在可行区间内做二分搜索找最大速度
    lo, hi = feasible_list[0], feasible_list[-1]
    if hi >= v_max:
        hi = v_max

    best_v = lo
    best_result = None

    while hi - lo > tol:
        mid = (lo + hi) / 2
        t_sim, T_sim = simulate(T1, T6, T7, T8, mid)
        ind = compute_indicators(t_sim, T_sim)

        if check_feasible(ind):
            best_v = mid
            best_result = {
                'v': mid, 'indicators': ind,
                't': t_sim, 'T': T_sim,
                'feasible': True
            }
            lo = mid
        else:
            hi = mid

    return best_v, best_result


# ==================== 问题3：最小化超过217°C面积 ====================

def _objective_problem3(x, speed_fixed=None, dt=2.0):
    """问题3目标函数：首次超过217°C到峰值之间的阴影面积

    x = [T1, T6, T7, T8, v]
    dt : 仿真时间步长（优化时用粗步长加速）
    """
    if speed_fixed is not None:
        T1, T6, T7, T8 = x
        v = speed_fixed
    else:
        T1, T6, T7, T8, v = x

    t, T = simulate(T1, T6, T7, T8, v, dt=dt, fast=True)
    ind = compute_indicators(t, T)

    # 如果不满足制程约束，返回大惩罚值
    if not check_feasible(ind):
        return 1e8

    return area_217_to_peak(t, T)


def _clip_to_bounds(x, bounds):
    """将候选解裁剪回变量边界内。"""
    clipped = np.array(x, dtype=float).copy()
    for i, (lo, hi) in enumerate(bounds):
        clipped[i] = np.clip(clipped[i], lo, hi)
    return clipped


def _constraint_violation_problem3(ind):
    """计算制程约束违反程度，0表示完全可行。"""
    violations = [
        max(0.0, ind['v_up_max'] - 3.0) / 3.0,
        max(0.0, -3.0 - ind['v_down_max']) / 3.0,
        max(0.0, 60.0 - ind['t_150_190']) / 60.0,
        max(0.0, ind['t_150_190'] - 120.0) / 60.0,
        max(0.0, 40.0 - ind['t_above_217']) / 40.0,
        max(0.0, ind['t_above_217'] - 90.0) / 40.0,
        max(0.0, 240.0 - ind['T_peak']) / 10.0,
        max(0.0, ind['T_peak'] - 250.0) / 10.0,
    ]
    return float(np.sum(np.square(violations)))


def _evaluate_problem3_candidate(x, dt=2.0, penalty_weight=1e5):
    """评价第三问候选解，返回惩罚目标和完整仿真指标。"""
    T1, T6, T7, T8, v = x
    t, T = simulate(T1, T6, T7, T8, v, dt=dt, fast=True)
    ind = compute_indicators(t, T)
    area = area_217_to_peak(t, T)
    feasible = check_feasible(ind)
    violation = _constraint_violation_problem3(ind)
    objective = area + penalty_weight * violation
    return {
        'objective': objective,
        'area': area,
        'feasible': feasible,
        'violation': violation,
        'indicators': ind,
        't': t,
        'T': T,
    }


def simulated_annealing_problem3(
        bounds=None, seed=42, max_iter=2000, initial_temp=120.0,
        final_temp=1e-3, cooling_rate=0.995, dt_search=2.0,
        dt_final=0.5, penalty_weight=1e5, base_step_frac=0.18,
        min_step_frac=0.05):
    """用模拟退火求解问题3。

    保留旧的 differential_evolution 求解函数，便于后续对比。
    退火搜索阶段使用较粗仿真步长，最终结果用0.5s重新计算。
    """
    if bounds is None:
        bounds = PROBLEM3_BOUNDS

    rng = np.random.default_rng(seed)
    bounds_arr = np.array(bounds, dtype=float)
    lows = bounds_arr[:, 0]
    highs = bounds_arr[:, 1]
    spans = highs - lows

    current_x = lows + rng.random(len(bounds)) * spans
    current = _evaluate_problem3_candidate(
        current_x, dt=dt_search, penalty_weight=penalty_weight
    )

    best_penalty_x = current_x.copy()
    best_penalty = current.copy()
    best_feasible_x = current_x.copy() if current['feasible'] else None
    best_feasible = current.copy() if current['feasible'] else None

    history = []
    temp = initial_temp
    base_step = base_step_frac * spans

    for it in range(max_iter):
        cooling_progress = max(temp / initial_temp, min_step_frac / base_step_frac)
        step_scale = base_step * cooling_progress
        candidate_x = current_x + rng.normal(0.0, step_scale)
        candidate_x = _clip_to_bounds(candidate_x, bounds)
        candidate = _evaluate_problem3_candidate(
            candidate_x, dt=dt_search, penalty_weight=penalty_weight
        )

        delta = candidate['objective'] - current['objective']
        accept = delta <= 0 or rng.random() < np.exp(-delta / max(temp, 1e-12))
        if accept:
            current_x = candidate_x
            current = candidate

        if current['objective'] < best_penalty['objective']:
            best_penalty_x = current_x.copy()
            best_penalty = current.copy()

        if current['feasible']:
            if best_feasible is None or current['area'] < best_feasible['area']:
                best_feasible_x = current_x.copy()
                best_feasible = current.copy()

        history.append({
            'iter': it,
            'temperature': temp,
            'objective': current['objective'],
            'area': current['area'],
            'violation': current['violation'],
            'feasible': current['feasible'],
            'best_area': np.nan if best_feasible is None else best_feasible['area'],
        })

        temp *= cooling_rate
        if temp < final_temp:
            break

    final_x = best_feasible_x if best_feasible is not None else best_penalty_x
    T1, T6, T7, T8, v = final_x
    t, T = simulate(T1, T6, T7, T8, v, dt=dt_final)
    ind = compute_indicators(t, T)
    area = area_217_to_peak(t, T)

    return {
        'x': final_x,
        'fun': area,
        'area': area,
        'T1': T1, 'T6': T6, 'T7': T7, 'T8': T8,
        'v': v,
        't': t, 'T': T,
        'indicators': ind,
        'feasible': check_feasible(ind),
        'violation': _constraint_violation_problem3(ind),
        'history': history,
        'seed': seed,
        'method': 'simulated_annealing',
    }


def optimize_problem3():
    """求解问题3：优化温区设定和速度最小化题图阴影面积

    变量：x=[T1, T6, T7, T8, v]
    范围：
        T1: [165, 185]
        T6: [185, 205]
        T7: [225, 245]
        T8: [245, 265]
        v:  [65, 100]
    """
    result = differential_evolution(
        _objective_problem3,
        PROBLEM3_BOUNDS,
        args=(None, 3.0),  # 粗步长加速
        seed=42,
        maxiter=200,
        tol=1e-3,
        popsize=15,
    )

    # 用精细步长重新计算最终结果
    T1, T6, T7, T8, v = result.x
    t, T = simulate(T1, T6, T7, T8, v, dt=0.5)
    ind = compute_indicators(t, T)
    area = area_217_to_peak(t, T)

    return {
        'x': result.x,
        'fun': result.fun,
        'area': area,
        'T1': T1, 'T6': T6, 'T7': T7, 'T8': T8,
        'v': v,
        't': t, 'T': T,
        'indicators': ind,
    }


# ==================== 问题4：对称性优化 ====================

def _problem4_terms(t, T, alpha=0.7, area_scale=PROBLEM4_AREA_SCALE):
    """Return problem 4 area, mirror symmetry, and combined objective."""
    S3 = area_217_to_peak(t, T)
    total_area = area_above_217(t, T)
    D = symmetry_error(t, T)
    S3_norm = S3 / area_scale if S3 > 0 else 0.0
    D_norm = D
    J = alpha * S3_norm + (1 - alpha) * D_norm
    return S3, total_area, D, S3_norm, D_norm, J


def _objective_problem4(x, alpha=0.7, dt=2.0,
                        area_scale=PROBLEM4_AREA_SCALE):
    """Problem 4 objective: problem-3 area plus mirror symmetry."""
    T1, T6, T7, T8, v = x
    t, T = simulate(T1, T6, T7, T8, v, dt=dt, fast=True)
    ind = compute_indicators(t, T)

    if not check_feasible(ind):
        return 1e8

    _, _, _, _, _, J = _problem4_terms(
        t, T, alpha=alpha, area_scale=area_scale
    )
    return J


def _evaluate_problem4_candidate(
        x, alpha=0.7, dt=2.0, penalty_weight=1e5,
        area_scale=PROBLEM4_AREA_SCALE):
    """Evaluate a problem 4 candidate with process penalties."""
    T1, T6, T7, T8, v = x
    t, T = simulate(T1, T6, T7, T8, v, dt=dt, fast=True)
    ind = compute_indicators(t, T)
    area, total_area, D, A_norm, D_norm, J = _problem4_terms(
        t, T, alpha=alpha, area_scale=area_scale
    )
    feasible = check_feasible(ind)
    violation = _constraint_violation_problem3(ind)
    objective = J + penalty_weight * violation
    return {
        'objective': objective,
        'J': J,
        'area': area,
        'total_area_above_217': total_area,
        'symmetry_error': D,
        'A_norm': A_norm,
        'D_norm': D_norm,
        'feasible': feasible,
        'violation': violation,
        'indicators': ind,
        't': t,
        'T': T,
    }


def simulated_annealing_problem4(
        alpha=0.7, bounds=None, seed=42, max_iter=2000,
        initial_temp=3.0, final_temp=1e-5, cooling_rate=0.995,
        dt_search=2.0, dt_final=0.5, penalty_weight=1e5,
        area_scale=PROBLEM4_AREA_SCALE):
    """Simulated annealing for problem 4."""
    if bounds is None:
        bounds = PROBLEM3_BOUNDS

    rng = np.random.default_rng(seed)
    bounds_arr = np.array(bounds, dtype=float)
    lows = bounds_arr[:, 0]
    highs = bounds_arr[:, 1]
    spans = highs - lows

    current_x = lows + rng.random(len(bounds)) * spans
    current = _evaluate_problem4_candidate(
        current_x, alpha=alpha, dt=dt_search,
        penalty_weight=penalty_weight, area_scale=area_scale
    )

    best_penalty_x = current_x.copy()
    best_penalty = current.copy()
    best_feasible_x = current_x.copy() if current['feasible'] else None
    best_feasible = current.copy() if current['feasible'] else None

    history = []
    temp = initial_temp
    base_step = 0.18 * spans

    for it in range(max_iter):
        cooling_progress = max(temp / initial_temp, 0.05)
        step_scale = base_step * cooling_progress
        candidate_x = current_x + rng.normal(0.0, step_scale)
        candidate_x = _clip_to_bounds(candidate_x, bounds)
        candidate = _evaluate_problem4_candidate(
            candidate_x, alpha=alpha, dt=dt_search,
            penalty_weight=penalty_weight, area_scale=area_scale
        )

        delta = candidate['objective'] - current['objective']
        accept = delta <= 0 or rng.random() < np.exp(-delta / max(temp, 1e-12))
        if accept:
            current_x = candidate_x
            current = candidate

        if current['objective'] < best_penalty['objective']:
            best_penalty_x = current_x.copy()
            best_penalty = current.copy()

        if current['feasible']:
            if best_feasible is None or current['J'] < best_feasible['J']:
                best_feasible_x = current_x.copy()
                best_feasible = current.copy()

        history.append({
            'iter': it,
            'temperature': temp,
            'objective': current['objective'],
            'J': current['J'],
            'area': current['area'],
            'total_area_above_217': current['total_area_above_217'],
            'symmetry_error': current['symmetry_error'],
            'A_norm': current['A_norm'],
            'D_norm': current['D_norm'],
            'violation': current['violation'],
            'feasible': current['feasible'],
            'best_J': np.nan if best_feasible is None else best_feasible['J'],
            'best_objective': best_penalty['objective'],
        })

        temp *= cooling_rate
        if temp < final_temp:
            break

    final_x = best_feasible_x if best_feasible is not None else best_penalty_x
    T1, T6, T7, T8, v = final_x
    t, T = simulate(T1, T6, T7, T8, v, dt=dt_final)
    ind = compute_indicators(t, T)
    A, total_area, D, A_norm, D_norm, J = _problem4_terms(
        t, T, alpha=alpha, area_scale=area_scale
    )

    return {
        'x': final_x,
        'fun': J,
        'J': J,
        'T1': T1, 'T6': T6, 'T7': T7, 'T8': T8,
        'v': v,
        'area': A,
        'total_area_above_217': total_area,
        'symmetry_error': D,
        'A_norm': A_norm,
        'D_norm': D_norm,
        'alpha': alpha,
        'area_scale': area_scale,
        't': t, 'T': T,
        'indicators': ind,
        'feasible': check_feasible(ind),
        'violation': _constraint_violation_problem3(ind),
        'history': history,
        'seed': seed,
        'method': 'simulated_annealing',
    }


def optimize_problem4(alpha=0.7, area_scale=PROBLEM4_AREA_SCALE):
    """Solve problem 4 with problem-3 area and mirror symmetry."""
    result = differential_evolution(
        _objective_problem4,
        PROBLEM3_BOUNDS,
        args=(alpha, 3.0, area_scale),
        seed=42,
        maxiter=200,
        tol=1e-3,
        popsize=15,
    )

    T1, T6, T7, T8, v = result.x
    t, T = simulate(T1, T6, T7, T8, v, dt=0.5)
    ind = compute_indicators(t, T)
    A, total_area, D, A_norm, D_norm, J = _problem4_terms(
        t, T, alpha=alpha, area_scale=area_scale
    )

    return {
        'x': result.x,
        'fun': J,
        'J': J,
        'T1': T1, 'T6': T6, 'T7': T7, 'T8': T8,
        'v': v,
        'area': A,
        'total_area_above_217': total_area,
        'symmetry_error': D,
        'A_norm': A_norm,
        'D_norm': D_norm,
        'alpha': alpha,
        'area_scale': area_scale,
        't': t, 'T': T,
        'indicators': ind,
        'feasible': check_feasible(ind),
        'violation': _constraint_violation_problem3(ind),
    }
