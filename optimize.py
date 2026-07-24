"""优化函数：问题2、3、4"""
import numpy as np
from scipy.optimize import differential_evolution

from simulator import simulate
from constraints import (compute_indicators, check_feasible,
                          area_above_217, symmetry_error)


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
    """问题3目标函数：超过217°C部分的面积

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

    return area_above_217(t, T)


def optimize_problem3():
    """求解问题3：优化温区设定和速度最小化超过217°C面积

    变量：x=[T1, T6, T7, T8, v]
    范围：
        T1: [160, 190]
        T6: [190, 220]
        T7: [220, 250]
        T8: [240, 270]
        v:  [65, 100]
    """
    bounds = [
        (160, 190),   # T1: 温区1~5
        (190, 220),   # T6: 温区6
        (220, 250),   # T7: 温区7
        (240, 270),   # T8: 温区8~9
        (65, 100),    # v: 传送带速度
    ]

    result = differential_evolution(
        _objective_problem3,
        bounds,
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
    area = area_above_217(t, T)

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

def _objective_problem4(x, alpha=0.5, dt=2.0):
    """问题4目标函数：J = α·A_norm + (1-α)·D_norm

    x = [T1, T6, T7, T8, v]
    alpha: 面积项权重 (default 0.5)
    dt: 仿真步长（优化时用粗步长加速）
    """
    T1, T6, T7, T8, v = x
    t, T = simulate(T1, T6, T7, T8, v, dt=dt, fast=True)
    ind = compute_indicators(t, T)

    if not check_feasible(ind):
        return 1e8

    A = area_above_217(t, T)
    D = symmetry_error(t, T)

    # 归一化因子（基于典型值估计）
    A_norm = A / 2000.0 if A > 0 else 0.0
    D_norm = D / 200.0 if D > 0 else 0.0

    return alpha * A_norm + (1 - alpha) * D_norm


def optimize_problem4(alpha=0.5):
    """求解问题4：在问题3基础上增加对称性约束

    变量同问题3：x=[T1, T6, T7, T8, v]
    目标函数：J = α·A_norm + (1-α)·D_norm
    """
    bounds = [
        (160, 190),   # T1
        (190, 220),   # T6
        (220, 250),   # T7
        (240, 270),   # T8
        (65, 100),    # v
    ]

    result = differential_evolution(
        _objective_problem4,
        bounds,
        args=(alpha, 3.0),  # 粗步长加速
        seed=42,
        maxiter=200,
        tol=1e-3,
        popsize=15,
    )

    # 精细步长重新计算
    T1, T6, T7, T8, v = result.x
    t, T = simulate(T1, T6, T7, T8, v, dt=0.5)
    ind = compute_indicators(t, T)
    A = area_above_217(t, T)
    D = symmetry_error(t, T)

    A_norm = A / 2000.0
    D_norm = D / 200.0

    return {
        'x': result.x,
        'fun': result.fun,
        'T1': T1, 'T6': T6, 'T7': T7, 'T8': T8,
        'v': v,
        'area': A,
        'symmetry_error': D,
        'A_norm': A_norm,
        'D_norm': D_norm,
        'alpha': alpha,
        't': t, 'T': T,
        'indicators': ind,
    }
