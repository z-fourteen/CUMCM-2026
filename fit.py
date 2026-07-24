"""参数辨识：一维/二维非线性最小二乘"""
import numpy as np
from scipy.optimize import minimize_scalar, least_squares
from ode_model import solve_ode, solve_ode_dual


# ==================== 模型一：单k优化 ====================

def objective(k, t_data, T_exp):
    """目标函数 J(k) = sum((T_exp - T_pred)^2)"""
    T_pred = solve_ode(k, t_data[-1], t_data)
    return np.sum((T_exp - T_pred)**2)


def fit_k(t_data, T_exp, k_min=0.0001, k_max=1.0):
    """搜索最优k* = argmin J(k)"""
    result = minimize_scalar(
        objective, args=(t_data, T_exp),
        bounds=(k_min, k_max), method='bounded'
    )
    return result.x, result.fun


# ==================== 模型二：双参数(kh, kc)优化 ====================

def residual(params, t_data, T_exp):
    """残差向量 r = T_pred - T_exp（least_squares 使用）"""
    kh, kc = params
    T_pred = solve_ode_dual(kh, kc, t_data[-1], t_data)
    return T_pred - T_exp


def fit_kh_kc(t_data, T_exp):
    """使用 least_squares 优化 kh, kc

    参数初值: kh=0.02, kc=0.003
    参数范围: [1e-5, 1e-5] ~ [0.1, 0.1]
    """
    result = least_squares(
        residual, x0=[0.02, 0.003],
        bounds=([1e-5, 1e-5], [0.1, 0.1]),
        args=(t_data, T_exp),
        verbose=0,
    )
    kh_opt, kc_opt = result.x
    J_min = np.sum(result.fun**2)  # SSE = sum(r^2)
    return kh_opt, kc_opt, J_min, result
