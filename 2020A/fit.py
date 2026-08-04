"""参数辨识：一维/二维/三维非线性最小二乘"""
import numpy as np
from scipy.optimize import minimize_scalar, least_squares
from ode_model import solve_ode, solve_ode_dual, solve_ode_dual_custom
from furnace import FurnaceTemperature


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


# ==================== 模型三：三参数(kh, kc, wc)优化 ====================

def residual_model3(params, t_data, T_exp):
    """模型三残差：优化 kh, kc, wc

    wc 为冷却段 5%-95% 过渡宽度 (cm)。
    使用 FurnaceTemperature 构造自定义 Tf(x)。
    """
    kh, kc, wc = params
    furnace = FurnaceTemperature(cooling_width=wc)
    T_pred = solve_ode_dual_custom(
        furnace.temperature, kh, kc,
        t_data[-1], t_data
    )
    return T_pred - T_exp


def fit_kh_kc_wc(t_data, T_exp):
    """使用 least_squares 优化 kh, kc, wc

    参数初值: kh=0.01967, kc=0.00591, wc=5
    参数范围:
        kh: [0.001, 0.1]
        kc: [0.0001, 0.05]
        wc: [5.0, 80.0]   (wc=0 不允许，防止 ac → ∞)
    """
    result = least_squares(
        residual_model3, x0=[0.01966939, 0.00591111, 5.0],
        bounds=([0.001, 0.0001, 5.0], [0.1, 0.05, 80.0]),
        args=(t_data, T_exp),
        verbose=2,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        max_nfev=200,
    )
    kh_opt, kc_opt, wc_opt = result.x
    J_min = np.sum(result.fun**2)
    return kh_opt, kc_opt, wc_opt, J_min, result


# ==================== 模型三升级：四参数(kh, kc, xc, wc)优化 ====================

def residual_model3_v2(params, t_data, T_exp):
    """模型三升级版残差：优化 kh, kc, xc, wc

    xc : 冷却过渡中心位置 (cm)
    wc : 冷却段 5%-95% 过渡宽度 (cm)
    """
    kh, kc, xc, wc = params
    furnace = FurnaceTemperature(cooling_center=xc, cooling_width=wc)
    T_pred = solve_ode_dual_custom(
        furnace.temperature, kh, kc,
        t_data[-1], t_data
    )
    return T_pred - T_exp


def fit_kh_kc_xc_wc(t_data, T_exp):
    """使用 least_squares 优化 kh, kc, xc, wc

    参数初值: kh=0.01967, kc=0.00591, xc=342, wc=30
    参数范围:
        kh: [0.001, 0.1]
        kc: [0.0001, 0.05]
        xc: [330.0, 370.0]
        wc: [10.0, 80.0]
    """
    result = least_squares(
        residual_model3_v2, x0=[0.01966939, 0.00591111, 342.0, 30.0],
        bounds=([0.001, 0.0001, 330.0, 10.0], [0.1, 0.05, 370.0, 80.0]),
        args=(t_data, T_exp),
        verbose=2,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        max_nfev=200,
    )
    kh_opt, kc_opt, xc_opt, wc_opt = result.x
    J_min = np.sum(result.fun**2)
    return kh_opt, kc_opt, xc_opt, wc_opt, J_min, result
