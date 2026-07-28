"""PCB热响应模型 ODE定义和求解"""
import numpy as np
from scipy.integrate import solve_ivp
from furnace import Tf

# 传输速度：70 cm/min = 1.1667 cm/s
V = 1.1667


# ==================== 模型一：单k模型 ====================

def ode(t, T, k):
    """dT/dt = k*(Tf(v*t) - T)"""
    return k * (Tf(V * t) - T)


def solve_ode(k, t_end, t_eval, method='RK45', max_step=0.5):
    """求解ODE得到预测温度（单k模型）"""
    sol = solve_ivp(
        ode, [0.0, t_end], [25.0], args=(k,),
        t_eval=t_eval, method=method,
        max_step=max_step,
        rtol=1e-8, atol=1e-10
    )
    return sol.y[0]


# ==================== 模型二：平滑切换双k模型 ====================

def smooth_k(delta_T, kh, kc, beta=0.5):
    """平滑切换的有效换热系数

    S(delta_T) = 1/(1+exp(-beta*delta_T))
    k_eff = kc + (kh-kc)*S

    参数
    ----
    delta_T : float or ndarray
        Tf - T，炉温与PCB温差
    kh : float
        加热阶段换热系数
    kc : float
        冷却阶段换热系数
    beta : float
        切换平滑度（固定为0.5）
    """
    S = 1 / (1 + np.exp(-beta * delta_T))
    return kc + (kh - kc) * S


def ode_dual(t, T, kh, kc):
    """dT/dt = k_eff(delta_T) * (Tf(v*t) - T)"""
    Tf_val = Tf(V * t)
    delta_T = Tf_val - T
    k_eff = smooth_k(delta_T, kh, kc)
    return k_eff * delta_T


def solve_ode_dual(kh, kc, t_end, t_eval, method='RK45', max_step=0.5):
    """求解平滑切换双k模型的ODE

    从t=0开始，初始条件T(0)=25°C。
    """
    sol = solve_ivp(
        ode_dual, [0.0, t_end], [25.0], args=(kh, kc),
        t_eval=t_eval, method=method,
        max_step=max_step,
        rtol=1e-8, atol=1e-10
    )
    return sol.y[0]


def solve_ode_dual_custom(Tf_func, kh, kc, t_end, t_eval,
                           method='RK45', max_step=0.5):
    """求解自定义 Tf 的平滑切换双k模型（模型三）

    同 solve_ode_dual，但使用用户提供的 Tf_func(x) 代替全局 Tf。

    参数
    ----
    Tf_func : callable — Tf(x) 炉温场函数
    kh : float — 加热换热系数
    kc : float — 冷却换热系数
    t_end : float — 终止时间 (s)
    t_eval : ndarray — 输出时间点
    """
    def ode(t, T):
        Tf_val = Tf_func(V * t)
        delta_T = Tf_val - T
        k_eff = smooth_k(delta_T, kh, kc)
        return k_eff * delta_T

    sol = solve_ivp(
        ode, [0.0, t_end], [25.0],
        t_eval=t_eval, method=method,
        max_step=max_step,
        rtol=1e-8, atol=1e-10
    )
    return sol.y[0]


def compute_k_eff(t, kh, kc, beta=0.5):
    """计算每个时刻的 k_eff 值（用于绘图）"""
    T_pred = solve_ode_dual(kh, kc, t[-1], t)
    Tf_vals = Tf(V * t)
    delta_T = Tf_vals - T_pred
    S = 1 / (1 + np.exp(-beta * delta_T))
    k_eff = kc + (kh - kc) * S
    return k_eff, delta_T
