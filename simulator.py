"""统一仿真接口 - 基于已验证的平滑切换双k模型"""
import numpy as np
from scipy.integrate import solve_ivp
from furnace import sigma

# ==================== 固定模型参数 ====================
KH = 0.01966939     # 加热换热系数
KC = 0.00591111     # 冷却换热系数
BETA = 0.5          # Sigmoid切换平滑度
A = 1.178           # 炉温场Sigmoid平滑参数

# ==================== 炉子几何参数 (cm) ====================
ENTRANCE_TO_ZONE1 = 25.0
ZONE_LENGTH = 52.5
GAP_LENGTH = 5.0
NUM_ZONES = 11

# Sigmoid中点位置 (来自已验证的furnace.py，保持不变)
SIGMOID_POSITIONS = [25, 200, 235.5, 271, 342]


def build_Tf(T1, T6, T7, T8):
    """根据温区设定构造炉内环境温度函数 Tf(x)

    构造方式：Tf(x)=T₀ + ΣΔᵢ·σ(a·(x-pᵢ))
    其中Δᵢ为相邻温区温差，pᵢ为Sigmoid中点位置。

    参数
    ----
    T1 : float — 小温区1~5温度 (°C)
    T6 : float — 小温区6温度 (°C)
    T7 : float — 小温区7温度 (°C)
    T8 : float — 小温区8~9温度 (°C)

    返回
    ----
    Tf : callable — Tf(x) 炉内环境温度函数
    """
    amps = [
        T1 - 25.0,     # 室温 → 温区1~5
        T6 - T1,       # 温区5 → 温区6
        T7 - T6,       # 温区6 → 温区7
        T8 - T7,       # 温区7 → 温区8~9
        25.0 - T8,     # 温区8~9 → 温区10~11 (冷却)
    ]

    def Tf(x):
        result = 25.0
        for amp, pos in zip(amps, SIGMOID_POSITIONS):
            result += amp * sigma(A * (x - pos))
        return result

    return Tf


def simulate(T1, T6, T7, T8, speed, t_max=None, dt=0.5, fast=False):
    """统一仿真接口

    使用已验证的平滑切换双k换热模型，求解PCB中心温度。

    参数
    ----
    T1 : float — 小温区1~5温度 (°C)
    T6 : float — 小温区6温度 (°C)
    T7 : float — 小温区7温度 (°C)
    T8 : float — 小温区8~9温度 (°C)
    speed : float — 传送带速度 (cm/min)
    t_max : float — 最大仿真时间 (s), default 500
    dt : float — 输出时间步长 (s), default 0.5
    fast : bool — 快速模式（低精度，用于优化搜索）

    返回
    ----
    t : ndarray — 时间序列 (s)
    T : ndarray — PCB中心温度 (°C)
    """
    Tf = build_Tf(T1, T6, T7, T8)
    v = speed / 60.0  # cm/min → cm/s

    # 自适应仿真时长：炉长652.5cm + 30cm余量
    if t_max is None:
        total_len = ZONE_POSITIONS[11][1]  # zone 11 end ≈ 652.5 cm
        t_max = (total_len + 30) / v * 1.1

    t_eval = np.arange(0, t_max + dt, dt)
    t_eval = t_eval[t_eval <= t_max]  # 确保不超出 t_span

    def ode_rhs(t, y):
        Tf_val = Tf(v * t)
        delta_T = Tf_val - y[0]
        k_eff = KC + (KH - KC) * sigma(BETA * delta_T)
        return [k_eff * delta_T]

    sol = solve_ivp(
        ode_rhs, [0.0, t_max], [25.0],
        t_eval=t_eval, method='RK45',
        max_step=0.5,
        rtol=1e-6 if fast else 1e-8,
        atol=1e-8 if fast else 1e-10
    )

    return sol.t, sol.y[0]


# ==================== 温区位置查询 ====================

def _build_zone_map():
    """构建温区编号→ (起始, 结束) 位置映射"""
    zones = {}
    start = ENTRANCE_TO_ZONE1
    for i in range(1, NUM_ZONES + 1):
        end = start + ZONE_LENGTH
        zones[i] = (start, end)
        start = end + GAP_LENGTH
    return zones


# 全局温区位置表
ZONE_POSITIONS = _build_zone_map()


def zone_center(zone_num):
    """温区中点位置 (cm)"""
    s, e = ZONE_POSITIONS[zone_num]
    return (s + e) / 2


def zone_end(zone_num):
    """温区结束位置 (cm)"""
    return ZONE_POSITIONS[zone_num][1]


def zone_start(zone_num):
    """温区起始位置 (cm)"""
    return ZONE_POSITIONS[zone_num][0]


def position_to_time(x, speed):
    """将空间位置转换为时间 t = x/v

    参数
    ----
    x : float — 位置 (cm)
    speed : float — 传送带速度 (cm/min)

    返回
    ----
    t : float — 时间 (s)
    """
    return x / (speed / 60.0)
