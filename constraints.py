"""炉温曲线工艺指标计算"""
import numpy as np


def compute_indicators(t, T):
    """计算所有工艺指标

    参数
    ----
    t : ndarray — 时间序列 (s)
    T : ndarray — PCB中心温度 (°C)

    返回
    ----
    dict — 包含指标:
        - v_up_max: 最大升温斜率 (°C/s)
        - v_down_max: 最大降温斜率 (°C/s)
        - t_150_190: 温度150~190°C持续时间 (s)
        - t_above_217: 温度超过217°C持续时间 (s)
        - T_peak: 峰值温度 (°C)
        - t_peak: 峰值时间 (s)
    """
    dt = t[1] - t[0]
    dT_dt = np.gradient(T, dt)
    T_peak = np.max(T)
    t_peak = t[np.argmax(T)]

    # 最大升温斜率（正的最大值）
    v_up_max = np.max(dT_dt)

    # 最大降温斜率（负的最小值，即降温最快的速率）
    v_down_max = np.min(dT_dt)

    # 温度在150~190°C之间的持续时间
    mask_150_190 = (T >= 150.0) & (T <= 190.0)
    t_150_190 = np.sum(mask_150_190) * dt if np.any(mask_150_190) else 0.0

    # 温度超过217°C的持续时间
    mask_above_217 = T >= 217.0
    t_above_217 = np.sum(mask_above_217) * dt if np.any(mask_above_217) else 0.0

    return {
        'v_up_max': v_up_max,
        'v_down_max': v_down_max,
        't_150_190': t_150_190,
        't_above_217': t_above_217,
        'T_peak': T_peak,
        't_peak': t_peak,
    }


def check_feasible(indicators):
    """检查是否满足所有制程界限

    制程界限：
        - 升温斜率 ≤ 3°C/s
        - 降温斜率 ≥ -3°C/s (绝对值 ≤ 3°C/s)
        - 150~190°C 持续时间 60~120s
        - 217°C以上持续时间 40~90s
        - 峰值温度 240~250°C

    参数
    ----
    indicators : dict — compute_indicators() 的返回值

    返回
    ----
    feasible : bool — 是否满足所有约束
    """
    checks = [
        indicators['v_up_max'] <= 3.0,
        indicators['v_down_max'] >= -3.0,
        60.0 <= indicators['t_150_190'] <= 120.0,
        40.0 <= indicators['t_above_217'] <= 90.0,
        240.0 <= indicators['T_peak'] <= 250.0,
    ]
    return all(checks)


def check_feasible_detail(indicators):
    """检查约束并返回详细结果"""
    return {
        '升温斜率 ≤ 3°C/s': (indicators['v_up_max'] <= 3.0,
                              f"{indicators['v_up_max']:.4f} ≤ 3"),
        '降温斜率 ≥ -3°C/s': (indicators['v_down_max'] >= -3.0,
                              f"{indicators['v_down_max']:.4f} ≥ -3"),
        '150~190°C 60~120s': (60.0 <= indicators['t_150_190'] <= 120.0,
                               f"{indicators['t_150_190']:.2f}s in [60,120]"),
        '>217°C 40~90s': (40.0 <= indicators['t_above_217'] <= 90.0,
                           f"{indicators['t_above_217']:.2f}s in [40,90]"),
        '峰值 240~250°C': (240.0 <= indicators['T_peak'] <= 250.0,
                            f"{indicators['T_peak']:.2f}°C in [240,250]"),
    }


def area_above_217(t, T):
    """计算超过217°C部分的面积 (∫max(T-217, 0) dt)

    用于问题3的目标函数。
    """
    mask = T > 217.0
    if not np.any(mask):
        return 0.0
    dt = t[1] - t[0]
    return np.trapz(T[mask] - 217.0, t[mask])


def symmetry_error(t, T):
    """计算峰值两侧超过217°C区域的对称误差

    D = ∫|T(t_peak-τ) - T(t_peak+τ)| dτ

    只对 T>217°C 的区域计算。
    """
    T_peak = np.max(T)
    t_peak = t[np.argmax(T)]

    # 找到 T>217 的区间
    mask = T >= 217.0
    if not np.any(mask):
        return 0.0

    # 截取 T>217 的左右部分
    t_above = t[mask]
    T_above = T[mask]

    # 找到 t_peak 在截取段中的位置
    peak_idx = np.searchsorted(t_above, t_peak)
    left = T_above[:peak_idx + 1]
    right = T_above[peak_idx:]
    t_left = t_above[:peak_idx + 1]
    t_right = t_above[peak_idx:]

    # 将左右对齐到相同长度，计算逐点误差
    n = min(len(left), len(right))
    if n < 2:
        return 0.0

    # 以峰值位置对齐
    T_left = left[-n:] if len(left) >= n else left
    T_right = right[:n] if len(right) >= n else right

    diff = np.abs(T_left - T_right)
    # 使用峰值右侧的时间作为积分变量
    tau = t_right[:n] - t_peak
    return np.trapz(diff, tau)
