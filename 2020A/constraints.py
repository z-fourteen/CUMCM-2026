"""炉温曲线工艺指标计算"""
import numpy as np


def _trapz(y, x):
    """兼容 NumPy 1.x/2.x 的梯形积分。"""
    if hasattr(np, 'trapezoid'):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


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

    保留用于统计整段高于217°C的总热暴露面积。
    """
    mask = T > 217.0
    if not np.any(mask):
        return 0.0
    dt = t[1] - t[0]
    return _trapz(T[mask] - 217.0, t[mask])


def area_217_to_peak(t, T):
    """计算题图阴影面积：首次超过217°C到峰值之间的面积。

    S = ∫[t_217, t_peak] max(T(t)-217, 0) dt
    其中 t_217 使用线性插值估计首次过217°C的时刻。
    """
    peak_idx = int(np.argmax(T))
    if peak_idx <= 0 or T[peak_idx] <= 217.0:
        return 0.0

    crossing_idx = None
    for i in range(peak_idx + 1):
        if T[i] >= 217.0:
            crossing_idx = i
            break

    if crossing_idx is None:
        return 0.0

    if crossing_idx == 0 or T[crossing_idx] == 217.0:
        t_cross = t[crossing_idx]
    else:
        t0, t1 = t[crossing_idx - 1], t[crossing_idx]
        T0, T1 = T[crossing_idx - 1], T[crossing_idx]
        t_cross = t0 + (217.0 - T0) * (t1 - t0) / (T1 - T0)

    t_seg = np.concatenate(([t_cross], t[crossing_idx:peak_idx + 1]))
    T_seg = np.concatenate(([217.0], T[crossing_idx:peak_idx + 1]))
    return _trapz(np.maximum(T_seg - 217.0, 0.0), t_seg)


def _first_crossing_time(t, T, level, end_idx):
    """Return first upward crossing time before or at end_idx."""
    for i in range(end_idx + 1):
        if T[i] >= level:
            if i == 0 or T[i] == level:
                return float(t[i])
            t0, t1 = t[i - 1], t[i]
            T0, T1 = T[i - 1], T[i]
            return float(t0 + (level - T0) * (t1 - t0) / (T1 - T0))
    return None


def _last_crossing_time(t, T, level, start_idx):
    """Return first downward crossing time after or at start_idx."""
    for i in range(start_idx + 1, len(T)):
        if T[i] <= level:
            if T[i] == level:
                return float(t[i])
            t0, t1 = t[i - 1], t[i]
            T0, T1 = T[i - 1], T[i]
            return float(t0 + (level - T0) * (t1 - t0) / (T1 - T0))
    return None


def symmetry_error(t, T, level=217.0, normalize=True, eps=1e-9):
    """Mirror symmetry error around peak time for the region above level.

    The compared profile is H(t)=max(T(t)-level, 0). The integration interval
    uses max(left_duration, right_duration), so an unmatched tail on either
    side contributes to the error.
    """
    peak_idx = int(np.argmax(T))
    if T[peak_idx] <= level:
        return 0.0

    t_peak = float(t[peak_idx])
    t_left = _first_crossing_time(t, T, level, peak_idx)
    t_right = _last_crossing_time(t, T, level, peak_idx)
    if t_left is None or t_right is None or t_right <= t_left:
        return 0.0

    dt = float(t[1] - t[0])
    tau_left = t_peak - t_left
    tau_right = t_right - t_peak
    tau_max = max(tau_left, tau_right)
    if tau_max <= 0:
        return 0.0

    tau = np.arange(0.0, tau_max + 0.5 * dt, dt)
    left_time = t_peak - tau
    right_time = t_peak + tau

    left_h = np.zeros_like(tau)
    right_h = np.zeros_like(tau)
    left_valid = left_time >= t_left
    right_valid = right_time <= t_right

    left_h[left_valid] = np.maximum(
        np.interp(left_time[left_valid], t, T) - level, 0.0
    )
    right_h[right_valid] = np.maximum(
        np.interp(right_time[right_valid], t, T) - level, 0.0
    )

    error_area = float(_trapz(np.abs(left_h - right_h), tau))
    if not normalize:
        return error_area

    total_area = area_above_217(t, T)
    return error_area / (total_area + eps)


