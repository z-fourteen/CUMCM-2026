"""绘图函数"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from furnace import Tf

# 设置中文字体
mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'STSong', 'FangSong']
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.unicode_minus'] = False


def plot_raw_data(t_data, T_exp):
    """绘制原始实验炉温曲线"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_data, T_exp, 'b-', linewidth=0.8, label='实验数据')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('温度 (°C)')
    ax.set_title('原始实验炉温曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_furnace_profile(x_range=(0, 400)):
    """绘制炉内环境温度场"""
    x = np.linspace(*x_range, 1000)
    T_env = Tf(x)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, T_env, 'r-', linewidth=1.5, label='Tf(x) 炉内环境温度')
    ax.set_xlabel('位置 (cm)')
    ax.set_ylabel('温度 (°C)')
    ax.set_title('炉内环境温度场')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_comparison(t_data, T_exp, T_pred, label='模型预测', color='r--', lw=1.2):
    """绘制实验曲线与模型预测曲线对比"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_data, T_exp, 'b-', linewidth=0.8, label='实验数据')
    ax.plot(t_data, T_pred, color, linewidth=lw, label=label)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('温度 (°C)')
    ax.set_title('实验曲线与模型预测曲线对比')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_comparison_dual(t_data, T_exp, T_pred1, T_pred2, kh, kc):
    """绘制两条模型曲线与实验数据对比"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_data, T_exp, 'b-', linewidth=0.8, label='实验数据')
    ax.plot(t_data, T_pred1, 'r--', linewidth=1.0, alpha=0.7,
            label='模型一 (k=0.014)')
    ax.plot(t_data, T_pred2, 'g-', linewidth=1.2,
            label=f'模型二 (kh={kh:.6f}, kc={kc:.6f})')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('温度 (°C)')
    ax.set_title('实验曲线与模型预测曲线对比')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_residuals(t_data, residuals, title='残差图'):
    """绘制残差图"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_data, residuals, 'ko', markersize=2, alpha=0.5, label='残差')
    ax.axhline(y=0, color='r', linestyle='--', linewidth=0.8)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('残差 (°C)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_keff(t_data, k_eff, delta_T, kh, kc):
    """绘制 k_eff(t) 曲线和温差曲线"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # 上：k_eff(t)
    ax1.plot(t_data, k_eff, 'g-', linewidth=1.2, label='k_eff(t)')
    ax1.axhline(y=kh, color='orange', linestyle='--', linewidth=0.8,
                label=f'kh={kh:.6f}')
    ax1.axhline(y=kc, color='purple', linestyle='--', linewidth=0.8,
                label=f'kc={kc:.6f}')
    ax1.set_ylabel('k_eff')
    ax1.set_title('有效换热系数 k_eff(t) 随时间变化')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 下：温差 delta_T
    ax2.plot(t_data, delta_T, 'm-', linewidth=0.8, label='ΔT = Tf - T')
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('温差 (°C)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, (ax1, ax2)
