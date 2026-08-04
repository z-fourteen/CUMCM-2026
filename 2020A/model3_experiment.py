#!/usr/bin/env python3
"""模型三升级实验：验证冷却中心位置和过渡宽度对拟合误差的影响

运行完整比较实验：
    模型一：单k（基准）
    模型二：双k（kh/kc）
    模型三升级：双k + 冷却中心 + 冷却宽度（kh/kc/xc/wc）

输出：
    - 模型三最优参数 kh, kc, xc, wc
    - 三模型误差对比表
    - 诊断图（对比/Tf/残差/冷却局部放大）
"""
import os
import numpy as np
import openpyxl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.unicode_minus'] = False

from furnace import Tf, FurnaceTemperature
from ode_model import solve_ode, solve_ode_dual, solve_ode_dual_custom
from fit import fit_k, fit_kh_kc, fit_kh_kc_xc_wc


# ============================================================
# 1. 加载实验数据
# ============================================================
def load_experimental_data(filepath='附件.xlsx'):
    """从 Excel 加载实验数据，返回 (t_data, T_exp)"""
    wb = openpyxl.load_workbook(filepath)
    ws = wb['Sheet1']
    rows = list(ws.iter_rows(values_only=True))
    # 跳过表头，第1列=时间(s)，第2列=温度(°C)
    t_data = np.array([float(r[0]) for r in rows[1:]])
    T_exp = np.array([float(r[1]) for r in rows[1:]])
    return t_data, T_exp


# ============================================================
# 2. 运行三个模型
# ============================================================
def run_model1(t_data, T_exp):
    """模型一：单k"""
    print("\n" + "=" * 50)
    print("  模型一：单k 参数辨识")
    print("=" * 50)
    k_opt, J_min = fit_k(t_data, T_exp)
    T_pred = solve_ode(k_opt, t_data[-1], t_data)
    return {'k': k_opt, 'J': J_min, 'T_pred': T_pred}


def run_model2(t_data, T_exp):
    """模型二：双k (kh/kc)，固定冷却宽度 wc=5"""
    print("\n" + "=" * 50)
    print("  模型二：双k (kh/kc) 参数辨识")
    print("=" * 50)
    kh_opt, kc_opt, J_min, result = fit_kh_kc(t_data, T_exp)
    T_pred = solve_ode_dual(kh_opt, kc_opt, t_data[-1], t_data)
    return {'kh': kh_opt, 'kc': kc_opt, 'J': J_min, 'T_pred': T_pred,
            'wc': 5.0, 'result': result}


def run_model3(t_data, T_exp):
    """模型三升级：双k + 冷却中心 + 冷却宽度 (kh/kc/xc/wc)"""
    print("\n" + "=" * 50)
    print("  模型三升级：双k + 冷却中心 + 冷却宽度 (kh/kc/xc/wc)")
    print("=" * 50)
    kh_opt, kc_opt, xc_opt, wc_opt, J_min, result = \
        fit_kh_kc_xc_wc(t_data, T_exp)
    furnace = FurnaceTemperature(cooling_center=xc_opt, cooling_width=wc_opt)
    T_pred = solve_ode_dual_custom(
        furnace.temperature, kh_opt, kc_opt,
        t_data[-1], t_data
    )
    return {'kh': kh_opt, 'kc': kc_opt, 'xc': xc_opt, 'wc': wc_opt,
            'J': J_min, 'T_pred': T_pred, 'result': result,
            'furnace': furnace}


# ============================================================
# 3. 误差指标
# ============================================================
def compute_metrics(t_data, T_exp, T_pred):
    """计算完整误差指标"""
    residual = T_exp - T_pred
    n = len(residual)
    rmse = np.sqrt(np.sum(residual**2) / n)
    mae = np.mean(np.abs(residual))

    # 峰值
    i_peak_exp = np.argmax(T_exp)
    i_peak_pred = np.argmax(T_pred)
    t_peak_exp = t_data[i_peak_exp]
    t_peak_pred = t_data[i_peak_pred]
    T_peak_exp = T_exp[i_peak_exp]
    T_peak_pred = T_pred[i_peak_pred]

    # 峰值误差
    peak_temp_error = abs(T_peak_exp - T_peak_pred)
    peak_time_error = abs(t_peak_exp - t_peak_pred)

    # 加热段 (t <= t_peak_exp) 和冷却段 (t > t_peak_exp)
    heat_mask = t_data <= t_peak_exp
    cool_mask = t_data > t_peak_exp
    heat_mae = np.mean(np.abs(residual[heat_mask])) if np.any(heat_mask) else 0.0
    cool_mae = np.mean(np.abs(residual[cool_mask])) if np.any(cool_mask) else 0.0

    # 超过 217°C 持续时间
    mask_above = T_exp >= 217.0
    t_above = np.sum(mask_above) * (t_data[1] - t_data[0]) if np.any(mask_above) else 0.0

    return {
        'RMSE': rmse,
        'MAE': mae,
        '加热MAE': heat_mae,
        '冷却MAE': cool_mae,
        '峰值温度误差': peak_temp_error,
        '峰值时间误差': peak_time_error,
        '超过217°C时间': t_above,
        'T_peak_exp': T_peak_exp,
        't_peak_exp': t_peak_exp,
        'residual': residual,
        'heat_mask': heat_mask,
        'cool_mask': cool_mask,
    }


# ============================================================
# 4. 绘图
# ============================================================
def plot_model3_comparison(t_data, T_exp, model3, metrics, out_dir):
    """生成模型三的诊断图集"""
    T_pred = model3['T_pred']

    # --- 图1：实验 vs 模型三预测 ---
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(t_data, T_exp, 'b-', linewidth=0.8, label='实验数据')
    ax.plot(t_data, T_pred, 'r--', linewidth=1.2,
            label=f"模型三 (kh={model3['kh']:.6f}, kc={model3['kc']:.6f},"
                  f" xc={model3['xc']:.1f}, wc={model3['wc']:.2f})")
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('温度 (°C)')
    ax.set_title('实验曲线与模型三预测对比')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '模型三_实验对比.png'), dpi=150)
    plt.close(fig)

    # --- 图2：Tf(x) 对比 ---
    x = np.linspace(0, 420, 2000)
    Tf_orig = Tf(x)
    furnace3 = model3['furnace']
    Tf_mod3 = furnace3.temperature(x)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(x, Tf_orig, 'b-', linewidth=1.2, label='原始 Tf(x) (wc=5)')
    ax.plot(x, Tf_mod3, 'r--', linewidth=1.2,
            label=f"模型三 Tf(x) (xc={model3['xc']:.1f}, wc={model3['wc']:.2f})")
    # 标注冷却过渡区
    mid = furnace3.cooling_center
    half = model3['wc'] / 2
    ax.axvspan(mid - half, mid + half, alpha=0.1, color='red',
               label=f'冷却过渡区 xc={model3["xc"]:.1f}±{model3["wc"]/2:.1f}cm')
    ax.set_xlabel('位置 (cm)')
    ax.set_ylabel('温度 (°C)')
    ax.set_title('炉内环境温度场 Tf(x) 对比 — 冷却段')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '模型三_Tfx对比.png'), dpi=150)
    plt.close(fig)

    # --- 图3：残差 ---
    fig, ax = plt.subplots(figsize=(11, 4))
    res = metrics['residual']
    ax.plot(t_data, res, 'ko', markersize=2, alpha=0.4, label='残差')
    ax.axhline(y=0, color='r', linestyle='--', linewidth=0.8)
    # 标注冷却段
    cool_mask = metrics['cool_mask']
    if np.any(cool_mask):
        t_peak = metrics['t_peak_exp']
        ax.axvline(x=t_peak, color='orange', linestyle=':', linewidth=0.8,
                   label=f'峰值时间 t={t_peak:.0f}s')
        # 在冷却段用不同颜色
        ax.plot(t_data[cool_mask], res[cool_mask], 'ro', markersize=2,
                alpha=0.4, label='冷却段残差')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('残差 (°C)')
    ax.set_title(f'模型三残差 (RMSE={metrics["RMSE"]:.3f}°C)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '模型三_残差图.png'), dpi=150)
    plt.close(fig)

    # --- 图4：冷却阶段局部放大 ---
    t_peak = metrics['t_peak_exp']
    # 从峰值前 20s 到结束
    mask_zoom = t_data >= (t_peak - 20)
    t_zoom = t_data[mask_zoom]
    T_exp_zoom = T_exp[mask_zoom]
    T_pred_zoom = T_pred[mask_zoom]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(t_zoom, T_exp_zoom, 'b-', linewidth=1.0, label='实验数据')
    ax.plot(t_zoom, T_pred_zoom, 'r--', linewidth=1.2, label='模型三预测')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('温度 (°C)')
    ax.set_title('冷却阶段局部放大（峰值 → 结束）')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '模型三_冷却放大.png'), dpi=150)
    plt.close(fig)

    # --- 图5：三模型对比 ---
    # (此图在 run_all 中统一绘制)


def plot_all_models(t_data, T_exp, models, metrics_all, out_dir):
    """三模型对比图"""
    colors = ['g-', 'b--', 'r-']
    labels = [
        f"模型一 (k={models[0]['k']:.6f})",
        f"模型二 (kh={models[1]['kh']:.6f}, kc={models[1]['kc']:.6f})",
        f"模型三 (kh={models[2]['kh']:.6f}, kc={models[2]['kc']:.6f},"
        f" xc={models[2]['xc']:.1f}, wc={models[2]['wc']:.2f})",
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    # 上图：温度曲线
    ax1.plot(t_data, T_exp, 'k-', linewidth=0.8, label='实验数据', alpha=0.7)
    for i in range(3):
        ax1.plot(t_data, models[i]['T_pred'], colors[i], linewidth=1.0,
                 label=labels[i])
    ax1.set_ylabel('温度 (°C)')
    ax1.set_title('三模型与实验数据对比')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 下图：残差对比
    for i in range(3):
        res = metrics_all[i]['residual']
        ax2.plot(t_data, res, colors[i], linewidth=0.6,
                 label=f'模型{"一二三"[i]} RMSE={metrics_all[i]["RMSE"]:.2f}°C',
                 alpha=0.6)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('残差 (°C)')
    ax2.set_title('残差对比')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '三模型对比.png'), dpi=150)
    plt.close(fig)


# ============================================================
# 5. 主流程
# ============================================================
def print_comparison_table(models, metrics_all):
    """打印三模型对比表"""
    print("\n" + "=" * 80)
    print("  模型对比结果")
    print("=" * 80)

    header = f"{'模型':<16} {'参数数':<8} {'RMSE(°C)':<10} {'MAE(°C)':<10} {'加热MAE':<10} {'冷却MAE':<10} {'峰值误差':<10}"
    print(header)
    print("-" * 80)

    model_names = ['模型一（单k）', '模型二（双k）', '模型三（+xc+wc）']
    for i in range(3):
        m = metrics_all[i]
        print(f"{model_names[i]:<18} {models[i].get('n_params', i+1):<8} "
              f"{m['RMSE']:<10.3f} {m['MAE']:<10.3f} "
              f"{m['加热MAE']:<10.3f} {m['冷却MAE']:<10.3f} "
              f"{m['峰值温度误差']:<10.3f}")

    # 对比模型二 vs 模型三
    print()
    imp = metrics_all[1]['RMSE'] - metrics_all[2]['RMSE']
    print(f"  RMSE 改善（模型二 → 三）: {imp:.3f} °C "
          f"({imp/metrics_all[1]['RMSE']*100:.1f}%)")
    imp_cool = metrics_all[1]['冷却MAE'] - metrics_all[2]['冷却MAE']
    print(f"  冷却 MAE 改善: {imp_cool:.3f} °C "
          f"({imp_cool/metrics_all[1]['冷却MAE']*100:.1f}%)")

    print("\n  模型三最优参数:")
    print(f"    kh = {models[2]['kh']:.6f}")
    print(f"    kc = {models[2]['kc']:.6f}")
    print(f"    xc = {models[2]['xc']:.2f} cm")
    print(f"    wc = {models[2]['wc']:.2f} cm")

    # 模型三详细指标
    m3 = metrics_all[2]
    print(f"\n  模型三详细指标:")
    print(f"    RMSE          = {m3['RMSE']:.4f} °C")
    print(f"    MAE           = {m3['MAE']:.4f} °C")
    print(f"    加热MAE       = {m3['加热MAE']:.4f} °C")
    print(f"    冷却MAE       = {m3['冷却MAE']:.4f} °C")
    print(f"    峰值温度误差  = {m3['峰值温度误差']:.4f} °C")
    print(f"    峰值时间误差  = {m3['峰值时间误差']:.4f} s")
    print(f"    实验峰值      = {m3['T_peak_exp']:.2f} °C @ {m3['t_peak_exp']:.1f}s")


def save_experiment_results(t_data, T_exp, models, metrics_all, out_dir):
    """保存实验结果到文本文件和CSV"""
    # 保存参数对比
    with open(os.path.join(out_dir, '实验结果.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("  模型三升级实验：冷却中心+冷却宽度对拟合误差的影响\n")
        f.write("=" * 70 + "\n\n")

        f.write("模型参数:\n")
        f.write(f"  模型一: k = {models[0]['k']:.6f}\n")
        f.write(f"  模型二: kh = {models[1]['kh']:.6f}, "
                f"kc = {models[1]['kc']:.6f}\n")
        f.write(f"  模型三: kh = {models[2]['kh']:.6f}, "
                f"kc = {models[2]['kc']:.6f}, "
                f"xc = {models[2]['xc']:.2f} cm, "
                f"wc = {models[2]['wc']:.2f} cm\n\n")

        f.write("误差对比:\n")
        header = f"{'模型':<16} {'RMSE(°C)':<10} {'MAE(°C)':<10} {'加热MAE':<10} {'冷却MAE':<10} {'峰值误差':<10}\n"
        f.write(header)
        f.write("-" * 70 + "\n")
        names = ['模型一（单k）', '模型二（双k）', '模型三（+xc+wc）']
        for i in range(3):
            m = metrics_all[i]
            f.write(f"{names[i]:<16} {m['RMSE']:<10.4f} {m['MAE']:<10.4f} "
                    f"{m['加热MAE']:<10.4f} {m['冷却MAE']:<10.4f} "
                    f"{m['峰值温度误差']:<10.4f}\n")

        f.write(f"\nRMSE 改善（模型二 → 三）: "
                f"{metrics_all[1]['RMSE'] - metrics_all[2]['RMSE']:.4f} °C\n")
        f.write(f"冷却 MAE 改善: "
                f"{metrics_all[1]['冷却MAE'] - metrics_all[2]['冷却MAE']:.4f} °C\n")

        f.write(f"\n\n模型三详细指标:\n")
        m3 = metrics_all[2]
        f.write(f"  RMSE          = {m3['RMSE']:.4f} °C\n")
        f.write(f"  MAE           = {m3['MAE']:.4f} °C\n")
        f.write(f"  加热MAE       = {m3['加热MAE']:.4f} °C\n")
        f.write(f"  冷却MAE       = {m3['冷却MAE']:.4f} °C\n")
        f.write(f"  峰值温度误差  = {m3['峰值温度误差']:.4f} °C\n")
        f.write(f"  峰值时间误差  = {m3['峰值时间误差']:.4f} s\n")
        f.write(f"  实验峰值      = {m3['T_peak_exp']:.2f} °C @ "
                f"{m3['t_peak_exp']:.1f}s\n")

    # 保存残差
    for i, name in enumerate(['模型一', '模型二', '模型三']):
        df = np.column_stack([t_data, models[i]['T_pred'],
                              metrics_all[i]['residual']])
        np.savetxt(
            os.path.join(out_dir, f'{name}_残差.csv'),
            df, delimiter=',',
            header='t(s),T_pred(°C),residual(°C)',
            comments='', fmt='%.6f'
        )


def run_all():
    """运行完整实验"""
    out_dir = '模型三result'
    os.makedirs(out_dir, exist_ok=True)

    # 1. 加载数据
    print("正在加载实验数据...")
    t_data, T_exp = load_experimental_data()
    print(f"  数据点: {len(t_data)} 个")
    print(f"  时间范围: {t_data[0]:.1f} ~ {t_data[-1]:.1f} s")

    # 2. 运行三个模型
    models = []
    models.append(run_model1(t_data, T_exp))
    models.append(run_model2(t_data, T_exp))
    models.append(run_model3(t_data, T_exp))

    # 3. 计算指标
    metrics_all = []
    for i in range(3):
        m = compute_metrics(t_data, T_exp, models[i]['T_pred'])
        metrics_all.append(m)
        # 为模型添加参数数量
        models[i]['n_params'] = [1, 2, 3][i]

    # 4. 打印对比表
    print_comparison_table(models, metrics_all)

    # 5. 生成诊断图
    print("\n正在生成诊断图...")
    plot_model3_comparison(t_data, T_exp, models[2], metrics_all[2], out_dir)
    plot_all_models(t_data, T_exp, models, metrics_all, out_dir)
    print(f"  图像已保存至 {out_dir}/")

    # 6. 保存结果
    save_experiment_results(t_data, T_exp, models, metrics_all, out_dir)
    print(f"  结果已保存至 {out_dir}/")

    # 7. 输出分析结论
    print("\n" + "=" * 70)
    print("  分析结论")
    print("=" * 70)

    m2 = metrics_all[1]
    m3 = metrics_all[2]
    rmse_imp = m2['RMSE'] - m3['RMSE']
    cool_imp = m2['冷却MAE'] - m3['冷却MAE']
    xc_opt = models[2]['xc']
    wc_opt = models[2]['wc']

    print(f"\n  模型三参数变化:")
    print(f"    xc: 342.0 (原始) → {xc_opt:.2f} cm")
    print(f"    wc:   5.0 (原始) → {wc_opt:.2f} cm")

    if rmse_imp > 1.0:
        print(f"\n  冷却段效应: 显著")
        print(f"  引入 (xc, wc) 后，RMSE 降低 {rmse_imp:.2f}°C"
              f" ({rmse_imp/m2['RMSE']*100:.1f}%)")
        print(f"  冷却 MAE 降低 {cool_imp:.2f}°C"
              f" ({cool_imp/m2['冷却MAE']*100:.1f}%)")
        if abs(xc_opt - 342) > 10:
            print(f"  → 冷却中心位置偏移 {xc_opt - 342:.1f}cm 是主要改进因素")
        if wc_opt > 15:
            print(f"  → 冷却过渡宽度展宽至 {wc_opt:.1f}cm 是主要改进因素")
        print(f"  → 冷却误差主要来自 Tf(x) 冷却边界建模")
    elif rmse_imp > 0.3:
        print(f"\n  冷却段效应: 中等")
        print(f"  引入 (xc, wc) 后，RMSE 降低 {rmse_imp:.2f}°C")
        if abs(xc_opt - 342) > 5:
            print(f"  → 冷却中心偏移 xc={xc_opt:.1f} 部分改善拟合")
        if wc_opt > 10:
            print(f"  → 冷却过渡展宽 wc={wc_opt:.1f} 部分改善拟合")
        print(f"  → 冷却误差部分来自 Tf(x) 建模，但仍需考虑双热容模型")
    else:
        print(f"\n  冷却段效应: 微小")
        print(f"  引入 (xc, wc) 后，RMSE 仅降低 {rmse_imp:.2f}°C")
        print(f"  → 冷却误差主要来自 PCB 内部热惯性（非 Tf(x) 形状或位置）")
        print(f"  → 需考虑升级为双热容模型")

    # 检查残差模式
    res3 = m3['residual']
    cool_mask = m3['cool_mask']
    if np.any(cool_mask):
        cool_res = res3[cool_mask]
        mid = len(cool_res) // 2
        if mid > 10:
            first_half = np.mean(cool_res[:mid])
            second_half = np.mean(cool_res[mid:])
            if first_half > 1.0 and second_half < -1.0:
                print(f"  ⚠ 冷却段残差仍呈现先正后负模式")
                print(f"  → 提示 PCB 内部热惯性效应显著")
                print(f"  → 建议进一步升级为双热容模型")

    print(f"\n  实验完成！")


if __name__ == '__main__':
    run_all()
