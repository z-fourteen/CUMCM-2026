from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "question2_signal_optimization"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid")

v1 = pd.read_csv(OUTPUT / "v1_signal_plan.csv")
v2 = pd.read_csv(OUTPUT / "v2_signal_plan.csv")
v2["场景"] = v2["日期类型"] + "-时段" + v2["时段编号"].astype(str)
plans = pd.concat([v1, v2], ignore_index=True)
plans["场景"] = plans["日期类型"] + "-时段" + plans["时段编号"].astype(str)
order = plans[["日期类型", "时段编号", "场景"]].drop_duplicates().sort_values(["日期类型", "时段编号"])["场景"]

fig, axes = plt.subplots(2, 1, figsize=(16, 10), constrained_layout=True)
summary = plans.groupby(["场景", "版本"], as_index=False)[["周期_C", "南北绿灯", "东西绿灯"]].mean()
summary["场景"] = pd.Categorical(summary["场景"], categories=order, ordered=True)
summary = summary.sort_values(["场景", "版本"])
for version, label, color in [(1, "第一版", "#2f6db0"), (2, "第二版", "#d97706")]:
    part = summary[summary["版本"] == version]
    axes[0].plot(part["场景"], part["周期_C"], marker="o", label=label, color=color)
    axes[1].plot(part["场景"], part["南北绿灯"], marker="o", label=f"{label}-南北", color=color)
    axes[1].plot(part["场景"], part["东西绿灯"], marker="s", linestyle="--", label=f"{label}-东西", color=color, alpha=.7)
axes[0].set_title("各时段优化周期"); axes[0].set_ylabel("周期 C (秒)")
axes[1].set_title("各时段平均绿灯配置"); axes[1].set_ylabel("绿灯时间 (秒)")
for ax in axes:
    ax.tick_params(axis="x", rotation=70); ax.legend(ncol=2)
fig.savefig(OUTPUT / "01_周期与绿灯配置.png", dpi=180)
plt.close(fig)

delay = plans.groupby(["场景", "版本"], as_index=False)["总交叉口延误_秒车"].mean()
delay["场景"] = pd.Categorical(delay["场景"], categories=order, ordered=True)
delay = delay.sort_values(["场景", "版本"])
fig, ax = plt.subplots(figsize=(16, 6), constrained_layout=True)
sns.barplot(data=delay, x="场景", y="总交叉口延误_秒车", hue="版本", palette=["#2f6db0", "#d97706"], ax=ax)
ax.set_title("第一版与第二版平均交叉口延误对比"); ax.set_xlabel(""); ax.set_ylabel("延误（秒·车）"); ax.tick_params(axis="x", rotation=70)
ax.legend(title="版本", labels=["第一版", "第二版"])
fig.savefig(OUTPUT / "02_版本延误对比.png", dpi=180)
plt.close(fig)

heat_data = v2.iloc[:, [2, 1, 7]].copy()
heat_data.columns = ["intersection", "period", "saturation"]
heat_data["scene"] = v2["场景"]
heat = heat_data.pivot_table(index="intersection", columns="scene", values="saturation", aggfunc="mean")
heat = heat.reindex(columns=order)
fig, ax = plt.subplots(figsize=(18, 5), constrained_layout=True)
sns.heatmap(heat, cmap="YlOrRd", vmin=0, vmax=max(.9, float(heat.max().max())), annot=False, ax=ax)
ax.set_title("第二版最大饱和度热力图"); ax.set_xlabel("日期类型-时段"); ax.set_ylabel("交叉口")
ax.tick_params(axis="x", rotation=70)
fig.savefig(OUTPUT / "03_v2_saturation_heatmap.png", dpi=180)
plt.close(fig)
