from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import solve_question1_kmeans as base


SOURCE = Path("outputs/question1_grouped_10min_masked")
OUTPUT = Path("outputs/question1_12phase_kmeans_10min")
DAY_TYPES = ["工作日", "周末及普通休息日", "五一节假日"]
TURNS = ["直行", "左转", "右转"]
DIRECTIONS = list(base.DIRECTION_NAMES.values())
PHASES = [f"{direction}-{turn}" for direction in DIRECTIONS for turn in TURNS]
COLORS = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB", "#882255"]


def slot_label(slot: int) -> str:
    minutes = int(slot) * 10
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def load_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    estimates = pd.read_csv(SOURCE / "07_逐日十分钟十二流向估计.csv", parse_dates=["日期"])
    estimates["相位"] = estimates["方向名称"] + "-" + estimates["转向"]
    typical = estimates.groupby(["日期类型", "半小时序号", "相位"], observed=True)["估计转向流量"].agg(
        平均流量="mean", 标准差="std", 有效日期数="count"
    ).reset_index()
    features = typical.pivot(index=["日期类型", "半小时序号"], columns="相位", values="平均流量").reset_index()
    for phase in PHASES:
        if phase not in features:
            features[phase] = 0.0
    return estimates, features[["日期类型", "半小时序号", *PHASES]], typical


def stability_score(features: np.ndarray, cluster_count: int) -> float:
    labels = [KMeans(n_clusters=cluster_count, n_init=20, random_state=seed).fit_predict(features) for seed in range(10)]
    scores = [adjusted_rand_score(labels[0], candidate) for candidate in labels[1:]]
    return float(np.mean(scores))


def smooth_labels(labels: np.ndarray, scaled: np.ndarray, minimum_slots: int = 3) -> np.ndarray:
    result = labels.copy()
    for _ in range(30):
        boundaries = [0] + [index for index in range(1, len(result)) if result[index] != result[index - 1]] + [len(result)]
        runs = [(boundaries[index], boundaries[index + 1]) for index in range(len(boundaries) - 1)]
        short = [(start, end) for start, end in runs if end - start < minimum_slots]
        if not short or len(runs) == 1:
            break
        start, end = short[0]
        candidates = set()
        if start > 0:
            candidates.add(result[start - 1])
        if end < len(result):
            candidates.add(result[end])
        if not candidates:
            break
        center = scaled[start:end].mean(axis=0)
        best = min(candidates, key=lambda label: np.linalg.norm(center - scaled[result == label].mean(axis=0)))
        result[start:end] = best
    return result


def cluster_features(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clustered_parts = []
    metric_parts = []
    center_parts = []
    for day_type in DAY_TYPES:
        part = features[features["日期类型"] == day_type].sort_values("半小时序号").copy()
        scaled = StandardScaler().fit_transform(part[PHASES])
        models = {}
        rows = []
        for cluster_count in range(3, 9):
            model = KMeans(n_clusters=cluster_count, n_init=50, random_state=2024).fit(scaled)
            models[cluster_count] = model
            rows.append({
                "日期类型": day_type,
                "K": cluster_count,
                "SSE": model.inertia_,
                "轮廓系数": silhouette_score(scaled, model.labels_),
                "CH指数": calinski_harabasz_score(scaled, model.labels_),
                "DB指数": davies_bouldin_score(scaled, model.labels_),
                "标签切换次数": int(np.count_nonzero(model.labels_[1:] != model.labels_[:-1])),
                "随机种子稳定性ARI": stability_score(scaled, cluster_count),
            })
        metrics = pd.DataFrame(rows)
        best_silhouette = metrics["轮廓系数"].max()
        eligible = metrics[(metrics["轮廓系数"] >= best_silhouette - 0.03) & (metrics["随机种子稳定性ARI"] >= 0.9)]
        if eligible.empty:
            eligible = metrics[metrics["轮廓系数"] >= best_silhouette - 0.03]
        best_k = int(eligible.sort_values(["标签切换次数", "DB指数", "K"]).iloc[0]["K"])
        metrics["入选"] = metrics["K"].eq(best_k)

        raw = models[best_k].labels_
        mean_totals = part.assign(raw=raw).groupby("raw")[PHASES].mean().sum(axis=1).sort_values()
        rank_map = {old: rank + 1 for rank, old in enumerate(mean_totals.index)}
        ranked = np.array([rank_map[label] for label in raw])
        smoothed = smooth_labels(ranked, scaled, minimum_slots=3)
        part["原始聚类"] = ranked
        part["连续化聚类"] = smoothed
        part["日期类型"] = day_type
        clustered_parts.append(part)
        metric_parts.append(metrics)

        for label in sorted(np.unique(smoothed)):
            center = part.loc[part["连续化聚类"] == label, PHASES].mean().to_dict()
            center_parts.append({"日期类型": day_type, "聚类类别": label, **center})
    return pd.concat(clustered_parts), pd.concat(metric_parts), pd.DataFrame(center_parts)


def build_periods(clustered: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapped_parts = []
    rows = []
    for day_type in DAY_TYPES:
        part = clustered[clustered["日期类型"] == day_type].sort_values("半小时序号").copy()
        labels = part["连续化聚类"].to_numpy()
        boundaries = [0] + [index for index in range(1, 144) if labels[index] != labels[index - 1]] + [144]
        period_ids = np.zeros(144, dtype=int)
        for number, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]), start=1):
            period_ids[start:end] = number
            rows.append({
                "日期类型": day_type,
                "时段编号": number,
                "开始时间": slot_label(start),
                "结束时间": "24:00" if end == 144 else slot_label(end),
                "十分钟区间数": end - start,
                "聚类类别": int(labels[start]),
            })
        part["时段编号"] = period_ids
        mapped_parts.append(part)
    return pd.concat(mapped_parts), pd.DataFrame(rows)


def summarize_periods(clustered: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = clustered.groupby(["日期类型", "时段编号"])[PHASES].mean().reset_index()
    long = summary.melt(id_vars=["日期类型", "时段编号"], var_name="相位", value_name="平均每10分钟流量")
    long[["方向", "转向"]] = long["相位"].str.split("-", n=1, expand=True)
    return summary, long


def save_figure(figure: plt.Figure, name: str) -> None:
    figure.tight_layout()
    figure.savefig(OUTPUT / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def make_plots(clustered: pd.DataFrame, metrics: pd.DataFrame, centers: pd.DataFrame, periods: pd.DataFrame, old_periods: pd.DataFrame) -> None:
    base.set_plot_style()
    plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 13, "axes.labelsize": 11, "legend.frameon": False})

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, day_type in zip(axes, DAY_TYPES):
        part = metrics[metrics["日期类型"] == day_type]
        axis.plot(part["K"], part["轮廓系数"], marker="o", linewidth=2)
        chosen = part[part["入选"]]
        axis.scatter(chosen["K"], chosen["轮廓系数"], s=110, color="#EE6677", zorder=3)
        axis.set_title(day_type, fontweight="bold"); axis.set_xlabel("K"); axis.grid(alpha=.2)
    axes[0].set_ylabel("轮廓系数")
    figure.suptitle("12相位K-means聚类数选择", fontsize=15, fontweight="bold")
    save_figure(figure, "01_聚类数选择")

    matrix = centers.set_index(["日期类型", "聚类类别"])[PHASES]
    figure, axis = plt.subplots(figsize=(15, max(5, .55 * len(matrix))))
    image = axis.imshow(matrix, aspect="auto", cmap="YlOrRd")
    axis.set_yticks(range(len(matrix))); axis.set_yticklabels([f"{day}-C{label}" for day, label in matrix.index])
    axis.set_xticks(range(12)); axis.set_xticklabels(PHASES, rotation=45, ha="right")
    axis.set_title("12相位聚类中心（辆/10分钟）", fontweight="bold")
    figure.colorbar(image, ax=axis, label="辆/10分钟")
    save_figure(figure, "02_十二相位聚类中心热力图")

    figure, axes = plt.subplots(3, 1, figsize=(15, 7), sharex=True)
    for axis, day_type in zip(axes, DAY_TYPES):
        part = clustered[clustered["日期类型"] == day_type].sort_values("半小时序号")
        axis.step(part["半小时序号"] / 6, part["原始聚类"], where="post", color="#BBBBBB", label="原始标签")
        axis.step(part["半小时序号"] / 6, part["连续化聚类"], where="post", color="#4477AA", linewidth=2, label="连续化标签")
        axis.set_title(day_type, loc="left", fontweight="bold"); axis.set_ylabel("类别"); axis.grid(alpha=.2)
    axes[0].legend(ncol=2); axes[-1].set_xlabel("时刻"); axes[-1].set_xticks(range(0, 25, 2))
    figure.suptitle("12相位聚类的时间连续化结果", fontsize=15, fontweight="bold")
    save_figure(figure, "03_聚类标签连续化")

    scaled_parts = []
    for day_type in DAY_TYPES:
        part = clustered[clustered["日期类型"] == day_type].copy()
        coords = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(part[PHASES]))
        part["PC1"], part["PC2"] = coords[:, 0], coords[:, 1]
        scaled_parts.append(part)
    projected = pd.concat(scaled_parts)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for axis, day_type in zip(axes, DAY_TYPES):
        part = projected[projected["日期类型"] == day_type]
        scatter = axis.scatter(part["PC1"], part["PC2"], c=part["连续化聚类"], cmap="tab10", s=28, alpha=.85)
        axis.set_title(day_type, fontweight="bold"); axis.set_xlabel("PC1"); axis.grid(alpha=.15)
    axes[0].set_ylabel("PC2")
    figure.suptitle("12维相位特征PCA投影", fontsize=15, fontweight="bold")
    save_figure(figure, "04_PCA聚类分布")

    comparison = []
    for day_type in DAY_TYPES:
        comparison.append({"日期类型": day_type, "四方向方案时段数": int((old_periods["日期类型"] == day_type).sum()), "十二相位方案时段数": int((periods["日期类型"] == day_type).sum())})
    compare = pd.DataFrame(comparison).set_index("日期类型")
    figure, axis = plt.subplots(figsize=(9, 5))
    compare.plot.bar(ax=axis, color=["#BBBBBB", "#4477AA"], rot=0)
    axis.set_ylabel("连续时段数"); axis.set_title("四方向方案与12相位方案复杂度对比", fontweight="bold"); axis.grid(axis="y", alpha=.2)
    save_figure(figure, "05_新旧方案时段数对比")

    turn_colors = {"直行": "#4477AA", "左转": "#EE6677", "右转": "#228833"}
    for day_type in DAY_TYPES:
        part = clustered[clustered["日期类型"] == day_type].sort_values("半小时序号")
        day_periods = periods[periods["日期类型"] == day_type]
        figure, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True)
        for axis, direction in zip(axes.flat, DIRECTIONS):
            for turn in TURNS:
                axis.plot(
                    part["半小时序号"] / 6,
                    part[f"{direction}-{turn}"],
                    color=turn_colors[turn],
                    linewidth=1.8,
                    label=turn,
                )
            for row in day_periods.itertuples():
                start = int(row.开始时间[:2]) + int(row.开始时间[3:]) / 60
                end = 24 if row.结束时间 == "24:00" else int(row.结束时间[:2]) + int(row.结束时间[3:]) / 60
                axis.axvspan(start, end, color=COLORS[(int(row.聚类类别) - 1) % len(COLORS)], alpha=.07)
                axis.axvline(start, color="#555555", linewidth=.8, alpha=.65)
                axis.text((start + end) / 2, .97, f"P{int(row.时段编号)}", transform=axis.get_xaxis_transform(), ha="center", va="top", fontsize=9)
            axis.set_title(direction, fontweight="bold")
            axis.set_ylabel("辆/10分钟")
            axis.grid(axis="y", alpha=.2)
            axis.set_xlim(0, 24)
        axes[0, 0].legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.02), borderaxespad=0)
        axes[-1, 0].set_xlabel("时刻"); axes[-1, 1].set_xlabel("时刻")
        axes[-1, 0].set_xticks(range(0, 25, 2)); axes[-1, 1].set_xticks(range(0, 25, 2))
        figure.suptitle(f"{day_type}：12相位流量曲线与最终时段边界", fontsize=16, fontweight="bold")
        figure.subplots_adjust(top=.90)
        save_figure(figure, f"06_{day_type}_流量曲线与时段边界")


def write_document(metrics: pd.DataFrame, periods: pd.DataFrame, centers: pd.DataFrame) -> None:
    chosen = metrics[metrics["入选"]]
    metric_lines = "\n".join(
        f"- {row.日期类型}：K={int(row.K)}，轮廓系数={row.轮廓系数:.3f}，CH指数={row.CH指数:.1f}，DB指数={row.DB指数:.3f}，稳定性ARI={row.随机种子稳定性ARI:.3f}。"
        for row in chosen.itertuples()
    )
    period_lines = "\n".join(
        f"- {day_type}：" + "；".join(f"P{row.时段编号} {row.开始时间}-{row.结束时间}（C{row.聚类类别}）" for row in group.itertuples())
        for day_type, group in periods.groupby("日期类型", sort=False)
    )
    document = """# 基于12相位向量的十分钟K-means聚类方案

## 一、方案修正原因

原方案使用四个进口方向流量构成四维特征，并使用总流量对类别排序，并非只按总流量聚类。但四方向特征无法区分同一进口方向内部直行、左转、右转的结构差异。两个区间即使进口总量接近，只要左转需求不同，其信号放行需求就可能明显不同。因此修正方案采用12相位流量向量聚类。

## 二、数据基础与计算顺序

本方案复用缺失掩码结果：方向2在2024年4月1日至4月17日标为未观测，不作为真实零流量。先利用车牌多路口轨迹估计每个日期、每个十分钟区间、每个进口方向的直行、左转和右转流量，再进行聚类。计算顺序为：

原始车辆记录 → 日期分类 → 进口流量统计 → 轨迹转向识别 → 未识别车辆比例估计 → 12相位十分钟流量 → 12维聚类 → 连续时段。

这样避免了“先按时段聚类，再用该时段估计相位比例”的循环依赖。

## 三、12维特征

对日期类型 \(g\) 和十分钟区间 \(s\)，构造

\[
\mathbf x_{g,s}=(q_{1S},q_{1L},q_{1R},q_{2S},q_{2L},q_{2R},q_{3S},q_{3L},q_{3R},q_{4S},q_{4L},q_{4R}).
\]

各分量是同类日期在相同十分钟位置的平均相位流量。每个分量分别进行Z-score标准化，防止大流量直行相位完全支配欧氏距离。

## 四、K-means与K值选择

对工作日、周末及普通休息日、五一节假日分别聚类，候选K为3至8。目标函数为

\[
\min\sum_k\sum_{s\in C_k}\|\mathbf z_{g,s}-\boldsymbol\mu_k\|_2^2.
\]

综合SSE、轮廓系数、Calinski-Harabasz指数、Davies-Bouldin指数、标签切换次数和十个随机种子的调整兰德指数ARI选择K。先保留轮廓系数距最优值不超过0.03且ARI不低于0.9的候选，再优先选择标签切换少、DB指数小、K较小的方案。

__METRICS__

## 五、时间连续化

K-means忽略时间顺序，十分钟粒度下可能产生孤立标签。对持续不足3个十分钟区间、即不足30分钟的片段，比较其12维中心与前后相邻类别中心的欧氏距离，将其并入更相似的相邻类别。连续化只改变短片段标签，不改动车流观测值。原始标签和连续化标签均保留，便于检验处理影响。

## 六、最终时段

__PERIODS__

## 七、时段相位流量

连续化完成后，对每个日期类型和连续时段内的十分钟12相位流量取平均。最终宽表每行对应一个“日期类型+时段”，12列分别对应四个进口方向的直行、左转、右转平均流量，单位为辆/10分钟。长表用于论文表格和进一步信号配时计算。

## 八、结果解释原则

类别编号按聚类中心总流量由小到大排序，但类别含义应结合12相位中心解释，而不能只称为低、中、高流量。例如某一类别可能表现为南北直行主导，另一类别可能表现为东西向或左转需求增强。聚类中心热力图用于识别这些结构。

## 九、模型优势与限制

优势是聚类特征直接对应题目要求的12个交通流向，更适合后续信号相位配置；同时考虑方向2结构性缺失、轨迹置信度和时间连续性。限制是12相位不是摄像头直接观测，而是基于轨迹覆盖率75.83%的估计结果。没有人工转向标签，因此仍需将结果称为相位流量估计，并报告轨迹覆盖率和敏感性。

## 十、输出文件

- `01_十分钟十二相位聚类特征.csv`：144个十分钟区间的12维特征及标签；
- `02_K值评价.csv`：聚类评价和稳定性；
- `03_十二相位聚类中心.csv`：各类别12相位中心；
- `04_连续时段定义.csv`：最终时段边界；
- `05_每时段每相位流量宽表.csv`：论文主结果；
- `06_每时段每相位流量长表.csv`：逐相位结果；
- PNG图片：K值选择、中心热力图、连续化、PCA分布及新旧方案对比。
""".replace("__METRICS__", metric_lines).replace("__PERIODS__", period_lines)
    (OUTPUT / "12相位聚类详细建模说明.md").write_text(document, encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    estimates, features, typical = load_features()
    clustered, metrics, centers = cluster_features(features)
    clustered, periods = build_periods(clustered)
    wide, long = summarize_periods(clustered)
    old_periods = pd.read_csv(SOURCE / "05_聚类时段定义.csv")

    clustered.to_csv(OUTPUT / "01_十分钟十二相位聚类特征.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUTPUT / "02_K值评价.csv", index=False, encoding="utf-8-sig")
    centers.to_csv(OUTPUT / "03_十二相位聚类中心.csv", index=False, encoding="utf-8-sig")
    periods.to_csv(OUTPUT / "04_连续时段定义.csv", index=False, encoding="utf-8-sig")
    wide.to_csv(OUTPUT / "05_每时段每相位流量宽表.csv", index=False, encoding="utf-8-sig")
    long.to_csv(OUTPUT / "06_每时段每相位流量长表.csv", index=False, encoding="utf-8-sig")
    typical.to_csv(OUTPUT / "07_相位特征有效样本统计.csv", index=False, encoding="utf-8-sig")
    make_plots(clustered, metrics, centers, periods, old_periods)
    write_document(metrics, periods, centers)
    manifest = {
        "feature_dimension": 12,
        "interval_minutes": 10,
        "selected_k": {row.日期类型: int(row.K) for row in metrics[metrics["入选"]].itertuples()},
        "period_counts": periods.groupby("日期类型").size().astype(int).to_dict(),
    }
    (OUTPUT / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
