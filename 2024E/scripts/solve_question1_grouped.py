from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import solve_question1_kmeans as base


OUTPUT = Path("outputs/question1_grouped_10min_masked")
INPUT = Path("附件2.csv")
DAY_TYPES = ["工作日", "周末及普通休息日", "五一节假日"]
SLOTS_PER_DAY = 144
MINUTES_PER_SLOT = 10
COLORS = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377"]

WEST = ["经三路-纬中路", "经二路-纬中路", "经一路-纬中路", "环西路-纬中路"]
EAST = ["纬中路-景区出入口", "经四路-纬中路", "经五路-纬中路", "环东路-纬中路"]
NORTH = ["经中路-纬一路", "经中路-环北路"]
SOUTH = ["经中路-环南路"]
BRANCH_MAP = {
    1: {"直行": WEST, "左转": SOUTH, "右转": NORTH},
    2: {"直行": EAST, "左转": NORTH, "右转": SOUTH},
    3: {"直行": NORTH, "左转": WEST, "右转": EAST},
    4: {"直行": SOUTH, "左转": EAST, "右转": WEST},
}
ROAD_CLASS = {
    direction: {road: (turn, rank) for turn, roads in turns.items() for rank, road in enumerate(roads)}
    for direction, turns in BRANCH_MAP.items()
}


def classify_date(date_series: pd.Series) -> pd.Series:
    dates = pd.to_datetime(date_series).dt.normalize()
    may_day = (dates >= "2024-05-01") & (dates <= "2024-05-05")
    qingming = (dates >= "2024-04-04") & (dates <= "2024-04-06")
    makeup_workdays = dates.isin(pd.to_datetime(["2024-04-07", "2024-04-28"]))
    ordinary_rest = ((dates.dt.dayofweek >= 5) | qingming) & ~makeup_workdays & ~may_day
    return pd.Series(np.select([may_day, ordinary_rest], [DAY_TYPES[2], DAY_TYPES[1]], default=DAY_TYPES[0]), index=dates.index)


def load_target_and_trajectories() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    target_parts = []
    total_rows = 0
    for chunk in pd.read_csv(INPUT, encoding="gb18030", chunksize=500_000):
        total_rows += len(chunk)
        part = chunk[chunk["交叉口"] == base.TARGET].copy()
        if not part.empty:
            target_parts.append(part)
    target = pd.concat(target_parts, ignore_index=True)
    target["时间"] = pd.to_datetime(target["时间"], errors="coerce")
    target = target.dropna(subset=["方向", "时间", "车牌号"]).drop_duplicates()
    target["方向"] = target["方向"].astype(int)
    target_plates = set(target["车牌号"].unique())

    trajectory_parts = []
    for chunk in pd.read_csv(
        INPUT, encoding="gb18030", usecols=["时间", "车牌号", "交叉口"], chunksize=500_000
    ):
        part = chunk[chunk["车牌号"].isin(target_plates)].copy()
        if not part.empty:
            trajectory_parts.append(part)
    trajectories = pd.concat(trajectory_parts, ignore_index=True)
    trajectories["时间"] = pd.to_datetime(trajectories["时间"], errors="coerce")
    trajectories = trajectories.dropna().drop_duplicates()
    diagnostics = {
        "source_rows": total_rows,
        "target_rows": len(target),
        "target_unique_plates": len(target_plates),
        "trajectory_rows": len(trajectories),
    }
    return target, trajectories, diagnostics


def infer_multihop(target: pd.DataFrame, trajectories: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_events = target.reset_index(drop=True).copy()
    target_events["事件ID"] = np.arange(len(target_events))
    target_lookup = target_events[["事件ID", "时间", "车牌号", "方向"]].rename(columns={"方向": "目标方向"})
    events = trajectories.merge(target_lookup, on=["时间", "车牌号"], how="left")
    events = events.sort_values(["车牌号", "时间", "交叉口"]).reset_index(drop=True)

    event_ids = events["事件ID"].dropna().astype(int)
    result = target_events.set_index("事件ID").copy()
    result["候选路口"] = None
    result["间隔秒"] = np.nan
    result["跳过记录数"] = np.nan
    grouped = events.groupby("车牌号", sort=False, observed=True)
    for shift in range(1, 5):
        next_road = grouped["交叉口"].shift(-shift)
        next_time = grouped["时间"].shift(-shift)
        rows = events["事件ID"].notna()
        ids = events.loc[rows, "事件ID"].astype(int)
        roads = next_road.loc[rows]
        seconds = (next_time.loc[rows] - events.loc[rows, "时间"]).dt.total_seconds()
        current_missing = result.loc[ids, "候选路口"].isna().to_numpy()
        valid = current_missing & roads.notna().to_numpy() & roads.ne(base.TARGET).to_numpy()
        if valid.any():
            chosen_ids = ids.iloc[np.flatnonzero(valid)].to_numpy()
            result.loc[chosen_ids, "候选路口"] = roads.iloc[np.flatnonzero(valid)].to_numpy()
            result.loc[chosen_ids, "间隔秒"] = seconds.iloc[np.flatnonzero(valid)].to_numpy()
            result.loc[chosen_ids, "跳过记录数"] = shift - 1

    result["转向"] = None
    result["置信度"] = "未识别"
    result["未识别原因"] = None
    for direction in sorted(base.DIRECTION_NAMES):
        direction_rows = result["方向"].eq(direction)
        for road, (turn, rank) in ROAD_CLASS[direction].items():
            matched = direction_rows & result["候选路口"].eq(road) & result["间隔秒"].between(15, 1800)
            result.loc[matched, "转向"] = turn
            confidence = "高" if rank == 0 else "中"
            result.loc[matched, "置信度"] = confidence

    missing = result["转向"].isna()
    result.loc[missing & result["候选路口"].isna(), "未识别原因"] = "无后续有效路口记录"
    result.loc[missing & result["候选路口"].notna() & result["间隔秒"].lt(15), "未识别原因"] = "时间间隔过短"
    result.loc[missing & result["候选路口"].notna() & result["间隔秒"].gt(1800), "未识别原因"] = "后续记录超过30分钟"
    incompatible = missing & result["候选路口"].notna() & result["未识别原因"].isna()
    result.loc[incompatible, "未识别原因"] = "后续路口与行驶方向冲突"
    result["日期"] = result["时间"].dt.normalize()
    result["日期类型"] = classify_date(result["日期"])
    result["半小时序号"] = result["时间"].dt.hour * 6 + result["时间"].dt.minute // 10

    summary = pd.concat(
        [
            result["置信度"].value_counts().rename_axis("类别").rename("数量"),
            result.loc[result["转向"].isna(), "未识别原因"].value_counts().rename_axis("类别").rename("数量"),
        ]
    ).reset_index()
    summary["占目标记录比例"] = summary["数量"] / len(result)
    return result.reset_index(), summary


def make_daily_flow(target: pd.DataFrame) -> pd.DataFrame:
    data = target.copy()
    data["日期"] = data["时间"].dt.normalize()
    data["日期类型"] = classify_date(data["日期"])
    data["半小时序号"] = data["时间"].dt.hour * 6 + data["时间"].dt.minute // 10
    dates = pd.DataFrame({"日期": pd.date_range(data["日期"].min(), data["日期"].max(), freq="D")})
    dates["日期类型"] = classify_date(dates["日期"])
    index = pd.MultiIndex.from_product(
        [dates["日期"], range(SLOTS_PER_DAY), sorted(base.DIRECTION_NAMES)], names=["日期", "半小时序号", "方向"]
    )
    daily = data.groupby(["日期", "半小时序号", "方向"]).size().reindex(index, fill_value=0).rename("流量").reset_index()
    daily = daily.merge(dates, on="日期", how="left")
    structural_missing = daily["方向"].eq(2) & daily["日期"].between(pd.Timestamp("2024-04-01"), pd.Timestamp("2024-04-17"))
    daily["观测有效"] = ~structural_missing
    daily.loc[structural_missing, "流量"] = np.nan
    daily["方向名称"] = daily["方向"].map(base.DIRECTION_NAMES)
    return daily


def slot_label(slot: int) -> str:
    minutes = int(slot) * MINUTES_PER_SLOT
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def build_periods_10min(profile: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = profile.copy()
    labels = result["聚类类别"].to_numpy()
    boundaries = [0] + [slot for slot in range(1, SLOTS_PER_DAY) if labels[slot] != labels[slot - 1]]
    period_id = np.zeros(SLOTS_PER_DAY, dtype=int)
    rows = []
    for number, start in enumerate(boundaries, start=1):
        end = boundaries[number] if number < len(boundaries) else SLOTS_PER_DAY
        period_id[start:end] = number
        rows.append({"时段编号": number, "开始时间": slot_label(start), "结束时间": "24:00" if end == SLOTS_PER_DAY else slot_label(end), "包含10分钟数": end - start, "聚类类别": int(labels[start])})
    result["时段编号"] = period_id
    return result, pd.DataFrame(rows)


def choose_clusters(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    profiles = []
    metrics_all = []
    periods_all = []
    for day_type in DAY_TYPES:
        subset = daily[daily["日期类型"] == day_type]
        profile = subset.groupby(["半小时序号", "方向"])["流量"].mean().unstack(fill_value=0)
        profile.columns = [f"{base.DIRECTION_NAMES[column]}平均流量" for column in profile.columns]
        profile["总平均流量"] = profile.sum(axis=1)
        profile = profile.reset_index()
        features = StandardScaler().fit_transform(profile[[f"{name}平均流量" for name in base.DIRECTION_NAMES.values()]])
        metrics = []
        models = {}
        for cluster_count in range(3, 7):
            model = KMeans(n_clusters=cluster_count, n_init=50, random_state=2024).fit(features)
            metrics.append({
                "日期类型": day_type,
                "K": cluster_count,
                "SSE": model.inertia_,
                "轮廓系数": silhouette_score(features, model.labels_),
                "CH指数": calinski_harabasz_score(features, model.labels_),
                "切换次数": int(np.count_nonzero(model.labels_[1:] != model.labels_[:-1])),
            })
            models[cluster_count] = model
        metric_frame = pd.DataFrame(metrics)
        best_score = metric_frame["轮廓系数"].max()
        eligible = metric_frame[metric_frame["轮廓系数"] >= best_score - 0.03]
        best_k = int(eligible.sort_values(["切换次数", "K"]).iloc[0]["K"])
        labels = models[best_k].labels_
        totals = profile.assign(label=labels).groupby("label")["总平均流量"].mean().sort_values()
        ranks = {label: rank + 1 for rank, label in enumerate(totals.index)}
        profile["聚类类别"] = [ranks[label] for label in labels]
        profile["聚类类别"] = base.smooth_short_runs(profile["聚类类别"].to_numpy(), profile)
        clustered, periods = build_periods_10min(profile)
        clustered["日期类型"] = day_type
        periods["日期类型"] = day_type
        periods["K"] = best_k
        profiles.append(clustered)
        metrics_all.append(metric_frame.assign(入选=metric_frame["K"].eq(best_k)))
        periods_all.append(periods)
    return pd.concat(profiles), pd.concat(metrics_all), pd.concat(periods_all)


def estimate_movements(daily: pd.DataFrame, tracks: pd.DataFrame, profiles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    identified = tracks[tracks["转向"].notna()].copy()
    identified["权重"] = identified["置信度"].map({"高": 1.0, "中": 0.75})
    counts = identified.groupby(["日期类型", "半小时序号", "方向", "转向"])["权重"].sum()
    index = pd.MultiIndex.from_product(
        [DAY_TYPES, range(SLOTS_PER_DAY), sorted(base.DIRECTION_NAMES), base.TURNS],
        names=["日期类型", "半小时序号", "方向", "转向"],
    )
    ratios = counts.reindex(index).groupby(level=[0, 1, 2]).transform(lambda values: values / values.sum())
    fallback = identified.groupby(["日期类型", "方向", "转向"])["权重"].sum()
    fallback = fallback.groupby(level=[0, 1]).transform(lambda values: values / values.sum())
    for key in ratios[ratios.isna()].index:
        ratios.loc[key] = fallback.get((key[0], key[2], key[3]), 1 / 3)
    ratios = ratios.groupby(level=[0, 1, 2]).transform(lambda values: values / values.sum()).rename("转向比例").reset_index()

    inflow = daily.rename(columns={"流量": "进口流量"})
    estimates = inflow.merge(ratios, on=["日期类型", "半小时序号", "方向"], how="left")
    estimates["估计转向流量"] = estimates["进口流量"] * estimates["转向比例"]
    mapping = profiles[["日期类型", "半小时序号", "时段编号"]]
    estimates = estimates.merge(mapping, on=["日期类型", "半小时序号"], how="left")
    summary = estimates.groupby(["日期类型", "时段编号", "方向", "转向"]).agg(
        平均每10分钟流量=("估计转向流量", "mean"),
        标准差=("估计转向流量", "std"),
        平均转向比例=("转向比例", "mean"),
        样本数=("估计转向流量", "size"),
    ).reset_index()
    summary["方向名称"] = summary["方向"].map(base.DIRECTION_NAMES)
    return estimates, summary


def save_figure(figure: plt.Figure, name: str) -> None:
    figure.tight_layout()
    figure.savefig(OUTPUT / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def make_plots(profiles: pd.DataFrame, metrics: pd.DataFrame, periods: pd.DataFrame, movements: pd.DataFrame, reasons: pd.DataFrame) -> None:
    base.set_plot_style()
    plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 13, "axes.labelsize": 11, "legend.frameon": False})
    figure, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    for axis, day_type in zip(axes, DAY_TYPES):
        part = profiles[profiles["日期类型"] == day_type]
        for direction, name in base.DIRECTION_NAMES.items():
            axis.plot(part["半小时序号"] / 6, part[f"{name}平均流量"], label=name, linewidth=2)
        for period, group in part.groupby("时段编号"):
            start, end = group["半小时序号"].min() / 6, (group["半小时序号"].max() + 1) / 6
            axis.axvspan(start, end, color=COLORS[(int(period) - 1) % len(COLORS)], alpha=.10)
            axis.axvline(start, color="#555555", linewidth=.7, alpha=.6)
            axis.text((start + end) / 2, .97, f"P{int(period)}", transform=axis.get_xaxis_transform(), ha="center", va="top")
        axis.set_title(day_type, loc="left", fontweight="bold")
        axis.set_ylabel("辆/10分钟")
        axis.grid(axis="y", alpha=.2)
    axes[0].legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.01), borderaxespad=0)
    axes[-1].set_xticks(range(0, 25, 2)); axes[-1].set_xlabel("时刻")
    figure.suptitle("三类日期典型日车流与 K-means 连续时段", fontsize=16, fontweight="bold")
    save_figure(figure, "01_三类日期流量与聚类时段")

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    cmap = ListedColormap(COLORS)
    for axis, day_type in zip(axes, DAY_TYPES):
        part = profiles[profiles["日期类型"] == day_type]
        band = np.repeat(part["聚类类别"].to_numpy()[None, :], 2, axis=0)
        axis.imshow(band, aspect="auto", cmap=cmap, interpolation="nearest", extent=[0, 24, 0, 1])
        axis.set_title(day_type, fontweight="bold"); axis.set_xlabel("时刻"); axis.set_yticks([])
        axis.set_xticks(range(0, 25, 4))
    figure.suptitle("10分钟区间聚类类别色带（颜色按流量等级排序）", fontsize=15, fontweight="bold")
    save_figure(figure, "02_聚类结果色带")

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, day_type in zip(axes, DAY_TYPES):
        part = metrics[metrics["日期类型"] == day_type]
        axis.plot(part["K"], part["轮廓系数"], marker="o", linewidth=2, label="轮廓系数")
        chosen = part[part["入选"]]
        axis.scatter(chosen["K"], chosen["轮廓系数"], s=100, color="#EE6677", zorder=3, label="选定K")
        axis.set_title(day_type, fontweight="bold"); axis.set_xlabel("K"); axis.grid(alpha=.2)
    axes[0].set_ylabel("轮廓系数"); axes[0].legend()
    figure.suptitle("三类日期聚类数选择", fontsize=15, fontweight="bold")
    save_figure(figure, "03_聚类数评价")

    pivot = movements.pivot_table(index=["日期类型", "时段编号"], columns=["方向名称", "转向"], values="平均每10分钟流量")
    figure, axis = plt.subplots(figsize=(15, max(7, .45 * len(pivot))))
    image = axis.imshow(pivot, aspect="auto", cmap="YlOrRd")
    axis.set_yticks(range(len(pivot))); axis.set_yticklabels([f"{a} P{b}" for a, b in pivot.index])
    axis.set_xticks(range(len(pivot.columns))); axis.set_xticklabels([f"{a}-{b}" for a, b in pivot.columns], rotation=45, ha="right")
    axis.set_title("三类日期各聚类时段 12 流向平均流量", fontweight="bold")
    figure.colorbar(image, ax=axis, label="辆/10分钟")
    save_figure(figure, "04_十二流向流量热力图")

    plot_reasons = reasons.sort_values("数量")
    figure, axis = plt.subplots(figsize=(10, 5.5))
    bars = axis.barh(plot_reasons["类别"], plot_reasons["数量"], color=["#4477AA" if x in ["高", "中"] else "#BBBBBB" for x in plot_reasons["类别"]])
    axis.bar_label(bars, labels=[f"{value:,}\n({ratio:.1%})" for value, ratio in zip(plot_reasons["数量"], plot_reasons["占目标记录比例"])], padding=4)
    axis.set_xlabel("目标路口记录数"); axis.set_title("轨迹转向识别覆盖与未识别原因", fontweight="bold"); axis.grid(axis="x", alpha=.2)
    save_figure(figure, "05_轨迹识别诊断")


def write_report(diagnostics: dict, tracks: pd.DataFrame, reasons: pd.DataFrame, periods: pd.DataFrame) -> None:
    identified = tracks["转向"].notna().sum()
    direct = tracks["置信度"].eq("高").sum()
    medium = tracks["置信度"].eq("中").sum()
    lines = "\n".join(
        f"- {day}: " + "；".join(f"P{r.时段编号} {r.开始时间}-{r.结束时间}" for r in group.itertuples())
        for day, group in periods.groupby("日期类型", sort=False)
    )
    report = f"""# 第一问分日期类型修正结果

## 日期口径

- 五一节假日：2024-05-01 至 2024-05-05。
- 工作日包含调休上班日 2024-04-07、2024-04-28。
- 周末及普通休息日包含普通周六、周日和清明假期 2024-04-04 至 2024-04-06，但不包含五一。

## 聚类时段

{lines}

## 相位识别说明

原结果中的 65.59% 是“能够由相邻路口轨迹直接判断转向的覆盖率”，不是与人工真实标签比较得到的准确率。原始摄像头位于停车线后方且没有记录车辆转向，因此数据本身不存在可直接计算监督准确率的真实标签。

修正版从只检查最近相邻路口扩展为检查完整上下游路口链：最近下游路口命中记为高置信度，允许中间路口漏拍后由更远下游路口命中记为中置信度。共 {diagnostics['target_rows']:,} 条目标记录，其中识别 {identified:,} 条，覆盖率 {identified / diagnostics['target_rows']:.2%}；高置信度 {direct:,} 条，中置信度 {medium:,} 条。

未识别的主要原因由 `03_identification_diagnostics.csv` 给出，包括车辆之后未再被主路摄像头捕获、车辆驶入停车场或支路、后续记录超过 30 分钟，以及轨迹方向与道路拓扑冲突。即使覆盖率提高，也不能把它直接表述为“准确率”；论文中应报告覆盖率、置信度和流量守恒，并把缺少真实转向标签列为模型限制。

## 估计方法

高置信度轨迹权重为 1，中置信度轨迹权重为 0.75。分别在工作日、周末及普通休息日、五一节假日内部，按半小时和进口方向计算转向比例，再将未识别车辆按该比例分配。每个进口方向的直行、左转、右转估计量之和严格等于其观测进口流量。
"""
    (OUTPUT / "修正结果说明.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target, trajectories, diagnostics = load_target_and_trajectories()
    daily = make_daily_flow(target)
    tracks, reasons = infer_multihop(target, trajectories)
    profiles, metrics, periods = choose_clusters(daily)
    estimates, movements = estimate_movements(daily, tracks, profiles)

    daily.to_csv(OUTPUT / "01_三类日期十分钟流量.csv", index=False, encoding="utf-8-sig")
    profiles.to_csv(OUTPUT / "02_三类日期聚类结果.csv", index=False, encoding="utf-8-sig")
    reasons.to_csv(OUTPUT / "03_identification_diagnostics.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUTPUT / "04_聚类评价指标.csv", index=False, encoding="utf-8-sig")
    periods.to_csv(OUTPUT / "05_聚类时段定义.csv", index=False, encoding="utf-8-sig")
    movements.to_csv(OUTPUT / "06_三类日期十二流向结果.csv", index=False, encoding="utf-8-sig")
    wide_movements = movements.assign(相位=movements["方向名称"] + "-" + movements["转向"]).pivot_table(
        index=["日期类型", "时段编号"], columns="相位", values="平均每10分钟流量", fill_value=0
    ).reset_index()
    wide_movements.to_csv(OUTPUT / "06_每时段每相位车流量宽表.csv", index=False, encoding="utf-8-sig")
    daily.groupby(["日期类型", "方向", "方向名称"], dropna=False)["观测有效"].agg(
        日期区间数="size", 有效区间数="sum"
    ).reset_index().to_csv(OUTPUT / "09_观测有效性统计.csv", index=False, encoding="utf-8-sig")
    estimates.to_csv(OUTPUT / "07_逐日十分钟十二流向估计.csv", index=False, encoding="utf-8-sig")
    tracks[["事件ID", "时间", "车牌号", "方向", "候选路口", "间隔秒", "转向", "置信度", "未识别原因"]].to_csv(
        OUTPUT / "08_目标路口轨迹识别明细.csv", index=False, encoding="utf-8-sig"
    )
    make_plots(profiles, metrics, periods, movements, reasons)
    write_report(diagnostics, tracks, reasons, periods)
    manifest = {**diagnostics, "identified": int(tracks["转向"].notna().sum()), "coverage": float(tracks["转向"].notna().mean())}
    (OUTPUT / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
