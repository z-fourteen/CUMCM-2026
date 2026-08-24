from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


TARGET = "经中路-纬中路"
DIRECTION_NAMES = {
    1: "东向西",
    2: "西向东",
    3: "南向北",
    4: "北向南",
}
ADJACENT = {
    "west": "经三路-纬中路",
    "east": "纬中路-景区出入口",
    "north": "经中路-纬一路",
    "south": "经中路-环南路",
}
NEXT_ROAD_TO_TURN = {
    1: {ADJACENT["west"]: "直行", ADJACENT["south"]: "左转", ADJACENT["north"]: "右转"},
    2: {ADJACENT["east"]: "直行", ADJACENT["north"]: "左转", ADJACENT["south"]: "右转"},
    3: {ADJACENT["north"]: "直行", ADJACENT["west"]: "左转", ADJACENT["east"]: "右转"},
    4: {ADJACENT["south"]: "直行", ADJACENT["east"]: "左转", ADJACENT["west"]: "右转"},
}
TURNS = ["直行", "左转", "右转"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2024 CUMCM E题问题1：K-means时段划分与转向流量估计")
    parser.add_argument("--input", type=Path, default=Path("附件2.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/question1_kmeans"))
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--random-state", type=int, default=2024)
    parser.add_argument("--min-travel-seconds", type=int, default=15)
    parser.add_argument("--max-travel-minutes", type=int, default=20)
    return parser.parse_args()


def set_plot_style() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False


def load_relevant_records(path: Path, chunksize: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    relevant_intersections = {TARGET, *ADJACENT.values()}
    target_parts: list[pd.DataFrame] = []
    trajectory_parts: list[pd.DataFrame] = []
    total_rows = 0
    bad_time_rows = 0

    for chunk in pd.read_csv(
        path,
        encoding="gb18030",
        usecols=["方向", "时间", "车牌号", "交叉口"],
        chunksize=chunksize,
    ):
        total_rows += len(chunk)
        chunk["时间"] = pd.to_datetime(chunk["时间"], errors="coerce")
        bad_time_rows += int(chunk["时间"].isna().sum())
        chunk = chunk.dropna(subset=["时间", "车牌号", "交叉口"])
        chunk["方向"] = pd.to_numeric(chunk["方向"], errors="coerce")
        chunk = chunk[chunk["方向"].isin(DIRECTION_NAMES)].copy()
        target_chunk = chunk[chunk["交叉口"] == TARGET]
        relevant_chunk = chunk[chunk["交叉口"].isin(relevant_intersections)]
        if not target_chunk.empty:
            target_parts.append(target_chunk)
        if not relevant_chunk.empty:
            trajectory_parts.append(relevant_chunk)

    target = pd.concat(target_parts, ignore_index=True)
    trajectories = pd.concat(trajectory_parts, ignore_index=True)
    duplicate_cols = ["方向", "时间", "车牌号", "交叉口"]
    target_duplicates = int(target.duplicated(duplicate_cols).sum())
    trajectory_duplicates = int(trajectories.duplicated(duplicate_cols).sum())
    target = target.drop_duplicates(duplicate_cols).copy()
    trajectories = trajectories.drop_duplicates(duplicate_cols).copy()
    diagnostics = {
        "source_rows": total_rows,
        "invalid_time_rows": bad_time_rows,
        "target_rows_after_cleaning": len(target),
        "target_duplicates_removed": target_duplicates,
        "trajectory_rows_after_cleaning": len(trajectories),
        "trajectory_duplicates_removed": trajectory_duplicates,
        "start_time": str(target["时间"].min()),
        "end_time": str(target["时间"].max()),
        "active_dates": int(target["时间"].dt.date.nunique()),
    }
    return target, trajectories, diagnostics


def add_time_fields(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["日期"] = result["时间"].dt.normalize()
    result["半小时序号"] = result["时间"].dt.hour * 2 + result["时间"].dt.minute // 30
    return result


def make_half_hour_tables(target: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = add_time_fields(target)
    all_dates = pd.date_range(target["日期"].min(), target["日期"].max(), freq="D")
    full_index = pd.MultiIndex.from_product(
        [all_dates, range(48), sorted(DIRECTION_NAMES)], names=["日期", "半小时序号", "方向"]
    )
    daily = (
        target.groupby(["日期", "半小时序号", "方向"], observed=True)
        .size()
        .reindex(full_index, fill_value=0)
        .rename("流量")
        .reset_index()
    )
    daily["方向名称"] = daily["方向"].map(DIRECTION_NAMES)
    profile = daily.groupby(["半小时序号", "方向"], observed=True)["流量"].agg(["mean", "std"]).reset_index()
    means = profile.pivot(index="半小时序号", columns="方向", values="mean").fillna(0)
    means = means.rename(columns={key: f"{value}平均流量" for key, value in DIRECTION_NAMES.items()})
    means["总平均流量"] = means.sum(axis=1)
    means = means.reset_index()
    means["开始时间"] = means["半小时序号"].map(slot_label)
    return daily, means


def slot_label(slot: int) -> str:
    minutes = int(slot) * 30
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def cluster_slots(profile: pd.DataFrame, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    flow_columns = [f"{name}平均流量" for name in DIRECTION_NAMES.values()]
    features = profile[flow_columns].to_numpy()
    scaled = StandardScaler().fit_transform(features)
    metrics = []
    models: dict[int, KMeans] = {}
    for cluster_count in range(2, 9):
        model = KMeans(n_clusters=cluster_count, n_init=50, random_state=random_state)
        labels = model.fit_predict(scaled)
        metrics.append(
            {
                "K": cluster_count,
                "SSE": model.inertia_,
                "轮廓系数": silhouette_score(scaled, labels),
                "原始切换次数": int(np.count_nonzero(labels[1:] != labels[:-1])),
            }
        )
        models[cluster_count] = model
    metrics_frame = pd.DataFrame(metrics)
    traffic_candidates = metrics_frame[metrics_frame["K"] >= 3]
    max_silhouette = traffic_candidates["轮廓系数"].max()
    eligible = traffic_candidates[traffic_candidates["轮廓系数"] >= max_silhouette - 0.03]
    best_k = int(eligible.sort_values(["原始切换次数", "K"]).iloc[0]["K"])
    labels = models[best_k].labels_

    cluster_totals = profile.assign(_label=labels).groupby("_label")["总平均流量"].mean().sort_values()
    semantic_labels = {old: rank + 1 for rank, old in enumerate(cluster_totals.index)}
    result = profile.copy()
    result["聚类类别"] = [semantic_labels[label] for label in labels]
    result["类别含义"] = result["聚类类别"].map(
        {rank: f"流量等级{rank}" for rank in range(1, best_k + 1)}
    )
    return result, metrics_frame, best_k


def circular_runs(labels: list[int]) -> list[list[int]]:
    runs: list[list[int]] = [[0]]
    for index in range(1, len(labels)):
        if labels[index] == labels[index - 1]:
            runs[-1].append(index)
        else:
            runs.append([index])
    if len(runs) > 1 and labels[runs[0][0]] == labels[runs[-1][0]]:
        runs[0] = runs[-1] + runs[0]
        runs.pop()
    return runs


def smooth_short_runs(labels: np.ndarray, profile: pd.DataFrame, minimum_slots: int = 2) -> np.ndarray:
    smoothed = labels.copy()
    flow_columns = [f"{name}平均流量" for name in DIRECTION_NAMES.values()]
    features = StandardScaler().fit_transform(profile[flow_columns])
    while True:
        runs = circular_runs(smoothed.tolist())
        short_runs = [run for run in runs if len(run) < minimum_slots]
        if not short_runs or len(runs) == 1:
            break
        run = short_runs[0]
        left_index = (run[0] - 1) % len(smoothed)
        right_index = (run[-1] + 1) % len(smoothed)
        candidates = {smoothed[left_index], smoothed[right_index]}
        run_center = features[run].mean(axis=0)
        best_label = min(
            candidates,
            key=lambda label: np.linalg.norm(
                run_center - features[smoothed == label].mean(axis=0)
            ),
        )
        smoothed[run] = best_label
    return smoothed


def build_periods(clustered: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = clustered.copy()
    result["平滑类别"] = smooth_short_runs(result["聚类类别"].to_numpy(), result)
    labels = result["平滑类别"].tolist()
    boundaries = [0]
    for slot in range(1, 48):
        if labels[slot] != labels[slot - 1]:
            boundaries.append(slot)
    period_id = np.zeros(48, dtype=int)
    period_rows = []
    for sequence, start in enumerate(boundaries, start=1):
        end = boundaries[sequence] if sequence < len(boundaries) else 48
        slots = list(range(start, end))
        period_id[slots] = sequence
        period_rows.append(
            {
                "时段编号": sequence,
                "开始时间": slot_label(start),
                "结束时间": "24:00" if end == 48 else slot_label(end),
                "包含半小时数": len(slots),
                "聚类类别": labels[start],
            }
        )
    result["时段编号"] = period_id
    return result, pd.DataFrame(period_rows)


def infer_turns(
    target: pd.DataFrame,
    trajectories: pd.DataFrame,
    min_seconds: int,
    max_minutes: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = trajectories.sort_values(["车牌号", "时间", "交叉口"]).copy()
    ordered["下一交叉口"] = ordered.groupby("车牌号", observed=True)["交叉口"].shift(-1)
    ordered["下一时刻"] = ordered.groupby("车牌号", observed=True)["时间"].shift(-1)
    target_tracks = ordered[ordered["交叉口"] == TARGET].copy()
    target_tracks["后续间隔秒"] = (target_tracks["下一时刻"] - target_tracks["时间"]).dt.total_seconds()
    valid_interval = target_tracks["后续间隔秒"].between(min_seconds, max_minutes * 60)
    target_tracks["转向"] = None
    for direction, mapping in NEXT_ROAD_TO_TURN.items():
        mask = (target_tracks["方向"] == direction) & valid_interval
        target_tracks.loc[mask, "转向"] = target_tracks.loc[mask, "下一交叉口"].map(mapping)

    identified = target_tracks.dropna(subset=["转向"]).copy()
    identified = add_time_fields(identified)
    counts = identified.groupby(["半小时序号", "方向", "转向"], observed=True).size().rename("已识别数").reset_index()
    totals = counts.groupby(["半小时序号", "方向"], observed=True)["已识别数"].transform("sum")
    counts["转向比例"] = counts["已识别数"] / totals

    full_index = pd.MultiIndex.from_product(
        [range(48), sorted(DIRECTION_NAMES), TURNS], names=["半小时序号", "方向", "转向"]
    )
    ratios = counts.set_index(["半小时序号", "方向", "转向"])["转向比例"].reindex(full_index)
    direction_fallback = identified.groupby(["方向", "转向"], observed=True).size()
    direction_fallback = direction_fallback / direction_fallback.groupby(level=0).transform("sum")
    for slot, direction, turn in ratios[ratios.isna()].index:
        ratios.loc[(slot, direction, turn)] = direction_fallback.get((direction, turn), 1 / 3)
    ratios = ratios.groupby(level=[0, 1]).transform(lambda values: values / values.sum()).rename("转向比例").reset_index()

    target_timed = add_time_fields(target)
    all_dates = pd.date_range(target_timed["日期"].min(), target_timed["日期"].max(), freq="D")
    full_index = pd.MultiIndex.from_product(
        [all_dates, range(48), sorted(DIRECTION_NAMES)], names=["日期", "半小时序号", "方向"]
    )
    daily_inflow = (
        target_timed.groupby(["日期", "半小时序号", "方向"], observed=True)
        .size()
        .reindex(full_index, fill_value=0)
        .rename("进口流量")
        .reset_index()
    )
    estimates = daily_inflow.merge(ratios, on=["半小时序号", "方向"], how="left")
    estimates["估计转向流量"] = estimates["进口流量"] * estimates["转向比例"]
    coverage = {
        "target_records_for_tracking": len(target_tracks),
        "turns_identified": len(identified),
        "identification_rate": len(identified) / len(target_tracks) if len(target_tracks) else 0,
    }
    return estimates, pd.DataFrame([coverage])


def summarize_movements(
    daily_estimates: pd.DataFrame, slot_periods: pd.DataFrame, periods: pd.DataFrame
) -> pd.DataFrame:
    mapping = slot_periods[["半小时序号", "时段编号"]]
    merged = daily_estimates.merge(mapping, on="半小时序号", how="left")
    summary = (
        merged.groupby(["时段编号", "方向", "转向"], observed=True)
        .agg(
            平均每半小时流量=("估计转向流量", "mean"),
            标准差=("估计转向流量", "std"),
            平均转向比例=("转向比例", "mean"),
            样本数=("估计转向流量", "size"),
        )
        .reset_index()
    )
    summary["方向名称"] = summary["方向"].map(DIRECTION_NAMES)
    summary = summary.merge(periods[["时段编号", "开始时间", "结束时间", "聚类类别"]], on="时段编号")
    columns = [
        "时段编号", "开始时间", "结束时间", "聚类类别", "方向", "方向名称", "转向",
        "平均每半小时流量", "标准差", "平均转向比例", "样本数",
    ]
    return summary[columns].sort_values(["时段编号", "方向", "转向"])


def save_plots(profile: pd.DataFrame, metrics: pd.DataFrame, movement_summary: pd.DataFrame, output: Path) -> None:
    set_plot_style()
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].plot(metrics["K"], metrics["SSE"], marker="o")
    axes[0].set(xlabel="聚类数 K", ylabel="SSE", title="肘部法")
    axes[1].plot(metrics["K"], metrics["轮廓系数"], marker="o")
    axes[1].set(xlabel="聚类数 K", ylabel="轮廓系数", title="轮廓系数")
    figure.tight_layout()
    figure.savefig(output / "01_k_selection.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(14, 6))
    x = profile["半小时序号"] / 2
    for name in DIRECTION_NAMES.values():
        axis.plot(x, profile[f"{name}平均流量"], label=name, linewidth=1.7)
    period_colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]
    for period, group in profile.groupby("时段编号"):
        start = group["半小时序号"].min() / 2
        end = (group["半小时序号"].max() + 1) / 2
        axis.axvspan(start, end, color=period_colors[(int(period) - 1) % len(period_colors)], alpha=0.08)
        axis.axvline(start, color="#666666", linewidth=0.7, alpha=0.55)
        axis.text((start + end) / 2, 0.98, f"P{int(period)}", transform=axis.get_xaxis_transform(), ha="center", va="top")
    axis.set(xlabel="时刻", ylabel="平均流量（辆/半小时）", title="目标路口典型日流量与聚类时段")
    axis.set_xticks(range(0, 25, 2))
    axis.legend(ncol=4)
    figure.tight_layout()
    figure.savefig(output / "02_typical_day_periods.png", dpi=180)
    plt.close(figure)

    pivot = movement_summary.pivot_table(
        index=["时段编号", "开始时间", "结束时间"],
        columns=["方向名称", "转向"], values="平均每半小时流量"
    )
    figure, axis = plt.subplots(figsize=(15, max(4.5, 0.55 * len(pivot))))
    image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="YlOrRd")
    axis.set_yticks(range(len(pivot)))
    axis.set_yticklabels([f"P{i[0]} {i[1]}-{i[2]}" for i in pivot.index])
    axis.set_xticks(range(len(pivot.columns)))
    axis.set_xticklabels([f"{item[0]}-{item[1]}" for item in pivot.columns], rotation=55, ha="right")
    axis.set_title("各时段12相位估计流量（辆/半小时）")
    figure.colorbar(image, ax=axis, label="辆/半小时")
    figure.tight_layout()
    figure.savefig(output / "03_movement_heatmap.png", dpi=180)
    plt.close(figure)


def write_report(
    output: Path,
    diagnostics: dict,
    best_k: int,
    metrics: pd.DataFrame,
    periods: pd.DataFrame,
    coverage: pd.DataFrame,
    movement_summary: pd.DataFrame,
) -> None:
    identification_rate = float(coverage.iloc[0]["identification_rate"])
    period_lines = "\n".join(
        f"- P{row.时段编号}: {row.开始时间}-{row.结束时间}，流量等级 {row.聚类类别}"
        for row in periods.itertuples()
    )
    peak = movement_summary.loc[movement_summary["平均每半小时流量"].idxmax()]
    report = f"""# 问题1：K-means时段划分与相位流量估计

## 数据与方法

- 原始记录 {diagnostics['source_rows']:,} 条，目标路口清洗后 {diagnostics['target_rows_after_cleaning']:,} 条。
- 统计范围为 {diagnostics['start_time']} 至 {diagnostics['end_time']}，共 {diagnostics['active_dates']} 天。
- 以半小时为基础区间，使用四个进口方向的典型日平均流量作为 K-means 特征，先标准化再聚类。
- 交通控制至少区分低、中、高三种状态，故在 K=3 至 K=8 中综合轮廓系数与时段连续性，选择 K={best_k}；K=2 仅作为基准列示，少于 1 小时的孤立类别并入相邻最相似类别。
- 利用同车牌从目标路口到四个相邻路口的后续记录识别直行、左转和右转，允许行程间隔 15 秒至 20 分钟；未识别车辆按对应半小时、对应进口方向的已识别转向比例估算。

## 时段划分

{period_lines}

## 关键结果

- 可直接识别转向的目标路口记录为 {int(coverage.iloc[0]['turns_identified']):,} 条，识别率 {identification_rate:.2%}。
- 最大的单相位平均流量出现在 P{int(peak['时段编号'])}（{peak['开始时间']}-{peak['结束时间']}）的“{peak['方向名称']}-{peak['转向']}”，约 {peak['平均每半小时流量']:.1f} 辆/半小时。
- 完整的 12 相位结果见 `06_period_movement_flow.csv`，其流量单位均为辆/半小时。

## 结果口径与限制

- 这里的“相位”按四个进口方向分别拆分为直行、左转、右转，共 12 个交通流向；实际信号配时可再将兼容流向组合为放行相位。
- 摄像头位于停车线后方，原始记录没有转向字段，因此转向结果属于基于轨迹的估计值。
- 车辆驶入沿线停车场、车牌漏拍或超过时间阈值会降低直接识别率；比例外推确保每个进口方向满足流量守恒。
- 当前结果是 36 天混合典型日。如用于精细信号配时，建议进一步分别计算工作日、普通周末和五一假期。
"""
    (output / "结果说明.md").write_text(report, encoding="utf-8")
    (output / "run_manifest.json").write_text(
        json.dumps(
            {"diagnostics": diagnostics, "selected_k": best_k, "identification_rate": identification_rate},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    target, trajectories, diagnostics = load_relevant_records(args.input, args.chunksize)
    daily, profile = make_half_hour_tables(target)
    clustered, metrics, best_k = cluster_slots(profile, args.random_state)
    slot_periods, periods = build_periods(clustered)
    daily_estimates, coverage = infer_turns(
        target, trajectories, args.min_travel_seconds, args.max_travel_minutes
    )
    movement_summary = summarize_movements(daily_estimates, slot_periods, periods)

    daily.to_csv(args.output / "01_daily_half_hour_direction_flow.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(args.output / "02_k_selection_metrics.csv", index=False, encoding="utf-8-sig")
    slot_periods.to_csv(args.output / "03_typical_day_clusters.csv", index=False, encoding="utf-8-sig")
    periods.to_csv(args.output / "04_period_definition.csv", index=False, encoding="utf-8-sig")
    daily_estimates.to_csv(args.output / "05_daily_movement_estimates.csv", index=False, encoding="utf-8-sig")
    movement_summary.to_csv(args.output / "06_period_movement_flow.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(args.output / "07_turn_identification_coverage.csv", index=False, encoding="utf-8-sig")
    save_plots(slot_periods, metrics, movement_summary, args.output)
    write_report(args.output, diagnostics, best_k, metrics, periods, coverage, movement_summary)
    print(f"完成：结果已写入 {args.output}")


if __name__ == "__main__":
    main()
