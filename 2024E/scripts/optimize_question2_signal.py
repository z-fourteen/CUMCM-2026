"""Question 2 signal optimization, versions 1 (static) and 2 (dynamic).

The input is Question 1's long movement-flow table (vehicles/10 minutes).
In the absence of measurements for the other 11 intersections, demand is
replicated with configurable intersection scale factors.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MOVEMENTS = ("南北直行", "南北左转", "东西直行", "东西左转", "南北右转", "东西右转")
DEFAULT_CYCLES = (60, 75, 90, 105, 120, 135, 150)


def load_demand(path: Path, scale: list[float]) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw["方向"] = raw["方向"].astype(str)
    raw["转向"] = raw["转向"].astype(str)
    direction = np.select(
        [raw["方向"].str.contains("南"), raw["方向"].str.contains("北"),
         raw["方向"].str.contains("东"), raw["方向"].str.contains("西")],
        ["南北", "南北", "东西", "东西"], default="东西")
    turn = raw["转向"].map(lambda value: "左转" if "左" in value else ("右转" if "右" in value else "直行"))
    raw["movement"] = pd.Series(direction, index=raw.index) + turn
    raw = raw[raw["movement"].isin(MOVEMENTS)]
    base = raw.groupby(["日期类型", "时段编号", "movement"], as_index=False)["平均每10分钟流量"].mean()
    periods = base[["日期类型", "时段编号"]].drop_duplicates().sort_values(["日期类型", "时段编号"])
    rows = []
    for intersection, factor in enumerate(scale, 1):
        part = base.copy()
        part["交叉口"] = f"I{intersection:02d}"
        part["流量"] = part["平均每10分钟流量"] * 6 * factor
        rows.append(part[["日期类型", "时段编号", "交叉口", "movement", "流量"]])
    return pd.concat(rows, ignore_index=True)


def delay(flow: float, cycle: float, green: float, saturation: float, queue: float = 0.0) -> tuple[float, float]:
    effective = saturation * green / cycle
    x = flow / max(effective, 1e-9)
    if x >= 1:
        return 1e5 + 1e4 * (x - 1), x
    ratio = green / cycle
    d = cycle * (1 - ratio) ** 2 / max(2 * (1 - x * ratio), 1e-6)
    return d + queue / max(effective - flow, 1e-6), x


def optimize(demand: pd.DataFrame, version: int, cycles: tuple[int, ...], pedestrian_factor: float) -> pd.DataFrame:
    rows = []
    for (day_type, period, intersection), part in demand.groupby(["日期类型", "时段编号", "交叉口"]):
        flows = part.set_index("movement")["流量"].reindex(MOVEMENTS, fill_value=0.0)
        best = None
        for cycle in cycles:
            for north_south_green in np.arange(20, cycle - 39, 5):
                east_west_green = cycle - 10 - north_south_green
                if east_west_green < 20:
                    continue
                greens = {"南北直行": north_south_green, "南北左转": north_south_green,
                          "东西直行": east_west_green, "东西左转": east_west_green}
                total_delay, max_x = 0.0, 0.0
                queue = 0.0
                for movement, flow in flows.items():
                    sat = 1800.0 * (pedestrian_factor if movement.endswith("右转") and version >= 2 else 1.0)
                    green = cycle if movement.endswith("右转") else greens[movement]
                    if version >= 2:
                        queue = max(0.0, flow - sat * green / cycle) * 0.25
                    d, x = delay(float(flow), cycle, green, sat, queue)
                    total_delay += flow * d
                    max_x = max(max_x, x)
                if max_x <= 0.90 and (best is None or total_delay < best[0]):
                    best = (total_delay, cycle, north_south_green, east_west_green, max_x)
        if best is None:
            best = (1e9, cycles[-1], cycles[-1] // 2 - 5, cycles[-1] // 2 - 5, 1.0)
        total, cycle, ns, ew, max_x = best
        rows.append({"日期类型": day_type, "时段编号": period, "交叉口": intersection,
                     "版本": version, "周期_C": cycle, "南北绿灯": ns, "东西绿灯": ew,
                     "最大饱和度": max_x, "总交叉口延误_秒车": total,
                     "右转行人冲突系数": pedestrian_factor if version >= 2 else 1.0})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/question2_signal_optimization"))
    parser.add_argument("--version", type=int, choices=(1, 2), default=1)
    parser.add_argument("--pedestrian-factor", type=float, default=0.75)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    input_path = args.input
    if input_path is None:
        candidates = list(Path("outputs/question1_12phase_kmeans_10min").glob("06_*.csv"))
        if not candidates:
            raise FileNotFoundError("No Question 1 movement-flow CSV found; pass --input explicitly")
        input_path = candidates[0]
    demand = load_demand(input_path, [1.0] * 12)
    result = optimize(demand, args.version, DEFAULT_CYCLES, args.pedestrian_factor)
    result.to_csv(args.output / f"v{args.version}_signal_plan.csv", index=False, encoding="utf-8-sig")
    manifest = {"version": args.version, "input": str(input_path), "intersections": 12,
                "cycle_candidates": DEFAULT_CYCLES, "main_road_only_objective": True,
                "right_turn": "always-permitted; pedestrian factor applied in v2",
                "queue_propagation": args.version >= 2,
                "demand_assumption": "Question 1 movement demand replicated with scale 1.0"}
    (args.output / f"v{args.version}_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
