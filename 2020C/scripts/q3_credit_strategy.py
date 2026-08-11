import argparse
import copy
import csv
from collections import Counter

from credit_common import (
    NO,
    OUTPUT_DIR,
    RISK_LEVELS,
    YES,
    clip,
    compute_thresholds,
    load_rate_table,
    safe_ratio,
    score_fields,
    score_q2,
    select_rate,
    strategy_summary,
    write_bar_chart_svg,
    write_csv,
    write_strategy_charts,
)


FIELD_CODE = "企业代号"
FIELD_NAME = "企业名称"
FIELD_CLASS = "估计信用等级"
FIELD_REVENUE = "有效销项金额"
FIELD_MONTHLY = "月均销项收入"
FIELD_MARGIN = "毛利率"
FIELD_MONTHS = "有效经营月份数"
FIELD_GAP = "最长销项断档月数"
FIELD_VOL = "销项收入波动系数"
FIELD_CUSTOMER_HHI = "客户集中度HHI"
FIELD_SUPPLIER_HHI = "供应商集中度HHI"
FIELD_MATCH = "进销匹配差异"
FIELD_SCORE = "信用得分"
FIELD_RISK = "风险等级"
FIELD_REJECT = "是否一票否决"
FIELD_REJECT_REASON = "一票否决原因"
FIELD_HIGH_POOL = "是否高风险池"
FIELD_HIGH_REASON = "高风险池原因"
FIELD_LOAN_STATUS = "是否放贷"
FIELD_INIT_AMOUNT = "初始建议额度_万元"
FIELD_CAP = "额度上限_万元"
FIELD_FINAL_AMOUNT = "最终贷款额度_万元"
FIELD_RATE = "建议贷款年利率"
FIELD_LOSS = "客户流失率"


INDUSTRY_RULES = [
    ("医疗药品与健康服务", "低冲击", ["药房", "药业", "医疗", "医药", "诊所", "健康"]),
    ("食品与基础民生", "低冲击", ["食品", "粮油", "农产品", "生鲜", "超市", "便利"]),
    ("科技与线上服务", "中冲击", ["网络科技", "软件", "信息", "技术服务", "互联网", "科技"]),
    ("装饰装修与家居", "高冲击", ["装饰", "装修", "门窗", "灯饰", "家居", "美居"]),
    ("文娱广告与线下服务", "高冲击", ["文化", "传媒", "广告", "营销策划", "体育", "培训"]),
    ("可选消费与门店经营", "高冲击", ["服饰", "童装", "美容", "花店", "工艺品", "个体经营"]),
    ("建筑与工程", "中高冲击", ["建筑", "工程", "建材", "园林", "市政", "消防"]),
    ("制造与设备", "中冲击", ["机械", "设备", "电子", "塑胶", "材料", "机电"]),
    ("物流运输", "中冲击", ["物流", "运输", "货运", "配送"]),
    ("普通批发商贸", "中冲击", ["商贸", "贸易", "物资", "供应链", "五金", "办公用品"]),
]


BASE_SHOCK = {
    "低冲击": {"rev": 0.05, "margin": 0.02, "vol": 0.10, "month": 0, "amount": 1.00},
    "中冲击": {"rev": 0.15, "margin": 0.05, "vol": 0.25, "month": 1, "amount": 0.90},
    "中高冲击": {"rev": 0.22, "margin": 0.075, "vol": 0.38, "month": 2, "amount": 0.82},
    "高冲击": {"rev": 0.30, "margin": 0.10, "vol": 0.50, "month": 2, "amount": 0.75},
}


SCENARIO_FACTOR = {
    "mild": 0.70,
    "base": 1.00,
    "severe": 1.30,
}


def classify_industry(name):
    for industry, shock_level, keywords in INDUSTRY_RULES:
        if any(keyword in name for keyword in keywords):
            return industry, shock_level
    return "其他服务", "中冲击"


def vulnerability_factor(row):
    factor = 1.0
    reasons = []
    checks = [
        (row[FIELD_CUSTOMER_HHI] > 0.80, 0.20, "客户集中度HHI>0.80"),
        (row[FIELD_SUPPLIER_HHI] > 0.80, 0.20, "供应商集中度HHI>0.80"),
        (row[FIELD_VOL] > 1.00, 0.15, "销项收入波动系数>1.00"),
        (row[FIELD_MARGIN] < 0.05, 0.15, "毛利率<5%"),
        (row[FIELD_MONTHS] < 12, 0.10, "有效经营月份数<12"),
        (row.get(FIELD_HIGH_POOL) == YES, 0.10, "第二问高风险池"),
    ]
    for matched, add, reason in checks:
        if matched:
            factor += add
            reasons.append(reason)
    return min(factor, 1.60), "；".join(reasons)


def apply_shock(rows, baseline_by_code, scenario):
    scenario_scale = SCENARIO_FACTOR[scenario]
    adjusted = []
    for row in copy.deepcopy(rows):
        baseline = baseline_by_code[row[FIELD_CODE]]
        row[FIELD_HIGH_POOL] = baseline[FIELD_HIGH_POOL]
        row[FIELD_HIGH_REASON] = baseline[FIELD_HIGH_REASON]
        industry, shock_level = classify_industry(row[FIELD_NAME])
        params = BASE_SHOCK[shock_level]
        factor, weak_reasons = vulnerability_factor(row)

        d_rev = min(params["rev"] * scenario_scale * factor, 0.60)
        d_margin = min(params["margin"] * scenario_scale * factor, 0.20)
        d_vol = min(params["vol"] * scenario_scale * factor, 1.00)
        d_month = round(params["month"] * scenario_scale * factor)

        row["行业类别"] = industry
        row["冲击等级"] = shock_level
        row["情景"] = scenario
        row["脆弱性修正系数"] = factor
        row["脆弱性原因"] = weak_reasons
        row["收入下降率"] = d_rev
        row["毛利率下降幅度"] = d_margin
        row["波动系数上升率"] = d_vol
        row["经营月份减少数"] = d_month
        row["调整前信用得分"] = baseline[FIELD_SCORE]
        row["调整前风险等级"] = baseline[FIELD_RISK]
        row["调整前贷款额度_万元"] = baseline[FIELD_FINAL_AMOUNT]
        row["调整前贷款年利率"] = baseline[FIELD_RATE]

        row[FIELD_REVENUE] *= 1 - d_rev
        row[FIELD_MONTHLY] *= 1 - d_rev
        row[FIELD_MARGIN] -= d_margin
        row[FIELD_VOL] *= 1 + d_vol
        row[FIELD_MONTHS] = max(0, row[FIELD_MONTHS] - d_month)
        row[FIELD_GAP] += d_month
        adjusted.append(row)
    return adjusted


def add_q3_high_pool_rules(rows):
    for row in rows:
        if row[FIELD_REJECT] == YES:
            continue
        reasons = [x for x in row[FIELD_HIGH_REASON].split("；") if x]
        if row["冲击等级"] == "高冲击" and row[FIELD_VOL] > 1.5:
            reasons.append("高冲击行业且销项收入波动系数>1.5")
        if row["冲击等级"] == "高冲击" and row[FIELD_MONTHS] < 12:
            reasons.append("高冲击行业且有效经营月份数<12")
        row[FIELD_HIGH_POOL] = YES if reasons else NO
        row[FIELD_HIGH_REASON] = "；".join(dict.fromkeys(reasons))
    return rows


def assign_strategy_q3(rows, total_credit_wan, scenario="base"):
    rates = load_rate_table()
    for row in rows:
        if row[FIELD_REJECT] == YES:
            row.update(
                {
                    FIELD_LOAN_STATUS: "不放贷",
                    FIELD_INIT_AMOUNT: 0.0,
                    FIELD_CAP: 0.0,
                    FIELD_FINAL_AMOUNT: 0.0,
                    FIELD_RATE: 0.0,
                    FIELD_LOSS: 0.0,
                    "冲击额度折减系数": 0.0,
                }
            )
            continue

        raw_amount = 10.0 + 90.0 * row[FIELD_SCORE] / 100.0
        if row[FIELD_RISK] == "低风险":
            amount, cap = clip(raw_amount, 70, 100), 100.0
        elif row[FIELD_RISK] == "较低风险":
            amount, cap = clip(raw_amount, 40, 80), 80.0
        elif row[FIELD_RISK] == "中风险":
            amount, cap = clip(raw_amount, 10, 50), 50.0
        else:
            amount, cap = clip(raw_amount, 10, 30), 30.0

        if row[FIELD_HIGH_POOL] == YES:
            amount, cap = min(amount, 30.0), min(cap, 30.0)
        if row["冲击等级"] == "高冲击" and row[FIELD_HIGH_POOL] == YES:
            row.update(
                {
                    FIELD_LOAN_STATUS: "高冲击高风险池暂停授信",
                    FIELD_INIT_AMOUNT: 0.0,
                    FIELD_CAP: 0.0,
                    FIELD_FINAL_AMOUNT: 0.0,
                    FIELD_RATE: 0.0,
                    FIELD_LOSS: 0.0,
                    "冲击额度折减系数": 0.0,
                }
            )
            continue

        discount = BASE_SHOCK[row["冲击等级"]]["amount"]
        amount *= discount
        cap *= discount
        if amount < 10:
            amount, cap = 0.0, 0.0

        rate, loss = select_rate(row, rates)
        row.update(
            {
                FIELD_LOAN_STATUS: "拟放贷" if amount >= 10 else "暂缓授信",
                FIELD_INIT_AMOUNT: amount,
                FIELD_CAP: cap,
                FIELD_FINAL_AMOUNT: 0.0,
                FIELD_RATE: rate if amount >= 10 else 0.0,
                FIELD_LOSS: loss if amount >= 10 else 0.0,
                "冲击额度折减系数": discount,
            }
        )

    priority = {"低风险": 0, "较低风险": 1, "中风险": 2, "高风险": 3}
    eligible = [r for r in rows if r[FIELD_REJECT] == NO and r[FIELD_INIT_AMOUNT] >= 10]
    eligible.sort(key=lambda r: (priority[r[FIELD_RISK]], r[FIELD_HIGH_POOL] == YES, -r[FIELD_SCORE]))

    deployable_credit_wan = total_credit_wan * 0.90 if scenario == "severe" else total_credit_wan
    remaining = deployable_credit_wan
    for row in eligible:
        if remaining < 10:
            row[FIELD_LOAN_STATUS] = "资金不足未放贷"
            continue
        amount = min(row[FIELD_INIT_AMOUNT], remaining)
        if amount < 10:
            row[FIELD_LOAN_STATUS] = "资金不足未放贷"
            continue
        row[FIELD_FINAL_AMOUNT] = amount
        row[FIELD_LOAN_STATUS] = "放贷"
        remaining -= amount

    if scenario != "severe" and remaining >= 0.01:
        for row in eligible:
            if row[FIELD_LOAN_STATUS] != "放贷":
                continue
            room = row[FIELD_CAP] - row[FIELD_FINAL_AMOUNT]
            if room <= 0:
                continue
            add = min(room, remaining)
            row[FIELD_FINAL_AMOUNT] += add
            remaining -= add
            if remaining < 0.01:
                break
    actual_allocated = deployable_credit_wan - remaining
    total_remaining = total_credit_wan - actual_allocated
    return rows, total_remaining


def risk_rank(level):
    return {level: idx for idx, level in enumerate(RISK_LEVELS)}.get(level, 99)


def compare_summary(baseline_rows, shock_rows, total_credit_wan, remaining):
    baseline_by_code = {row[FIELD_CODE]: row for row in baseline_rows}
    lent = [r for r in shock_rows if r[FIELD_LOAN_STATUS] == "放贷"]
    amount_sum = sum(r[FIELD_FINAL_AMOUNT] for r in lent)
    avg_rate = safe_ratio(sum(r[FIELD_FINAL_AMOUNT] * r[FIELD_RATE] for r in lent), amount_sum)
    baseline_lent = [r for r in baseline_rows if r[FIELD_LOAN_STATUS] == "放贷"]
    baseline_amount = sum(r[FIELD_FINAL_AMOUNT] for r in baseline_lent)
    baseline_rate = safe_ratio(sum(r[FIELD_FINAL_AMOUNT] * r[FIELD_RATE] for r in baseline_lent), baseline_amount)

    score_drops = [baseline_by_code[r[FIELD_CODE]][FIELD_SCORE] - r[FIELD_SCORE] for r in shock_rows]
    up_count = sum(risk_rank(r[FIELD_RISK]) > risk_rank(baseline_by_code[r[FIELD_CODE]][FIELD_RISK]) for r in shock_rows)
    new_high_pool = sum(
        r[FIELD_HIGH_POOL] == YES and baseline_by_code[r[FIELD_CODE]][FIELD_HIGH_POOL] == NO for r in shock_rows
    )
    new_reject = sum(r[FIELD_REJECT] == YES and baseline_by_code[r[FIELD_CODE]][FIELD_REJECT] == NO for r in shock_rows)
    amount_cut = sum(max(0.0, baseline_by_code[r[FIELD_CODE]][FIELD_FINAL_AMOUNT] - r[FIELD_FINAL_AMOUNT]) for r in shock_rows)

    row = {
        "年度信贷总额_万元": total_credit_wan,
        "实际放贷金额_万元": amount_sum,
        "剩余额度_万元": remaining,
        "放贷企业数": len(lent),
        "平均信用得分下降": safe_ratio(sum(score_drops), len(score_drops)),
        "风险等级上调企业数": up_count,
        "新增高风险池企业数": new_high_pool,
        "新增一票否决企业数": new_reject,
        "额度削减总额_万元": amount_cut,
        "加权平均利率": avg_rate,
        "基准放贷金额_万元": baseline_amount,
        "基准放贷企业数": len(baseline_lent),
        "基准加权平均利率": baseline_rate,
        "是否用满额度": YES if remaining < 0.01 else NO,
    }
    for level in RISK_LEVELS:
        row[f"{level}放贷数"] = sum(r[FIELD_LOAN_STATUS] == "放贷" and r[FIELD_RISK] == level for r in shock_rows)
    return row


def to_number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def load_csv_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return [{key: to_number(value) for key, value in row.items()} for row in csv.DictReader(fh)]


def load_baseline_inputs():
    q1_features = load_csv_rows(OUTPUT_DIR / "q1_enterprise_features.csv")
    q2_score_rows = load_csv_rows(OUTPUT_DIR / "q2_scorecard.csv")
    q2_strategy_rows = load_csv_rows(OUTPUT_DIR / "q2_credit_strategy.csv")

    strategy_by_code = {row[FIELD_CODE]: row for row in q2_strategy_rows}
    baseline_rows = []
    for row in q2_score_rows:
        merged = copy.deepcopy(row)
        merged.update(strategy_by_code.get(row[FIELD_CODE], {}))
        baseline_rows.append(merged)
    return q1_features, q2_score_rows, baseline_rows


def write_summary_md(path, row, risk_counts, industry_counts):
    lines = [
        "# 第三问执行摘要",
        "",
        f"- 年度信贷总额：{row['年度信贷总额_万元']:.2f} 万元",
        f"- 实际放贷金额：{row['实际放贷金额_万元']:.2f} 万元",
        f"- 剩余额度：{row['剩余额度_万元']:.2f} 万元",
        f"- 放贷企业数：{row['放贷企业数']} 家",
        f"- 平均信用得分下降：{row['平均信用得分下降']:.2f}",
        f"- 风险等级上调企业数：{row['风险等级上调企业数']} 家",
        f"- 新增高风险池企业数：{row['新增高风险池企业数']} 家",
        f"- 新增一票否决企业数：{row['新增一票否决企业数']} 家",
        f"- 额度削减总额：{row['额度削减总额_万元']:.2f} 万元",
        f"- 加权平均利率：{row['加权平均利率']:.4%}",
        "",
        "## 冲击后风险等级分布",
        "",
    ]
    for level in RISK_LEVELS:
        lines.append(f"- {level}：{risk_counts[level]} 家")
    lines.extend(["", "## 行业冲击等级分布", ""])
    for level in ["低冲击", "中冲击", "中高冲击", "高冲击"]:
        lines.append(f"- {level}：{industry_counts[level]} 家")
    path.write_text("\n".join(lines), encoding="utf-8-sig")


def write_q3_charts(rows, sensitivity_rows):
    chart_paths = list(write_strategy_charts("q3", "第三问", rows, sensitivity_rows))
    chart_dir = OUTPUT_DIR / "charts"
    scenario_label = {"mild": "温和", "base": "基准", "severe": "严重"}

    amount_items = [
        (scenario_label.get(row["情景"], row["情景"]), float(row["实际放贷金额_万元"]))
        for row in sensitivity_rows
    ]
    remaining_items = [
        (scenario_label.get(row["情景"], row["情景"]), float(row["剩余额度_万元"]))
        for row in sensitivity_rows
    ]
    lent_items = [
        (scenario_label.get(row["情景"], row["情景"]), float(row["放贷企业数"]))
        for row in sensitivity_rows
    ]
    high_pool_items = [
        (scenario_label.get(row["情景"], row["情景"]), float(row["新增高风险池企业数"]))
        for row in sensitivity_rows
    ]
    decision_counts = Counter(row[FIELD_LOAN_STATUS] for row in rows)
    decision_order = [
        "放贷",
        "资金不足未放贷",
        "暂缓授信",
        "高冲击高风险池暂停授信",
        "不放贷",
    ]
    decision_items = [(label, decision_counts[label]) for label in decision_order if decision_counts[label] > 0]
    decision_items.extend(
        (label, count)
        for label, count in decision_counts.items()
        if label not in decision_order and count > 0
    )
    shock_counts = Counter(row["冲击等级"] for row in rows)
    shock_items = [(level, shock_counts[level]) for level in ["低冲击", "中冲击", "中高冲击", "高冲击"]]

    extra_outputs = [
        chart_dir / "q3_scenario_actual_amount.svg",
        chart_dir / "q3_scenario_remaining_amount.svg",
        chart_dir / "q3_scenario_lent_count.svg",
        chart_dir / "q3_scenario_new_high_pool.svg",
        chart_dir / "q3_shock_level_distribution.svg",
    ]
    write_bar_chart_svg(chart_dir / "q3_decision_counts.svg", "第三问：授信决策分布", decision_items, unit="家")
    write_bar_chart_svg(extra_outputs[0], "第三问：不同冲击情景实际放贷金额", amount_items, unit="万元")
    write_bar_chart_svg(extra_outputs[1], "第三问：不同冲击情景剩余额度", remaining_items, unit="万元")
    write_bar_chart_svg(extra_outputs[2], "第三问：不同冲击情景放贷企业数", lent_items, unit="家")
    write_bar_chart_svg(extra_outputs[3], "第三问：不同冲击情景新增高风险池企业数", high_pool_items, unit="家")
    write_bar_chart_svg(extra_outputs[4], "第三问：基准情景行业冲击等级分布", shock_items, unit="家")
    chart_paths.extend(extra_outputs)
    return chart_paths


def run_shock_scenario(q2_features, baseline_rows, q1_thresholds, total_credit_wan, scenario):
    baseline_by_code = {row[FIELD_CODE]: row for row in baseline_rows}
    shocked_features = apply_shock(q2_features, baseline_by_code, scenario)
    shocked_scored = score_q2(shocked_features, q1_thresholds)
    shocked_scored = add_q3_high_pool_rules(shocked_scored)
    rows, remaining = assign_strategy_q3(shocked_scored, total_credit_wan, scenario)
    summary = compare_summary(baseline_rows, rows, total_credit_wan, remaining)
    summary["情景"] = scenario
    return rows, remaining, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-credit-wan", type=float, default=10000.0)
    parser.add_argument("--scenario", choices=sorted(SCENARIO_FACTOR), default="base")
    args = parser.parse_args()

    q1_features, q2_features, baseline_rows = load_baseline_inputs()
    q1_thresholds = compute_thresholds(q1_features)

    rows, remaining, summary = run_shock_scenario(
        q2_features, baseline_rows, q1_thresholds, args.total_credit_wan, args.scenario
    )
    sensitivity_rows = []
    for scenario in ["mild", "base", "severe"]:
        _, _, scenario_summary = run_shock_scenario(
            q2_features, baseline_rows, q1_thresholds, args.total_credit_wan, scenario
        )
        sensitivity_rows.append(scenario_summary)

    risk_counts = Counter(r[FIELD_RISK] for r in rows)
    shock_counts = Counter(r["冲击等级"] for r in rows)

    detail_fields = [
        FIELD_CODE,
        FIELD_NAME,
        "行业类别",
        "冲击等级",
        "情景",
        "脆弱性修正系数",
        "脆弱性原因",
        "收入下降率",
        "毛利率下降幅度",
        "波动系数上升率",
        "经营月份减少数",
        "调整前信用得分",
        FIELD_SCORE,
        "调整前风险等级",
        FIELD_RISK,
        "调整前贷款额度_万元",
        FIELD_FINAL_AMOUNT,
        "调整前贷款年利率",
        FIELD_RATE,
    ]
    q3_score_fields = ["行业类别", "冲击等级", "脆弱性修正系数"] + score_fields(False)
    q3_strategy_fields = ["行业类别", "冲击等级", "脆弱性修正系数", "冲击额度折减系数"] + [
        FIELD_CODE,
        FIELD_NAME,
        FIELD_CLASS,
        FIELD_SCORE,
        FIELD_RISK,
        FIELD_REJECT,
        FIELD_REJECT_REASON,
        FIELD_HIGH_POOL,
        FIELD_HIGH_REASON,
        FIELD_LOAN_STATUS,
        FIELD_INIT_AMOUNT,
        FIELD_CAP,
        FIELD_FINAL_AMOUNT,
        FIELD_RATE,
        FIELD_LOSS,
    ]
    summary_fields = list(summary.keys())

    write_csv(OUTPUT_DIR / "q3_shock_detail.csv", rows, detail_fields)
    write_csv(OUTPUT_DIR / "q3_scorecard.csv", rows, q3_score_fields)
    write_csv(OUTPUT_DIR / "q3_credit_strategy.csv", rows, q3_strategy_fields)
    write_csv(OUTPUT_DIR / "q3_comparison_summary.csv", [summary], summary_fields)
    write_csv(OUTPUT_DIR / "q3_sensitivity.csv", sensitivity_rows, ["情景"] + [f for f in summary_fields if f != "情景"])
    write_summary_md(OUTPUT_DIR / "q3_summary.md", summary, risk_counts, shock_counts)
    chart_paths = write_q3_charts(rows, sensitivity_rows)

    print(f"enterprise_count={len(rows)}")
    print(f"output_dir={OUTPUT_DIR}")
    print("charts=" + ",".join(str(path) for path in chart_paths))
    print(f"remaining_credit_wan={remaining:.2f}")


if __name__ == "__main__":
    main()
