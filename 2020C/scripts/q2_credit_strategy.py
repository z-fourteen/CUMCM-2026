import argparse
import copy
from collections import Counter

from credit_common import (
    OUTPUT_DIR,
    RISK_LEVELS,
    YES,
    assign_strategy,
    budget_fields,
    budget_sensitivity,
    compute_thresholds,
    feature_fields,
    load_feature_rows,
    score_fields,
    score_q2,
    strategy_fields,
    write_strategy_charts,
    write_csv,
)


def write_budget_md(path, rows):
    lines = [
        "# 第二问年度信贷总额敏感性测试",
        "",
        "| 年度总额(万元) | 实际放贷(万元) | 剩余(万元) | 放贷企业数 | 是否用满 | 加权平均利率 |",
        "| ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['年度信贷总额_万元']:.0f} | {row['实际放贷金额_万元']:.2f} | "
            f"{row['剩余额度_万元']:.2f} | {row['放贷企业数']} | {row['是否用满额度']} | "
            f"{row['加权平均利率']:.4%} |"
        )
    lines.append("")
    lines.append("第二问题设年度信贷总额为 10000 万元；敏感性测试用于观察资金规模变化下的可执行性。")
    path.write_text("\n".join(lines), encoding="utf-8-sig")


def write_summary(path, rows, budget_rows, total_credit_wan, remaining, q1_thresholds):
    lent = [r for r in rows if r["是否放贷"] == "放贷"]
    rejected = [r for r in rows if r["是否一票否决"] == YES]
    high_pool = [r for r in rows if r["是否高风险池"] == YES]
    amount_sum = sum(r["最终贷款额度_万元"] for r in lent)
    avg_rate = 0.0 if amount_sum == 0 else sum(r["最终贷款额度_万元"] * r["建议贷款年利率"] for r in lent) / amount_sum
    risk_counts = Counter(r["风险等级"] for r in rows)
    class_counts = Counter(r["估计信用等级"] for r in rows)

    lines = [
        "# 第二问执行摘要",
        "",
        f"- 年度信贷总额：{total_credit_wan:.2f} 万元",
        f"- 实际分配额度：{amount_sum:.2f} 万元",
        f"- 剩余额度：{remaining:.2f} 万元",
        f"- 放贷企业数：{len(lent)}",
        f"- 一票否决企业数：{len(rejected)}",
        f"- 高风险池企业数：{len(high_pool)}，该字段只作为过程标记，不作为最终风险等级",
        f"- 加权平均利率：{avg_rate:.4%}",
        "",
        "## 最终风险等级分布",
        "",
    ]
    for level in RISK_LEVELS:
        lines.append(f"- {level}：{risk_counts[level]} 家")

    lines.extend(["", "## 估计信用等级分布", ""])
    for level in ["类A", "类B", "类C", "类D"]:
        lines.append(f"- {level}：{class_counts[level]} 家")

    lines.extend(
        [
            "",
            "## 迁移自第一问的主要阈值",
            "",
            f"- 有效销项金额 5% 分位：{q1_thresholds['revenue_p05']:.2f}",
            f"- 有效销项金额 10% 分位：{q1_thresholds['revenue_p10']:.2f}",
            f"- 有效销项金额 90% 分位：{q1_thresholds['revenue_p90']:.2f}",
            "",
            "## 预算敏感性结论",
            "",
        ]
    )
    for row in budget_rows:
        lines.append(
            f"- 总额{row['年度信贷总额_万元']:.0f}万元：实际放贷{row['实际放贷金额_万元']:.2f}万元，"
            f"放贷{row['放贷企业数']}家，是否用满：{row['是否用满额度']}"
        )
    path.write_text("\n".join(lines), encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-credit-wan", type=float, default=10000.0)
    parser.add_argument("--budgets", default="5000,8000,10000,12000,15000,20000")
    args = parser.parse_args()

    q1_features = load_feature_rows("*123*.xlsx", has_credit_record=True)
    q1_thresholds = compute_thresholds(q1_features)

    q2_features = load_feature_rows("*302*.xlsx", has_credit_record=False)
    scored_rows = score_q2(q2_features, q1_thresholds)

    rows = copy.deepcopy(scored_rows)
    rows, remaining = assign_strategy(rows, args.total_credit_wan)
    budget_rows = budget_sensitivity(scored_rows, [float(x) for x in args.budgets.split(",") if x.strip()])

    write_csv(OUTPUT_DIR / "q2_enterprise_features.csv", rows, feature_fields(False))
    write_csv(OUTPUT_DIR / "q2_scorecard.csv", rows, score_fields(False))
    write_csv(OUTPUT_DIR / "q2_credit_strategy.csv", rows, strategy_fields(False))
    write_csv(OUTPUT_DIR / "q2_budget_sensitivity.csv", budget_rows, budget_fields())
    write_budget_md(OUTPUT_DIR / "q2_budget_sensitivity.md", budget_rows)
    write_summary(OUTPUT_DIR / "q2_summary.md", rows, budget_rows, args.total_credit_wan, remaining, q1_thresholds)
    chart_paths = write_strategy_charts("q2", "第二问", rows, budget_rows)

    print(f"enterprise_count={len(rows)}")
    print(f"output_dir={OUTPUT_DIR}")
    print("charts=" + ",".join(str(path) for path in chart_paths))
    print(f"remaining_credit_wan={remaining:.2f}")


if __name__ == "__main__":
    main()
