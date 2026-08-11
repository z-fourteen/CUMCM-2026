import copy
import csv
import html
import math
import re
import statistics
import zipfile
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from xml.etree import ElementTree as ET


WORK_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = WORK_DIR.parent
RESOURCE_DIR = ROOT_DIR / "resource" / "2020C"
OUTPUT_DIR = WORK_DIR / "outputs"

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS = {"main": MAIN_NS}

YES = "是"
NO = "否"
VALID_INVOICE = "有效发票"
VOID_INVOICE = "作废发票"
RISK_LEVELS = ["低风险", "较低风险", "中风险", "高风险"]


def excel_date_to_month(serial):
    try:
        value = float(serial)
    except (TypeError, ValueError):
        return ""
    day = datetime(1899, 12, 30) + timedelta(days=value)
    return day.strftime("%Y-%m")


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def safe_ratio(num, den, default=0.0):
    return default if den == 0 else num / den


def clip(value, low, high):
    return max(low, min(high, value))


def percentile(values, p):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    pos = (len(ordered) - 1) * p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def positive_score(value, low, high):
    if high <= low:
        return 50.0
    return clip((value - low) / (high - low) * 100, 0, 100)


def negative_score(value, low, high):
    if high <= low:
        return 50.0
    return clip((high - value) / (high - low) * 100, 0, 100)


def bounded_score(value, good_low, good_high, hard_low, hard_high):
    if good_low <= value <= good_high:
        return 100.0
    if value < good_low:
        return positive_score(value, hard_low, good_low)
    return negative_score(value, good_high, hard_high)


def col_index(cell_ref):
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    idx = 0
    for ch in letters:
        idx = idx * 26 + ord(ch.upper()) - 64
    return idx - 1


def shared_strings(zf):
    values = []
    if "xl/sharedStrings.xml" not in zf.namelist():
        return values
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    for si in root.findall(".//main:si", NS):
        values.append("".join(t.text or "" for t in si.findall(".//main:t", NS)))
    return values


def workbook_sheets(zf):
    wb = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rid_to_target = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
    result = {}
    for sheet in wb.findall(".//main:sheets/main:sheet", NS):
        name = sheet.attrib["name"]
        rid = sheet.attrib[f"{{{REL_NS}}}id"]
        result[name] = "xl/" + rid_to_target[rid].replace("\\", "/")
    return result


def cell_value(cell, strings):
    kind = cell.attrib.get("t")
    value = cell.find("main:v", NS)
    inline = cell.find("main:is", NS)
    if kind == "s" and value is not None and value.text is not None:
        idx = int(value.text)
        return strings[idx] if idx < len(strings) else value.text
    if kind == "inlineStr" and inline is not None:
        return "".join(t.text or "" for t in inline.findall(".//main:t", NS))
    return value.text if value is not None and value.text is not None else ""


def iter_rows(zf, target, strings):
    with zf.open(target) as fh:
        for _, elem in ET.iterparse(fh, events=("end",)):
            if not elem.tag.endswith("}row"):
                continue
            cells = {}
            for cell in elem.findall("main:c", NS):
                cells[col_index(cell.attrib.get("r", "A"))] = cell_value(cell, strings)
            if cells:
                max_col = max(cells)
                yield [cells.get(i, "") for i in range(max_col + 1)]
            elem.clear()


def find_attachment(pattern):
    matches = sorted(RESOURCE_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"未找到附件：{RESOURCE_DIR / pattern}")
    return matches[0]


def enterprise_sort_key(row):
    return int(re.sub(r"\D", "", row["企业代号"]) or 0)


def init_stats(info):
    return {
        "code": info["code"],
        "name": info["name"],
        "rating": info.get("rating", ""),
        "defaulted": info.get("defaulted", NO),
        "in_valid_amt": 0.0,
        "out_valid_amt": 0.0,
        "in_total_rows": 0,
        "out_total_rows": 0,
        "in_valid_rows": 0,
        "out_valid_rows": 0,
        "in_void_rows": 0,
        "out_void_rows": 0,
        "in_negative_rows": 0,
        "out_negative_rows": 0,
        "in_partners": Counter(),
        "out_partners": Counter(),
        "in_months": Counter(),
        "out_months": Counter(),
    }


def load_enterprises(zf, target, strings, has_credit_record):
    enterprises = {}
    for i, row in enumerate(iter_rows(zf, target, strings)):
        if i == 0 or len(row) < 2:
            continue
        code = row[0]
        enterprises[code] = {
            "code": code,
            "name": row[1],
            "rating": row[2] if has_credit_record and len(row) > 2 else "",
            "defaulted": row[3] if has_credit_record and len(row) > 3 else NO,
        }
    return enterprises


def aggregate_invoice(zf, target, strings, stats_by_code, kind):
    prefix = "in" if kind == "input" else "out"
    for i, row in enumerate(iter_rows(zf, target, strings)):
        if i == 0 or len(row) < 8:
            continue
        code = row[0]
        if code not in stats_by_code:
            continue
        stats = stats_by_code[code]
        amount = to_float(row[4])
        status = str(row[7] or "").strip()
        month = excel_date_to_month(row[2])
        partner = row[3]

        stats[f"{prefix}_total_rows"] += 1
        if status == VOID_INVOICE:
            stats[f"{prefix}_void_rows"] += 1
        if amount < 0:
            stats[f"{prefix}_negative_rows"] += 1
        if status == VALID_INVOICE:
            stats[f"{prefix}_valid_rows"] += 1
            stats[f"{prefix}_valid_amt"] += amount
            if partner:
                stats[f"{prefix}_partners"][partner] += abs(amount)
            if month:
                stats[f"{prefix}_months"][month] += amount


def hhi(counter):
    total = sum(abs(v) for v in counter.values())
    if total <= 0:
        return 1.0
    return sum((abs(v) / total) ** 2 for v in counter.values())


def longest_gap(month_counter):
    months = sorted(month_counter)
    if not months:
        return 999
    parsed = [datetime.strptime(m, "%Y-%m") for m in months]
    gaps = []
    for prev, curr in zip(parsed, parsed[1:]):
        gaps.append((curr.year - prev.year) * 12 + curr.month - prev.month - 1)
    return max(gaps) if gaps else 0


def coeff_var(values):
    values = [v for v in values if v > 0]
    if len(values) <= 1:
        return 1.0
    mean = statistics.mean(values)
    return 1.0 if mean == 0 else statistics.pstdev(values) / mean


def make_features(stats_by_code, include_credit_fields):
    rows = []
    for stats in stats_by_code.values():
        out_amt = stats["out_valid_amt"]
        in_amt = stats["in_valid_amt"]
        out_month_count = len(stats["out_months"])
        active_month_count = len(set(stats["out_months"]) | set(stats["in_months"]))
        row = {
            "企业代号": stats["code"],
            "企业名称": stats["name"],
            "有效销项金额": out_amt,
            "有效进项金额": in_amt,
            "月均销项收入": safe_ratio(out_amt, max(out_month_count, 1)),
            "毛利率": safe_ratio(out_amt - in_amt, out_amt, default=-1.0),
            "有效销项发票数": stats["out_valid_rows"],
            "有效进项发票数": stats["in_valid_rows"],
            "销项作废率": safe_ratio(stats["out_void_rows"], stats["out_total_rows"]),
            "进项作废率": safe_ratio(stats["in_void_rows"], stats["in_total_rows"]),
            "销项负数率": safe_ratio(stats["out_negative_rows"], stats["out_total_rows"]),
            "进项负数率": safe_ratio(stats["in_negative_rows"], stats["in_total_rows"]),
            "销项交易月份数": out_month_count,
            "进项交易月份数": len(stats["in_months"]),
            "有效经营月份数": active_month_count,
            "最长销项断档月数": longest_gap(stats["out_months"]),
            "销项收入波动系数": coeff_var(list(stats["out_months"].values())),
            "客户数": len(stats["out_partners"]),
            "供应商数": len(stats["in_partners"]),
            "客户集中度HHI": hhi(stats["out_partners"]),
            "供应商集中度HHI": hhi(stats["in_partners"]),
            "进销匹配差异": abs(safe_ratio(in_amt, out_amt, default=10.0) - 1.0),
        }
        if include_credit_fields:
            row["信誉评级"] = stats["rating"]
            row["是否违约"] = stats["defaulted"]
        rows.append(row)
    rows.sort(key=enterprise_sort_key)
    return rows


def load_feature_rows(attachment_pattern, has_credit_record):
    zf = zipfile.ZipFile(find_attachment(attachment_pattern))
    strings = shared_strings(zf)
    sheets = workbook_sheets(zf)
    enterprises = load_enterprises(zf, sheets["企业信息"], strings, has_credit_record)
    stats_by_code = {code: init_stats(info) for code, info in enterprises.items()}
    aggregate_invoice(zf, sheets["进项发票信息"], strings, stats_by_code, "input")
    aggregate_invoice(zf, sheets["销项发票信息"], strings, stats_by_code, "output")
    return make_features(stats_by_code, has_credit_record)


def compute_thresholds(rows):
    valid_revenues = [r["有效销项金额"] for r in rows if r["有效销项金额"] > 0]
    monthly_revenues = [r["月均销项收入"] for r in rows if r["月均销项收入"] > 0]
    customer_counts = [r["客户数"] for r in rows]
    supplier_counts = [r["供应商数"] for r in rows]
    return {
        "revenue_p05": percentile(valid_revenues, 0.05),
        "revenue_p10": percentile(valid_revenues, 0.10),
        "revenue_p90": percentile(valid_revenues, 0.90),
        "monthly_p10": percentile(monthly_revenues, 0.10),
        "monthly_p90": percentile(monthly_revenues, 0.90),
        "customer_p10": percentile(customer_counts, 0.10),
        "customer_p90": percentile(customer_counts, 0.90),
        "supplier_p10": percentile(supplier_counts, 0.10),
        "supplier_p90": percentile(supplier_counts, 0.90),
    }


def score_operating_dimensions(row, thresholds):
    max_void_rate = max(row["销项作废率"], row["进项作废率"])
    max_negative_rate = max(row["销项负数率"], row["进项负数率"])

    revenue_score = positive_score(row["有效销项金额"], thresholds["revenue_p10"], thresholds["revenue_p90"])
    monthly_score = positive_score(row["月均销项收入"], thresholds["monthly_p10"], thresholds["monthly_p90"])
    margin_score = bounded_score(row["毛利率"], 0.05, 0.45, -0.30, 1.00)
    scale_score = 0.45 * revenue_score + 0.25 * monthly_score + 0.30 * margin_score

    month_score = positive_score(row["有效经营月份数"], 6, 36)
    gap_score = negative_score(row["最长销项断档月数"], 0, 12)
    vol_score = negative_score(row["销项收入波动系数"], 0.2, 2.0)
    continuity_score = 0.45 * month_score + 0.30 * gap_score + 0.25 * vol_score

    void_score = negative_score(max_void_rate, 0.00, 0.25)
    neg_score = negative_score(max_negative_rate, 0.00, 0.20)
    match_score = negative_score(row["进销匹配差异"], 0.00, 2.00)
    authenticity_score = 0.35 * void_score + 0.30 * neg_score + 0.35 * match_score

    customer_score = positive_score(row["客户数"], thresholds["customer_p10"], thresholds["customer_p90"])
    supplier_score = positive_score(row["供应商数"], thresholds["supplier_p10"], thresholds["supplier_p90"])
    customer_hhi_score = negative_score(row["客户集中度HHI"], 0.02, 0.80)
    supplier_hhi_score = negative_score(row["供应商集中度HHI"], 0.02, 0.80)
    resilience_score = 0.25 * (customer_score + supplier_score + customer_hhi_score + supplier_hhi_score)

    row.update(
        {
            "最大作废率": max_void_rate,
            "最大负数率": max_negative_rate,
            "经营规模分": scale_score,
            "经营连续性分": continuity_score,
            "交易真实性分": authenticity_score,
            "抗风险能力分": resilience_score,
        }
    )
    return row


def risk_level_from_score(score):
    if score >= 80:
        return "低风险"
    if score >= 65:
        return "较低风险"
    if score >= 50:
        return "中风险"
    return "高风险"


def level_basis(row):
    return (
        f"信用得分={row['信用得分']:.2f}，有效销项金额={row['有效销项金额']:.2f}，"
        f"毛利率={row['毛利率']:.2%}，有效经营月份={row['有效经营月份数']}，"
        f"最大作废率={row['最大作废率']:.2%}，最大负数率={row['最大负数率']:.2%}，"
        f"进销匹配差异={row['进销匹配差异']:.2f}，"
        f"客户HHI={row['客户集中度HHI']:.3f}，供应商HHI={row['供应商集中度HHI']:.3f}"
    )


def reject_reasons_for_row(row, thresholds, allow_rating_reject):
    reasons = []
    if allow_rating_reject and row.get("信誉评级") == "D":
        reasons.append("信誉评级D")
    if row["有效销项发票数"] == 0 or row["有效销项金额"] <= 0:
        reasons.append("无正常销项交易")
    elif row["有效销项金额"] <= thresholds["revenue_p05"]:
        reasons.append(f"有效销项金额{row['有效销项金额']:.2f} <= 5%分位{thresholds['revenue_p05']:.2f}")
    if row["有效经营月份数"] < 6:
        reasons.append(f"有效经营月份数{row['有效经营月份数']} < 6")
    if row["最大作废率"] > 0.40:
        reasons.append(f"最大作废率{row['最大作废率']:.2%} > 40%")
    if row["最大负数率"] > 0.30:
        reasons.append(f"最大负数率{row['最大负数率']:.2%} > 30%")
    if row["有效销项金额"] > 0 and row["进销匹配差异"] > 4.0:
        reasons.append(f"进销匹配差异{row['进销匹配差异']:.2f} > 4")
    return reasons


def high_pool_reasons_for_row(row):
    reasons = []
    if row.get("是否违约") == YES:
        reasons.append("历史违约记录")
    if row["信用得分"] < 50:
        reasons.append(f"信用得分{row['信用得分']:.2f} < 50")
    if row["最大作废率"] > 0.25:
        reasons.append(f"最大作废率{row['最大作废率']:.2%} > 25%")
    if row["最大负数率"] > 0.15:
        reasons.append(f"最大负数率{row['最大负数率']:.2%} > 15%")
    if row["进销匹配差异"] > 2.5:
        reasons.append(f"进销匹配差异{row['进销匹配差异']:.2f} > 2.5")
    return reasons


def score_q1(rows, thresholds):
    for row in rows:
        score_operating_dimensions(row, thresholds)
        rating_base = {"A": 100.0, "B": 80.0, "C": 60.0, "D": 0.0}.get(row["信誉评级"], 50.0)
        credit_base_score = clip(rating_base - (25.0 if row["是否违约"] == YES else 0.0), 0, 100)
        total_score = (
            0.25 * row["经营规模分"]
            + 0.20 * row["经营连续性分"]
            + 0.25 * row["交易真实性分"]
            + 0.20 * row["抗风险能力分"]
            + 0.10 * credit_base_score
        )
        risk_level = risk_level_from_score(total_score)
        reject_reasons = reject_reasons_for_row(row, thresholds, allow_rating_reject=True)
        if row["是否违约"] == YES and not reject_reasons:
            risk_level = "高风险"
        row["信用基础分"] = credit_base_score
        row["信用得分"] = total_score
        row["风险得分"] = 100 - total_score
        row["风险等级"] = risk_level
        high_pool_reasons = [] if reject_reasons else high_pool_reasons_for_row(row)
        row["是否一票否决"] = YES if reject_reasons else NO
        row["一票否决原因"] = "；".join(reject_reasons)
        row["是否高风险池"] = YES if high_pool_reasons else NO
        row["高风险池原因"] = "；".join(high_pool_reasons)
        row["分级依据指标"] = level_basis(row)
    return rows


def estimate_credit_classes(rows):
    preliminary = []
    for row in rows:
        pre_score = (
            0.30 * row["经营规模分"]
            + 0.25 * row["经营连续性分"]
            + 0.25 * row["交易真实性分"]
            + 0.20 * row["抗风险能力分"]
        )
        row["经营综合预评分"] = pre_score
        preliminary.append(pre_score)
    p25 = percentile(preliminary, 0.25)
    p60 = percentile(preliminary, 0.60)
    p85 = percentile(preliminary, 0.85)
    for row in rows:
        score = row["经营综合预评分"]
        if score >= p85:
            row["估计信用等级"] = "类A"
            row["信用基础分"] = 90.0
        elif score >= p60:
            row["估计信用等级"] = "类B"
            row["信用基础分"] = 75.0
        elif score >= p25:
            row["估计信用等级"] = "类C"
            row["信用基础分"] = 60.0
        else:
            row["估计信用等级"] = "类D"
            row["信用基础分"] = 30.0


def score_q2(rows, thresholds):
    for row in rows:
        score_operating_dimensions(row, thresholds)
    estimate_credit_classes(rows)
    for row in rows:
        total_score = (
            0.25 * row["经营规模分"]
            + 0.20 * row["经营连续性分"]
            + 0.25 * row["交易真实性分"]
            + 0.20 * row["抗风险能力分"]
            + 0.10 * row["信用基础分"]
        )
        row["信用得分"] = total_score
        row["风险得分"] = 100 - total_score
        row["风险等级"] = risk_level_from_score(total_score)
        reject_reasons = reject_reasons_for_row(row, thresholds, allow_rating_reject=False)
        high_pool_reasons = [] if reject_reasons else high_pool_reasons_for_row(row)
        row["是否一票否决"] = YES if reject_reasons else NO
        row["一票否决原因"] = "；".join(reject_reasons)
        row["是否高风险池"] = YES if high_pool_reasons else NO
        row["高风险池原因"] = "；".join(high_pool_reasons)
        row["分级依据指标"] = level_basis(row)
    return rows


def load_rate_table():
    zf = zipfile.ZipFile(find_attachment("*利率*.xlsx"))
    strings = shared_strings(zf)
    target = next(iter(workbook_sheets(zf).values()))
    rates = []
    for i, row in enumerate(iter_rows(zf, target, strings)):
        if i < 2 or len(row) < 4:
            continue
        rate = to_float(row[0])
        if rate > 0:
            rates.append({"rate": rate, "A": to_float(row[1]), "B": to_float(row[2]), "C": to_float(row[3])})
    return rates


def rate_grade(row):
    if row.get("信誉评级") in {"A", "B", "C"}:
        return row["信誉评级"]
    return {"类A": "A", "类B": "B", "类C": "C", "类D": "C"}.get(row.get("估计信用等级"), "C")


def select_rate(row, rates):
    if row["是否一票否决"] == YES:
        return 0.0, 0.0
    intervals = {
        "低风险": (0.04, 0.07),
        "较低风险": (0.07, 0.10),
        "中风险": (0.10, 0.12),
        "高风险": (0.12, 0.15),
    }
    lo, hi = intervals[row["风险等级"]]
    grade = rate_grade(row)
    candidates = [r for r in rates if lo <= r["rate"] <= hi]
    if not candidates:
        return (lo + hi) / 2, 0.0
    best = max(candidates, key=lambda r: r["rate"] * (1 - r[grade]))
    return best["rate"], best[grade]


def assign_strategy(rows, total_credit_wan):
    rates = load_rate_table()
    for row in rows:
        if row["是否一票否决"] == YES:
            row.update(
                {
                    "是否放贷": "不放贷",
                    "初始建议额度_万元": 0.0,
                    "额度上限_万元": 0.0,
                    "最终贷款额度_万元": 0.0,
                    "建议贷款年利率": 0.0,
                    "客户流失率": 0.0,
                }
            )
            continue

        raw_amount = 10.0 + 90.0 * row["信用得分"] / 100.0
        if row["风险等级"] == "低风险":
            amount, cap = clip(raw_amount, 70, 100), 100.0
        elif row["风险等级"] == "较低风险":
            amount, cap = clip(raw_amount, 40, 80), 80.0
        elif row["风险等级"] == "中风险":
            amount, cap = clip(raw_amount, 10, 50), 50.0
        else:
            amount, cap = clip(raw_amount, 10, 30), 30.0

        if row["是否高风险池"] == YES:
            amount, cap = min(amount, 30.0), min(cap, 30.0)

        rate, loss = select_rate(row, rates)
        row.update(
            {
                "是否放贷": "拟放贷",
                "初始建议额度_万元": amount,
                "额度上限_万元": cap,
                "最终贷款额度_万元": 0.0,
                "建议贷款年利率": rate,
                "客户流失率": loss,
            }
        )

    priority = {"低风险": 0, "较低风险": 1, "中风险": 2, "高风险": 3}
    eligible = [r for r in rows if r["是否一票否决"] == NO]
    eligible.sort(key=lambda r: (priority[r["风险等级"]], r["是否高风险池"] == YES, -r["信用得分"]))

    remaining = total_credit_wan
    for row in eligible:
        if remaining < 10:
            row["是否放贷"] = "资金不足未放贷"
            continue
        amount = min(row["初始建议额度_万元"], remaining)
        if amount < 10:
            row["是否放贷"] = "资金不足未放贷"
            continue
        row["最终贷款额度_万元"] = amount
        row["是否放贷"] = "放贷"
        remaining -= amount

    if remaining >= 0.01:
        for row in eligible:
            if row["是否放贷"] != "放贷":
                continue
            room = row["额度上限_万元"] - row["最终贷款额度_万元"]
            if room <= 0:
                continue
            add = min(room, remaining)
            row["最终贷款额度_万元"] += add
            remaining -= add
            if remaining < 0.01:
                break
    return rows, remaining


def strategy_summary(rows, total_credit_wan, remaining):
    lent = [r for r in rows if r["是否放贷"] == "放贷"]
    amount_sum = sum(r["最终贷款额度_万元"] for r in lent)
    avg_rate = safe_ratio(sum(r["最终贷款额度_万元"] * r["建议贷款年利率"] for r in lent), amount_sum)
    result = {
        "年度信贷总额_万元": total_credit_wan,
        "实际放贷金额_万元": amount_sum,
        "剩余额度_万元": remaining,
        "放贷企业数": len(lent),
        "一票否决企业数": sum(r["是否一票否决"] == YES for r in rows),
        "高风险池企业数": sum(r["是否高风险池"] == YES for r in rows),
        "加权平均利率": avg_rate,
        "是否用满额度": YES if remaining < 0.01 else NO,
        "未用满原因": "" if remaining < 0.01 else "通过风控门槛企业的风险等级额度上限合计低于年度信贷总额",
    }
    for level in RISK_LEVELS:
        result[f"{level}放贷数"] = sum(r["是否放贷"] == "放贷" and r["风险等级"] == level for r in rows)
    return result


def budget_sensitivity(scored_rows, budgets):
    rows = []
    for budget in budgets:
        rows_copy = copy.deepcopy(scored_rows)
        rows_copy, remaining = assign_strategy(rows_copy, budget)
        rows.append(strategy_summary(rows_copy, budget, remaining))
    return rows


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            clean = {}
            for field in fields:
                value = row.get(field, "")
                clean[field] = f"{value:.6f}" if isinstance(value, float) else value
            writer.writerow(clean)


def feature_fields(has_credit_record):
    fields = ["企业代号", "企业名称"]
    if has_credit_record:
        fields += ["信誉评级", "是否违约"]
    else:
        fields += ["估计信用等级"]
    return fields + [
        "有效销项金额",
        "有效进项金额",
        "月均销项收入",
        "毛利率",
        "有效销项发票数",
        "有效进项发票数",
        "销项作废率",
        "进项作废率",
        "销项负数率",
        "进项负数率",
        "有效经营月份数",
        "最长销项断档月数",
        "销项收入波动系数",
        "客户数",
        "供应商数",
        "客户集中度HHI",
        "供应商集中度HHI",
        "进销匹配差异",
    ]


def score_fields(has_credit_record):
    fields = feature_fields(has_credit_record) + [
        "经营规模分",
        "经营连续性分",
        "交易真实性分",
        "抗风险能力分",
        "信用基础分",
        "信用得分",
        "风险得分",
        "最大作废率",
        "最大负数率",
        "是否一票否决",
        "一票否决原因",
        "是否高风险池",
        "高风险池原因",
        "风险等级",
        "分级依据指标",
    ]
    if not has_credit_record:
        fields.insert(fields.index("信用基础分"), "经营综合预评分")
    return fields


def strategy_fields(has_credit_record):
    fields = ["企业代号", "企业名称"]
    if has_credit_record:
        fields += ["信誉评级", "是否违约"]
    else:
        fields += ["估计信用等级"]
    return fields + [
        "信用得分",
        "风险等级",
        "分级依据指标",
        "是否一票否决",
        "一票否决原因",
        "是否高风险池",
        "高风险池原因",
        "是否放贷",
        "初始建议额度_万元",
        "额度上限_万元",
        "最终贷款额度_万元",
        "建议贷款年利率",
        "客户流失率",
    ]


def budget_fields():
    return [
        "年度信贷总额_万元",
        "实际放贷金额_万元",
        "剩余额度_万元",
        "放贷企业数",
        "一票否决企业数",
        "高风险池企业数",
        "低风险放贷数",
        "较低风险放贷数",
        "中风险放贷数",
        "高风险放贷数",
        "加权平均利率",
        "是否用满额度",
        "未用满原因",
    ]


def _fmt_num(value):
    if abs(value - round(value)) < 1e-6:
        return f"{value:.0f}"
    return f"{value:.2f}"


def _svg_text(x, y, text, size=14, anchor="middle", weight="400", fill="#1f2937"):
    escaped = html.escape(str(text))
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
        f'font-weight="{weight}" fill="{fill}" '
        f'font-family="Microsoft YaHei, SimHei, Arial, sans-serif">{escaped}</text>'
    )


def write_bar_chart_svg(path, title, items, unit="", width=920, height=520):
    path.parent.mkdir(parents=True, exist_ok=True)
    margin_left, margin_right, margin_top, margin_bottom = 100, 40, 70, 95
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom
    max_value = max([value for _, value in items] + [1])
    bar_gap = 18
    bar_w = max(18, (chart_w - bar_gap * (len(items) - 1)) / max(len(items), 1))
    palette = ["#2563eb", "#16a34a", "#f59e0b", "#dc2626", "#7c3aed", "#0891b2"]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        _svg_text(width / 2, 34, title, size=22, weight="700"),
        f'<line x1="{margin_left}" y1="{margin_top + chart_h}" x2="{width - margin_right}" y2="{margin_top + chart_h}" stroke="#9ca3af"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + chart_h}" stroke="#9ca3af"/>',
    ]
    for i in range(5):
        value = max_value * i / 4
        y = margin_top + chart_h - chart_h * i / 4
        parts.append(f'<line x1="{margin_left}" y1="{y}" x2="{width - margin_right}" y2="{y}" stroke="#e5e7eb"/>')
        parts.append(_svg_text(margin_left - 12, y + 5, _fmt_num(value), size=12, anchor="end", fill="#6b7280"))

    for idx, (label, value) in enumerate(items):
        x = margin_left + idx * (bar_w + bar_gap)
        bar_h = chart_h * value / max_value if max_value else 0
        y = margin_top + chart_h - bar_h
        color = palette[idx % len(palette)]
        parts.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{bar_h}" rx="4" fill="{color}"/>')
        parts.append(_svg_text(x + bar_w / 2, y - 8, f"{_fmt_num(value)}{unit}", size=13, weight="600", fill="#111827"))
        parts.append(_svg_text(x + bar_w / 2, margin_top + chart_h + 30, label, size=13, fill="#374151"))

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_budget_sensitivity_svg(path, title, rows, width=980, height=540):
    path.parent.mkdir(parents=True, exist_ok=True)
    margin_left, margin_right, margin_top, margin_bottom = 105, 45, 75, 85
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom
    budgets = [float(r["年度信贷总额_万元"]) for r in rows]
    actuals = [float(r["实际放贷金额_万元"]) for r in rows]
    max_value = max(budgets + actuals + [1])
    step = chart_w / max(len(rows) - 1, 1)

    def point(values, idx):
        x = margin_left + step * idx
        y = margin_top + chart_h - chart_h * values[idx] / max_value
        return x, y

    budget_points = [point(budgets, i) for i in range(len(rows))]
    actual_points = [point(actuals, i) for i in range(len(rows))]
    budget_path = " ".join(f"{x:.2f},{y:.2f}" for x, y in budget_points)
    actual_path = " ".join(f"{x:.2f},{y:.2f}" for x, y in actual_points)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        _svg_text(width / 2, 34, title, size=22, weight="700"),
        f'<line x1="{margin_left}" y1="{margin_top + chart_h}" x2="{width - margin_right}" y2="{margin_top + chart_h}" stroke="#9ca3af"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + chart_h}" stroke="#9ca3af"/>',
    ]
    for i in range(5):
        value = max_value * i / 4
        y = margin_top + chart_h - chart_h * i / 4
        parts.append(f'<line x1="{margin_left}" y1="{y}" x2="{width - margin_right}" y2="{y}" stroke="#e5e7eb"/>')
        parts.append(_svg_text(margin_left - 12, y + 5, f"{_fmt_num(value)}", size=12, anchor="end", fill="#6b7280"))

    parts.append(f'<polyline points="{budget_path}" fill="none" stroke="#94a3b8" stroke-width="3" stroke-dasharray="7 5"/>')
    parts.append(f'<polyline points="{actual_path}" fill="none" stroke="#2563eb" stroke-width="4"/>')
    for idx, row in enumerate(rows):
        bx, by = budget_points[idx]
        ax, ay = actual_points[idx]
        parts.append(f'<circle cx="{bx}" cy="{by}" r="4" fill="#94a3b8"/>')
        parts.append(f'<circle cx="{ax}" cy="{ay}" r="5" fill="#2563eb"/>')
        parts.append(_svg_text(ax, ay - 12, _fmt_num(actuals[idx]), size=12, weight="600", fill="#1d4ed8"))
        parts.append(_svg_text(bx, margin_top + chart_h + 28, f"{_fmt_num(float(row['年度信贷总额_万元']))}", size=12, fill="#374151"))
    parts.append('<rect x="690" y="54" width="18" height="4" fill="#2563eb"/>')
    parts.append(_svg_text(765, 62, "实际放贷金额", size=13, anchor="start"))
    parts.append('<line x1="690" y1="82" x2="708" y2="82" stroke="#94a3b8" stroke-width="3" stroke-dasharray="7 5"/>')
    parts.append(_svg_text(765, 86, "年度信贷总额", size=13, anchor="start"))
    parts.append(_svg_text(width / 2, height - 18, "年度信贷总额（万元）", size=13, fill="#6b7280"))
    parts.append(_svg_text(20, margin_top + chart_h / 2, "金额（万元）", size=13, anchor="start", fill="#6b7280"))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_strategy_charts(prefix, title, rows, budget_rows):
    chart_dir = OUTPUT_DIR / "charts"
    risk_counts = Counter(r["风险等级"] for r in rows)
    risk_items = [(level, risk_counts[level]) for level in RISK_LEVELS]
    loan_items = [
        (level, sum(float(r["最终贷款额度_万元"]) for r in rows if r["风险等级"] == level and r["是否放贷"] == "放贷"))
        for level in RISK_LEVELS
    ]
    decision_counts = Counter(r["是否放贷"] for r in rows)
    decision_order = ["放贷", "资金不足未放贷", "不放贷"]
    decision_items = [(label, decision_counts[label]) for label in decision_order if decision_counts[label] > 0]

    outputs = [
        chart_dir / f"{prefix}_risk_distribution.svg",
        chart_dir / f"{prefix}_loan_by_risk.svg",
        chart_dir / f"{prefix}_decision_counts.svg",
        chart_dir / f"{prefix}_budget_sensitivity.svg",
    ]
    write_bar_chart_svg(outputs[0], f"{title}：风险等级分布", risk_items, unit="家")
    write_bar_chart_svg(outputs[1], f"{title}：各风险等级放贷金额", loan_items, unit="万元")
    write_bar_chart_svg(outputs[2], f"{title}：授信决策分布", decision_items, unit="家")
    write_budget_sensitivity_svg(outputs[3], f"{title}：年度总额敏感性", budget_rows)
    return outputs
