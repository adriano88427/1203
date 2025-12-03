# -*- coding: utf-8 -*-
"""
带参数双因子报告生成。
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List

import pandas as pd

from .fa_report_utils import HTMLReportBuilder, render_metric_cards, render_table, render_alert


def generate_dual_param_reports(results: List[Dict], report_options: Dict[str, any]) -> Dict[str, str]:
    output_dir = report_options.get("output_dir")
    os.makedirs(output_dir, exist_ok=True)
    prefix = report_options.get("param_prefix", "双因子带参数")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summaries = [item["summary"] for item in results]
    summary_df = pd.DataFrame(summaries)
    csv_name = f"{prefix}汇总_{timestamp}.csv"
    csv_path = os.path.join(output_dir, csv_name)
    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    html_name = f"{prefix}分析_{timestamp}.html"
    html_path = os.path.join(output_dir, html_name)
    html_content = _build_html(summary_df, results, timestamp)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    excel_name = f"{prefix}数据_{timestamp}.xlsx"
    excel_path = os.path.join(output_dir, excel_name)
    _write_excel(results, excel_path)

    return {"csv_path": csv_path, "html_path": html_path, "excel_path": excel_path}


def _build_html(summary_df: pd.DataFrame, results: List[Dict], timestamp: str) -> str:
    builder = HTMLReportBuilder("带参数双因子分析报告", f"生成时间 {timestamp}")
    cards = render_metric_cards(
        [
            ("因子对数", str(len(results))),
            ("平均最佳年化", f"{summary_df['best_annual_return'].mean():.3f}"),
            ("平均协同增益", f"{summary_df['synergy'].mean():.3f}"),
        ]
    )
    builder.add_section("核心指标", cards)

    top = summary_df.sort_values("best_annual_return", ascending=False).head(10)
    if top.empty:
        builder.add_section("最佳双因子区间", render_alert("暂无有效区间。"))
    else:
        table = render_table(
            top,
            columns=[
                "factor_a",
                "factor_b",
                "best_range",
                "best_annual_return",
                "worst_range",
                "worst_annual_return",
                "synergy",
            ],
            headers=["因子A", "因子B", "最佳区间", "最佳年化", "最差区间", "最差年化", "协同增益"],
            formatters={
                "best_annual_return": lambda v: f"{float(v):.3f}",
                "worst_annual_return": lambda v: f"{float(v):.3f}",
                "synergy": lambda v: f"{float(v):.3f}",
            },
        )
        builder.add_section("最佳双因子区间", table)

    detail_sections = []
    for item in results[:3]:
        grid = item["grid"]
        summary = item["summary"]
        detail_sections.append(
            f"<h3>{summary['factor_a']} × {summary['factor_b']}</h3>"
            + grid.to_html(index=False, float_format=lambda v: f"{v:.4f}")
        )
    if detail_sections:
        builder.add_section("部分区间详情", "".join(detail_sections))

    return builder.render()


def _write_excel(results: List[Dict], excel_path: str):
    writer = pd.ExcelWriter(excel_path, engine="openpyxl")
    for item in results:
        sheet_name = f"{item['summary']['factor_a']}_{item['summary']['factor_b']}"
        clean_name = sheet_name[:30]
        item["grid"].to_excel(writer, sheet_name=clean_name, index=False)
    writer.close()
