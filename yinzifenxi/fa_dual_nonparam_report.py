# -*- coding: utf-8 -*-
"""
非参数双因子分析报告生成。
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List

import pandas as pd

from .fa_report_utils import HTMLReportBuilder, render_metric_cards, render_table, render_alert


def generate_dual_nonparam_reports(results: List[Dict], report_options: Dict[str, any]) -> Dict[str, str]:
    """生成 CSV 和 HTML 报告，返回路径。"""
    output_dir = report_options.get("output_dir")
    os.makedirs(output_dir, exist_ok=True)
    prefix = report_options.get("nonparam_prefix", "双因子非参数")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summaries = [item["summary"] for item in results]
    summary_df = pd.DataFrame(summaries)
    csv_name = f"{prefix}汇总_{timestamp}.csv"
    csv_path = os.path.join(output_dir, csv_name)
    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    html_name = f"{prefix}分析_{timestamp}.html"
    html_path = os.path.join(output_dir, html_name)
    html_content = _build_html_report(results, report_options, timestamp)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return {"csv_path": csv_path, "html_path": html_path}


def _build_html_report(results: List[Dict], report_options: Dict[str, any], timestamp: str) -> str:
    summaries = [item["summary"] for item in results]
    df = pd.DataFrame(summaries)

    builder = HTMLReportBuilder("双因子非参数分析报告", f"生成时间 {timestamp}")
    avg_synergy = df["synergy"].mean()
    avg_ic = df["combined_ic"].mean()
    avg_long_short = df["long_short"].mean()

    cards = render_metric_cards(
        [
            ("分析因子对数", str(len(results))),
            ("平均协同IC", f"{avg_synergy:.4f}"),
            ("平均组合IC", f"{avg_ic:.4f}"),
            ("平均多空差", f"{avg_long_short:.4f}"),
        ]
    )
    builder.add_section("核心指标", cards)

    ranked = df.sort_values("synergy", key=lambda s: s.abs(), ascending=False).head(
        report_options.get("max_rank_display", 10)
    )
    if ranked.empty:
        builder.add_section("因子对排行榜", render_alert("暂无可用因子对。"))
    else:
        table = render_table(
            ranked,
            columns=[
                "factor_a",
                "factor_b",
                "ic_a",
                "ic_b",
                "combined_ic",
                "synergy",
                "long_short",
                "sample_size",
            ],
            headers=["因子A", "因子B", "IC(A)", "IC(B)", "组合IC", "协同效应", "多空收益差", "样本量"],
            formatters={
                "ic_a": lambda x: f"{float(x):.4f}",
                "ic_b": lambda x: f"{float(x):.4f}",
                "combined_ic": lambda x: f"{float(x):.4f}",
                "synergy": lambda x: f"{float(x):+.4f}",
                "long_short": lambda x: f"{float(x):.4f}",
            },
        )
        builder.add_section("因子对排行榜", table)

    if report_options.get("heatmap_enabled", True):
        heatmap_sections = []
        for item in results[:3]:
            grid = item.get("grid")
            summary = item["summary"]
            if isinstance(grid, pd.DataFrame) and not grid.empty:
                grid_reset = grid.copy()
                grid_reset.index = [f"A{i}" for i in grid_reset.index]
                grid_reset.columns = [f"B{j}" for j in grid_reset.columns]
                heatmap_sections.append(
                    f"<h3>{summary['factor_a']} × {summary['factor_b']}</h3>"
                    + grid_reset.to_html(border=0, classes='table heatmap', float_format=lambda v: f"{v:.4f}")
                )
        if heatmap_sections:
            builder.add_section("典型热力图", "".join(heatmap_sections))

    return builder.render()
