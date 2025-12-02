# -*- coding: utf-8 -*-
"""
数据完整性验证模块

用于在正式分析前对多表合并结果进行强制校验，并输出详细的验证报告。
"""
from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import pandas as pd
import numpy as np

from .fa_config import COLUMN_ALIGNMENT_RULES


@dataclass
class DataValidationResult:
    passed: bool
    report_lines: List[str]


class DataValidator:
    """
    针对多文件输入的数据验证器：
    1. 统计每个文件/工作表的行数，判断是否全部读取成功
    2. 根据信号日期列评估年份/交易日覆盖情况
    3. 计算估算的样本覆盖比例，并在阈值不足时直接阻断后续分析
    """

    def __init__(
        self,
        file_paths: Sequence[str],
        diagnostics: Optional[Iterable],
        combined_df: Optional[pd.DataFrame],
        required_columns: Optional[Sequence[str]] = None,
        min_file_coverage: float = 1.0,
        min_row_coverage: float = 0.8,
        min_year_coverage: float = 0.8,
        enable_debug: Optional[bool] = None,
    ):
        self.file_paths = [os.path.abspath(p) for p in file_paths if p]
        self.diagnostics = list(diagnostics or [])
        self.df = combined_df if combined_df is not None else pd.DataFrame()
        self.required_columns = [col for col in (required_columns or []) if col]
        self.min_file_coverage = min_file_coverage
        self.min_row_coverage = min_row_coverage
        self.min_year_coverage = min_year_coverage
        env_debug = os.environ.get("FA_VALIDATOR_DEBUG")
        if enable_debug is not None:
            self.debug_enabled = bool(enable_debug)
        elif env_debug is None:
            self.debug_enabled = True
        else:
            self.debug_enabled = env_debug.strip().lower() not in ("0", "false", "no")

    def run(self) -> DataValidationResult:
        lines: List[str] = ["\n[VALIDATION] === 数据完整性验证 ==="]
        expected_files = len(self.file_paths)
        file_stats = self._aggregate_file_stats()
        self._debug(
            f"启动多表格对比验证：期望文件 {expected_files} 个，解析诊断 {len(self.diagnostics)} 条",
            lines,
        )

        total_loaded_rows = len(self.df)
        valid_files = [
            path
            for path in self.file_paths
            if file_stats.get(path, {}).get("rows", 0) > 0
        ]
        valid_file_count = len(valid_files)
        coverage_ratio = (
            valid_file_count / expected_files if expected_files else 1.0
        )

        avg_rows_per_valid = (
            total_loaded_rows / valid_file_count if valid_file_count > 0 else 0.0
        )
        estimated_total_rows = (
            avg_rows_per_valid * expected_files if expected_files else total_loaded_rows
        )
        row_coverage_ratio = (
            (total_loaded_rows / estimated_total_rows)
            if estimated_total_rows > 0
            else 0.0
        )

        self._debug(
            f"聚合完成：成功文件 {valid_file_count} 个，总样本 {total_loaded_rows} 行，估算覆盖率 {row_coverage_ratio:.2%}",
            lines,
        )

        lines.append(
            f"  - 期望文件数: {expected_files}, 成功读取: {valid_file_count}, 覆盖率: {coverage_ratio:.1%}"
        )
        if valid_file_count:
            lines.append(
                f"  - 合并后总样本: {total_loaded_rows} 行，"
                f"估算理论样本: {int(estimated_total_rows)} 行，覆盖率: {row_coverage_ratio:.1%}"
            )
        else:
            lines.append("  - 合并后总样本: 0 行")

        lines.append("  - 逐文件读入情况：")
        for path in self.file_paths:
            info = file_stats.get(path)
            base_name = os.path.basename(path)
            if not info or info["rows"] == 0:
                lines.append(f"    · {base_name}: 未读取 (0 行)")
                continue
            percent = (
                (info["rows"] / total_loaded_rows) * 100
                if total_loaded_rows > 0
                else 0.0
            )
            sheet_desc = "; ".join(
                f"{sheet}({rows})" for sheet, rows in info["sheets"]
            ) or "无"
            lines.append(
                f"    · {base_name}: {info['rows']} 行，占比 {percent:.1f}% ，工作表 {sheet_desc}"
            )
            for issue in info["issues"]:
                lines.append(f"        [问题] {issue}")

        missing_files = [
            os.path.basename(path)
            for path in self.file_paths
            if path not in valid_files
        ]
        if missing_files:
            self._debug(
                "存在未读取的文件: " + ", ".join(missing_files),
                lines,
            )
        if missing_files:
            lines.append(
                f"  - 警告：以下文件未成功读取任何样本：{', '.join(missing_files)}"
            )

        year_info = self._summarize_year_coverage()
        lines.extend(year_info["lines"])
        observed_years = year_info["year_count"]
        self._debug(
            f"年份覆盖检查：检测到 {observed_years} 个年份记录",
            lines,
        )

        column_info = self._analyze_column_consistency()
        lines.extend(column_info["lines"])
        if column_info.get("failed"):
            self._debug(
                "列对齐检查发现异常，请关注上方 FAIL 信息",
                lines,
            )
        else:
            self._debug("列对齐检查通过", lines)

        passed = True
        if expected_files and coverage_ratio < self.min_file_coverage:
            passed = False
            lines.append("  - [FAIL] 文件覆盖率低于要求，存在未读入的表格")
        if expected_files and row_coverage_ratio < self.min_row_coverage:
            passed = False
            lines.append("  - [FAIL] 样本覆盖率偏低，读取行数远低于理论值")
        if expected_files and observed_years / expected_files < self.min_year_coverage:
            passed = False
            lines.append("  - [FAIL] 信号日期年份覆盖不足，无法满足多年的分析要求")
        if column_info.get("failed"):
            passed = False
        if total_loaded_rows == 0:
            passed = False
            lines.append("  - [FAIL] 未读取到任何有效样本，无法继续分析")

        if passed:
            lines.append("[VALIDATION] 数据验证通过")
            self._debug("多表格对比验证完成：状态 PASS", lines)
        else:
            lines.append("[VALIDATION] 数据验证未通过")
            self._debug("多表格对比验证完成：状态 FAIL", lines)

        return DataValidationResult(passed=passed, report_lines=lines)

    def _debug(self, message: str, buffer: Optional[List[str]] = None):
        """记录调试信息到日志"""
        if not getattr(self, "debug_enabled", False):
            return
        text = f"[DEBUG][DataValidator] {message}"
        if buffer is not None:
            buffer.append(text)
        else:
            print(text)

    def _aggregate_file_stats(self) -> "OrderedDict[str, dict]":
        stats: "OrderedDict[str, dict]" = OrderedDict()
        for diag in self.diagnostics:
            file_key = os.path.abspath(getattr(diag, "file_path", ""))
            entry = stats.setdefault(
                file_key,
                {"rows": 0, "sheets": [], "issues": []},
            )
            entry["rows"] += getattr(diag, "rows", 0) or 0
            sheet_name = getattr(diag, "sheet_name", "Sheet")
            entry["sheets"].append((sheet_name, getattr(diag, "rows", 0) or 0))
            issues = self._collect_diag_issues(diag, self.required_columns)
            entry["issues"].extend(issues)
        return stats

    @staticmethod
    def _collect_diag_issues(diag, required_columns: Sequence[str]) -> List[str]:
        issues: List[str] = []
        unmapped = getattr(diag, "unmapped_columns", None) or []
        if unmapped:
            issues.append(f"未识别列: {', '.join(unmapped)}")
        conversions = getattr(diag, "conversion_failures", None) or {}
        for column, rate in conversions.items():
            if rate >= 0.5:
                issues.append(f"{column} 转换失败率 {rate:.0%}")
        present_cols = set(getattr(diag, "present_columns", []))
        missing_required = [
            col for col in required_columns if col not in present_cols
        ]
        if missing_required:
            issues.append(f"缺少必需列: {', '.join(missing_required)}")
        notes = getattr(diag, "notes", None) or []
        issues.extend(notes)
        return issues

    def _summarize_year_coverage(self) -> dict:
        lines: List[str] = []
        if self.df is None or self.df.empty:
            lines.append("  - 未提供合并数据，无法检查年份覆盖")
            return {"lines": lines, "year_count": 0, "years": []}

        date_column = None
        for candidate in ("信号日期", "signal_date", "日期", "交易日期"):
            if candidate in self.df.columns:
                date_column = candidate
                break

        if date_column is None:
            lines.append("  - 未找到信号日期/交易日期列，无法统计年份覆盖")
            return {"lines": lines, "year_count": 0, "years": []}

        date_series = pd.to_datetime(self.df[date_column], errors="coerce")
        valid_dates = date_series.dropna()
        unique_years = sorted(
            set(int(year) for year in valid_dates.dt.year.dropna().unique())
        )
        unique_days = int(valid_dates.dt.normalize().nunique())
        if unique_years:
            lines.append(
                f"  - 覆盖年份: {', '.join(str(y) for y in unique_years)} "
                f"(共 {len(unique_years)} 年)，覆盖交易日 {unique_days} 天"
            )
        else:
            lines.append("  - 未获取到有效年份信息")
        return {"lines": lines, "year_count": len(unique_years), "years": unique_years}

    def _analyze_column_consistency(self) -> dict:
        lines: List[str] = ["  - 列值一致性检查"]
        failed = False
        if not COLUMN_ALIGNMENT_RULES:
            lines.append("    · 未配置列对齐规则")
            return {"lines": lines, "failed": False}
        if self.df is None or self.df.empty:
            lines.append("    [WARN] 数据为空，跳过列对齐检查")
            return {"lines": lines, "failed": False}

        date_column = None
        for candidate in ("信号日期", "signal_date", "日期", "交易日期"):
            if candidate in self.df.columns:
                date_column = candidate
                break
        if date_column is None:
            lines.append("    [WARN] 未找到日期列，无法执行列对齐检查")
            return {"lines": lines, "failed": False}

        date_series = pd.to_datetime(self.df[date_column], errors="coerce")
        year_series = date_series.dt.year
        available_years = sorted(int(y) for y in year_series.dropna().unique())
        if not available_years:
            lines.append("    [WARN] ????????????????")
            return {"lines": lines, "failed": False}

        for column, rule in COLUMN_ALIGNMENT_RULES.items():
            if column not in self.df.columns:
                continue
            col_series = pd.to_numeric(self.df[column], errors="coerce")
            year_stats = {}
            for year in available_years:
                mask = year_series == year
                year_values = col_series[mask].dropna()
                if year_values.empty:
                    continue
                stats = {
                    "count": len(year_values),
                    "median": float(year_values.median()),
                    "abs_max": float(year_values.abs().max()),
                    "share_over1": float((year_values.abs() > 1).mean()),
                }
                year_stats[year] = stats
                abs_max = rule.get("abs_max")
                if abs_max is not None and stats["abs_max"] > abs_max + 1e-9:
                    lines.append(
                        f"    [FAIL] ? {column} ? {year} ????? {abs_max*100:.0f}% ????? {stats['abs_max']*100:.1f}%?"
                    )
                    failed = True
            if not year_stats:
                continue

            baseline_candidates = [
                stats["median"]
                for stats in year_stats.values()
                if np.isfinite(stats["median"]) and stats["count"] >= rule.get("min_samples", 100)
            ]
            if not baseline_candidates:
                baseline_candidates = [
                    stats["median"]
                    for stats in year_stats.values()
                    if np.isfinite(stats["median"])
                ]
            if not baseline_candidates:
                continue
            baseline_median = float(np.median(np.abs(baseline_candidates)))
            if not np.isfinite(baseline_median) or baseline_median == 0:
                baseline_median = 1.0

            for year, stats in year_stats.items():
                median_val = stats["median"]
                if not np.isfinite(median_val) or stats["count"] < max(50, rule.get("min_samples", 100) * 0.2):
                    continue
                factor = self._infer_scale_factor(baseline_median, median_val, rule)
                if abs(factor - 1.0) > 0.05:
                    mask = year_series == year
                    self.df.loc[mask, column] = self.df.loc[mask, column] * factor
                    col_series = pd.to_numeric(self.df[column], errors="coerce")
                    lines.append(
                        f"    [FIX] ? {column} {year} ???? {factor:g} ???????"
                    )
                    stats["median"] = float(median_val * factor)
                    stats["abs_max"] = float(pd.to_numeric(self.df.loc[mask, column], errors="coerce").abs().max())
                    abs_max = rule.get("abs_max")
                    if abs_max is not None and stats["abs_max"] > abs_max + 1e-9:
                        lines.append(
                            f"    [FAIL] ? {column} {year} ?????????? {abs_max*100:.0f}%"
                        )
                        failed = True

            medians = [abs(stats["median"]) for stats in year_stats.values() if np.isfinite(stats["median"])]
            if medians:
                global_median = float(np.median(medians)) or 1.0
                for year, stats in year_stats.items():
                    median_val = abs(stats["median"])
                    if global_median and (median_val > global_median * 5 or median_val < global_median * 0.2):
                        lines.append(
                            f"    [FAIL] ? {column} {year} ????? {median_val:.4f} ?????????"
                        )
                        failed = True
            else:
                lines.append(f"    [WARN] ? {column} ??????????????")

        return {"lines": lines, "failed": failed}

    @staticmethod
    def _infer_scale_factor(reference: float, candidate: float, rule: dict) -> float:
        if not np.isfinite(reference) or not np.isfinite(candidate) or candidate == 0:
            return 1.0
        ratio = reference / candidate
        candidates = rule.get("scale_candidates", [10, 100, 0.1, 0.01])
        for factor in candidates:
            if factor <= 0:
                continue
            lower = factor * 0.7
            upper = factor * 1.3
            if lower <= ratio <= upper:
                return factor
        return 1.0

    
