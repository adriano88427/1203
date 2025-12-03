# -*- coding: utf-8 -*-
"""
带参数双因子分析管道。
"""
from __future__ import annotations

import itertools
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .fa_config import RETURN_COLUMN
from .fa_stat_utils import calc_max_drawdown
from .fa_dual_param_report import generate_dual_param_reports


def run_dual_param_pipeline(
    parameterized_analyzer,
    dual_config: Dict[str, Any],
    report_options: Dict[str, Any],
    logger=None,
) -> Dict[str, str]:
    data = getattr(parameterized_analyzer, "processed_data", None)
    if data is None or data.empty:
        print("[WARN] 双因子带参数分析缺少 processed_data，已跳过")
        return {}

    factor_pairs = _select_param_pairs(parameterized_analyzer, dual_config)
    if not factor_pairs:
        print("[WARN] 未找到用于带参数双因子的因子对")
        return {}

    min_samples = dual_config.get("param_min_samples", 300)
    results: List[Dict[str, Any]] = []
    default_bins = dual_config.get("param_default_bins", 3)
    for pair in factor_pairs:
        pair_result = _analyze_parameterized_pair(
            data,
            pair,
            dual_config.get("param_ranges", {}),
            min_samples,
            default_bins,
        )
        if pair_result:
            results.append(pair_result)

    if not results:
        print("[WARN] 双因子带参数分析未产生有效结果")
        return {}

    return generate_dual_param_reports(results, report_options)


def _select_param_pairs(parameterized_analyzer, dual_config: Dict[str, Any]) -> List[Tuple[str, str]]:
    pairs: Sequence[Tuple[str, str]] = dual_config.get("param_factor_pairs") or []
    if pairs:
        filtered = [
            (a, b)
            for a, b in pairs
            if a in parameterized_analyzer.factors and b in parameterized_analyzer.factors
        ]
        if filtered:
            return filtered
    # 默认使用单因子列表前若干项
    factors = parameterized_analyzer.factors[: dual_config.get("nonparam_top_n", 6)]
    return list(itertools.combinations(factors, 2))[: dual_config.get("max_factor_pairs", 20)]


def _analyze_parameterized_pair(
    df: pd.DataFrame,
    pair: Tuple[str, str],
    range_config: Dict[str, Any],
    min_samples: int,
    default_bins: int,
) -> Optional[Dict[str, Any]]:
    factor_a, factor_b = pair
    if factor_a not in df.columns or factor_b not in df.columns or RETURN_COLUMN not in df.columns:
        return None

    ranges_a = _resolve_ranges(df[factor_a], range_config.get(factor_a), default_bins)
    ranges_b = _resolve_ranges(df[factor_b], range_config.get(factor_b), default_bins)
    if not ranges_a or not ranges_b:
        return None

    records: List[Dict[str, Any]] = []
    for idx_a, range_a in enumerate(ranges_a):
        mask_a = _build_mask(df[factor_a], range_a)
        for idx_b, range_b in enumerate(ranges_b):

            mask_b = _build_mask(df[factor_b], range_b)
            subset = df[mask_a & mask_b]
            returns = subset[RETURN_COLUMN].dropna()
            if len(returns) < min_samples:
                continue
            avg = float(returns.mean())
            std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
            annual_return = avg * 252
            annual_std = std * np.sqrt(252) if std > 0 else np.nan
            sharpe = (annual_return / annual_std) if annual_std and np.isfinite(annual_std) else np.nan
            drawdown = calc_max_drawdown(returns)
            win_rate = float((returns > 0).mean())

            records.append(
                {
                    "factor_a": factor_a,
                    "factor_b": factor_b,
                    "range_a": range_a,
                    "range_b": range_b,
                    "range_a_label": _format_range_label(idx_a, range_a),
                    "range_b_label": _format_range_label(idx_b, range_b),
                    "avg_return": avg,
                    "annual_return": annual_return,
                    "annual_std": annual_std,
                    "sharpe": sharpe,
                    "samples": len(returns),
                    "win_rate": win_rate,
                    "max_drawdown": drawdown,
                }
            )

    if not records:
        return None

    pair_df = pd.DataFrame(records)
    best = pair_df.sort_values("annual_return", ascending=False).iloc[0]
    worst = pair_df.sort_values("annual_return", ascending=True).iloc[0]
    summary = {
        "factor_a": factor_a,
        "factor_b": factor_b,
        "best_range": f"{best['range_a_label']} & {best['range_b_label']}",
        "best_annual_return": best["annual_return"],
        "worst_range": f"{worst['range_a_label']} & {worst['range_b_label']}",
        "worst_annual_return": worst["annual_return"],
        "synergy": best["annual_return"] - max(worst["annual_return"], 0),
    }
    return {"summary": summary, "grid": pair_df}


def _resolve_ranges(series: pd.Series, config_ranges, default_bins: int) -> List[Tuple[float, float]]:
    """将用户配置或默认分位转换为数值区间。"""
    if config_ranges:
        normalized = []
        for item in config_ranges:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                normalized.append((float(item[0]), float(item[1])))
        if normalized:
            return normalized

    bins = max(2, int(default_bins))
    quantile_points = np.linspace(0, 1, bins + 1)
    values = series.quantile(quantile_points).tolist()
    ranges = []
    for idx in range(bins):
        low, high = values[idx], values[idx + 1]
        if low == high:
            continue
        ranges.append((float(low), float(high)))
    return ranges


def _build_mask(series: pd.Series, value_range: Tuple[float, float]) -> pd.Series:
    low, high = value_range
    return (series >= low) & (series <= high)


def _format_range_label(index: int, value_range: Tuple[float, float]) -> str:
    return f"区间{index + 1}[{value_range[0]:.4f}~{value_range[1]:.4f}]"
