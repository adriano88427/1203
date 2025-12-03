# -*- coding: utf-8 -*-
"""
非参数双因子分析管道。

以现有 FactorAnalysis 的结果为基础，筛选因子对、计算二维分位表现，
并将摘要交给报告模块生成独立文件。
"""
from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .fa_config import RETURN_COLUMN
from .fa_stat_utils import calc_max_drawdown, custom_spearman_corr
from .fa_dual_nonparam_report import generate_dual_nonparam_reports


def run_dual_nonparam_pipeline(
    analyzer,
    dual_config: Dict[str, Any],
    report_options: Dict[str, Any],
    logger=None,
) -> Dict[str, str]:
    """
    执行非参数双因子分析并生成报告。
    """
    data = getattr(analyzer, "processed_data", None)
    if data is None or data.empty:
        print("[WARN] 双因子分析缺少 processed_data，已跳过")
        return {}

    factor_pairs = _select_factor_pairs(analyzer, dual_config)
    if not factor_pairs:
        print("[WARN] 未找到合规的双因子组合，已跳过")
        return {}

    min_samples = dual_config.get("min_samples", 800)
    bins = max(3, int(dual_config.get("nonparam_bins", 5)))

    if dual_config.get("enable_prescreen", True):
        factor_pairs = _prescreen_factor_pairs(
            data,
            factor_pairs,
            dual_config.get("max_factor_pairs", len(factor_pairs)),
        )
        if not factor_pairs:
            print("[WARN] 预筛后无可用因子对")
            return {}

    results: List[Dict[str, Any]] = []
    for pair in factor_pairs:
        result = _analyze_factor_pair(data, pair, bins, min_samples)
        if result:
            results.append(result)

    if not results:
        print("[WARN] 双因子分析未产生有效结果")
        return {}

    report_paths = generate_dual_nonparam_reports(results, report_options)
    return report_paths


def _select_factor_pairs(analyzer, dual_config: Dict[str, Any]) -> List[Tuple[str, str]]:
    """根据配置和单因子表现挑选双因子组合。"""
    explicit_pairs: Sequence[Tuple[str, str]] = dual_config.get("nonparam_factor_pairs") or []
    if explicit_pairs:
        pairs = [(a, b) for a, b in explicit_pairs if a in analyzer.factors and b in analyzer.factors]
    else:
        analysis_results = getattr(analyzer, "analysis_results", {}) or {}
        ranked = sorted(
            analysis_results.items(),
            key=lambda item: abs(item[1].get("ic_mean", 0.0)),
            reverse=True,
        )
        top_n = max(2, int(dual_config.get("nonparam_top_n", 6)))
        top_factors = [name for name, _ in ranked[:top_n]]
        pairs = list(itertools.combinations(top_factors, 2))

    max_pairs = max(1, int(dual_config.get("max_factor_pairs", 30)))
    if len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]
    return pairs


def _prescreen_factor_pairs(
    df: pd.DataFrame,
    factor_pairs: List[Tuple[str, str]],
    limit: int,
) -> List[Tuple[str, str]]:
    """通过简化 IC 算法预筛因子对，优先处理协同潜力高的组合。"""
    quick_scores: List[Tuple[Tuple[str, str], float]] = []
    for idx, pair in enumerate(factor_pairs):
        if idx % 10 == 0:
            print(f"[INFO] 双因子预筛进度 {idx + 1}/{len(factor_pairs)}")
        cols = [pair[0], pair[1], RETURN_COLUMN]
        if any(col not in df.columns for col in cols):
            continue
        subset = df[cols].dropna()
        if len(subset) < 200:
            continue
        ic1 = subset[pair[0]].corr(subset[RETURN_COLUMN])
        ic2 = subset[pair[1]].corr(subset[RETURN_COLUMN])
        combined = (
            (subset[pair[0]].rank(pct=True) + subset[pair[1]].rank(pct=True)) / 2.0
        ).corr(subset[RETURN_COLUMN])
        score = (combined or 0.0) - max(abs(ic1 or 0.0), abs(ic2 or 0.0))
        quick_scores.append((pair, abs(score)))

    if not quick_scores:
        return factor_pairs[:limit]
    quick_scores.sort(key=lambda item: item[1], reverse=True)
    selected = [pair for pair, _ in quick_scores[:limit]]
    return selected


def _analyze_factor_pair(
    df: pd.DataFrame,
    pair: Tuple[str, str],
    bins: int,
    min_samples: int,
) -> Optional[Dict[str, Any]]:
    """对单个因子对执行二维分组，并计算协同指标。"""
    factor_a, factor_b = pair
    required_cols = [factor_a, factor_b, RETURN_COLUMN]
    if any(col not in df.columns for col in required_cols):
        return None

    subset = df[required_cols].dropna()
    if len(subset) < min_samples:
        return None

    try:
        subset = subset.copy()
        subset["bucket_a"] = pd.qcut(subset[factor_a], q=bins, labels=False, duplicates="drop")
        subset["bucket_b"] = pd.qcut(subset[factor_b], q=bins, labels=False, duplicates="drop")
    except ValueError:
        return None

    if subset["bucket_a"].nunique() < 2 or subset["bucket_b"].nunique() < 2:
        return None

    group_stats = (
        subset.groupby(["bucket_a", "bucket_b"])[RETURN_COLUMN]
        .agg(["mean", "count"])
        .rename(columns={"mean": "avg_return", "count": "samples"})
        .reset_index()
    )
    grid = group_stats.pivot(index="bucket_a", columns="bucket_b", values="avg_return")

    ic_a = custom_spearman_corr(subset[factor_a].values, subset[RETURN_COLUMN].values) or 0.0
    ic_b = custom_spearman_corr(subset[factor_b].values, subset[RETURN_COLUMN].values) or 0.0
    combined_factor = (subset[factor_a].rank(pct=True) + subset[factor_b].rank(pct=True)) / 2.0
    combined_ic = custom_spearman_corr(combined_factor.values, subset[RETURN_COLUMN].values) or 0.0

    long_short = float(group_stats["avg_return"].max() - group_stats["avg_return"].min())
    synergy = combined_ic - max(abs(ic_a), abs(ic_b))
    drawdown = calc_max_drawdown(subset[RETURN_COLUMN])

    summary = {
        "factor_a": factor_a,
        "factor_b": factor_b,
        "ic_a": ic_a,
        "ic_b": ic_b,
        "combined_ic": combined_ic,
        "synergy": synergy,
        "sample_size": int(len(subset)),
        "long_short": long_short,
        "max_drawdown": drawdown,
    }
    return {"summary": summary, "grid": grid, "raw": group_stats}
