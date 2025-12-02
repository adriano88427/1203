# -*- coding: utf-8 -*-
"""
带参数因子分析辅助函数。

当前用于承载评分逻辑，后续可逐步扩展（例如分类、权重等）。
"""

import pandas as pd


def _fa_score_parameterized_factors(factor_results):
    """
    根据分组统计结果为每个参数区间打分，返回 DataFrame。

    Args:
        factor_results (dict): {因子名: {'group_stats': DataFrame, ...}}
    """
    all_scores = []

    metric_samples = {
        'return': [],
        'sharpe': [],
        'std': [],
        'drawdown': [],
    }

    for results in factor_results.values():
        group_stats = results['group_stats']
        for _, group in group_stats.iterrows():
            if pd.notna(group['年化收益率']):
                metric_samples['return'].append(float(group['年化收益率']))
            if pd.notna(group['年化夏普比率']):
                metric_samples['sharpe'].append(float(group['年化夏普比率']))
            if pd.notna(group['年化收益标准差']):
                metric_samples['std'].append(float(group['年化收益标准差']))
            if pd.notna(group['最大回撤']):
                metric_samples['drawdown'].append(float(group['最大回撤']))

    def _score_linear(value, best, worst, higher_better=True):
        if value is None or pd.isna(value):
            return 0.0
        val = float(value)
        if higher_better:
            if val <= worst:
                return 0.0
            if val >= best:
                return 10.0
            return (val - worst) / (best - worst) * 10.0
        else:
            if val >= worst:
                return 0.0
            if val <= best:
                return 10.0
            return (worst - val) / (worst - best) * 10.0

    def _calc_bounds(samples, higher_better=True, fallback_best=1.0, fallback_worst=0.0):
        series = pd.Series(samples).dropna()
        if series.empty:
            return fallback_best, fallback_worst
        q_low = series.quantile(0.05)
        q_high = series.quantile(0.95)
        center = series.median()
        span = (q_high - q_low) or (abs(center) * 0.1 + 1e-3)
        best = center + span if higher_better else center - span
        worst = center - span if higher_better else center + span
        if best == worst:
            span = abs(best) * 0.1 + 1e-3
            if higher_better:
                best += span
                worst -= span
            else:
                best -= span
                worst += span
        best += span * 0.3 if higher_better else -span * 0.3
        worst -= span * 0.3 if higher_better else span * 0.3
        return best, worst

    bounds = {
        'return': _calc_bounds(metric_samples['return'], True, 1.0, 0.0),
        'sharpe': _calc_bounds(metric_samples['sharpe'], True, 1.0, 0.0),
        'std': _calc_bounds(metric_samples['std'], False, 0.5, 3.0),
        'drawdown': _calc_bounds(metric_samples['drawdown'], False, 0.1, 0.7),
    }

    for factor, results in factor_results.items():
        group_stats = results['group_stats']
        factor_semantic = results.get('semantic')
        for _, group in group_stats.iterrows():
            param_range = group['参数区间']
            win_rate = group['胜率']
            max_drawdown = group['最大回撤']
            ann_return = group['年化收益率']
            ann_std = group['年化收益标准差']
            sharpe_ratio = group['年化夏普比率']

            win_score = _score_linear(win_rate, best=0.65, worst=0.45, higher_better=True)
            return_score = _score_linear(ann_return, *bounds['return'], higher_better=True)
            sharpe_score = _score_linear(sharpe_ratio, *bounds['sharpe'], higher_better=True)
            std_score = _score_linear(ann_std, *bounds['std'], higher_better=False)
            drawdown_value = float(max_drawdown) if max_drawdown is not None else None
            drawdown_score = _score_linear(drawdown_value, *bounds['drawdown'], higher_better=False)

            total_score = (
                return_score * 0.30
                + sharpe_score * 0.30
                + std_score * 0.10
                + drawdown_score * 0.30
            )
            factor_direction = "正向" if ann_return >= 0 else "负向"

            all_scores.append({
                '因子名称': factor,
                '参数区间': param_range,
                '区间序号': group.get('分组'),
                '胜率': win_rate,
                '最大回撤': max_drawdown,
                '平均每笔收益率': group.get('平均收益'),
                '交易日数量': group.get('交易日数量'),
                '样本数量': group.get('样本数量'),
                '数据年份': group.get('数据年份'),
                '年化收益率': ann_return,
                '年化收益标准差': ann_std,
                '年化夏普比率': sharpe_ratio,
                '因子方向': factor_direction,
                '综合得分': total_score,
                '胜率得分': win_score,
                '收益率得分': return_score,
                '夏普得分': sharpe_score,
                '风险得分': std_score,
                '回撤得分': drawdown_score,
                '值语义': factor_semantic,
            })

    df = pd.DataFrame(all_scores)
    if df.empty:
        return df

    df['相邻平滑后得分'] = df['综合得分']

    for factor, group in df.groupby('因子名称', sort=False):
        sort_column = '区间序号' if '区间序号' in group.columns else None
        if sort_column and group[sort_column].notna().any():
            ordered = group.sort_values(sort_column, kind='mergesort')
        else:
            ordered = group

        ordered_indices = ordered.index.to_list()
        for pos, idx in enumerate(ordered_indices):
            current = float(df.at[idx, '综合得分'])
            prev_score = float(df.at[ordered_indices[pos - 1], '综合得分']) if pos > 0 else None
            next_score = float(df.at[ordered_indices[pos + 1], '综合得分']) if pos < len(ordered_indices) - 1 else None

            prev_weight = 0.25 if prev_score is not None else 0.0
            curr_weight = 0.5
            next_weight = 0.25 if next_score is not None else 0.0
            weight_sum = prev_weight + curr_weight + next_weight

            smoothed = (
                (prev_score if prev_score is not None else 0.0) * prev_weight +
                current * curr_weight +
                (next_score if next_score is not None else 0.0) * next_weight
            ) / weight_sum if weight_sum > 0 else current

            df.at[idx, '相邻平滑后得分'] = smoothed

    return df
