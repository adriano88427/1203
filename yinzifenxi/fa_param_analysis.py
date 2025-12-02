# -*- coding: utf-8 -*-
"""
带参数因子分析模块，承载 ParameterizedFactorAnalyzer。
本模块已在 yinzifenxi1119_split.py 中被引用，用于生成带参数因子报告。
"""

import os
import unicodedata
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict

from .excel_parser import load_excel_sources, DEFAULT_PARSE_CONFIG
from .fa_config import (
    DEFAULT_DATA_FILES,
    FACTOR_COLUMNS,
    RETURN_COLUMN,
    DATA_PARSE_CONFIG,
)
from .fa_nonparam_analysis import DEFAULT_GROUP_COUNT, DEFAULT_USE_PEARSON
from .fa_param_helpers import _fa_score_parameterized_factors
from .fa_param_report import _fa_generate_parameterized_report
from .fa_stat_utils import (
    calc_max_drawdown,
    custom_spearman_corr,
    rolling_window_analysis,
    sample_sensitivity_analysis,
)
from .excel_parser import FactorNormalizer, NormalizationInfo


class ParameterizedFactorAnalyzer:
    """专门针对带参数因子的综合分析器"""
    
    def __init__(self, data, file_path=None):
        """初始化综合因子分析器"""
        self.data = data
        default_paths = DEFAULT_DATA_FILES if file_path is None else file_path
        if isinstance(default_paths, (list, tuple, set)):
            resolved_paths = [str(p) for p in default_paths if p]
        elif default_paths:
            resolved_paths = [str(default_paths)]
        else:
            resolved_paths = []
        self.file_paths = resolved_paths
        self.file_path = self.file_paths[0] if self.file_paths else None
        enabled_flags = DATA_PARSE_CONFIG.get("column_enabled", {}) or {}
        self.factors = [
            factor for factor in FACTOR_COLUMNS
            if enabled_flags.get(factor, "是") != "否"
        ]
        self.factor_list = self.factors  # 修复：添加factor_list属性
        self.disabled_factors = [
            factor for factor in FACTOR_COLUMNS
            if enabled_flags.get(factor, "是") == "否"
        ]
        self.return_col = RETURN_COLUMN
        self.sqrt_annualization_factor = np.sqrt(252)
        self.annualization_factor = 252
        self.parse_config = DEFAULT_PARSE_CONFIG.copy()
        if DATA_PARSE_CONFIG:
            self.parse_config.update(DATA_PARSE_CONFIG)
        self.parse_diagnostics = []
        self.unavailable_columns = set()
        self.normalizer = FactorNormalizer()
        self.normalization_stats: Dict[str, NormalizationInfo] = {}
        
        # 确保数据有效
        if self.data is None or self.data.empty:
            print("错误: 没有有效数据")
            # 不返回值，让对象仍可被创建但处于无效状态
        else:
            self.data = self._normalize_dataframe_columns(self.data)
        if getattr(self, "disabled_factors", None):
            print("[INFO] 以下因子因配置被禁用: " + ", ".join(self.disabled_factors))

    @staticmethod
    def _clean_column_name(name):
        """移除隐藏空字符、警告提示等异常内容，保证列名可匹配。"""
        if name is None:
            return ""
        text = str(name)
        text = unicodedata.normalize("NFKC", text)
        text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
        # 去掉不可见控制字符（WPS提示中常见）
        text = "".join(ch for ch in text if unicodedata.category(ch) not in ("Cf", "Cc"))
        text = " ".join(text.split())
        return text.strip()

    def _normalize_dataframe_columns(self, df):
        """对DataFrame列名做清洗与去重，确保后续因子匹配可靠。"""
        if df is None or df.empty:
            return df
        canonical_map = {
            self._clean_column_name(name): name
            for name in set(list(self.factors) + [self.return_col])
        }
        rename_map = {}
        for col in df.columns:
            cleaned = self._clean_column_name(col)
            canonical = canonical_map.get(cleaned)
            rename_map[col] = canonical if canonical else (cleaned if cleaned else str(col))
        normalized = df.rename(columns=rename_map)

        counts = {}
        unique_cols = []
        for idx, name in enumerate(normalized.columns):
            base = name if name else f"Unnamed_{idx+1}"
            count = counts.get(base, 0)
            unique_name = base if count == 0 else f"{base}.{count}"
            counts[base] = count + 1
            unique_cols.append(unique_name)
        normalized.columns = unique_cols
        return normalized
    
    def load_data(self):
        """从文件加载数据"""
        if self.data is not None:
            self.data = self._normalize_dataframe_columns(self.data)
            return True
        if not self.file_paths:
            print("数据加载失败: 未指定数据文件")
            return False

        try:
            parsed = load_excel_sources(self.file_paths, self.parse_config)
        except Exception as exc:
            print(f"数据加载失败: {exc}")
            return False

        if parsed.data is None or parsed.data.empty:
            print("数据加载失败: 未能成功读取任何文件")
            return False

        for diag in parsed.diagnostics:
            filename = os.path.basename(diag.file_path)
            print(f"加载文件 {filename}[{diag.sheet_name}]: {diag.rows} 行")

        self.data = self._normalize_dataframe_columns(parsed.data)
        self.parse_diagnostics = parsed.diagnostics
        self.unavailable_columns = set(parsed.unavailable_columns or [])
        self._report_parse_integrity()
        return True
# 已删除复杂的自适应年化计算方法，使用优化版本（第1085行）
# 优化版本特点：
# 1. 使用标准复利年化方法作为主要计算方式
# 2. 保留CAGR方法作为对比方法
# 3. 删除线性年化方法（忽视复利效应）
# 4. 增强数据特征分析和验证机制
    
    def preprocess_data(self):
        """预处理数据"""
        if self.data is None or self.data.empty:
            print("错误: 没有数据可处理")
            return False

        self.data = self._normalize_dataframe_columns(self.data)
        try:
            # 复制数据
            df = self.data.copy()

            self.normalization_stats = {}
            normalization_targets = list(dict.fromkeys(list(self.factors) + [self.return_col]))
            for col in normalization_targets:
                if col not in df.columns:
                    continue
                raw_series = df[col]
                diagnostics = self._get_column_diagnostics(col)
                normalized_series, info = self.normalizer.normalize(col, raw_series, diagnostics)
                if normalized_series.notna().any():
                    df[col] = normalized_series
                    self.normalization_stats[col] = info
                    if info.applied_scale:
                        print(f"[INFO] 列 '{col}' 自动缩放: {info.applied_scale}")
                    elif info.semantic == "percent" and info.detected_percent_pattern:
                        print(f"[INFO] 列 '{col}' 识别到百分比格式并已转换为小数")
                else:
                    fallback = self._fallback_numeric_conversion(raw_series)
                    df[col] = fallback
                    self.normalization_stats[col] = info
                    print(f"[WARN] 列 '{col}' 正规化失败，使用回退转换")

            self._log_normalization_notes()
            
            # 处理收益率列（保证为数值）
            if not pd.api.types.is_numeric_dtype(df[self.return_col]):
                df[self.return_col] = pd.to_numeric(df[self.return_col], errors='coerce')
            
            # 确保日期列正确处理
            if '信号日期' in df.columns:
                try:
                    df['信号日期'] = pd.to_datetime(df['信号日期'], errors='coerce')
                except:
                    print("警告：无法转换信号日期列")
            
            # 删除缺失值
            original_len = len(df)
            df = df.dropna(subset=[self.return_col])
            print(f"数据预处理完成，分析使用 {len(df)} 行有效数据 (删除了 {original_len - len(df)} 行缺失值)")
            
            self.processed_data = df
            self._log_data_integrity_report(df)
            return True
            
        except Exception as e:
            print(f"数据预处理失败: {e}")
            return False
    
    def _log_normalization_notes(self):
        """在日志中输出数据标准化说明表格。"""
        stats = getattr(self, "normalization_stats", {}) or {}
        if not stats:
            return
        rows = []
        for column, info in stats.items():
            applied = "--" if info.applied_scale is None else f"{info.applied_scale:g}"
            notes = info.notes or ("自动识别百分比" if info.detected_percent_pattern else "--")
            rows.append({
                "列名": column,
                "语义": info.semantic,
                "展示格式": info.display,
                "缩放": applied,
                "备注": notes,
            })
        if rows:
            table = pd.DataFrame(rows, columns=["列名", "语义", "展示格式", "缩放", "备注"])
            print("\n[INFO] === 数据标准化说明 ===")
            print(table.to_string(index=False))

    def _report_parse_integrity(self):
        """根据解析诊断信息输出每个文件的字段覆盖情况。"""
        diagnostics = getattr(self, "parse_diagnostics", [])
        if not diagnostics:
            return
        required = set(self.factors + [self.return_col])
        print("\n[DATA] === 数据文件字段完整性检查 ===")
        for diag in diagnostics:
            filename = os.path.basename(diag.file_path)
            header = f"  - {filename}[{diag.sheet_name}]"
            missing = sorted(required - set(diag.present_columns))
            if missing:
                print(f"{header} 缺少列: {', '.join(missing)}")
            else:
                print(f"{header} OK (行数: {diag.rows})")
            issues = [
                f"{col}:{rate:.0%}"
                for col, rate in diag.conversion_failures.items()
                if rate >= 0.8
            ]
            if issues:
                print(f"      转换失败率过高: {', '.join(issues)}")

    def _log_data_integrity_report(self, df):
        """对整合后的数据按年份检查样本覆盖情况，并记录到日志。"""
        print("\n[DATA] === 多表样本覆盖情况 ===")
        if '信号日期' not in df.columns:
            print("  - 缺少信号日期列，无法执行年度覆盖检查")
            return
        valid_dates = df['信号日期'].dropna()
        if valid_dates.empty:
            print("  - 所有信号日期均为空，无法检查年度覆盖")
            return
        df = df.copy()
        df['__year__'] = pd.to_datetime(df['信号日期']).dt.year
        year_counts = df['__year__'].value_counts().sort_index()
        for year, count in year_counts.items():
            unique_days = df.loc[df['__year__'] == year, '信号日期'].dt.normalize().nunique()
            print(f"  - {int(year)} 年：样本 {int(count)} 条，覆盖交易日 {unique_days} 天")

        available_years = sorted(year_counts.index.tolist())
        coverage_threshold = 1
        factor_alerts = []
        for factor in self.factors:
            if factor not in df.columns:
                factor_alerts.append((factor, "列缺失"))
                continue
            counts = (
                df.groupby('__year__')[factor]
                .apply(lambda s: s.notna().sum())
                .to_dict()
            )
            missing = [
                str(year)
                for year in available_years
                if counts.get(year, 0) < coverage_threshold
            ]
            if missing:
                factor_alerts.append((factor, f"缺少年份: {', '.join(missing)}"))
        if factor_alerts:
            print("[DATA] 因子覆盖告警：")
            for name, msg in factor_alerts:
                print(f"  - {name}: {msg}")
        else:
            print("[DATA] 所有因子在各年份均存在有效样本")
        df.drop(columns='__year__', inplace=True, errors='ignore')

    def _get_column_diagnostics(self, column_name: str):
        for diag in getattr(self, "parse_diagnostics", []):
            columns = getattr(diag, "present_columns", []) or []
            if column_name in columns:
                return diag
        return None

    @staticmethod
    def _fallback_numeric_conversion(series: pd.Series) -> pd.Series:
        try:
            col_series = series.astype(str).str.strip()
            has_percent = col_series.str.contains('%').any()
            cleaned = col_series.str.replace('%', '', regex=False)
            numeric_series = pd.to_numeric(cleaned, errors='coerce')
            if has_percent:
                numeric_series = numeric_series / 100
            return numeric_series
        except Exception:
            return pd.to_numeric(series, errors='coerce')
    
    def calculate_comprehensive_metrics(self, factor_col):
        """计算综合指标"""
        df_clean = self.processed_data.dropna(subset=[factor_col, self.return_col]).copy()
        has_signal_date = '信号日期' in df_clean.columns
        norm_info = self.normalization_stats.get(factor_col)
        semantic = norm_info.semantic if norm_info else None
        if has_signal_date:
            df_clean = df_clean.sort_values('信号日期')
        else:
            print(f"警告: 因子 {factor_col} 缺少信号日期，最大回撤按原始顺序估算")
        
        if len(df_clean) < 10:
            print(f"警告: 因子 {factor_col} 有效数据不足")
            return None
        
        try:
            # 计算分组收益（默认使用配置中的等分数量）
            group_count = DEFAULT_GROUP_COUNT
            df_clean['分组'] = pd.qcut(df_clean[factor_col], q=group_count, labels=False, duplicates='drop')
            
            # 计算每组的统计指标
            group_stats = []
            total_samples = len(df_clean)
            
            for group_id in range(group_count):
                group_data = df_clean[df_clean['分组'] == group_id]
                
                if len(group_data) == 0:
                    continue
                
                # 获取该组的因子值范围
                factor_values = group_data[factor_col]
                min_val = factor_values.min()
                max_val = factor_values.max()
                param_range = self.normalizer.format_range(
                    factor_col,
                    float(min_val),
                    float(max_val),
                    semantic=semantic,
                )
                
                # 计算该组的收益统计
                returns = group_data[self.return_col]
                avg_return = returns.mean()
                return_std = returns.std()
                win_rate = (returns > 0).mean()

                # 构建用于最大回撤的日度组合收益
                trade_days = len(returns.dropna())
                date_span = None
                start_date = None
                end_date = None
                used_daily_aggregation = False
                drawdown_series = returns
                year_label = "--"
                if has_signal_date and group_data['信号日期'].notna().any():
                    dated_group = group_data.dropna(subset=['信号日期'])
                    if not dated_group.empty:
                        start_date = dated_group['信号日期'].min()
                        end_date = dated_group['信号日期'].max()
                        if pd.notna(start_date) and pd.notna(end_date):
                            date_span = f"{start_date.date()}~{end_date.date()}"
                        year_values = (
                            dated_group['信号日期']
                            .dropna()
                            .dt.year
                            .dropna()
                            .astype(int)
                            .tolist()
                        )
                        if year_values:
                            unique_years = sorted(set(year_values))
                            if len(unique_years) == 1:
                                year_label = str(unique_years[0])
                            else:
                                is_contiguous = unique_years == list(range(unique_years[0], unique_years[-1] + 1))
                                if is_contiguous:
                                    year_label = f"{unique_years[0]}-{unique_years[-1]}"
                                else:
                                    year_label = "、".join(str(y) for y in unique_years)
                        daily_returns = (
                            dated_group.groupby('信号日期')[self.return_col]
                            .mean()
                            .sort_index()
                        )
                        if len(daily_returns) >= 3:
                            drawdown_series = daily_returns
                            trade_days = len(daily_returns.dropna())
                            used_daily_aggregation = True
                        else:
                            print(
                                f"提示: 因子 {factor_col} 区间 {param_range} 有效交易日不足，使用原序列计算最大回撤"
                            )

                if isinstance(drawdown_series, pd.Series):
                    daily_returns_series = drawdown_series.dropna()
                else:
                    daily_returns_series = pd.Series(drawdown_series).dropna()
                trade_days = len(daily_returns_series)
                daily_mean = daily_returns_series.mean() if trade_days > 0 else np.nan
                daily_std = daily_returns_series.std(ddof=1) if trade_days > 1 else np.nan

                observation_days = trade_days
                if start_date is not None and end_date is not None:
                    observation_days = max((end_date - start_date).days + 1, trade_days)
                observation_years = np.nan
                if observation_days and observation_days > 0:
                    observation_years = max(observation_days / 252, 1 / 252)

                annualization_method = "CAGR"
                annualized_return = np.nan
                if trade_days > 0 and pd.notna(observation_years) and observation_years > 0:
                    clipped_returns = daily_returns_series.clip(lower=-0.99)
                    try:
                        total_return = float(np.prod(1 + clipped_returns.values) - 1)
                    except Exception:
                        total_return = np.nan
                    final_value = 1 + (total_return if not pd.isna(total_return) else 0)
                    if pd.notna(total_return) and final_value > 0:
                        try:
                            annualized_return = final_value ** (1 / observation_years) - 1
                        except Exception:
                            annualization_method = "LinearFallback"
                            annualized_return = (
                                daily_mean * 252 if pd.notna(daily_mean) else np.nan
                            )
                    else:
                        annualization_method = "LinearFallback"
                        annualized_return = (
                            daily_mean * 252 if pd.notna(daily_mean) else np.nan
                        )
                else:
                    annualization_method = "LinearFallback"
                    annualized_return = daily_mean * 252 if pd.notna(daily_mean) else np.nan

                if pd.notna(daily_std) and daily_std > 0:
                    annualized_std = daily_std * np.sqrt(252)
                else:
                    annualized_std = np.nan

                sharpe_ratio = np.nan
                if pd.notna(daily_std) and daily_std > 0 and pd.notna(daily_mean):
                    sharpe_ratio = (daily_mean / daily_std) * np.sqrt(252)
                elif pd.notna(daily_mean):
                    sharpe_ratio = np.inf if daily_mean > 0 else 0.0

                downside_returns = daily_returns_series[daily_returns_series < 0]
                sortino_ratio = np.nan
                if len(downside_returns) >= 2:
                    downside_std_daily = downside_returns.std(ddof=1)
                    if pd.notna(downside_std_daily) and downside_std_daily > 0 and pd.notna(daily_mean):
                        sortino_ratio = (daily_mean / downside_std_daily) * np.sqrt(252)
                elif len(downside_returns) == 0 and pd.notna(daily_mean) and daily_mean > 0:
                    sortino_ratio = np.inf
                else:
                    sortino_ratio = 0.0 if pd.notna(daily_mean) and daily_mean <= 0 else sortino_ratio

                sample_notes = []
                if trade_days < 20:
                    sample_notes.append(f"有效交易日仅 {trade_days} 天，年化结果偏差较大")
                if not used_daily_aggregation:
                    sample_notes.append("缺少有效信号日期，按样本顺序估算年化指标")
                sample_note = "；".join(sample_notes) if sample_notes else None

                max_drawdown = calc_max_drawdown(drawdown_series)
                
                group_stats.append({
                    '分组': group_id + 1,
                    '参数区间': param_range,
                    '平均收益': avg_return,
                    '收益标准差': return_std,
                    '胜率': win_rate,
                    '最大回撤': max_drawdown,
                    '交易日数量': trade_days,
                    '数据年份': year_label,
                    '观察区间': date_span,
                    '日度收益均值': daily_mean,
                    '日度收益波动': daily_std,
                    '观测期年数': observation_years,
                    '年化收益估算方式': annualization_method,
                    '年化样本提示': sample_note,
                    '年化收益率': annualized_return,
                    '年化收益标准差': annualized_std,
                    '年化夏普比率': sharpe_ratio,
                    '年化索提诺比率': sortino_ratio,
                    '样本数量': len(group_data)
                })
            
            if not group_stats:
                return None
            
            group_stats_df = pd.DataFrame(group_stats)
            
            # 计算多空收益（最高组 - 最低组）
            long_short_return = group_stats_df['年化收益率'].max() - group_stats_df['年化收益率'].min()
            
            return {
                'group_stats': group_stats_df,
                'long_short_return': long_short_return,
                'total_samples': total_samples,
                'factor_col': factor_col,
                'semantic': semantic,
            }
            
        except Exception as e:
            print(f"计算因子 {factor_col} 综合指标时出错: {e}")
            return None
    
    def calculate_ic(self, factor_col, use_pearson=None):
        """计算信息系数IC"""
        if use_pearson is None:
            use_pearson = DEFAULT_USE_PEARSON
        if not hasattr(self, 'processed_data'):
            df_clean = self.data.dropna(subset=[factor_col, self.return_col])
        else:
            df_clean = self.processed_data.dropna(subset=[factor_col, self.return_col])
        
        if len(df_clean) < 2:
            return np.nan, np.nan, np.nan, np.nan
        
        try:
            factor_values = df_clean[factor_col].values
            return_values = df_clean[self.return_col].values

            if use_pearson:
                ic = np.corrcoef(factor_values, return_values)[0, 1]
            else:
                ic = custom_spearman_corr(factor_values, return_values)
            
            # 计算IC的均值和标准差（使用滚动窗口）
            window_size = min(30, len(df_clean) // 3)
            if window_size < 5:
                return ic, np.nan, np.nan, np.nan
            
            rolling_ic = []
            for i in range(window_size, len(df_clean)):
                subset = df_clean.iloc[i-window_size:i]
                if use_pearson:
                    corr = subset[factor_col].corr(subset[self.return_col])
                else:
                    corr = custom_spearman_corr(
                        subset[factor_col].values,
                        subset[self.return_col].values
                    )
                if not np.isnan(corr):
                    rolling_ic.append(corr)
            
            if len(rolling_ic) < 2:
                return ic, np.nan, np.nan, np.nan
            
            ic_mean = np.mean(rolling_ic)
            ic_std = np.std(rolling_ic)
            
            # 计算t统计量和p值
            if ic_std > 0:
                t_stat = ic_mean / (ic_std / np.sqrt(len(rolling_ic)))
                try:
                    from scipy.stats import t
                    p_value = 2 * (1 - t.cdf(abs(t_stat), len(rolling_ic) - 1))
                except:
                    p_value = np.nan
            else:
                t_stat = np.nan
                p_value = np.nan
            
            return ic_mean, ic_std, t_stat, p_value
        
        except Exception as e:
            print(f"计算IC时出错: {e}")
            return np.nan, np.nan, np.nan, np.nan
    
    def score_factors(self, factor_results):
        """对因子进行综合评分"""
        return _fa_score_parameterized_factors(factor_results)

    def generate_parameterized_report(self):
        """生成带参数因子的详细TXT/CSV报告"""
        return _fa_generate_parameterized_report(self)

    def analyze_rolling_ic(self, factor_col, window_sizes=(30, 60), compute_ic_decay=True, save_plots=False):
        """
        针对带参数数据提供滚动IC分析便捷入口。
        """
        if getattr(self, 'processed_data', None) is None or factor_col not in self.processed_data.columns:
            print(f"滚动窗口分析失败：缺少因子 {factor_col} 的有效数据")
            return {}
        df = self.processed_data[['信号日期', factor_col, self.return_col]].dropna()
        return rolling_window_analysis(
            df,
            factor_col,
            self.return_col,
            window_sizes=window_sizes,
            compute_ic_decay=compute_ic_decay,
            save_plots=save_plots,
        )

    def analyze_sample_sensitivity(self, factor_col, sample_sizes=(0.8, 0.9, 1.0), n_iterations=100):
        """
        利用通用样本敏感性工具复核带参数分组的稳健性。
        """
        if getattr(self, 'processed_data', None) is None or factor_col not in self.processed_data.columns:
            print(f"样本敏感性分析失败：缺少因子 {factor_col} 的有效数据")
            return {}
        df = self.processed_data[[factor_col, self.return_col]].dropna()
        return sample_sensitivity_analysis(
            df,
            factor_col,
            self.return_col,
            sample_sizes=sample_sizes,
            n_iterations=n_iterations,
        )

