# -*- coding: utf-8 -*-
"""
非参数因子分析模块，从 yinzifenxi1119_split.py 拆分而来。
当前仅包含 FactorAnalysis 类的文本拷贝，为避免引入行为变动，
暂未接入主流程。
"""

import os
import re
import numpy as np
import pandas as pd
import scipy
from datetime import datetime
from typing import Sequence, Dict

from .excel_parser import load_excel_sources, DEFAULT_PARSE_CONFIG
from .fa_data_validator import DataValidator
from .fa_config import (
    DEFAULT_DATA_FILE,
    DEFAULT_DATA_FILES,
    FACTOR_COLUMNS,
    RETURN_COLUMN,
    build_report_path,
    SEGMENT_MIN_SAMPLES,
    SEGMENT_MIN_DAILY,
    MARKET_SEGMENT_RULES,
    DATA_PARSE_CONFIG,
)
from .fa_stat_utils import (
    ensure_list,
    safe_len,
    safe_ensure_list,
    custom_spearman_corr,
    robust_correlation,
    mann_whitney_u_test,
    bootstrap_confidence_interval,
    kendall_tau_corr,
    rolling_window_analysis,
    temporal_stability_analysis,
    sample_sensitivity_analysis,
    calculate_standard_annual_return,
    safe_calculate_annual_return,
    validate_annual_return_calculation,
)
from .excel_parser import FactorNormalizer, NormalizationInfo
from .fa_nonparam_helpers import (
    _fa_classify_factors_by_ic,
    _fa_generate_factor_classification_overview,
    _fa_get_suggested_weight,
    _fa_get_scoring_standards,
    compute_integrated_factor_scores,
)
from .fa_nonparam_report import (
    _fa_generate_summary_report,
    _fa_generate_factor_analysis_report,
    _fa_generate_positive_factors_analysis,
    _fa_generate_negative_factors_analysis,
)

try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except Exception:
    HAS_PLOT = False
    plt = None
    print("警告: matplotlib不可用，相关可视化功能将被跳过")

try:
    from scipy import stats
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False
    stats = None
    print("警告: scipy不可用，部分统计功能将被简化")


# 隐式的预处理默认参数（符合A股常用实践，无需用户干预）
DEFAULT_PROCESS_FACTORS = True
DEFAULT_FACTOR_METHOD = 'standardize'
DEFAULT_WINSORIZE = True
DEFAULT_WINSORIZE_LIMITS = (0.01, 0.99)
DEFAULT_USE_PEARSON = False
DEFAULT_GROUP_COUNT = 10


# 注意：此类目前只是被动拷贝，真正运行的实现仍位于 yinzifenxi1119_split.py。
# 当前仍以 yinzifenxi1119_split.py 中的版本为准。



class FactorAnalysis:
    def __init__(self, file_path=None, data=None):
        """
        初始化因子分析类
        
        Args:
            file_path: 数据文件路径（Excel或CSV）
            data: 直接传入的DataFrame数据
        """
        # 支持单文件或多文件输入（列表/元组），默认使用配置中的所有文件
        default_paths = DEFAULT_DATA_FILES if file_path is None else file_path
        if isinstance(default_paths, (list, tuple, set)):
            resolved_paths = [str(p) for p in default_paths if p]
        elif default_paths:
            resolved_paths = [str(default_paths)]
        else:
            resolved_paths = []
        if not resolved_paths and DEFAULT_DATA_FILE:
            resolved_paths = [DEFAULT_DATA_FILE]
        self.file_paths = resolved_paths
        self.file_path = self.file_paths[0] if self.file_paths else None
        self.data = data
        enabled_flags = DATA_PARSE_CONFIG.get("column_enabled", {}) or {}
        self.factors = [
            factor for factor in FACTOR_COLUMNS
            if enabled_flags.get(factor, "是") != "否"
        ]
        self.disabled_factors = [
            factor for factor in FACTOR_COLUMNS
            if enabled_flags.get(factor, "是") == "否"
        ]
        self.return_col = RETURN_COLUMN
        self.analysis_results = {}
        self.segment_overview = {}
        self.parse_config = DEFAULT_PARSE_CONFIG.copy()
        if DATA_PARSE_CONFIG:
            self.parse_config.update(DATA_PARSE_CONFIG)
        
        # 初始化异常统计数据
        self.anomaly_stats = {
            'factor_processing': {},
            'missing_values': {},
            'outliers': {},
            'unique_value_check': {},
            'duplicate_rows': {},
            'data_cleaning': {},
            'ic_calculation': {}
        }
        self.parse_diagnostics = []
        self.unavailable_factors = set()
        self.loaded_sources = []
        if getattr(self, "disabled_factors", None):
            print("[INFO] 以下因子因配置被禁用: " + ", ".join(self.disabled_factors))
        self.normalizer = FactorNormalizer()
        self.normalization_stats: Dict[str, NormalizationInfo] = {}
        
        # 如果没有直接传入数据且有文件路径，则加载数据
        if self.data is None and self.file_path:
            self.load_data()

    def _determine_market_segment(self, stock_code):
        """
        根据证券代码判断所属板块：主板、创业板、科创板、北交所等。
        """
        if stock_code is None or (isinstance(stock_code, float) and np.isnan(stock_code)):
            return "未知"
        code_str = str(stock_code).strip().upper()
        if not code_str:
            return "未知"
        normalized = code_str.replace('.', '')
        digits = "".join(ch for ch in normalized if ch.isdigit())

        def starts_with_any(value, prefixes):
            return any(value.startswith(p) for p in prefixes)

        for segment, alpha_prefixes, digit_prefixes in MARKET_SEGMENT_RULES:
            if alpha_prefixes and starts_with_any(normalized, alpha_prefixes):
                return segment
            if digits and digit_prefixes and starts_with_any(digits, digit_prefixes):
                return segment

        return "其他"

    def _ensure_market_segment_column(self, df):
        """
        确保预处理数据中包含 market_segment 列，并统计板块样本占比。
        """
        if 'market_segment' not in df.columns:
            df['market_segment'] = df['股票代码'].apply(self._determine_market_segment)
        segment_counts = df['market_segment'].value_counts(dropna=False)
        total = float(len(df)) if len(df) else 1.0
        self.segment_overview = {
            seg: {
                'count': int(count),
                'ratio': float(count) / total if total else 0.0
            }
            for seg, count in segment_counts.items()
        }
        return df
    
    def _calculate_adaptive_annual_returns(self, avg_returns, characteristics, method_info):
        """
        执行自适应年化计算 - 优化版本
        
        核心改进：
        1. 使用标准复利年化方法作为主要计算方式
        2. 保留CAGR方法作为对比方法
        3. 删除线性年化方法（忽视复利效应）
        4. 增强数据特征分析和验证机制
        
        Args:
            avg_returns: 平均收益数据，包含分组收益信息
            characteristics: 数据特征分析结果
            method_info: 选择的年化方法
            
        Returns:
            dict: 年化计算结果，包含标准复利年化、CAGR对比和验证信息
        """
        try:
            print(f"    [处理] 开始自适应年化计算优化...")
            
            # 持股周期和数据特征
            holding_period = characteristics['holding_period_days']
            observation_years = characteristics['observation_period_years']
            total_trades = characteristics['total_trades']
            
            # 步骤1: 计算持股周期总收益率
            # 使用分组平均收益作为持股周期收益率
            period_total_returns = avg_returns['平均收益']
            
            # 步骤2: 应用标准复利年化方法（主要方法）
            print(f"    [统计] 应用标准复利年化方法...")
            standard_annual_returns = []
            validation_results = []
            
            for i, total_return in enumerate(period_total_returns):
                # 修复4: 使用安全版本的年化计算函数，防止类型错误
                annual_return, _ = safe_calculate_annual_return(total_return, observation_years)
                standard_annual_returns.append(annual_return)
                
                # 验证计算结果的正确性
                validation = validate_annual_return_calculation(annual_return, observation_years, total_return)
                validation_results.append(validation)
                
                # 输出验证结果（前3组详细显示）
                if i < 3 and validation['valid']:
                    print(f"      组{i+1}: 总收益{total_return:.4f} -> 年化{annual_return:.4f} ({annual_return*100:.2f}%)")
                elif i < 3:
                    print(f"      组{i+1}: 验证失败 - {validation.get('error', '未知错误')}")
            
            # 转换为numpy数组
            standard_annual_returns = np.array(standard_annual_returns)
            
            # 步骤3: 计算CAGR复合年化（对比方法）
            print(f"    [上升] 计算CAGR复合年化（对比方法）...")
            cagr_annual_returns = []
            
            for total_return in period_total_returns:
                # 修复4: 使用安全版本的年化计算函数，防止类型错误
                # CAGR计算：(1 + 总收益率)^(1/年数) - 1
                # 这与标准复利年化是相同的数学公式
                cagr_return, _ = safe_calculate_annual_return(total_return, observation_years)
                cagr_annual_returns.append(cagr_return)
            
            cagr_annual_returns = np.array(cagr_annual_returns)
            
            # 步骤4: 移除线性年化方法（方法A/B）
            # 不再计算传统的线性年化收益率
            
            # 步骤5: 计算年化风险指标
            print(f"    [下降] 计算年化风险指标...")
            daily_std_returns = avg_returns['收益标准差'] / holding_period
            
            # 使用标准复利年化收益率计算风险指标
            # 年化标准差：考虑复利效应的波动率调整
            annual_std = daily_std_returns * np.sqrt(observation_years * 252 / holding_period)

            # 修复2: 数组形状兼容性检查，防止broadcast错误
            # 确保annual_std是1D数组
            if annual_std.ndim > 1:
                annual_std = annual_std.flatten()
            # 确保standard_annual_returns是1D数组
            if standard_annual_returns.ndim > 1:
                standard_annual_returns = standard_annual_returns.flatten()

            # 年化夏普比率（基于标准复利年化收益率）
            annual_sharpe = np.where(annual_std > 0,
                                    standard_annual_returns / annual_std,
                                    0.0)

            # 年化索提诺比率（简化处理，下行风险使用标准差代替）
            annual_sortino = np.where(annual_std > 0,
                                     standard_annual_returns / annual_std,
                                     0.0)
            
            # 步骤6: 数据质量评估和验证统计
            print(f"    [OK] 数据质量评估...")
            valid_annual_returns = standard_annual_returns[np.isfinite(standard_annual_returns)]
            
            quality_stats = {
                'total_groups': len(standard_annual_returns),
                'valid_groups': len(valid_annual_returns),
                'validation_success_rate': sum(v['valid'] for v in validation_results) / len(validation_results) if validation_results else 0,
                'mean_annual_return': np.mean(valid_annual_returns) if len(valid_annual_returns) > 0 else np.nan,
                'std_annual_return': np.std(valid_annual_returns) if len(valid_annual_returns) > 0 else np.nan,
                'min_annual_return': np.min(valid_annual_returns) if len(valid_annual_returns) > 0 else np.nan,
                'max_annual_return': np.max(valid_annual_returns) if len(valid_annual_returns) > 0 else np.nan
            }
            
            print(f"      有效年化计算: {quality_stats['valid_groups']}/{quality_stats['total_groups']}")
            print(f"      验证成功率: {quality_stats['validation_success_rate']:.1%}")
            print(f"      平均年化收益率: {quality_stats['mean_annual_return']*100:.2f}%" if not np.isnan(quality_stats['mean_annual_return']) else "      平均年化收益率: N/A")
            
            # 步骤7: 构建结果字典
            results = {
                # 主要结果（标准复利年化）
                'standard_compound_annual_return': standard_annual_returns,
                'main_annual_return': standard_annual_returns,  # 保持兼容性
                
                # 对比方法（CAGR）
                'cagr_annual_return': cagr_annual_returns,
                
                # 风险指标
                'annual_std': annual_std,
                'annual_sharpe': annual_sharpe,
                'annual_sortino': annual_sortino,
                
                # 基础数据
                'daily_avg_returns': period_total_returns / holding_period,
                'base_frequency': method_info['frequency_base'],
                'daily_std_returns': daily_std_returns,
                
                # 新增：数据质量信息
                'quality_stats': quality_stats,
                'validation_results': validation_results,
                'observation_years': observation_years,
                
                # 计算方法标识
                'calculation_method': 'standard_compound',
                'comparison_method': 'cagr_based',
                'deprecated_methods': ['linear_annual_return_a', 'linear_annual_return_b']
            }
            
            print(f"    [OK] 自适应年化计算完成")
            return results
            
        except Exception as e:
            print(f"    [ERROR] 年化计算出错: {str(e)}")
            # 返回安全的默认值
            n_groups = len(avg_returns)
            return {
                'standard_compound_annual_return': np.full(n_groups, np.nan),
                'main_annual_return': np.full(n_groups, np.nan),
                'cagr_annual_return': np.full(n_groups, np.nan),
                'annual_std': np.full(n_groups, np.nan),
                'annual_sharpe': np.full(n_groups, np.nan),
                'annual_sortino': np.full(n_groups, np.nan),
                'daily_avg_returns': np.full(n_groups, np.nan),
                'base_frequency': np.nan,
                'daily_std_returns': np.full(n_groups, np.nan),
                'quality_stats': {'error': str(e)},
                'validation_results': [],
                'observation_years': np.nan,
                'calculation_method': 'error_fallback',
                'comparison_method': 'error_fallback',
                'deprecated_methods': []
            }
    
    def load_data(self):
        """从文件加载数据（支持多文件合并）"""
        if self.data is not None:
            return True
        if not self.file_paths:
            print("[ERROR] 未找到任何数据文件路径，请检查配置")
            return False

        try:
            parsed = load_excel_sources(self.file_paths, self.parse_config)
        except Exception as exc:
            print(f"[ERROR] Excel 解析失败: {exc}")
            return False

        if parsed.data is None or parsed.data.empty:
            print("[ERROR] Excel 解析结果为空，请检查数据文件内容")
            return False

        self.data = parsed.data
        self.parse_diagnostics = parsed.diagnostics
        self.unavailable_factors = set(parsed.unavailable_columns or [])
        self.loaded_sources = [
            {
                'path': diag.file_path,
                'file_name': os.path.basename(diag.file_path),
                'columns': set(diag.present_columns),
                'sheet': diag.sheet_name,
                'rows': getattr(diag, 'rows', 0) or 0,
                'year': self._infer_year_from_name(diag.file_path),
            }
            for diag in self.parse_diagnostics
        ]

        merged_count = len(parsed.diagnostics)
        print(
            f"[INFO] 数据加载成功: {len(self.data)} 行, 列数 {self.data.shape[1]} "
            f"(解析 {merged_count} 个工作表)"
        )
        if self.unavailable_factors:
            print(
                "[WARN] 以下因子列在解析过程中无法可靠转换，已暂时跳过: "
                + ", ".join(sorted(self.unavailable_factors))
            )
        return True

    def validate_data_sources(self):
        """
        在预处理之前执行数据完整性验证，确保所有表格均成功读取并覆盖预期年份。
        """
        validator = DataValidator(
            self.file_paths,
            getattr(self, "parse_diagnostics", []),
            self.data,
            required_columns=[self.return_col, *self.factors],
        )
        result = validator.run()

        def _emit(line: str):
            if line is None:
                return
            text = line.rstrip()
            if not text:
                return
            text = text.lstrip("\n")
            stripped = text.lstrip()
            needs_info_prefix = True
            if stripped.startswith(("[INFO]", "[WARN]", "[ERROR]", "[STEP]", "[OK]", "[DEBUG]")):
                needs_info_prefix = False
            elif stripped.startswith("[VALIDATION]"):
                needs_info_prefix = True
            elif stripped.startswith("["):
                needs_info_prefix = False
            if needs_info_prefix:
                text = f"[INFO] {text}"
            print(text)

        for line in result.report_lines:
            _emit(line)
        return result.passed

    def _infer_year_from_name(self, path):
        try:
            filename = os.path.basename(path or "")
        except Exception:
            filename = str(path)
        match = re.search(r"(20\d{2})", filename)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        return None

    def validate_processed_coverage(self, min_years=5, min_trading_days=800):
        """
        预处理后再次校验样本覆盖范围，防止因缺失值导致仅剩单一年度。
        """
        lines = ["\n[VALIDATION] === 样本覆盖复核 ==="]
        df = getattr(self, 'processed_data', None)
        if df is None or df.empty:
            lines.append("  - 预处理结果为空，无法继续分析")
            for line in lines:
                print(line)
            return False

        if '信号日期' not in df.columns:
            lines.append("  - 缺少信号日期列，无法评估样本覆盖")
            for line in lines:
                print(line)
            return False

        valid_dates = pd.to_datetime(df['信号日期'], errors='coerce').dropna()
        year_counts = valid_dates.dt.year.dropna().value_counts().sort_index()
        unique_years = list(year_counts.index)
        unique_days = int(valid_dates.dt.normalize().nunique())
        lines.append(
            f"  - 预处理后覆盖年份: {', '.join(str(y) for y in unique_years) if unique_years else '无'}"
            f" (共 {len(unique_years)} 年)，覆盖交易日 {unique_days} 天"
        )

        factor_alerts = []
        for factor in self.factors:
            factor_df = df[['信号日期', factor]].dropna()
            if factor_df.empty:
                factor_alerts.append({
                    'factor': factor,
                    'message': "无有效样本",
                    'missing_years': unique_years,
                    'factor_years': [],
                })
                continue
            factor_years = sorted(pd.to_datetime(factor_df['信号日期'], errors='coerce').dropna().dt.year.unique())
            if len(factor_years) < min_years:
                missing_years = sorted(set(unique_years) - set(factor_years))
                factor_alerts.append({
                    'factor': factor,
                    'message': f"仅覆盖 {len(factor_years)} 年",
                    'missing_years': missing_years,
                    'factor_years': factor_years,
                })

        total_source_rows = sum(src.get('rows', 0) for src in getattr(self, 'loaded_sources', [])) or len(self.data) or 1

        for alert in factor_alerts:
            factor = alert['factor']
            missing_years = alert.get('missing_years', [])
            factor_years = alert.get('factor_years', [])
            lines.append(
                f"    · 因子 {factor}: 已覆盖年份 {', '.join(str(y) for y in factor_years) if factor_years else '无'}，"
                f"缺失年份 {', '.join(str(y) for y in missing_years) if missing_years else '未知'}（{alert.get('message')}）"
            )
            for year in missing_years:
                year_sample = year_counts.get(year, 0)
                year_share = (year_sample / len(df)) if len(df) else 0
                lines.append(
                    f"      - 年度 {year}: 预处理样本 {year_sample} 行，占全量 {year_share:.1%}"
                )
                sources = [
                    src for src in getattr(self, 'loaded_sources', [])
                    if src.get('year') == year
                ]
                if not sources:
                    lines.append("        · 未找到对应源文件信息（可能无法识别年份）")
                    continue
                for src in sources:
                    src_rows = src.get('rows', 0)
                    src_share = (src_rows / total_source_rows) if total_source_rows else 0
                    has_factor = factor in src.get('columns', set())
                    reason = "缺少该因子列" if not has_factor else "列存在但有效记录缺失"
                    lines.append(
                        f"        · 文件 {src.get('file_name', '未知')}:{src_rows} 行，占合并数据 {src_share:.1%}，{reason}"
                    )

        passed = True
        if len(unique_years) < min_years:
            lines.append("  - [FAIL] 信号日期年份少于要求")
            passed = False
        if unique_days < min_trading_days:
            lines.append("  - [FAIL] 覆盖交易日过少，无法代表完整区间")
            passed = False
        if factor_alerts:
            lines.append("  - [FAIL] 存在因子在多年份缺失，结果失真")
            passed = False

        if passed:
            lines.append("[VALIDATION] 样本覆盖复核通过")
        else:
            lines.append("[VALIDATION] 样本覆盖复核未通过")

        for line in lines:
            print(line)
        return passed

    def _report_missing_columns(self, missing_cols: Sequence[str]):
        diag_list = getattr(self, 'parse_diagnostics', [])
        if diag_list:
            for col in missing_cols:
                missing_files = [
                    f"{os.path.basename(diag.file_path)}[{diag.sheet_name}]"
                    for diag in diag_list
                    if col not in diag.present_columns
                ]
                if missing_files:
                    print(f"  -> 列 '{col}' 在以下文件/工作表缺失: {', '.join(missing_files)}")
        elif getattr(self, "loaded_sources", None):
            for col in missing_cols:
                missing_files = [
                    os.path.basename(src['path'])
                    for src in self.loaded_sources
                    if col not in src.get('columns', set())
                ]
                if missing_files:
                    print(f"  -> 列 '{col}' 在以下文件中缺失: {', '.join(missing_files)}")

    def _get_column_diagnostics(self, column_name: str):
        for diag in getattr(self, "parse_diagnostics", []):
            columns = getattr(diag, "present_columns", []) or []
            if column_name in columns:
                return diag
        return None

    @staticmethod
    def _fallback_numeric_conversion(series: pd.Series) -> pd.Series:
        """旧版字符串→数值转换逻辑，供Normalizer失败时使用。"""
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
    
    def apply_factor_processing(self, df, factor_col, method='standardize', winsorize=True, winsorize_limits=(0.01, 0.99)):
        """
        对因子数据进行处理（标准化和缩尾处理）
        
        Args:
            df: 数据框
            factor_col: 因子列名
            method: 处理方法，'standardize'（标准化）或 'normalize'（归一化）
            winsorize: 是否进行缩尾处理
            winsorize_limits: 缩尾处理的分位数范围
            
        Returns:
            处理后的数据框
        """
        # 复制数据以避免修改原始数据
        processed_df = df.copy()
        
        # 记录处理前的统计信息
        original_stats = {
            'mean': processed_df[factor_col].mean(),
            'std': processed_df[factor_col].std(),
            'min': processed_df[factor_col].min(),
            'max': processed_df[factor_col].max(),
            'q1': processed_df[factor_col].quantile(0.25),
            'q3': processed_df[factor_col].quantile(0.75)
        }
        
        # 缩尾处理
        if winsorize:
            lower_limit = processed_df[factor_col].quantile(winsorize_limits[0])
            upper_limit = processed_df[factor_col].quantile(winsorize_limits[1])
            processed_df[factor_col] = processed_df[factor_col].clip(lower=lower_limit, upper=upper_limit)
            
            # 记录缩尾处理信息（安全检查确保键存在）
            if factor_col not in self.anomaly_stats['factor_processing']:
                self.anomaly_stats['factor_processing'][factor_col] = {}
            winsorized_count = ((df[factor_col] < lower_limit) | (df[factor_col] > upper_limit)).sum()
            self.anomaly_stats['factor_processing'][factor_col]['winsorized_count'] = winsorized_count
            self.anomaly_stats['factor_processing'][factor_col]['winsorize_limits'] = winsorize_limits
        
        # 标准化或归一化处理
        if method == 'standardize':
            # 标准化：(x - mean) / std
            mean_val = processed_df[factor_col].mean()
            std_val = processed_df[factor_col].std()
            if std_val > 0:
                processed_df[factor_col] = (processed_df[factor_col] - mean_val) / std_val
                if factor_col not in self.anomaly_stats['factor_processing']:
                    self.anomaly_stats['factor_processing'][factor_col] = {}
                self.anomaly_stats['factor_processing'][factor_col]['method'] = 'standardize'
                self.anomaly_stats['factor_processing'][factor_col]['params'] = {'mean': mean_val, 'std': std_val}
            else:
                print(f"警告：因子 {factor_col} 标准差为0，无法标准化")
                if factor_col not in self.anomaly_stats['factor_processing']:
                    self.anomaly_stats['factor_processing'][factor_col] = {}
                self.anomaly_stats['factor_processing'][factor_col]['method'] = 'none'
                self.anomaly_stats['factor_processing'][factor_col]['params'] = {'reason': 'std=0'}
                
        elif method == 'normalize':
            # 归一化：(x - min) / (max - min)
            min_val = processed_df[factor_col].min()
            max_val = processed_df[factor_col].max()
            if max_val > min_val:
                processed_df[factor_col] = (processed_df[factor_col] - min_val) / (max_val - min_val)
                if factor_col not in self.anomaly_stats['factor_processing']:
                    self.anomaly_stats['factor_processing'][factor_col] = {}
                self.anomaly_stats['factor_processing'][factor_col]['method'] = 'normalize'
                self.anomaly_stats['factor_processing'][factor_col]['params'] = {'min': min_val, 'max': max_val}
            else:
                print(f"警告：因子 {factor_col} 最大值等于最小值，无法归一化")
                if factor_col not in self.anomaly_stats['factor_processing']:
                    self.anomaly_stats['factor_processing'][factor_col] = {}
                self.anomaly_stats['factor_processing'][factor_col]['method'] = 'none'
                self.anomaly_stats['factor_processing'][factor_col]['params'] = {'reason': 'max=min'}
        
        # 记录处理后的统计信息
        processed_stats = {
            'mean': processed_df[factor_col].mean(),
            'std': processed_df[factor_col].std(),
            'min': processed_df[factor_col].min(),
            'max': processed_df[factor_col].max(),
            'q1': processed_df[factor_col].quantile(0.25),
            'q3': processed_df[factor_col].quantile(0.75)
        }
        
        # 记录处理后的统计信息
        if factor_col not in self.anomaly_stats['factor_processing']:
            self.anomaly_stats['factor_processing'][factor_col] = {}
        if factor_col not in self.anomaly_stats['factor_processing']:
            self.anomaly_stats['factor_processing'][factor_col] = {}
        self.anomaly_stats['factor_processing'][factor_col]['original_stats'] = original_stats
        self.anomaly_stats['factor_processing'][factor_col]['processed_stats'] = processed_stats
        
        return processed_df
    
    def preprocess_data(
        self,
        process_factors=None,
        factor_method=None,
        winsorize=None,
        winsorize_limits=None,
    ):
        """
        数据预处理方法 - 处理百分比字符串和数值转换

        Args:
            process_factors: 是否处理因子数据；默认采用A股标准（标准化）
            factor_method: 因子处理方法 ('standardize', 'normalize', 'rank')
            winsorize: 是否进行缩尾处理（默认开启）
            winsorize_limits: 缩尾处理的上下限分位数

        Returns:
            bool: 预处理是否成功
        """
        if self.data is None or self.data.empty:
            print("错误: 没有数据可处理")
            return False
        
        try:
            # 应用默认的A股通用预处理方式（允许调用者传入特殊需求时覆盖）
            if process_factors is None:
                process_factors = DEFAULT_PROCESS_FACTORS
            if factor_method is None:
                factor_method = DEFAULT_FACTOR_METHOD
            if winsorize is None:
                winsorize = DEFAULT_WINSORIZE
            if winsorize_limits is None:
                winsorize_limits = DEFAULT_WINSORIZE_LIMITS

            # 复制数据
            df = self.data.copy()
            
            # 初始化异常数据统计信息字典
            self.anomaly_stats = {
                'missing_values': {},
                'outliers': {},
                'unique_value_check': {},
                'duplicate_rows': {},
                'factor_processing': {}
            }
            
            # 检查必要的列是否存在
            active_factors = [f for f in self.factors if f not in self.unavailable_factors]
            required_cols = ['股票代码', '股票名称', '信号日期', self.return_col]
            required_cols.extend(active_factors)
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"错误：缺少必要的列: {missing_cols}")
                self._report_missing_columns(missing_cols)
                return False
            
            self.normalization_stats = {}
            columns_to_normalize = list(dict.fromkeys(active_factors + [self.return_col]))
            for col in columns_to_normalize:
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
            
            # 确保日期列正确处理
            if '信号日期' in df.columns:
                try:
                    df['信号日期'] = pd.to_datetime(df['信号日期'], errors='coerce')
                except:
                    print("警告：无法转换信号日期列")
            
            # 处理数值型因子并记录异常信息
            for factor in self.factors:
                if factor in self.unavailable_factors:
                    print(f"[WARN] 因子 '{factor}' 在解析阶段被标记为不可用，已跳过处理")
                    continue
                print(f"处理因子: {factor}")
                # 初始化因子处理统计信息
                self.anomaly_stats['factor_processing'][factor] = {
                    'processed': process_factors,
                    'method': factor_method,
                    'params': {'winsorize': winsorize, 'winsorize_limits': winsorize_limits}
                }
                
                # 保证因子列为数值型（Normalizer 失败时已回退）
                if not pd.api.types.is_numeric_dtype(df[factor]):
                    df[factor] = pd.to_numeric(df[factor], errors='coerce')
                
                # 记录缺失值信息
                missing_count = df[factor].isna().sum()
                self.anomaly_stats['missing_values'][factor] = missing_count
                if missing_count > 0:
                    print(f"  因子 {factor} 有 {missing_count} 个缺失值 ({missing_count/len(df):.2%})")
                
                # 3倍标准差检测异常值（仅记录不删除）
                valid_data = df[factor].dropna()
                if len(valid_data) > 0:
                    mean_val = valid_data.mean()
                    std_val = valid_data.std()
                    lower_bound_std = mean_val - 3 * std_val
                    upper_bound_std = mean_val + 3 * std_val
                    outlier_count_std = len(valid_data[(valid_data < lower_bound_std) | (valid_data > upper_bound_std)])
                    
                    self.anomaly_stats['outliers'][factor] = {
                        'count': outlier_count_std,
                        'percentage': outlier_count_std/len(df) if len(df) > 0 else 0,
                        'bounds': {'lower': lower_bound_std, 'upper': upper_bound_std}
                    }
                    
                    if outlier_count_std > 0:
                        print(f"  因子 {factor}: 检测到 {outlier_count_std} 个3倍标准差异常值 ({outlier_count_std/len(df):.2%})")
                
                # 唯一值比例检查（低唯一值比例可能表示数据问题）
                if df[factor].nunique() < len(df) * 0.05 and len(df) > 0:
                    unique_pct = df[factor].nunique() / len(df)
                    self.anomaly_stats['unique_value_check'][factor] = unique_pct
                    print(f"  警告：因子 {factor} 唯一值比例过低 ({unique_pct:.2%})，可能存在数据质量问题")
                
                # 如果需要对因子进行处理
                if process_factors:
                    print(f"  对因子 {factor} 进行 {factor_method} 处理")
                    df = self.apply_factor_processing(df, factor, factor_method, winsorize, winsorize_limits)
            
            # 确保收益率列为数值型
            if not pd.api.types.is_numeric_dtype(df[self.return_col]):
                df[self.return_col] = pd.to_numeric(df[self.return_col], errors='coerce')
            
            # 记录收益率列的缺失值和异常值
            missing_return_count = df[self.return_col].isna().sum()
            self.anomaly_stats['missing_values'][self.return_col] = missing_return_count
            if missing_return_count > 0:
                print(f"收益率列 {self.return_col} 有 {missing_return_count} 个缺失值")
            
            # 记录重复行信息
            duplicate_rows = df.duplicated().sum()
            self.anomaly_stats['duplicate_rows']['count'] = duplicate_rows
            if duplicate_rows > 0:
                print(f"检测到 {duplicate_rows} 行重复数据")
            
            # 注意：根据用户要求，我们不删除任何数据，只记录异常信息
            # 仅删除收益率和因子列的缺失值行，以确保分析有意义的数据
            original_len = len(df)
            df = df.dropna(subset=[self.return_col])
            self.anomaly_stats['missing_values']['total_removed'] = original_len - len(df)
            
            if len(df) < original_len:
                print(f"数据清理：删除 {original_len - len(df)} 行缺失值")
            
            # 记录原始数据统计信息
            self.anomaly_stats['original_data_count'] = original_len
            self.anomaly_stats['analyzed_data_count'] = len(df)
            
            # 简化样本筛选信息输出
            final_count = len(df)
            removed_count = original_len - final_count
            if removed_count > 0:
                print(f"数据预处理完成：保留 {final_count}/{original_len} 行有效数据")
            
            df = self._ensure_market_segment_column(df)
            self.processed_data = df
            print(f"[INFO] 数据预处理完成：可用样本 {len(df)} 行，记录因子 {len(self.factors)} 个")
            return True

        except Exception as e:
            print(f"[ERROR] 数据预处理失败: {e}")
            return False
    
    def calculate_ic(self, factor_col, use_pearson=None, use_robust_corr=False, use_kendall=False,
                     use_nonparam_test=False, compute_bootstrap_ci=False, n_bootstrap=1000):
        """
        计算因子IC值，支持多种稳健性统计方法
        
        Args:
            factor_col: 因子列名
            use_pearson: 是否使用Pearson相关系数（默认从配置读取，默认为Spearman）
            use_robust_corr: 是否使用稳健相关系数（Spearman + Kendall组合），默认为False
            use_kendall: 是否使用Kendall's Tau相关系数，默认为False
            use_nonparam_test: 是否进行非参数检验，默认为False
            compute_bootstrap_ci: 是否计算Bootstrap置信区间，默认为False
            n_bootstrap: Bootstrap重抽样次数，默认1000次
            
        Returns:
            tuple: (IC均值, IC标准差, t统计量, p值, 额外统计结果字典)
        """
        # 使用预处理后的数据，确保因子处理生效
        df = self.processed_data if hasattr(self, 'processed_data') and self.processed_data is not None else self.data.copy()
        if 'market_segment' not in df.columns:
            df = self._ensure_market_segment_column(df)
            if hasattr(self, 'processed_data') and self.processed_data is not None:
                self.processed_data = df
        
        # 确保数据有效
        if df.empty or factor_col not in df.columns or self.return_col not in df.columns:
            print(f"警告: 数据为空或列名不存在")
            return (np.nan, np.nan, np.nan, np.nan, {})
        
        corr_type = "Pearson" if use_pearson else "Spearman"
        print(f"计算因子 {factor_col} 的 {corr_type} IC值")
        
        daily_sample_counts = []
        for date, group in df.groupby('信号日期'):
            valid_data = group.dropna(subset=[factor_col, self.return_col])
            daily_sample_counts.append(len(valid_data))
        avg_daily_samples = float(np.mean(daily_sample_counts)) if daily_sample_counts else 0.0
        median_daily_samples = float(np.median(daily_sample_counts)) if daily_sample_counts else 0.0
        p25 = float(np.percentile(daily_sample_counts, 25)) if daily_sample_counts else 0.0
        p75 = float(np.percentile(daily_sample_counts, 75)) if daily_sample_counts else 0.0
        daily_sample_std = float(np.std(daily_sample_counts, ddof=1)) if len(daily_sample_counts) > 1 else 0.0
        daily_sample_cv = (daily_sample_std / avg_daily_samples) if avg_daily_samples > 0 else np.nan
        total_sample_volume = float(np.sum(daily_sample_counts)) if daily_sample_counts else 0.0
        sorted_counts = sorted(daily_sample_counts, reverse=True)
        top5 = sum(sorted_counts[:5]) if sorted_counts else 0.0
        top5_share = (top5 / total_sample_volume) if total_sample_volume > 0 else 0.0

        if avg_daily_samples >= 8:
            min_samples_per_day = 5
            ic_window_days = 1
            mode = "高样本量模式"
        elif avg_daily_samples >= 4:
            min_samples_per_day = 3
            ic_window_days = 2
            mode = "中样本量模式"
        else:
            min_samples_per_day = 2
            ic_window_days = 3
            mode = "低样本量模式"

        total_dates = len(df['信号日期'].dropna().unique())
        daily_ics = []
        skipped_dates = 0
        extra_stats = {
            'daily_points': 0,
            'ic_mode': "daily" if ic_window_days == 1 else "daily_window",
            'avg_daily_samples': avg_daily_samples,
            'median_daily_samples': median_daily_samples,
            'p25_daily_samples': p25,
            'p75_daily_samples': p75,
            'min_samples_per_day': min_samples_per_day,
            'ic_window_days': ic_window_days,
            'screening_mode': mode,
            'total_dates': total_dates,
            'skipped_dates': 0,
            'skip_ratio': 0.0,
            'daily_sample_std': daily_sample_std,
            'daily_sample_cv': daily_sample_cv,
            'daily_top5_share': top5_share,
            'avg_trades_per_day': avg_daily_samples,
            'trade_count_cv': daily_sample_cv,
            'ic_low_sample_mode': mode == "低样本量模式",
            'ic_window_note': f"{mode}, 窗口{ic_window_days}日, min样本{min_samples_per_day}",
            'daily_sample_volume': total_sample_volume,
        }
        factor_valid_df = df.dropna(subset=[factor_col, self.return_col]).copy()
        total_factor_samples = float(len(factor_valid_df))
        segment_counts_series = factor_valid_df['market_segment'].value_counts(dropna=False)
        extra_stats['segment_counts'] = {
            seg: {
                'count': int(count),
                'ratio': (float(count) / total_factor_samples) if total_factor_samples else 0.0
            }
            for seg, count in segment_counts_series.items()
        }
        extra_stats['segment_metrics'] = {}
        extra_stats['segment_warnings'] = []
        extra_stats['segment_primary'] = None
        extra_stats['segment_primary_ratio'] = 0.0
        extra_stats['segment_secondary'] = None
        extra_stats['segment_secondary_ratio'] = 0.0
        extra_stats['segment_primary_ic'] = None
        extra_stats['segment_secondary_ic'] = None
        segment_ratios = [
            info.get('ratio', 0.0)
            for info in extra_stats['segment_counts'].values()
            if isinstance(info, dict)
        ]
        extra_stats['segment_concentration'] = max(segment_ratios) if segment_ratios else 0.0
        extra_stats['segment_total'] = len(extra_stats['segment_counts'])
        extra_stats['segment_valid_count'] = 0

        print(
            f"  [IC] 样本统计: 日均 {avg_daily_samples:.2f} | 中位 {median_daily_samples:.2f} | "
            f"P25 {p25:.2f} | P75 {p75:.2f} → min_samples={min_samples_per_day}, "
            f"窗口={ic_window_days}日（{mode}）"
        )

        def _compute_overall_reference(silent=False, include_diagnostics=False, source_df=None):
            """
            计算整体IC引用值，可作为评分与回退时的备用数据。
            """
            try:
                working_df = source_df if source_df is not None else df
                if working_df is None or working_df.empty:
                    return None
                valid_data = working_df.dropna(subset=[factor_col, self.return_col])
                factor_data = np.asarray(valid_data[factor_col], dtype=float)
                return_data = np.asarray(valid_data[self.return_col], dtype=float)
                valid_mask = np.isfinite(factor_data) & np.isfinite(return_data)
                factor_data = factor_data[valid_mask]
                return_data = return_data[valid_mask]

                if source_df is None:
                    local_avg_samples = avg_daily_samples
                else:
                    unique_dates = working_df['信号日期'].dropna().unique()
                    date_count = len(unique_dates) or 1
                    local_avg_samples = float(len(valid_data)) / float(date_count)

                if local_avg_samples >= 5:
                    min_overall_samples = 25
                    min_factor_variability = 5
                    min_return_variability = 5
                elif local_avg_samples >= 3:
                    min_overall_samples = 15
                    min_factor_variability = 3
                    min_return_variability = 3
                else:
                    min_overall_samples = 10
                    min_factor_variability = 2
                    min_return_variability = 2

                if len(factor_data) < min_overall_samples:
                    if not silent:
                        print(f"  警告: 整体数据量不足，无法计算有效的整体IC值")
                    return None

                factor_variability = len(np.unique(factor_data))
                return_variability = len(np.unique(return_data))
                if factor_variability < min_factor_variability or return_variability < min_return_variability:
                    if not silent:
                        print(f"  警告: 整体数据变异性不足 - 因子唯一值: {factor_variability}, 收益率唯一值: {return_variability}")
                    return None

                if use_pearson:
                    overall_ic = np.corrcoef(factor_data, return_data)[0, 1] if len(factor_data) >= 2 else np.nan
                elif HAS_SCIPY:
                    from scipy.stats import spearmanr
                    overall_ic, _ = spearmanr(factor_data, return_data) if len(factor_data) >= 2 else (np.nan, None)
                else:
                    try:
                        factor_rank = pd.Series(factor_data).rank().values
                        return_rank = pd.Series(return_data).rank().values
                        overall_ic = np.corrcoef(factor_rank, return_rank)[0, 1]
                    except Exception:
                        overall_ic = np.nan

                if np.isnan(overall_ic) or not np.isfinite(overall_ic):
                    if not silent:
                        print("  警告: 无法计算有效的整体IC值")
                    return None

                mode_label = "Pearson" if use_pearson else "Spearman"
                prefix = "[参考] " if silent else ""
                print(f"  {prefix}成功计算整体IC值: {overall_ic:.6f} ({mode_label}, 样本 {len(factor_data)})")

                n = len(factor_data)
                ic_std = np.sqrt((1 - overall_ic**2) / (n - 2)) if n > 2 else np.nan
                overall_ir = np.nan
                if ic_std is not None and not np.isnan(ic_std) and abs(ic_std) > 1e-12:
                    overall_ir = overall_ic / ic_std
                t_stat = np.nan
                p_value = np.nan
                if n >= min_overall_samples and abs(overall_ic) < 1.0:
                    t_stat = overall_ic * np.sqrt((n - 2) / (1 - overall_ic**2))
                    dof = n - 2
                    if HAS_SCIPY:
                        from scipy.stats import t
                        p_value = 2 * (1 - t.cdf(abs(t_stat), dof))
                    else:
                        import math
                        t_abs = abs(t_stat)
                        p_value = 2 * (1 - 0.5 * (1 + math.erf(t_abs / math.sqrt(2))))

                ci_width = None
                if ic_std is not None and not np.isnan(ic_std):
                    ci_width = 2 * ic_std

                metrics = {
                    'overall_ic': overall_ic,
                    'overall_ic_std': ic_std,
                    'overall_ir': overall_ir,
                    'overall_t_stat': t_stat,
                    'overall_p_value': p_value,
                    'overall_sample_size': n,
                    'overall_factor_unique': factor_variability,
                    'overall_return_unique': return_variability,
                    'overall_min_required': min_overall_samples,
                    'overall_mode': mode_label,
                    'overall_ci_width': ci_width,
                }

                if include_diagnostics:
                    if use_kendall or use_robust_corr:
                        metrics['overall_kendall_tau'] = kendall_tau_corr(factor_data, return_data)
                    if use_robust_corr:
                        metrics['overall_robust_corr'] = robust_correlation(factor_data, return_data)
                    if use_nonparam_test:
                        metrics['overall_mann_whitney_u'] = mann_whitney_u_test(factor_data, return_data)
                    if compute_bootstrap_ci:
                        bootstrap_results = bootstrap_confidence_interval(
                            factor_data, return_data, n_bootstrap=n_bootstrap)
                        metrics['overall_bootstrap_ci'] = bootstrap_results
                return metrics
            except Exception as exc:
                if silent:
                    print(f"  [参考] 整体IC计算失败: {exc}")
                else:
                    print(f"  计算整体IC值时出错: {exc}")
            return None
        if not hasattr(self, 'anomaly_stats'):
            self.anomaly_stats = {}
        if 'ic_calculation' not in self.anomaly_stats:
            self.anomaly_stats['ic_calculation'] = {}
        self.anomaly_stats['ic_calculation'][factor_col] = {
            'total_dates': total_dates,
            'processed_dates': 0,
            'skipped_dates': 0,
            'skipped_reasons': [],
            'avg_daily_samples': avg_daily_samples,
            'median_daily_samples': median_daily_samples,
            'p25_daily_samples': p25,
            'screening_mode': mode,
            'min_samples_per_day': min_samples_per_day,
            'ic_window_days': ic_window_days,
            'segment_warnings': [],
        }

        pending_frames = []
        pending_dates = []

        def _record_skip(reason):
            nonlocal skipped_dates
            for dt in pending_dates or []:
                msg = f"日期 {dt}: {reason}"
                self.anomaly_stats['ic_calculation'][factor_col]['skipped_reasons'].append(msg)
            skipped_dates += len(pending_dates) if pending_dates else 1

        def _flush_pending(reason):
            nonlocal pending_frames, pending_dates
            if pending_dates:
                _record_skip(reason)
            pending_frames = []
            pending_dates = []

        def _compute_ic_from_dataframe(valid_data, date_label):
            factor_values = valid_data[factor_col]
            return_values = valid_data[self.return_col]
            factor_std = factor_values.std()
            return_std = return_values.std()
            if factor_std <= 0 or return_std <= 0:
                return None, f"{date_label}: 标准差为零 (factor_std={factor_std:.6f}, return_std={return_std:.6f})"

            if avg_daily_samples >= 8:
                min_factor_variability = 5
                min_return_variability = 5
            elif avg_daily_samples >= 4:
                min_factor_variability = 3
                min_return_variability = 3
            else:
                min_factor_variability = 2
                min_return_variability = 2

            factor_variability = factor_values.nunique()
            return_variability = return_values.nunique()
            if factor_variability < min_factor_variability:
                return None, f"{date_label}: 因子值变异性不足 (唯一值: {factor_variability}, 要求: {min_factor_variability})"
            if return_variability < min_return_variability:
                return None, f"{date_label}: 收益率变异性不足 (唯一值: {return_variability}, 要求: {min_return_variability})"

            try:
                if use_pearson:
                    factor_values_clean = np.asarray(factor_values, dtype=float)
                    return_values_clean = np.asarray(return_values, dtype=float)
                    mask = ~(np.isnan(factor_values_clean) | np.isnan(return_values_clean))
                    factor_values_clean = factor_values_clean[mask]
                    return_values_clean = return_values_clean[mask]
                    if len(factor_values_clean) >= 2:
                        ic_value = np.corrcoef(factor_values_clean, return_values_clean)[0, 1]
                    else:
                        return None, f"{date_label}: 有效样本<2，无法计算Pearson"
                else:
                    ic_value = custom_spearman_corr(factor_values, return_values)
            except Exception as exc:
                return None, f"{date_label}: 计算 IC 出错 - {exc}"

            if ic_value is None or np.isnan(ic_value) or not np.isfinite(ic_value):
                return None, f"{date_label}: 计算结果为NaN或无穷大"
            # winsor 以避免极值
            ic_value = float(np.clip(ic_value, -0.99, 0.99))
            return ic_value, None

        for date, group in df.groupby('信号日期'):
            valid_data = group.dropna(subset=[factor_col, self.return_col])
            if valid_data.empty:
                self.anomaly_stats['ic_calculation'][factor_col]['skipped_reasons'].append(
                    f"日期 {date}: 无有效样本")
                skipped_dates += 1
                continue

            pending_frames.append(valid_data)
            pending_dates.append(date)
            combined = pd.concat(pending_frames, axis=0)

            if len(combined) < min_samples_per_day:
                if len(pending_frames) >= ic_window_days:
                    reason = f"聚合{len(pending_frames)}天后样本仍不足 (需要≥{min_samples_per_day}，实际:{len(combined)})"
                    _flush_pending(reason)
                continue

            ic_value, failure_reason = _compute_ic_from_dataframe(combined, f"日期 {pending_dates[0]}~{pending_dates[-1]}")
            if ic_value is None:
                _flush_pending(failure_reason or "计算失败")
                continue

            daily_ics = ensure_list(daily_ics, "daily_ics")
            daily_ics.append(ic_value)
            self.anomaly_stats['ic_calculation'][factor_col]['processed_dates'] += len(pending_dates)
            print(f"  IC窗口 {pending_dates[0]}~{pending_dates[-1]}: {ic_value:.4f} (样本 {len(combined)})")
            pending_frames = []
            pending_dates = []

        if pending_dates:
            _flush_pending("窗口结束但样本仍不足")

        daily_ics = ensure_list(daily_ics, "daily_ics")
        daily_numeric = []
        for ic in daily_ics:
            if isinstance(ic, (int, float, np.number)) and np.isfinite(ic):
                daily_numeric.append(float(ic))
        daily_ics = daily_numeric
        effective_dates = len(daily_ics)
        skip_ratio = (skipped_dates / total_dates) if total_dates else 0.0
        extra_stats['daily_points'] = effective_dates
        extra_stats['effective_dates'] = effective_dates
        extra_stats['skipped_dates'] = skipped_dates
        extra_stats['skip_ratio'] = skip_ratio
        extra_stats['daily_missing_ratio'] = skip_ratio
        extra_stats['qualified_day_ratio'] = (effective_dates / total_dates) if total_dates else 0.0
        self.anomaly_stats['ic_calculation'][factor_col]['skipped_dates'] = skipped_dates
        self.anomaly_stats['ic_calculation'][factor_col]['effective_dates'] = effective_dates

        print(
            f"  [IC] 日度IC窗口结果: 有效 {effective_dates}/{total_dates}，"
            f"跳过 {skipped_dates} ({skip_ratio:.1%})"
        )

        if daily_ics:
            ic_mean = np.mean(daily_ics)
            ic_std = np.std(daily_ics, ddof=1) if len(daily_ics) > 1 else 0.0
            reference_metrics = _compute_overall_reference(silent=True, include_diagnostics=False)
            if reference_metrics:
                extra_stats['overall_metrics'] = reference_metrics
                for key, value in reference_metrics.items():
                    if key.startswith('overall_'):
                        extra_stats[key] = value
        segment_metrics = {}
        segment_warnings = []
        for segment, info in extra_stats['segment_counts'].items():
            seg_count = info.get('count', 0)
            if seg_count < SEGMENT_MIN_SAMPLES:
                segment_warnings.append(
                    f"{segment}: 样本不足（{seg_count}<{SEGMENT_MIN_SAMPLES}）"
                )
                continue
            seg_df = factor_valid_df[factor_valid_df['market_segment'] == segment]
            seg_daily_counts = seg_df.groupby('信号日期').size()
            seg_avg_daily = float(seg_daily_counts.mean()) if not seg_daily_counts.empty else 0.0
            if seg_avg_daily < SEGMENT_MIN_DAILY:
                segment_warnings.append(
                    f"{segment}: 日均样本{seg_avg_daily:.1f}<阈值{SEGMENT_MIN_DAILY}"
                )
                continue
            seg_metrics = _compute_overall_reference(silent=True, include_diagnostics=False, source_df=seg_df)
            if seg_metrics:
                seg_metrics['segment_sample_size'] = int(seg_count)
                seg_metrics['segment_ratio'] = info.get('ratio', 0.0)
                seg_metrics['segment_avg_daily_samples'] = seg_avg_daily
                segment_metrics[segment] = seg_metrics
            else:
                segment_warnings.append(f"{segment}: 无法计算整体IC")
        extra_stats['segment_metrics'] = segment_metrics
        extra_stats['segment_valid_count'] = len(segment_metrics)
        extra_stats['segment_warnings'] = segment_warnings
        self.anomaly_stats['ic_calculation'][factor_col]['segment_warnings'] = segment_warnings
        sorted_segments = sorted(
            extra_stats['segment_counts'].items(),
            key=lambda item: item[1].get('count', 0),
            reverse=True,
        )
        if sorted_segments:
            primary_seg = sorted_segments[0][0]
            extra_stats['segment_primary'] = primary_seg
            extra_stats['segment_primary_ratio'] = sorted_segments[0][1].get('ratio', 0.0)
            extra_stats['segment_primary_ic'] = segment_metrics.get(primary_seg, {}).get('overall_ic')
        if len(sorted_segments) > 1:
            secondary_seg = sorted_segments[1][0]
            extra_stats['segment_secondary'] = secondary_seg
            extra_stats['segment_secondary_ratio'] = sorted_segments[1][1].get('ratio', 0.0)
            extra_stats['segment_secondary_ic'] = segment_metrics.get(secondary_seg, {}).get('overall_ic')

        if len(daily_ics) >= 5 and ic_std > 0:
            t_stat = ic_mean / (ic_std / np.sqrt(len(daily_ics)))
            dof = len(daily_ics) - 1
            if HAS_SCIPY:
                from scipy.stats import t
                p_value = 2 * (1 - t.cdf(abs(t_stat), dof))
            else:
                import math
                p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2))))
            if use_kendall or use_robust_corr:
                from scipy.stats import kendalltau
                kt, _ = kendalltau(range(len(daily_ics)), daily_ics)
                extra_stats['kendall_tau'] = kt
            if use_robust_corr:
                extra_stats['robust_corr'] = robust_correlation(range(len(daily_ics)), daily_ics)
            if use_nonparam_test:
                extra_stats['wilcoxon_test'] = mann_whitney_u_test(daily_ics, [0.0] * len(daily_ics))
            if compute_bootstrap_ci:
                extra_stats['bootstrap_ci'] = bootstrap_confidence_interval(
                    daily_ics,
                    None,
                    statistic='mean',
                    n_bootstrap=n_bootstrap,
                )
            return (ic_mean, ic_std, t_stat, p_value, extra_stats)
        elif daily_ics:
            print("  警告: 有效IC值数量不足或标准差为0，无法计算t统计量")
            return (ic_mean, ic_std, np.nan, np.nan, extra_stats)

        extra_stats['fallback_reason'] = (
            f"有效日度IC点数不足（{effective_dates}/{total_dates}）"
        )
        print("  警告: 没有成功计算任何日度IC，尝试整体计算")
        extra_stats['ic_mode'] = "overall"
        overall_metrics = _compute_overall_reference(silent=False, include_diagnostics=True)
        if overall_metrics:
            extra_stats['overall_metrics'] = overall_metrics
            for key, value in overall_metrics.items():
                if key.startswith('overall_'):
                    extra_stats[key] = value
            return (
                overall_metrics.get('overall_ic', np.nan),
                overall_metrics.get('overall_ic_std', np.nan),
                overall_metrics.get('overall_t_stat', np.nan),
                overall_metrics.get('overall_p_value', np.nan),
                extra_stats,
            )
        print("  无法计算IC值")
        return (np.nan, np.nan, np.nan, np.nan, extra_stats)
    
    def calculate_group_returns(self, factor_col, n_groups=None):
        """
        计算分组收益 - 使用简单的等分分组方式：直接对完整因子数据排序后平均分配，不按日期分组处理
        
        Args:
            factor_col: 因子列名
            n_groups: 分组数量（默认读取配置中的 group_count）
            
        Returns:
            dict: 包含分组收益和多空收益的字典
        """
        # 使用预处理后的数据，确保因子处理生效
        df = self.processed_data if hasattr(self, 'processed_data') and self.processed_data is not None else self.data.copy()
        if n_groups is None:
            n_groups = DEFAULT_GROUP_COUNT
        
        # 确保数据有效
        if df.empty or factor_col not in df.columns or self.return_col not in df.columns:
            print(f"警告: 数据为空或列名不存在")
            return None
        
        print(f"- - - - - - - - - - - - - - - - - - - - - - - - - - - - - : {len(df)}")
        df = df.copy()
        df[factor_col] = pd.to_numeric(df[factor_col], errors="coerce")
        df[self.return_col] = pd.to_numeric(df[self.return_col], errors="coerce")
        
        # - - - - - - - - - - - - - - - - - - - - - - - 
        total_samples = len(df)
        df_clean = df.dropna(subset=[factor_col, self.return_col]).copy()
        valid_samples = len(df_clean)
        removed_samples = total_samples - valid_samples
        
        if len(df_clean) == 0:
            print(f"警告: 去除因子值为空的行后没有剩余数据")
            return None
        
        # 使用简单的等分分组方式
        # 1. 将因子数据从小到大排列
        # 2. 平均分成n_groups份
        df_clean = df_clean.sort_values(by=factor_col)
        
        # 计算每个样本应该属于哪个分组
        total_samples = len(df_clean)
        group_size = total_samples // n_groups
        remainder = total_samples % n_groups
        
        # 创建分组标签
        groups = []
        for i in range(n_groups):
            # 前remainder个分组每个多分配1个样本
            group_count = group_size + (1 if i < remainder else 0)
            groups.extend([i+1] * group_count)
        
        # 分配分组标签
        df_clean['分组'] = groups[:total_samples]  # 确保长度匹配
        
        # 验证分组样本数量分布
        group_counts = df_clean['分组'].value_counts().sort_index()
        if len(group_counts) > 0:
            min_count = group_counts.min()
            max_count = group_counts.max()
            # 如果样本数量差异超过10%，发出警告
            if max_count > min_count * 1.1:
                print(f"警告: 分组样本数量分布不均，最小: {min_count}, 最大: {max_count}")
        
        # 计算每组的平均收益、标准差和样本数量
        # 同时计算每组因子值的最小值和最大值
        group_stats = df_clean.groupby('分组').agg({
            self.return_col: ['mean', 'std', 'count'],
            factor_col: ['min', 'max']
        }).reset_index()
        
        # 重命名列
        group_stats.columns = ['分组', '平均收益', '收益标准差', '样本数量', '因子最小值', '因子最大值']

        # 确保关键统计列为数值型，避免字符串在后续计算中参与运算
        numeric_columns = ['平均收益', '收益标准差', '样本数量', '因子最小值', '因子最大值']
        for col in numeric_columns:
            if col in group_stats.columns:
                group_stats[col] = pd.to_numeric(group_stats[col], errors='coerce')
        
        # 创建参数区间列
        # 使用更统一的格式化方法，避免显示不一致
        # 注释掉对百分比因子的特殊处理，统一使用原始数据
        # percentage_columns = [
        #     '日最大跌幅百分比', '信号当日收盘涨跌幅', '信号后一日开盘涨跌幅', 
        #     '次日开盘后总体下跌幅度', '前10日最大涨幅', '当日回调', '持股2日收益率'
        # ]
        
        # 注释掉对百分比因子的特殊处理，统一使用原始数据
        # if factor_col in percentage_columns:
        #     # 对于百分比因子，将值乘以100恢复为原始百分比形式
        #     group_stats['参数区间'] = group_stats.apply(lambda x: f"{x['因子最小值']*100:.4f}-{x['因子最大值']*100:.4f}", axis=1)
        # else:
        #     # 对于非百分比因子，保持原样
        #     group_stats['参数区间'] = group_stats.apply(lambda x: f"{x['因子最小值']:.4f}-{x['因子最大值']:.4f}", axis=1)
            
        # 使用原始数据，不进行任何转换，修改为使用"到"分隔符避免Excel解析错误
        group_stats['参数区间'] = group_stats.apply(lambda x: f"{x['因子最小值']:.4f}到{x['因子最大值']:.4f}", axis=1)
        
        # 检测并警告异常大的区间跨度
        for idx, row in group_stats.iterrows():
            span = row['因子最大值'] - row['因子最小值']
            # 计算所有组的平均跨度
            avg_span = group_stats['因子最大值'].mean() - group_stats['因子最小值'].mean()
            # 如果某组跨度超过平均跨度的3倍，发出警告
            if span > 3 * avg_span and span > 0:
                print(f"警告: 第{int(row['分组'])}组的参数区间跨度异常大: {span:.4f}")
        
        # 保留需要的列
        avg_returns = group_stats[['分组', '平均收益', '收益标准差', '样本数量', '参数区间']]
        
        # 简化的分组完成提示
        total_samples = avg_returns['样本数量'].sum()
        print(f"分组完成，共 {n_groups} 组，总样本数: {total_samples}")
        
        # 确保分组从1到n_groups连续
        expected_groups = list(range(1, n_groups + 1))
        for g in expected_groups:
            if g not in avg_returns['分组'].values:
                print(f"警告: 未找到分组 {g}，创建默认记录")
                new_row = pd.DataFrame({
                    '分组': [g],
                    '平均收益': [0.0],
                    '收益标准差': [0.0],
                    '样本数量': [0]
                })
                avg_returns = pd.concat([avg_returns, new_row], ignore_index=True)
        
        # 按分组排序
        avg_returns = avg_returns.sort_values('分组').reset_index(drop=True)
        
        # 计算t统计量和p值
        t_stats = []
        p_values = []
        
        for _, row in avg_returns.iterrows():
            group_num = row['分组']
            # 从原始数据中获取该分组的数据
            group_data = df_clean[df_clean['分组'] == group_num][self.return_col]
            
            if len(group_data) > 1:
                # 计算样本标准差
                std = group_data.std(ddof=1)
                if std > 0:
                    # 计算t统计量
                    t_stat = row['平均收益'] / (std / np.sqrt(len(group_data)))
                    
                    # 计算p值
                    if HAS_SCIPY:
                        from scipy.stats import t
                        p_value = 2 * (1 - t.cdf(abs(t_stat), len(group_data) - 1))
                        # 为非常小的p值设置最小值，避免显示为0
                        p_value = max(p_value, 1e-10)
                    else:
                        # 不使用scipy时，使用数学公式计算
                        import math
                        dof = len(group_data) - 1
                        
                        # 对于大自由度，可以使用正态近似
                        if dof > 30:
                            z_score = t_stat
                            p_value = 2 * (1 - 0.5 * (1 + math.erf(z_score / math.sqrt(2))))
                        else:
                            # 对于小自由度，使用更简单的近似
                            p_value = 2 * math.exp(-t_stat**2 / 2)
                        # 为非常小的p值设置最小值
                        p_value = max(p_value, 1e-10)
                else:
                    t_stat = np.nan
                    p_value = np.nan
            else:
                t_stat = np.nan
                p_value = np.nan
            
            t_stats.append(t_stat)
            p_values.append(p_value)
        
        avg_returns['T统计量'] = t_stats
        avg_returns['P值'] = p_values
        
        # 计算每个分组的胜率、最大回撤、夏普率和索提诺比率
        win_rates = []
        max_drawdowns = []
        sharpe_ratios = []
        sortino_ratios = []
        
        for _, row in avg_returns.iterrows():
            group_num = row['分组']
            # 从原始数据中获取该分组的数据
            group_data = df_clean[df_clean['分组'] == group_num][self.return_col]
            
            if len(group_data) > 0:
                # 计算胜率：收益为正的样本占比
                positive_count = (group_data > 0).sum()
                win_rate = positive_count / len(group_data) if len(group_data) > 0 else 0
                win_rates.append(win_rate)
                
                # 计算最大回撤
                if len(group_data) > 1:
                    cumulative_returns = (1 + group_data).cumprod()
                    running_max = cumulative_returns.expanding().max()
                    drawdown = (cumulative_returns - running_max) / running_max
                    max_drawdown = abs(drawdown.min())
                    max_drawdowns.append(max_drawdown)
                else:
                    max_drawdowns.append(0.0)
                
                # 计算单期夏普率（用于年化转换）
                if row['收益标准差'] > 0:
                    single_period_sharpe = row['平均收益'] / row['收益标准差']
                    sharpe_ratios.append(single_period_sharpe)
                else:
                    sharpe_ratios.append(0.0)
                
                # 计算单期索提诺比率（用于年化转换）
                downside_returns = group_data[group_data < 0]
                if len(downside_returns) > 0 and len(downside_returns) > 1:
                    downside_std = downside_returns.std(ddof=1)
                    if downside_std > 0:
                        single_period_sortino = row['平均收益'] / downside_std
                        sortino_ratios.append(single_period_sortino)
                    else:
                        sortino_ratios.append(0.0)
                else:
                    sortino_ratios.append(0.0)
            else:
                win_rates.append(0.0)
                max_drawdowns.append(0.0)
                sharpe_ratios.append(0.0)
                sortino_ratios.append(0.0)
        
        # 添加新的列到结果中
        avg_returns['胜率'] = win_rates
        avg_returns['最大回撤'] = max_drawdowns
        avg_returns['夏普率'] = sharpe_ratios  # 添加单期夏普比率
        avg_returns['索提诺比率'] = sortino_ratios  # 添加单期索提诺比率
        
        # [目标] 自适应年化收益率计算系统
        # 自动分析原始数据特征，智能选择年化算法
        print(f"  [分析] 开始自适应年化计算分析...")
        
        # 步骤1: 自动分析原始数据特征
        data_characteristics = self._analyze_data_characteristics()
        
        # 步骤2: 基于数据特征选择最优年化算法
        annualization_method = self._select_optimal_annualization_method(data_characteristics)
        
        # 步骤3: 执行年化计算
        annual_results = self._calculate_adaptive_annual_returns(avg_returns, data_characteristics, annualization_method)
        
        # 应用计算结果 - 优化版本
        # 主要年化收益率：使用标准复利年化结果
        avg_returns['年化收益率'] = annual_results['main_annual_return']
        
        # 对比方法：CAGR复合年化收益率
        avg_returns['CAGR年化收益率'] = annual_results['cagr_annual_return']
        
        # 风险指标
        avg_returns['年化收益标准差'] = annual_results['annual_std']
        avg_returns['年化夏普比率'] = annual_results['annual_sharpe']
        avg_returns['年化索提诺比率'] = annual_results['annual_sortino']
        
        # 新增：数据质量信息列
        if 'quality_stats' in annual_results:
            quality = annual_results['quality_stats']
            avg_returns['年化计算成功率'] = quality.get('validation_success_rate', 0)
            avg_returns['有效分组数'] = quality.get('valid_groups', 0)
        
        # 打印详细分析结果
        self._print_annualization_analysis(data_characteristics, annualization_method, annual_results)
        
        # 计算多空收益（高分组 - 低分组）
        long_short_return = np.nan
        if len(avg_returns) >= 2:
            max_group = avg_returns['分组'].max()
            min_group = avg_returns['分组'].min()
            
            max_return = avg_returns.loc[avg_returns['分组'] == max_group, '平均收益'].values[0]
            min_return = avg_returns.loc[avg_returns['分组'] == min_group, '平均收益'].values[0]
            
            # 计算高因子组与低因子组的收益差
            long_short_return = max_return - min_return
            print(f"  多空收益（高-低分组）: {long_short_return:.4f}")
        
        segment_summary = {}
        if 'market_segment' in df_clean.columns:
            for seg, seg_df in df_clean.groupby('market_segment'):
                if seg_df.empty:
                    continue
                seg_returns = seg_df[self.return_col].dropna()
                avg_ret = float(seg_returns.mean()) if len(seg_returns) > 0 else np.nan
                win_rate_seg = float((seg_returns > 0).mean()) if len(seg_returns) > 0 else np.nan
                segment_summary[seg] = {
                    'avg_return': avg_ret,
                    'win_rate': win_rate_seg,
                    'count': int(len(seg_df)),
                }
        segment_recommendation = None
        if segment_summary:
            total_segment_samples = sum(info.get('count', 0) for info in segment_summary.values())
            sorted_by_return = sorted(
                segment_summary.items(),
                key=lambda item: (
                    item[1].get('avg_return')
                    if item[1].get('avg_return') is not None and not np.isnan(item[1].get('avg_return'))
                    else -np.inf
                ),
                reverse=True,
            )
            recommendation_parts = []
            best_seg, best_info = sorted_by_return[0]
            best_return = best_info.get('avg_return')
            if len(segment_summary) == 1:
                recommendation_parts.append(
                    f"仅覆盖{best_seg}（{best_info.get('count', 0)}条）样本，建议在该板块内评估"
                )
            elif best_return is not None and not np.isnan(best_return):
                other_positive = [
                    info.get('avg_return')
                    for seg, info in sorted_by_return[1:]
                    if info.get('avg_return') is not None and not np.isnan(info.get('avg_return')) and info.get('avg_return') > 0
                ]
                if best_return > 0 and not other_positive:
                    recommendation_parts.append(
                        f"仅在{best_seg}呈正收益，其它板块≤0，建议限定于{best_seg}"
                    )
                else:
                    runner_return = sorted_by_return[1][1].get('avg_return') if len(sorted_by_return) > 1 else None
                    if runner_return is not None and not np.isnan(runner_return):
                        delta = best_return - runner_return
                        threshold = max(0.01, abs(runner_return) * 0.5)
                        if delta >= threshold:
                            recommendation_parts.append(
                                f"{best_seg}收益明显领先（Δ={delta:.3f}），优先配置该板块"
                            )
            if total_segment_samples > 0:
                primary_share = best_info.get('count', 0) / total_segment_samples
                if primary_share >= 0.7:
                    recommendation_parts.append(
                        f"{best_seg}样本占比{primary_share * 100:.1f}%，注意板块集中风险"
                    )
            if recommendation_parts:
                segment_recommendation = "；".join(recommendation_parts)

        return {
            'avg_returns': avg_returns,
            'long_short_return': long_short_return,
            'segment_summary': segment_summary,
            'segment_recommendation': segment_recommendation,
        }
    
    def _analyze_data_characteristics(self):
        """
        自动分析原始数据特征
        
        Returns:
            dict: 数据特征分析结果
        """
        df_source = getattr(self, 'processed_data', None)
        if df_source is None or df_source.empty:
            df_source = self.data if self.data is not None else pd.DataFrame()
        df = df_source if df_source is not None else pd.DataFrame()
        
        default_characteristics = {
            'total_trades': len(df),
            'avg_trade_interval': 2.0,
            'actual_annual_trades': 164.0,
            'observation_period_years': 5.18,
            'holding_period_days': 2,
            'trade_frequency_category': '高频'
        }

        # 计算交易频率和持股周期
        if df is not None and not df.empty and '信号日期' in df.columns and self.return_col in df.columns:
            signal_dates = pd.to_datetime(df['信号日期'], errors='coerce')
            valid_dates = signal_dates.dropna().sort_values()

            if len(valid_dates) > 0:
                date_diff = valid_dates.diff().dt.days
                date_diff_clean = date_diff.dropna()
                avg_interval = date_diff_clean.mean() if not date_diff_clean.empty else np.nan

                if pd.notna(avg_interval) and avg_interval > 0:
                    actual_trades_per_year = 365 / avg_interval
                else:
                    actual_trades_per_year = 365  # 默认值

                if len(valid_dates) >= 2:
                    total_days = (valid_dates.iloc[-1] - valid_dates.iloc[0]).days
                    observation_period = total_days / 365.25 if total_days > 0 else 1
                else:
                    observation_period = 1  # 默认观测期

                holding_period = 2  # 从数据特征知道是2日持有
                trade_interval_value = float(avg_interval) if pd.notna(avg_interval) else np.nan

                return {
                    'total_trades': len(df),
                    'avg_trade_interval': trade_interval_value,
                    'actual_annual_trades': actual_trades_per_year,
                    'observation_period_years': observation_period,
                    'holding_period_days': holding_period,
                    'trade_frequency_category': '高频' if actual_trades_per_year > 100 else ('中频' if actual_trades_per_year > 20 else '低频')
                }

        return default_characteristics
    
    def _select_optimal_annualization_method(self, characteristics):
        """
        基于数据特征选择最优年化算法 - 优化版本
        
        核心改进：
        1. 优先选择标准复利年化方法（数学最严谨）
        2. 基于数据质量特征智能选择算法
        3. 保留CAGR方法作为对比验证
        4. 添加数据质量评估机制
        
        Args:
            characteristics: 数据特征分析结果
            
        Returns:
            dict: 选择的年化方法参数，包含新的标准复利方法标识
        """
        try:
            actual_trades = characteristics['actual_annual_trades']
            holding_period = characteristics['holding_period_days']
            frequency_category = characteristics['trade_frequency_category']
            observation_years = characteristics['observation_period_years']
            total_trades = characteristics['total_trades']
            
            print(f"    [分析] 智能选择最优年化算法...")
            
            # 数据质量评估指标
            data_quality_score = 0
            frequency_stability_score = 0
            
            # 1. 数据完整性评估
            if observation_years >= 2.0 and total_trades >= 50:
                data_quality_score += 0.4  # 观测期足够长且交易样本充足
            if actual_trades >= 20 and actual_trades <= 300:
                data_quality_score += 0.3  # 年交易频率在合理范围内
            if holding_period > 0 and holding_period <= 30:
                data_quality_score += 0.3  # 持股周期合理
            
            # 2. 频率稳定性评估
            if frequency_category in ['高频', '中频']:
                frequency_stability_score += 0.5  # 高频和中频交易相对稳定
            if actual_trades > 10:  # 有足够的交易频率数据
                frequency_stability_score += 0.5
            
            # 3. 智能算法选择
            # 主要标准：优先使用标准复利年化方法
            if data_quality_score >= 0.7:
                # 高质量数据：优先使用标准复利年化方法
                selected_method = {
                    'primary_method': 'standard_compound',
                    'comparison_method': 'cagr_based',
                    'reason': f'高质量数据（得分{data_quality_score:.2f}），使用标准复利年化方法（数学最严谨）',
                    'frequency_base': actual_trades,
                    'data_quality_score': data_quality_score,
                    'frequency_stability_score': frequency_stability_score,
                    'optimization_reason': '数据质量优秀，优先选择数学上最严谨的标准复利年化方法'
                }
                print(f"      [OK] 选择：标准复利年化（数据质量得分: {data_quality_score:.2f}）")
                
            elif frequency_stability_score >= 0.7:
                # 中等质量数据：仍优先使用标准复利年化，但加强验证
                selected_method = {
                    'primary_method': 'standard_compound',
                    'comparison_method': 'cagr_based',
                    'reason': f'频率稳定性良好（得分{frequency_stability_score:.2f}），使用标准复利年化方法',
                    'frequency_base': actual_trades,
                    'data_quality_score': data_quality_score,
                    'frequency_stability_score': frequency_stability_score,
                    'optimization_reason': '频率稳定性良好，使用标准复利年化方法，加强结果验证'
                }
                print(f"      [OK] 选择：标准复利年化（频率稳定性得分: {frequency_stability_score:.2f}）")
                
            else:
                # 较低质量数据：使用标准复利年化但增加保守性检查
                selected_method = {
                    'primary_method': 'standard_compound',
                    'comparison_method': 'cagr_based',
                    'reason': f'数据质量一般，仍使用标准复利年化方法但加强验证',
                    'frequency_base': actual_trades,
                    'data_quality_score': data_quality_score,
                    'frequency_stability_score': frequency_stability_score,
                    'optimization_reason': '数据质量一般，使用标准复利年化方法，但需要加强验证和保守性处理'
                }
                print(f"      [警告]  选择：标准复利年化（加强验证模式）")
            
            # 添加技术参数
            selected_method.update({
                'observation_period_years': observation_years,
                'holding_period_days': holding_period,
                'frequency_category': frequency_category,
                'is_optimized': True,  # 标记为优化版本
                'deprecated_methods': ['linear_annual_return_a', 'linear_annual_return_b']  # 标记已移除的方法
            })
            
            print(f"      [统计] 算法特征分析:")
            print(f"         数据质量得分: {data_quality_score:.2f}")
            print(f"         频率稳定性得分: {frequency_stability_score:.2f}")
            print(f"         观测期长度: {observation_years:.2f}年")
            print(f"         年交易频率: {actual_trades:.1f}次")
            
            return selected_method
            
        except Exception as e:
            print(f"      [ERROR] 算法选择出错: {str(e)}")
            # 返回安全的默认选择
            return {
                'primary_method': 'standard_compound',
                'comparison_method': 'cagr_based',
                'reason': f'算法选择失败，使用默认标准复利年化方法',
                'frequency_base': 252,
                'data_quality_score': 0,
                'frequency_stability_score': 0,
                'is_optimized': False,
                'error': str(e)
            }
    
    def _print_annualization_analysis(self, characteristics, method_info, results):
        """
        打印详细年化分析结果 - 优化版本
        
        Args:
            characteristics: 数据特征
            method_info: 选择的年化方法
            results: 年化计算结果（优化后的结构）
        """
        print(f"  [统计] 数据特征分析:")
        print(f"     总交易次数: {characteristics['total_trades']}次")
        print(f"     年化交易频率: {characteristics['actual_annual_trades']:.1f}次/年")
        print(f"     平均交易间隔: {characteristics['avg_trade_interval']:.1f}天")
        print(f"     观测期长度: {characteristics['observation_period_years']:.2f}年")
        print(f"     持股周期: {characteristics['holding_period_days']}天")
        print(f"     交易频率分类: {characteristics['trade_frequency_category']}")
        
        print(f"\n  [目标] 选择年化算法:")
        print(f"     算法类型: {method_info['primary_method']}")
        print(f"     选择理由: {method_info['reason']}")
        print(f"     频率基础: {method_info['frequency_base']:.1f}次/年")
        
        print(f"\n  [结果] 年化计算结果（优化后）:")
        
        # 主要方法：标准复利年化
        if 'standard_compound_annual_return' in results:
            valid_standard = results['standard_compound_annual_return'][np.isfinite(results['standard_compound_annual_return'])]
            if len(valid_standard) > 0:
                print(f"     [强] 标准复利年化收益率: {float(valid_standard.mean()):.6f} ({float(valid_standard.mean())*100:.4f}%)")
                print(f"        (数学公式: (1+总收益率)^(1/年数) - 1)")
            else:
                print(f"     [强] 标准复利年化收益率: 无有效数据")
        
        # 对比方法：CAGR年化
        if 'cagr_annual_return' in results:
            valid_cagr = results['cagr_annual_return'][np.isfinite(results['cagr_annual_return'])]
            if len(valid_cagr) > 0:
                print(f"     [上升] CAGR复合年化收益率: {float(valid_cagr.mean()):.6f} ({float(valid_cagr.mean())*100:.4f}%)")
                print(f"        (对比方法，与标准复利年化数学等价)")
            else:
                print(f"     [上升] CAGR复合年化收益率: 无有效数据")
        
        # 数据质量评估
        if 'quality_stats' in results:
            quality = results['quality_stats']
            print(f"\n  [列表] 数据质量评估:")
            print(f"     有效分组数: {quality.get('valid_groups', 0)}/{quality.get('total_groups', 0)}")
            print(f"     验证成功率: {quality.get('validation_success_rate', 0):.1%}")
            
            if not np.isnan(quality.get('mean_annual_return', np.nan)):
                print(f"     年化收益率范围: {quality.get('min_annual_return', 0)*100:.2f}% 到 {quality.get('max_annual_return', 0)*100:.2f}%")
        
        # 风险指标
        valid_std = results['annual_std'][np.isfinite(results['annual_std'])]
        valid_sharpe = results['annual_sharpe'][np.isfinite(results['annual_sharpe'])]
        valid_sortino = results['annual_sortino'][np.isfinite(results['annual_sortino'])]
        
        if len(valid_std) > 0:
            print(f"     年化收益标准差: {float(valid_std.mean()):.6f} ({float(valid_std.mean())*100:.4f}%)")
        if len(valid_sharpe) > 0:
            print(f"     年化夏普比率: {float(valid_sharpe.mean()):.4f}")
        if len(valid_sortino) > 0:
            print(f"     年化索提诺比率: {float(valid_sortino.mean()):.4f}")
        
        # 计算方法说明
        print(f"\n  [工具] 计算方法说明:")
        print(f"     [INFO] 主要方法: 标准复利年化（数学最严谨）")
        print(f"     [INFO] 对比方法: CAGR复合年化（验证一致性）")
        print(f"     [INFO] 验证机制: 反向计算验证结果准确性")
        
        # 计算方法差异分析
        if 'standard_compound_annual_return' in results and 'cagr_annual_return' in results:
            standard_mean = np.nanmean(results['standard_compound_annual_return'])
            cagr_mean = np.nanmean(results['cagr_annual_return'])
            
            if not (np.isnan(standard_mean) or np.isnan(cagr_mean)) and cagr_mean != 0:
                method_diff = standard_mean / cagr_mean
                print(f"     [尺度] 方法一致性检验: {method_diff:.6f} (应接近1.0)")
                if abs(method_diff - 1.0) < 0.001:
                    print(f"        [OK] 两种方法结果一致，验证通过")
                else:
                    print(f"        [警告]  方法间存在差异，需要检查")
    
    def calculate_factor_stats(self, factor_col):
        """
        计算因子统计指标，包括异常数据统计
        
        Args:
            factor_col: 因子列名
            
        Returns:
            dict: 因子统计信息，包含异常数据统计指标
        """
        df = self.processed_data
        factor_data = df[factor_col].dropna()
        total_samples = len(df)
        valid_samples = len(factor_data)
        
        # 计算基本统计量（这些只需要pandas和numpy）
        stats_info = {
            '均值': factor_data.mean(),
            '标准差': factor_data.std(),
            '最小值': factor_data.min(),
            '最大值': factor_data.max(),
            '中位数': factor_data.median(),
            '偏度': factor_data.skew(),
            '峰度': factor_data.kurtosis(),
            '样本数': total_samples,
            '有效样本数': valid_samples,
            '缺失样本比例': (total_samples - valid_samples) / total_samples if total_samples > 0 else 0
        }
        
        # 计算异常数据统计指标
        # 1. IQR离群值检测（不删除，只统计）
        Q1 = factor_data.quantile(0.25)
        Q3 = factor_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound_iqr = Q1 - 1.5 * IQR
        upper_bound_iqr = Q3 + 1.5 * IQR
        extreme_lower_bound = Q1 - 3.0 * IQR
        extreme_upper_bound = Q3 + 3.0 * IQR
        
        # 温和离群值（1.5-3倍IQR）
        mild_outliers = factor_data[(factor_data < lower_bound_iqr) | (factor_data > upper_bound_iqr)]
        # 极端离群值（>3倍IQR）
        extreme_outliers = factor_data[(factor_data < extreme_lower_bound) | (factor_data > extreme_upper_bound)]
        
        stats_info['IQR'] = IQR
        stats_info['Q1'] = Q1
        stats_info['Q3'] = Q3
        stats_info['温和离群值数量'] = len(mild_outliers) - len(extreme_outliers)
        stats_info['温和离群值比例'] = (len(mild_outliers) - len(extreme_outliers)) / valid_samples if valid_samples > 0 else 0
        stats_info['极端离群值数量'] = len(extreme_outliers)
        stats_info['极端离群值比例'] = len(extreme_outliers) / valid_samples if valid_samples > 0 else 0
        stats_info['总离群值比例'] = len(mild_outliers) / valid_samples if valid_samples > 0 else 0
        
        # 2. 极值比例（分位数之外的数据）
        p1 = factor_data.quantile(0.01)
        p99 = factor_data.quantile(0.99)
        p05 = factor_data.quantile(0.05)
        p95 = factor_data.quantile(0.95)
        
        extreme_low_values = factor_data[factor_data < p1]
        extreme_high_values = factor_data[factor_data > p99]
        
        stats_info['1%分位数'] = p1
        stats_info['99%分位数'] = p99
        stats_info['5%分位数'] = p05
        stats_info['95%分位数'] = p95
        stats_info['1%以下极值比例'] = len(extreme_low_values) / valid_samples if valid_samples > 0 else 0
        stats_info['99%以上极值比例'] = len(extreme_high_values) / valid_samples if valid_samples > 0 else 0
        
        # 3. 数据分布异常检测
        # 偏度异常检测（绝对值大于2表示严重偏斜）
        skewness = factor_data.skew()
        stats_info['偏度异常'] = abs(skewness) > 2
        stats_info['偏度异常程度'] = '严重偏斜' if abs(skewness) > 2 else ('中等偏斜' if abs(skewness) > 1 else '近似对称')
        
        # 峰度异常检测（绝对值大于3表示分布异常陡峭或平坦）
        kurtosis = factor_data.kurtosis()
        stats_info['峰度异常'] = abs(kurtosis) > 3
        stats_info['峰度异常程度'] = '极端' if abs(kurtosis) > 3 else ('中等' if abs(kurtosis) > 1 else '正常')
        
        # 4. 唯一值比例（低唯一值比例可能表示数据离散度不足）
        unique_count = factor_data.nunique()
        unique_ratio = unique_count / valid_samples if valid_samples > 0 else 0
        stats_info['唯一值数量'] = unique_count
        stats_info['唯一值比例'] = unique_ratio
        stats_info['离散度不足'] = unique_ratio < 0.05
        
        # 5. 零值和极值检查
        zero_count = (factor_data == 0).sum()
        stats_info['零值数量'] = zero_count
        stats_info['零值比例'] = zero_count / valid_samples if valid_samples > 0 else 0
        
        # 检查是否存在异常大的数值变化
        if valid_samples > 1:
            max_min_ratio = abs(factor_data.max() / factor_data.min()) if factor_data.min() != 0 else float('inf')
            stats_info['最大值/最小值比率'] = max_min_ratio
            stats_info['数值范围异常'] = max_min_ratio > 10000 or factor_data.max() > 1e6 or factor_data.min() < -1e6
        
        # 使用scipy计算更详细的统计信息（仅当scipy可用时）
        if HAS_SCIPY:
            # 计算Jarque-Bera正态性检验
            jb_stat, jb_pvalue = stats.jarque_bera(factor_data)
            stats_info['Jarque-Bera统计量'] = jb_stat
            stats_info['Jarque-Bera p值'] = jb_pvalue
            stats_info['非正态分布'] = jb_pvalue < 0.05  # p值小于0.05拒绝正态分布假设
            
            # 计算Shapiro-Wilk正态性检验
            if len(factor_data) <= 5000:  # Shapiro-Wilk在大样本上计算较慢
                sw_stat, sw_pvalue = stats.shapiro(factor_data)
                stats_info['Shapiro-Wilk统计量'] = sw_stat
                stats_info['Shapiro-Wilk p值'] = sw_pvalue
                stats_info['Shapiro-Wilk非正态'] = sw_pvalue < 0.05
        
        # 6. 异常因子识别（为后续报告做准备）
        stats_info['可能异常因子'] = any([
            stats_info['总离群值比例'] > 0.2,  # 超过20%的离群值
            stats_info['极端离群值比例'] > 0.05,  # 超过5%的极端离群值
            stats_info['偏度异常'],
            stats_info['峰度异常'],
            stats_info['离散度不足'],
            '数值范围异常' in stats_info and stats_info['数值范围异常'],
            stats_info['零值比例'] > 0.5  # 超过50%的零值
        ])
        
        return stats_info
    
    # 移除了验证函数，优化计算逻辑确保结果正确
    
    def run_factor_analysis(self, use_pearson=None):
        """
        运行所有因子的分析

        Args:
            use_pearson: 是否使用Pearson相关系数计算IC值（默认从配置读取）
        """
        if not self.preprocess_data():
            return False
        
        # 添加总体样本统计和打印
        print(f"\n===== 因子分析总体样本统计 =====")
        print(f"原始数据总样本数: {len(self.data)}")
        print(f"预处理后数据总样本数: {len(self.processed_data)}")
        print(f"数据保留率: {len(self.processed_data)/len(self.data)*100:.2f}%")
        print(f"待分析因子数量: {len(self.factors)}")
        
        # 统计各因子的有效样本数
        factor_valid_samples = {}
        for factor in self.factors:
            if factor in self.processed_data.columns:
                valid_samples = self.processed_data[factor].notna().sum()
                factor_valid_samples[factor] = valid_samples
                print(f"因子 {factor}: {valid_samples} 个有效样本 ({valid_samples/len(self.processed_data)*100:.2f}%)")
            else:
                factor_valid_samples[factor] = 0
                print(f"因子 {factor}: 0 个有效样本 (因子不存在)")
        
        # 计算收益率列的有效样本数
        return_valid_samples = self.processed_data[self.return_col].notna().sum()
        print(f"收益率列 {self.return_col}: {return_valid_samples} 个有效样本 ({return_valid_samples/len(self.processed_data)*100:.2f}%)")
        print("==================================\n")
        
        if use_pearson is None:
            use_pearson = DEFAULT_USE_PEARSON

        # 确定相关系数类型
        corr_type = "Pearson" if use_pearson else "Spearman"
        print(f"\n开始因子分析，使用 {self.return_col} 作为收益率计算标准")
        print(f"使用 {corr_type} 相关系数计算IC值")
        
        for factor in self.factors:
            print(f"\n=== 分析因子: {factor} ===")
            
            # 检查因子是否存在且有效
            if factor not in self.processed_data.columns:
                print(f"跳过因子 {factor}: 数据中不存在该因子")
                continue
            
            # 计算因子基本统计
            stats_info = self.calculate_factor_stats(factor)
            print(f"分析因子: {factor}")
            
            # 计算IC值 - 启用所有新增的稳健性统计方法
            ic_mean, ic_std, t_stat, p_value, extra_stats = self.calculate_ic(
                factor, 
                use_pearson=use_pearson,
                use_robust_corr=True,    # 启用稳健相关系数
                use_kendall=True,        # 启用Kendall's Tau
                use_nonparam_test=True,  # 启用非参数检验
                compute_bootstrap_ci=True # 启用Bootstrap置信区间
            )
            
            # 显示额外的统计信息
            if extra_stats:
                print(f"  额外稳健性统计信息:")
                for key, value in extra_stats.items():
                    if isinstance(value, tuple):
                        print(f"    {key}: {value}")
                    elif isinstance(value, list) and len(value) > 0:
                        print(f"    {key}: {len(value)}个Bootstrap样本")
                    elif isinstance(value, (int, float, np.number)):
                        print(f"    {key}: {value:.3f}")
                    else:
                        print(f"    {key}: {value}")
            
            # 添加缺失值检查和警告
            if np.isnan(ic_std):
                print(f"  警告: {factor} 的IC标准差计算失败或缺失")
                ir = np.nan
            else:
                # 直接按数学定义计算IR值，不添加任何人为限制
                ir = ic_mean / ic_std if ic_std != 0 else np.nan
                if np.isnan(ir) or not np.isfinite(ir):
                    print(f"  警告: {factor} 的IR值计算异常（可能是IC标准差为0）")
            
            print(f"IC分析结果: IC均值={ic_mean:.3f}, IR值={ir:.3f}")
            
            # 计算分组收益
            group_results = self.calculate_group_returns(factor)
            if group_results:
                print(f"\n分组收益分析:")
                preview = group_results['avg_returns'].head(5)
                print(preview.to_string(index=False, float_format='%.3f'))
                if len(group_results['avg_returns']) > len(preview):
                    print("  … 其余分组已省略，详见报告文件")
                print(f"\n多空收益(最高组-最低组): {group_results['long_short_return']:.3f}")
            if isinstance(extra_stats, dict):
                if group_results:
                    extra_stats['segment_summary'] = group_results.get('segment_summary')
                    extra_stats['segment_recommendation'] = group_results.get('segment_recommendation')
                else:
                    extra_stats.setdefault('segment_summary', None)
                    extra_stats.setdefault('segment_recommendation', None)
            
            # 保存分析结果
            self.analysis_results[factor] = {
                'stats_info': stats_info,
                'ic_mean': ic_mean,
                'ic_std': ic_std,
                'ir': ir,
                't_stat': t_stat,
                'p_value': p_value,
                'group_results': group_results,
                'extra_stats': extra_stats
            }
            
            # 注释掉自动生成CSV文件的代码，将在用户选择因子后再生成
            # if group_results and 'avg_returns' in group_results:
            #     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            #     safe_factor_name = factor.replace('/', '_').replace('\\', '_').replace(':', '_')
            #     csv_filename = f'十等分分组收益_{safe_factor_name}_{timestamp}.csv'
            #     group_results['avg_returns'].to_csv(csv_filename, index=False, encoding='utf-8-sig')
            #     print(f"  因子 {factor} 的分组收益表格已保存至: {csv_filename}")
        
        print("因子分析完成")
        
        return True
    
    def plot_factor_distribution(self):
        """
        绘制因子分布图
        """
        if not HAS_PLOT:
            print("可视化功能不可用，跳过因子分布绘图")
            return False
            
        if not hasattr(self, 'processed_data'):
            print("错误：请先运行因子分析")
            return False
        
        print("\n绘制因子分布图...")
        n_factors = len(self.factors)
        n_cols = min(2, n_factors)
        n_rows = (n_factors + n_cols - 1) // n_cols
        
        plt.figure(figsize=(12, 4 * n_rows))
        
        for i, factor in enumerate(self.factors):
            if factor not in self.processed_data.columns:
                continue
            
            ax = plt.subplot(n_rows, n_cols, i + 1)
            sns.histplot(self.processed_data[factor].dropna(), kde=True, ax=ax)
            plt.title(f'{factor} 分布')
            plt.xlabel(factor)
            plt.ylabel('频次')
        
        plt.tight_layout()
        plt.savefig('因子分布图.png', dpi=300, bbox_inches='tight')
        print("因子分布图已保存为 '因子分布图.png'")
        # 移除plt.show()以避免阻塞和KeyboardInterrupt错误
        plt.close()  # 关闭当前图像以释放内存
        return True
    
    def plot_group_returns(self):
        """
        绘制分组收益图
        """
        if not HAS_PLOT:
            print("可视化功能不可用，跳过分组收益绘图")
            return False
            
        if not self.analysis_results:
            print("错误：请先运行因子分析")
            return False
        
        print("\n绘制分组收益图...")
        n_factors = len(self.analysis_results)
        n_cols = min(2, n_factors)
        n_rows = (n_factors + n_cols - 1) // n_cols
        
        plt.figure(figsize=(12, 4 * n_rows))
        
        for i, (factor, results) in enumerate(self.analysis_results.items()):
            if 'group_results' not in results or results['group_results'] is None:
                continue
            
            ax = plt.subplot(n_rows, n_cols, i + 1)
            group_returns = results['group_results']['avg_returns']
            
            sns.barplot(x='分组', y='平均收益', data=group_returns, ax=ax)
            plt.title(f'{factor} 分组收益')
            plt.xlabel('分组')
            plt.ylabel(f'{self.return_col} (平均)')
            
            # 添加数值标签
            for j, row in group_returns.iterrows():
                ax.text(j, row['平均收益'], f'{row['平均收益']:.3f}', 
                        ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('分组收益图.png', dpi=300, bbox_inches='tight')
        print("分组收益图已保存为 '分组收益图.png'")
        # 移除plt.show()以避免阻塞和KeyboardInterrupt错误
        plt.close()  # 关闭当前图像以释放内存
        return True
    
    def _get_new_scoring_weights(self, is_negative_factor=False):
        """
        获取新的评分权重配置（基于改进报告建议）
        
        Args:
            is_negative_factor: 是否为负向因子
            
        Returns:
            dict: 权重配置字典
        """
        if is_negative_factor:
            # 负向因子专门评分体系（基于建议3）
            return {
                'ic_strength': 0.40,    # 负向强度评分（40%权重）
                'significance': 0.30,   # 统计显著性（30%权重）
                'stability': 0.20,      # IR值稳定性（20%权重）
                'return_performance': 0.10  # 收益表现（10%权重）
            }
        else:
            # 正向因子权重配置（基于建议2）
            return {
                'ic_mean': 0.35,        # IC均值（35%权重，提升预测能力重要性）
                'significance': 0.25,   # 统计显著性（25%权重，提升可靠性重视度）
                'ir_value': 0.20,       # IR值（20%权重，降低稳定性权重）
                'return_performance': 0.20  # 多空收益（20%权重，降低收益权重）
            }

    def _calculate_improved_scores(self, ic_mean, ic_std, ir, t_stat, p_value, long_short_return):
        """
        基于新标准的改进评分计算（综合建议1、2、3）
        
        Args:
            ic_mean: IC均值
            ic_std: IC标准差
            ir: IR值
            t_stat: t统计量
            p_value: p值
            long_short_return: 多空收益
            
        Returns:
            dict: 包含所有维度评分的字典
        """
        # 识别因子类型
        factor_type = self._identify_factor_type(ic_mean, long_short_return)
        is_negative_factor = ic_mean < 0
        
        # 获取对应权重配置
        weights = self._get_new_scoring_weights(is_negative_factor)
        
        if is_negative_factor:
            # 负向因子专门评分体系（建议3）
            scores = self._calculate_negative_factor_scores(
                ic_mean, ir, p_value, long_short_return, weights
            )
        else:
            # 正向因子评分体系（建议2）
            scores = self._calculate_positive_factor_scores(
                ic_mean, ir, p_value, long_short_return, weights
            )
        
        # 计算加权总分
        total_score = sum(scores[metric] * weight
                         for metric, weight in weights.items())
        
        scores['total_score'] = total_score
        scores['factor_type'] = factor_type
        scores['is_negative'] = is_negative_factor
        
        return scores

    def _calculate_positive_factor_scores(self, ic_mean, ir, p_value, long_short_return, weights):
        """
        计算正向因子各维度得分（基于建议2的权重配置）
        """
        scores = {}
        
        # 1. IC均值评分（35%权重）- 预测能力
        ic_score = self._score_ic_mean_new_standard(ic_mean)
        scores['ic_mean'] = ic_score
        
        # 2. 统计显著性评分（25%权重）- 可靠性
        sig_score = self._score_statistical_significance_new(p_value)
        scores['significance'] = sig_score
        
        # 3. IR值评分（20%权重）- 稳定性
        ir_score = self._score_ir_value_new_standard(ir)
        scores['ir_value'] = ir_score
        
        # 4. 多空收益评分（20%权重）- 收益能力
        return_score = self._score_long_short_return_new_standard(long_short_return)
        scores['return_performance'] = return_score
        
        return scores

    def _calculate_negative_factor_scores(self, ic_mean, ir, p_value, long_short_return, weights):
        """
        计算负向因子各维度得分（基于建议3的专门体系）
        """
        scores = {}
        
        # 1. 负向强度评分（40%权重）- |IC均值|越大越好
        negative_strength_score = self._score_negative_intensity(abs(ic_mean))
        scores['ic_strength'] = negative_strength_score
        
        # 2. 统计显著性评分（30%权重）- p值显著性
        sig_score = self._score_statistical_significance_new(p_value)
        scores['significance'] = sig_score
        
        # 3. IR值稳定性（20%权重）- 稳定性表现
        stability_score = self._score_stability_new(ir)
        scores['stability'] = stability_score
        
        # 4. 收益表现（10%权重）- 多空收益
        return_score = self._score_return_performance_negative(long_short_return)
        scores['return_performance'] = return_score
        
        return scores

    def _apply_domestic_standards(self, total_score, ic_mean, ir, factor_type, is_negative=False):
        """
        应用国内量化实践标准（基于建议1）
        
        Args:
            total_score: 总分
            ic_mean: IC均值
            ir: IR值
            factor_type: 因子类型
            is_negative: 是否为负向因子
            
        Returns:
            tuple: (rating, status, usage)
        """
        # 国内实践标准（建议1）
        # A级因子：IC均值>0.08，IR>0.3
        # B级因子：IC均值>0.05，IR>0.2
        
        # 特别处理：IC均值>0.12且多空收益>0.04的优秀因子（保持特殊机制）
        if not is_negative and ic_mean >= 0.12 and abs(ir) >= 0.3:
            if total_score >= 3.5:
                return 'A+', '卓越', '强烈推荐使用'
            elif total_score >= 3.0:
                return 'A', '优秀', '推荐使用'
        
        # 基于国内实践标准的评级
        if not is_negative:
            # 正向因子评级标准（结合国内实践）
            if ic_mean >= 0.08 and ir >= 0.3:
                # 国内A级标准
                if total_score >= 3.5:
                    rating = 'A+'
                elif total_score >= 3.0:
                    rating = 'A'
                else:
                    rating = 'A-'  # 保持对优秀IC因子的特殊评级
            elif ic_mean >= 0.05 and ir >= 0.2:
                # 国内B级标准
                if total_score >= 2.5:
                    rating = 'B+'
                elif total_score >= 2.0:
                    rating = 'B'
                else:
                    rating = 'B-'  # 基于国内标准调整
            elif ic_mean >= 0.02:
                # 国内C级标准
                if total_score >= 1.5:
                    rating = 'C+'
                elif total_score >= 1.0:
                    rating = 'C'
                else:
                    rating = 'C-'
            else:
                rating = 'D'  # 无效因子
        else:
            # 负向因子评级（基于绝对IC值和稳定性）
            abs_ic = abs(ic_mean)
            if abs_ic >= 0.08 and abs(ir) >= 0.3:
                rating = 'A-'  # 负向因子使用A-表示优秀反向因子
            elif abs_ic >= 0.05 and abs(ir) >= 0.2:
                rating = 'B+'
            elif abs_ic >= 0.03:
                rating = 'B'
            elif abs_ic >= 0.02:
                rating = 'C+'
            else:
                rating = 'D'
        
        # 状态和使用建议映射
        status_mapping = {
            'A+': '卓越', 'A': '优秀', 'A-': '优秀', 'B+': '良好',
            'B': '一般', 'C+': '较弱', 'C': '弱', 'D': '无效'
        }
        
        usage_mapping = {
            'A+': '强烈推荐使用', 'A': '推荐使用', 'A-': '推荐使用',
            'B+': '可考虑使用', 'B': '谨慎使用', 'C+': '不推荐使用',
            'C': '不建议使用', 'D': '避免使用'
        }
        
        status = status_mapping.get(rating, '未知')
        usage = usage_mapping.get(rating, '需重新评估')
        
        return rating, status, usage

    def _score_ic_mean_new_standard(self, ic_mean):
        """
        基于国内量化实践的IC均值评分（建议1）
        A级：>0.08，B级：>0.05
        """
        abs_ic = abs(ic_mean)
        if abs_ic >= 0.12:
            return 4.0  # 优秀（超额奖励机制）
        elif abs_ic >= 0.08:
            return 3.5  # 国内A级标准
        elif abs_ic >= 0.05:
            return 3.0  # 国内B级标准
        elif abs_ic >= 0.02:
            return 2.0  # 有效阈值
        elif abs_ic >= 0.01:
            return 1.0  # 弱
        else:
            return 0.5  # 极弱

    def _score_ir_value_new_standard(self, ir):
        """
        基于建议2的IR值评分（降低权重但保持重要性）
        """
        abs_ir = abs(ir)
        if abs_ir >= 1.5:
            return 2.5  # 极强（保持原有高分）
        elif abs_ir >= 1.0:
            return 2.0  # 强
        elif abs_ir >= 0.5:
            return 1.5  # 中等（国内常见标准）
        elif abs_ir >= 0.3:
            return 1.0  # 弱
        elif abs_ir >= 0.15:
            return 0.8  # 较弱
        else:
            return 0.5  # 极弱

    def _score_statistical_significance_new(self, p_value):
        """
        基于建议2的统计显著性评分（提升权重至25%）
        """
        if np.isnan(p_value):
            return 0.3  # 数据缺失
        
        if p_value < 0.01:
            return 1.0  # 高度显著
        elif p_value < 0.05:
            return 0.8  # 显著（国内实践重视但不过度依赖）
        elif p_value < 0.1:
            return 0.6  # 边缘显著
        else:
            return 0.3  # 不显著

    def _score_long_short_return_new_standard(self, long_short_return):
        """
        基于建议2的多空收益评分（降低权重至20%）
        """
        if np.isnan(long_short_return):
            return 1.0  # 数据缺失，默认中等
        
        abs_return = abs(long_short_return)
        if abs_return >= 0.04:
            return 2.0  # 优秀（降低满分，保持重要性）
        elif abs_return >= 0.03:
            return 1.8  # 强
        elif abs_return >= 0.02:
            return 1.5  # 中等
        elif abs_return >= 0.01:
            return 1.0  # 弱
        else:
            return 0.5  # 极弱

    def _score_negative_intensity(self, abs_ic_mean):
        """
        基于建议3的负向强度评分
        """
        if abs_ic_mean >= 0.1:
            return 4.0  # 强负向
        elif abs_ic_mean >= 0.07:
            return 3.5  # 中强负向
        elif abs_ic_mean >= 0.05:
            return 3.0  # 中等负向
        elif abs_ic_mean >= 0.03:
            return 2.0  # 弱负向
        else:
            return 1.0  # 极弱负向

    def _score_stability_new(self, ir):
        """
        基于建议3的稳定性评分（负向因子20%权重）
        """
        abs_ir = abs(ir)
        if abs_ir >= 1.5:
            return 2.0  # 极强稳定性
        elif abs_ir >= 1.0:
            return 1.5  # 强稳定性
        elif abs_ir >= 0.5:
            return 1.0  # 中等稳定性
        elif abs_ir >= 0.2:
            return 0.8  # 一般稳定性
        else:
            return 0.5  # 较差稳定性

    def _score_return_performance_negative(self, long_short_return):
        """
        基于建议3的负向因子收益表现评分（10%权重）
        """
        if np.isnan(long_short_return):
            return 0.5  # 数据缺失
        
        # 负向因子希望多空收益为负
        if long_short_return < -0.02:
            return 1.0  # 优秀反向收益
        elif long_short_return < -0.01:
            return 0.8  # 良好反向收益
        elif long_short_return < 0:
            return 0.6  # 一般反向收益
        else:
            return 0.3  # 收益为正但因子为负向，可能存在数据问题

    def _generate_improved_detailed_reason(self, rating, ic_mean, ir, p_value, long_short_return, factor_type, scores):
        """
        生成改进的详细理由说明
        """
        reasons = []
        
        # 因子类型标识
        factor_direction = "负向" if ic_mean < 0 else "正向"
        
        if rating in ['A+', 'A', 'A-']:
            reasons.append(f"[OK] 优秀表现：{rating}级{factor_direction}因子，具有强预测能力和高收益性")
            reasons.append(f"-  IC均值{abs(ic_mean):.3f}，{'超过国内A级标准(>0.08)' if abs(ic_mean) > 0.08 else '接近国内A级标准'}")
            reasons.append(f"-  IR值{abs(ir):.3f}，稳定性{'优秀' if abs(ir) > 1.5 else '良好' if abs(ir) > 1.0 else '一般'}")
            if p_value < 0.05:
                reasons.append(f"-  统计显著(p值={p_value:.3f})")
            reasons.append(f"-  类型：{factor_type}")
            reasons.append("使用建议：强烈推荐使用，可作为组合核心配置，权重15-25%")
            
        elif rating == 'B+':
            reasons.append(f"-  良好表现：B+级{factor_direction}因子，符合国内B级标准")
            reasons.append(f"-  IC均值{abs(ic_mean):.3f}，{'达到国内B级标准(>0.05)' if abs(ic_mean) > 0.05 else '接近国内B级标准'}")
            reasons.append(f"-  IR值{abs(ir):.3f}，稳定性一般")
            if p_value < 0.1:
                reasons.append(f"-  p值={p_value:.3f}，{'统计显著' if p_value < 0.05 else '边缘显著'}")
            reasons.append(f"-  类型：{factor_type}")
            reasons.append("使用建议：可谨慎使用，权重控制在10%以内，加强监控")
            
        elif rating == 'B':
            reasons.append(f"-  一般表现：B级{factor_direction}因子，具有基础预测能力")
            reasons.append(f"-  IC均值{abs(ic_mean):.3f}，预测能力一般")
            reasons.append(f"-  IR值{abs(ir):.3f}，稳定性有限")
            reasons.append(f"-  类型：{factor_type}")
            reasons.append("使用建议：谨慎使用，权重控制在5%以内，定期评估")
            
        elif rating in ['C+', 'C']:
            reasons.append(f"✗ 表现不佳：该因子{'C+' if rating == 'C+' else 'C'}级")
            reasons.append(f"-  IC均值{abs(ic_mean):.3f}，预测能力不足")
            reasons.append(f"-  IR值{abs(ir):.3f}，稳定性较差")
            if p_value >= 0.05:
                reasons.append(f"-  p值={p_value:.3f}，统计不显著")
            reasons.append(f"-  类型：{factor_type}")
            if rating == 'C+':
                reasons.append("使用建议：不推荐使用，如需使用请严格控制权重5%以下")
            else:
                reasons.append("使用建议：避免使用，确定无效，继续使用可能造成损失")
        
        return '\n'.join(reasons)

    def _evaluate_factor_performance(self, ic_mean, ic_std, ir, t_stat, p_value, long_short_return):
        """
        改进的因子性能评估函数（综合应用建议1、2、3）
        
        新的评分体系：
        - 正向因子：IC均值35% + 统计显著性25% + IR值20% + 多空收益20%
        - 负向因子：负向强度40% + 统计显著性30% + 稳定性20% + 收益表现10%
        - 评级标准：采用国内量化实践标准
        """
        
        # 使用新的评分计算方法
        scores = self._calculate_improved_scores(
            ic_mean, ic_std, ir, t_stat, p_value, long_short_return
        )
        
        # 应用国内量化实践标准进行评级
        rating, status, usage = self._apply_domestic_standards(
            scores['total_score'], ic_mean, ir, scores['factor_type'], scores['is_negative']
        )
        
        # 生成详细理由
        detailed_reason = self._generate_improved_detailed_reason(
            rating, ic_mean, ir, p_value, long_short_return, scores['factor_type'], scores
        )
        
        return {
            'score': round(scores['total_score'], 1),
            'rating': rating,
            'status': status,
            'usage': usage,
            'detailed_reason': detailed_reason,
            'factor_type': scores['factor_type'],
            # 新增：详细维度得分
            'ic_score': scores.get('ic_mean', scores.get('ic_strength', 0)),
            'significance_score': scores['significance'],
            'stability_score': scores.get('ir_value', scores.get('stability', 0)),
            'return_score': scores['return_performance'],
            'is_negative_factor': scores['is_negative']
        }
    
    def _generate_detailed_reason(self, rating, ic_mean, ir, p_value, long_short_return, factor_type):
        """
        根据新评级和因子指标生成精简的详细理由
        """
        reasons = []
        
        if rating in ['A+', 'A']:
            reasons.append(f"[OK] 优秀表现：{rating}级因子，具有强预测能力和高收益性")
            reasons.append(f"-  IC均值{abs(ic_mean):.3f}，预测能力强")
            reasons.append(f"-  多空收益{abs(long_short_return):.3f}，收益表现卓越")
            reasons.append(f"-  IR值{abs(ir):.3f}，稳定性{'优秀' if abs(ir) > 1.5 else '良好'}")
            if p_value < 0.05:
                reasons.append(f"-  统计显著(p值={p_value:.3f})")
            reasons.append(f"-  类型：{factor_type}")
            reasons.append("使用建议：强烈推荐使用，可作为组合核心配置，权重15-25%")
            
        elif rating == 'B+':
            reasons.append(f"-  良好表现：B+级因子，具有中等预测能力和收益性")
            reasons.append(f"-  IC均值{abs(ic_mean):.3f}，预测能力{'中等' if abs(ic_mean) > 0.05 else '一般'}")
            reasons.append(f"-  多空收益{abs(long_short_return):.3f}，收益表现{'优秀' if abs(long_short_return) > 0.02 else '一般'}")
            reasons.append(f"-  IR值{abs(ir):.3f}，稳定性一般")
            if p_value < 0.05:
                reasons.append(f"-  统计显著(p值={p_value:.3f})")
            else:
                reasons.append(f"-  p值={p_value:.3f}")
            reasons.append(f"-  类型：{factor_type}")
            reasons.append("使用建议：可谨慎使用，权重控制在10%以内，加强监控")
            
        elif rating == 'B':
            reasons.append(f"-  一般表现：B级因子，具有基础预测能力")
            reasons.append(f"-  IC均值{abs(ic_mean):.3f}，预测能力一般")
            reasons.append(f"-  多空收益{abs(long_short_return):.3f}，收益表现一般")
            reasons.append(f"-  IR值{abs(ir):.3f}，稳定性有限")
            if p_value < 0.05:
                reasons.append(f"-  统计显著(p值={p_value:.3f})")
            else:
                reasons.append(f"-  p值={p_value:.3f}")
            reasons.append(f"-  类型：{factor_type}")
            reasons.append("使用建议：谨慎使用，权重控制在5%以内，定期评估")
            
        elif rating in ['C+', 'C']:
            reasons.append(f"✗ 表现不佳：该因子{'C+' if rating == 'C+' else 'C'}级")
            reasons.append(f"-  IC均值{abs(ic_mean):.3f}，预测能力不足")
            reasons.append(f"-  IR值{abs(ir):.3f}，稳定性较差")
            if p_value >= 0.05:
                reasons.append(f"-  p值={p_value:.3f}，统计不显著")
            reasons.append(f"-  类型：{factor_type}")
            if rating == 'C+':
                reasons.append("使用建议：不推荐使用，如需使用请严格控制权重5%以下")
            else:
                reasons.append("使用建议：避免使用，确定无效，继续使用可能造成损失")
        
        return '\n'.join(reasons)
    
    def _identify_factor_type(self, ic_mean, long_short_return):
        """
        识别因子类型：线性、非线性、无效
        """
        if abs(ic_mean) < 0.02:
            return "无效因子"
        elif not np.isnan(long_short_return) and abs(long_short_return) > abs(ic_mean) * 2:
            return "非线性因子"
        else:
            return "线性因子"
    
    def _generate_data_driven_comparison(self, summary_df):
        """
        生成基于客观数据的因子对比分析
        """
        comparison_results = {
            'objective_analysis': {},
            'data_driven_insights': {}
        }
        
        # 1. 客观数据验证
        if not summary_df.empty:
            # 计算各指标的统计分布
            ic_mean_values = summary_df['IC均值'].dropna()
            ir_values = summary_df['IR值'].dropna()
            ls_returns = summary_df['多空收益'].dropna()
            
            # 为每个因子生成客观评级
            factor_objective_grades = []
            
            for _, row in summary_df.iterrows():
                factor_name = row['因子名称']
                ic_mean = row['IC均值']
                ir = row['IR值']
                ls_return = row.get('多空收益', np.nan)
                
                # 客观评级逻辑
                objective_score = 0
                
                # IC均值客观评分
                if abs(ic_mean) >= 0.1:
                    objective_score += 30
                    ic_grade = "优秀"
                elif abs(ic_mean) >= 0.05:
                    objective_score += 20
                    ic_grade = "良好"
                elif abs(ic_mean) >= 0.02:
                    objective_score += 10
                    ic_grade = "一般"
                else:
                    ic_grade = "较差"
                
                # 多空收益客观评分（核心指标）
                if not np.isnan(ls_return) and abs(ls_return) >= 0.03:
                    objective_score += 35
                    ls_grade = "卓越"
                elif not np.isnan(ls_return) and abs(ls_return) >= 0.02:
                    objective_score += 25
                    ls_grade = "优秀"
                elif not np.isnan(ls_return) and abs(ls_return) >= 0.01:
                    objective_score += 15
                    ls_grade = "良好"
                else:
                    ls_grade = "一般"
                
                # IR值客观评分
                if abs(ir) >= 1.5:
                    objective_score += 25
                    ir_grade = "优秀"
                elif abs(ir) >= 1.0:
                    objective_score += 20
                    ir_grade = "良好"
                elif abs(ir) >= 0.5:
                    objective_score += 10
                    ir_grade = "一般"
                else:
                    ir_grade = "较差"
                
                # 统计显著性客观评分
                p_value = row.get('p值', np.nan)
                if not np.isnan(p_value):
                    if p_value < 0.01:
                        objective_score += 10
                        sig_grade = "高度显著"
                    elif p_value < 0.05:
                        objective_score += 8
                        sig_grade = "显著"
                    elif p_value < 0.1:
                        objective_score += 5
                        sig_grade = "边缘显著"
                    else:
                        objective_score += 2
                        sig_grade = "不显著"
                else:
                    sig_grade = "数据缺失"
                    objective_score += 2
                
                # 生成客观评级
                if objective_score >= 85:
                    objective_grade = "A+"
                elif objective_score >= 75:
                    objective_grade = "A"
                elif objective_score >= 65:
                    objective_grade = "B+"
                elif objective_score >= 55:
                    objective_grade = "B"
                elif objective_score >= 40:
                    objective_grade = "C+"
                elif objective_score >= 25:
                    objective_grade = "C"
                else:
                    objective_grade = "D"
                
                factor_objective_grades.append({
                    'factor_name': factor_name,
                    'objective_score': objective_score,
                    'objective_grade': objective_grade,
                    'ic_grade': ic_grade,
                    'ls_grade': ls_grade,
                    'ir_grade': ir_grade,
                    'sig_grade': sig_grade,
                    'ic_mean': ic_mean,
                    'ls_return': ls_return,
                    'ir': ir
                })
            
            comparison_results['objective_analysis'] = {
                'factor_grades': factor_objective_grades,
                'summary_stats': {
                    'total_factors': len(factor_objective_grades),
                    'excellent_factors': len([f for f in factor_objective_grades if f['objective_score'] >= 75]),
                    'good_factors': len([f for f in factor_objective_grades if 65 <= f['objective_score'] < 75]),
                    'average_ic': ic_mean_values.mean() if len(ic_mean_values) > 0 else np.nan,
                    'average_ls_return': ls_returns.mean() if len(ls_returns) > 0 else np.nan,
                    'top_performer': max(factor_objective_grades, key=lambda x: x['objective_score']) if factor_objective_grades else None
                }
            }
        
        return comparison_results
    def _generate_executive_summary(self, summary_df):
        """
        生成精简执行摘要
        """
        # 统计因子整体表现
        total_factors = len(summary_df)
        
        # 按新评价体系重新评估所有因子
        factor_scores = []
        factor_ratings = []
        factor_types = []
        factor_names = []
        
        for _, row in summary_df.iterrows():
            factor_name = row['因子名称']
            ic_mean = row['IC均值']
            ic_std = row['IC标准差']
            ir = row['IR值']
            t_stat = row.get('t统计量', np.nan)
            p_value = row.get('p值', np.nan)
            long_short_return = row.get('多空收益', np.nan)
            
            eval_result = self._evaluate_factor_performance(ic_mean, ic_std, ir, t_stat, p_value, long_short_return)
            factor_scores.append(eval_result['score'])
            factor_ratings.append(eval_result['rating'])
            factor_types.append(eval_result['factor_type'])
            factor_names.append(factor_name)
        
        # 转换为DataFrame进行分析
        analysis_df = pd.DataFrame({
            '因子名称': factor_names,
            '综合得分': factor_scores,
            '评级': factor_ratings,
            '因子类型': factor_types
        })
        
        # 最佳因子
        if not analysis_df.empty and '综合得分' in analysis_df.columns and not analysis_df['综合得分'].isna().all():
            best_factor_idx = analysis_df['综合得分'].idxmax()
            best_factor = analysis_df.iloc[best_factor_idx]
        else:
            # 处理异常情况
            best_factor = None
            print("警告: 无法确定最佳因子（可能是因为数据为空或缺少'综合得分'列）")
        
        # 因子分布统计
        rating_counts = analysis_df['评级'].value_counts()
        type_counts = analysis_df['因子类型'].value_counts()
        
        # 优秀因子数量 (A+和A级)
        excellent_count = rating_counts.get('A+', 0) + rating_counts.get('A', 0)
        good_count = rating_counts.get('B+', 0) + rating_counts.get('B', 0)
        poor_count = rating_counts.get('C+', 0) + rating_counts.get('C', 0) + rating_counts.get('D', 0)
        
        # 生成精简摘要内容
        summary_lines = []
        
        # 核心摘要
        summary_lines.append(f"本次分析共评估 {total_factors} 个因子，整体表现{'良好' if excellent_count >= 3 else '一般' if good_count >= 2 else '较差'}。")
        
        # 核心发现
        if excellent_count > 0 and best_factor is not None:
            summary_lines.append(f"-  发现 {excellent_count} 个优秀因子(A+和A级)，其中 {best_factor['因子名称']} 表现最佳(评级:{best_factor['评级']})。")
        elif excellent_count > 0:
            summary_lines.append(f"-  发现 {excellent_count} 个优秀因子(A+和A级)，但无法确定最佳因子。")
        
        # 因子类型分析
        if type_counts.get('非线性因子', 0) > 0:
            summary_lines.append(f"-  检测到 {type_counts['非线性因子']} 个非线性因子，建议采用分组选股策略。")
        
        # 投资建议
        if excellent_count >= 2:
            summary_lines.append(f"💡 投资建议：采用多因子组合策略，重点配置A级以上因子。")
        elif good_count >= 1:
            summary_lines.append(f"💡 投资建议：选择表现最佳的2-3个因子进行策略构建。")
        else:
            summary_lines.append(f"💡 投资建议：当前因子表现有限，建议重新因子开发。")
        
        # 风险提示
        if poor_count >= total_factors * 0.5:
            summary_lines.append("[警告]  风险提示：多数因子表现较差，存在模型失效风险，建议谨慎使用。")
        elif poor_count >= total_factors * 0.3:
            summary_lines.append("[警告]  风险提示：部分因子表现不佳，需结合其他指标验证。")
        
        return '\n'.join(summary_lines)
    
    def classify_factors_by_ic(self):
        """
        根据IC均值将因子分为正向和负向两类
        
        Returns:
            tuple: (positive_factors_df, negative_factors_df) 两个DataFrame，按IC均值排序
        """
        return _fa_classify_factors_by_ic(self)
    
    def generate_factor_classification_overview(self):
        """
        生成因子分类概览
        
        Returns:
            str: 概览信息字符串
        """
        return _fa_generate_factor_classification_overview(self)
    
    def _get_suggested_weight(self, rating, is_positive):
        """
        根据因子评级获取建议权重
        
        Args:
            rating: 因子评级
            is_positive: 是否为正向因子
        
        Returns:
            str: 建议权重范围
        """
        return _fa_get_suggested_weight(self, rating, is_positive)
     
    def _get_scoring_standards(self):
        """
        获取因子评分标准的详细说明
        
        Returns:
            str: 评分标准说明文本
        """
        return _fa_get_scoring_standards(self)
    
    def generate_positive_factors_analysis(self, summary_mode=False):
        """
        - - - - - - - - - - - - 

        Returns:
            str: - - - - - - - - - - 
        """
        return _fa_generate_positive_factors_analysis(self, summary_mode=summary_mode)

    def generate_negative_factors_analysis(self, summary_mode=False):
        """
        - - - - - - - - - - - - 

        Returns:
            str: - - - - - - - - - - 
        """
        return _fa_generate_negative_factors_analysis(self, summary_mode=summary_mode)
    
    def analyze_rolling_ic(self, factor_col, window_sizes=(30, 60), compute_ic_decay=True, save_plots=False):
        """
        使用通用滚动窗口分析工具评估因子在不同时间窗内的IC表现。
        """
        if self.processed_data is None or factor_col not in self.processed_data.columns:
            print(f"滚动窗口分析失败：未找到因子 {factor_col} 的预处理数据")
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
    
    def analyze_temporal_stability(self, factor_results=None):
        """
        调用通用时序稳定性工具，评估因子IC的时间一致性。
        """
        factor_results = factor_results or getattr(self, 'analysis_results', {})
        return temporal_stability_analysis(factor_results)
    
    def analyze_sample_sensitivity(self, factor_col, sample_sizes=(0.8, 0.9, 1.0), n_iterations=100):
        """
        使用统一的样本敏感性分析工具，评估不同抽样比例下IC的波动情况。
        """
        if self.processed_data is None or factor_col not in self.processed_data.columns:
            print(f"样本敏感性分析失败：未找到因子 {factor_col} 的预处理数据")
            return {}
        df = self.processed_data[[factor_col, self.return_col]].dropna()
        return sample_sensitivity_analysis(
            df,
            factor_col,
            self.return_col,
            sample_sizes=sample_sizes,
            n_iterations=n_iterations,
        )
    
    def generate_factor_analysis_report(self, summary_df, process_factors=False, factor_method='standardize', winsorize=False, summary_mode=False):
        """
        - - - - - - - - - - - 

        Args:
            summary_df: - - - - - - - - - 
            process_factors: - - - - - - - - - - 
            factor_method: - - - - - - - 'standardize'- - - - - -  'normalize'- - - - - 
            winsorize: - - - - - - - - - 
        """
        return _fa_generate_factor_analysis_report(
            self,
            summary_df,
            process_factors=process_factors,
            factor_method=factor_method,
            winsorize=winsorize,
            summary_mode=summary_mode,
        )

    def generate_auxiliary_analysis_report(
        self,
        factors=None,
        window_sizes=(30, 60),
        sample_sizes=(0.8, 0.9, 1.0),
        n_iterations=100,
    ):
        """
        收集滚动IC、时序稳定性与样本敏感性的统计信息。
        结果缓存到 self.auxiliary_stats，并输出结构化摘要 CSV。
        """
        if not hasattr(self, 'processed_data') or self.processed_data is None:
            print("辅助分析报告生成失败：请先执行数据预处理")
            return None

        analysis_results = getattr(self, 'analysis_results', None)
        if not analysis_results:
            print("辅助分析报告生成失败：请先完成因子分析")
            return None

        if not isinstance(factors, (list, tuple)):
            factors = None
        target_factors = factors or list(analysis_results.keys()) or list(self.factors)
        target_factors = [
            factor for factor in target_factors
            if factor in analysis_results and factor in self.processed_data.columns
        ]
        if not target_factors:
            print("辅助分析报告生成失败：没有可用于生成报告的因子")
            return None

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_rows = []
        auxiliary_stats = {}
        score_rows = []

        def _safe_average(values):
            cleaned = []
            for val in values:
                if val is None:
                    continue
                try:
                    if pd.isna(val):
                        continue
                except Exception:
                    pass
                try:
                    cleaned.append(float(val))
                except (TypeError, ValueError):
                    continue
            if not cleaned:
                return None
            return float(np.mean(cleaned))

        def _summarize_auxiliary_metrics(entry):
            metrics = {}
            rolling_payload = entry.get('rolling') or {}
            cv_vals, half_vals, maic_vals = [], [], []
            for payload in rolling_payload.values():
                stability = payload.get('stability') or {}
                decay = payload.get('decay') or {}
                cv_vals.append(stability.get('coefficient_of_variation'))
                maic_vals.append(stability.get('mean_abs_ic'))
                half_vals.append(decay.get('half_life'))
            metrics['rolling_cv_avg'] = _safe_average(cv_vals)
            metrics['rolling_half_life_avg'] = _safe_average(half_vals)
            metrics['rolling_maic_avg'] = _safe_average(maic_vals)

            temporal = entry.get('temporal') or {}
            ic_stability = temporal.get('ic_stability') or {}
            trends = temporal.get('temporal_trends') or {}
            rank_stats = temporal.get('rank_stability') or {}
            trend_corr = ic_stability.get('trend_correlation')
            if trend_corr is None:
                trend_corr = trends.get('trend_correlation')
            metrics['temporal_autocorr_lag1'] = ic_stability.get('autocorr_lag1')
            metrics['temporal_trend_corr'] = trend_corr
            metrics['temporal_sign_changes'] = trends.get('sign_changes')
            metrics['temporal_mean_reversion'] = trends.get('mean_reversion_strength')
            metrics['temporal_rank_volatility'] = rank_stats.get('ranking_volatility')

            sample = entry.get('sample') or {}
            effects = sample.get('sample_size_effects') or {}
            std_vals, mean_vals, success_vals = [], [], []
            for stats in effects.values():
                if not stats:
                    continue
                std_vals.append(stats.get('ic_std'))
                mean_vals.append(stats.get('ic_mean'))
                success_vals.append(stats.get('success_rate'))
            metrics['sample_ic_mean_avg'] = _safe_average(mean_vals)
            metrics['sample_ic_std_avg'] = _safe_average(std_vals)
            metrics['sample_success_rate_avg'] = _safe_average(success_vals)
            robustness = sample.get('robustness_metrics') or {}
            metrics['sample_cross_variance'] = robustness.get('mean_variance_across_samples')
            metrics['sample_best_ratio'] = robustness.get('best_sample_size')
            metrics['sample_most_stable_ratio'] = robustness.get('most_stable_sample_size')

            return metrics

        for factor in target_factors:
            factor_result = analysis_results.get(factor, {})
            row = {
                '因子': factor,
                'ic_mean': factor_result.get('ic_mean'),
                'ic_std': factor_result.get('ic_std'),
                'ir': factor_result.get('ir'),
                't_stat': factor_result.get('t_stat'),
                'p_value': factor_result.get('p_value'),
                'long_short_return': (factor_result.get('group_results') or {}).get('long_short_return'),
            }
            extra_stats = factor_result.get('extra_stats') or {}
            overall_metrics = extra_stats.get('overall_metrics') or {}
            if not overall_metrics:
                overall_metrics = {
                    key: extra_stats.get(key)
                    for key in extra_stats.keys()
                    if key.startswith('overall_')
                }

            def _resolve_overall(field):
                if overall_metrics and field in overall_metrics and overall_metrics[field] is not None:
                    return overall_metrics[field]
                return extra_stats.get(field)

            row['overall_ic'] = _resolve_overall('overall_ic')
            row['overall_ir'] = _resolve_overall('overall_ir')
            row['overall_p_value'] = _resolve_overall('overall_p_value')
            row['overall_sample_size'] = _resolve_overall('overall_sample_size')
            row['overall_mode'] = _resolve_overall('overall_mode')
            row['overall_t_stat'] = _resolve_overall('overall_t_stat')
            row['overall_ci_width'] = _resolve_overall('overall_ci_width')

            row['ic_mode'] = extra_stats.get('ic_mode')
            row['ic_window_days'] = extra_stats.get('ic_window_days')
            row['min_samples_per_day'] = extra_stats.get('min_samples_per_day')
            row['avg_daily_samples'] = extra_stats.get('avg_daily_samples')
            row['median_daily_samples'] = extra_stats.get('median_daily_samples')
            row['p25_daily_samples'] = extra_stats.get('p25_daily_samples')
            row['p75_daily_samples'] = extra_stats.get('p75_daily_samples')
            row['daily_sample_std'] = extra_stats.get('daily_sample_std')
            row['daily_sample_cv'] = extra_stats.get('daily_sample_cv')
            row['daily_top5_share'] = extra_stats.get('daily_top5_share')
            row['daily_points'] = extra_stats.get('daily_points')
            row['effective_dates'] = extra_stats.get('effective_dates')
            row['skip_ratio'] = extra_stats.get('skip_ratio')
            row['daily_missing_ratio'] = extra_stats.get('daily_missing_ratio')
            row['qualified_day_ratio'] = extra_stats.get('qualified_day_ratio')
            row['avg_trades_per_day'] = extra_stats.get('avg_trades_per_day')
            row['trade_count_cv'] = extra_stats.get('trade_count_cv')
            row['ic_window_note'] = extra_stats.get('ic_window_note')
            row['screening_mode'] = extra_stats.get('screening_mode')
            row['segment_primary'] = extra_stats.get('segment_primary')
            row['segment_primary_ratio'] = extra_stats.get('segment_primary_ratio')
            row['segment_secondary'] = extra_stats.get('segment_secondary')
            row['segment_secondary_ratio'] = extra_stats.get('segment_secondary_ratio')
            row['segment_warning'] = "; ".join(extra_stats.get('segment_warnings', [])) if extra_stats.get('segment_warnings') else ""
            row['segment_primary_ic'] = extra_stats.get('segment_primary_ic')
            row['segment_secondary_ic'] = extra_stats.get('segment_secondary_ic')
            row['segment_recommendation'] = extra_stats.get('segment_recommendation')

            aux_entry = {'rolling': {}, 'temporal': {}, 'sample': {}}

            # 1. 滚动窗口统计
            rolling_result = self.analyze_rolling_ic(
                factor,
                window_sizes=window_sizes,
                compute_ic_decay=True,
                save_plots=False,
            )
            ic_series_for_stability = []
            if rolling_result and rolling_result.get('rolling_ic'):
                for window in window_sizes:
                    stats = rolling_result['rolling_ic'].get(window)
                    if not stats:
                        continue
                    ic_values = stats.get('ic_values') or []
                    sample_count = len(ic_values)
                    if not ic_series_for_stability and ic_values:
                        ic_series_for_stability = ic_values
                    row[f'rolling_{window}_mean_ic'] = stats.get('mean_ic')
                    row[f'rolling_{window}_ic_std'] = stats.get('ic_std')
                    row[f'rolling_{window}_window_count'] = sample_count
                    aux_entry['rolling'][window] = {
                        'stats': stats,
                        'sample_count': sample_count,
                    }
                for window, metrics in (rolling_result.get('stability_metrics') or {}).items():
                    row[f'rolling_{window}_cv'] = metrics.get('coefficient_of_variation')
                    row[f'rolling_{window}_persistence'] = metrics.get('persistence')
                    row[f'rolling_{window}_mean_abs_ic'] = metrics.get('mean_abs_ic')
                    aux_entry['rolling'].setdefault(window, {})['stability'] = metrics
                for window, decay in (rolling_result.get('ic_decay') or {}).items():
                    if not decay:
                        continue
                    row[f'rolling_{window}_half_life'] = decay.get('half_life')
                    row[f'rolling_{window}_decay_rate'] = decay.get('decay_rate')
                    row[f'rolling_{window}_initial_ic'] = decay.get('initial_ic')
                    row[f'rolling_{window}_final_ic'] = decay.get('final_ic')
                    aux_entry['rolling'].setdefault(window, {})['decay'] = decay

            # 2. 时序稳定性
            temporal_payload = {'ic_values': ic_series_for_stability} if ic_series_for_stability else None
            temporal_result = self.analyze_temporal_stability(temporal_payload) if temporal_payload else {}
            aux_entry['temporal'] = temporal_result or {}
            if temporal_result and any(temporal_result.values()):
                ic_stability = temporal_result.get('ic_stability') or {}
                if ic_stability:
                    row['temporal_autocorr_lag1'] = ic_stability.get('autocorr_lag1')
                    row['temporal_trend_corr'] = ic_stability.get('trend_correlation')
                    row['temporal_ic_volatility'] = ic_stability.get('ic_volatility')
                temporal_trend = temporal_result.get('temporal_trends') or {}
                if temporal_trend:
                    row['temporal_linear_trend'] = temporal_trend.get('linear_trend')
                    row['temporal_sign_changes'] = temporal_trend.get('sign_changes')
                    row['temporal_mean_reversion'] = temporal_trend.get('mean_reversion_strength')
                rank_stability = temporal_result.get('rank_stability') or {}
                if rank_stability:
                    row['temporal_rank_change'] = rank_stability.get('mean_ranking_change')
                    row['temporal_rank_volatility'] = rank_stability.get('ranking_volatility')

            # 3. 样本敏感性
            sample_result = self.analyze_sample_sensitivity(
                factor,
                sample_sizes=sample_sizes,
                n_iterations=n_iterations,
            )
            aux_entry['sample'] = sample_result or {}
            if sample_result and sample_result.get('sample_size_effects'):
                for size in sample_sizes:
                    stats = sample_result['sample_size_effects'].get(size)
                    if not stats:
                        continue
                    success_rate = stats.get('success_rate')
                    size_label = int(size * 100)
                    row[f'sample_{size_label}_ic_mean'] = stats.get('ic_mean')
                    row[f'sample_{size_label}_ic_std'] = stats.get('ic_std')
                    row[f'sample_{size_label}_ic_q25'] = stats.get('ic_q25')
                    row[f'sample_{size_label}_ic_q75'] = stats.get('ic_q75')
                    row[f'sample_{size_label}_success_rate'] = success_rate
                robustness = sample_result.get('robustness_metrics')
                if robustness:
                    row['sample_ic_stability_across_samples'] = robustness.get('ic_stability_across_samples')
                    row['sample_mean_variance_across_samples'] = robustness.get('mean_variance_across_samples')
                    row['sample_best_ratio'] = robustness.get('best_sample_size')
                    row['sample_most_stable_ratio'] = robustness.get('most_stable_sample_size')

            metric_summary = _summarize_auxiliary_metrics(aux_entry)
            row.update(metric_summary)
            aux_entry['metric_summary'] = metric_summary

            integrated_scores = compute_integrated_factor_scores(factor_result, aux_entry)
            aux_entry['integrated_scores'] = integrated_scores
            row['base_score'] = integrated_scores['base_score']
            row['overall_score'] = integrated_scores['overall_score']
            row['rolling_score'] = integrated_scores['rolling_score']
            row['temporal_score'] = integrated_scores['temporal_score']
            row['sample_score'] = integrated_scores['sample_score']
            row['final_score'] = integrated_scores['final_score']
            row['stability_score'] = integrated_scores['stability_score']
            row['rating'] = integrated_scores['rating']

            score_row = {
                '因子': factor,
                'ic_mean': row.get('ic_mean'),
                'ic_std': row.get('ic_std'),
                'ir': row.get('ir'),
                'long_short_return': row.get('long_short_return'),
                'segment_primary': row.get('segment_primary'),
                'segment_primary_ratio': row.get('segment_primary_ratio'),
                'segment_secondary': row.get('segment_secondary'),
                'segment_secondary_ratio': row.get('segment_secondary_ratio'),
                'segment_warning': row.get('segment_warning'),
                'segment_primary_ic': row.get('segment_primary_ic'),
                'segment_secondary_ic': row.get('segment_secondary_ic'),
                'segment_recommendation': row.get('segment_recommendation'),
                'daily_points': row.get('daily_points'),
                'skip_ratio': row.get('skip_ratio'),
                'ic_mode': row.get('ic_mode'),
                'avg_daily_samples': row.get('avg_daily_samples'),
                'daily_sample_cv': row.get('daily_sample_cv'),
                'base_score': integrated_scores['base_score'],
                'overall_ic': row.get('overall_ic'),
                'overall_ir': row.get('overall_ir'),
                'overall_p_value': row.get('overall_p_value'),
                'overall_sample_size': row.get('overall_sample_size'),
                'overall_score': integrated_scores['overall_score'],
                'rolling_score': integrated_scores['rolling_score'],
                'temporal_score': integrated_scores['temporal_score'],
                'sample_score': integrated_scores['sample_score'],
                'final_score': integrated_scores['final_score'],
                'stability_score': integrated_scores['stability_score'],
                'rating': integrated_scores['rating'],
            }
            score_row.update(metric_summary)
            component_weights = integrated_scores.get('component_weights') or {}
            reliability_scores = integrated_scores.get('reliability_scores') or {}
            reliability_labels = integrated_scores.get('reliability_labels') or {}
            for key in ['base', 'overall', 'rolling', 'temporal', 'sample']:
                if key in component_weights:
                    row[f'weight_{key}'] = component_weights[key]
                    score_row[f'weight_{key}'] = component_weights[key]
                if key in reliability_scores:
                    row[f'reliability_{key}'] = reliability_scores[key]
                    score_row[f'reliability_{key}'] = reliability_scores[key]
                if key in reliability_labels:
                    row[f'reliability_label_{key}'] = reliability_labels[key]
                    score_row[f'reliability_label_{key}'] = reliability_labels[key]
            row['weight_notes'] = integrated_scores.get('weight_notes')
            score_row['weight_notes'] = integrated_scores.get('weight_notes')
            score_rows.append(score_row)

            factor_result['integrated_scores'] = integrated_scores
            factor_result['auxiliary_stats'] = aux_entry
            factor_result['reliability'] = {
                'scores': reliability_scores,
                'labels': reliability_labels,
                'weights': component_weights,
                'notes': integrated_scores.get('weight_notes'),
            }
            aux_entry['reliability'] = factor_result['reliability']

            summary_rows.append(row)
            auxiliary_stats[factor] = aux_entry

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            self.latest_auxiliary_summary = summary_df
            print("[OK] 辅助分析摘要已计算（不再单独导出CSV）")
        
        self.auxiliary_stats = auxiliary_stats
        if score_rows:
            score_df = pd.DataFrame(score_rows)
            self.latest_auxiliary_scores = score_df
            print("[OK] 辅助稳健性评分摘要已计算（不再单独导出CSV）")

        if hasattr(self, 'analysis_results'):
            for factor, aux_entry in auxiliary_stats.items():
                base_result = self.analysis_results.get(factor)
                if not base_result:
                    continue
                integrated_scores = aux_entry.get('integrated_scores')
                if not integrated_scores:
                    integrated_scores = compute_integrated_factor_scores(base_result, aux_entry)
                self.analysis_results[factor]['integrated_scores'] = integrated_scores
                self.analysis_results[factor]['auxiliary_stats'] = aux_entry

        print("[OK] 辅助分析数据已缓存，可在报告和评分中使用")
        return auxiliary_stats
    
    def generate_summary_report(self):
        """
        - - - - - - - - 
        """
        return _fa_generate_summary_report(self)
    
    def run_filtered_factor_analysis(self, filter_conditions, use_pearson=None):
        """
        运行带参数的因子分析
        
        Args:
            filter_conditions: 过滤条件字典，格式为 {factor_name: (operator, value)}
                              例如：{"信号发出时上市天数": (">", 1200), "信号当日收盘涨跌幅": ("<", -19.9)}
                              支持的操作符：'>', '<', '>=', '<=', '==', '!='
            use_pearson: 是否使用Pearson相关系数计算IC值，默认为False（使用Spearman相关系数）
        
        Returns:
            bool: 分析是否成功
        """
    
    def optimize_factor_parameter(self, factor_name, operator, initial_value, optimize_metric='long_short_return', use_pearson=None):
        """
        优化因子参数，只保留10等分数据测试
        
        Args:
            factor_name: 要优化的因子名称
            operator: 操作符（实际已不再使用，保留兼容性）
            initial_value: 初始参数值（实际已不再使用，保留兼容性）
            optimize_metric: 优化指标
            use_pearson: 是否使用Pearson相关系数计算IC值（默认从配置读取）
        
        Returns:
            dict: 包含10等分测试结果的字典
        """
        if use_pearson is None:
            use_pearson = DEFAULT_USE_PEARSON
        if use_pearson is None:
            use_pearson = DEFAULT_USE_PEARSON
        group_count = DEFAULT_GROUP_COUNT

        # 创建日志文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = build_report_path(f"参数优化_{factor_name}_decile_test_{timestamp}.txt")
        
        # 日志函数
        def log(message):
            print(message)
            with open(log_filename, 'a', encoding='utf-8') as f:
                f.write(message + '\n')
        
        # 保存原始数据
        original_processed_data = self.processed_data.copy()
        
        # 检查因子是否存在且为数值型
        if factor_name not in self.processed_data.columns:
            error_msg = f"错误：因子 '{factor_name}' 在数据中不存在"
            log(error_msg)
            return None
        
        # 记录参数优化开始信息
        log("\n" + "="*80)
        log(f"参数优化开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log(f"优化因子: {factor_name}")
        log(f"优化指标: {optimize_metric}")
        log("="*80)
        
        # 开始等分数据测试
        log("\n" + "="*80)
        log(f"开始{group_count}等分数据测试")
        log("="*80)
        
        # 按因子值从小到大排序并平均分成指定份数
        temp_data = original_processed_data.copy()
        if not pd.api.types.is_numeric_dtype(temp_data[factor_name]):
            temp_data[factor_name] = pd.to_numeric(temp_data[factor_name], errors='coerce')
        
        # 去除NaN值
        temp_data = temp_data.dropna(subset=[factor_name])
        
        total_data_count = len(temp_data)
        log(f"数据总量: {total_data_count} 行")
        
        # 如果数据量足够，进行等分测试
        if total_data_count > 0:
            # 按因子值排序
            temp_data_sorted = temp_data.sort_values(by=factor_name)

            # 平均分组
            n_bins = group_count
            bin_size = len(temp_data_sorted) // n_bins
            decile_performances = []
            
            for i in range(n_bins):
                # 计算每个分位的数据范围
                start_idx = i * bin_size
                end_idx = len(temp_data_sorted) if i == n_bins - 1 else (i + 1) * bin_size
                
                # 获取该分位的数据
                decile_data = temp_data_sorted.iloc[start_idx:end_idx].copy()
                
                # 获取该分位的因子值范围
                decile_min = decile_data[factor_name].min()
                decile_max = decile_data[factor_name].max()
                
                log(f"\n=== 第 {i+1} 等分测试（共{len(decile_data)}行） ===")
                log(f"因子值范围: {decile_min:.3f} 到 {decile_max:.3f}")
                
                # 临时更新处理后的数据
                self.processed_data = decile_data
                
                # 计算性能指标
                try:
                    ic_mean, ic_std, _, _, _ = self.calculate_ic(factor_name, use_pearson=use_pearson)
                    ir = ic_mean / ic_std if not np.isnan(ic_std) and ic_std != 0 else 0.0
                except Exception as e:
                    log(f"  计算IC值时出错: {str(e)}")
                    ic_mean = 0.0
                    ir = 0.0
                
                try:
                    group_results = self.calculate_group_returns(factor_name)
                    long_short_return = group_results.get('long_short_return', 0.0) if group_results else 0.0
                    if np.isnan(long_short_return):
                        long_short_return = 0.0
                except Exception as e:
                    log(f"  计算分组收益时出错: {str(e)}")
                    long_short_return = 0.0
                
                # 计算胜率
                try:
                    win_rate = 0.0
                    if group_results and 'avg_returns' in group_results:
                        avg_returns = group_results['avg_returns']
                        if not avg_returns.empty:
                            positive_groups = (avg_returns['mean_return'] > 0).sum()
                            if len(avg_returns) > 0:
                                win_rate = (positive_groups / len(avg_returns)) * 100
                except Exception as e:
                    log(f"  计算胜率时出错: {str(e)}")
                    win_rate = 0.0
                
                # 存储分位性能
                decile_performances.append({
                    'decile': i + 1,
                    'factor_min': decile_min,
                    'factor_max': decile_max,
                    'data_count': len(decile_data),
                    'long_short_return': long_short_return,
                    'ir': ir,
                    'ic_mean': ic_mean,
                    'win_rate': win_rate
                })
                
                # 记录测试结果
                log(f"  多空收益: {long_short_return:.3f}%")
                log(f"  IR值: {ir:.3f}")
                log(f"  IC均值: {ic_mean:.3f}")
                log(f"  胜率: {win_rate:.1f}%")
            
            # 恢复原始数据
            self.processed_data = original_processed_data.copy()
            
            # 保存10等分测试结果到CSV
            if decile_performances:
                decile_df = pd.DataFrame(decile_performances)
                decile_df = decile_df.fillna(0)
                
                decile_csv_filename = build_report_path(f"十等分测试_{factor_name}_{timestamp}.csv")
                decile_df.to_csv(decile_csv_filename, index=False, encoding='utf-8-sig')
                
                # 找出表现最好的分位
                if optimize_metric == 'long_short_return':
                    best_decile = max(decile_performances, key=lambda x: x['long_short_return'])
                elif optimize_metric == 'ir':
                    best_decile = max(decile_performances, key=lambda x: x['ir'])
                else:
                    best_decile = max(decile_performances, key=lambda x: x['win_rate'])
                
                log(f"\n最佳表现分位: 第 {best_decile['decile']} 等分")
                log(f"因子值范围: {best_decile['factor_min']:.3f} 到 {best_decile['factor_max']:.3f}")
                log(f"多空收益: {best_decile['long_short_return']:.3f}%")
                
                # 存储初始结果
                results = {
                    'best_decile': best_decile,
                    'all_deciles': decile_performances,
                    'csv_file': decile_csv_filename
                }
                
                # 记录参数优化结束信息
                log("\n" + "="*80)
                log(f"参数优化结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                log(f"日志文件保存路径: {log_filename}")
                
                # 恢复原始数据
                self.processed_data = original_processed_data
                
                return results
            else:
                log("警告: 没有生成有效的10等分测试结果")
                return None
        else:
            log("警告: 因子值范围内没有足够的数据进行10等分测试")
            return None
    
    def _generate_filtered_summary_report(self, filtered_analysis_results, condition_str):
        """
        生成带条件的因子分析汇总报告
        
        Args:
            filtered_analysis_results: 过滤后的分析结果
            condition_str: 条件描述字符串
        """
        # 创建汇总表
        summary_data = []
        
        for factor, results in filtered_analysis_results.items():
            row = {
                '因子名称': factor,
                'IC均值': results['ic_mean'],
                'IC标准差': results['ic_std'],
                'IR值': results['ir'],
                't统计量': results['t_stat'],
                'p值': results['p_value']
            }
            
            if 'group_results' in results and results['group_results'] is not None:
                row['多空收益'] = results['group_results']['long_short_return']
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        
        print(f"\n=== 带条件的因子分析汇总报告 ({condition_str}) ===")
        print(summary_df.to_string(index=False, float_format='%.3f'))
        
        # 保存汇总报告，设置小数位数为3位
        # 使用条件字符串的简短版本作为文件名
        safe_condition_str = condition_str.replace(' ', '').replace('>', 'gt').replace('<', 'lt').replace('=', 'eq').replace('且', '_')
        # 添加时间戳到文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'因子分析汇总报告_{safe_condition_str}_{timestamp}.csv'
        summary_df_rounded = summary_df.round(3)
        report_path = build_report_path(filename)
        summary_df_rounded.to_csv(report_path, index=False, encoding='utf-8-sig')
        print(f"\n带条件的汇总报告已保存到 '{report_path}'")
        
        return summary_df
    
    def run_full_analysis(self):
        """
        运行完整的因子分析流程
        """
        if not self.run_factor_analysis():
            return False
        
        print("\n=== 开始可视化分析结果 ===")
        
        # 绘制因子分布图（只有在可视化功能可用时）
        if HAS_PLOT:
            self.plot_factor_distribution()
        
        # 绘制分组收益图（只有在可视化功能可用时）
        if HAS_PLOT:
            self.plot_group_returns()
        
        # 生成汇总报告
        self.generate_summary_report()
        
        print("\n=== 因子分析完成 ===")
        return True




# 删除重复的main函数定义，只保留末尾的完整版本
