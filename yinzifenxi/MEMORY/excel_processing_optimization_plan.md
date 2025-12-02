# Excel 数据处理代码优化方案（第二阶段）

> 目标：继续提升 Excel 数据解析/预处理的一致性，消除冗余逻辑与潜在矛盾，确保 `FactorNormalizer` 成为单一事实来源。  
> 适用文件：`fa_nonparam_analysis.py`、`fa_param_analysis.py`、`fa_param_report.py`、`fa_data_validator.py`、`excel_parser.py` 等。

## 1. 冗余转换与缩放清理

### 1.1 `fa_nonparam_analysis.py:722-808`
- **问题**：在 `FactorNormalizer` 已对 `active_factors + return_col` 完成语义识别与缩放后，仍保留旧逻辑（字符串 → 数值 → `/100`）。  
- **风险**：重复打印日志、再次覆盖 `Normalizer` 的结果，若列已被转换为 float 再 `astype(str)` 可能丢失信息。  
- **优化**：
  1. 删除 `columns_to_normalize` 循环中旧的字符串转换段，仅在 `FactorNormalizer` 失败（返回 `info.notes` 含错误）时才执行回退逻辑。
  2. 对收益率列 (`self.return_col`) 也使用 `Normalizer`，若 `info.applied_scale` 为 `None` 则再尝试一次回退。

### 1.2 `fa_param_analysis.py:154-203`
- **问题**：完全同上；`FactorNormalizer` + 旧逻辑并存，重复工作。  
- **优化**：与非参数分析保持一致，保留 `Normalizer`，回退逻辑封装成 `self._coerce_numeric(col, df)`，仅在 `Normalizer` 失败时调用。

## 2. normalization_stats 的使用与输出

### 2.1 `fa_nonparam_analysis`
- **问题**：`self.normalization_stats` 记录了信息，但后续没有使用。  
- **方案**：
  - 在日志中输出“数据标准化说明”，与参数化报告一致。
  - 将每个因子的 `NormalizationInfo` 注入 `analysis_results[factor]['normalization']`，供 `fa_nonparam_report`/报告 CSV 使用。

### 2.2 `fa_param_report`
- 已新增“数据标准化说明”板块；同样的组件可复用到非参数报告（`fa_nonparam_report.py`），保持一致体验。

## 3. DataValidator 语义对齐

- 目前 `_analyze_column_consistency()` 仍依赖 `COLUMN_ALIGNMENT_RULES` 手工配置的 `abs_max/scale_candidates`，与 `FactorNormalizer` 的语义识别脱节。
- **改进**：
  1. 在 `DataValidator` 初始化时创建 `FactorNormalizer`，利用 `semantic`/`scale_hint` 来判断不同年份是否存在单位冲突；
  2. 若某列按语义应该是 percent，却检测到年度 median > 2，则自动提示 `[WARN] 列 xxx 年度 y 可能未缩放`；
  3. 将当前的 `scale_candidates` 逻辑退为回退路径（当 `FactorNormalizer` 无法判断时才使用）。

## 4. ExcelParser 与 Normalizer 的协同

- `FactorNormalizer` 现定义在 `excel_parser.py` 内，但 `ExcelParser` 本身并未使用它。  
- **建议**：
  1. 在 `ParseDiagnostics` 中记录 `raw_semantic`/`applied_scale` 等信息（由 `Normalizer` 输出）。  
  2. 将 `ValueConverter` 输出的 DataFrame 再交给 `Normalizer` 做一次语义识别，这样主流程在 `load_excel_sources` 后即可知道哪些列已缩放、哪些需要关注。  
  3. 若 `Normalizer` 在解析阶段已经完成缩放，可在 `ParsedData` 中附带 `normalization_info`，方便不同分析器复用。

## 5. 日志与告警统一

- `[INFO] 列 'X' 自动缩放`、`已自动尝试将列 'X' 转换为数值类型` 等日志风格不一致。  
- **优化**：
  - 新增一个 `NormalizationLogger`（或在 `FactorNormalizer` 内封装 `log_action()`），统一输出格式，如 `[NORMALIZE] Column=X semantic=percent scale=0.01 reason=detected_percent_pattern`。
  - 替换 `print(f"已自动尝试…")` 这类文案。

## 6. 测试与回归计划

1. **单元测试**：针对 `FactorNormalizer` 的 `normalize()` / `format_range()` 建立覆盖，确认删除旧逻辑后仍能处理字符串百分比、纯数值百分比等场景。
2. **集成测试**：运行非参数与带参数完整流程，验证：
   - 日志中只出现一次“自动缩放”提示；
   - `带参数因子分析数据_*.xlsx` 及非参数报告的“参数区间”均保持一致单位；
   - `DataValidator` 的列对齐检查会输出新的语义警告（或确认通过）。
3. **回归**：对旧版（未优化）和新版输出进行 diff，重点检查 `当日回调`、`机构持股比例(%)` 等关键字段的区间与统计是否一致。

## 7. 实施顺序建议

1. **清理冗余转换** （fa_nonparam_analysis / fa_param_analysis），确保 `FactorNormalizer` 成为唯一入口；
2. **输出 normalization_stats** 到报告层，复用“数据标准化说明”；
3. **增强 DataValidator**（引入 `FactorNormalizer`）；
4. **ExcelParser 协同**（可视情况安排到下一阶段）；
5. **统一日志样式** + 补充测试。

此方案聚焦于解决现阶段的“重复逻辑/信息未使用/验证分散”问题，为后续继续深化 Excel 语义识别和异常纠偏打下基础。
