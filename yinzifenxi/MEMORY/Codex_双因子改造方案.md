# Codex · 双因子分析改造方案

> 目标：在不改动既有单因子分析行为的前提下，引入「非参数双因子」与「带参数双因子」分析及其独立报告，并通过配置开关灵活控制执行范围。

---

## 1. 配置层改造 (`fa_config.py`)

1. **分析开关**
   - 新增 `ANALYSIS_SWITCHES` 字典（示例：`{"single_nonparam": True, "single_param": True, "dual_nonparam": False, "dual_param": False}`）。
   - `yinzifenxi_main.py` 在运行前读取开关，决定是否执行子流程。

2. **双因子默认配置**
   - `DUAL_FACTOR_SETTINGS = { "nonparam_factor_pairs": [], "nonparam_bins": 5, "nonparam_top_n": 5, "param_factor_pairs": [], "param_ranges": {}, "param_min_samples": 50 }`。
   - 支持 “显式列出因子对” 或 “自动取单因子表现最好的前 N 个因子做组合”。
   - `param_ranges` 允许：`{ "因子A": [(0,0.1),(0.1,0.2)], "因子B": ["top10%","bottom30%"] }`。

3. **报告默认选项**
   - `DUAL_REPORT_OPTIONS` 指定标题、输出文件名前缀、热力图/排行榜默认参数，供新报告模块复用。

4. **轻量校验函数**
   - `validate_dual_config()`：运行时检查配置合法性（因子存在、区间格式正确）。不通过时打印 `[WARN]` 并跳过执行。

---

## 2. 主流程挂钩 (`yinzifenxi_main.py`)

1. **现有单因子流程保持原状**
   - `FactorAnalysis` 与 `ParameterizedFactorAnalyzer` 逻辑及报告调用不变。

2. **双因子入口**
   - 在单因子分析完成后，根据开关顺序执行：
     1. `DualFactorAnalysis`（非参数）
     2. `DualParameterizedAnalyzer`（带参数）
   - 每个阶段独立 `try/except`，出现异常只影响本阶段。

3. **数据复用**
   - 非参数双因子直接使用 `FactorAnalysis` 的 `processed_data` / `analysis_results`。
   - 带参数双因子使用 `ParameterizedFactorAnalyzer` 的 `processed_data`，避免重复IO。

4. **CLI/日志**
   - 启动时打印当前开关状态；如全部关闭，仅提示“未启用任何分析”。
   - 新步骤遵循 `[INFO]/[WARN]/[ERROR]` 格式写入日志文件。

---

## 3. 非参数双因子分析 (`fa_dual_nonparam_analysis.py`)

1. **核心类：`DualFactorAnalysis`**
   - 初始化接收 `FactorAnalysis` 实例与配置。
   - 生成因子对：优先使用配置列表，否则根据 `analysis_results` 取 TopN 两两组合。
   - 对每个因子对执行：
     - Winsorize/标准化同单因子设置。
     - 分别对两个因子做等分排序（默认 5×5 网格，可配置）。
     - 计算每个网格的 `平均收益 / 胜率 / 样本数 / 最大回撤`。
     - 计算协同 IC（如 2D rank IC 或 双变量回归系数）及显著性、IR、长短组合收益。
     - 输出 `grid_stats`（DataFrame）与 `summary_metrics`（dict）。

2. **辅助函数**
   - `generate_factor_pairs()`：处理显式/自动组合逻辑。
   - `compute_dual_grid_metrics()`：实现二维分桶与统计。
   - `format_dual_summary()`：归整为报告所需结构。

3. **结果结构**
   ```python
   {
       ("因子A","因子B"): {
           "summary": {...},
           "grid": pd.DataFrame,
           "heatmap": ndarray/df,
           "warnings": [...]
       },
       ...
   }
   ```

---

## 4. 非参数双因子报告 (`fa_dual_nonparam_report.py`)

1. **CSV 汇总**
   - 列：`因子A/因子B/协同IC/IR/最佳组合收益/最差组合收益/样本量/显著性/备注`。

2. **HTML 报告**
   - 沿用 `HTMLReportBuilder`：
     - 核心指标卡：分析对数、平均协同 IC、平均多空收益等。
     - 因子对排行榜：按 `协同IC` 或 `long-short` 排序，展示前 N 名。
     - 典型网格热力图：可输出为表格或图片（若图片，使用 `matplotlib` 保存为 base64）。
     - 策略建议与风险提示引用单因子模板，文案针对双因子场景调整。

3. **文件命名**
   - 例如：`双因子非参数汇总_YYYYMMDD.csv`、`双因子非参数分析_YYYYMMDD.html`，路径使用 `build_report_path`。

---

## 5. 带参数双因子分析 (`fa_dual_param_analysis.py`)

1. **核心类：`DualParameterizedAnalyzer`**
   - 输入 `ParameterizedFactorAnalyzer` 或其 `processed_data`。
   - 读取配置的因子对与区间，支持：
     - 静态区间 (`[(0,0.1), (0.1,0.3)]`)
     - 百分位语法（`"top10%"`）→ 运行时换算实际阈值。
   - 对每个组合网格计算：
     - 日/年化收益、波动、胜率、最大回撤、样本量、交易日数。
     - 协同增益指标（双因子收益 vs 单因子最优收益、组合协方差等）。
   - 结果结构仿照单因子带参：`factor_pair -> {"grid_stats": df, "summary": dict}`。

2. **复用逻辑**
   - 从单因子带参中抽象共通函数（如 `_normalize_dataframe_columns`, `_assign_user_forced_groups`, 年化计算），避免重复实现，保证指标一致。

3. **输出接口**
   - `generate_dual_parameterized_report()` 返回 HTML 文件路径。
   - 同时生成 Excel（含高亮），调用 `fa_dual_param_report` 中的写入函数。

---

## 6. 带参数双因子报告 (`fa_dual_param_report.py`)

1. **概览**
   - 展示双因子对数量、平均年化收益、平均最大回撤等卡片。

2. **排行榜**
   - 正向/负向双因子区间榜（参考单因子带参 Top5 结构），新增“协同增益”列。

3. **详情卡片**
   - 每个因子对展示：
     - 区间矩阵（表格/热力图）；
     - 指标列表（年化收益/波动/夏普/胜率/最大回撤/观测期）；
     - 文本化建议/样本提示。

4. **Excel 输出**
   - 写入 `带参数双因子分析数据_时间戳.xlsx`。
   - 可沿用 `_apply_param_csv_highlight`，或新增 `_apply_dual_highlight` 高亮多列。

---

## 7. 共有工具与异常处理

1. **工具复用**
   - 继续使用 `fa_stat_utils` 中的 `custom_spearman_corr`, `calc_max_drawdown`, `safe_calculate_annual_return`，必要时新增 2D 分位辅助函数。
   - 若某些函数被单/双因子共享（如分组排序），考虑提取到 `fa_stat_utils` 或新建 `fa_dual_utils`。

2. **异常与日志**
   - 每个分析器内部捕捉配置异常（缺因子/区间）→ `[WARN]` 并跳过该组合。
   - 严重错误（计算失败）→ `[ERROR]`，但不影响其他组合。

---

## 8. 验证与回归建议

1. **功能验证**
   - 默认只跑单因子 → 确认输出与改造前一致。
   - 开启非参数双因子 → 检查 CSV/HTML 生成、数据合理性。
   - 配置特定因子对及区间 → 验证带参双因子结果、Excel 高亮。
   - 关闭全部开关 → 程序应只打印提示并退出。

2. **文档/测试记录**
   - 在 `MEMORY` 中记录测试结论或脚本使用说明，以便后续维护。

---

## 9. 文件结构预览

```
MEMORY/
  └─ Codex_双因子改造方案.md   # 本文档
yinzifenxi_main.py             # 新增开关控制&双因子入口
fa_config.py                   # 配置开关+双因子参数
fa_dual_nonparam_analysis.py   # 新增：非参数双因子分析
fa_dual_nonparam_report.py     # 新增：非参数双因子报告
fa_dual_param_analysis.py      # 新增：带参数双因子分析
fa_dual_param_report.py        # 新增：带参数双因子报告
...（其余单因子文件保持不变）
```

---

**结论**：上述步骤在保留既有单因子行为的同时，引入配置化的双因子分析与报告生成流程，满足“非参数 + 带参数”、A股主流方法、配置开关、报告独立等全部需求。
