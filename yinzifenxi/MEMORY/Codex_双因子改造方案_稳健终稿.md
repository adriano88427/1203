# Codex · 双因子分析改造方案（稳健终稿）

> 目标：保持既有单因子行为零改动的前提下，引入“非参数双因子 + 带参数双因子”功能。方案沿用 Codex 版本的完整结构设计，并吸收 Claude 保守方案的优点（渐进式开关、性能限制、最小侵入式调用与独立容错），以获得在需求覆盖与实施风险之间更稳健的折衷。

---

## 1. 总体设计原则

1. **功能完整性**：满足“非参数 + 带参数”双因子分析、报告格式与单因子保持一致、符合 A 股主流实践等原始需求。
2. **最小侵入性**：仅在 `fa_config.py` 与 `yinzifenxi_main.py` 做可控改动；所有新逻辑置于独立模块（`fa_dual_*`），减少对已有代码的影响。
3. **渐进式开关**：吸收 Claude 方案的“开关控制”思想，将双因子开关、性能参数集中在配置中，并提供环境变量覆盖能力，便于灰度启用。
4. **性能/安全兜底**：借鉴 Claude 方案的保守限制，控制最大组合数量、最小样本量、网格规模等，确保新功能不会拖慢主流程。
5. **一致的数据口径**：与 Codex 方案一样复用 `FactorAnalysis`/`ParameterizedFactorAnalyzer` 的预处理结果，避免 Claude 方案那样“直接读原始 Excel”导致口径不一致。
6. **独立报告与容错**：双因子分析失败不影响单因子流程；输出到独立文件，报告风格与单因子一致但互不覆盖；并提供降级提示。

---

## 2. 配置层方案（`fa_config.py`）

### 2.1 分析开关与性能参数
新增配置块，支持默认值 + 环境变量覆盖：
```python
ANALYSIS_SWITCHES = {
    "single_nonparam": True,
    "single_param": True,
    "dual_nonparam": os.getenv("FA_DUAL_NONPARAM", "false").lower() == "true",
    "dual_param": os.getenv("FA_DUAL_PARAM", "false").lower() == "true",
}
```

```python
DUAL_FACTOR_SETTINGS = {
    "nonparam_bins": int(os.getenv("FA_DUAL_NONPARAM_BINS", "5")),  # 默认5×5
    "nonparam_top_n": int(os.getenv("FA_DUAL_NONPARAM_TOP_N", "6")),  # 取单因子Top N组合
    "max_factor_pairs": int(os.getenv("FA_DUAL_MAX_PAIRS", "30")),  # 确保性能可控
    "min_samples": int(os.getenv("FA_DUAL_MIN_SAMPLES", "800")),
    "param_factor_pairs": [],
    "param_ranges": {},
    "param_min_samples": int(os.getenv("FA_DUAL_PARAM_MIN_SAMPLES", "300")), 
}
```

### 2.2 报告/路径默认值
```python
DUAL_REPORT_OPTIONS = {
    "nonparam_prefix": "双因子非参数",
    "param_prefix": "双因子带参数",
    "output_dir": os.getenv("FA_DUAL_OUTPUT_DIR", "baogao/dual_factors"),
    "heatmap_enabled": True,
    "max_rank_display": 10,
}
```

### 2.3 校验函数
提供 `validate_dual_config()`：
- 检查 bins/top_n/max_pairs 是否在合理范围（例如 bins ∈ [3,10]，max_pairs ≤ 100）。
- 若配置非法，打印 `[WARN]` 并回落到默认值，避免启动失败。

---

## 3. 主流程挂钩（`yinzifenxi_main.py`）

### 3.1 善用 existing `logger`
- 保持单因子流程逻辑完全不变。
- 在主流程完成后，以“可控插桩”的形式调用双因子流程：
  ```python
  if ANALYSIS_SWITCHES.get("dual_nonparam"):
      try:
          run_dual_nonparam_analysis(analyzer, logger)
      except Exception as exc:
          print(f"[ERROR] 双因子非参数分析失败: {exc}")
  ```
- 同样的模式用于带参数双因子。

### 3.2 函数式入口 + 容错
- `run_dual_nonparam_analysis` 、`run_dual_param_analysis` 分别放在新模块里导出，小函数逻辑仅负责校验开关、传递配置、打印日志。
- 即使双因子模块出现异常也不会中断单因子流程，符合 Claude 方案“独立运行”优点。

### 3.3 数据复用与限制
- 非参数路径直接复用 `FactorAnalysis` 的 `processed_data`、`analysis_results`、`factors`、`return_col`，避免重新读取文件。
- 带参数路径复用 `ParameterizedFactorAnalyzer.processed_data`（若不存在则重新构建）。
- 在调用前检查样本量/因子数量，如果不满足 `min_samples` 等限制则打印 `[WARN]` 并跳过，参考 Claude 的保守策略。

---

## 4. 非参数双因子分析 (`fa_dual_nonparam_analysis.py`)

### 4.1 核心职责
1. 接收 `FactorAnalysis` 对象 + `DUAL_FACTOR_SETTINGS`。
2. 根据配置生成因子对：优先显式配置，缺省时取单因子表现最好的 `nonparam_top_n` 个因子两两组合，并限制总数不超过 `max_factor_pairs`。
3. 对每对因子执行：
   - 根据 `nonparam_bins` 做二维分位分箱（默认 5×5），支持 winsorize/标准化与单因子保持一致。
   - 计算每个格子的平均收益、胜率、样本量、最大回撤、信息比等；同时计算协同 IC（如 2D 排序 Spearman 或等权组合 vs 收益的相关性）以及 p 值/IR。
   - 输出 `grid_stats`（DataFrame）和 `summary_metrics`（dict）。
4. 存储结果结构：
   ```python
   {
     ("因子A","因子B"): {
        "summary": {...},
        "grid": pd.DataFrame,
        "heatmap": np.ndarray 或 DataFrame,
        "warnings": [...]  # 例如样本不足
     },
     ...
   }
   ```

### 4.2 性能/质量控制
- 每个因子对最少样本 `min_samples`，不足则跳过。
- 超过 `max_factor_pairs` 自动截断，日志提示。
- 计算完成后根据 `synergy_score`、`combined_ic` 排序，便于报告展示。

### 4.3 可选增强
- 若需要吸纳 Claude 的“快速粗筛”思路，可先计算简化 IC（线性叠加）进行预筛，再对排名前 N 的因子对做精细二维分析，从而兼顾速度与准确性。

---

## 5. 非参数双因子报告 (`fa_dual_nonparam_report.py`)

### 5.1 输出内容
1. **CSV 汇总**：列出因子对、协同 IC、IR、最佳/最差组合收益、样本数、显著性、备注等 `MEMORY/Codex_双因子改造方案.md:79-94`。
2. **HTML 报告**：沿用 `HTMLReportBuilder`，包含：
   - 核心指标卡（分析对数、平均协同 IC、平均多空收益）。
   - 因子对排行榜（前 N 名，显示协同 IC、组合收益等）。
   - 典型热力图（可插入 Base64 图或 HTML 表格）。
   - 策略建议/风险提示（参考单因子模板，文案针对双因子场景调整）。

### 5.2 独立目录 + 命名
- 输出目录读取 `DUAL_REPORT_OPTIONS["output_dir"]`，默认 `baogao/dual_factors`。
- 文件命名如 `双因子非参数汇总_YYYYMMDD_HHMM.csv`、`双因子非参数分析_YYYYMMDD_HHMM.html`。

### 5.3 异常与降级
- 若无有效因子对，则 CSV/HTML 中输出提示段落，且在主流程日志打印 `[WARN]`，借鉴 Claude 方案的恢复机制。

---

## 6. 带参数双因子分析 (`fa_dual_param_analysis.py`)

### 6.1 核心职责
1. 接收 `ParameterizedFactorAnalyzer` 或其 `processed_data`。
2. 根据配置 `param_factor_pairs`、`param_ranges` 生成组合；若配置为空可默认使用单因子得分前 N 的因子及其默认区间。
3. 对每个因子对的区间网格计算：
   - 日/年化收益、波动、夏普、索提诺、胜率、最大回撤、交易日数等，与单因子带参保持一致。
   - 协同增益指标（双因子收益相对单因子 max 的提升、组合协方差等）。
4. 结果结构与单因子带参一样：
   ```python
   {
     ("因子A","因子B"): {
        "grid_stats": DataFrame,  # 行列为区间组合
        "summary": {...}
     },
     ...
   }
   ```

### 6.2 复用 & 限制
- 抽象/复用单因子带参中的 `_normalize_dataframe_columns`、`_assign_user_forced_groups`、年化计算等逻辑，确保口径一致。
- 最少样本数 `param_min_samples`，不足则跳过该区间组合并记录 warning。
- 限制每个因子对/区间数量以控制计算量。

### 6.3 兼容保守策略
- 在配置中允许 `FA_DUAL_PARAM_MAX_PAIRS`、`FA_DUAL_PARAM_MAX_GROUPS` 等环境变量覆盖，借鉴 Claude 方案的性能防护思想。

---

## 7. 带参数双因子报告 (`fa_dual_param_report.py`)

### 7.1 输出内容
- **HTML**：概览卡片（平均年化收益/最大回撤）、正向/负向双因子排行榜（新增“协同增益”列）、详情卡片（每个因子对区间矩阵 + 指标列表 + 建议/提示）。
- **Excel**：数据明细与高亮（可复用 `fa_param_report.py` 中的高亮函数，或新增 `_apply_dual_highlight`）。
- 文件命名参照非参数双因子格式，换成 `双因子带参数分析_时间戳.html` 等。

### 7.2 失败降级
- 若无有效数据，HTML 提示“暂无可用的双因子带参数结果”；主流程打印 `[WARN]`。

---

## 8. 共享工具与守护机制

1. **工具复用**：继续调用 `fa_stat_utils`（`custom_spearman_corr`, `calc_max_drawdown`, `safe_calculate_annual_return` 等）；新增通用函数可以集中在 `fa_dual_utils.py`，但避免直接修改现有 util，减少回归风险。
2. **性能/内存控制**：吸收 Claude 保守策略，配置中限制 `max_factor_pairs`、`bins`、`min_samples`、`max_memory_mb` 等，运行时监控统计并输出日志。
3. **异常处理**：每个分析器内部 try/except；单个因子对失败只记 warning，不影响其他任务；整体失败也不阻断主流程。

---

## 9. 验证计划

1. 关闭所有双因子开关：确认输出与改造前完全一致。
2. 仅开启非参数双因子：生成 CSV/HTML，检查指标正确性、网格展示、性能日志。
3. 仅开启带参数双因子：验证区间指标、Excel 高亮、协同增益指标。
4. 同时开启双路径：确认各自成果独立输出且互不干扰。
5. 调整环境变量（如 `FA_DUAL_MAX_PAIRS=10`）观察是否生效。

---

## 10. 文件/模块清单

```
MEMORY/
  ├─ Codex_双因子改造方案.md           # 初版
  └─ Codex_双因子改造方案_稳健终稿.md  # 本文档
yinzifenxi_main.py                    # 添加双因子入口及容错封装
fa_config.py                          # 新增开关 + 配置 + 校验
fa_dual_nonparam_analysis.py          # 新增：非参数双因子分析器
fa_dual_nonparam_report.py            # 新增：非参数双因子报告
fa_dual_param_analysis.py             # 新增：带参数双因子分析器
fa_dual_param_report.py               # 新增：带参数双因子报告
fa_dual_utils.py（可选）               # 复用辅助函数（若有共性逻辑）
```

---

## 11. 风险评估与缓解

| 风险点 | 描述 | 缓解措施 |
| --- | --- | --- |
| 功能未满足 | 若双因子实现不完整会偏离需求 | 本方案明确包含非参 + 带参 + 报告 + 配置开关全链路 |
| 影响主流程 | 新逻辑导致单因子结果异常 | 仅在主流程尾部调用，且 try/except；开关默认关闭 |
| 性能压力 | 因子对过多、网格过大 | `max_factor_pairs`、`nonparam_bins` 等限制 + 日志提醒 |
| 维护成本 | 模块新增较多 | 通过独立文件、明确职责、代码注释控制复杂度 |
| 数据不一致 | 不复用已有数据会导致口径差异 | 复用 `FactorAnalysis/ParameterizedFactorAnalyzer` 的预处理结果 |

---

**结论**：在 Codex 方案全面满足需求的基础上引入 Claude 方案的“渐进开关 + 性能限制 + 独立容错”优点，可在保证功能完整性的同时把握实施风险。建议按照本终稿执行，并在编码阶段紧贴上述模块划分与配置约定，确保交付可回滚、可扩展、可维护的双因子分析能力。\
