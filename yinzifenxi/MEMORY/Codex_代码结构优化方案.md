# Codex · 双因子扩展后代码结构优化方案

> 目的：在现有双因子功能落地的基础上，进一步减少重复、降低耦合、提升可维护性。以下建议按“配置 → 主流程 → 分析模块 → 报告模块 → 共享工具 → 验证策略”顺序给出，方便分阶段实施。

---

## 1. 配置层（`fa_config.py`）

1. **配置拆分**  
   - 现有 `fa_config.py` 同时承担“基础路径、单因子参数、双因子配置”等职责，文件较长。建议将双因子相关配置独立到 `fa_dual_config.py` 或在末尾采用更明显的分割注释，便于快速定位。

2. **只读/深拷贝约束**  
   - `DUAL_FACTOR_SETTINGS` 等默认值含可变对象（列表、字典）。即便 `validate_dual_config()` 返回深拷贝，外部仍可能直接引用全局变量。建议：
     - 在模块导出时仅提供 `get_dual_config()`（返回深拷贝），不暴露原始字典；或
     - 将默认值声明为 `MappingProxyType`，从源头保证只读。

3. **Report Options 校验**  
   - 新增 `validate_dual_report_options()`，统一检查 `output_dir` 是否存在、`max_rank_display` 是否为正、`heatmap_enabled` 是否布尔值，避免报告模块中重复判断。

4. **环境变量映射集中化**  
   - 当前 `ANALYSIS_SWITCHES` 同时读取默认值与环境变量。可考虑提供 `load_switches_from_env()`，将“读取→转换→校验”逻辑集中，主流程仅处理返回值。

---

## 2. 主流程（`yinzifenxi_main.py`）

1. **函数拆解/复用**  
   - `main()` 仍包含大量打印/循环，建议将“单因子渲染”“参数化分析”提炼为单独函数，方便独立测试，也便于在 CLI 之外调用。

2. **logger 使用**  
   - `_run_dual_*_workflow` 接收 `logger` 却未调用；如果只需标准输出，可去掉该参数；如需记录到文件，则内部应改用 `logger.write()` 保持日志来源一致。

3. **配置传递**  
   - 每次调用 `validate_dual_config()` 都返回新副本。若后续要同时传递给多个模块，可考虑在 `main()` 里创建一次 `dual_config` & `dual_report_options` 副本，并保证只读，防止模块间意外修改。

4. **参数化分析重用**  
   - `parameterized_analyzer = ParameterizedFactorAnalyzer(analyzer.data.copy())` 与 `run_dual_param_pipeline()` 使用同一数据，可将其构建逻辑抽象为函数，以避免出错时重复复制大 DataFrame。

---

## 3. 非参数双因子模块（`fa_dual_nonparam_analysis.py`）

1. **Pipeline 与分析解耦**  
   - `run_dual_nonparam_pipeline` 同时负责数据筛选、预筛、分析与报告调用。建议拆成：`select_pairs`、`evaluate_pairs`、`create_report_payload`，由主流程决定是否生成报告，以降低耦合。

2. **预筛改进**  
   - `_prescreen_factor_pairs` 目前采用 Pearson 相关，可考虑复用 `custom_spearman_corr` 或引入更轻量的 rank-based 评分，保持口径一致。预筛可输出 top N 评分以便调试。

3. **日志与进度**  
   - 预筛、分析阶段都在 `print`，建议引入可注入式 logger（默认 `print`），便于单元测试捕获输出。

4. **复用辅助函数**  
   - `_analyze_factor_pair` 与带参数版有多处相似逻辑（过滤、收益统计、协同指标）。可新增 `fa_dual_utils.py` 内含 `prepare_subset(df, pair)`、`compute_combined_ic()` 等工具，减少复制。

---

## 4. 带参数双因子模块（`fa_dual_param_analysis.py`）

1. **区间生成策略**  
   - `_resolve_ranges` 目前仅支持等距分位；可根据配置增加 “自定义列表 + 百分位语法（如 top10%）+ 默认分位” 三种策略，以更贴近业务。

2. **输出数据解耦**  
   - 与非参数模块类似，分析函数可只返回结构化数据，不直接调用 `generate_dual_param_reports`，便于在未来新增 API 或不同报告形式。

3. **重复导入清理**  
   - 当前导入了 `os`、`datetime` 等未使用模块，可清理以降低 lint 噪音。

4. **性能统计**  
   - 可以在 `run_dual_param_pipeline` 中增添简单的统计（耗时、成功/失败组合数），便于监控大数据量下的运行状况。

---

## 5. 报告模块（`fa_dual_nonparam_report.py` & `fa_dual_param_report.py`）

1. **共用模板**  
   - 两个报告生成流程十分相似，可抽象公共函数：`write_dual_csv(summary_df, prefix)`、`build_dual_html(title, cards, sections)`，减少重复，保持风格同步。

2. **报告配置验证**  
   - 在写文件前可调用 `validate_dual_report_options()`，若配置不合法（如路径不可写）应提前报错而非静默失败。

3. **Excel 生成**  
   - `generate_dual_param_reports` 中的 `_write_excel` 建议用 `with pd.ExcelWriter(...) as writer:`，并在多 sheet 写入时对 sheet 名进行重名处理以防异常。

4. **热力图/详情定制**  
   - 可预留钩子（例如 `report_options["extra_sections"]`）以便后续扩展，不必修改底层组件。

---

## 6. 共享工具 / 目录结构优化

1. **新增 `fa_dual_utils.py`（建议）**  
   - 集中存放共享函数（区间解析、协同评分、日志适配器等），供分析/报告模块调用，进一步降低耦合。

2. **目录层次**  
   - 将 `fa_dual_*` 文件放入子包（如 `dual/analysis.py`, `dual/report.py`）有助于组织结构，也避免顶层 `yinzifenxi` 目录愈发膨胀。

3. **减少重复导入**  
   - 当前多个模块重复导入 `datetime`、`os` 等大多数未使用，可以通过工具模块统一输出时间戳/路径生成函数。

---

## 7. 验证策略

1. **测试矩阵**  
   - 在自动化或手工测试中覆盖以下场景：
     - 双因子开关关闭：确认单因子结果不变。
     - 仅非参数开启 / 仅带参数开启。
     - 自定义因子对 + 自定义区间。
     - 数据不足时的降级路径（warnings/log）。

2. **性能监控**  
   - 在预筛/分析环节增加耗时和因子对数量日志；对大数据时留意 CPU/内存。

3. **报告一致性**  
   - 对比双因子报告与单因子报告风格，保证 UI 一致；必要时可引入 UI 自动化或 diff 工具。

---

**小结**：当前双因子功能已满足主要需求，但通过拆分配置、解耦分析与报告、复用辅助函数以及增强日志/验证，可以让整体结构更清晰、模块间更独立，也为后续新增功能或测试打下基础。建议按模块优先级逐步实施以上优化措施。*** End Patch
