---
type: nav
title: 时间线 (Log)
updated: 2026-06-22
---


# 时间线 (Log)

追加式记录。每条以 `## [日期] 操作 | 标题` 开头，方便 `grep "^## \[" log.md | tail` 取最近记录。

## [2026-06-17] init | 搭建 wiki 框架
按 Karpathy LLM Wiki 方法论建立三层结构与目录约定（见 [[CLAUDE]]）。创建 `concepts/`、`components/`、`sources/` 三个目录与导航层 `README` / `index` / `log` / `glossary`。

## [2026-06-17] ingest | 仓库代码（src/sol_execbench）
读取并分析核心源码：`sol_score.py`、`core/bench/{reward_hack,correctness,timing,clock_lock,io}.py`、`driver/{problem_packager.py, templates/eval_driver.py}`。沉淀为 [[components/architecture-and-flow]]、[[components/evaluation-driver]]、[[concepts/timing-and-reproducibility]]、[[concepts/correctness-protocol]]、[[concepts/reward-hacking-defenses]]、[[concepts/sol-score]]。

## [2026-06-17] ingest | docs/ 四个 schema 文档
读取 `docs/{definition,workload,solution,trace}.md`。沉淀为 [[components/data-models]]，并回填 [[sources/code-and-docs-map]]。

## [2026-06-17] ingest | 技术报告 arXiv:2603.19173
逐节读取 HTML 版（摘要 / §1 引言 / §2 相关工作 / §3 构造 / §4 数据集与评测 / §5 实验 / §6 结论）。沉淀为 [[sources/tech-report]]，并支撑 [[concepts/design-philosophy]]、[[concepts/speed-of-light-and-solar]]、[[concepts/sol-score]]、[[concepts/benchmark-construction]]、[[concepts/reward-hacking-defenses]] 的"为什么"论证。

## [2026-06-17] lint | 首版一致性检查
核对 SOL Score 公式（代码 `sol_score.py` ↔ 报告 §4.3 Eq.2）、reward-hack 防御表（代码 `reward_hack.py` ↔ 报告表 3）、计时参数（代码默认 vs 报告 §4.4 描述）三处一致性，已在相应页面标注代码与报告的差异点。

## [2026-06-17] query | SOL-ExecBench 与 SOLAR 是否强依赖
用户提问：本仓库是否强依赖 NVlabs/SOLAR。验证 `pyproject.toml`、全仓 `import solar` grep、`sol_score.py` 唯一使用点——结论：代码完全解耦，仅方法论强绑定（SOLAR 离线产出 `T_SOL` 烘焙进数据集，SOL Score 把它当纯 float 消费）。回填到 [[concepts/speed-of-light-and-solar]] §6 "与 SOL-ExecBench 的耦合关系"。

## [2026-06-22] query | benchmark 算子用什么 DSL 写 + cuTile 算子清单
用户两问：(1) 被测 kernel 支持哪些 DSL；(2) cuTile 写的算子有哪些。核对 `core/data/solution.py` 的 `SupportedLanguages` 枚举与全仓 `examples/**/solution*.json`。结论：solution 支持 7 类 DSL（pytorch/triton/cutlass/cudnn/cute_dsl/cutile/cuda_cpp），reference 始终 PyTorch；cuTile 当前仅 1 个样例算子 `attn_proj_residual_cutile`（Jamba 注意力输出投影+残差，与 `cute_dsl` 同算子双实现）。附带发现：11 个样例里 10 个 `DPS=false`，与 schema 默认 `true` 相反。回填到 [[components/data-models]] §3.1（新增"各 DSL 样例算子清单"表 + 两条观察），并更新 [[sources/code-and-docs-map]] 的 `examples/` 行指向它。

## [2026-06-22] query | 深剖 attn_proj_residual_cutile 的 cuTile 实现
用户要专门的分析报告。精读 `examples/cutile/jamba_attn_proj/{kernel.py,solution_cutile.json,definition.json}` 与对照组 `examples/cute_dsl/jamba_attn_proj/kernel.py`。新建 [[components/example-attn-proj-cutile]]：逐段剖析 `run` 编排 / `matmul_kernel`（tile MMA + group-M swizzle + fp32 累加 + ZERO padding + `num_ctas=ByTarget(sm_100=2)` 双-CTA） / `add_kernel`，提炼 cuTile 原语速记表。**关键发现**：定义名为 *fused* o_proj+residual，但 cuTile 版实为两 kernel + 物化中间张量、并未融合（cute_dsl 版亦然，且其 GEMM 直接外包 torch、仅手写 add）——已诚实标注为静态阅读的定性判断。接入 [[index]] 导航 + 速查，并从 [[components/data-models]] §3.1 链入。

## [2026-06-22] lint | 全库一致性体检（含新 cuTile 页）
对 18 个页面做体检并核对代码。**核对通过**：`eval_driver.py` 实为 655 行（各页"~650"准确）；cuTile 样例 [[components/example-attn-proj-cutile]] 的行号引用、`ConstInt = ct.Constant[int]` 均与 `examples/cutile/jamba_attn_proj/kernel.py` 一致；"11 样例 / 10 个 DPS=false"经 `find examples -name 'solution*.json'` 实测吻合；四级分类与精度分布各自加总 = 235；无孤儿页、无断裂 wikilink、无"被提及却缺主页"的概念。**修复两处**：(1) [[glossary]] 的 Solution 行只列 7 种语言，但 `core/data/solution.py::SupportedLanguages` 实为 **9** 值（漏 `cudnn_frontend`、`cublas`）——已对齐到 [[components/data-models]] §3 的权威清单；(2) [[components/example-attn-proj-cutile]] §2 的 `run` 代码片段把 `tm=128,...` 写成元组内关键字参（非法 Python），且 host grid 用 `cdiv` 而源码是 `math.ceil`——已改为位置参 + `ceil` 以忠实于源码。

---

### 待办 / 可扩展来源 (open threads)
- SOLAR 仓库（github.com/NVlabs/SOLAR）实现细节尚未 ingest——目前对 SOLAR 的分析基于技术报告 §4.2 与 README，未读其源码。
- `core/data/*.py` 的 Pydantic 校验逻辑只读了 `docs/` 描述与部分调用点，未逐行读源码；如需更深可补 ingest。
- competition 保留的 10 个问题（245 中未公开的部分）不在公开数据集内。
