---
type: nav
title: 时间线 (Log)
updated: 2026-06-17
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

---

### 待办 / 可扩展来源 (open threads)
- SOLAR 仓库（github.com/NVlabs/SOLAR）实现细节尚未 ingest——目前对 SOLAR 的分析基于技术报告 §4.2 与 README，未读其源码。
- `core/data/*.py` 的 Pydantic 校验逻辑只读了 `docs/` 描述与部分调用点，未逐行读源码；如需更深可补 ingest。
- competition 保留的 10 个问题（245 中未公开的部分）不在公开数据集内。
