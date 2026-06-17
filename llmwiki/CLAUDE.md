---
type: schema
title: SOL-ExecBench LLM Wiki — Schema & 维护约定
updated: 2026-06-17
---

# SOL-ExecBench LLM Wiki — Schema

本文件是这套 wiki 的 **schema 层**（对应 Karpathy [LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) 模式中的 `CLAUDE.md`/`AGENTS.md`）。它告诉 LLM agent：这套 wiki 怎么组织、有哪些约定、在 ingest / query / lint 时该遵循什么流程。**人类负责选材、提问、把关；LLM 负责读、写、交叉引用、记账。**

> 这套 wiki 的研究对象是 **SOL-ExecBench**：NVIDIA 面向 AI 生成 GPU kernel 的"光速（Speed-of-Light）"评测框架。目标是把它的**设计思想**与**具体方法**讲透，证据来自仓库代码、`docs/`、以及技术报告 arXiv:2603.19173。

---

## 1. 三层架构（Three Layers）

| 层 | 位置 | 谁拥有 | 是否可变 |
|----|------|--------|----------|
| **Raw sources（原始来源）** | 仓库 `src/`、`docs/`、`README.md`、技术报告 arXiv:2603.19173 | 上游项目 | 对 wiki 只读、不可改 |
| **The wiki（本目录）** | `llmwiki/**.md` | LLM agent | LLM 持续增改 |
| **The schema（本文件）** | `llmwiki/CLAUDE.md` | 人 + LLM 共同演进 | 随约定调整 |

原始来源是"真理之源"。wiki 页面里的每一条结论都应能回溯到某个源文件的具体位置（文件路径 + 行号/函数名，或技术报告章节号）。

---

## 2. 目录结构（Directory Layout）

```
llmwiki/
├── CLAUDE.md      # 本文件：schema 与维护约定
├── README.md      # 入口：这是什么、怎么用、从哪开始读
├── index.md       # 内容目录（catalog）：每页一行摘要，按类别组织
├── log.md         # 时间线（append-only）：ingest / query / lint 记录
├── glossary.md    # 术语表（中英对照）
├── concepts/      # 概念页：方法论与设计思想的深度剖析
│   ├── design-philosophy.md
│   ├── speed-of-light-and-solar.md
│   ├── sol-score.md
│   ├── reward-hacking-defenses.md
│   ├── correctness-protocol.md
│   ├── timing-and-reproducibility.md
│   └── benchmark-construction.md
├── components/    # 实体页：代码模块 / 架构组件
│   ├── architecture-and-flow.md
│   ├── data-models.md
│   └── evaluation-driver.md
└── sources/       # 原始来源摘要
    ├── tech-report.md
    └── code-and-docs-map.md
```

### 页面类型（Page Types）

- **概念页（concept）** — 一个设计思想或方法（如 SOL Score、reward hacking 防御）。回答"是什么 / 为什么这么设计 / 有什么权衡"。
- **实体页（component）** — 一个代码模块或架构组件。回答"在代码里怎么落地的"。
- **来源摘要（source）** — 对某个原始来源（技术报告、docs）的浓缩，便于检索回溯。
- **导航页** — `index.md`（内容）、`log.md`（时间）、`README.md`（入口）、`glossary.md`（术语）。

---

## 3. 页面约定（Conventions）

1. **YAML frontmatter**：每页顶部带 `type / title / updated / sources / tags`，便于 Dataview 之类工具聚合。
2. **Wikilink 交叉引用**：页面间用 `[[concepts/sol-score]]` 形式互链；关键概念第一次出现时务必链到它的主页。
3. **可回溯**：技术结论后给出证据锚点，例如 `src/sol_execbench/core/bench/timing.py::ShiftingMemoryPoolAllocator` 或 `技术报告 §4.4.1`。
4. **单一主页原则**：每个概念有且只有一个"主页"，其它页面引用而非复制。发现重复就合并。
5. **语言**：正文中文；`SOL Score`、`roofline`、`reward hacking`、`CUPTI`、`DPS`、`NVFP4` 等技术术语保留英文。
6. **诚实标注不确定性**：代码与报告冲突、或某处未读透，明确写出（例如"报告说 CUDA events，代码默认 CUPTI——见 [[concepts/timing-and-reproducibility]]"）。

---

## 4. 三种操作（Operations）

### Ingest（摄入新来源）
读新来源 → 与人确认要点 → 在 `sources/` 写/更新摘要 → 更新相关 `concepts/`、`components/` 页 → 更新 `index.md` → 在 `log.md` 追加一条 `ingest` 记录。一次摄入通常会触碰多页。

### Query（提问）
先读 `index.md` 定位相关页 → 钻取细节 → 给出带引用的回答。**好的回答应回填为新页面或某页的新小节**，让探索沉淀进 wiki，而不是消失在对话里。

### Lint（体检）
定期自查：页面间矛盾、被新来源推翻的旧结论、无入链的孤儿页、被提到却没有主页的概念、缺失的交叉引用、可补的空白。把发现写进 `log.md` 并修复。

---

## 5. log.md 约定

追加式（append-only），每条以统一前缀开头，便于 `grep "^## \[" log.md | tail` 取最近记录：

```
## [2026-06-17] ingest | 技术报告 §4 评测框架
## [2026-06-17] lint   | 修复 sol-score 与 timing 页的交叉引用
```

---

## 6. 当前状态（Status）

- v0.1（2026-06-17）：基于一次性 ingest（仓库代码 + `docs/` + 技术报告 arXiv:2603.19173）搭建首版框架并填充深度分析。后续来源（新版本代码、competition 子集、SOLAR 仓库细节）可按上面的 ingest 流程增量并入。
