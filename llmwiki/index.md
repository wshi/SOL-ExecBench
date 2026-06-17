---
type: nav
title: 内容目录 (Index)
updated: 2026-06-17
---

# 内容目录 (Index)

本 wiki 全部页面的目录。每页一行摘要。提问时先读这里定位，再钻取细节。

## 导航 (Navigation)

| 页面 | 摘要 |
|------|------|
| [[README]] | 入口：这套 wiki 是什么、从哪开始读、一句话抓住设计精髓 |
| [[CLAUDE]] | Schema 层：三层架构、目录约定、ingest/query/lint 流程 |
| [[index]] | 本页：内容目录 |
| [[log]] | 时间线：ingest / lint 记录 |
| [[glossary]] | 术语表（中英对照） |

## 概念 (Concepts) — 设计思想与方法论

| 页面 | 摘要 |
|------|------|
| [[concepts/design-philosophy]] | **核心综述**：从"打败软件 baseline"到"逼近硬件 SOL"的范式转变；五条 benchmark 设计准则；三大设计支柱 |
| [[concepts/speed-of-light-and-solar]] | Speed-of-Light 思想；SOLAR 管线（graph extractor → agentic einsum converter → roofline analyzer）；T_SOL 公式与局限 |
| [[concepts/sol-score]] | SOL Score 公式 `S=1/(1+(T_k−T_SOL)/(T_b−T_SOL))`；三锚点性质；correctness gate；为何用算术平均；与 speedup 的对比实验 |
| [[concepts/reward-hacking-defenses]] | 三类作弊（concurrency / state-caching / environment）与对应防御；静态 LLM-judge + 动态运行时检测；14.5% 命中率 |
| [[concepts/correctness-protocol]] | 10 轮随机重测；torch.allclose 风格容差 (atol,rtol,matched_ratio)；inf/NaN/全零 sanity；容差离线标定 1.25× margin |
| [[concepts/timing-and-reproducibility]] | CUPTI vs CUDA events；L2 cache flush；ShiftingMemoryPoolAllocator；clock lock(1500MHz)；warmup/iterations |
| [[concepts/benchmark-construction]] | 124 模型 → 7,400 子图 → 235 问题；四级分类 L1/L2/Quant/FIB；提取/筛选/校验管线；精度分布 |

## 组件 (Components) — 代码架构

| 页面 | 摘要 |
|------|------|
| [[components/architecture-and-flow]] | 三层架构 CLI / Driver / Core；端到端执行流；子进程隔离的设计动机 |
| [[components/data-models]] | Definition / Workload / Solution / Trace 四个 Pydantic 模型的设计；symbolic axes (const/var/expr)；DPS |
| [[components/evaluation-driver]] | `ProblemPackager` 暂存与编译；`eval_driver.py` ~650 行自包含脚本逐段剖析；防御检查的插入点 |

## 来源 (Sources) — 原始材料摘要

| 页面 | 摘要 |
|------|------|
| [[sources/tech-report]] | 技术报告 arXiv:2603.19173 逐节要点（摘要、相关工作、构造、SOLAR、度量、评测框架、实验） |
| [[sources/code-and-docs-map]] | 仓库代码与 `docs/` 的来源索引：每个关键文件是什么、对应哪个 wiki 页 |

---

### 按主题速查

- **"为什么换评测锚点"** → [[concepts/design-philosophy]]、[[concepts/sol-score]]
- **"硬件极限怎么算出来的"** → [[concepts/speed-of-light-and-solar]]
- **"怎么防作弊"** → [[concepts/reward-hacking-defenses]]、[[components/evaluation-driver]]
- **"怎么保证测得准、测得稳"** → [[concepts/timing-and-reproducibility]]、[[concepts/correctness-protocol]]
- **"数据集从哪来"** → [[concepts/benchmark-construction]]
- **"代码怎么跑起来的"** → [[components/architecture-and-flow]]、[[components/evaluation-driver]]
