---
type: nav
title: SOL-ExecBench LLM Wiki — 入口
updated: 2026-06-17
---

# SOL-ExecBench LLM Wiki

一套用 [Karpathy LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) 方法论搭建的**知识库**，目标是把 **SOL-ExecBench** 的设计思想与具体方法讲透。

> **SOL-ExecBench**（Speed-of-Light ExecBench）是 NVIDIA 的 GPU kernel 评测框架，用来评估 AI（agentic optimizer / LLM）生成的 kernel。它的根本主张：**别再比"打败软件 baseline 快多少"，要比"离硬件光速（Speed-of-Light）还差多少"。** —— 技术报告 arXiv:[2603.19173](https://arxiv.org/abs/2603.19173)。

---

## 这套 wiki 是什么

- 它**不是**对 README 的复述，而是结合**仓库代码 + `docs/` + 技术报告**，对 SOL-ExecBench 的每一个核心设计决策做"是什么 / 为什么 / 有何权衡"的深度剖析。
- 三层结构（详见 [[CLAUDE]]）：**原始来源**（只读）→ **wiki 页面**（本目录，LLM 维护）→ **schema**（`CLAUDE.md`，约定）。
- 维护方式见 `CLAUDE.md` 的 ingest / query / lint 流程。

## 从哪开始读

1. **先看大图** → [[concepts/design-philosophy]]：从"打败 baseline"到"逼近 SOL"的范式转变，一页讲清整个项目的"为什么"。
2. **核心度量** → [[concepts/sol-score]] 与 [[concepts/speed-of-light-and-solar]]：SOL Score 公式、SOLAR roofline 管线。
3. **可信评测三支柱** → [[concepts/correctness-protocol]]、[[concepts/timing-and-reproducibility]]、[[concepts/reward-hacking-defenses]]。
4. **数据集怎么来的** → [[concepts/benchmark-construction]]。
5. **代码怎么落地的** → [[components/architecture-and-flow]] → [[components/evaluation-driver]] → [[components/data-models]]。

完整目录见 [[index]]；术语见 [[glossary]]；来源回溯见 [[sources/tech-report]] 与 [[sources/code-and-docs-map]]。

## 一句话抓住设计精髓

| 维度 | 传统 kernel benchmark | SOL-ExecBench |
|------|----------------------|---------------|
| 评测锚点 | 可变的软件 baseline（如 PyTorch eager） | 解析推导的硬件 **SOL 下界**（固定上限） |
| 度量 | speedup（越大越好，但无上限、可被刷） | **SOL Score** ∈ [0,1]，0.5=持平 baseline，1.0=触到硬件极限 |
| 问题来源 | 旧模型 / 合成 / 单算子 | 124 个**生产级模型**抽取的 235 个真实子图 |
| 对抗鲁棒性 | 弱 | sandbox：clock lock + L2 flush + 子进程隔离 + reward-hack 静/动态检测 |

> 实验里：speedup 与"离 SOL 多远"几乎不相关（r=0.10），而 SOL Score 与"填补了多少优化空间"近乎完全相关（r=0.981）。这就是为什么要换锚点。详见 [[concepts/sol-score]]。
