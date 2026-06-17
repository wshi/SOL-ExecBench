---
type: concept
title: Speed-of-Light 思想与 SOLAR 管线
updated: 2026-06-17
sources: [arXiv:2603.19173 §2.2/§4.2, README.md, github.com/NVlabs/SOLAR]
tags: [SOL, SOLAR, roofline, einsum, T_SOL]
---

# Speed-of-Light 思想与 SOLAR 管线

回答"硬件极限到底怎么算出来"——这是 [[concepts/sol-score]] 的输入 `T_SOL` 的来源。

## 1. 什么是 Speed-of-Light（SOL）

**SOL = 某计算在目标硬件上理论可达的最短运行时间**（下界）。它不是某个实现跑出来的，而是从工作量与硬件峰值能力**解析推导**的。

报告 §2.2 点明出处：用 SOL 下界来评测，对齐的是 **NVIDIA 内部 kernel 团队的方法论**——他们衡量 kernel 一直是看"达到硬件 SOL 的百分比"，而非相对软件 baseline。SOL-ExecBench 把这套内部尺子公开化、自动化。

理论基础是经典的 **roofline 模型**（Williams et al. 2009）：任何 kernel 的性能上限由"峰值算力"和"峰值内存带宽"两条线中**更紧的那条**决定。

## 2. T_SOL 公式（roofline）

报告 §4.2 Eq.(1)：

```
T_SOL = max( Total FLOPs / Compute Throughput ,
             Total Fused Bytes / Memory Bandwidth )
```

- 取 `max` 的含义：要么被算力卡住（compute-bound），要么被带宽卡住（memory-bound），运行时间不可能短于两者中更大的那个。
- **Fused Bytes（融合后字节数）** 是关键词：分析器会考虑**图级融合与预取**，只算真正必须进出片外内存的字节，而不是把每个算子的读写朴素相加。这让下界更贴近"一个好 kernel 实际能做到的"。

报告 §4.2 给的具体例子（来自 Jamba-Reasoning-3B 的一个 L1 问题：矩阵乘 + 残差加）：
- Total FLOPs ≈ 107.4G，融合内存 ≈ 126MB，算术强度 ≈ 427 → **compute-bound**
- 在 B200@1.5GHz 上 T_SOL ≈ **0.06 ms**

> 算术强度（arithmetic intensity）= FLOPs / Bytes。它落在 roofline 的哪一侧，决定了 kernel 是算力受限还是带宽受限——这正是 kernel 优化时最先要判断的事。

## 3. SOLAR 管线：从 PyTorch 程序到 T_SOL

**SOLAR = SOL Analysis for Runtime**（开源于 github.com/NVlabs/SOLAR）。输入一段 PyTorch 程序 + 输入形状，输出 T_SOL。三个分析阶段（报告 §4.2，图 3）：

```
PyTorch 程序 + 输入形状
      │
 ① Graph Extractor      ── torchview，forward hooks 抓 live forward 的算子图 + 中间张量形状
      │  (尊重 eager 执行与动态控制流，无需静态图)
      ▼
   算子图 (matmul / add / transpose …, 带 shape)
      │
 ② Agentic Einsum Converter ── 把算子翻成"扩展 einsum 表达式"(index-based 规范形)
      │  · 已知算子 → 查持久 lookup table 直接转换
      │  · 未见算子 → LLM agent 生成候选转换函数，emulate 结果与原算子比对自校验后入表
      ▼
   扩展 einsum 图 (显式暴露张量迭代空间与计算模式)
      │
 ③ SOL Analyzer        ── 从 einsum 图导出 FLOPs 与内存流量；按 roofline + 目标频率算 T_SOL
      │  · 考虑图级 fusion / prefetch
      │  · 支持 Orojenesis：按片上 buffer 容量推导更紧的数据搬移下界
      ▼
   T_SOL
```

### ① Graph Extractor
基于 `torchview`，用 **forward hooks** 在一次真实 forward 中收集张量元数据。这样做的好处：**尊重 PyTorch 的 eager 执行与动态控制流**，能抓到真实执行路径，而不需要把模型静态图化（很多前沿模型有数据依赖的控制流，静态图化要么做不到要么失真）。

### ② Agentic Einsum Converter
把算子统一翻译成**扩展 einsum**（广义爱因斯坦求和，index-based 记号）。这个规范形的价值在于：它**显式暴露张量的迭代空间和计算模式**，从而能机械地数出 FLOPs 与内存流量。
- 例子（报告 §4.2）：matmul `ACB,BH→ACH`（一个 contraction 节点）+ 投影后的 elementwise add `ACH,ACH→ACH`。
- **自动化 + 自校验**：没见过的算子由 LLM agent 生成转换函数，并通过"emulate einsum 结果 vs 原 PyTorch 算子结果"自动验证后才入库。lookup table 持久化，越用越全——这是"agentic"的体现。

### ③ SOL Analyzer
拿 einsum 图 + 硬件规格（峰值算力、带宽、目标频率）套 roofline 公式。两个精细化：
- **图级 fusion / prefetch**：只计必要的片外字节。
- **Orojenesis**（Huang et al. 2024）：朴素 roofline 假设所有数据都能驻留片上做完美复用，会**高估**可达性能；Orojenesis 把"片上 buffer 容量有限"建模进来，给出更紧（更现实）的数据搬移下界。

## 4. SOLAR 的两个已知局限（报告 §4.2，诚实披露）

1. **只看形状不看数值**：无法捕捉值依赖的优化（压缩、常量传播、结构化/重复数据带来的访存或代数化简收益）。
2. **SOL bound 实践中未必"紧"**：硬件可能 power capping 或 thermal throttling，真实极限可能略低于解析值。

这两条局限正是 [[concepts/sol-score]] 里设置"审计信号"的原因——当出现 `T_k < T_SOL`（候选比理论极限还快）时，要么 SOLAR 算松了，要么有 reward hacking，需人工复查。

## 5. 它如何接进度量

SOLAR 产出的 `T_SOL` 是固定的"满分线"，与"持平线"`T_b`（scoring baseline）一起喂给 SOL Score。三者关系：

```
T_SOL  ≤  T_k  ……  T_b      （理想情形：候选介于极限与基线之间）
 满分(S=1)   候选     0.5线
```

完整数学见 [[concepts/sol-score]]。

## 延伸阅读
- 度量怎么用 T_SOL：[[concepts/sol-score]]
- 为什么要换这个锚点：[[concepts/design-philosophy]]
- 数据集里各精度/算子的分布（决定 compute/memory-bound 比例）：[[concepts/benchmark-construction]]
- 报告 §4.2 原文要点：[[sources/tech-report]]
