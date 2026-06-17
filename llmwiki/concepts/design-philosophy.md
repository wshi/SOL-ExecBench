---
type: concept
title: 核心设计哲学 — 从"打败 baseline"到"逼近 SOL"
updated: 2026-06-17
sources: [arXiv:2603.19173 §1/§2/§3, README.md]
tags: [philosophy, synthesis, motivation]
---

# 核心设计哲学：从"打败 baseline"到"逼近 SOL"

> 这是整套 wiki 的**综述页**。读懂这一页，就抓住了 SOL-ExecBench 所有具体方法背后的"为什么"。

## 1. 问题：speedup 这把尺子坏了

随着 agentic AI 系统越来越会写、会调 GPU kernel，"怎么评测它们"反而成了瓶颈。主流 kernel benchmark（KernelBench、TritonBench、BackendBench 等）几乎都用同一把尺子：**相对某个软件 baseline（通常是 PyTorch eager）的 speedup**。

这把尺子有三个根本缺陷：

1. **锚点会动**。baseline 是软件实现，PyTorch 一升级、baseline 一换，"快了多少"就变了。横向、纵向都不可比。
2. **没有上限，掩盖真实差距**。speedup 可以无限大，但"比 PyTorch 快 10×"完全可能**离硬件极限还差 10×以上**。报告 §5.2 用实验坐实了这点：在 SOL-ExecBench 上，speedup 与"离 SOL 还有多远"几乎不相关（log-log 下 Pearson **r=0.10**）。一把奖励"比烂 baseline 快"的尺子，会把"还有巨大优化空间"的 kernel 排到前面。
3. **容易被刷**。baseline 越慢越容易"超越"，于是优化目标退化成"找个更弱的对比对象"。

报告 §1 把这件事拔高到产业层面：每代 GPU（Blackwell 引入 NVFP4、第 5 代 Tensor Core 等）都在快速加入新特性，数据中心又把能效当硬约束，人工逐个写 SOL kernel 跟不上"硬件特性节奏 × 模型复杂度增长"。所以**AI 写 kernel 不是可选项而是必需品**——也因此，评测它们的尺子必须是对的。

## 2. 范式转变：换一个不会动、有上限、刷不了的锚点

SOL-ExecBench 的根本主张（也是它名字的来历）：

> **不要问"比软件 baseline 快多少"，要问"离硬件 Speed-of-Light（光速 / 理论极限）还差多少"。**

"Speed-of-Light"借自物理学——光速是不可超越的上限。在 NVIDIA 内部 kernel 工程里，性能一直是按"达到硬件 SOL 的百分比"来衡量的，而不是相对软件 baseline。SOL-ExecBench 把这套**工业界内部方法论**固化成了一个公开 benchmark（报告 §2.2）。

这个锚点的好处正好对症前面三个缺陷：

- **不会动**：T_SOL 由 FLOPs、字节数、硬件峰值算力/带宽**解析推导**（roofline），是物理量，不随软件版本漂移。怎么算见 [[concepts/speed-of-light-and-solar]]。
- **有上限**：度量被归一化到 [0,1]，1.0 就是触到硬件极限，天然暴露"还剩多少头"。见 [[concepts/sol-score]]。
- **难刷**：锚点是物理极限而非弱对手，"找更烂的 baseline"这条捷径直接消失。

## 3. 五条 benchmark 设计准则（报告 §1）

报告明确：一个面向 agentic kernel 优化的 benchmark 必须同时满足五条，而**现有 benchmark 没有一个同时满足**：

1. 覆盖当前前沿与新兴架构（dense Transformer、MoE、SSM、线性注意力、hybrid、多模态……）；
2. 包含"必须用上新硬件特性与低精度格式才能跑到最优"的问题；
3. 同时含训练后期（post-training）与推理（inference）负载，即 forward **和** backward；
4. 用**硬件极限**而非可变软件 baseline 作评测目标；
5. 评测基础设施要能扛**对抗式优化**（anti reward-hacking）。

SOL-ExecBench 就是冲着同时满足这五条设计的（与同类 benchmark 的对比见 [[sources/tech-report]] 表 1）。

## 4. 由准则推出的三个具体设计（数据集 / 度量 / 评测）

```
准则 1,2,3  ─►  应用驱动的数据集     ─►  [[concepts/benchmark-construction]]
准则 4      ─►  SOL 锚定的度量       ─►  [[concepts/sol-score]] + [[concepts/speed-of-light-and-solar]]
准则 5      ─►  可信、抗作弊的评测     ─►  正确性 / 计时复现 / 反作弊三支柱
```

### 支柱一：应用驱动的数据集（Application-grounded）
从 124 个**生产级 + 新兴**模型抽取 7,400 个计算子图，筛成 235 个问题，分 L1/L2/Quant/FIB 四级，覆盖 forward+backward、BF16/FP8/NVFP4，每问题约 16 个动态形状 workload。"问题必须来自真实模型"保证算子类型、张量形状、数据类型都反映真正在跑/将要跑的负载。详见 [[concepts/benchmark-construction]]。

### 支柱二：SOL 锚定的度量（Hardware-grounded metric）
**SOLAR** 管线从 PyTorch reference 解析推导每个问题的 T_SOL；**SOL Score** 用 `S=1/(1+(T_k−T_SOL)/(T_b−T_SOL))` 把"候选 kernel 填补了多少 baseline→SOL 的空间"映射到 [0,1]。0.5=持平 baseline，1.0=触到硬件极限。详见 [[concepts/sol-score]]、[[concepts/speed-of-light-and-solar]]。

### 支柱三：可信、抗作弊的评测（Robust harness）
报告 §4.4 把它总结为一个 sandbox：**GPU clock locking + L2 cache clearing + 子进程隔离 + 针对常见 reward-hacking 的静/动态检测**。这背后是三个互补的子系统：

- **正确性协议** —— 多轮随机重测 + 工作负载级容差，先证明"算对了"才给性能分。见 [[concepts/correctness-protocol]]。
- **计时与复现** —— CUPTI 计时 + 冷 L2 + 平移内存池 + 锁频，让"测得准、测得稳"。见 [[concepts/timing-and-reproducibility]]。
- **反作弊防御** —— 把观察到的三类作弊（concurrency / state-caching / environment）逐一堵死。见 [[concepts/reward-hacking-defenses]]。

## 5. 一个贯穿全局的关键区分：reference ≠ baseline

这是理解整套设计的"枢纽"，很多人会混淆（报告 §4.4）：

| | **reference（参考实现）** | **scoring baseline（评分基线）** |
|---|---|---|
| 角色 | 定义**语义**、用于正确性比对 | 锚定 **SOL Score 的 0.5 中点** |
| 写法 | 朴素、可读、可移植的 PyTorch；**可能很慢** | 高性能实现（由 agentic optimizer 产出） |
| 是否公开 | 公开（在 Definition 里） | **当前不公开**，且可随 SOTA 更新 |
| 是否计入度量 | 否（只判对错） | 是（T_b） |

把两者分开，是为了既能用一份"清晰但慢"的 PyTorch 代码定义"正确"，又能用一份"强但保密"的实现定义"持平线"——后者还能随时换更强的，让 benchmark 始终保持挑战性。这也解释了一个常见困惑：Trace 里的 speedup_factor 可能极大，因为它是相对**朴素 reference**算的，而非相对 baseline（见 [[components/data-models]]）。

## 6. 设计哲学一览（自检清单）

- 评测锚点 = **物理极限**，不是可变对手。
- 度量 = **有界、可解释、可跨问题平均**（[0,1]，0.5/1.0 双锚点）。
- 数据 = **真实模型子图**，覆盖架构广度 + forward/backward + 低精度。
- 可信 = **正确性闸门 + 复现性工程 + 反作弊**，三者缺一不可。
- 透明 = reference 公开、SOLAR 开源、harness 开源；只有 baseline 与 competition 子集保留。

## 延伸阅读
- 度量数学：[[concepts/sol-score]]
- 极限怎么算：[[concepts/speed-of-light-and-solar]]
- 三支柱：[[concepts/correctness-protocol]] · [[concepts/timing-and-reproducibility]] · [[concepts/reward-hacking-defenses]]
- 数据集：[[concepts/benchmark-construction]]
- 代码落地：[[components/architecture-and-flow]]
- 报告原文要点：[[sources/tech-report]]
