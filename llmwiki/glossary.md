---
type: nav
title: 术语表 (Glossary)
updated: 2026-06-17
---

# 术语表 (Glossary)

中英对照 + 一句话释义。详细展开见各自主页。

| 术语 | 含义 | 主页 |
|------|------|------|
| **Speed-of-Light (SOL)** | 光速：某 kernel 在目标硬件上理论可达的最小运行时间（下界）。NVIDIA 内部以"达到 SOL 的百分比"衡量 kernel 质量。 | [[concepts/speed-of-light-and-solar]] |
| **SOLAR** | *SOL Analysis for Runtime*。从 PyTorch 程序解析推导 T_SOL 的管线：graph extractor → agentic einsum converter → roofline analyzer。 | [[concepts/speed-of-light-and-solar]] |
| **T_SOL** | SOLAR 估出的光速运行时间 = `max(FLOPs/算力, 融合字节数/带宽)`（roofline）。 | [[concepts/speed-of-light-and-solar]] |
| **T_b（scoring baseline）** | 评分基线运行时间。一个**高性能**实现（非 reference），锚定 SOL Score 的 0.5 中点；当前不公开、可随 SOTA 更新。 | [[concepts/sol-score]] |
| **T_k** | 候选 kernel 实测运行时间。 | [[concepts/sol-score]] |
| **SOL Score (S)** | `S = 1/(1+(T_k−T_SOL)/(T_b−T_SOL))` ∈ [0,1]。0.5=持平 baseline，1.0=触到硬件 SOL。 | [[concepts/sol-score]] |
| **roofline** | 屋顶线模型：性能受 max(峰值算力, 峰值带宽) 约束；SOL Analyzer 的理论基础。 | [[concepts/speed-of-light-and-solar]] |
| **Orojenesis** | 在 roofline 基础上、按片上缓存容量推导更紧的数据搬移下界；SOLAR 支持它来收紧 T_SOL。 | [[concepts/speed-of-light-and-solar]] |
| **reward hacking** | agent 钻评测漏洞刷分而非真正写出更快 kernel。三类：concurrency / state-caching / environment。 | [[concepts/reward-hacking-defenses]] |
| **DPS (Destination-Passing Style)** | 目标传递风格：输出张量预分配、作为最后几个实参传入、原地写、不返回。默认约定。 | [[components/data-models]] |
| **Definition** | 问题规格：算子契约（symbolic axes、输入输出张量、PyTorch reference）。 | [[components/data-models]] |
| **Workload** | 把 Definition 实例化：给 var 轴绑定具体值 + 指定输入来源 + 容差。每问题约 16 个。 | [[components/data-models]] |
| **Solution** | 候选 kernel：源码文件 + 构建规格 + 入口函数。支持 PyTorch/Triton/CUTLASS/cuDNN/CuTe DSL/cuTile/CUDA C++。 | [[components/data-models]] |
| **Trace** | 单次评测结果的不可变记录：status + correctness + performance + environment。 | [[components/data-models]] |
| **symbolic axes** | 符号维度，分三类：`const`（定义期固定）/ `var`（workload 绑定）/ `expr`（由其它轴算出）。 | [[components/data-models]] |
| **CUPTI** | CUDA Profiling Tools Interface。硬件级 profiling，测真实 GPU kernel 执行时间，排除 CPU launch 开销；代码默认计时法。 | [[concepts/timing-and-reproducibility]] |
| **L2 cache flush** | 每次计时迭代前清零 256MB device buffer，强制冷 cache，避免命中假象。 | [[concepts/timing-and-reproducibility]] |
| **ShiftingMemoryPoolAllocator** | 每迭代把 `data_ptr` 平移 256B 的内存池，破坏"用地址当缓存键"的作弊，且避免 cudaMalloc 干扰计时。 | [[concepts/timing-and-reproducibility]] |
| **clock lock** | 用 `nvidia-smi` 把 GPU/DRAM 频率钉死（B200=1500MHz），消除 boost 抖动带来的 10–30% 方差。 | [[concepts/timing-and-reproducibility]] |
| **matched_ratio** | 通过容差的元素占比阈值（默认 0.99）；与 atol/rtol/max_error_cap 共同定义正确性。 | [[concepts/correctness-protocol]] |
| **NVFP4 / FP8 / BF16** | 低精度浮点格式。NVFP4=Blackwell 第 5 代 Tensor Core 的 4-bit 格式（16 元素块缩放）；FP8 块缩放；BF16 主流。 | [[concepts/benchmark-construction]] |
| **L1 / L2 / Quant / FIB** | 四个 benchmark 分类：单算子 / 多算子融合块 / 量化 / FlashInfer 推理原语。 | [[concepts/benchmark-construction]] |
| **fast_p** | KernelBench 的度量：正确且 speedup>p× 的 kernel 占比。SOL-ExecBench 用 SOL Score 取代之。 | [[sources/tech-report]] |

> 命名来历：**Speed-of-Light** 借自物理学的"光速=不可超越的上限"，在 NVIDIA kernel 工程语境里指"硬件理论极限"。整套 benchmark 的精神就是把评测从"和别人比"换成"和物理上限比"。
