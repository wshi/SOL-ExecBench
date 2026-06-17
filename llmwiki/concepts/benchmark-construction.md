---
type: concept
title: Benchmark 构造 — 数据集从哪来
updated: 2026-06-17
sources: [arXiv:2603.19173 §3/§4.1]
tags: [dataset, taxonomy, extraction, precision, coverage]
---

# Benchmark 构造：数据集从哪来

回答设计准则 1/2/3（应用驱动、用上新硬件特性、覆盖 forward+backward 与低精度）如何落到一个具体数据集。详见 [[concepts/design-philosophy]] §4 支柱一。

## 1. 三条构造原则（报告 §3）

1. **应用驱动（Application-grounded）**：问题必须由 SOTA 与新兴模型架构驱动，使算子类型、张量形状、数据类型反映**真正在生产里跑/将要跑**的负载。
2. **逼出最新硬件特性**：问题与度量目标要鼓励"用上新 GPU 架构特性才能跑到最优"的解（例如 Blackwell 第 5 代 Tensor Core 上的 NVFP4）。
3. **覆盖 post-training 生命周期**：跨越 fine-tuning、RLHF、推理服务——**含 forward 与 backward**、含低精度（FP8、NVFP4），而非只有推理 forward。

## 2. 提取管线：124 模型 → 7,400 子图 → 235 问题（报告 §3.2，图 1）

```
124 源模型 (6 域)
   │  Model preparation：载入架构定义，抽出源码 + 配置常量(hidden size, heads, dtype)
   ▼
   │  Subgraph extraction：前沿 LLM 分析每个模型，产出"常量内联"的独立 PyTorch 子图
   │   · forward 顺序生成(便于去重)；backward 并行生成；量化模型用专门 prompt
   ▼
7,400 子图
   │  Curation & sampling：每子图按 11 维刻画(算子类型/域/精度/计算强度/前后向…)
   │   · 分层抽样保证多样与均衡；维持单核 vs 多核融合的目标比例；为量化预留名额
   │   · 每个选中子图由 LLM driver generator 转成 benchmark 问题
   ▼
   │  Validation（三道）：
   │   1. 人类专家多轮 review + LLM review：问题良构、抓对子图、reference 正确
   │   2. 执行式校验：跨全部 workload 验数值正确，容差由重复 reference 运行标定
   │   3. 跑 agentic kernel optimizer：暴露"规格漏洞"(agent 靠钻定义歧义而非真写快 kernel 刷高 speedup)
   │   → 任一不过、或易被 spec gaming 的问题，剪掉
   ▼
245 个验证通过 → 公开 235 + 保留 10 给即将到来的 competition
```

几个值得注意的设计：
- **解耦 curation 与 extraction**：子图池可在不重跑提取的情况下**重采样**，以瞄准不同 benchmark 目标。这让数据集可演化。
- **用 agentic optimizer 当"红队"做校验**：构造期就跑优化器去**找规格漏洞**，把"能被钻空子"的问题提前剪掉——这与运行期的 [[concepts/reward-hacking-defenses]] 是同一精神的两端（构造期防 spec gaming，运行期防 reward hacking）。
- **LLM 深度介入但有校验闭环**：提取、转换、审查都用 LLM，但都配人工或执行式验证兜底。

## 3. 源模型覆盖：6 域 124 模型（报告 §3.1）

| 域 | 模型数 | 代表 & 引入的算子 |
|----|--------|-------------------|
| 大语言模型 (LLM) | 61 | Llama-3.x/Gemma-3/Phi-4(dense)、DeepSeek-V3/R1·Qwen3-Coder-480B·GLM-4.7(MoE)、Kimi-K2；GQA、SwiGLU MoE dispatch、multi-token prediction |
| 扩散 (文/图) | 24 | SD 变体、FLUX.1/2、HunyuanImage、Qwen-Image-Edit、Sana、HiDream；adaLN、dual-stream joint attention、VAE；文本扩散 LLaDA-8B 的并行去噪 |
| 视觉 | 6 | SAM-HQ、ConvNextV2、VMamba、NAFNet、Swin2SR、MaskGIT；窗口注意力 |
| 音频/语音 | 9 | Whisper、Parakeet-TDT、Canary(ASR)；Voxtral、OpenVoice、Kokoro、XTTS-v2(TTS)；conformer encoder |
| 视频 | 2 | Wan2.2-T2V；3D RoPE 空间注意力 |
| 多模态/hybrid | 22 | Qwen3-VL/Omni、Llama-3.2-Vision、Gemma-3n、DeepSeek-OCR；Jamba、Nemotron-H、RWKV-v7(SSM/hybrid) |

覆盖广度本身就是设计目标：报告 §1 论证，benchmark 必须覆盖这种架构广度，**既为保持代表性，也为向硬件设计者预示未来负载趋势**（kernel 需求与硬件设计互相塑造）。

## 4. 四级分类（报告 §4，表 2）

| 类别 | 描述 | 数量 | 精度 | 例子 |
|------|------|------|------|------|
| **L1** | 从真实模型抽的**单算子** kernel，神经网络计算的积木 | 94 | BF16/FP32 | GQA、RMSNorm、SwiGLU、RoPE |
| **L2** | **多算子融合** kernel，完整计算块；比 L1 复杂 3–10× | 82 | BF16/FP32 | decoder layer、MoE dispatch、SSM chunk scan、cross-attention |
| **Quant** | 显式低精度计算的 kernel（18 个 FP8 块缩放，15 个 NVFP4 16 元素块缩放） | 33 | FP8/NVFP4 | FP8 MLA projection、NVFP4 MoE expert、FP8 MoE gate |
| **FIB** | 来自 3 个生产模型族(Llama-3.1-8B, Qwen3-30B-A3B, DeepSeek-V3/R1)的**独立推理原语**（并入 FlashInfer-Bench 的 26 个） | 26 | BF16/FP8 | fused attention、FP8 MoE、RMSNorm |
| 合计 | | **235** | | |

L1+L2 = 176（75%），是主体；Quant 与 FIB 全为 forward。

## 5. 问题特征分布（报告 §4.1，图 2）

- **前后向**：189 forward（80%）+ 46 backward（20%）。backward 覆盖如 MoE routing 的梯度 scatter、带 softcapping 的 backward softmax、fused norm-residual 链的反传。
- **算子类型**：attention 主导 81（35%）→ MoE 36（15%）→ normalization 27（12%）→ embedding/位置编码 20（9%）→ linear/projection 16（7%）→ 其它 13、fused blocks 11、GEMM 10、MLP/activation 10、conv 6、SSM/Mamba 5。
- **域**：LLM 153（65%）最多，覆盖全部四级；扩散/视觉集中在 L1/L2。
- **精度**（主数据张量 dtype，非累加 buffer）：BF16 107（46%）→ FP32 79（34%）→ FP8 19（8%）→ NVFP4 15（6%）→ FP16 12（5%）→ Mixed 3（整数/布尔主导，如 mask 构造、MoE 路由排序、多模态位置索引）。

## 6. Workload：动态形状（报告 §3.3/§4.1）

- 每问题约 **16 个 workload**（FIB 7–48），覆盖动态轴如 batch size ∈ {1,…,64}、seq length ∈ {128,…,8192}。
- **78 个问题（33%）用 custom 输入生成**，应对结构化输入：paged KV cache、MoE routing 张量、稀疏 attention mask。这对应数据模型里的 `custom_inputs_entrypoint` 机制（见 [[components/data-models]]）。

## 7. 规格格式（报告 §3.3）

每个问题三件套，遵循**扩展自 FlashInfer Trace 的 schema**：
- **Definition**：名字、op_type、typed symbolic axes（const/var/expr）、输入输出张量 shape/dtype、reference。
- **Reference**：自包含 PyTorch，顶层 `run()`；需结构化输入的问题再定义 `get_inputs()`（代码里即 `custom_inputs_entrypoint`）。
- **Workloads**：每问题多个动态形状实例的具体轴值。

数据模型的工程实现详见 [[components/data-models]]。

## 延伸阅读
- 这套数据集服务于哪条设计准则：[[concepts/design-philosophy]]
- 精度格式与 roofline（compute/memory-bound）：[[concepts/speed-of-light-and-solar]]
- 三件套的代码模型：[[components/data-models]]
- 与其它 benchmark 的对比（表 1）：[[sources/tech-report]]
