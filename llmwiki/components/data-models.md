---
type: component
title: 数据模型 — Definition / Workload / Solution / Trace
updated: 2026-06-22
sources: [docs/{definition,workload,solution,trace}.md, src/sol_execbench/core/data/, arXiv:2603.19173 §3.3]
tags: [data-model, pydantic, schema, DPS, axes]
---

# 数据模型：Definition / Workload / Solution / Trace

四个 Pydantic v2 模型构成 benchmark 的"契约语言"。它们既是内存对象，也是落盘/IPC 的序列化格式（见 [[components/architecture-and-flow]] §5）。schema 扩展自 FlashInfer Trace（报告 §3.3）。

```
Definition  ──"被哪个 Solution 解"──►  Solution
    │                                     │
    │实例化(绑定 var 轴 + 指定输入来源)     │
    ▼                                     ▼
Workload  ───────────► 评测 ◄────────────┘
                         │
                         ▼
                       Trace (单次运行的不可变记录)
```

## 1. Definition — 算子契约（"是什么"）

定义一个计算工作负载的形式规格。**身份规则**：两个 kernel 共享同一 Definition ⟺ 有相同的轴、相同的角色（const/var）、相同的 const 值。

### Symbolic axes（符号维度）—— 设计精髓
轴分三类，这是支撑"一个问题、多个动态形状 workload"的关键抽象：

| 类型 | 何时确定 | 例子 |
|------|----------|------|
| `const` | **定义期**固定 | `hidden_size: {type: const, value: 4096}` |
| `var` | **每个 workload 运行期**绑定 | `batch_size: {type: var}` |
| `expr` | 由其它轴**算出** | `total_tokens: {type: expr, expression: "batch_size * seq_len"}` |

`expr` 支持 `+ - * / // % **`、括号、一元正负。这套设计让一个 Definition 能干净地参数化出几十个形状各异的 workload，而无需重复定义结构（对应报告 §3.3 的"dynamically-shaped workloads"）。

### 其它关键字段
- `inputs`/`outputs`：每个张量给 `shape`（轴名列表；`[]`=0维张量；`null`=Python 标量）+ `dtype`。支持 dtype 含 `float8_e4m3fn/e5m2`、`float4_e2m1fn_x2`（NVFP4 打包）等低精度（见 [[concepts/benchmark-construction]]）。
- `reference`：PyTorch 源码字符串，必须有顶层 `run`；提倡显式逐步、避开 `torch.nn.functional` 捷径以求清晰。它是**功能规格**，不是性能基线（见 [[concepts/correctness-protocol]] §6）。
- `constraints`：轴间断言，如 `"H_qo == H_kv * H_r"`（GQA）。
- `custom_inputs_entrypoint`：见下。

### custom_inputs_entrypoint —— 结构化输入的生成钩子
有些输入纯随机生成不合法：router indices（须是合法专家索引）、softmax 输出（须按维求和为 1）、ragged 序列长度（须非降、和为总 token）、paged KV cache。于是 Definition 可在 reference 里定义一个生成函数：

```python
def fn(axes_and_scalars: dict[str,int], device) -> dict[str, torch.Tensor]: ...
```

评测器对 workload 里标 `"type":"custom"` 的输入调它生成。报告 §4.1：**78 个问题（33%）用 custom 输入**。代码侧 `core/bench/io.py::gen_inputs` 还有一套**启发式张量生成**（按名字猜：`*_weight`→正态/√fan_in、`norm_weight`→全 1、`causal_mask`→上三角填 -inf、softmax 输出→真 softmax、SSM 的 `A`→负值衰减…），让"随机输入"也尽量贴合算子语义、避免数值退化。

## 2. Workload — 把 Definition 变可执行（"用什么数据跑"）

给所有 `var` 轴绑定具体整数值，并指定每个输入的数据来源。一个问题约 16 个 workload，存为 JSONL（一行一个）。

### 输入描述符（`type` 选择来源策略）
| type | 含义 |
|------|------|
| `random` | 按 Definition 的 shape/dtype 随机生成（经启发式） |
| `scalar` | 固定标量值直接传入（对应 `shape:null` 的输入） |
| `safetensors` | 从 `.safetensors` 文件按 `tensor_key` 加载真实数据（KV-cache 索引、真实权重等） |
| `custom` | 由 Definition 的 `custom_inputs_entrypoint` 生成 |

### tolerance —— 逐 workload 的正确性边界
`max_atol`(1e-2) / `max_rtol`(1e-2) / `required_matched_ratio`(0.99) / `max_error_cap`(null) / `allow_negative_inf`(false)。**逐 workload** 而非全局——容易的形状可收紧、难的可放松。完整语义与"为什么这么设计"见 [[concepts/correctness-protocol]] §3。

## 3. Solution — 候选 kernel（"怎么实现的"）

`name` / `definition`（解哪个问题）/ `author` / `spec` / `sources`。

### spec — 构建规格
- `languages`：`pytorch`/`triton`/`cute_dsl`/`cutile`/`cudnn_frontend`/`cuda_cpp`/`cutlass`/`cudnn`/`cublas`。**C++ 与 Python 不能混**。
- `target_hardware`：`["B200"]` / `["local"]` / 两者。
- `entry_point`：`"file::func"`，评测器调的函数。
- `destination_passing_style`（默认 **true**）：见下。
- `dependencies`：受限集合（CUTLASS_3_7、cutlass、triton、cublas、cudnn、torch），按当前环境校验、缺失即快速失败。
- `compile_options`：C++ 的 `cflags`/`cuda_cflags`(默认 `-O3 --use_fast_math`)/`ld_flags`(默认 `-lcuda`)。

### DPS（Destination-Passing Style）—— 默认调用约定
```python
# DPS=true（默认）：输出预分配、作为最后实参传入、原地写、不返回
def run(input, weight, eps, output):
    output[:] = normalize(input, weight, eps)
# DPS=false：返回值风格
def run(input, weight, eps):
    return normalize(input, weight, eps)
```
实参顺序：先 inputs（按 Definition.inputs 顺序）后 outputs（按 Definition.outputs 顺序）。**默认 DPS** 的工程意义：输出 buffer 由评测器预分配，计时时配合 ShiftingMemoryPoolAllocator 复用、并能在每迭代清零（见 [[concepts/timing-and-reproducibility]] §3）；评测器对两种约定都支持（`eval_driver._call_and_collect_outputs`，见 [[components/evaluation-driver]]）。

`sources` 是 `{path, content}` 数组，评测期被重建成目录结构（path 须相对、无 `..`、不重复）。

### 3.1 `examples/` 各 DSL 样例算子清单
仓库 `examples/` 下每种 DSL 各放了 1–3 个样例解，是理解"某 DSL 在本框架里怎么落地"的最短路径。当前全量清单（按 DSL 分组；`DPS` 列= `destination_passing_style`）：

| DSL (`languages`) | 算子 / 问题 | solution 文件 | 入口 | DPS |
|---|---|---|---|---|
| `pytorch` | gemma3 SwiGLU | `examples/pytorch/gemma3_swiglu/` | `kernel.py::run` | false |
| `pytorch` | linear backward | `examples/pytorch/linear_backward/` | `kernel.py::run` | false |
| `triton` | nemotron RMSNorm | `examples/triton/nemotron_rms_norm/` | `kernel.py::run` | false |
| `triton` | olmo3 post-norm + residual | `examples/triton/olmo3_post_norm/` | `kernel.py::run` | false |
| `triton` | RMSNorm（前向） | `examples/triton/rmsnorm/` | `kernel.py::rmsnorm_fwd` | **true** |
| `cutlass` | GEMM | `examples/cutlass/gemm/` | `main.cpp::run` | false |
| `cudnn` | softmax | `examples/cudnn/softmax/` | `main.cu::run` | false |
| `cuda_cpp` | flux RoPE | `examples/cuda_cpp/flux_rope/` | `kernel.cu::run` | false |
| `cuda_cpp` | RMSNorm（Gemini-2.5-pro 产出） | `examples/cuda_cpp/rmsnorm/` | `main.cpp::run` | false |
| `cute_dsl` | Jamba 注意力输出投影 + 残差 | `examples/cute_dsl/jamba_attn_proj/` | `kernel.py::run` | false |
| **`cutile`** | **Jamba 注意力输出投影 + 残差** | **`examples/cutile/jamba_attn_proj/`** | **`kernel.py::run`** | **false** |

观察（截至 2026-06-22 回填）：

- **cuTile 当前只有 1 个样例算子**（`attn_proj_residual_cutile`），与 `cute_dsl` 是**同一个 Jamba 算子的两种 DSL 实现**——便于对照 GEMM + residual add 在 CuTe DSL vs cuTile 下的写法。cuTile 实现拆成 `matmul_kernel`（`ct.mma()`，tile 128×256×64）+ `add_kernel` 两个 kernel；测试用 `@pytest.mark.requires_cutile` 跑（需 SM 100+，对应 [[concepts/timing-and-reproducibility]] 的 Blackwell 门槛）。`tests/sol_execbench/samples/jamba_attn_proj/` 下有一份镜像，`tests/docker/dependencies/test_cutile.py` 是依赖冒烟测试（向量加，非算子样例）。**逐段深度剖析见 [[components/example-attn-proj-cutile]]**（含一个反直觉发现：名为 fused 实际并未融合）。
- **`DPS` 默认是 `true`（§3 DPS 小节），但 11 个样例里 10 个写成 `false`（return-value 风格）**，唯一 `true` 的是 `examples/triton/rmsnorm`。即：默认值与样例的主流写法相反——样例倾向直接返回输出。评测器对两种约定都支持（见 [[components/evaluation-driver]]）。

## 4. Trace — 单次运行的不可变记录（"结果如何"）

把一个 Solution × 一个 workload 的完整评测结果原子化记录；所有 Trace 的集合就是结果数据库。

### status 枚举（状态机出口）
`PASSED` / `INCORRECT_SHAPE` / `INCORRECT_NUMERICAL` / `INCORRECT_DTYPE` / `RUNTIME_ERROR` / `COMPILE_ERROR` / `TIMEOUT` / `REWARD_HACK` / `INVALID_REFERENCE`。

### correctness / performance 的 nullable 规则 —— 体现"先对后快"
| status | correctness | performance |
|--------|-------------|-------------|
| PASSED | 有 | 有 |
| INCORRECT_NUMERICAL | 有 | **无** |
| 其它（SHAPE/DTYPE/RUNTIME/COMPILE/TIMEOUT/REWARD_HACK/INVALID_REFERENCE） | **无** | **无** |

这张表是 [[concepts/sol-score]] §5 "正确性闸门 C∈{0,1}"在数据层的硬编码：**只有 PASSED 才带 performance**，任何不正确/作弊/出错都拿不到性能数 → 后续算 SOL Score 时 C=0、性能分为零。

### 其它
- `performance`：`latency_ms`(候选) / `reference_latency_ms` / `speedup_factor`（= ref/candidate）。注：speedup 相对**朴素 reference**，所以"很大"正常（`docs/trace.md` 注明；也见 [[concepts/design-philosophy]] §5）。
- `environment`：`hardware` + `libs`（版本快照）——评测可复现的元数据。
- `correctness`：`max_relative_error` / `max_absolute_error` / `has_nan` / `has_inf`。

> 设计取向：Trace **自包含且不可变**——含 workload 配置、环境快照、结果，所以单条 Trace 脱离上下文也能解释一次运行。`eval_driver._emit` 用 `allow_nan=False` 序列化，任何意外 NaN/Inf 会立刻报错而非产出非法 JSON。

## 延伸阅读
- 模型如何在进程间流动：[[components/architecture-and-flow]]
- 评测状态机怎么产出这些 status：[[components/evaluation-driver]]
- 容差/正确性细节：[[concepts/correctness-protocol]]
- DPS 与计时的配合：[[concepts/timing-and-reproducibility]]
