---
type: concept
title: 计时方法与复现性
updated: 2026-06-17
sources: [arXiv:2603.19173 §4.4/§5.1, src/sol_execbench/core/bench/{timing,clock_lock,io}.py]
tags: [timing, reproducibility, CUPTI, L2-cache, clock-lock]
---

# 计时方法与复现性

"可信评测"第二支柱：让 `T_k`（候选运行时间，喂给 [[concepts/sol-score]]）**测得准、测得稳、测不假**。GPU 计时极易被各种效应污染——本页讲 SOL-ExecBench 怎么逐个消除。

## 1. 计时方法：CUPTI 优先，CUDA events 兜底

代码 `core/bench/timing.py::time_runnable` 提供两种方法，**默认 `methodology="cupti"`**：

- **CUPTI（CUDA Profiling Tools Interface）**：`bench_gpu_time_with_cupti`。硬件级 profiling（nsys 同款），测**真实 GPU kernel 执行时间**，**排除 CPU 端 launch 开销**。它用 activity tracing 抓 kernel 的 start/end 时间戳，按 correlation_id 把一次迭代内的 kernel 归并，取 `max(end)−min(start)` 作为该次 GPU 跨度。需要 CUPTI ≥ 13（CUDA 13+）。
- **CUDA events 兜底**：`bench_time_with_cuda_events`，派生自 `triton.testing.do_bench`（MIT），但加了三处修正：每个 start event 前显式同步、warmup 期清 L2、setup 回调用于克隆参数。

> CUPTI 的隐藏好处之一是**抗作弊**：它会校验每次迭代的 kernel 名集合一致（`generate_kernel_string`），若 kernel 集合在迭代间变化就抛错——这能抓"按输入切换不同 kernel"之类的把戏。

> ⚠️ **代码 vs 报告的一处差异**（按 [[CLAUDE]] 的诚实约定标注）：报告 §4.4 描述计时为"**CUDA events**，10 warmup + 50 timed，**3 trials 取 mean**"；而当前代码 `time_runnable` 默认走 **CUPTI**、`return_mode="median"`，单次调用跑 `warmup + rep` 次（由 `BenchmarkConfig` 给 warmup=10、iterations=50）。报告可能描述的是论文实验时的配置或更早版本；两者在"10 warmup / 50 timed / 锁频 / 冷 L2 / 克隆输入"这些**实质方法上一致**，差异在计时后端与聚合统计量。

## 2. 冷 L2 cache：每迭代清零 256MB

`timing.py` 在每次（warmup 与 timed）迭代前清一块 L2 缓冲：

```python
def _get_empty_cache_for_benchmark(device):
    cache_size = get_l2_cache_size(device) * 2   # L2 的两倍，保险
    return torch.empty(int(cache_size), dtype=torch.int8, device=device)
# 每迭代： cache.zero_()
```

报告 §4.4：每次计时迭代前清零 **256MB** device buffer。动机：若不清，前一次迭代的数据还在 L2 里，下一次"命中缓存"会测出**虚低**的运行时间（cache 命中假象）。强制冷 cache → 测到的是真实的、可复现的访存代价。CUPTI 路径按迭代测量，逐迭代 flush 正好对齐。

## 3. ShiftingMemoryPoolAllocator：一举三得

`core/bench/io.py::ShiftingMemoryPoolAllocator` 是计时设计里最精巧的一块。它预分配一个"略大于输入"的内存池，每次 `get_unique_args()` 把源数据拷到**前移一个偏移**的位置，于是每次迭代拿到的张量 `data_ptr` 都不同：

```python
_POOL_ALIGNMENT = 256   # 每迭代平移 256B
pool_numel = storage_span + (total_iterations - 1) * stride_numel
```

它同时解决三个问题：

1. **反 state-caching 作弊**：agent 想用 `data_ptr` 当缓存键"算一次反复返回"，但每迭代地址都变 256B，缓存键失效（报告 §4.4.1 明确这是防御机制；见 [[concepts/reward-hacking-defenses]] §B）。
2. **排除 cudaMalloc 干扰**：所有迭代的张量**预先一次性分配**，计时区间内不触发 `cudaMalloc`。`time_runnable` 注释直言：若不这样，cudaMalloc 会让测得的 kernel 时间**增加约 300%**。
3. **显存几乎不膨胀**：池只比输入大 `(iters−1)×256B`/张量，且保留 stride（包括 `expand()` 的 stride-0 广播只存物理 footprint），所以 VRAM 维持在约 1× 输入大小。

输出张量（DPS）则每次在新偏移**清零**后给出，保证每次迭代从干净状态开始。

## 4. 克隆参数：每次迭代用新鲜输入

`timing.py::clone_args` 与 allocator 的 `get_unique_args` 一起，保证每次计时迭代**从独立数据开始**，不复用上一迭代被 kernel 改过的状态。报告 §4.4："clone tensor arguments so each run starts from fresh inputs with new addresses"。这既保计时干净，也再添一道反作弊（防"原地累积/状态泄漏"）。

## 5. GPU 锁频：消除 boost 抖动

`core/bench/clock_lock.py` 用 `sudo nvidia-smi` 把 GPU 与 DRAM 频率钉死：

- 动机（模块 docstring）：GPU boost clock 在热压力下浮动会带来 **10–30% 的延迟方差**；锁频消除它。
- B200 锁到 **1,500 MHz**（报告 §4.4/§5.1；与 SOLAR 算 T_SOL 时用的目标频率一致——见 [[concepts/speed-of-light-and-solar]] 例子 "B200@1.5GHz"）。频率可被环境变量 `SOL_EXECBENCH_GPU_CLK_MHZ` / `SOL_EXECBENCH_DRAM_CLK_MHZ` 覆盖，预设表按设备名查（`config/device_config.py`）。
- 锁后等 3 秒再 `verify_clocks`（查 `nvidia-smi` 实际频率是否在 ±50MHz 内），不达标就解锁并失败——**不假装锁成功**。
- 评测器侧：`are_clocks_locked()` 读 `SOL_EXECBENCH_CLOCKS_LOCKED` 环境变量。若用户要求 `lock_clocks=True` 但实际没锁，`eval_driver` 会**拒绝所有 workload**（`RUNTIME_ERROR`），而不是给出不可信的数。

> T_SOL 在 1.5GHz 下推导、T_k 也在 1.5GHz 下实测——**理论极限与实测在同一频率下可比**，这是 SOL Score 有意义的前提。

## 6. 序列化执行

报告 §4.4：给定 GPU 上的 benchmark **串行执行**（不并发跑多个解争抢资源），跨多 GPU 时才 round-robin 调度。串行 + 锁频 + 冷 L2 + 平移池，共同把"同一个解多次评测得到同一个数"的复现性拉满。

## 7. 实验环境（报告 §5.1，复现锚点）

| 项 | 值 |
|----|----|
| 硬件 | NVIDIA DGX B200，8×B200，每卡 192GB HBM3e，8 TB/s 带宽 |
| 频率 | SM 锁定 1,500 MHz |
| 软件 | CUDA 13.1.1 · cuDNN 9.17.1 · PyTorch 2.9.0 · driver 580.95 |
| 计时 | 10 warmup + 50 timed（每卡单 GPU 运行） |

## 防污染清单（一页速记）
- launch 开销 → **CUPTI** 排除
- cache 命中假象 → **每迭代清 256MB L2**
- cudaMalloc 干扰（+300%）→ **预分配内存池**
- 地址缓存作弊 → **每迭代平移 data_ptr 256B**
- 状态泄漏 → **克隆输入**
- boost 抖动（10–30%）→ **锁频 1.5GHz**
- 资源争抢 → **串行执行**

## 延伸阅读
- 这些结构性手段同时是反作弊：[[concepts/reward-hacking-defenses]]
- 测出的 T_k 怎么变成分数：[[concepts/sol-score]]
- T_SOL 同频率推导：[[concepts/speed-of-light-and-solar]]
- 计时在驱动流程里的位置：[[components/evaluation-driver]]
