---
type: component
title: 架构与端到端执行流
updated: 2026-06-17
sources: [CLAUDE.md(repo), src/sol_execbench/{cli,driver,core}, arXiv:2603.19173 §4.4]
tags: [architecture, execution-flow, subprocess-isolation]
---

# 架构与端到端执行流

把"设计哲学"落到代码组织。SOL-ExecBench 代码库分三层，评测以**子进程**为执行单元。

## 1. 三层架构

```
┌─────────────────────────────────────────────────────────────┐
│ CLI 层   src/sol_execbench/cli/main.py                        │
│   Click 入口 `sol-execbench`；解析参数；调度 driver；          │
│   解析 traces；Rich 表格展示结果                               │
└───────────────┬─────────────────────────────────────────────┘
                │ 调用
┌───────────────▼─────────────────────────────────────────────┐
│ Driver 层  src/sol_execbench/driver/                          │
│   ProblemPackager：把问题文件(definition/workload/solution/   │
│     sources)暂存到临时目录，产出"编译"与"执行"两条 shell 命令  │
│   templates/eval_driver.py：在子进程里运行的自包含评测脚本     │
│   templates/build_ext.py：C++/CUDA 编译脚本                    │
└───────────────┬─────────────────────────────────────────────┘
                │ 用到
┌───────────────▼─────────────────────────────────────────────┐
│ Core 层  src/sol_execbench/core/                              │
│   core/data/：Pydantic v2 模型 Definition/Solution/Workload/  │
│     Trace（+ dtypes/shapes/base_model）→ [[components/data-models]] │
│   core/bench/：计时(timing) 正确性(correctness) 锁频(clock_   │
│     lock) 反作弊(reward_hack) IO(io) 配置(config)             │
└─────────────────────────────────────────────────────────────┘
```

职责切分清楚：**CLI 管交互、Driver 管编排与隔离、Core 管"评测科学"**（怎么算对、怎么测准、怎么防作弊）。SOL Score 计算 `sol_score.py` 独立在顶层（评测产出 latency，分数后算——见 [[concepts/sol-score]]）。

## 2. 端到端执行流

仓库 `CLAUDE.md` 概括的主流程，结合 `problem_packager.py` 与 `eval_driver.py`：

```
CLI 解析参数
  │
  ▼
ProblemPackager 暂存文件到临时目录
  ├─ 写 definition.json / workload.jsonl / solution.json / config.json
  └─ 写出 solution.sources 里的源码文件
  │
  ▼
（若 C++/CUDA）子进程编译： python build_ext.py → benchmark_kernel.so
  └─ 自动注入 -gencode（Blackwell→sm_100a；local→探测本机 SM）
  │
  ▼
子进程执行： python eval_driver.py
  ├─ 加载问题、exec reference、import 用户解
  ├─ 逐 workload：正确性(10 轮) → 反作弊检查 → 计时
  └─ 每个 workload 产出一行 JSONL Trace 到 stdout
  │
  ▼
CLI 解析 stdout 的 JSONL → Trace 对象 → Rich 表格 / 文件输出
```

> 两阶段（编译在"compile server"、执行在"GPU server"）的物理分离也体现在 eval_driver 里：它**禁掉** `torch.utils.cpp_extension.load/load_inline`，强制 C++/CUDA 走 build_ext 这条受控编译路径（见 [[components/evaluation-driver]] §2）。

## 3. 子进程隔离：为什么这是核心设计

报告 §4.4 把"每个解在专属子进程编译执行"列为评测框架的关键设计，动机有四：

1. **状态隔离**：一个解改坏了全局状态（monkey-patch、泄漏显存、注入线程），不会污染后续解的评测。
2. **抗作弊的天然边界**：配合 [[concepts/reward-hacking-defenses]]——进程级隔离让"跨解缓存/串状态"这类攻击无从下手。
3. **可调度**：支持跨多 GPU 的 round-robin 调度；失败的 worker 可丢弃重启而不打断整个 benchmark。
4. **超时护栏**：300 秒超时防 hang / 死循环（对应 Trace 的 `TIMEOUT` 状态）。

报告还提到：reference 输出与 reference 计时数据**单独准备**，通过 IPC 传给 solution worker——进一步把"可信的 ground truth"与"不可信的候选"在进程层面分开。

## 4. stdout/stderr 协议：JSONL over stdout

`eval_driver.py` 启动第一件事就是**把 stdout 重定向到 stderr**，只保留原始 stdout fd 用于打印 JSON：

```python
_real_stdout_fd = os.dup(1)      # 留住真 stdout
os.dup2(2, 1)                    # fd 1 改指 stderr
```

于是 torch/triton/JIT 的各种噪声全进 stderr，**只有干净的 JSONL Trace 走真 stdout**。`ProblemPackager.convert_stdout_to_traces` 只挑以 `{` 开头的行解析为 Trace。这个"带内协议靠流分离"的设计让评测可被脚本可靠消费（`scripts/run_dataset.py` 批量跑就靠它）。

## 5. 暂存目录的构造（ProblemPackager）

`driver/problem_packager.py` 在临时目录里铺好一切：
- 一次性写 `definition.json` / `workload.jsonl` / `solution.json` / `config.json`，再展开 `solution.sources` 的每个源文件（保留相对路径结构）。
- `compile()`：仅 C++/CUDA；写 `build_ext.py`，注入 gencode（`_inject_gencode_flags`：Blackwell→`sm_100a`，因为 tcgen05/TMEM 指令需要；LOCAL→`_get_local_sm()` 探测本机），返回命令与产物路径 `benchmark_kernel.so`。
- `execute()`：写 `eval_driver.py`；C++ 解要求 `.so` 已存在，返回 `["python","eval_driver.py"]`。
- `__del__` 默认清理暂存目录（除非 `--keep-staging`）。

> Pydantic 模型直接 `model_dump_json()` 落盘、子进程再 parse 回来——数据模型既是内存契约也是**进程间序列化格式**（见 [[components/data-models]]）。

## 6. 运行入口与工具
- 单题：`sol-execbench <problem_dir> --solution solution.json`（CLI）。
- 批量：`scripts/run_dataset.py` 跑整个数据集，已过的题默认跳过（除非 `--rerun`）。
- 容器：`scripts/run_docker.sh --build` 进 NVIDIA 镜像（src/tests/data 自动挂载）；`--lock-clocks` 锁频。

## 延伸阅读
- 子进程脚本逐段剖析：[[components/evaluation-driver]]
- 数据模型/序列化契约：[[components/data-models]]
- 隔离如何服务反作弊：[[concepts/reward-hacking-defenses]]
- 计时/锁频/复现：[[concepts/timing-and-reproducibility]]
