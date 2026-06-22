---
type: source
title: 代码库与 docs 来源索引
updated: 2026-06-22
tags: [source, code-map, provenance]
---

# 代码库与 docs 来源索引

把"原始来源（代码 + docs）"映射到分析它的 wiki 页，便于回溯。路径相对仓库根 `SOL-ExecBench/`。

## 核心源码 `src/sol_execbench/`

| 文件 | 是什么 | 主要沉淀到 |
|------|--------|------------|
| `sol_score.py` | SOL Score 公式实现（含 `T_b≤T_SOL` 边界分支） | [[concepts/sol-score]] |
| `cli/main.py` | Click CLI 入口 `sol-execbench`；解析参数、调度、Rich 表格 | [[components/architecture-and-flow]] |
| `driver/problem_packager.py` | 暂存问题文件、产出编译/执行命令、注入 gencode、解析 JSONL traces | [[components/architecture-and-flow]]、[[components/evaluation-driver]] |
| `driver/templates/eval_driver.py` | 子进程里运行的自包含评测脚本（~650 行；正确性/计时/反作弊编排） | [[components/evaluation-driver]] |
| `driver/templates/build_ext.py` | C++/CUDA 编译脚本（产出 benchmark_kernel.so） | [[components/architecture-and-flow]] |
| `core/bench/timing.py` | CUPTI / CUDA events 计时；冷 L2；`time_runnable`；`ShiftingMemoryPoolAllocator` 入口 | [[concepts/timing-and-reproducibility]] |
| `core/bench/correctness.py` | `compute_error_stats`、`check_tensor_sanity`、`set_seed`；容差判定 | [[concepts/correctness-protocol]] |
| `core/bench/reward_hack.py` | 反作弊原语：monkey-patch / thread / lazy / 完整性快照 | [[concepts/reward-hacking-defenses]] |
| `core/bench/clock_lock.py` | `nvidia-smi` 锁/验/解 GPU+DRAM 频率；`are_clocks_locked` | [[concepts/timing-and-reproducibility]] |
| `core/bench/io.py` | 输入生成（启发式 + safetensors + custom）、输出分配、`ShiftingMemoryPoolAllocator` 实现 | [[concepts/timing-and-reproducibility]]、[[components/data-models]] |
| `core/bench/config/` | `BenchmarkConfig`（warmup/iterations/seed/lock_clocks/benchmark_reference）、`device_config`（锁频预设表） | [[components/evaluation-driver]]、[[concepts/timing-and-reproducibility]] |
| `core/data/definition.py` | Definition 模型：symbolic axes、约束、轴解析、custom_inputs_entrypoint | [[components/data-models]] |
| `core/data/solution.py` | Solution 模型：spec、languages、DPS、dependencies、compile_options | [[components/data-models]] |
| `core/data/workload.py` | Workload 模型：输入描述符、`ToleranceSpec` | [[components/data-models]]、[[concepts/correctness-protocol]] |
| `core/data/trace.py` | Trace 模型：status 枚举、correctness/performance/environment | [[components/data-models]] |
| `core/data/{dtypes,shapes,base_model,json_utils}.py` | dtype 字符串↔torch、形状解析、Pydantic 基类、JSON 工具 | [[components/data-models]] |

> 注（按 [[CLAUDE]] 诚实约定）：`core/data/*.py` 的 Pydantic 校验逻辑主要据 `docs/` 与调用点理解，未逐行精读；如需更深可补 ingest（见 [[log]] open threads）。

## 文档 `docs/`

| 文件                   | 内容                                                                                          | 主要沉淀到                                                        |
| -------------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `docs/definition.md` | Definition schema：axes(const/var/expr)、inputs/outputs、reference、custom_inputs_entrypoint、例子 | [[components/data-models]]                                   |
| `docs/workload.md`   | Workload schema：random/scalar/safetensors/custom、tolerance、评测器如何用 workload                  | [[components/data-models]]、[[concepts/correctness-protocol]] |
| `docs/solution.md`   | Solution schema：spec、DPS、compile_options、dependencies、各语言签名与例子                              | [[components/data-models]]                                   |
| `docs/trace.md`      | Trace schema：status、correctness/performance/environment、nullable 表、例子                       | [[components/data-models]]                                   |

## 顶层与脚本

| 文件 | 内容 |
|------|------|
| `README.md` | 项目总览、SOL-Score/SOLAR/B200 提及、CLI 用法、benchmark config、citation | 散见各页 |
| `CLAUDE.md`(repo 根) | 给 Claude Code 的代码库指南：架构三层、执行流、两种调用约定、测试标记 | [[components/architecture-and-flow]] |
| `scripts/run_dataset.py` | 批量评测整个数据集（已过题默认跳过） | [[components/architecture-and-flow]] |
| `scripts/run_docker.sh` / `download_data.sh` | 启容器（挂载 src/tests/data，可锁频）/ 下数据集 | [[components/architecture-and-flow]] |
| `examples/` | 各 DSL 的示例解（pytorch/triton/cutlass/cudnn/cute_dsl/cutile/cuda_cpp）；全量算子清单见 [[components/data-models]] §3.1 | [[components/data-models]] |

## 外部来源
- 技术报告 arXiv:2603.19173 → 逐节要点见 [[sources/tech-report]]
- SOLAR 工具 github.com/NVlabs/SOLAR（**未 ingest 源码**，仅据报告 §4.2 + README 分析）→ [[concepts/speed-of-light-and-solar]]
- HuggingFace 数据集 nvidia/SOL-ExecBench；Leaderboard research.nvidia.com/benchmarks/sol-execbench
