---
type: component
title: 评测驱动 — eval_driver.py 逐段剖析
updated: 2026-06-17
sources: [src/sol_execbench/driver/templates/eval_driver.py, driver/problem_packager.py]
tags: [eval-driver, orchestration, state-machine, defenses]
---

# 评测驱动：eval_driver.py 逐段剖析

`driver/templates/eval_driver.py` 是被写进暂存目录、在**子进程**里运行的自包含评测脚本（约 650 行）。它是"评测科学"真正发生的地方——所有正确性、计时、反作弊检查在这里按精确顺序编排。配套的 `ProblemPackager` 见 [[components/architecture-and-flow]] §5。

## 1. 启动：流分离（先于 import torch）

```python
_real_stdout_fd = os.dup(1); os.dup2(2, 1)   # fd1→stderr，留住真 stdout 打 JSON
import torch                                   # 必须在重定向之后
```

目的：torch/triton/JIT 噪声进 stderr，**只有 JSONL Trace 走真 stdout**。`_emit()` 用 `json.dumps(..., allow_nan=False)` 写真 stdout——意外 NaN/Inf 立即报错而非产非法 JSON。详见 [[components/architecture-and-flow]] §4。

## 2. 防篡改快照（先于 import 用户代码）

这是反作弊的"地基"——**在用户代码进来之前**给评测器自身拍照：

```python
_CRITICAL_NAMES = ["time_runnable","compute_error_stats","check_monkey_patch",
  "check_lazy_outputs","check_thread_injection","check_eval_integrity",
  "_call_and_collect_outputs","gen_inputs","allocate_outputs","normalize_outputs","make_eval"]
_integrity_snapshot = snapshot_critical_functions(globals(), _CRITICAL_NAMES)
_check_integrity = check_eval_integrity     # 存局部引用：改 globals() 里的名字也没用
```

同时 `reward_hack.py` 模块加载时已捕获 `id(torch.cuda.Event.elapsed_time)`。两道快照分别防"评测逻辑被替换"和"计时函数被 monkey-patch"（见 [[concepts/reward-hacking-defenses]] §3）。

## 3. 加载与 import：受控编译边界

- exec reference 代码：写到真实文件 `_reference.py` 再 exec——因为 `@triton.jit` 等装饰器会 `inspect.getsourcelines()` 回读源码。
- import 用户解：
  - C++/CUDA 解：从 `benchmark_kernel.so` 加载（须 compile 阶段已产出）。
  - Python 解：**禁用** `torch.utils.cpp_extension.load/load_inline`：

    ```python
    _cpp_ext.load = _blocked_cpp_ext_load
    _cpp_ext.load_inline = _blocked_cpp_ext_load
    ```

    强制 C++/CUDA 编译走 compile server 的 build_ext 受控路径，**堵死在 GPU server 上现场编译**这条绕过源码审查的口子。
- import 用户代码后立刻 `check_eval_integrity` 一次：抓 import 期施加的篡改（如经 `sys.modules['__main__']`）。命中则给所有 workload emit `REWARD_HACK` 并退出。

## 4. 调用约定适配（DPS / 返回值）

```python
def _call_and_collect_outputs(fn, inputs, dps, ...):
    if dps:
        outputs = allocate_outputs(...); fn(*inputs, *outputs); sync(); return outputs
    else:
        result = fn(*inputs); sync(); return [normalize_outputs(result)[n] for n in names]
```

两种约定都支持（见 [[components/data-models]] §3）。注意调用后**显式 `torch.cuda.synchronize()`**——把异步工作强制落地，既保正确性比对拿到真实结果，也防"把活留在异步队列里逃避计时"。

## 5. 每个 workload 的状态机

核心循环 `for _workload in workloads:`，顺序如下（emit 对应 status 后 `continue`/`break` 跳过计时）：

```
workload 开始
  │ 清上一 workload 的张量 + gc + empty_cache（防跨 workload 显存累积 OOM）
  │ 解析 var 轴 → resolved_axes
  │ 加载 safetensors（失败→RUNTIME_ERROR）
  │
  ├─ 正确性循环 for _round in range(10):
  │     gen_inputs（失败→RUNTIME_ERROR, break）
  │     跑 reference（失败→INVALID_REFERENCE, break）   ← ground truth
  │     跑 user_fn（失败→RUNTIME_ERROR, break）          ← round0 兼作 JIT warmup
  │     if round==0:                                     ← 只做一次的结构检查
  │        check_eval_integrity（命中→REWARD_HACK）
  │        记录 threads_before = active_count()          ← 取 JIT 稳定后的线程基线
  │        check_lazy_outputs（FakeTensor 等→REWARD_HACK）
  │        shape 检查→INCORRECT_SHAPE；dtype 检查→INCORRECT_DTYPE
  │     每轮：compute_error_stats（超差→INCORRECT_NUMERICAL, break）
  │
  ├─ check_monkey_patch（计时前，命中→REWARD_HACK）
  ├─ time_runnable(user_fn, warmup=10, rep=50)（失败→RUNTIME_ERROR）   ← 测 T_k
  ├─ check_thread_injection(threads_before, active_count())（命中→REWARD_HACK）  ← 抓计时期注入
  ├─ （可选）time_runnable(ref_fn) 算 speedup —— 默认关闭
  └─ emit PASSED（带 correctness + performance）
```

几个编排上的讲究：
- **10 轮、每轮重生成随机输入**：抓非确定性 bug + 反 state-caching（见 [[concepts/correctness-protocol]] §2）。
- **结构检查只在 round 0**（不随输入变），**数值检查每轮**（要反复确认）。
- **threads_before 取 round 0 之后**：JIT（torch.compile/Triton/CuTe DSL）首次调用才 spawn 基础设施线程，先让它稳定，再在计时后比对，才能干净地抓到"计时期间"注入的线程。
- **monkey-patch 检查紧贴计时前**、**thread-injection 检查紧贴计时后**——把防御卡在它们各自能生效的时间点。
- 防御检查通过 `_reward_hack_check` 包装：命中即 emit `REWARD_HACK` trace 并返回 True。

## 6. 配置驱动行为
- `BenchmarkConfig`（`config.json`）：`warmup_runs`=10、`iterations`=50、`seed`=200、`lock_clocks`、`benchmark_reference`（默认 false——因为朴素 reference 可能极慢，>1h，会主导总时长）。
- 锁频校验：`lock_clocks=True` 但实际未锁 → 拒绝所有 workload（`RUNTIME_ERROR`），不给不可信的数（见 [[concepts/timing-and-reproducibility]] §5）。`set_seed(seed)` 保输入可复现。

## 7. ProblemPackager 侧（编排的另一半）
`driver/problem_packager.py`：把模型 `model_dump_json()` 落盘暂存、展开源码；`compile()` 注入 gencode（Blackwell→`sm_100a`，需 tcgen05/TMEM）；`execute()` 返回运行命令；`convert_stdout_to_traces()` 把 JSONL stdout 解析回 Trace（只取 `{` 开头行）。完整见 [[components/architecture-and-flow]] §5。

## 一句话总结
eval_driver 把"先证明算对（10 轮）→ 确认没作弊（沿时间线插桩）→ 才测速度（CUPTI+冷L2+平移池）"这条**不可绕过的顺序**固化进一个自包含、流分离、子进程隔离的脚本——这就是 SOL-ExecBench "可信评测"在代码层的全部落点。

## 延伸阅读
- 三层架构与子进程隔离动机：[[components/architecture-and-flow]]
- 防御原语实现：[[concepts/reward-hacking-defenses]]
- 正确性细节：[[concepts/correctness-protocol]]
- 计时细节：[[concepts/timing-and-reproducibility]]
- 状态枚举与 nullable 表：[[components/data-models]]
