---
type: concept
title: 正确性协议
updated: 2026-06-17
sources: [arXiv:2603.19173 §4.4, src/sol_execbench/core/bench/correctness.py, docs/workload.md, driver/templates/eval_driver.py]
tags: [correctness, tolerance, validation]
---

# 正确性协议

"可信评测"第一支柱：**先证明算对，再谈多快**。性能分有一个硬闸门 `C∈{0,1}`——没通过正确性的 kernel 性能分一律为零（见 [[concepts/sol-score]] §5）。

## 1. 检查顺序：从结构到数值，层层把关

报告 §4.4 + 代码 `eval_driver.py` 的检查链（每个 workload）：

```
1. 物化 reference 输出（pinned reference，return-value 风格）—— ground truth
2. 跑候选 kernel（DPS 或 return-value）
3. round 0 结构检查：lazy outputs → shape → dtype
4. 数值检查（每轮都做）：tensor sanity → 容差比对
   └─ 跨 10 轮、每轮重新随机生成输入
```

任一步失败就 emit 对应状态（`INVALID_REFERENCE` / `INCORRECT_SHAPE` / `INCORRECT_DTYPE` / `INCORRECT_NUMERICAL` / `REWARD_HACK`）并跳过该 workload，不进入计时。状态机详见 [[components/evaluation-driver]]。

## 2. 多轮随机重测（10 轮）—— 兼任反作弊

代码 `eval_driver.py` 对每个 workload 跑 `for _round in range(10)`，**每轮 `gen_inputs` 重新随机生成输入**。报告 §4.4 的动机有二：
1. **抓非确定性 bug 与输入依赖的正确性问题**——只测一次可能蒙混过关。
2. **反 state-caching 作弊**——agent 想"算一次缓存结果反复用"，但每轮输入都变，缓存就失效（见 [[concepts/reward-hacking-defenses]] §B）。

分工很讲究：**结构性检查（lazy/shape/dtype）和防御快照只在 round 0 做一次**（这些不随输入变），**数值检查每轮都做**（这才是要反复确认的）。round 0 还兼作 warmup——JIT 编译器（torch.compile / Triton / CuTe DSL）在首次调用时才 spawn 基础设施线程，把这个噪声放在正式计时之前。

## 3. 容差定义：torch.allclose 风格 + 三个加固

正确性不是"逐元素精确相等"——GPU kernel 用不同精度/累加顺序，本来就有数值差异。容差是**逐 workload**的（`docs/workload.md`、`ToleranceSpec`）：

| 字段 | 默认 | 含义 |
|------|------|------|
| `max_atol` | 1e-2 | 绝对误差阈值 |
| `max_rtol` | 1e-2 | 相对误差阈值 |
| `required_matched_ratio` | 0.99 | **必须通过容差的元素占比** |
| `max_error_cap` | null | 最大绝对误差硬上限（无视 matched_ratio 直接判废） |
| `allow_negative_inf` | false | 为 true 时，两边同为 -inf 的位置算对并排除出误差计算 |

判定公式（`core/bench/correctness.py::compute_error_stats`）：

```
元素通过 ⟺ |output − reference| ≤ max_atol + max_rtol·|reference|   （torch.allclose 风格）
workload 通过 ⟺ matched_ratio ≥ required_matched_ratio
               且（若设了 max_error_cap）最大绝对误差 ≤ max_error_cap
```

三个加固细节，每个都堵一种"看似对其实错"：
- **matched_ratio < 1（默认 0.99）**：允许极少数元素超差——容忍正常数值噪声，但不容忍系统性错误。
- **max_error_cap（"cuDNN 模式"）**：防"大多数元素对、个别离群点误差任意大"被放过。代码注释直接点名这是为 cuDNN 这类情况设的。
- **rel_error 用 `max_atol` 当除数下限**（`torch.clamp(|y|, min=max_atol)`）：避免在接近零处除以极小值导致相对误差爆炸。

> 容差是**逐 workload**的，不是全局：同一问题里，容易的形状可以收紧、数值难的配置可以放松（`docs/workload.md`）。这让正确性标准贴合每个具体配置的数值特性。

## 4. Tensor sanity：inf / NaN / 全零

`core/bench/correctness.py::check_tensor_sanity` 在算误差**之前**先体检，三条铁律：

1. **非有限值一律判错**：任一张量含 inf/NaN 即失败——**即便两边在同一位置都是 inf 也拒**（注释："非有限值意味着退化数值，不是正确输出"）。唯一例外是 `allow_negative_inf=true` 时两边同为 -inf 的位置（attention mask 等场景的合法 -inf）。
2. **NaN 优先于 Inf**：报 `has_nan` 还是 `has_inf` 时 NaN 优先——因为 NaN 是更严重的失败（Inf 可能只是溢出，NaN 意味着未定义计算）。
3. **全零输出检测**：reference 有非平凡值（范数 > 0）而候选全零 → 立即判错。这堵的是"什么都不算、直接返回零张量"的退化解。

## 5. 容差怎么定出来的：离线标定 + 安全裕度

报告 §4.4 披露了容差的来源（这往往被忽略，却是可信的关键）：

- **BF16/FP32 问题**：容差**离线标定**——在随机输入上**反复探测 reference**，测出它自身的数值波动，再对所需绝对容差乘 **1.25× 安全裕度**。即容差不是拍脑袋，而是从"reference 自己跑多次有多大抖动"反推的。
- **量化 / sampling 问题用专门评测器**：量化 kernel 对 cuBLAS reference（经 PyTorch）比对，cuBLAS 不可用时退回 FP32 reference 比对。
- 报告 §3.2 校验阶段也提到，执行式校验"用从重复 reference 运行标定的容差"验证全部 workload 的数值正确性。

这条把容差和"精度降级"作弊联系起来：因为容差是按 reference 真实精度需求紧标定的，FP32 偷用 FP16 算往往会被容差挡掉（见 [[concepts/reward-hacking-defenses]] §C）。

## 6. reference 的角色：功能规格，不是性能基线

报告 §4.4 反复强调（也见 [[concepts/design-philosophy]] §5）：reference 写来**为可移植、可读、功能覆盖**，可能很慢，应被当作**功能规格**而非性能基线。所以：
- 正确性比对的对象是 reference；
- 但 SOL Score 的 0.5 锚点是**另一个**高性能 scoring baseline `T_b`；
- Trace 里 `speedup_factor` 相对 reference 算，所以"很大"是正常的（`docs/trace.md` 也特意注明）。

## 延伸阅读
- 正确性如何当性能分的闸门：[[concepts/sol-score]] §5
- 多轮随机如何反作弊：[[concepts/reward-hacking-defenses]]
- 检查链在驱动里的精确位置与状态机：[[components/evaluation-driver]]
- 容差字段所在的数据模型：[[components/data-models]]
